import gym
import torch
import torch.nn.functional as F
import numpy as np

# 策略网络，输入状态s，输出动作概率 πθ(a|s)
class PolicyNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(PolicyNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        # 这里输出的即是 πθ(a|s)，多项分布概率
        return F.softmax(self.fc2(x), dim=1)

# REINFORCE算法
class REINFORCE:
    def __init__(self, state_dim, hidden_dim, action_dim, learning_rate, gamma,
                 device):
        self.policy_net = PolicyNet(state_dim, hidden_dim,
                                    action_dim).to(device)
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(),
                                          lr=learning_rate)
        self.gamma = gamma  # 折扣因子 γ
        self.device = device

    def take_action(self, state):
        # 状态输入网络，获得动作概率 πθ(a|s)
        state = torch.tensor([state], dtype=torch.float).to(self.device)
        probs = self.policy_net(state)
        # 按概率采样动作 a_t ~ πθ(·|s_t)
        action_dist = torch.distributions.Categorical(probs)
        action = action_dist.sample()
        return action.item()

    def update(self, transition_dict):
        reward_list = transition_dict['rewards']  # 奖励序列 {r_t}
        state_list = transition_dict['states']    # 状态序列 {s_t}
        action_list = transition_dict['actions']  # 动作序列 {a_t}

        G = 0  # 初始化累计折扣奖励 G_t
        self.optimizer.zero_grad()  # 梯度清零

        # 从轨迹终点向前遍历，计算每个时间步的梯度贡献
        for i in reversed(range(len(reward_list))):
            reward = reward_list[i]  # 即时奖励 r_t
            state = torch.tensor([state_list[i]], dtype=torch.float).to(self.device)
            action = torch.tensor([action_list[i]]).view(-1, 1).to(self.device)

            # 计算 πθ(a_t|s_t)
            action_prob = self.policy_net(state).gather(1, action)
            log_prob = torch.log(action_prob)  # 计算 log πθ(a_t|s_t)
            # 公式中 ∇θ log πθ(a_t|s_t) 的梯度计算由loss.backward触发

            # 计算累计折扣奖励 G_t = r_t + γ * G_{t+1}
            # 对应公式中 G_t = ∑_{t'=t}^T γ^{t'-t} r_{t'}
            G = self.gamma * G + reward

            # 计算损失函数（负的策略梯度估计）
            # 对应 REINFORCE 梯度公式 ∇θ J(θ) ≈ E[ G_t ∇θ log πθ(a_t|s_t) ]
            loss = -log_prob * G

            # 反向传播，计算梯度 ∇θ J(θ)
            loss.backward()

        # 梯度更新，执行参数优化
        self.optimizer.step()

def main():
    learning_rate = 1e-3
    num_episodes = 1000
    hidden_dim = 128
    gamma = 0.98
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    env_name = "CartPole-v0"
    env = gym.make(env_name)
    env.seed(0)
    torch.manual_seed(0)
    np.random.seed(0)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    agent = REINFORCE(state_dim, hidden_dim, action_dim, learning_rate, gamma, device)
    return_list = []

    for i_episode in range(num_episodes):
        state = env.reset()
        transition_dict = {'states': [], 'actions': [], 'rewards': []}
        episode_return = 0
        done = False

        while not done:
            action = agent.take_action(state)  # 动作采样 a_t ~ πθ(·|s_t)
            next_state, reward, done, _ = env.step(action)

            # 存储轨迹数据
            transition_dict['states'].append(state)
            transition_dict['actions'].append(action)
            transition_dict['rewards'].append(reward)

            state = next_state
            episode_return += reward

        return_list.append(episode_return)
        agent.update(transition_dict)  # 使用轨迹更新策略参数 θ

        if (i_episode + 1) % 10 == 0:
            avg_return = np.mean(return_list[-10:])
            print(f"Episode {i_episode+1}, Average Return: {avg_return:.2f}")

if __name__ == "__main__":
    main()
