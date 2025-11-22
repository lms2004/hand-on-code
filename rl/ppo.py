import torch
import torch.nn as nn
import torch.nn.functional as F

##########################################
# 🧮 GAE（广义优势估计）公式对应代码
##########################################
def compute_advantage(gamma, lmbda, td_delta):
    """
    计算 GAE： A_t = Σ (γλ)^l * δ_{t+l}
    参数：
        gamma：折扣因子
        lmbda：GAE 衰减系数
        td_delta：TD 残差 δ_t = r + γ V(s') - V(s)
    返回：
        advantage：每个时间步的优势值
    """
    td_delta = td_delta.detach().cpu().numpy()
    advantage = 0.0
    advantage_list = []
    for delta in reversed(td_delta):
        advantage = gamma * lmbda * advantage + delta
        advantage_list.append(advantage)
    advantage_list.reverse()
    return torch.tensor(advantage_list, dtype=torch.float32)

##########################################
# 🧩 策略网络的 PPO Loss（公式核心实现）
##########################################
class PolicyLoss(nn.Module):
    """
    策略损失函数，对应 PPO 的 min/clip 目标函数
    """
    def __init__(self, clip_eps: float = 0.2):
        super().__init__()
        self.clip_eps = clip_eps

    def forward(self, log_probs, old_log_probs, advantages, action_mask=None):
        ratio = (log_probs - old_log_probs).exp()  # e^{logπ - logπ_old}
        surr1 = ratio * advantages
        surr2 = ratio.clamp(1 - self.clip_eps, 1 + self.clip_eps) * advantages
        loss = -torch.min(surr1, surr2)
        loss = masked_mean(loss, action_mask, dim=-1).mean()
        return loss

##########################################
# 📈 值函数损失（Value Function Clipping）
##########################################
class ValueLoss(nn.Module):
    """
    值函数损失，对 value 使用 clip 避免更新过大
    """
    def __init__(self, clip_eps: float = 0.2):
        super().__init__()
        self.clip_eps = clip_eps

    def forward(self, values, old_values, returns, action_mask=None):
        values_clipped = old_values + (values - old_values).clamp(-self.clip_eps, self.clip_eps)
        surr1 = (values_clipped - returns) ** 2
        surr2 = (values - returns) ** 2
        loss = torch.max(surr1, surr2)
        loss = masked_mean(loss, action_mask, dim=-1).mean()
        return 0.5 * loss

##########################################
# 🤼 Pairwise Loss：用于训练奖励模型 RM
##########################################
class PairWiseLoss(nn.Module):
    """
    用于 reward model 的 pairwise 训练
    chosen_reward 应该高于 reject_reward
    """
    def forward(self, chosen_reward, reject_reward, margin=None):
        if margin is not None:
            loss = -F.logsigmoid(chosen_reward - reject_reward - margin)
        else:
            loss = -F.logsigmoid(chosen_reward - reject_reward)
        return loss.mean()

##########################################
# 💰 奖励计算：r + KL 罚项（或加 bonus）
##########################################
def compute_reward(r, kl, action_mask, kl_coef=0.01):
    """
    奖励 = r - β * KL，r 只加到最后一个 step（例如 EOS）
    """
    kl_reward = -kl_coef * kl
    eos_idx = action_mask.size(1) - 1 - action_mask.long().fliplr().argmax(dim=1, keepdim=True)
    last_reward = torch.zeros_like(kl).scatter(1, eos_idx, r.unsqueeze(1).to(kl.dtype))
    total_reward = last_reward + kl_reward
    return total_reward

##########################################
# 🔢 mask 平均函数，用于忽略 padding token
##########################################
def masked_mean(tensor, mask=None, dim=-1):
    if mask is None:
        return tensor.mean(dim)
    return (tensor * mask).sum(dim) / mask.sum(dim)

##########################################
# 🔁 PPO 更新示例
##########################################
def ppo_train_step(
    policy_model,         # 策略网络
    value_model,          # 值网络
    optimizer,            # 优化器
    states,               # 状态 s_t（或 prompt）
    actions,              # 行动 a_t（或生成 token）
    old_log_probs,        # 老策略 logπ_old
    returns,              # GAE 加权后的 Return
    advantages,           # GAE Advantage
    values,               # 老的值估计
    action_mask=None,     # mask 掉 padding
    kl=None,              # KL divergence
    ref_log_probs=None    # 参考策略（可选）
):
    # forward current policy
    log_probs = policy_model.get_log_prob(states, actions)
    new_values = value_model(states)

    # policy loss
    policy_loss = PolicyLoss()(log_probs, old_log_probs, advantages, action_mask)

    # value loss
    value_loss = ValueLoss()(new_values, values, returns, action_mask)

    # total loss
    loss = policy_loss + value_loss

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return {
        'policy_loss': policy_loss.item(),
        'value_loss': value_loss.item(),
        'total_loss': loss.item()
    }
