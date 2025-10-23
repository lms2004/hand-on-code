import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

class DPOLoss(nn.Module):
    """
    Direct Preference Optimization (DPO) Loss
    -----------------------------------------
    参数:
        beta (float): 控制 reward 尺度的温度参数（相当于 logit 的放缩系数）
        label_smoothing (float): 在正/负 logits 之间的平滑因子 (0 表示标准 DPO)
        ipo (bool): IPO 变体标志（此实现中未使用）

    返回（forward）:
        loss: 标量张量（batch 平均）
        chosen_rewards: 用于监控的已 detach 的 chosen reward
        rejected_rewards: 用于监控的已 detach 的 rejected reward
    """
    def __init__(self, beta: float, label_smoothing: float = 0.0, ipo: bool = False) -> None:
        super().__init__()
        self.beta = float(beta)
        self.label_smoothing = float(label_smoothing)
        self.ipo = bool(ipo)

    def forward(
        self,
        policy_chosen_logps: torch.Tensor,      # log π(a|x) 对于被选中的样本 (batch,)
        policy_rejected_logps: torch.Tensor,    # log π(a'|x) 对于被拒绝的样本 (batch,)
        reference_chosen_logps: torch.Tensor,   # log π_ref(a|x) (batch,)
        reference_rejected_logps: torch.Tensor, # log π_ref(a'|x) (batch,)
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # ---------------------------------------------------------------------
        # 1) 计算策略和参考策略的 log-ratio（对数比）
        #    π_logratio = log π(chosen) - log π(rejected)
        #    ref_logratio = log π_ref(chosen) - log π_ref(rejected)
        #    数学表示：
        #       r_π = log π(a|x) - log π(a'|x)
        #       r_ref = log π_ref(a|x) - log π_ref(a'|x)
        #    形状: (batch,)
        # ---------------------------------------------------------------------
        pi_logratios = policy_chosen_logps - policy_rejected_logps
        ref_logratios = reference_chosen_logps - reference_rejected_logps

        # ---------------------------------------------------------------------
        # 2) 计算二者的差作为最终的对比 logits（policy 相对于 reference 的优势）
        #    logits = r_π - r_ref
        #    数学表示：
        #       z = r_π - r_ref = (log π(chosen)-log π(rejected)) - (log π_ref(chosen)-log π_ref(rejected))
        #    直观：z>0 表示 policy 更偏向 chosen 相比 reference，z<0 则相反
        # ---------------------------------------------------------------------
        logits = pi_logratios - ref_logratios  # shape: (batch,)

        # ---------------------------------------------------------------------
        # 3) 将 logits 放缩并计算 DPO 的 logistic 形式损失（含 label smoothing）
        #    原始 DPO/二分类损失构造类似二元交叉熵：
        #       对于正类 (chosen 更优) 使用 -log sigmoid(beta * z)
        #       若使用 label smoothing 则混合 -log sigmoid(-beta*z)
        #
        #    losses_i = -(1 - ε) * log σ(β z_i) - ε * log σ(-β z_i)
        #    其中 σ 为 sigmoid，ε = label_smoothing
        #    注：当 ε=0 时，losses_i = -log σ(β z_i)
        # ---------------------------------------------------------------------
        losses = (
            -F.logsigmoid(self.beta * logits) * (1 - self.label_smoothing)
            - F.logsigmoid(-self.beta * logits) * self.label_smoothing
        )

        # ---------------------------------------------------------------------
        # 4) 对 batch 求均值作为最终标量损失
        #    loss = mean_i losses_i
        # ---------------------------------------------------------------------
        loss = losses.mean()

        # ---------------------------------------------------------------------
        # 5) 为监控计算 reward（detach，避免反向传播通过 reward）
        #    DPO 中常用的 reward 定义（成比例于 log π - log π_ref）：
        #       r_chosen = β * (log π(chosen) - log π_ref(chosen))
        #       r_rejected = β * (log π(rejected) - log π_ref(rejected))
        #    这里乘以 β 以保持与 logits 的尺度一致（可视作温度缩放）
        #    detach() 保证这些张量不会参与梯度计算，只供监控/记录使用
        # ---------------------------------------------------------------------
        chosen_rewards = (self.beta * (policy_chosen_logps - reference_chosen_logps)).detach()
        rejected_rewards = (self.beta * (policy_rejected_logps - reference_rejected_logps)).detach()

        return loss, chosen_rewards, rejected_rewards



# ==========  测试代码部分（可直接运行）  ==========

class ToyPolicy(nn.Module):
    """
    简化的 policy，用来演示 DPO Loss 的梯度传播。
    它包含两个参数张量：chosen 和 rejected。
    """
    def __init__(self, batch_size: int):
        super().__init__()
        self.chosen = nn.Parameter(torch.randn(batch_size) * 0.1)
        self.rejected = nn.Parameter(torch.randn(batch_size) * 0.1)

    def forward(self):
        return self.chosen, self.rejected


def run_demo(seed=42, device='cpu'):
    torch.manual_seed(seed)
    device = torch.device(device)

    batch_size = 8
    model = ToyPolicy(batch_size).to(device)

    # 固定 reference 模型的 logits（模拟参考策略）
    reference_chosen = torch.randn(batch_size, device=device) * 0.05
    reference_rejected = torch.randn(batch_size, device=device) * 0.05

    dpo = DPOLoss(beta=1.0, label_smoothing=0.1).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

    print("Initial model chosen logits:", model.chosen.data.cpu().numpy())
    print("Initial model rejected logits:", model.rejected.data.cpu().numpy())

    for step in range(1, 11):
        optimizer.zero_grad()
        policy_chosen, policy_rejected = model()

        loss, chosen_rewards, rejected_rewards = dpo(
            policy_chosen_logps=policy_chosen,
            policy_rejected_logps=policy_rejected,
            reference_chosen_logps=reference_chosen,
            reference_rejected_logps=reference_rejected,
        )

        loss.backward()
        optimizer.step()

        if step <= 3 or step % 5 == 0:
            print(
                f"Step {step:2d} | loss={loss.item():.6f} | "
                f"mean chosen_reward={chosen_rewards.mean().item():.6f} | "
                f"mean rejected_reward={rejected_rewards.mean().item():.6f}"
            )

    print("\nFinal model chosen logits:", model.chosen.data.cpu().numpy())
    print("Final model rejected logits:", model.rejected.data.cpu().numpy())
    print("Gradients (chosen):", model.chosen.grad.cpu().numpy())
    print("Gradients (rejected):", model.rejected.grad.cpu().numpy())


if __name__ == "__main__":
    run_demo()
