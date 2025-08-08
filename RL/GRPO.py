import torch
import torch.nn.functional as F

# 模拟一个简易模型用于测试
class DummyModel(torch.nn.Module):
    def forward(self, input_ids, attention_mask):
        vocab_size = 100
        # 模拟 logits 输出 [batch_size, seq_len, vocab_size]
        logits = torch.randn(input_ids.size(0), input_ids.size(1), vocab_size)
        return type('Obj', (object,), {'logits': logits})


class GRPOTrainer:
    def __init__(self, epsilon=0.2, kl_coeff=0.1, num_iterations=1):
        self.epsilon = epsilon        # PPO clipping 的 ε 超参数
        self.kl_coeff = kl_coeff      # KL 正则项的系数 β
        self.num_iterations = num_iterations  # 控制是否使用旧策略 log_probs

    # 获取每个 token 的 log 概率
    def _get_per_token_logps(self, model, input_ids, attention_mask, logits_to_keep):
        # 模型前向传播，得到 logits
        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
        # 截取只保留 completion 部分的 logits（忽略 prompt）
        logits = logits[:, -logits_to_keep-1:-1]  # 右移对齐目标 token
        log_probs = F.log_softmax(logits, dim=-1)

        # 提取 target token 的 log_prob
        target_ids = input_ids[:, -logits_to_keep:]
        per_token_logps = torch.gather(log_probs, dim=2, index=target_ids.unsqueeze(-1)).squeeze(-1)
        return per_token_logps

    # 对每个 group 内的 advantage 做归一化
    def _normalize_advantages_by_group(self, advantages, num_items_in_batch):
        # 若没有分组信息，则对全 batch 做归一化
        if num_items_in_batch is None:
            return (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # 按 group 归一化
        norm_advantages = torch.empty_like(advantages)
        idx = 0
        for group_size in num_items_in_batch:
            group_adv = advantages[idx:idx + group_size]
            norm = (group_adv - group_adv.mean()) / (group_adv.std(unbiased=False) + 1e-8)
            norm_advantages[idx:idx + group_size] = norm
            idx += group_size
        return norm_advantages

    # 计算 GRPO 损失函数
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        GRPO 损失函数实现，结构如下：
        J_GRPO(θ) = E[ min(r_t * A_t, clip(r_t, 1-ε, 1+ε)*A_t) - β * D_KL[π_θ || π_ref] ]
        """

        # 1. 准备输入数据
        prompt_ids, prompt_mask = inputs["prompt_ids"], inputs["prompt_mask"]
        completion_ids, completion_mask = inputs["completion_ids"], inputs["completion_mask"]

        # 拼接 prompt + completion，生成完整输入序列
        input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
        logits_to_keep = completion_ids.size(1)  # 只关注 completion 部分的预测概率

        # 2. 当前策略 π_θ 的 log 概率
        per_token_logps = self._get_per_token_logps(model, input_ids, attention_mask, logits_to_keep)

        # 3. KL 散度项：D_KL[π_θ || π_ref]
        ref_per_token_logps = inputs["ref_per_token_logps"]
        per_token_kl = torch.exp(ref_per_token_logps - per_token_logps) - (ref_per_token_logps - per_token_logps) - 1
        kl_loss = self.kl_coeff * per_token_kl.mean()

        # 4. 归一化的 advantage 值（按 group）
        advantages = self._normalize_advantages_by_group(inputs["advantages"], num_items_in_batch)  # shape: [B]

        # 5. PPO 概率比计算 r_t = π_θ / π_old
        old_per_token_logps = inputs["old_per_token_logps"] if self.num_iterations > 1 else per_token_logps.detach()
        coef_1 = torch.exp(per_token_logps - old_per_token_logps)
        coef_2 = torch.clamp(coef_1, 1 - self.epsilon, 1 + self.epsilon)

        # 6. 最小截断损失项：min(r_t * A_t, clip(r_t) * A_t)
        per_token_loss1 = coef_1 * advantages.unsqueeze(1)
        per_token_loss2 = coef_2 * advantages.unsqueeze(1)
        per_token_loss = -torch.min(per_token_loss1, per_token_loss2)

        # 7. 最终 GRPO 损失：PPO 损失 + KL 正则项
        loss = per_token_loss.mean() + kl_loss

        if return_outputs:
            return loss, per_token_logps
        return loss


# 模拟一批输入
B, T = 6, 5  # batch size = 6, 每条 completion 长度 = 5

inputs = {
    "prompt_ids": torch.randint(0, 100, (B, 5)),
    "prompt_mask": torch.ones(B, 5),
    "completion_ids": torch.randint(0, 100, (B, T)),
    "completion_mask": torch.ones(B, T),
    "ref_per_token_logps": torch.randn(B, T),       # 模拟参考策略的 logp
    "old_per_token_logps": torch.randn(B, T),       # 模拟旧策略的 logp
    "advantages": torch.randn(B)                    # 每个样本一个 advantage
}

# 模拟三组，每组两个样本
num_items_in_batch = [2, 2, 2]

# 初始化 GRPO 训练器与模型
trainer = GRPOTrainer(epsilon=0.2, kl_coeff=0.05)
model = DummyModel()

# 计算损失
loss = trainer.compute_loss(model, inputs, num_items_in_batch=num_items_in_batch)
print("GRPO 损失值:", loss.item())
