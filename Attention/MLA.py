#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DeepSeek-V2 MLA Demo — 带公式注释版本
验证 MLA 的低秩压缩 + 解耦 RoPE
"""

import torch
import torch.nn.functional as F


# =======================================
# 🔹 RoPE 旋转函数
# 公式:  RoPE(x) = x·cosθ + rotate(x)·sinθ
# =======================================
def apply_rope_x(x, cos, sin):
    """
    x: [B, H, L, D]
    cos/sin: [1, 1, L, D]
    """
    x1, x2 = x[..., ::2], x[..., 1::2]
    x_rot = torch.stack([-x2, x1], dim=-1).reshape_as(x)
    return x * cos + x_rot * sin


# =======================================
# 🔹 MLA 模块（DeepSeek-V2 实现）
# =======================================
class MLA(torch.nn.Module):
    def __init__(self, d_model, n_heads, max_len=1024, rope_theta=10000.0):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.dh = d_model // n_heads
        self.q_proj_dim = d_model // 2           # ↓ 公式 (W^{DQ})
        self.kv_proj_dim = (2 * d_model) // 3    # ↓ 公式 (W^{DKV})

        # 每个 head 的拆分维度
        self.qk_nope_dim = self.dh // 2   # 不带 RoPE 的部分
        self.qk_rope_dim = self.dh // 2   # 带 RoPE 的部分

        # ===== Q 投影层 =====
        # 公式:  c_t^Q = W^{DQ} h_t
        self.W_dq = torch.nn.Parameter(0.01 * torch.randn((d_model, self.q_proj_dim)))
        # 公式:  q_t^C = W^{UQ} c_t^Q
        self.W_uq = torch.nn.Parameter(0.01 * torch.randn((self.q_proj_dim, d_model)))
        self.q_layernorm = torch.nn.LayerNorm(self.q_proj_dim)

        # ===== KV 投影层 =====
        # 公式:  c_t^{KV} = W^{DKV} h_t
        self.W_dkv = torch.nn.Parameter(0.01 * torch.randn((d_model, self.kv_proj_dim + self.qk_rope_dim)))
        # 公式:  [K^C, V^C] = W^{UKV} c_t^{KV}
        self.W_ukv = torch.nn.Parameter(
            0.01 * torch.randn((self.kv_proj_dim, d_model + (n_heads * self.qk_nope_dim)))
        )
        self.kv_layernorm = torch.nn.LayerNorm(self.kv_proj_dim)

        # 输出投影:  u_t = W^O [o_{t,1};…;o_{t,n_h}]
        self.W_o = torch.nn.Parameter(0.01 * torch.randn((d_model, d_model)))

        # ===== RoPE 缓存 =====
        freqs = 1.0 / (rope_theta ** (torch.arange(0, self.dh, 2).float() / self.dh))
        emb = torch.outer(torch.arange(max_len).float(), freqs)
        cos_cached = emb.cos()[None, None, :, :]
        sin_cached = emb.sin()[None, None, :, :]
        self.register_buffer("cos_cached", cos_cached)
        self.register_buffer("sin_cached", sin_cached)

    # =======================================
    # 前向传播
    # =======================================
    def forward(self, x, kv_cache=None, past_length=0):
        """
        输入:  x ∈ ℝ^{B×S×d_model}
        输出:  u_t, c^{KV}
        """
        B, S, D = x.size()

        # -------------------------------------------------
        # Step1️⃣ KV 低秩压缩
        # -------------------------------------------------
        # 公式:  c_j^{KV} = W^{DKV} h_j
        if kv_cache is None:
            compressed_kv = x @ self.W_dkv
            # 拆分:  KV_for_lora → c^{KV};  K_for_rope → RoPE部分
            KV_for_lora, K_for_rope = torch.split(compressed_kv,
                                                  [self.kv_proj_dim, self.qk_rope_dim], dim=-1)
            KV_for_lora = self.kv_layernorm(KV_for_lora)
        else:
            # 推理阶段: 拼接旧缓存
            new_kv = x @ self.W_dkv
            compressed_kv = torch.cat([kv_cache, new_kv], dim=1)
            new_kv, new_K_for_rope = torch.split(new_kv, [self.kv_proj_dim, self.qk_rope_dim], dim=-1)
            old_kv, old_K_for_rope = torch.split(kv_cache, [self.kv_proj_dim, self.qk_rope_dim], dim=-1)
            new_kv = self.kv_layernorm(new_kv)
            old_kv = self.kv_layernorm(old_kv)
            KV_for_lora = torch.cat([old_kv, new_kv], dim=1)
            K_for_rope = torch.cat([old_K_for_rope, new_K_for_rope], dim=1)

        # -------------------------------------------------
        # Step2️⃣ 低秩恢复 Key/Value
        # -------------------------------------------------
        # 公式:  [K^C, V^C] = W^{UKV} c^{KV}
        KV = KV_for_lora @ self.W_ukv
        KV = KV.view(B, -1, self.n_heads, self.dh + self.qk_nope_dim).transpose(1, 2)
        # 拆分:  K^C, V^C
        K, V = torch.split(KV, [self.qk_nope_dim, self.dh], dim=-1)
        S_full = K.size(2)

        # -------------------------------------------------
        # Step3️⃣ 计算 RoPE Key
        # -------------------------------------------------
        # 公式:  k_j^R = RoPE(W^{KR} h_j)
        K_for_rope = K_for_rope.view(B, -1, 1, self.qk_rope_dim).transpose(1, 2)
        cos_k = self.cos_cached[:, :, :S_full, :self.qk_rope_dim // 2].repeat(1, 1, 1, 2)
        sin_k = self.sin_cached[:, :, :S_full, :self.qk_rope_dim // 2].repeat(1, 1, 1, 2)
        K_for_rope = apply_rope_x(K_for_rope, cos_k, sin_k)
        K_for_rope = K_for_rope.repeat(1, self.n_heads, 1, 1)

        # -------------------------------------------------
        # Step4️⃣ 计算 Query（含解耦 RoPE）
        # -------------------------------------------------
        # (a) 压缩:  c_t^Q = W^{DQ} h_t
        compressed_q = x @ self.W_dq
        compressed_q = self.q_layernorm(compressed_q)
        # (b) 恢复:  q_t^C = W^{UQ} c_t^Q
        Q = compressed_q @ self.W_uq
        Q = Q.view(B, -1, self.n_heads, self.dh).transpose(1, 2)
        # (c) 拆分:  Q=[Q_nope, Q_rope]
        Q, Q_for_rope = torch.split(Q, [self.qk_nope_dim, self.qk_rope_dim], dim=-1)
        # (d) 加 RoPE:  q_t^R = RoPE(W^{QR} c_t^Q)
        cos_q = self.cos_cached[:, :, past_length:past_length + S, :self.qk_rope_dim // 2].repeat(1, 1, 1, 2)
        sin_q = self.sin_cached[:, :, past_length:past_length + S, :self.qk_rope_dim // 2].repeat(1, 1, 1, 2)
        Q_for_rope = apply_rope_x(Q_for_rope, cos_q, sin_q)

        # -------------------------------------------------
        # Step5️⃣ 拼接解耦分支
        # -------------------------------------------------
        # 公式:  q_t = [q_t^C ; q_t^R],  k_j = [k_j^C ; k_j^R]
        q_heads = torch.cat([Q, Q_for_rope], dim=-1)
        k_heads = torch.cat([K, K_for_rope], dim=-1)
        v_heads = V

        # -------------------------------------------------
        # Step6️⃣ 注意力打分与加权
        # -------------------------------------------------
        # 公式:  α_{t,j} = softmax_j( (q_t k_j^T)/√(d_h+d_h^R) )
        mask = torch.ones((S, S_full), device=x.device)
        mask = torch.tril(mask, diagonal=past_length)
        sq_mask = mask[None, None, :, :] == 1
        x_out = F.scaled_dot_product_attention(q_heads, k_heads, v_heads, attn_mask=sq_mask)
        # 公式:  o_t = Σ_j α_{t,j} v_j^C
        x_out = x_out.transpose(1, 2).reshape(B, S, D)
        # 输出:  u_t = W^O [o_{t,1};…;o_{t,n_h}]
        x_out = x_out @ self.W_o.T

        return x_out, compressed_kv


# =======================================
# 🔹 调试入口
# =======================================
def main():
    torch.manual_seed(42)
    d_model, n_heads, seq_len, batch = 256, 8, 8, 2
    model = MLA(d_model=d_model, n_heads=n_heads, max_len=128)
    x = torch.randn(batch, seq_len, d_model)
    out, kv = model(x)
    print(f"✅ 输入: {x.shape} → 输出: {out.shape}, 缓存: {kv.shape}")

if __name__ == "__main__":
    main()
