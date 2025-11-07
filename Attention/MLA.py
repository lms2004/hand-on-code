#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DeepSeek-V2 MLA Demo — 行内公式注释版（完整版）
验证 MLA 的低秩键值压缩 + 解耦 RoPE
作者：ChatGPT (GPT-5)
"""

import torch
import torch.nn.functional as F


# =========================================================
# 🔹 RoPE 旋转函数
# 公式: RoPE(x) = x·cosθ + rotate(x)·sinθ
# =========================================================
def apply_rope_x(x, cos, sin):
    x1, x2 = x[..., ::2], x[..., 1::2]                            # 拆分偶/奇维度 → x=[x₁, x₂]
    x_rot = torch.stack([-x2, x1], dim=-1).reshape_as(x)          # rotate(x) = [-x₂, x₁]
    return x * cos + x_rot * sin                                  # 式(R): RoPE(x)=x·cosθ+rotate(x)·sinθ


# =========================================================
# 🔹 MLA 模块
# =========================================================
class MLA(torch.nn.Module):
    def __init__(self, d_model, n_heads, max_len=1024, rope_theta=10000.0):
        super().__init__()
        self.d_model = d_model                                     # 模型维度 d_model
        self.n_heads = n_heads                                     # 注意力头数 n_h
        self.dh = d_model // n_heads                               # 每个 head 的维度 d_h

        self.q_proj_dim = d_model // 2                             # 式(QD): c_t^{Q} = W^{DQ} h_t
        self.kv_proj_dim = (2 * d_model) // 3                      # 式(9):  c_t^{KV} = W^{DKV} h_t

        self.qk_nope_dim = self.dh // 2                            # 不带RoPE部分 → q_t^{C}, k_j^{C}
        self.qk_rope_dim = self.dh // 2                            # 带RoPE部分 → q_t^{R}, k_j^{R}

        # ===== Q投影 =====
        self.W_dq = torch.nn.Parameter(0.01 * torch.randn((d_model, self.q_proj_dim)))   # 式(QD): 低秩压缩矩阵 W^{DQ}
        self.W_uq = torch.nn.Parameter(0.01 * torch.randn((self.q_proj_dim, d_model)))   # 式(QU): 低秩恢复矩阵 W^{UQ}
        self.q_layernorm = torch.nn.LayerNorm(self.q_proj_dim)                           # LayerNorm(c_t^{Q})

        # ===== KV投影 =====
        self.W_dkv = torch.nn.Parameter(0.01 * torch.randn((d_model, self.kv_proj_dim + self.qk_rope_dim)))  
        # 式(9): [c_t^{KV}, k_{raw,t}^{R}] = W^{DKV} h_t

        self.W_ukv = torch.nn.Parameter(
            0.01 * torch.randn((self.kv_proj_dim, d_model + (n_heads * self.qk_nope_dim)))
        )  # 式(10)+(11): [K^{C}, V^{C}] = W^{UKV} c_t^{KV}
        self.kv_layernorm = torch.nn.LayerNorm(self.kv_proj_dim)    # ˆc_t^{KV} = LayerNorm(c_t^{KV})

        # ===== 输出投影 =====
        self.W_o = torch.nn.Parameter(0.01 * torch.randn((d_model, d_model)))            # 式(U): u_t = W^{O} o_t

        # ===== RoPE 缓存 =====
        freqs = 1.0 / (rope_theta ** (torch.arange(0, self.dh, 2).float() / self.dh))    # θ_i = θ^{-2i/d_h}
        emb = torch.outer(torch.arange(max_len).float(), freqs)                          # θ_{pos,i} = pos * freq_i
        cos_cached = emb.cos()[None, None, :, :]                                         # cosθ
        sin_cached = emb.sin()[None, None, :, :]                                         # sinθ
        self.register_buffer("cos_cached", cos_cached)
        self.register_buffer("sin_cached", sin_cached)

    # =========================================================
    # 🔹 前向传播
    # =========================================================
    def forward(self, x, kv_cache=None, past_length=0):
        B, S, D = x.size()                                                              # 输入 h_t ∈ ℝ^{B×S×d_model}

        # -----------------------------------------------------
        # Step1️⃣ KV 低秩压缩
        # 式(9): [c_t^{KV}, k_{raw,t}^{R}] = W^{DKV} h_t
        # -----------------------------------------------------
        if kv_cache is None:
            compressed_kv = x @ self.W_dkv                                              # 式(9): 计算 W^{DKV} h_t
            KV_for_lora, K_for_rope = torch.split(compressed_kv, [self.kv_proj_dim, self.qk_rope_dim], dim=-1)
            KV_for_lora = self.kv_layernorm(KV_for_lora)                                # ˆc_t^{KV} = LN(c_t^{KV})
        else:
            new_kv = x @ self.W_dkv                                                     # 式(9): 当前 token 的 [c_t^{KV}, k_{raw,t}^{R}]
            compressed_kv = torch.cat([kv_cache, new_kv], dim=1)                        # 拼接缓存 → [c_{1:t}^{KV}, k_{raw,1:t}^{R}]
            new_kv, new_Kr = torch.split(new_kv, [self.kv_proj_dim, self.qk_rope_dim], dim=-1)
            old_kv, old_Kr = torch.split(kv_cache, [self.kv_proj_dim, self.qk_rope_dim], dim=-1)
            new_kv = self.kv_layernorm(new_kv)                                          # ˆc_t^{KV}
            old_kv = self.kv_layernorm(old_kv)                                          # ˆc_{1:t-1}^{KV}
            KV_for_lora = torch.cat([old_kv, new_kv], dim=1)                            # 拼接低秩潜向量序列
            K_for_rope = torch.cat([old_Kr, new_Kr], dim=1)                             # 拼接RoPE原始键分支

        # -----------------------------------------------------
        # Step2️⃣ 从低秩潜向量恢复 Key/Value
        # 式(10) + (11): [K^{C}, V^{C}] = W^{UKV} c^{KV}
        # -----------------------------------------------------
        KV = KV_for_lora @ self.W_ukv                                                   # 应用恢复矩阵 W^{UKV}
        KV = KV.view(B, -1, self.n_heads, self.dh + self.qk_nope_dim).transpose(1, 2)   # reshape成[B,nH,S,dh+dh_nope]
        K, V = torch.split(KV, [self.qk_nope_dim, self.dh], dim=-1)                     # 分得 K^{C}, V^{C}
        S_full = K.size(2)                                                              # 全序列长度（含历史）

        # -----------------------------------------------------
        # Step3️⃣ 计算 RoPE 键分支
        # 式(R): k_j^{R} = RoPE(k_{raw,j}^{R})
        # -----------------------------------------------------
        K_for_rope = K_for_rope.view(B, -1, 1, self.qk_rope_dim).transpose(1, 2)        # [B,1,S_full,D_rope]
        cos_k = self.cos_cached[:, :, :S_full, :self.qk_rope_dim // 2].repeat(1, 1, 1, 2)
        sin_k = self.sin_cached[:, :, :S_full, :self.qk_rope_dim // 2].repeat(1, 1, 1, 2)
        K_for_rope = apply_rope_x(K_for_rope, cos_k, sin_k)                             # k_j^{R} = RoPE(k_{raw,j}^{R})
        K_for_rope = K_for_rope.repeat(1, self.n_heads, 1, 1)                           # 复制到所有 head

        # -----------------------------------------------------
        # Step4️⃣ 计算 Query
        # 式(QD): c_t^{Q} = W^{DQ} h_t
        # 式(QU): q_t^{C} = W^{UQ} c_t^{Q}
        # 式(R):  q_t^{R} = RoPE(q_{raw,t}^{R})
        # -----------------------------------------------------
        compressed_q = x @ self.W_dq                                                    # 式(QD): 计算低秩压缩向量 c_t^{Q}
        compressed_q = self.q_layernorm(compressed_q)                                   # ˆc_t^{Q}
        Q = compressed_q @ self.W_uq                                                    # 式(QU): 计算恢复向量 q_t^{C}
        Q = Q.view(B, -1, self.n_heads, self.dh).transpose(1, 2)                        # [B,nH,S,dh]
        Q, Q_for_rope = torch.split(Q, [self.qk_nope_dim, self.qk_rope_dim], dim=-1)    # 拆分 q_t^{C}, q_t^{R}
        cos_q = self.cos_cached[:, :, past_length:past_length + S, :self.qk_rope_dim // 2].repeat(1, 1, 1, 2)
        sin_q = self.sin_cached[:, :, past_length:past_length + S, :self.qk_rope_dim // 2].repeat(1, 1, 1, 2)
        Q_for_rope = apply_rope_x(Q_for_rope, cos_q, sin_q)                             # q_t^{R} = RoPE(q_{raw,t}^{R})

        # -----------------------------------------------------
        # Step5️⃣ 拼接解耦分支
        # 式: q_t = [q_t^{C}; q_t^{R}],  k_j = [k_j^{C}; k_j^{R}],  v_j = v_j^{C}
        # -----------------------------------------------------
        q_heads = torch.cat([Q, Q_for_rope], dim=-1)                                    # 拼接得到完整 q_t
        k_heads = torch.cat([K, K_for_rope], dim=-1)                                    # 拼接得到完整 k_j
        v_heads = V                                                                     # v_j 不变

        # -----------------------------------------------------
        # Step6️⃣ 注意力计算
        # 式(A): α_{t,j} = softmax_j( (q_t k_j^T)/√d )
        # 式(O): o_t = Σ_j α_{t,j} v_j
        # -----------------------------------------------------
        mask = torch.ones((S, S_full), device=x.device)                                 # 构造因果mask
        mask = torch.tril(mask, diagonal=past_length)
        sq_mask = mask[None, None, :, :] == 1
        x_out = F.scaled_dot_product_attention(q_heads, k_heads, v_heads, attn_mask=sq_mask)  # o_t = Σ α_{t,j} v_j
        x_out = x_out.transpose(1, 2).reshape(B, S, D)                                  # 合并所有head输出
        x_out = x_out @ self.W_o.T                                                      # 式(U): u_t = W^{O} o_t

        return x_out, compressed_kv


# =========================================================
# 🔹 调试入口
# =========================================================
def main():
    torch.manual_seed(42)
    d_model, n_heads, seq_len, batch = 256, 8, 8, 2
    model = MLA(d_model=d_model, n_heads=n_heads, max_len=128)
    x = torch.randn(batch, seq_len, d_model)
    out, kv = model(x)
    print(f"✅ 输入: {x.shape} → 输出: {out.shape}, 缓存: {kv.shape}")

if __name__ == "__main__":
    main()
