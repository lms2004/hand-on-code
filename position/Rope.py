#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🔹 Rotary Position Embedding (RoPE) Demo
---------------------------------------
基于 Llama3 实现的复数形式 RoPE 示例
公式参考：
f(q_m, m) = q_m e^{i m θ}

其中：
  - q_m: 第 m 个 token 的查询向量
  - θ: 不同维度对应的频率角
  - e^{i m θ}: 通过复数旋转实现相对位置编码
"""

import torch
import numpy as np


# ==============================================================
# 🧩 1️⃣ 预计算旋转频率 (公式对应: e^{i m θ})
# ==============================================================

def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    """
    返回预计算的频率tensor,形状为 (end, dim // 2),数据类型为complex64(复数)
    
    数学公式：
        θ_k = 1 / θ^{(2k / d)}
        freqs[m, k] = m * θ_k
        freqs_cis[m, k] = e^{i * freqs[m, k]}
    """
    # (1) 每两个维度共享同一个频率分量 θ_k
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    # (2) 序列位置索引 t = [0, 1, ..., end-1]
    t = torch.arange(end, device=freqs.device, dtype=torch.float32)
    # (3) 外积：生成每个位置的角度 mθ_k  →  freqs[m, k] = t[m] * freqs[k]
    freqs = torch.outer(t, freqs)
    # (4) 复数形式: e^{iθ} = cosθ + i·sinθ
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis


# ==============================================================
# 🧩 2️⃣ 广播形状对齐 (辅助函数)
# ==============================================================

def reshape_for_broadcast(freqs_cis, x):
    """
    调整 freqs_cis 形状，使其可与 Q/K 广播匹配
    数学意义：让每个批次与 head 共享相同的旋转角
    """
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)


# ==============================================================
# 🧩 3️⃣ 应用 RoPE 旋转 (核心公式)
# ==============================================================

def apply_rotary_emb(xq, xk, freqs_cis):
    """
    对输入 Q, K 向量进行旋转位置编码
    
    数学公式：
        q'_m = q_m e^{i m θ}
        k'_m = k_m e^{i m θ}
    """
    # (1) 将实数对 [q_{2i}, q_{2i+1}] 转为复数 q_i = q_{2i} + i·q_{2i+1}
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    # (2) 对齐广播形状
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    # (3) 复数乘法实现旋转: q'_m = q_m × e^{i m θ}
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    # (4) 转回原始类型
    return xq_out.type_as(xq), xk_out.type_as(xk)


# ==============================================================
# 🧩 4️⃣ Demo: 验证 RoPE 旋转效果
# ==============================================================

if __name__ == "__main__":
    torch.manual_seed(0)

    # 模拟输入：batch=1, seq_len=4, num_heads=1, head_dim=8
    B, L, H, D = 1, 4, 1, 8
    xq = torch.randn(B, L, H, D)
    xk = torch.randn(B, L, H, D)

    print("原始 Q 向量：")
    print(xq[0, :, 0])

    # 预计算频率 (相当于 e^{iθ})
    freqs_cis = precompute_freqs_cis(dim=D, end=L)
    print("\n预计算旋转频率 freqs_cis（前2个位置示例）:")
    print(freqs_cis[:2])

    # 应用 RoPE
    xq_rot, xk_rot = apply_rotary_emb(xq, xk, freqs_cis)

    print("\n旋转后 Q' 向量：")
    print(xq_rot[0, :, 0])

    # 验证旋转前后模长是否一致（仅旋转，不改变幅度）
    orig_norm = torch.norm(xq, dim=-1)
    new_norm = torch.norm(xq_rot, dim=-1)
    print("\n模长变化（应几乎相等）:")
    print(torch.allclose(orig_norm, new_norm, atol=1e-5))
