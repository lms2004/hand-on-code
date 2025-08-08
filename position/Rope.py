import torch

# 🧠 用于预计算 RoPE 频率：𝜃⁽ⁱ⁾ = 1 / θ^(i / d), 构造旋转相位因子 e^{jθ}
def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    """
    🧮 预计算旋转频率的复数表示
    📐 输出形状: (end, dim // 2)，复数值 e^{jωt}
    """

    # 🔢 构造频率尺度: 𝜔ₖ = 1 / θ^(k/d)，其中 k = 0, 2, ..., dim-2（取一半频率）
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))

    # 🕒 生成时间步序列: t = [0, 1, 2, ..., end-1]
    t = torch.arange(end, device=freqs.device, dtype=torch.float32)

    # 🔁 外积得到所有时间与频率组合: tᵢ × 𝜔ⱼ，对应每个位置和频率的旋转角度 θᵢⱼ
    freqs = torch.outer(t, freqs)  # shape: (end, dim//2)

    # 🔄 转换为复数 e^{jθ} = cosθ + jsinθ，模长为1，极角为频率×位置
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # 📌 complex64
    return freqs_cis

# 🎛️ 为了广播，将 freq_cis reshape 成与 xq/xk 可对齐的形状
def reshape_for_broadcast(freqs_cis, x):
    """
    🧩 调整频率矩阵 freqs_cis 的形状，使其可广播到 x 的 shape
    💡 实际是将 (seq_len, head_dim//2) → (1, seq_len, 1, head_dim//2)
    """
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])  # (seq_len, head_dim//2)

    # 🧱 构造广播形状：在除 seq_len 和 dim 外插入 1
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)

# ✨ 位置编码核心：将旋转频率 freqs_cis 应用于 Q、K 向量
def apply_rotary_emb(xq, xk, freqs_cis):
    """
    📐 输入：
        xq, xk: [batch, seq_len, n_heads, head_dim]
        freqs_cis: [seq_len, head_dim // 2] (复数旋转因子)
    📤 输出：
        应用旋转后的 xq, xk，shape 相同
    """

    # 🧊 先将最后的 head_dim 视作复数：实+虚，变为 [*, head_dim//2] 复数
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))

    # 🧩 调整频率形状以支持广播：从 [seq_len, dim//2] → [1, seq_len, 1, dim//2]
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)

    # 💫 复数乘法：相当于平面旋转操作 (cosθ + jsinθ)，将位置信息编码入向量
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)  # ↩️ 转回实数向量
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)

    # 🔁 返回原始类型（float16/float32），保持和输入一致
    return xq_out.type_as(xq), xk_out.type_as(xk)

def main():
    # 🧾 模拟参数配置
    batch_size = 2
    seq_len = 4
    n_heads = 2
    head_dim = 8  # 必须是偶数（才能被转换为复数）

    # 🌟 构造模拟的 Q、K 向量（可以视作 transformer attention 的输入）
    xq = torch.randn(batch_size, seq_len, n_heads, head_dim)
    xk = torch.randn(batch_size, seq_len, n_heads, head_dim)

    # 📐 预计算 RoPE 旋转频率 e^{jθ}
    freqs_cis = precompute_freqs_cis(head_dim, seq_len)

    # 🌀 应用旋转位置编码（RoPE）
    xq_rot, xk_rot = apply_rotary_emb(xq, xk, freqs_cis)

    # 📊 打印输出对比（仅展示前几个元素）
    print("Original Q (before RoPE):\n", xq[0, 0])
    print("\nRotated Q (after RoPE):\n", xq_rot[0, 0])
    print("\nOriginal K (before RoPE):\n", xk[0, 0])
    print("\nRotated K (after RoPE):\n", xk_rot[0, 0])

if __name__ == "__main__":
    main()