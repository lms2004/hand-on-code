import numpy as np
import matplotlib.pyplot as plt

# =============================
# 生成模拟数据
# =============================
np.random.seed(0)

X = np.random.randn(100, 3)                # X ∈ ℝ^{m×n}，输入矩阵 
true_w = np.array([2.0, -3.0, 1.0])        # 真实权重向量 w*
y = (X @ true_w + np.random.randn(100) * 0.5).reshape(-1, 1)  # y = Xw* + ε，添加噪声项 ε ~ N(0, 0.5²)

# 通用参数设置
lr = 0.1          # 学习率 α
lambd = 0.1       # 正则化强度 λ
epochs = 1000
m = X.shape[0]    # 样本数 m

# =============================
# L2 正则化（Ridge Regression）
# =============================
# 损失函数：
#   J(w, b) = (1/(2m)) * Σ_i (y_i - ŷ_i)^2  +  (λ/2) * ||w||²
# 其中 ŷ = Xw + b
# 梯度：
#   ∂J/∂w = (1/m) * Xᵀ (Xw + b - y)  +  λw
#   ∂J/∂b = (1/m) * Σ_i (ŷ_i - y_i)
# 偏置 b 不进行正则化
# =============================
w_l2 = np.zeros((3,1))    # 初始化 w = 0
b_l2 = 0.0            # 初始化 b = 0

for epoch in range(epochs):
    # ---------- 前向传播 ----------
    y_pred = X @ w_l2 + b_l2
    # 公式：ŷ = Xw + b

    # ---------- 计算误差 ----------
    error = y_pred - y
    # 公式：error = ŷ - y

    # ---------- 计算梯度 ----------
    dw = (X.T @ error) / m + lambd * w_l2
    # 公式：
    #   ∂J/∂w = (1/m) * Xᵀ (ŷ - y) + λw
    # 第一项来自平方误差项，第二项来自 L2 正则项 (λ/2)||w||² 的导数 = λw

    db = np.sum(error) / m
    # 公式：
    #   ∂J/∂b = (1/m) * Σ(ŷ_i - y_i)
    # 偏置项不正则化

    # ---------- 参数更新 ----------
    w_l2 -= lr * dw
    # 公式：w := w - α * ∂J/∂w

    b_l2 -= lr * db
    # 公式：b := b - α * ∂J/∂b

# 打印结果对比
print("learned w (L2):", w_l2)
print("learned b (L2):", b_l2)
print("true w:", true_w)


# =============================
# L1 正则化（Lasso Regression）
# =============================
# 损失函数：
#   J(w, b) = (1/(2m)) * Σ_i (y_i - ŷ_i)^2  +  λ * ||w||₁
# 梯度（次梯度）：
#   ∂J/∂w = (1/m) * Xᵀ (Xw + b - y)  +  λ * sign(w)
#   ∂J/∂b = (1/m) * Σ_i (ŷ_i - y_i)
# 其中 sign(w) 为符号函数：
#   sign(w_i) = { +1, 若 w_i > 0;  -1, 若 w_i < 0;  任意 ∈ [-1,1], 若 w_i = 0 }
# =============================
w_l1 = np.zeros((3,1))
b_l1 = 0.0

for epoch in range(epochs):
    # ---------- 前向传播 ----------
    y_pred = X @ w_l1 + b_l1
    # 公式：ŷ = Xw + b

    # ---------- 计算误差 ----------
    error = y_pred - y
    # 公式：error = ŷ - y

    # ---------- 计算梯度 ----------
    dw = (X.T @ error) / m + lambd * np.sign(w_l1)
    # 公式：
    #   ∂J/∂w = (1/m) * Xᵀ (ŷ - y) + λ sign(w)
    # 注意 sign(w) 在 w=0 处不可导，此处用次梯度近似

    db = np.sum(error) / m
    # 公式：
    #   ∂J/∂b = (1/m) * Σ(ŷ_i - y_i)

    # ---------- 参数更新 ----------
    w_l1 -= lr * dw
    # 公式：w := w - α * ∂J/∂w

    b_l1 -= lr * db
    # 公式：b := b - α * ∂J/∂b


# =============================
# 结果比较
# =============================
print("真实权重:\n", true_w)
print("L2 正则化权重（Ridge）:\n", w_l2)
print("L1 正则化权重（Lasso）:\n", w_l1)
