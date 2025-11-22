import numpy as np
import matplotlib.pyplot as plt

# =============================
# 生成模拟数据
# =============================
np.random.seed(0)

X = np.random.randn(100, 3)
true_w = np.array([2.0, -3.0, 1.0])
y = (X @ true_w + np.random.randn(100) * 0.5).reshape(-1, 1)

# 通用参数设置
lr = 0.1
lambd = 0.1
epochs = 1000
m = X.shape[0]

# =============================
# L2 正则化（Ridge Regression）
# =============================
w_l2 = np.zeros((3, 1))
b_l2 = 0.0
loss_history_l2 = []

for epoch in range(epochs):
    y_pred = X @ w_l2 + b_l2
    error = y_pred - y

    # 计算梯度
    dw = (X.T @ error) / m + lambd * w_l2
    db = np.sum(error) / m

    # 参数更新
    w_l2 -= lr * dw
    b_l2 -= lr * db

    # 计算损失 (包含正则项)
    loss = (np.sum(error ** 2) / (2 * m)) + (lambd / 2) * np.sum(w_l2 ** 2)
    loss_history_l2.append(loss)

print("learned w (L2):", w_l2.ravel())
print("learned b (L2):", b_l2)
print("true w:", true_w)

# =============================
# L1 正则化（Lasso Regression）
# =============================
w_l1 = np.zeros((3, 1))
b_l1 = 0.0
loss_history_l1 = []

for epoch in range(epochs):
    y_pred = X @ w_l1 + b_l1
    error = y_pred - y

    dw = (X.T @ error) / m + lambd * np.sign(w_l1)
    db = np.sum(error) / m

    w_l1 -= lr * dw
    b_l1 -= lr * db

    # 计算损失 (包含正则项)
    # (np.sum(error ** 2) / (2 * m)) MSE 损失 + 正则化损失
    loss = (np.sum(error ** 2) / (2 * m)) + lambd * np.sum(np.abs(w_l1))
    loss_history_l1.append(loss)

print("真实权重:", true_w)
print("L2 正则化权重（Ridge）:", w_l2.ravel())
print("L1 正则化权重（Lasso）:", w_l1.ravel())

# =============================
# 可视化损失函数收敛过程
# =============================
plt.figure(figsize=(8, 5))
plt.plot(loss_history_l2, label="L2 Loss (Ridge)")
plt.plot(loss_history_l1, label="L1 Loss (Lasso)")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("L1 vs L2 Regularization Loss Convergence")
plt.legend()
plt.grid(True)
plt.show()
