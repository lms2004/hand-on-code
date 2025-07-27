import numpy as np

# 🎲 构造数据：y = 2x + 1 加噪声（模拟真实场景）
np.random.seed(0)
X = np.linspace(0, 10, 100)                     # 100 个输入样本
Y = 2 * X + 1 + np.random.randn(*X.shape) * 1.5 # 加噪声的线性函数

# 初始化参数
w = 0.0  # 初始斜率
b = 0.0  # 初始偏置
lr = 0.01  # 学习率（步长）
epochs = 1000  # 迭代次数
m = len(X)  # 样本数

# 🔁 梯度下降
for epoch in range(epochs):
    # 1️⃣ 预测值：ŷ = wx + b
    Y_pred = w * X + b

    # 2️⃣ 误差项：error = ŷ - y
    error = Y_pred - Y

    # 3️⃣ 计算梯度
    # 🎯 损失函数 J(w,b) = (1/2m) * Σ(wx + b - y)^2
    # 🧠 对 w 求偏导：∂J/∂w = (1/m) * Σ(error * x)
    dw = (1/m) * np.dot(error, X)

    # 🧠 对 b 求偏导：∂J/∂b = (1/m) * Σ(error)
    db = (1/m) * np.sum(error)

    # 4️⃣ 更新参数（梯度下降公式）
    # w = w - α * ∂J/∂w
    # b = b - α * ∂J/∂b
    w -= lr * dw
    b -= lr * db

    # 5️⃣ 计算当前损失，用于观察收敛
    # 🎯 Loss = (1/2m) * Σ(ŷ - y)^2
    loss = (1/(2*m)) * np.sum(error**2)

    if epoch % 10 == 0:
        print(f"Epoch {epoch:03d}: Loss={loss:.4f}, w={w:.4f}, b={b:.4f}")
