import numpy as np

def compute_f1(y_true, y_pred):
    """
    🎯 计算 F1 分数，用于二分类任务评价模型性能
    包括 Precision（查准率）、Recall（召回率）和 F1（调和平均）
    
    示例：
    y_true = [1, 0, 1, 1, 0]
    y_pred = [1, 1, 1, 0, 0]
    
    TP = 2（预测为1，实际也为1）
    FP = 1（预测为1，实际为0）
    FN = 1（预测为0，实际为1）

    Precision = 2 / (2 + 1) = 0.6667
    Recall = 2 / (2 + 1) = 0.6667
    F1 = 2 * P * R / (P + R) = 0.6667
    """

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # ✅ TP: 真正例（预测=1 且 真实=1）
    TP = np.sum((y_pred == 1) & (y_true == 1))

    # ✅ FP: 假正例（预测=1 但真实=0）
    FP = np.sum((y_pred == 1) & (y_true == 0))

    # ✅ FN: 假负例（预测=0 但真实=1）
    FN = np.sum((y_pred == 0) & (y_true == 1))

    # 🧠 Precision（查准率）= TP / (TP + FP)
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0

    # 🧠 Recall（召回率）= TP / (TP + FN)
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0

    # 🧠 F1-score = 2 * P * R / (P + R)
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return precision, recall, f1

import numpy as np

def myF1(y_true, y_pred):
    """
    自定义 F1 分数计算函数（适用于二分类任务）

    输入：
        y_true: 真实标签（list 或 array），如 [1, 0, 1]
        y_pred: 预测标签（list 或 array），如 [1, 1, 0]

    返回：
        precision, recall, f1_score 三个浮点数
    """

    # ✅ 第一步：转换为 NumPy 数组，支持向量化逻辑计算
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # ✅ 第二步：计算 TP / FP / FN
    TP = np.sum((y_true == 1) & (y_pred == 1))  # 真正例：预测 1 且实际也是 1
    FP = np.sum((y_true == 0) & (y_pred == 1))  # 假正例：预测 1 但实际是 0
    FN = np.sum((y_true == 1) & (y_pred == 0))  # 假负例：预测 0 但实际是 1

    # ⚠️ 易错点 1：括号必须加！否则计算顺序错了
    # 错误写法：TP / TP + FP → 实际执行为 (TP / TP) + FP
    precision = TP / (TP + FP) if TP + FP > 0 else 0
    recall = TP / (TP + FN) if TP + FN > 0 else 0
    F1 = 2 * recall * precision / (recall + precision) if (recall + precision) > 0 else 0
    return precision, recall, F1



y_true = [1, 0, 1, 1, 0]
y_pred = [1, 1, 1, 0, 0]

p1, r1, f1_1 = compute_f1(y_true, y_pred)
p2, r2, f1_2 = myF1(y_true, y_pred)

print(f"[compute_f1] Precision: {p1:.4f}, Recall: {r1:.4f}, F1 Score: {f1_1:.4f}")
print(f"[myF1      ] Precision: {p2:.4f}, Recall: {r2:.4f}, F1 Score: {f1_2:.4f}")

