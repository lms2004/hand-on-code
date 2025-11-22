import numpy as np

def recall(y_true, y_pred):
    """
    Compute recall for binary classification.

    Recall = TP / (TP + FN)
    If denominator is 0, return 0.0
    """

    # True Positive: actual=1 & pred=1
    TP = np.sum((y_true == 1) & (y_pred == 1))

    # False Negative: actual=1 & pred=0
    FN = np.sum((y_true == 1) & (y_pred == 0))

    # Avoid division by zero
    if TP + FN == 0:
        return 0.0

    recall_value = TP / (TP + FN)
    return round(recall_value, 3)


# =====================================================
# Test Case 1（你给的第一个测试样例）
# Expected: 1.0
# =====================================================
y_true = np.array([1, 0, 1, 1, 0, 0])
y_pred = np.array([1, 0, 1, 1, 0, 0])
print(recall(y_true, y_pred))


# =====================================================
# Test Case 2（你给的第二个测试样例）
# Expected: 0.333
# =====================================================
y_true = np.array([1, 0, 1, 1, 0, 0])
y_pred = np.array([1, 0, 0, 0, 0, 1])
print(recall(y_true, y_pred))
