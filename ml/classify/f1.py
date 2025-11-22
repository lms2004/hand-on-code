import numpy as np

# ---------------------------------------------------
# F-Score Function
# ---------------------------------------------------
def f_score(y_true, y_pred, beta):
    """
    Calculate F-Score for a binary classification task.

    :param y_true: Numpy array of true labels
    :param y_pred: Numpy array of predicted labels
    :param beta: The weight of precision in the harmonic mean
    :return: F-Score rounded to three decimal places
    """

    # True Positives
    TP = np.sum((y_true == 1) & (y_pred == 1))

    # False Positives
    FP = np.sum((y_true == 0) & (y_pred == 1))

    # False Negatives
    FN = np.sum((y_true == 1) & (y_pred == 0))

    # Precision
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0

    # Recall
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0

    # F-beta Score
    beta2 = beta ** 2
    if precision + recall == 0:
        f = 0.0
    else:
        f = (1 + beta2) * precision * recall / (beta2 * precision + recall)

    return round(f, 3)


# ---------------------------------------------------
# Test Case Runner
# ---------------------------------------------------
def run_tests():
    print("Test Case  测试用例")

    # Input
    y_true = np.array([1, 0, 1, 1, 0, 1])
    y_pred = np.array([1, 0, 1, 0, 0, 1])
    beta = 1

    print("import numpy as np y_true = np.array([1, 0, 1, 1, 0, 1]) y_pred = np.array([1, 0, 1, 0, 0, 1]) beta = 1 print(f_score(y_true, y_pred, beta))")

    # Expected
    expected = 0.857
    print("Expected  预期")
    print(expected)

    # Your Output
    result = f_score(y_true, y_pred, beta)
    print("Your Output  《深度机器学习 | 练习题》")
    print(result)

    # Compare
    print("Passed  通过" if result == expected else "Failed  未通过")


# ---------------------------------------------------
# Run Tests
# ---------------------------------------------------
if __name__ == "__main__":
    run_tests()
