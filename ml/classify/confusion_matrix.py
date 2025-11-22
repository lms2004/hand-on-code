from collections import Counter

# ------------------------------------
# 1. 定义 confusion_matrix 函数
# ------------------------------------
def confusion_matrix(data):
    TP = FN = FP = TN = 0

    for y_true, y_pred in data:
        if y_true == 1 and y_pred == 1:
            TP += 1
        elif y_true == 1 and y_pred == 0:
            FN += 1
        elif y_true == 0 and y_pred == 1:
            FP += 1
        elif y_true == 0 and y_pred == 0:
            TN += 1

    return [[TP, FN], [FP, TN]]

# ------------------------------------
# 2. 定义测试函数（自动验证）
# ------------------------------------
def run_test(test_name, data, expected):
    print(f"=== {test_name} ===")

    result = confusion_matrix(data)
    print("Your Output:   ", result)
    print("Expected:      ", expected)

    if result == expected:
        print("Result:        PASSED ✔\n")
    else:
        print("Result:        FAILED ✘\n")


# ------------------------------------
# 3. 测试用例 1（题目示例）
# ------------------------------------
data1 = [[1, 1], [1, 0], [0, 1], [0, 0], [0, 1]]
expected1 = [[1, 1], [2, 1]]

run_test("Test Case 1", data1, expected1)


# ------------------------------------
# 4. 测试用例 2（你提供的测试）
# ------------------------------------
data2 = [
    [0, 1], [1, 0], [1, 1], [0, 1], [0, 0], [1, 0], [0, 1],
    [1, 1], [0, 0], [1, 0], [1, 1], [0, 0], [1, 0], [0, 1],
    [1, 1], [1, 1], [1, 0]
]
expected2 = [[5, 5], [4, 3]]

run_test("Test Case 2", data2, expected2)
