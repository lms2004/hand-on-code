import numpy as np

# ---------------------------------------------------
# Generate Random Subsets Function (官方标准解)
# ---------------------------------------------------
def get_random_subsets(X, y, n_subsets, replacements=True, seed=None):
    """
    Generate random subsets of a dataset.

    Each subset contains 50% of the samples (floor(n/2)),
    sampled with or without replacement according to `replacements`.
    """
    if seed is not None:
        np.random.seed(seed)

    n_samples = len(X)
    subset_size = n_samples // 2   # 根据题目示例：每个子集取 50%

    subsets = []
    for _ in range(n_subsets):
        indices = np.random.choice(
            n_samples,
            size=subset_size,
            replace=replacements
        )
        subsets.append((X[indices], y[indices]))

    return subsets


# ---------------------------------------------------
# Test Case Runner
# ---------------------------------------------------
def run_tests():

    print("\n================== Test Case 1 ==================")
    print("replacements = False  (无放回, 50% 样本)")

    X1 = np.array([[1, 2],
                   [3, 4],
                   [5, 6],
                   [7, 8],
                   [9, 10]])
    y1 = np.array([1, 2, 3, 4, 5])

    expected1 = [
        (np.array([[3, 4], [9, 10]]), np.array([2, 5])),
        (np.array([[7, 8], [3, 4]]), np.array([4, 2])),
        (np.array([[3, 4], [1, 2]]), np.array([2, 1]))
    ]

    result1 = get_random_subsets(X1, y1, 3, replacements=False, seed=42)

    print("Expected  预期")
    print(expected1)
    print("\nYour Output 《深度机器学习 | 练习题》")
    print(result1)

    passed1 = all(
        np.array_equal(result1[i][0], expected1[i][0]) and
        np.array_equal(result1[i][1], expected1[i][1])
        for i in range(3)
    )
    print("Passed  通过" if passed1 else "Failed  未通过")


    print("\n================== Test Case 2 ==================")
    print("replacements = True  (有放回, 50% 样本)")

    X2 = np.array([[1, 1],
                   [2, 2],
                   [3, 3],
                   [4, 4]])
    y2 = np.array([10, 20, 30, 40])

    expected2 = [
        (np.array([[3, 3], [4, 4]]), np.array([30, 40]))
    ]

    result2 = get_random_subsets(X2, y2, 1, replacements=True, seed=42)

    print("Expected  预期")
    print(expected2)
    print("\nYour Output 《深度机器学习 | 练习题》")
    print(result2)

    passed2 = (
        np.array_equal(result2[0][0], expected2[0][0]) and
        np.array_equal(result2[0][1], expected2[0][1])
    )
    print("Passed  通过" if passed2 else "Failed  未通过")


# ---------------------------------------------------
# Run Tests
# ---------------------------------------------------
if __name__ == "__main__":
    run_tests()
