import numpy as np

# ---------------------------------------------------
# K-Fold Cross-Validation Function
# ---------------------------------------------------
def k_fold_cross_validation(X: np.ndarray, y: np.ndarray, k=5, shuffle=True):
    n_samples = len(X)
    indices = np.arange(n_samples)

    if shuffle:
        np.random.shuffle(indices)

    # Compute fold sizes
    fold_sizes = np.full(k, n_samples // k, dtype=int)
    fold_sizes[:n_samples % k] += 1

    # Generate folds
    folds = []
    current = 0
    for fold_size in fold_sizes:
        start, stop = current, current + fold_size
        test_idx = indices[start:stop]
        train_idx = np.concatenate((indices[:start], indices[stop:]))
        folds.append((train_idx.tolist(), test_idx.tolist()))
        current = stop

    return folds


# ---------------------------------------------------
# Test Case Runner
# ---------------------------------------------------
def run_tests():
    print("Test 1  测试 1\n")
    print("Test 2  测试 2\n")
    print("Test 3  测试 3\n")
    print("Test 4  测试 4")

    print("Test Case  测试用例")

    np.random.seed(42)

    X = np.array([0,1,2,3,4,5,6,7,8,9])
    y = np.array([0,1,2,3,4,5,6,7,8,9])

    print("import numpy as np np.random.seed(42) print(k_fold_cross_validation(...))")

    # Expected output
    expected = [
        ([2, 3, 4, 5, 6, 7, 8, 9], [0, 1]),
        ([0, 1, 4, 5, 6, 7, 8, 9], [2, 3]),
        ([0, 1, 2, 3, 6, 7, 8, 9], [4, 5]),
        ([0, 1, 2, 3, 4, 5, 8, 9], [6, 7]),
        ([0, 1, 2, 3, 4, 5, 6, 7], [8, 9])
    ]

    print("Expected  预期")
    print(expected)

    # Your Output
    result = k_fold_cross_validation(X, y, k=5, shuffle=False)
    print("Your Output  《深度机器学习 | 练习题》")
    print(result)

    # Compare
    passed = result == expected
    print("Passed  通过" if passed else "Failed  未通过")


# ---------------------------------------------------
# Run Tests
# ---------------------------------------------------
if __name__ == "__main__":
    run_tests()
