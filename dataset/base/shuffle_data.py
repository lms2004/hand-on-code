import numpy as np

# ---------------------------------------------------
# Random Shuffle Function
# ---------------------------------------------------
def shuffle_data(X, y, seed=None):
    """
    Randomly shuffle X and y while keeping correspondence.
    """

    if seed is not None:
        np.random.seed(seed)

    indices = np.arange(len(X))
    np.random.shuffle(indices)

    return X[indices], y[indices]


# ---------------------------------------------------
# Test Case Runner
# ---------------------------------------------------
def run_tests():
    print("Test Case  测试用例")
    print("print(shuffle_data(np.array([[1, 2], [3, 4], [5, 6], [7, 8]]), np.array([1, 2, 3, 4]), seed=42))")
    print("打印(打乱数据(np.数组([[1, 2], [3, 4], [5, 6], [7, 8]]), np.数组([1, 2, 3, 4]), 种子=42))")

    X = np.array([[1, 2], 
                  [3, 4], 
                  [5, 6], 
                  [7, 8]])
    y = np.array([1, 2, 3, 4])

    expected_X = np.array([[3, 4],
                           [7, 8],
                           [1, 2],
                           [5, 6]])
    expected_y = np.array([2, 4, 1, 3])

    print("Expected  预期")
    print((expected_X, expected_y))

    # Your Output
    X_out, y_out = shuffle_data(X, y, seed=42)
    print("Your Output  《深度机器学习 | 练习题》")
    print((X_out, y_out))

    passed = np.array_equal(X_out, expected_X) and np.array_equal(y_out, expected_y)
    print("Passed  通过" if passed else "Failed  未通过")


# ---------------------------------------------------
# Run Tests
# ---------------------------------------------------
if __name__ == "__main__":
    run_tests()
