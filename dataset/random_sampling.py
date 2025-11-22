import numpy as np

# ---------------------------------------------------
# Generate Random Subsets Function (官方标准解)
# ---------------------------------------------------
import numpy as np

def get_random_subsets(X, y, n_subsets, replacements=True, seed=42):
    np.random.seed(seed)

    n, m = X.shape
    
    subset_size = n if replacements else n // 2
    idx = np.array([np.random.choice(n, subset_size, replace=replacements) for _ in range(n_subsets)])
    # convert all ndarrays to lists
    return [(X[idx][i].tolist(), y[idx][i].tolist()) for i in range(n_subsets)]
    

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

    expected1 = [[[3, 4], [9, 10]], [2, 5], [[7, 8], [3, 4]], [4, 2], [[3, 4], [1, 2]], [2, 1]]

    result1 = get_random_subsets(X1, y1, 3, replacements=False, seed=42)

    print("Expected  预期")
    print(expected1)
    print("\nYour Output 《深度机器学习 | 练习题》")
    print(result1)

    # Convert result to flat list format for comparison
    flat_result1 = []
    for x, y in result1:
        flat_result1.append(x)
        flat_result1.append(y)
    
    passed1 = flat_result1 == expected1
    print("Passed  通过" if passed1 else "Failed  未通过")


    print("\n================== Test Case 2 ==================")
    print("replacements = True  (有放回, 50% 样本)")

    X2 = np.array([[1, 1],
                   [2, 2],
                   [3, 3],
                   [4, 4]])
    y2 = np.array([10, 20, 30, 40])

    expected2 = [([[3, 3], [4, 4], [1, 1], [3, 3]], [30, 40, 10, 30])]

    result2 = get_random_subsets(X2, y2, 1, replacements=True, seed=42)

    print("Expected  预期")
    print(expected2)
    print("\nYour Output 《深度机器学习 | 练习题》")
    print(result2)

    passed2 = result2 == expected2
    print("Passed  通过" if passed2 else "Failed  未通过")


# ---------------------------------------------------
# Run Tests
# ---------------------------------------------------
if __name__ == "__main__":
    run_tests()
