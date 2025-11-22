import math

# ---------------------------------------------------
# K-Nearest Neighbors Function
# ---------------------------------------------------
def k_nearest_neighbors(points, query_point, k):
    """
    Find k nearest neighbors to a query point
    
    Args:
        points: List of tuples representing points [(x1, y1), (x2, y2), ...]
        query_point: Tuple representing query point (x, y)
        k: Number of nearest neighbors to return
    
    Returns:
        List of k nearest neighbor points as tuples
    """

    # Compute Euclidean distances
    distances = []
    for p in points:
        dist = math.sqrt(sum((pi - qi) ** 2 for pi, qi in zip(p, query_point)))
        distances.append((dist, p))

    # Sort by smallest distance
    distances.sort(key=lambda x: x[0])

    # Return k closest points
    return [point for _, point in distances[:k]]


# ---------------------------------------------------
# Test Case Runner
# ---------------------------------------------------
def run_tests():
    print("Test Case  测试用例")
    print("print(k_nearest_neighbors([(1, 2), (3, 4), (1, 1), (5, 6), (2, 3)], (2, 2), 3))")
    print("打印(k_nearest_neighbors([(1, 2), (3, 4), (1, 1), (5, 6), (2, 3)], (2, 2), 3))")

    points = [(1, 2), (3, 4), (1, 1), (5, 6), (2, 3)]
    query_point = (2, 2)
    k = 3

    expected = [(1, 2), (2, 3), (1, 1)]
    print("Expected  预期")
    print(expected)

    result = k_nearest_neighbors(points, query_point, k)

    print("Your Output  《深度机器学习 | 练习题》")
    print(result)

    # Compare
    print("Passed  通过" if result == expected else "Failed  未通过")


# ---------------------------------------------------
# Run Tests
# ---------------------------------------------------
if __name__ == "__main__":
    run_tests()
