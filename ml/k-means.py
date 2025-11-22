# -------------------------------
# k-Means 聚类算法实现
# -------------------------------

def k_means_clustering(points: list[tuple[float, float]], 
                       k: int, 
                       initial_centroids: list[tuple[float, float]], 
                       max_iterations: int) -> list[tuple[float, float]]:
    
    import math
    
    # 欧氏距离
    def distance(p1, p2):
        return math.sqrt(sum((a - b) ** 2 for a, b in zip(p1, p2)))
    
    centroids = initial_centroids

    for _ in range(max_iterations):
        # 1. 将每个点分配给最近的质心
        clusters = [[] for _ in range(k)]
        for p in points:
            nearest = min(range(k), key=lambda i: distance(p, centroids[i]))
            clusters[nearest].append(p)

        # 2. 更新质心
        new_centroids = []
        for cluster in clusters:
            if cluster:
                dim = len(cluster[0])
                mean = tuple(
                    round(sum(point[d] for point in cluster) / len(cluster), 4)
                    for d in range(dim)
                )
                new_centroids.append(mean)
            else:
                new_centroids.append(None)

        # 3. 若质心不再变化，则提前停止
        if new_centroids == centroids:
            break

        centroids = new_centroids

    return centroids


# -------------------------------
# 测试工具函数
# -------------------------------

def run_test(test_name, points, k, init_c, max_iter, expected):
    print(f"\nTest Case: {test_name}")
    print("Input:")
    print(" points =", points)
    print(" k =", k)
    print(" initial_centroids =", init_c)
    print(" max_iterations =", max_iter)
    
    print("\nExpected:")
    print(" ", expected)

    result = k_means_clustering(points, k, init_c, max_iter)
    
    print("\nYour Output:")
    print(" ", result)
    
    if result == expected:
        print("Passed ✓")
    else:
        print("Failed ✗")


# -------------------------------
# 测试用例（含你题目里的示例）
# -------------------------------
if __name__ == "__main__":
    
    # Test Case 1（题目示例）
    run_test(
        "Example Case",
        points=[(1,2),(1,4),(1,0),(10,2),(10,4),(10,0)],
        k=2,
        init_c=[(1,1),(10,1)],
        max_iter=10,
        expected=[(1.0, 2.0), (10.0, 2.0)]
    )