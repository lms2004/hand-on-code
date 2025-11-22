import numpy as np

# ------------------------------------
# 1. 动量优化器实现（支持标量 / 数组）
# ------------------------------------
def momentum_optimizer(parameter, grad, velocity, learning_rate=0.01, momentum=0.9):
    # 更新动量
    updated_velocity = momentum * velocity + learning_rate * grad
    
    # 更新参数
    updated_parameter = parameter - updated_velocity
    
    # 四舍五入保持题目格式
    return np.round(updated_parameter, 3), np.round(updated_velocity, 3)


# ------------------------------------
# 2. 自动测试函数
# ------------------------------------
def run_test(name, parameter, grad, velocity, lr, momentum, expected_param, expected_vel):
    print(f"=== {name} ===")
    result_param, result_vel = momentum_optimizer(parameter, grad, velocity, lr, momentum)
    
    print("Your Output:")
    print(result_param, result_vel)
    
    print("Expected:")
    print(expected_param, expected_vel)

    if np.allclose(result_param, expected_param) and np.allclose(result_vel, expected_vel):
        print("Result: PASSED ✔\n")
    else:
        print("Result: FAILED ✘\n")


# ------------------------------------
# 3. 你的测试用例（数组版本）
# ------------------------------------
parameter = np.array([1., 2.])
grad = np.array([0.1, 0.2])
velocity = np.array([0.5, 1.0])

expected_param = np.array([0.999, 1.998])
expected_vel = np.array([0.001, 0.002])

# ------------------------------------
# 4. 运行验证
# ------------------------------------
run_test(
    "Array Test Case",
    parameter,
    grad,
    velocity,
    0.01,   # learning_rate
    0.0,    # momentum
    expected_param,
    expected_vel
)
