import numpy as np

# 蒙特卡洛模拟示例
np.random.seed(0)

# 模拟1000次实验
n_simulations = 1000
simulated_data = np.random.normal(0, 1, n_simulations)  # 假设一个正态分布

# 计算模拟数据的平均值
mean_simulation = np.mean(simulated_data)

print(mean_simulation)  # 输出模拟数据的平均值

