import pandas as pd
import numpy as np

# 读取数据
df = pd.read_csv("Driver_Behavior.csv")

# 特征列
features = [
    'speed_kmph', 'accel_x', 'accel_y', 'brake_pressure',
    'steering_angle', 'throttle', 'lane_deviation',
    'phone_usage', 'headway_distance', 'reaction_time'
]

# 标签列
label_col = "behavior_label"

typical_samples = {}

for label, group in df.groupby(label_col):

    # 计算该类的平均值
    mean_vector = group[features].mean()

    # 计算每一行到均值的欧氏距离
    distances = np.linalg.norm(group[features] - mean_vector, axis=1)

    # 找到距离最小的一行
    idx = distances.argmin()

    typical_samples[label] = df.loc[idx]

# 输出结果
for k, v in typical_samples.items():
    print(f"\n{k} 的典型驾驶数据：")
    print(v)
