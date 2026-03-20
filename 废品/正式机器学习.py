# 导入必备库
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['WenQuanYi Zen Hei']  # 解决中文显示
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示

# ---------------------- 步骤1：数据读取与列名处理（解决空格问题） ----------------------
df = pd.read_excel('D:/驾驶数据.xlsx')

# 1. 去除列名前后空格（关键：解决“ 侧向加速度”问题）
df.columns = df.columns.str.strip()
print("✅ 处理后列名：")
for i, col in enumerate(df.columns, 1):
    print(f"{i}. '{col}'")

# 2. 确认字段匹配（你的7个字段）
expected_cols = ['时间', '纵向加速度', '侧向加速度', '垂向加速度(叠加了重力加速度)', '速度', '朝向', '横摆角速度（rad/s）']
missing_cols = [col for col in expected_cols if col not in df.columns]
if missing_cols:
    print(f"⚠️  缺失字段：{missing_cols}，请检查Excel字段名！")
    exit()  # 字段不匹配则退出，避免后续报错
else:
    print("✅ 所有字段匹配成功！")

# ---------------------- 步骤2：时间字段处理与时序分段（核心修复） ----------------------
# 假设时间字段是“数值型秒数”（如0, 0.1, 0.2...），按“每5秒”分段（可调整）
segment_duration = 5  # 分段时长（秒），根据数据密度调整（数据密可设2秒，疏可设10秒）

# 1. 时间字段处理（确保是数值型）
if df['时间'].dtype != 'float64':
    df['时间'] = pd.to_datetime(df['时间']).astype('int64') / 1e9  # 转成秒数（若原是datetime）
df = df.sort_values('时间').reset_index(drop=True)  # 按时间排序

# 2. 给每行分配“分段ID”（每segment_duration秒为一段）
df['分段ID'] = np.floor((df['时间'] - df['时间'].min()) / segment_duration).astype(int)

# 3. 检查分段数量（确保至少2段，否则聚类无意义）
segment_count = df['分段ID'].nunique()
if segment_count < 2:
    print(f"⚠️ 当前分段数仅{segment_count}段，需减小分段时长！")
    segment_duration = 2  # 自动减小分段时长
    df['分段ID'] = np.floor((df['时间'] - df['时间'].min()) / segment_duration).astype(int)
    segment_count = df['分段ID'].nunique()
    print(f"✅ 调整后分段数：{segment_count}段（时长{segment_duration}秒）")

print(f"\n=== 数据时序信息 ===")
print(f"总数据行数：{len(df)} 行")
print(f"时间范围：{df['时间'].min():.2f} ~ {df['时间'].max():.2f} 秒")
print(f"时序分段数：{segment_count} 段（每段{segment_duration}秒）")

# ---------------------- 步骤3：重构特征提取（按分段计算特征，避免重复） ----------------------
# 按“分段ID”分组，计算每个时间段的特征（每个分段1个样本，保证样本数充足）
feature_df = df.groupby('分段ID').agg({
    # 1. 速度特征（每段的统计值，各段不同）
    '速度': ['mean', 'std', lambda x: (x > 90).sum()/len(x)],  # 平均速度、速度波动、高速占比
    # 2. 纵向加速度特征（加速/刹车激烈度）
    '纵向加速度': ['max', 'min', lambda x: (x > 1.5).sum()/len(x), lambda x: (x < -1.5).sum()/len(x)],  # 最大正/负加速、激烈加速/刹车占比
    # 3. 侧向加速度特征（转向激烈度）
    '侧向加速度': [lambda x: x.abs().max(), lambda x: (x.abs() > 1).sum()/len(x)],  # 最大侧向加速、激烈转向占比
    # 4. 横摆角速度特征（转向稳定性）
    '横摆角速度（rad/s）': ['std', lambda x: (x.abs() > 0.5).sum()/len(x)]  # 横摆波动、高横摆占比
}).reset_index()

# 重命名特征列（简化列名）
feature_df.columns = [
    '分段ID',
    '平均速度', '速度波动', '高速占比',
    '最大正纵向加速度', '最大负纵向加速度', '激烈加速占比', '急刹车占比',
    '最大侧向加速度', '激烈转向占比',
    '横摆角速度波动', '高横摆占比'
]

# 最终特征矩阵（去掉分段ID，只保留特征）
features = feature_df.drop('分段ID', axis=1)
print(f"\n=== 特征矩阵信息 ===")
print(f"样本数（分段数）：{len(features)} 个")
print(f"特征数：{len(features.columns)} 个")
print(f"特征列表：{features.columns.tolist()}")

# ---------------------- 步骤4：机器学习聚类（样本数充足，可正常运行） ----------------------
# 1. 特征标准化（消除量纲）
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# 2. 选择最优聚类数（2-3类，基于轮廓系数）
best_k = 2
best_score = -1
max_k = min(3, len(features))  # 聚类数不超过样本数（避免样本不够）
for k in range(2, max_k + 1):
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(features_scaled)
    score = silhouette_score(features_scaled, labels)
    if score > best_score:
        best_score = score
        best_k = k

print(f"\n=== 聚类结果 ===")
print(f"最优驾驶风格类别数：{best_k}类")
print(f"轮廓系数（聚类效果）：{best_score:.3f}（越接近1越好）")

# 3. 给每个分段打风格标签
feature_df['驾驶风格标签'] = KMeans(n_clusters=best_k, random_state=42, n_init=10).fit_predict(features_scaled)

# 4. 标签命名（按激烈程度划分）
style_names = {}
for label in range(best_k):
    label_data = feature_df[feature_df['驾驶风格标签'] == label]
    # 激烈程度得分（综合关键指标）
    intense_score = (
        label_data['激烈加速占比'].mean() * 2 +
        label_data['急刹车占比'].mean() * 2 +
        label_data['激烈转向占比'].mean() * 1.5 +
        label_data['速度波动'].mean() * 0.5
    )
    # 命名规则
    if intense_score > 0.6:
        style_names[label] = '激进型'
    elif intense_score > 0.3:
        style_names[label] = '温和型'
    else:
        style_names[label] = '平稳型'

# 添加风格名称，并关联回原始数据
feature_df['驾驶风格'] = feature_df['驾驶风格标签'].map(style_names)
df = df.merge(feature_df[['分段ID', '驾驶风格']], on='分段ID', how='left')

# ---------------------- 步骤5：结果输出（占比图、雷达图、报告） ----------------------
# 1. 统计各风格占比（按时间分段数计算，更合理）
style_stats = feature_df['驾驶风格'].value_counts()
print(f"\n=== 各驾驶风格占比（按时间分段） ===")
for style, count in style_stats.items():
    print(f"{style}：{count}段，占比{count/len(feature_df)*100:.1f}%")

# 2. 绘制风格占比饼图
plt.figure(figsize=(8, 6))
colors = ['#FF5252', '#4CAF50', '#2196F3'][:best_k]  # 激进红、温和绿、平稳蓝
plt.pie(style_stats, labels=style_stats.index, autopct='%1.1f%%',
        startangle=90, colors=colors, textprops={'fontsize': 12})
plt.title('驾驶风格占比分布（按时间分段）', fontsize=16, pad=20)
plt.savefig('驾驶风格占比图.png', dpi=300, bbox_inches='tight')  # 保存在代码同级目录
plt.close()
print(f"\n✅ 占比图已保存：驾驶风格占比图.png")

# 3. 绘制关键特征雷达图（对比各风格）
radar_features = ['激烈加速占比', '急刹车占比', '激烈转向占比', '速度波动']
# 计算各风格的特征均值
style_radar_data = feature_df.groupby('驾驶风格')[radar_features].mean()

# 雷达图绘制
angles = np.linspace(0, 2*np.pi, len(radar_features), endpoint=False).tolist()
angles += angles[:1]  # 闭合图形

plt.figure(figsize=(10, 8))
ax = plt.subplot(111, polar=True)

for idx, style in enumerate(style_radar_data.index):
    values = style_radar_data.loc[style].tolist()
    values += values[:1]  # 闭合数据
    ax.plot(angles, values, label=style, linewidth=3, color=colors[idx])
    ax.fill(angles, values, alpha=0.2, color=colors[idx])

ax.set_xticks(angles[:-1])
ax.set_xticklabels(radar_features, fontsize=11)
ax.set_ylim(0, max(style_radar_data.values.flatten()) * 1.2)  # 调整y轴范围
ax.set_title('不同驾驶风格关键特征对比', fontsize=16, pad=30)
plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1), fontsize=12)
plt.savefig('驾驶风格特征雷达图.png', dpi=300, bbox_inches='tight')
print(f"✅ 特征雷达图已保存：驾驶风格特征雷达图.png")

# 4. 生成详细分析报告（Excel）
report_data = []
for style in style_stats.index:
    style_segments = feature_df[feature_df['驾驶风格'] == style]
    style_raw_data = df[df['驾驶风格'] == style]
    report_data.append({
        '驾驶风格': style,
        '时间分段数': len(style_segments),
        '占比(%)': round(len(style_segments)/len(feature_df)*100, 1),
        '覆盖数据行数': len(style_raw_data),
        '平均速度(km/h)': round(style_segments['平均速度'].mean(), 1),
        '激烈加速占比(%)': round(style_segments['激烈加速占比'].mean()*100, 1),
        '急刹车占比(%)': round(style_segments['急刹车占比'].mean()*100, 1),
        '激烈转向占比(%)': round(style_segments['激烈转向占比'].mean()*100, 1),
        '核心结论': f"该风格占总驾驶时间的{round(len(style_segments)/len(feature_df)*100,1)}%，"
                  f"平均速度{round(style_segments['平均速度'].mean(),1)}km/h，"
                  f"{'存在较多激烈加速/刹车' if (style_segments['激烈加速占比'].mean()+style_segments['急刹车占比'].mean())>0.2 else '驾驶动作较平缓'}"
    })

report_df = pd.DataFrame(report_data)
report_df.to_excel('驾驶风格详细分析报告.xlsx', index=False)
print(f"✅ 详细报告已保存：驾驶风格详细分析报告.xlsx")

# 5. 输出核心总结
print(f"\n🎉 分析完成！核心结论：")
dominant_style = style_stats.index[0]
dominant_ratio = style_stats.iloc[0]/len(feature_df)*100
print(f"你的驾驶以【{dominant_style}】为主，占总驾驶时间的{dominant_ratio:.1f}%")
if dominant_style == '激进型':
    print("建议：减少急加速和急刹车，转向时降低操作幅度，提升驾驶平稳性。")
elif dominant_style == '温和型':
    print("建议：整体驾驶较平稳，可适当减少高速行驶时间，进一步降低能耗。")
else:
    print("建议：驾驶风格非常平稳，继续保持当前操作习惯即可。")