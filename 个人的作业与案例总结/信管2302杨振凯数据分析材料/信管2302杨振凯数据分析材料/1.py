import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 检查文件是否存在
file_path = '实验1数据.xlsx'
if not os.path.exists(file_path):
    print(f"错误：找不到文件 '{file_path}'")
    # 创建模拟数据用于演示
    import numpy as np
    np.random.seed(42)
    satisfaction_scores = np.random.normal(75, 10, 100).astype(int)
    satisfaction_scores = np.clip(satisfaction_scores, 0, 100)  # 确保分数在0-100之间
    df = pd.DataFrame({'满意度评分': satisfaction_scores})
    print("已创建模拟数据用于演示")
else:
    # 从Excel文件读取数据
    try:
        df = pd.read_excel(file_path, sheet_name='员工工作满意度评分', header=None, skiprows=2, nrows=100, usecols=[0])
        df.columns = ['满意度评分']
        print(f"成功读取文件，数据形状：{df.shape}")
    except Exception as e:
        print(f"读取文件时出错：{e}")
        # 创建模拟数据作为后备
        import numpy as np
        np.random.seed(42)
        satisfaction_scores = np.random.normal(75, 10, 100).astype(int)
        satisfaction_scores = np.clip(satisfaction_scores, 0, 100)  # 确保分数在0-100之间
        df = pd.DataFrame({'满意度评分': satisfaction_scores})
        print("已创建模拟数据用于演示")

# 对数据排序
sorted_data = df.sort_values(by='满意度评分')
print("\n排序后的数据（前5行）：")
print(sorted_data.head())

# 定义分组区间
bins = [0, 50, 60, 70, 80, 90, 100]
labels = ['50以下', '50-60', '60-70', '70-80', '80-90', '90-100']

# 使用cut函数进行分组
df['分组'] = pd.cut(df['满意度评分'], bins=bins, labels=labels, right=True)

# 生成频数分布表
freq_table = df['分组'].value_counts().sort_index()
print("\n频数分布表：")
print(freq_table)

# 计算频率
print("\n频率分布表：")
freq_table_pct = df['分组'].value_counts(normalize=True).sort_index() * 100
print(freq_table_pct.round(2))

# 绘制直方图
plt.figure(figsize=(10, 6))
plt.hist(df['满意度评分'], bins=bins, edgecolor='black', alpha=0.7, color='#4CAF50')
plt.title('员工工作满意度直方图')
plt.xlabel('满意度评分')
plt.ylabel('频数')
plt.grid(axis='y', linestyle='--', alpha=0.7)

# 在直方图上方添加数值标签
heights, _, _ = plt.hist(df['满意度评分'], bins=bins, edgecolor='black', alpha=0.7, color='#4CAF50')
for i, height in enumerate(heights):
    plt.text((bins[i] + bins[i+1])/2, height + 0.5, f'{int(height)}', 
             ha='center', va='bottom', fontweight='bold')

# 保存图表
plt.tight_layout()
plt.savefig('employee_satisfaction_histogram.png', dpi=300)
print("\n直方图已保存为 'employee_satisfaction_histogram.png'")

# 显示图表
plt.close()  # 不显示图表，只保存

# 额外的数据分析
print("\n数据基本统计：")
print(df['满意度评分'].describe())

# 创建饼图显示不同满意度级别的占比
plt.figure(figsize=(8, 8))
freq_table.plot(kind='pie', autopct='%1.1f%%', startangle=90, colors=sns.color_palette('pastel'))
plt.title('员工工作满意度分布饼图')
plt.ylabel('')  # 移除y轴标签
plt.tight_layout()
plt.savefig('employee_satisfaction_pie_chart.png', dpi=300)
print("饼图已保存为 'employee_satisfaction_pie_chart.png'")
plt.close()

print("\n分析完成！")


# 计算累积频数
cumulative_freq = freq_table.cumsum()

# 绘制累积分布图
plt.figure(figsize=(10, 6))
cumulative_freq.plot(kind='line', marker='o')
plt.title('员工工作满意度累积分布图')
plt.xlabel('满意度分组')
plt.ylabel('累积频数')
plt.grid(True)
plt.show()


# 使用频数分布表绘制折线图
plt.figure(figsize=(10, 6))
freq_table.plot(kind='line', marker='o')
plt.title('员工工作满意度频数分布折线图')
plt.xlabel('满意度分组')
plt.ylabel('频数')
plt.grid(True)
plt.show()


plt.figure(figsize=(8, 8))
freq_table.plot(kind='pie', autopct='%1.1f%%')
plt.title('员工工作满意度频数分布饼图')
plt.ylabel('')  # 隐藏y轴标签
plt.show()


# 假设饮料数据在另一个文件中，这里我们使用一个示例DataFrame
# 实际数据可以类似读取
drink_data = pd.DataFrame({
    '饮料': ['可乐', '雪碧', '橙汁', '绿茶', '奶茶', '咖啡'],
    '销售数量': [150, 120, 90, 60, 30, 20]
})

# 按销售数量降序排序
drink_data = drink_data.sort_values(by='销售数量', ascending=False)

drink_data['累积百分比'] = drink_data['销售数量'].cumsum() / drink_data['销售数量'].sum() * 100


fig, ax1 = plt.subplots(figsize=(10, 6))

# 绘制条形图
ax1.bar(drink_data['饮料'], drink_data['销售数量'], color='C0')
ax1.set_xlabel('饮料')
ax1.set_ylabel('销售数量', color='C0')
ax1.tick_params(axis='y', labelcolor='C0')

# 绘制折线图
ax2 = ax1.twinx()
ax2.plot(drink_data['饮料'], drink_data['累积百分比'], color='C1', marker='o')
ax2.set_ylabel('累积百分比', color='C1')
ax2.tick_params(axis='y', labelcolor='C1')

plt.title('饮料销售帕累托图')
plt.show()



# 假设我们有一个包含工作绩效的数据列，这里我们随机生成一些数据用于示例
import numpy as np
np.random.seed(0)
df['工作绩效'] = np.random.randint(1, 10, size=len(df))

# 绘制散点图
plt.figure(figsize=(10, 6))
plt.scatter(df['满意度评分'], df['工作绩效'])
plt.title('员工工作满意度与工作绩效散点图')
plt.xlabel('员工工作满意度')
plt.ylabel('工作绩效')

# 添加趋势线
z = np.polyfit(df['满意度评分'], df['工作绩效'], 1)
p = np.poly1d(z)
plt.plot(df['满意度评分'], p(df['满意度评分']), "r--")

plt.show()
