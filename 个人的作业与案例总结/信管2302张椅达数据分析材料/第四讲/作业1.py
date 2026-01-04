import pandas as pd
from snownlp import SnowNLP
import matplotlib.pyplot as plt

# 1. 读取Excel文件
file_path = r"D:\用户数据采集\作业\用户数据\第4讲 感知世界：情感分析\第4讲 感知世界：情感分析\restaurant-comments.xlsx"
df = pd.read_excel(file_path)

print(f"数据读取成功！共有 {len(df)} 条评论")

# 2. 情感分析函数
def analyze_sentiment_snownlp(text):
    """
    使用SnowNLP进行情感分析
    返回值：情感得分 (0-1之间，越接近1表示越积极)
    """
    try:
        s = SnowNLP(str(text))
        # SnowNLP的情感分析得分在0-1之间
        # 0-0.4：负面，0.4-0.6：中性，0.6-1：正面
        return s.sentiments
    except Exception as e:
        print(f"分析文本时出错：{e}")
        return None

# 3. 批量分析情感
print("正在分析情感，请稍候...")
df['sentiment_score'] = df['comments'].apply(analyze_sentiment_snownlp)

# 4. 情感分类
def classify_sentiment(score):
    """
    根据情感得分进行分类
    """
    if score is None:
        return '未知'
    elif score >= 0.6:
        return '正面'
    elif score <= 0.4:
        return '负面'
    else:
        return '中性'

df['sentiment_label'] = df['sentiment_score'].apply(classify_sentiment)

# 5. 显示分析结果
print("\n=== 分析结果预览 ===")
print(df[['comments', 'sentiment_score', 'sentiment_label']].head(10))

print("\n=== 情感分布统计 ===")
sentiment_stats = df['sentiment_label'].value_counts()
print(sentiment_stats)

# 6. 保存结果到新Excel文件
output_path = r"D:\用户数据采集\作业\用户数据\第4讲 感知世界：情感分析\restaurant-comments-analyzed.xlsx"
df.to_excel(output_path, index=False)
print(f"\n分析结果已保存到：{output_path}")

# 7. 可选：生成可视化图表
def generate_visualization():
    """生成情感分析的可视化图表"""
    plt.figure(figsize=(10, 6))
    
    # 子图1：情感分布饼图
    plt.subplot(1, 2, 1)
    sentiment_stats.plot.pie(autopct='%1.1f%%', startangle=90)
    plt.title('情感分布比例')
    plt.ylabel('')
    
    # 子图2：情感得分直方图
    plt.subplot(1, 2, 2)
    plt.hist(df['sentiment_score'].dropna(), bins=20, alpha=0.7, color='skyblue')
    plt.xlabel('情感得分')
    plt.ylabel('评论数量')
    plt.title('情感得分分布')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 保存图表
    chart_path = r"D:\用户数据采集\作业\用户数据\第4讲 感知世界：情感分析\sentiment_analysis_chart.png"
    plt.savefig(chart_path, dpi=300, bbox_inches='tight')
    print(f"可视化图表已保存到：{chart_path}")
    plt.show()

# 8. 生成详细统计报告
def generate_report():
    """生成详细的分析报告"""
    print("\n" + "="*50)
    print("            情感分析详细报告")
    print("="*50)
    
    total_comments = len(df)
    positive = len(df[df['sentiment_label'] == '正面'])
    negative = len(df[df['sentiment_label'] == '负面'])
    neutral = len(df[df['sentiment_label'] == '中性'])
    
    print(f"总评论数：{total_comments}")
    print(f"正面评论：{positive} ({positive/total_comments*100:.1f}%)")
    print(f"负面评论：{negative} ({negative/total_comments*100:.1f}%)")
    print(f"中性评论：{neutral} ({neutral/total_comments*100:.1f}%)")
    print(f"平均情感得分：{df['sentiment_score'].mean():.3f}")
    
    # 显示最有代表性的评论
    print("\n=== 最具代表性的评论 ===")
    print("\n最积极的评论（前3）：")
    for _, row in df.nlargest(3, 'sentiment_score').iterrows():
        print(f"得分：{row['sentiment_score']:.3f} - {row['comments'][:100]}...")
    
    print("\n最消极的评论（前3）：")
    for _, row in df.nsmallest(3, 'sentiment_score').iterrows():
        print(f"得分：{row['sentiment_score']:.3f} - {row['comments'][:100]}...")

# 询问用户是否需要可视化
generate_vis = input("\n是否生成可视化图表？(y/n): ").lower()
if generate_vis == 'y':
    generate_visualization()

# 生成报告
generate_report()