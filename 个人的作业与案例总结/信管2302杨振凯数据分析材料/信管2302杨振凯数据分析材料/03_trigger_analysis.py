import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

# ---------------------- 配置（修复路径问题） ----------------------
INPUT_FILE = r"E:\大学奇奇怪怪的作业\实验\sentiment_result.csv"  # 情感分析结果路径
OUTPUT_DIR = r"E:\大学奇奇怪怪的作业\实验\trigger_sentiment.csv"  # 绝对路径，去掉结尾反斜杠
plt.rcParams["font.sans-serif"] = ["SimHei", "DejaVu Sans"]  # 中文显示
plt.rcParams["axes.unicode_minus"] = False

# 创建输出目录（确保存在）
os.makedirs(OUTPUT_DIR, exist_ok=True)  # 自动创建目录，无需手动拼接
OUTPUT_TRIGGER = os.path.join(OUTPUT_DIR, "china_trigger_sentiment.csv")  # 正确拼接文件路径
OUTPUT_HEATMAP = os.path.join(OUTPUT_DIR, "china_trigger_heatmap.png")

# ---------------------- 《进击的巨人》核心触发点（中国观众关注焦点） ----------------------
CORE_TRIGGERS = [
    "结局", "艾伦", "三笠", "阿尔敏", "利威尔", "地鸣", "灭世", "自由", "宿命",
    "烂尾", "逻辑混乱", "三观", "集体牺牲", "帕拉迪岛", "巨人之力", "伏笔", "回收"
]

# ---------------------- 触发点情感统计 ----------------------
def analyze_triggers():
    # 读取情感分析结果
    df = pd.read_csv(INPUT_FILE, encoding="utf-8")
    trigger_results = []

    for trigger in CORE_TRIGGERS:
        # 筛选包含该触发点的评论（不区分大小写，处理空值）
        trigger_df = df[df["原始评论"].str.contains(trigger, na=False, case=False)]
        if len(trigger_df) < 15:  # 样本不足15条跳过（无统计意义）
            continue
        
        # 计算情感指标
        avg_score = round(trigger_df["情感得分"].mean(), 2)
        sentiment_dist = trigger_df["情感倾向"].value_counts(normalize=True) * 100
        
        # 保存结果
        trigger_results.append({
            "触发点": trigger,
            "相关评论数": len(trigger_df),
            "平均情感强度": avg_score,
            "正向占比(%)": round(sentiment_dist.get("正向", 0.0), 2),
            "负向占比(%)": round(sentiment_dist.get("负向", 0.0), 2),
            "中性占比(%)": round(sentiment_dist.get("中性", 0.0), 2)
        })

    # 转换为DataFrame并排序
    trigger_df = pd.DataFrame(trigger_results)
    trigger_df = trigger_df.sort_values("平均情感强度", ascending=False)  # 按情感强度降序
    
    # 保存触发点分析结果
    trigger_df.to_csv(OUTPUT_TRIGGER, index=False, encoding="utf-8")
    print(f"触发点分析完成！结果保存至：{OUTPUT_TRIGGER}")

    # 输出关键发现
    print(f"\n【中国观众情感触发点Top5（正向最强）】")
    top_positive = trigger_df.head(5)[["触发点", "平均情感强度", "相关评论数"]]
    print(top_positive.to_string(index=False))
    
    print(f"\n【中国观众情感触发点Bottom5（负向最强）】")
    top_negative = trigger_df.tail(5)[["触发点", "平均情感强度", "相关评论数"]]
    print(top_negative.to_string(index=False))

    # 绘制热力图
    plot_heatmap(trigger_df)

# ---------------------- 绘制触发点情感热力图 ----------------------
def plot_heatmap(trigger_df):
    # 筛选有效触发点（相关评论数≥15）
    valid_triggers = trigger_df[trigger_df["相关评论数"] >= 15]
    if len(valid_triggers) < 5:
        print("有效触发点不足5个，跳过热力图绘制")
        return

    # 整理热力图数据（触发点×情感占比）
    heatmap_data = valid_triggers[["触发点", "正向占比(%)", "负向占比(%)", "中性占比(%)"]]
    heatmap_data = heatmap_data.set_index("触发点")  # 触发点作为行索引

    # 绘制热力图
    plt.figure(figsize=(12, 8))
    sns.heatmap(
        heatmap_data,
        cmap="RdYlGn",  # 红（负向）-黄（中性）-绿（正向）
        annot=True,     # 显示数值
        fmt=".1f",      # 保留1位小数
        cbar_kws={"label": "情感占比(%)"},  # 颜色条标签
        linewidths=0.5  # 格子边框宽度
    )
    plt.title("中国观众《进击的巨人》核心触发点情感占比热力图", fontsize=14, pad=20)
    plt.xticks(rotation=0)  # 列标签水平显示
    plt.yticks(rotation=45) # 行标签倾斜显示
    plt.tight_layout()      # 调整布局防止文字重叠
    
    # 保存热力图
    plt.savefig(OUTPUT_HEATMAP, dpi=300, bbox_inches="tight")  # 高清保存
    plt.close()
    print(f"热力图生成完成！保存至：{OUTPUT_HEATMAP}")

# ---------------------- 执行入口 ----------------------
if __name__ == "__main__":
    analyze_triggers()