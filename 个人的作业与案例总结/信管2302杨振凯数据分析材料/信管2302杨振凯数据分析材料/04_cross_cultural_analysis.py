import pandas as pd
import os

OUTPUT_FILE = "cross_cultural_correlation.csv"

# 读取前面步骤的结果数据
sentiment_df = pd.read_csv("E:\大学奇奇怪怪的作业\实验\sentiment_result.csv", encoding="utf-8")
keywords_df = pd.read_csv("E:\大学奇奇怪怪的作业\实验\top_keywords.csv", encoding="utf-8")
trigger_df = pd.read_csv("E:\大学奇奇怪怪的作业\实验\trigger_sentiment.csv\china_trigger_sentiment.csv", encoding="utf-8")

# 1. 提取情感分布差异
sentiment_dist = sentiment_df.groupby(["国家", "情感倾向"]).size().unstack(fill_value=0)
sentiment_dist["正向占比"] = (sentiment_dist["正向"] / sentiment_dist.sum(axis=1) * 100).round(2)
sentiment_dist["负向占比"] = (sentiment_dist["负向"] / sentiment_dist.sum(axis=1) * 100).round(2)

# 2. 提取关键词差异（Top10）
top_keywords = keywords_df[keywords_df["排名"] <= 10].pivot_table(
    index=["国家", "情感倾向"],
    values="关键词",
    aggfunc=lambda x: ", ".join(x)
).reset_index()

# 3. 提取触发点情感差异（Top5正负向）
trigger_top = trigger_df[trigger_df["相关评论数"] >= 15].sort_values("平均情感强度", ascending=False)
top_positive_triggers = trigger_top.groupby("国家").head(3)[["国家", "触发点", "平均情感强度"]]
top_negative_triggers = trigger_top.groupby("国家").tail(3)[["国家", "触发点", "平均情感强度"]]

# 生成跨文化关联表模板
correlation_data = []
for country in ["中国", "日本", "美国"]:
    # 情感分布
    country_sentiment = sentiment_dist.loc[country]
    # 正向关键词
    pos_keywords = top_keywords[(top_keywords["国家"] == country) & (top_keywords["情感倾向"] == "正向")]["关键词"].values[0]
    # 负向关键词
    neg_keywords = top_keywords[(top_keywords["国家"] == country) & (top_keywords["情感倾向"] == "负向")]["关键词"].values[0]
    # 正向触发点
    pos_triggers = top_positive_triggers[top_positive_triggers["国家"] == country]["触发点"].tolist()
    # 负向触发点
    neg_triggers = top_negative_triggers[top_negative_triggers["国家"] == country]["触发点"].tolist()

    correlation_data.append({
        "国家": country,
        "正向占比(%)": country_sentiment["正向占比"],
        "负向占比(%)": country_sentiment["负向占比"],
        "正向核心关键词": pos_keywords,
        "负向核心关键词": neg_keywords,
        "正向触发点": ", ".join(pos_triggers),
        "负向触发点": ", ".join(neg_triggers),
        "文化维度关联（霍夫斯泰德）": "",  # 手动填写：个人主义/集体主义等
        "阐释结论": ""  # 手动填写：差异原因分析
    })

# 保存模板
correlation_df = pd.DataFrame(correlation_data)
correlation_df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8")
print(f"跨文化关联表模板生成完成！保存至：{OUTPUT_FILE}")
print("\n请手动填写「文化维度关联」和「阐释结论」列，完成跨文化分析。")