import pandas as pd
import requests
import time
import json

# ---------------------- 配置参数 ----------------------
DEEPSEEK_API_KEY = "sk-5a102bf40b204935afdf202dd12f7658"  # 替换为你的API密钥
API_URL = "https://api.deepseek.com/v1/chat/completions"
INPUT_FILE = r"E:\大学奇奇怪怪的作业\实验\douban_comments_converted.csv"  # 步骤1生成的标准数据（原始字符串）
OUTPUT_FILE = r"E:\大学奇奇怪怪的作业\实验\sentiment_result.csv"  # 原始字符串修复路径

# 情感分析提示词（聚焦《进击的巨人》中国观众评论）
SENTIMENT_PROMPT = """
你是专业的情感分析师，专注于《进击的巨人》中国观众评论分析。
严格按以下要求输出：
1. 情感倾向：仅输出「正向」「负向」「中性」（正向=赞美/感动/认同；负向=批判/失望/反感；中性=客观描述）；
2. 情感得分：-1到1的浮点数（保留2位小数，越接近1越正向，越接近-1越负向）；
3. 输出格式：仅JSON，无额外文字，示例：{"情感倾向": "负向", "情感得分": -0.75}
"""

# ---------------------- 核心函数 ----------------------
def analyze_sentiment(comment):
    """调用API分析单条评论情感"""
    headers = {
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "deepseek-chat",
        "messages": [
            {"role": "system", "content": SENTIMENT_PROMPT},
            {"role": "user", "content": f"评论：{comment}"}
        ],
        "temperature": 0.1,
        "max_tokens": 100
    }

    retry_count = 3
    while retry_count > 0:
        try:
            response = requests.post(API_URL, headers=headers, data=json.dumps(payload), timeout=30)
            response.raise_for_status()
            result = response.json()
            content = result["choices"][0]["message"]["content"].strip()
            sentiment = json.loads(content)
            # 规范得分
            sentiment["情感得分"] = round(float(sentiment["情感得分"]), 2)
            sentiment["情感得分"] = max(min(sentiment["情感得分"], 1.0), -1.0)
            return sentiment
        except Exception as e:
            retry_count -= 1
            print(f"API调用失败（剩余{retry_count}次）：{str(e)}")
            time.sleep(5 if "429" in str(e) else 2)
    return {"情感倾向": "中性", "情感得分": 0.0}

def batch_analyze():
    """批量分析所有评论"""
    # 读取标准数据
    df = pd.read_csv(INPUT_FILE, encoding="utf-8")
    print(f"待分析数据量：{len(df)}条（中国观众评论）")

    # 批量处理
    results = []
    batch_size = 50
    for idx, row in df.iterrows():
        if (idx + 1) % batch_size == 0:
            print(f"已处理 {idx + 1}/{len(df)} 条")
        
        # 分析情感
        comment = row["原始评论"].strip()
        sentiment = analyze_sentiment(comment)
        
        # 保存结果
        results.append({
            "评论ID": row["评论ID"],
            "国家": row["国家"],
            "原始评论": comment,
            "分词后评论": row["分词后评论"],
            "情感倾向": sentiment["情感倾向"],
            "情感得分": sentiment["情感得分"]
        })
        
        time.sleep(0.3)  # 避免限流

    # 保存结果
    result_df = pd.DataFrame(results)
    result_df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8")

    # 输出统计（修复语法错误：for 倾向 加空格）
    print(f"\n情感分析完成！结果保存至：{OUTPUT_FILE}")
    sentiment_dist = result_df["情感倾向"].value_counts(normalize=True) * 100
    print(f"\n中国观众情感分布：")
    for 倾向, ratio in sentiment_dist.items():  # 修复：for和倾向之间加空格
        print(f"{倾向}：{ratio:.2f}%（{result_df['情感倾向'].value_counts()[倾向]}条）")
    print(f"平均情感得分：{result_df['情感得分'].mean():.2f}")

# ---------------------- 执行 ----------------------
if __name__ == "__main__":
    batch_analyze()