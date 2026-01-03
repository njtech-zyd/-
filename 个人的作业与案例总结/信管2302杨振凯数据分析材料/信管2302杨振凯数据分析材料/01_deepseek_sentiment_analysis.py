# ---------------------- 依赖库安装（命令行执行） ----------------------
# pip install pandas requests

# ---------------------- 导入库 ----------------------
import pandas as pd
import requests
import time
import json

# ---------------------- 配置参数 ----------------------
# 直接填写你的DeepSeek API密钥（替换下面的字符串）
DEEPSEEK_API_KEY = "sk-5a102bf40b204935afdf202dd12f7658"  # 关键：替换为自己的API密钥
API_URL = "https://api.deepseek.com/v1/chat/completions"
LANGUAGE_MAP = {"中国": "zh", "日本": "ja", "美国": "en"}  # 语言映射

# 情感分析提示词（确保API输出标准化JSON）
SENTIMENT_PROMPT = """
你是专业的跨语言情感分析师，专注于动漫《进击的巨人》观众评论分析。
严格按以下要求输出结果：
1. 情感判断：仅输出「正向」「负向」「中性」三者之一（正向=赞美/感动/认同；负向=批判/失望/反感；中性=客观描述无情感）；
2. 情感得分：-1到1之间的浮点数（保留2位小数，越接近1越正向，越接近-1越负向）；
3. 输出格式：仅返回JSON，无任何额外文字，示例：{"情感倾向": "正向", "情感得分": 0.92}
"""

# ---------------------- 核心函数 ----------------------
def analyze_sentiment_by_deepseek(comment, language):
    """单条评论情感分析（API调用，含重试机制）"""
    headers = {
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "deepseek-chat",
        "messages": [
            {"role": "system", "content": SENTIMENT_PROMPT},
            {"role": "user", "content": f"语言：{language}\n评论：{comment}"}
        ],
        "temperature": 0.1,
        "max_tokens": 100
    }

    retry_count = 3  # 最大重试3次
    while retry_count > 0:
        try:
            response = requests.post(
                API_URL, headers=headers, data=json.dumps(payload), timeout=30
            )
            response.raise_for_status()
            result = response.json()
            content = result["choices"][0]["message"]["content"].strip()
            sentiment = json.loads(content)

            # 验证并规范结果
            sentiment["情感得分"] = round(float(sentiment["情感得分"]), 2)
            sentiment["情感得分"] = max(min(sentiment["情感得分"], 1.0), -1.0)
            return sentiment

        except requests.exceptions.RequestException as e:
            retry_count -= 1
            print(f"API调用失败（剩余重试{retry_count}次）：{str(e)}")
            if "429" in str(e):  # 限流，延迟10秒
                time.sleep(10)
            else:
                time.sleep(2)
        except (json.JSONDecodeError, KeyError):
            retry_count -= 1
            print(f"结果解析失败（剩余重试{retry_count}次）")
            time.sleep(2)

    return {"情感倾向": "中性", "情感得分": 0.0}  # 重试失败默认中性

def batch_sentiment_analysis(input_file, output_file, batch_size=50):
    """批量处理所有评论情感分析"""
    # 读取数据
    df = pd.read_csv(input_file, encoding="utf-8")
    required_cols = ["评论ID", "国家", "原始评论", "分词后评论"]
    assert all(col in df.columns for col in required_cols), f"缺少必要字段：{required_cols}"
    print(f"总数据量：{len(df)}条，开始情感分析...")

    # 批量处理
    results = []
    for idx, row in df.iterrows():
        # 进度提示
        if (idx + 1) % batch_size == 0:
            print(f"已处理 {idx + 1}/{len(df)} 条")

        # 调用API
        sentiment = analyze_sentiment_by_deepseek(
            comment=row["原始评论"],
            language=LANGUAGE_MAP.get(row["国家"], "en")
        )

        # 保存结果
        results.append({
            "评论ID": row["评论ID"],
            "国家": row["国家"],
            "原始评论": row["原始评论"],
            "分词后评论": row["分词后评论"],
            "情感倾向": sentiment["情感倾向"],
            "情感得分": sentiment["情感得分"]
        })

        # 避免限流延迟
        time.sleep(0.3)

    # 保存输出文件
    result_df = pd.DataFrame(results)
    result_df.to_csv(output_file, index=False, encoding="utf-8")
    print(f"\n情感分析完成！结果保存至：{output_file}")
    print("\n情感倾向分布统计：")
    print(result_df.groupby(["国家", "情感倾向"]).size().unstack(fill_value=0))

# ---------------------- 执行入口（替换为你的文件路径） ----------------------
if __name__ == "__main__":
    batch_sentiment_analysis(
        input_file="E:\大学奇奇怪怪的作业\实验\douban_comments_converted.csv",  # 预处理输入文件
        output_file="E:\大学奇奇怪怪的作业\实验\sentiment_result.csv"  # 情感分析输出文件
    )