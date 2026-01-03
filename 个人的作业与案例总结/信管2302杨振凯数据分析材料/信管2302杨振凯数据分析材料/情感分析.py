import requests
import json
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

# -------------------------- 【关键适配：你的豆瓣评论文件参数】--------------------------
API_KEY = "sk-5a102bf40b204935afdf202dd12f7658"  # 替换为你的 DeepSeek API Key
INPUT_FILE = "E:\大学奇奇怪怪的作业\实验\douban_FINAL_CLEANED.csv"  # 你的豆瓣评论文件路径（固定）
OUTPUT_FILE = "E:\大学奇奇怪怪的作业\实验\douban_comments_sentiment_result.csv"  # 输出结果路径（保存在同目录）
COMMENT_COLUMN = "comment"  # 你的评论列名（已确认是 comment）
BATCH_SIZE = 500  # 1 批即可处理完 461 条数据（免费额度足够）
MAX_WORKERS = 10  # 并发线程数（平衡效率和限流）
RETRY_TIMES = 3  # 失败自动重试次数
MODEL = "deepseek-chat"  # 稳定的通用对话模型
TEXT_MAX_LENGTH = 10000  # 截断超长文本（避免 API 报错）

# -------------------------- 【工具函数：无需修改】--------------------------
# 1. 读取豆瓣评论数据（保留所有原始列：username、rating、time 等）
def load_comments(file_path):
    df = pd.read_csv(file_path, encoding="utf-8-sig")
    # 数据预处理：去重（避免重复评论）、截断超长文本、保留所有原始列
    df = df.drop_duplicates(subset=[COMMENT_COLUMN])  # 按评论去重
    df[COMMENT_COLUMN] = df[COMMENT_COLUMN].astype(str).str[:TEXT_MAX_LENGTH]  # 截断
    df = df.reset_index(drop=True)
    print(f"✅ 成功加载 {len(df)} 条豆瓣有效评论（原始数据 461 条，去重后剩余）")
    print(f"📊 原始列：{df.columns.tolist()}（将保留所有列，新增情感分析结果列）")
    return df

# 2. 单条评论情感分析（带原生重试，避免依赖 tenacity）
def analyze_single_comment(comment):
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    # 精准提示词（适配多语言，确保输出格式统一）
    prompt = f"""
    对以下豆瓣评论做情感分析，严格遵守：
    1. 情感标签仅返回 positive（正面）、negative（负面）、neutral（中性）；
    2. 置信度保留 4 位小数（0~1，越接近 1 越可信）；
    3. 语言识别返回缩写（如 zh=中文、en=英文）；
    4. 仅输出 JSON 字符串，无额外文字，字段：sentiment、confidence、language。
    
    评论内容：{comment}
    """
    data = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0  # 零随机性，结果稳定
    }
    # 原生重试逻辑
    retry_intervals = [1, 2, 4]
    for attempt in range(RETRY_TIMES):
        try:
            time.sleep(0.1)  # 降低限流风险
            response = requests.post(
                url="https://api.deepseek.com/v1/chat/completions",
                headers=headers,
                json=data,
                timeout=15
            )
            response.raise_for_status()
            model_res = response.json()
            result_json = model_res["choices"][0]["message"]["content"].strip()
            return json.loads(result_json)
        except Exception as e:
            if attempt == RETRY_TIMES - 1:
                raise e
            time.sleep(retry_intervals[attempt])

# 3. 批量处理（保留原始列，新增情感结果，原生进度提示）
def batch_analyze_comments(df):
    all_results = []
    total = len(df)
    print(f"\n===== 开始批量分析 {total} 条豆瓣评论 =====")
    
    # 多线程处理
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_idx = {
            executor.submit(analyze_single_comment, row[COMMENT_COLUMN]): row 
            for _, row in df.iterrows()  # 保留每行原始数据（username、rating 等）
        }
        processed_count = 0
        for future in as_completed(future_to_idx):
            processed_count += 1
            # 显示进度（每 20 条更新一次）
            if processed_count % 20 == 0 or processed_count == total:
                print(f"进度：{processed_count}/{total} [{(processed_count/total)*100:.1f}%]")
            
            row = future_to_idx[future]
            try:
                sentiment_res = future.result()
                # 合并原始数据和情感结果
                result_row = row.to_dict()  # 原始列（username、rating 等）
                result_row.update({
                    "情感标签": sentiment_res["sentiment"],
                    "情感置信度": sentiment_res["confidence"],
                    "文本语言": sentiment_res["language"],
                    "分析状态": "成功"
                })
            except Exception as e:
                result_row = row.to_dict()
                result_row.update({
                    "情感标签": None,
                    "情感置信度": None,
                    "文本语言": None,
                    "分析状态": f"失败：{str(e)[:40]}"
                })
            all_results.append(result_row)
    
    # 保存结果（包含所有原始列 + 情感分析列）
    result_df = pd.DataFrame(all_results)
    result_df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")
    return result_df

# -------------------------- 【执行入口：直接运行】--------------------------
if __name__ == "__main__":
    try:
        # 1. 加载数据
        comments_df = load_comments(INPUT_FILE)
        
        # 2. 批量分析
        result_df = batch_analyze_comments(comments_df)
        
        # 3. 输出分析报告
        print("\n===== 豆瓣评论情感分析完成！=====")
        success_df = result_df[result_df["分析状态"] == "成功"]
        fail_df = result_df[result_df["分析状态"] != "成功"]
        
        print(f"📊 整体统计：")
        print(f"   - 总评论数：{len(result_df)}")
        print(f"   - 成功分析：{len(success_df)} 条（{len(success_df)/len(result_df)*100:.1f}%）")
        print(f"   - 失败分析：{len(fail_df)} 条")
        
        if len(success_df) > 0:
            print(f"\n❤️  情感分布：")
            sentiment_count = success_df["情感标签"].value_counts()
            for sent, count in sentiment_count.items():
                sent_cn = {"positive": "正面", "negative": "负面", "neutral": "中性"}[sent]
                print(f"   - {sent_cn}评论：{count} 条（{count/len(success_df)*100:.1f}%）")
        
        print(f"\n💾 结果文件已保存至：{OUTPUT_FILE}")
        print(f"   包含列：{result_df.columns.tolist()}")
        print("\n🔍 结果预览（前 2 行）：")
        print(result_df[["username", "rating", "comment", "情感标签", "情感置信度"]].head(2))
    
    except Exception as e:
        print(f"\n❌ 程序异常：{str(e)}")
        if "API_KEY" in str(e) or "401" in str(e):
            print("   提示：请检查 API Key 是否有效（去 DeepSeek 控制台重新生成）")