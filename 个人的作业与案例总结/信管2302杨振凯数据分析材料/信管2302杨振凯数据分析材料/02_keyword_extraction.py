import pandas as pd
import matplotlib.pyplot as plt
import warnings  # 新增：导入warnings模块
from sklearn.feature_extraction.text import TfidfVectorizer
from wordcloud import WordCloud

warnings.filterwarnings("ignore")

# ---------------------- 配置 ----------------------
INPUT_FILE = r"E:\大学奇奇怪怪的作业\实验\sentiment_result.csv"  # 步骤2情感分析结果
OUTPUT_KEYWORDS = r"E:\大学奇奇怪怪的作业\实验\top_keywords.csv"
OUTPUT_WORDCLOUD_DIR = r"E:\大学奇奇怪怪的作业\实验\wordclouds"
TOP_N = 50  # Top50关键词
plt.rcParams["font.sans-serif"] = ["SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

# 创建输出文件夹
import os
os.makedirs(OUTPUT_WORDCLOUD_DIR, exist_ok=True)

# ---------------------- 加载停用词 ----------------------
def load_chinese_stopwords():
    # 中文停用词表（适配豆瓣评论）
    stopwords = [
        "的", "了", "是", "我", "在", "有", "和", "就", "不", "人", "都", "一", "一个", "上", "也",
        "很", "到", "说", "要", "去", "你", "会", "着", "没", "看", "过", "还", "而", "这", "那",
        "但", "把", "让", "被", "从", "比", "跟", "为", "与", "及", "等", "或", "都", "全", "各",
        "每", "只", "仅", "又", "再", "更", "最", "挺", "比较", "非常", "特别", "真的", "感觉", "觉得"
    ]
    return stopwords

# ---------------------- 关键词提取 ----------------------
def extract_keywords():
    # 读取数据
    df = pd.read_csv(INPUT_FILE, encoding="utf-8")
    stopwords = load_chinese_stopwords()

    # 按情感倾向分组提取关键词
    all_keywords = []
    for sentiment in ["正向", "负向", "中性"]:
        sentiment_df = df[df["情感倾向"] == sentiment]
        if len(sentiment_df) < 30:
            print(f"警告：{sentiment}样本不足30条，跳过提取")
            continue
        
        # TF-IDF配置（适配中文评论）
        vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),  # 1-2元词（如“结局”“烂尾”“逻辑混乱”）
            max_features=2000,
            stop_words=stopwords
        )
        tfidf_matrix = vectorizer.fit_transform(sentiment_df["分词后评论"])
        feature_names = vectorizer.get_feature_names_out()
        tfidf_scores = tfidf_matrix.sum(axis=0).A1

        # 排序取Top50
        sorted_idx = tfidf_scores.argsort()[::-1]
        top_keywords = [(feature_names[i], tfidf_scores[i]) for i in sorted_idx[:TOP_N]]
        
        # 保存关键词
        for rank, (word, score) in enumerate(top_keywords, 1):
            all_keywords.append({
                "国家": "中国",
                "情感倾向": sentiment,
                "关键词": word,
                "TF-IDF得分": round(score, 4),
                "排名": rank
            })
        
        # 生成词云
        generate_wordcloud([word for word, _ in top_keywords], sentiment)

    # 保存关键词结果
    keywords_df = pd.DataFrame(all_keywords)
    keywords_df.to_csv(OUTPUT_KEYWORDS, index=False, encoding="utf-8")
    print(f"关键词提取完成！保存至：{OUTPUT_KEYWORDS}")

    # 输出Top10关键词
    print(f"\n中国观众各情感倾向Top10关键词：")
    for sentiment in ["正向", "负向", "中性"]:
        top10 = keywords_df[(keywords_df["情感倾向"] == sentiment) & (keywords_df["排名"] <= 10)]
        if not top10.empty:
            print(f"\n{sentiment}：{', '.join(top10['关键词'].tolist())}")

# ---------------------- 词云生成 ----------------------
def generate_wordcloud(keywords, sentiment):
    keywords_text = " ".join(keywords)
    # 词云配置（中文适配）
    wordcloud = WordCloud(
        width=800, height=400,
        background_color="white",
        font_path="msyh.ttc",  # Windows系统默认微软雅黑字体路径
        colormap={"正向": "Greens", "负向": "Reds", "中性": "Blues"}[sentiment],
        max_words=50,
        random_state=42
    ).generate(keywords_text)

    # 保存词云
    save_path = f"{OUTPUT_WORDCLOUD_DIR}china_{sentiment}_wordcloud.png"
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.title(f"中国观众-{sentiment}关键词词云", fontsize=16)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"生成词云：{save_path}")

# ---------------------- 执行 ----------------------
if __name__ == "__main__":
    extract_keywords()