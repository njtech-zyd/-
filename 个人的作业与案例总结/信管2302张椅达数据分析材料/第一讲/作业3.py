# 导入必要的库
import jieba
from collections import Counter

# 定义文本内容（统一引号格式）
text = """
落实"企业管理年"主题，加强QEHS三体系建设，
通过自动化、数字化、智能化升级改造加快新一代信息技术与企业生产经营融合，
打造精益制造能力，提升精细化管理水平，助力公司从"制造"升级为"智造"，
从而提高经营效率和效益:通过集团一体化智数管理平台建设与运营，
丰富企业供应链管理、生产工艺控制等管理工具，不断增强生产经营过程数据获取与分析能力，
强化全过程一体化管理，提高自动化、数字化、智能化的供应链管理能力，
为体系安全稳定运行与管理水平提升保驾护航，致力打造安全智能化工厂;
打造助剂互联网技术合作和商务合作平台，构建具有国际竞争力的供应链体系。
"""

# 定义停用词
stopwords = ['\n', '，', '、', ':', ';', '"', "'", '。', '与', '和', '等', '为', '从', '的', '了', '在']

# 1. 分词处理
words = jieba.lcut(text)

# 2. 过滤停用词
filtered_words = [word for word in words if word not in stopwords]

print("过滤停用词后的分词结果（前50个词）：")
print(filtered_words[:50])
print()

# 3. 定义要统计的特殊词汇
target_words = ['数字化', '智能化', '安全', '自动化', '供应链', '管理', '制造', '智造']

# 4. 统计词频
word_counts = Counter(filtered_words)

# 输出特定词汇的词频统计结果
print("特定词汇词频统计结果：")
for word in target_words:
    if word in word_counts:
        print(f"'{word}': {word_counts[word]}次")
    else:
        print(f"'{word}': 0次")

# 5. 输出所有词汇的词频（按频率降序）
print("\n所有词汇词频统计（前20个）：")
for word, count in word_counts.most_common(20):
    print(f"'{word}': {count}次")

# 6. 分析数字技术相关词汇
print("\n数字技术相关词汇统计：")
tech_words = ['数字化', '智能化', '自动化', '信息技术', '智数', '互联网', '数据']
for word in tech_words:
    if word in word_counts:
        print(f"'{word}': {word_counts[word]}次")
    else:
        print(f"'{word}': 0次")

# 7. 分析安全管理相关词汇
print("\n安全管理相关词汇统计：")
safety_words = ['安全', '体系', 'QEHS', '稳定', '保驾护航']
for word in safety_words:
    if word in word_counts:
        print(f"'{word}': {word_counts[word]}次")
    else:
        print(f"'{word}': 0次")