# ==================== 第一部分：指定词汇词频统计 ====================

# 定义需要统计的词汇
terms = ['黄旭华', '核潜艇', '国立交通大学']

# 打印指定的词汇列表
print("指定统计的词汇：", terms)

# 创建空字典存储统计结果
terms_dict = {}

# 打开文本文件
f_txt = open('D:\\用户数据采集\\作业\\第二讲\\科学家博物馆-黄旭华传记序言.txt', encoding='utf-8')
data_txt = f_txt.read()
f_txt.close()

# 统计指定词汇出现的次数
for term in terms:
    terms_dict[term] = data_txt.count(term)

# 打印统计结果
print("\n指定词汇统计结果：")
print(terms_dict)

# ==================== 第二部分：全文词频统计 ====================

import jieba
from collections import Counter
import re

def full_text_word_frequency(text, top_n=20):
    """
    对全文进行词频统计
    :param text: 文本内容
    :param top_n: 显示前N个高频词
    :return: 词频字典
    """
    # 使用jieba进行中文分词
    words = jieba.lcut(text)
    
    # 过滤掉标点符号、空格和单个字符
    filtered_words = []
    for word in words:
        # 只保留长度大于1的中文字符
        if len(word) > 1 and re.search(r'[\u4e00-\u9fff]', word):
            filtered_words.append(word)
    
    # 统计词频
    word_counts = Counter(filtered_words)
    
    # 获取前N个高频词
    top_words = word_counts.most_common(top_n)
    
    return dict(top_words)

# 对全文进行词频统计
full_word_dict = full_text_word_frequency(data_txt, top_n=20)

print("\n全文词频统计（前20个高频词）：")
for word, count in full_word_dict.items():
    print(f"{word}: {count}次")

# ==================== 第三部分：数据可视化 ====================

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams

# 设置中文字体
def make_chinese_plot_ready():
    try:
        rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
    except:
        rcParams['font.sans-serif'] = ['Microsoft YaHei']  # 或使用微软雅黑
    rcParams['axes.unicode_minus'] = False

# 定义画图函数
def draw_dict(mydict, title="词频统计", figsize=(12, 6)):
    make_chinese_plot_ready()
    df = pd.DataFrame(list(mydict.items()), columns=['词语', '频次'])
    df = df.sort_values('频次', ascending=False)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # 柱状图
    df.set_index('词语')['频次'].plot(kind='bar', ax=ax1, color='skyblue')
    ax1.set_title(f'{title} - 柱状图')
    ax1.set_xlabel('词语')
    ax1.set_ylabel('出现次数')
    ax1.tick_params(axis='x', rotation=45)
    
    # 饼图（只显示前10个）
    if len(df) > 10:
        top10 = df.head(10)
        ax2.pie(top10['频次'], labels=top10['词语'], autopct='%1.1f%%', startangle=90)
    else:
        ax2.pie(df['频次'], labels=df['词语'], autopct='%1.1f%%', startangle=90)
    ax2.set_title(f'{title} - 饼图')
    
    plt.tight_layout()
    plt.show()

# 可视化指定词汇统计结果
print("\n正在生成可视化图表...")
draw_dict(terms_dict, "指定词汇词频统计")

# 可视化全文高频词统计结果
draw_dict(full_word_dict, "全文高频词统计")

# ==================== 第四部分：详细统计报告 ====================

print("\n" + "="*50)
print("作业完成报告")
print("="*50)

# 1. 文本基本信息
total_chars = len(data_txt)
print(f"1. 文本基本信息：")
print(f"   文本总字数：{total_chars} 字")
print(f"   文本预览（前200字）：")
print(f"   {data_txt[:200]}...")

# 2. 指定词汇详细统计
print(f"\n2. 指定词汇详细统计：")
for term, count in terms_dict.items():
    print(f"   '{term}' 出现 {count} 次")

# 3. 全文词频统计摘要
print(f"\n3. 全文词频统计摘要：")
print(f"   统计到的不同词语数量：{len(full_word_dict)} 个")
print(f"   前5个高频词：")
for i, (word, count) in enumerate(list(full_word_dict.items())[:5], 1):
    print(f"     第{i}名：'{word}' - {count}次")

# 4. 数据分析
print(f"\n4. 数据分析：")
print(f"   '黄旭华' 在全文中的出现频率：{terms_dict.get('黄旭华', 0)/total_chars*10000:.2f}‱")
print(f"   '核潜艇' 在全文中的出现频率：{terms_dict.get('核潜艇', 0)/total_chars*10000:.2f}‱")

# 5. 保存结果到文件
output_data = {
    '指定词汇统计': terms_dict,
    '全文高频词统计': full_word_dict,
    '文本信息': {
        '总字数': total_chars,
        '不同词语数': len(full_word_dict)
    }
}

import json
with open('词频统计结果.json', 'w', encoding='utf-8') as f:
    json.dump(output_data, f, ensure_ascii=False, indent=2)

print(f"\n5. 结果已保存到 '词频统计结果.json' 文件")
print("="*50)