
# 导入jieba库
import jieba

# 打开并读取文本文件（三国演义前10回）
article = open('D:\用户数据采集\作业\第二讲\sanguo_10.txt', 'r', encoding='utf-8').read()

# 手动设计停用词和符号集合
dele = {'。', '！', '？', '的', '“', '”', '（', '）', ' ', '》', '《', '，'}

# 加入字典中没有的新词
jieba.add_word('老贼')

# 分词处理
words = list(jieba.cut(article))

# 创建字典存储词-词频
articleDict = {}

# 使用集合去除停用词
articleSet = set(words) - dele

# 统计词频（只统计长度大于1的词）
for w in articleSet:
    if len(w) > 1:
        articleDict[w] = words.count(w)

# 对词典中的词按词频降序排序
articlelist = sorted(articleDict.items(), key=lambda x: x[1], reverse=True)

# 输出词频前100个
for i in range(100):
    print(articlelist[i])