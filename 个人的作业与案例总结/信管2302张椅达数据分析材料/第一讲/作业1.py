# 第三段文本处理
text3 = "黄旭华，1926年3月12日出生于广东省汕尾市，原籍广东省揭阳市。1949年毕业于上海交通大学。历任北京海军核潜艇研究室副总工程师、中船重工集团公司核潜艇总体研究设计所研究员、名誉所长。1994年当选为中国工程院院士。"

import jieba

# 1. 基本分词
seg_list = jieba.cut(text3)
print("基本分词结果：")
print('/'.join(seg_list))
print()

# 2. 加入用户词典（直接在代码中添加）
user_dict_list = [
    '黄旭华',
    '北京海军核潜艇研究室',
    '中船重工集团公司',
    '核潜艇总体研究设计所',
    '中国工程院院士',
    '名誉所长',
    '长短期记忆网络',
    '时间递归神经网络',
    '上海交通大学'
]

# 将用户词典添加到jieba
for word in user_dict_list:
    jieba.add_word(word, freq=1000, tag='n')

seg_list_dict = jieba.cut(text3)
print("加入用户词典后：")
print('/'.join(seg_list_dict))
print()

# 3. 加入停用词（直接在代码中定义）
stopwords = [
    '，', '。', '、', '的', '于', '为', '是', '在', '了', '和', '就', '都', '而', '及', 
    '与', '着', '或', '一个', '没有', '有', '很', '非常', '吗', '呢', '吧', '啊', '嗯', 
    '哦', '呀', '哟', '唉', '喂', '咦', '哼', '呸'
]

seg_list_stopw = jieba.cut(text3)

final = ''
for seg in seg_list_stopw:
    if seg not in stopwords:
        final += seg + '/'

print("加入停用词后：")
print(final)