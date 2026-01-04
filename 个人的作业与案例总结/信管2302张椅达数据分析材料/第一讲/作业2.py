import jieba

# 1. 第一段文本基本分词
seg_list1 = jieba.cut("曾经有一份真诚的爱情摆在我的面前，我没有珍惜，等到失去的时候才追悔莫及，人世间最痛苦的事情莫过于此。如果上天能够给我一个重新来过的机会，我会对那个女孩子说三个字：‘我爱你’。如果非要给这份爱加上一个期限，我希望是，一万年")
print("第一段基本分词结果：")
print('$'.join(seg_list1))
print()

# 2. 第二段文本基本分词
seg_list2 = jieba.cut("LSTM（Long Short-Term Memory）是长短期记忆网络，是一种时间递归神经网络，适合于处理和预测时间序列中间隔和延迟相对较长的重要事件。")
print("第二段基本分词结果：")
print('@'.join(seg_list2))
print()

# 3. 第二段文本加入用户词典
# 先在代码中添加用户词典
user_dict_list = [
    '长短期记忆网络',
    '时间递归神经网络',
    'LSTM',
    'Long Short-Term Memory'
]

# 将用户词典添加到jieba
for word in user_dict_list:
    jieba.add_word(word, freq=1000, tag='n')

seg_list_dict = jieba.cut("LSTM（Long Short-Term Memory）是长短期记忆网络，是一种时间递归神经网络，适合于处理和预测时间序列中间隔和延迟相对较长的重要事件。")
print("第二段加入用户词典后：")
print('/'.join(seg_list_dict))
print()

# 4. 第一段文本加入停用词
# 直接在代码中定义停用词
stopwords = [
    '，', '。', '、', '的', '于', '为', '是', '在', '了', '和', '就', '都', '而', '及', 
    '与', '着', '或', '一个', '没有', '有', '很', '非常', '吗', '呢', '吧', '啊', '嗯', 
    '哦', '呀', '哟', '唉', '喂', '咦', '哼', '呸', '‘', '’', '：'
]

seg_list_stopw = jieba.cut("曾经有一份真诚的爱情摆在我的面前，我没有珍惜，等到失去的时候才追悔莫及，人世间最痛苦的事情莫过于此。如果上天能够给我一个重新来过的机会，我会对那个女孩子说三个字：‘我爱你’。如果非要给这份爱加上一个期限，我希望是，一万年")

final = ''
#这是一行注释，进行分词结果的过滤
for seg in seg_list_stopw:
    if seg not in stopwords:
        final += seg + '/' #叠加，累加

print("第一段加入停用词后：")
print(final)