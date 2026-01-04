# 文件的打开与关闭
f_name = open('D:/用户数据采集/作业/第二讲/name.txt', encoding='ANSI')
data_name = f_name.read()
print(data_name[:50])
f_name.close()

# 将文本转化为列表
names = data_name.split('|')  # split一下names就是列表
print(names)

# 学习一种新的数据结构，字典
name_dict = {}

# 打开《三国演义》文本
f_txt = open('D:/用户数据采集/作业/第二讲/sanguo.txt', encoding='ANSI')
data_txt = f_txt.read()
f_txt.close()

print(data_txt[:100])

# 用count函数统计文本中的词汇
for name in names:
    name_dict[name] = data_txt.count(name)

print(name_dict)

# 定义中文字体设置函数
def make_chinese_plot_ready():
    from matplotlib import rcParams
    # rcParams['font.family'] = 'sans-serif' # 从百度下载这个字体WenQuanYi Micro Hei，放到电脑的字体库中。怎么做？百度
    rcParams['font.sans-serif'] = ['FangSong']  # 或者直接使用电脑有的字体 FangSong
    rcParams['axes.unicode_minus'] = False

# 定义绘制词频柱状图函数
def draw_dict(mydict, figsize=(8, 5)):
    import pandas as pd
    import matplotlib.pyplot as plt
    
    make_chinese_plot_ready()
    
    # 将字典转换为DataFrame
    df = pd.DataFrame(list(mydict.items()), columns=['name', 'times'])
    
    # 绘制柱状图并排序
    df.set_index('name')['times'].sort_values(ascending=False).plot(kind='bar', figsize=figsize)
    plt.tight_layout()

# 设置matplotlib在Jupyter中内嵌显示
# %pylab inline  # 在Jupyter中使用，独立运行时注释掉

# 绘制人名词频统计图
draw_dict(name_dict)

# 武器统计部分
f_weapon = open('D:/用户数据采集/作业/第二讲/weapon.txt', encoding='utf-8')
data_weapon = f_weapon.read()
print(data_weapon[:100])

weapons_origin = data_weapon.split('\n')
print(weapons_origin[:40])

weapons = []
for weapon in weapons_origin:
    if weapon != '':
        weapons.append(weapon)

weapon_dict = {}
for weapon in weapons:
    weapon_dict[weapon] = data_txt.count(weapon)

print(weapon_dict)

# 绘制武器词频统计图
draw_dict(weapon_dict)

# 保存清晰的图像
import matplotlib.pyplot as plt
plt.savefig('weapon000.png')