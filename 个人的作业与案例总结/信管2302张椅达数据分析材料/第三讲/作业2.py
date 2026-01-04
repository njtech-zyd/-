import jieba
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import numpy as np
from PIL import Image  # 用于处理图像形状
import warnings
warnings.filterwarnings('ignore')
import os

# 1. 读取文本文件
file_path = r"D:\用户数据采集\作业\用户数据\第3讲 感知世界：词云与可视化\第3讲 感知世界：词云与可视化\功勋科学家黄旭华传记序言文本.txt"

try:
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()
    print(f"文本读取成功，长度：{len(text)} 字符")
except FileNotFoundError:
    print(f"文件未找到，请检查路径：{file_path}")
    # 如果文件不存在，使用示例文本
    text = """
    黄旭华是中国核潜艇事业的先驱者和奠基人之一，隐姓埋名三十载，为我国第一代核潜艇的研制呕心沥血。
    他率领团队攻克了无数技术难关，成功研制出我国第一艘核潜艇，使中国成为世界上第五个拥有核潜艇的国家。
    他的奉献精神和科学成就，是共和国功勋科学家的杰出代表，激励着一代又一代科技工作者。
    """
    print("已使用示例文本进行演示")

# 2. 中文分词
print("正在进行中文分词...")
words = jieba.lcut(text)
print(f"分词数量：{len(words)}")
# 过滤掉单个字符和空白
words_filtered = [word for word in words if len(word) > 1 and word.strip() != '']
text_cut = " ".join(words_filtered)

# 3. 设置中文字体 - 修改这部分！！！
# 尝试多种可能的字体路径
font_path_options = [
    "simsun.ttf",  # 当前目录
    "simsun.ttc",  # 有些系统中是.ttc格式
    "C:/Windows/Fonts/simsun.ttc",  # Windows常见路径
    "C:/Windows/Fonts/simsun.ttf",  # Windows常见路径
    "C:/Windows/Fonts/msyh.ttc",    # 微软雅黑
    "C:/Windows/Fonts/msyh.ttf",    # 微软雅黑
    "C:/Windows/Fonts/simhei.ttf",  # 黑体
    "C:/Windows/Fonts/simkai.ttf",  # 楷体
]

font_path = None
for fp in font_path_options:
    if os.path.exists(fp):
        font_path = fp
        print(f"找到字体文件：{font_path}")
        break

if font_path is None:
    # 如果都找不到，尝试使用matplotlib的字体
    import matplotlib.font_manager as fm
    fonts = fm.findSystemFonts()
    # 查找中文字体
    chinese_fonts = [f for f in fonts if any(name in f.lower() for name in ['simsun', 'msyh', 'simhei', 'simkai', 'pingfang'])]
    if chinese_fonts:
        font_path = chinese_fonts[0]
        print(f"使用系统字体：{font_path}")
    else:
        # 如果还是找不到，使用默认字体（可能会显示方框）
        font_path = None
        print("警告：未找到中文字体文件，词云可能显示为方框")

# 4. 创建词云对象（基础版本）
print("生成词云中...")
wordcloud_params = {
    'width': 1200,
    'height': 800,
    'background_color': 'white',
    'max_words': 200,
    'max_font_size': 150,
    'min_font_size': 10,
    'collocations': False,
    'random_state': 42,
    'contour_width': 1,
    'contour_color': 'steelblue'
}

# 如果找到字体，添加字体参数
if font_path:
    wordcloud_params['font_path'] = font_path

try:
    wordcloud = WordCloud(**wordcloud_params).generate(text_cut)
    
    # 5. 显示词云
    plt.figure(figsize=(14, 10))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title("功勋科学家黄旭华传记序言 - 词云图", fontsize=16, pad=20)
    plt.tight_layout()
    plt.show()
    
    # 6. 保存词云图
    output_path = "黄旭华传记词云.png"
    wordcloud.to_file(output_path)
    print(f"词云已保存至：{output_path}")
    
except Exception as e:
    print(f"生成词云时出错：{e}")
    print("尝试使用默认参数重新生成...")
    # 去掉字体参数再试一次
    wordcloud = WordCloud(**{k: v for k, v in wordcloud_params.items() if k != 'font_path'}).generate(text_cut)
    plt.figure(figsize=(14, 10))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title("功勋科学家黄旭华传记序言 - 词云图", fontsize=16, pad=20)
    plt.tight_layout()
    plt.show()
    output_path = "黄旭华传记词云.png"
    wordcloud.to_file(output_path)

# 7. 使用蒙版图片 - 黄旭华图片
mask_image_path = r"D:\用户数据采集\作业\用户数据\第3讲 感知世界：词云与可视化\第3讲 感知世界：词云与可视化\huangxuhua.jpg"
print(f"\n尝试使用蒙版图片：{mask_image_path}")

try:
    # 加载蒙版图片
    if os.path.exists(mask_image_path):
        print("找到蒙版图片，正在处理...")
        
        # 打开图片并转换为灰度
        mask_image = Image.open(mask_image_path)
        
        # 显示原始蒙版图片
        plt.figure(figsize=(10, 8))
        plt.imshow(mask_image)
        plt.axis('off')
        plt.title("蒙版图片 - 黄旭华", fontsize=14, pad=20)
        plt.tight_layout()
        plt.show()
        
        # 转换为灰度图并二值化作为蒙版
        # WordCloud需要二维数组作为mask，白色部分(255)将被忽略，黑色部分(0)将显示词云
        mask_array = np.array(mask_image.convert('L'))  # 转换为灰度
        
        # 如果图片背景是白色，我们需要反转（通常白色部分表示蒙版，黑色部分显示词云）
        # 但我们希望人像部分显示词云，背景透明，所以需要根据图片调整
        # 这里我们假设背景较亮（接近白色），人像较暗（接近黑色）
        
        # 方法1：直接使用灰度图作为蒙版（暗色部分显示词云）
        mask = mask_array
        
        # 方法2：二值化处理（如果需要更清晰的边缘）
        # threshold = 128
        # mask = np.where(mask_array > threshold, 255, 0).astype(np.uint8)
        
        # 方法3：反转（如果背景是黑色，人像是白色）
        # mask = 255 - mask_array
        
        print(f"蒙版图片尺寸：{mask.shape}")
        
        # 创建带蒙版的词云
        mask_params = {
            'background_color': 'white',
            'max_words': 300,  # 蒙版可能需要更多词
            'max_font_size': 100,
            'min_font_size': 8,
            'random_state': 42,
            'colormap': 'plasma',  # 使用更鲜艳的颜色映射
            'mask': mask,
            'contour_width': 0,  # 去掉轮廓线
            'mode': 'RGBA'  # 使用RGBA模式支持透明度
        }
        
        if font_path:
            mask_params['font_path'] = font_path
        
        # 生成蒙版词云
        print("生成蒙版词云中...")
        wordcloud_mask = WordCloud(**mask_params).generate(text_cut)
        
        # 显示蒙版词云
        plt.figure(figsize=(14, 12))
        plt.imshow(wordcloud_mask, interpolation='bilinear')
        plt.axis('off')
        plt.title("功勋科学家黄旭华传记序言 - 蒙版词云", fontsize=16, pad=20)
        plt.tight_layout()
        plt.show()
        
        # 保存蒙版词云
        mask_output = "黄旭华传记词云_蒙版.png"
        wordcloud_mask.to_file(mask_output)
        print(f"蒙版词云已保存至：{mask_output}")
        
        # 可选：生成带有背景色的蒙版词云
        print("\n生成带有背景色的蒙版词云...")
        mask_params_with_bg = mask_params.copy()
        mask_params_with_bg['background_color'] = 'lightblue'  # 设置背景色
        mask_params_with_bg['mode'] = 'RGB'  # 使用RGB模式
        
        wordcloud_mask_bg = WordCloud(**mask_params_with_bg).generate(text_cut)
        
        plt.figure(figsize=(14, 12))
        plt.imshow(wordcloud_mask_bg, interpolation='bilinear')
        plt.axis('off')
        plt.title("功勋科学家黄旭华传记序言 - 蒙版词云（带背景色）", fontsize=16, pad=20)
        plt.tight_layout()
        plt.show()
        
        bg_output = "黄旭华传记词云_蒙版_背景色.png"
        wordcloud_mask_bg.to_file(bg_output)
        print(f"带背景色的蒙版词云已保存至：{bg_output}")
        
    else:
        print(f"未找到蒙版图片：{mask_image_path}")
        print("将使用圆形蒙版替代...")
        
        # 如果找不到图片，使用圆形蒙版
        mask = np.zeros((800, 1200), dtype=np.uint8)
        center_x, center_y = 600, 400
        radius = 350
        y, x = np.ogrid[:800, :1200]
        mask_area = (x - center_x)**2 + (y - center_y)**2 <= radius**2
        mask[mask_area] = 255
        
        shape_params = {
            'background_color': 'white',
            'max_words': 150,
            'contour_width': 2,
            'contour_color': 'navy',
            'random_state': 42,
            'mask': mask
        }
        
        if font_path:
            shape_params['font_path'] = font_path
        
        wordcloud_shape = WordCloud(**shape_params).generate(text_cut)
        
        plt.figure(figsize=(14, 10))
        plt.imshow(wordcloud_shape, interpolation='bilinear')
        plt.axis('off')
        plt.title("功勋科学家黄旭华传记序言 - 圆形词云", fontsize=16, pad=20)
        plt.tight_layout()
        plt.show()
        
        shape_output = "黄旭华传记词云_圆形.png"
        wordcloud_shape.to_file(shape_output)
        print(f"圆形词云已保存至：{shape_output}")
        
except Exception as e:
    print(f"蒙版词云生成失败：{e}")
    import traceback
    traceback.print_exc()

# 8. 显示词频最高的20个词
from collections import Counter
word_freq = Counter(words_filtered)
top_words = word_freq.most_common(20)

print("\n词频最高的20个词：")
print("-" * 30)
for word, freq in top_words:
    print(f"{word:<8} : {freq:>4}")
print("-" * 30)

# 9. 可选：添加停用词过滤
stopwords = set(['我们', '他们', '一个', '没有', '这个', '但是', '因为', '所以', '就是', '可以'])
words_without_stop = [word for word in words_filtered if word not in stopwords]
text_without_stop = " ".join(words_without_stop)

clean_params = {
    'width': 1200,
    'height': 800,
    'background_color': 'white',
    'max_words': 150,
    'random_state': 42,
    'colormap': 'viridis'
}

if font_path:
    clean_params['font_path'] = font_path

wordcloud_clean = WordCloud(**clean_params).generate(text_without_stop)

plt.figure(figsize=(14, 10))
plt.imshow(wordcloud_clean, interpolation='bilinear')
plt.axis('off')
plt.title("功勋科学家黄旭华传记序言 - 过滤停用词后", fontsize=16, pad=20)
plt.tight_layout()
plt.show()

clean_output = "黄旭华传记词云_过滤后.png"
wordcloud_clean.to_file(clean_output)
print(f"过滤后词云已保存至：{clean_output}")

print("\n程序执行完成！")