"""
豆瓣评论数据格式转换工具
功能：将原始豆瓣评论数据转换为 ['评论ID', '国家', '原始评论', '分词后评论'] 格式
"""

import pandas as pd
import numpy as np
import jieba
import uuid
from datetime import datetime
from tqdm import tqdm
import argparse
import warnings
warnings.filterwarnings('ignore')

def init_jieba():
    """初始化jieba分词器"""
    print("正在初始化中文分词器...")
    jieba.initialize()
    print("分词器初始化完成")

def generate_comment_id(method="uuid", prefix="COMMENT", length=16):
    """生成评论ID，支持UUID和时间戳两种方式"""
    if method == "uuid":
        return str(uuid.uuid4()).replace('-', '')[:length]
    elif method == "timestamp":
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        random_suffix = ''.join([str(np.random.randint(0, 10)) for _ in range(6)])
        return f"{prefix}_{timestamp}_{random_suffix}"
    else:
        raise ValueError("ID生成方法仅支持 'uuid' 和 'timestamp'")

def segment_chinese_text(text, min_length=1, stop_words=None):
    """中文文本分词处理"""
    if pd.isna(text) or str(text).strip() == "":
        return ""
    
    text_clean = str(text).strip()
    words = jieba.lcut(text_clean)
    
    # 过滤短词
    if min_length > 1:
        words = [word for word in words if len(word) >= min_length]
    
    # 过滤停用词
    if stop_words and isinstance(stop_words, list):
        words = [word for word in words if word not in stop_words]
    
    return " ".join(words)

def convert_douban_format(input_path, output_path, country_default="未知", 
                         id_method="uuid", min_word_length=1):
    """主转换函数"""
    print("="*50)
    print("豆瓣评论数据格式转换工具")
    print("="*50)
    
    # 初始化
    init_jieba()
    
    # 读取数据
    print(f"\\n正在读取原始数据: {input_path}")
    try:
        df = pd.read_csv(input_path, encoding='utf-8')
    except UnicodeDecodeError:
        df = pd.read_csv(input_path, encoding='gbk')
    
    # 数据转换
    print("\\n开始数据格式转换...")
    new_df = pd.DataFrame()
    
    # 生成评论ID
    print("正在生成评论ID...")
    new_df['评论ID'] = [generate_comment_id(method=id_method) for _ in range(len(df))]
    
    # 设置国家字段
    new_df['国家'] = country_default
    
    # 原始评论
    new_df['原始评论'] = df['comment'].astype(str).fillna("")
    
    # 分词处理
    print("正在进行中文分词...")
    tqdm.pandas(desc="中文分词")
    new_df['分词后评论'] = new_df['原始评论'].progress_apply(
        lambda x: segment_chinese_text(x, min_length=min_word_length)
    )
    
    # 保存结果
    new_df.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"\\n数据已保存到: {output_path}")
    print(f"转换完成！共处理 {len(new_df)} 条评论")
    
    return new_df

# 运行示例
if __name__ == "__main__":
    # 配置参数
    input_file = "E:\大学奇奇怪怪的作业\实验\douban_FINAL_CLEANED.csv"      # 输入文件路径
    output_file = "E:\大学奇奇怪怪的作业\实验\douban_comments_converted.csv" # 输出文件路径
    
    # 执行转换
    converted_data = convert_douban_format(
        input_path=input_file,
        output_path=output_file,
        country_default="中国",       # 国家默认值
        id_method="uuid",             # ID生成方式
        min_word_length=1             # 最小词长度
    )