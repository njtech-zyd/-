import pandas as pd

# 读取原始豆瓣文件
file_path = "E:\大学奇奇怪怪的作业\实验\douban_comments_converted.csv"

# 自动检测编码读取
def read_csv_with_encoding(file_path):
    encodings = ["utf-8", "gbk", "gb2312", "latin-1"]
    for encoding in encodings:
        try:
            return pd.read_csv(file_path, encoding=encoding)
        except UnicodeDecodeError:
            continue
    raise ValueError("无法识别文件编码")

df = read_csv_with_encoding(file_path)

# 1. 补充“国家”字段（豆瓣评论均为中国观众）
df["国家"] = "中国"

# 2. 字段映射：统一为分析所需的标准字段（按实际文件字段名调整！）
# 请根据你的文件字段名修改以下映射（示例：若原始字段为comment_text→原始评论，seg_comment→分词后评论）
field_mapping = {
    "comment_id":"评论ID", 
    "国家": "国家",# 替换为你文件中的“评论唯一标识”字段名
    "原始评论": "原始评论",   # 替换为你文件中的“原始评论文本”字段名
    "分词后评论": "分词后评论",  # 替换为你文件中的“分词后评论”字段名
    
}

# 筛选并重命名字段
standard_df = df[list(field_mapping.keys())].rename(columns=field_mapping)

# 3. 处理空值（删除无评论内容的行）
standard_df = standard_df.dropna(subset=["原始评论", "分词后评论"])
standard_df = standard_df[standard_df["原始评论"].str.strip() != ""]

# 4. 保存为标准分析数据（供后续步骤使用）
output_path = "E:\大学奇奇怪怪的作业\实验\reprocessed_data.csv"
standard_df.to_csv(output_path, index=False, encoding="utf-8")

print(f"数据预处理完成！")
print(f"标准数据总行数：{len(standard_df)}")
print(f"标准数据字段：{standard_df.columns.tolist()}")
print(f"数据保存路径：{output_path}")
print(f"\n前3行预览：")
print(standard_df.head(3))