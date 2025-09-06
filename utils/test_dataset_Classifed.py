import os
import shutil
import re

# 定义文件路径
# label_dir = r'D:\AA_CodeText\Jiang_program\swin_transformer_paper\swin_transformer_paper\data_test\label'
# ASDF_dir = r'D:\AA_CodeText\Jiang_program\swin_transformer_paper\swin_transformer_paper\data_test\target_ASDF'
signal_dir = r'D:\AA_CodeText\Jiang_program\swin_transformer_paper\swin_transformer_paper\data_test\target_signal_new'
output_root = r'D:\AA_CodeText\Jiang_program\swin_transformer_paper\swin_transformer_paper\data_test\classified'

# 定义分类范围矩阵
# ranges = [
#     [1, 250, -5],
#     [251, 500, 0],
#     [501, 800, 5],
#     [801, 1500, 10],
#     [1501, 2100, 15],
#     [2101, 2500, 20]
# ]
ranges = [
    [1, 250, -5],
    [251, 500, 0],
    [501, 750, 5],
    [751, 1000, 10],
    [1001, 1250, 15],
]


# 创建输出目录结构
db_values = sorted(set(db for _, _, db in ranges))
for db in db_values:
    db_dir = os.path.join(output_root, f"{db}DB")
    os.makedirs(os.path.join(db_dir, "label"), exist_ok=True)
    os.makedirs(os.path.join(db_dir, "target_signal"), exist_ok=True)
    os.makedirs(os.path.join(db_dir, "target_ASDF"), exist_ok=True)


# 文件处理函数
def process_files(file_dir, pattern, file_type):
    for filename in os.listdir(file_dir):
        # 提取文件序号
        match = re.search(pattern, filename)
        if not match:
            continue

        file_index = int(match.group(1))

        # 确定所属的2500区块
        block_index = (file_index - 1) // 1250
        block_offset = block_index * 1250
        relative_index = file_index - block_offset

        # 查找对应的DB值
        db_value = None
        for start, end, db in ranges:
            if start <= relative_index <= end:
                db_value = db
                break

        if db_value is None:
            continue

        # 构建目标路径
        dest_dir = os.path.join(output_root, f"{db_value}DB", file_type)
        src_path = os.path.join(file_dir, filename)
        dest_path = os.path.join(dest_dir, filename)

        # 复制文件
        shutil.copy2(src_path, dest_path)


# # 处理label文件 (格式: 1.txt)
# process_files(label_dir, r'^(\d+)\.txt$', "label")

# 处理label文件 (格式: 1.txt)
process_files(signal_dir, r'^(\d+)\.txt$', "target_signal")

# # 处理target文件 (格式: GASF_Image_1.jpg)
# process_files(ASDF_dir, r'^GASF_Image_(\d+)\.jpg$', "target_ASDF")

print("文件分类完成！")