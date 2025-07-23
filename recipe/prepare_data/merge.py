import os
import json
from collections import defaultdict

def merge_jsonl_files_from_folder(folder_path, output_file, demo_source):
    """
    合并指定文件夹下的所有 JSONL 文件，将相同问题的答案合并到一起，并保留其他字段。

    :param folder_path: 包含 JSONL 文件的文件夹路径
    :param output_file: 输出的 JSONL 文件路径
    """
    # 获取文件夹中所有 .jsonl 文件的路径
    file_paths = [
        os.path.join(folder_path, fname)
        for fname in os.listdir(folder_path)
        if fname.endswith('.jsonl')
    ]

    if not file_paths:
        print(f"警告：文件夹 {folder_path} 中没有找到任何 .jsonl 文件。")
        return

    print(f"file_paths found: {file_paths}")

    # 使用 defaultdict 构建问题-数据映射
    question_data = {}

    # 遍历每个文件
    for file_path in file_paths:
        print(f"Merging {file_path}")
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                # 解析每一行为字典
                data = json.loads(line.strip())
                
                # 提取问题和答案
                question = data.get('problem')
                answer = data.get('demos')
                demo_source = data.get("demo_source", demo_source)
                
                if not question or not answer:
                    print(f"警告：跳过无效数据行（文件 {file_path}）：{line}")
                    continue
                
                # 如果问题第一次出现，初始化数据结构
                if question not in question_data:
                    question_data[question] = {key: value for key, value in data.items() if key != 'model_solution'}
                    # things we need mergw
                    question_data[question]['demos'] = []  # 初始化答案列表
                    
                
                # 将当前答案添加到答案列表中
                question_data[question]['demos'].extend(answer)

    # 将合并后的数据写入输出文件
    with open(output_file, 'w', encoding='utf-8') as f:
        for merged_data in question_data.values():
            # 写入 JSONL 文件
            f.write(json.dumps(merged_data, ensure_ascii=False) + '\n')

# 示例用法
if __name__ == "__main__":
    # 输入文件夹路径
    input_folder = "/mnt/jfs2/cth/085b13/_data/raw_dataset/distillation/luffy_train/DeepSeek-R1-Distill-Qwen-32B"
    demo_source = input_folder.split("/")[-1]
    # 输出文件路径
    output_file = "/mnt/jfs2/cth/085b13/_data/raw_dataset/distillation/luffy_train//luffy_train_r1_distill_32b_unverified.jsonl"

    # 调用函数合并文件
    merge_jsonl_files_from_folder(input_folder, output_file, demo_source)
    print("合并完成，结果已保存到:", output_file)