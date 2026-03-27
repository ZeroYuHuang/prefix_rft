import os
import json
import argparse
from collections import defaultdict
from datasets import Dataset
from recipe.prefix_rft_v2.custom_rewards.math_oat import rfn

def merge_jsonl_files_from_folder(folder_path, output_file, merged_key):
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
    uid_data = {}

    # 遍历每个文件
    for file_path in file_paths:
        print(f"Merging {file_path}")
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                # 解析每一行为字典
                data = json.loads(line.strip())
                # 提取问题和答案
                uid = data.get('uid')
                # 如果问题第一次出现，初始化数据结构
                if uid not in uid_data:
                    uid_data[uid] = {key: value for key, value in data.items() if key != merged_key}
                    # things we need mergw
                    uid_data[uid][merged_key] = []  # 初始化答案列表
                # 将当前答案添加到答案列表中
                uid_data[uid][merged_key].extend(data[merged_key])

    # 将合并后的数据写入输出文件
    # with open(output_file, 'w', encoding='utf-8') as f:
        # for merged_data in uid_data.values():
        #     # 写入 JSONL 文件
        #     f.write(json.dumps(merged_data, ensure_ascii=False) + '\n')
    dataset = list(uid_data.values())
    return dataset

# 示例用法
if __name__ == "__main__":
    # 输入文件夹路径
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_folder", type=str, required=True)
    parser.add_argument("--output_file", type=str, required=True)
    parser.add_argument("--merged_key", type=str, required=True)

    args = parser.parse_args()
    for k, v in vars(args).items():
        print(f"{k}: {v}")
    # input_folder = "/mnt/jfs2/cth/085b13/_data/raw_dataset/distillation/luffy_train/DeepSeek-R1-Distill-Qwen-7B"
    # demo_source = input_folder.split("/")[-1]
    # # 输出文件路径
    # output_file = "/mnt/jfs2/cth/085b13/_data/raw_dataset/distillation/luffy_train//luffy_train_r1_distill_7b_unverified.jsonl"
    # 调用函数合并文件
    dataset = merge_jsonl_files_from_folder(args.input_folder, args.output_file, args.merged_key)
    dataset = Dataset.from_list(dataset)
    print(dataset)

    # verify
    def verify_correctness(example):
        correctness = []
        gt = example["reward_model"]['ground_truth']
        example["analysis_corr"] = [rfn(None, gt, response_str=d)['score'] for d in example['analysis_gen']]
        return example

    dataset = dataset.map(function=verify_correctness, num_proc=16)
    print(dataset)
    print(f"Save to {args.output_file}")
    dataset.to_json(args.output_file)
            
    
