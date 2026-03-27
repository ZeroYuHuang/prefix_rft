import os
import uuid
import numpy as np
from datasets import load_dataset, concatenate_datasets
from transformers import AutoTokenizer
from custom_rewards.math_oat import rfn, rfn_after_think

if __name__ == "__main__":
    data_files = "/mnt/jfs2/cth/085b13/_data/raw_dataset/distillation/luffy_train//luffy_train_r1_distill_1p5b_unverified.jsonl"
    HOME_DIR = "/mnt/jfs2/cth/085b13"
    dataset = load_dataset("json", data_files=data_files)
    print(dataset)
    seed, size = 42, 16
    save_dir = f"{HOME_DIR}/_data/processed_dataset_new/luffy_analysis_1p5b_{size}k_seed_{seed}_sys_wo_format"
    sys_prompt_wo_format = 'Your task is to follow a systematic, thorough reasoning process before providing the final solution. This involves analyzing, summarizing, exploring, reassessing, and refining your thought process through multiple iterations. Structure your response into two sections: Thought and Solution. Each thought should include detailed analysis, brainstorming, verification, and refinement of ideas. In the Solution section, provide the final, logical, and accurate answer, clearly derived from the exploration in the Thought section. Include the answer in \\boxed{} for closed-form results like multiple choices or mathematical solutions'
    tok = AutoTokenizer.from_pretrained(f"{HOME_DIR}/_models/Qwen2.5-Math-7B")
    
    def change_sys_prompt(example, idx):
        raw_prompt = example["prompt"][1]['content']
        example["prompt"] = [
            {"content": sys_prompt_wo_format, "role": "system"},
            {"content": raw_prompt, "role": "user"}
        ]
        return example
    
    dataset = dataset.map(function=change_sys_prompt, with_indices=True, num_proc=16)

    def get_label_and_length(example, idx, tok, rfn):
        demos =  example["demos"]
        # get length
        example["num_demo"] = len(demos)
        example["demos_len"] = [len(tok.tokenize(d)) for d in demos]
        # get correctness
        gt = example["reward_model"]['ground_truth']
        example["correctness"] = [rfn(None, gt, response_str=d)['score'] for d in demos]
        example["avg_correctness"] = np.mean(example["correctness"])
        example["difficulty"] = np.sum(example["correctness"])
        return example

    dataset = dataset.map(
        function=get_label_and_length, 
        with_indices=True, 
        num_proc=16,
        fn_kwargs={
            'tok': tok,
            'rfn': rfn
        }
    )    
    
    # filter out imcomplete demos
    def filter_out_imcomplete_demos(example):
        demos = example["demos"]
        demos_len = example["demos_len"]
        correctness = example['correctness']
        new_demos, new_demos_len, new_correctness = [], [], []
        for d, dl, corr, in zip(demos, demos_len, correctness):
            if dl < 8200:
                new_demos.append(d)
                new_demos_len.append(dl)
                new_correctness.append(corr)
        example["demos"] = new_demos
        example["num_demo"] = len(new_demos)
        example["demos_len"] = new_demos_len
        example["correctness"] = new_correctness
        example["avg_correctness"] = np.mean(new_correctness) if len(new_correctness) > 0 else -1
        example["difficulty"] = np.sum(new_correctness) if len(new_correctness) > 0 else 0
        return example
    dataset = dataset.map(function=filter_out_imcomplete_demos, num_proc=16)

    def keep_only_one_demo(example):
        demos = example["demos"]
        demos_len = example["demos_len"]
        correctness = example['correctness']
        new_demos, new_demos_len = [], []
        for d, dl, corr, in zip(demos, demos_len, correctness):
            if corr:
                new_demos.append(d)
                new_demos_len.append(dl)
                break
        example["demos"] = new_demos
        example["demos_len"] = new_demos_len
        example["old_demos"] = demos
        example["old_demos_len"] = demos_len
        example["old_correctness"] = correctness
        return example
    dataset = dataset.map(function=keep_only_one_demo, num_proc=16)
    dataset = dataset.filter(lambda x: len(x["demos"]) > 0, num_proc=16)
    print(dataset)

    print("Uniformly sample across difficulty")
    difficuties = set(dataset["train"]["difficulty"])
    split_datasets_by_filter = {}
    for level in difficuties:
        filtered_dataset = dataset['train'].filter(lambda example: example['difficulty'] == level)
        # 只有当筛选出的数据集不为空时，才添加到字典中
        if len(filtered_dataset) > 0:
            split_datasets_by_filter[level] = filtered_dataset
        print(level, len(filtered_dataset))
    
    sampled_splits = []
    sampled_valid_splits = []
    target_size = size * 1024
    num_per_level = target_size // 8
    for level, split_dataset in split_datasets_by_filter.items():
        if level == 0:
            continue
        shuffled_subset = split_dataset.shuffle(seed=42)
        sampled_splits.append(shuffled_subset.select(range(num_per_level)))
        sampled_valid_splits.append(shuffled_subset.select(range(num_per_level // 4)))
        print(f"从难度 {level} 中采样了 {num_per_level} 条样本。")

    # 3. 将所有采样出的数据集合并成一个大的数据集
    dataset = concatenate_datasets(sampled_splits)
    val_dataset = concatenate_datasets(sampled_valid_splits)
    print(dataset)
    print(val_dataset)

    dataset = dataset.remove_columns(["avg_correctness", "correctness"])
    val_dataset = val_dataset.remove_columns(["avg_correctness", "correctness"])

    def add_response_and_uid(example):
        example["response"] = example["demos"][0]
        example["uid"] = str(uuid.uuid4())
        return example

    dataset = dataset.map(function=add_response_and_uid, num_proc=16)
    val_dataset = val_dataset.map(function=add_response_and_uid, num_proc=16)
    print(dataset)
    print(f"Save to {save_dir}/train.parquet")
    dataset.to_parquet(os.path.join(save_dir, f'train.parquet'))
    print(f"Save to {save_dir}/valid.parquet")
    val_dataset.to_parquet(os.path.join(save_dir, f'valid.parquet'))

    
"""
Save to /mnt/jfs2/cth/085b13/_data/processed_dataset_new/luffy_analysis_1p5b_4k_seed_42_sys_wo_format/train.parquet
"""
    
