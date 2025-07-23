import os
import json
from datasets import load_dataset, Dataset
from custom_rewards.math_oat import rfn, rfn_after_think
from transformers import AutoTokenizer
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

if __name__ == "__main__":
    
    data_files = "/mnt/jfs2/cth/085b13/_data/processed_dataset_new/luffy_train_distill1p5b_sys_wo_format_all_8k/train.parquet"
    save_dir = "/mnt/jfs2/cth/085b13/_data/processed_dataset_new/luffy_train_distill1p5b_sys_wo_format_all_8k_correct_only"
    dataset = load_dataset("parquet", data_files=data_files)['train']
    print(dataset)        

    def filter_uncorrect_demos(example):
        demos = example["demos"]
        correctness = example["correctness"]
        assert len(demos) == len(correctness)
        new_demos = []
        for d, c in zip(demos, correctness):
            if c:
                new_demos.append(d)
        example["demos"] = new_demos
        return example

    dataset = dataset.map(filter_uncorrect_demos, num_proc=16)
    dataset = dataset.remove_columns(['num_demo', 'demos_len', 'correctness', 'avg_correctness'])
    print(f"Save to {save_dir}/train.parquet")
    dataset.to_parquet(os.path.join(save_dir, f'train.parquet'))
        