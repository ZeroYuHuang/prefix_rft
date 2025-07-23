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
    data_files = "/mnt/jfs2/cth/085b13/_data/raw_dataset/distillation/luffy_train//luffy_train_r1_distill_1p5b_unverified.jsonl"
    HOME_DIR = "/mnt/jfs2/cth/085b13"
    dataset_name = "luffy_train_distill1p5b_sys_wo_format_all_8k"
    save_dir = f"{HOME_DIR}/_data/processed_dataset_new/{dataset_name}"
    dataset = load_dataset("json", data_files=data_files)
    print(dataset)
    """
    Dataset({
        features: ['data_source', 'prompt', 'ability', 'reward_model', 'extra_info', 'problem', 'demos'],
        num_rows: 45714
    })
    """
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

    def filter_out_imcomplete_demos(example):
        demos = example["demos"]
        demos_len = example["demos_len"]
        correctness = example['correctness']
        new_demos, new_demos_len, new_correctness = [], [], []
        for d, dl, corr, in zip(demos, demos_len, correctness):
            if dl < 8000:
                new_demos.append(d)
                new_demos_len.append(dl)
                new_correctness.append(corr)
        example["demos"] = new_demos
        example["num_demo"] = len(new_demos)
        example["demos_len"] = new_demos_len
        example["correctness"] = new_correctness
        example["avg_correctness"] = np.mean(new_correctness) if len(new_correctness) > 0 else -1
        return example


    dataset = dataset.map(
        function=filter_out_imcomplete_demos,
        num_proc=16
    )

    data_split = dataset['train']
    print("\nConverting dataset to pandas DataFrame for plotting...")
    df = data_split.to_pandas()

    # Visualize num_demo and save the figure
    print("\nVisualizing 'num_demo' distribution...")
    plt.figure(figsize=(12, 7))
    ax = sns.histplot(data=df, x='num_demo', discrete=True, shrink=0.8)
    for p in ax.patches:
        height = p.get_height()
        if height > 0: # 只为高度大于0的柱子添加标签
            ax.annotate(f'{int(height)}', 
                        (p.get_x() + p.get_width() / 2., height), 
                        ha = 'center', va = 'center', 
                        xytext = (0, 9), 
                        textcoords = 'offset points')
    plt.title('Distribution of Number of Demos per Example')
    plt.xlabel('Number of Demonstrations')
    plt.ylabel('Count')
    plt.grid(axis='y', alpha=0.75)
    num_demo_fig_path = f"/data/stat_num_demo_{dataset_name}.png"
    plt.savefig(num_demo_fig_path)
    plt.close()
    print(f"Saved 'num_demo' visualization to {num_demo_fig_path}")

    # Visualize avg_correctness and save the figure
    print("Visualizing 'avg_correctness' distribution...")
    plt.figure(figsize=(12, 7))
    ax = sns.histplot(data=df, x='avg_correctness', kde=True, bins=20)
    for p in ax.patches:
        height = p.get_height()
        if height > 0: # 只为高度大于0的柱子添加标签
            ax.annotate(f'{int(height)}', 
                        (p.get_x() + p.get_width() / 2., height), 
                        ha = 'center', va = 'center', 
                        xytext = (0, 9), 
                        textcoords = 'offset points')
    plt.title('Distribution of Average Correctness Score')
    plt.xlabel('Average Correctness')
    plt.ylabel('Count')
    plt.grid(axis='y', alpha=0.75)
    avg_correctness_fig_path = f"/data/stat_avg_correctness_{dataset_name}.png"
    plt.savefig(avg_correctness_fig_path)
    plt.close()
    print(f"Saved 'avg_correctness' visualization to {avg_correctness_fig_path}")


    print(dataset)

    dataset = dataset['train']
    print(f"Save to {save_dir}/train.parquet")
    dataset.to_parquet(os.path.join(save_dir, f'train.parquet'))
        

