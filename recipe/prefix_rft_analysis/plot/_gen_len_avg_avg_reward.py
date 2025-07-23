# -*- coding: utf-8 -*-

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from datasets import load_dataset, Dataset
from tqdm import tqdm
from transformers import AutoTokenizer

# --- 全局配置和路径 ---

# 1. Hugging Face Tokenizer
print("Loading Tokenizer...")
try:
    # Using a fast tokenizer is crucial for performance
    TOKENIZER = AutoTokenizer.from_pretrained("/mnt/jfs2/cth/085b13/_models/Qwen2.5-Math-1.5B", use_fast=True)
except Exception as e:
    print(f"Fatal Error: Could not load tokenizer. Please check the path. Error: {e}")
    exit()
print("Tokenizer loaded successfully.")

# 2. 目录和文件名配置
ROOT_DIR = "/mnt/jfs2/cth/085b13/_analysis_v2"
RFT_DIR = "rft_Qwen2.5-Math-1.5B_luffy_analysis_1p5b_16k_seed_42_sys_wo_format_num_empty_8_clip_0.8_25-06-21-Jun-06-1750459811"
SFT_DIR = "sft_Qwen2.5-Math-1.5B_luffy_analysis_1p5b_16k_seed_42_sys_wo_format_lr5e-5_25-06-23-Jun-06-1750677874"
Prefix_RFT_DIR = "rft_Qwen2.5-Math-1.5B_luffy_analysis_1p5b_16k_seed_42_sys_wo_format_num_empty_7_clip_0.8_25-06-21-Jun-06-1750458724"
GEN_FILE = "gen_t1.0_top_p_0.95_max_tokens8192.json"
length_step = 640
correctness_steps = [512, 544, 576, 608]

# 3. 输出目录
OUTPUT_DIR = "/data/_figs/"


def calculate_avg_correctness_across_steps(model_path: str) -> pd.DataFrame:
    """
    (Optimized) Calculates Prefix-RFT's average correctness using efficient batch loading and grouping.
    """
    json_paths = [os.path.join(model_path, f"global_step{step}", GEN_FILE) for step in correctness_steps]
    existing_paths = [p for p in json_paths if os.path.exists(p)]

    if not existing_paths:
        print("Error: No correctness files found for Prefix-RFT.")
        return pd.DataFrame()

    print("\nCalculating baseline correctness from Prefix-RFT model (Optimized)...")
    # Load all JSON files into a single dataset
    dataset = load_dataset('json', data_files=existing_paths, split='train', cache_dir='./.cache/prefix_rft_corr_combined')
    dataset = dataset.to_pandas()
    # Use the highly efficient group_by operation
    grouped_dataset = dataset.groupby('uid')
    
    results = []
    for uid, group in tqdm(grouped_dataset, desc="  Aggregating correctness"):
        # 在这里, 'uid' 就是分组的名称 (例如 'problem_123')
        # 'group' 就是一个只包含该 uid 所有行的 DataFrame
        all_scores = [score for sublist in group['analysis_corr'] for score in sublist]
        if all_scores:
            avg_correctness = np.mean(all_scores)
            results.append({'uid': uid, 'avg_correctness': avg_correctness})

    return pd.DataFrame(results)


def _calculate_batch_token_lengths(batch):
    """
    (UPDATED) Helper function for dataset.map.
    It now filters out generations with token length > 8192 before averaging.
    """
    avg_lengths = []
    for gen_list in batch['analysis_gen']:
        if not gen_list:
            avg_lengths.append(0)
            continue

        # Tokenize all generations in the list once
        all_token_ids = TOKENIZER(gen_list).input_ids
        
        # Get all token lengths
        all_token_lengths = [len(ids) for ids in all_token_ids]
        
        # --- NEW: Filter out generations that were likely truncated ---
        filtered_lengths = [length for length in all_token_lengths if length < 1000000]
        
        # Calculate the mean of the filtered lengths
        if filtered_lengths:
            avg_len = np.mean(filtered_lengths)
        else:
            # Handle the edge case where all generations for a prompt were truncated
            avg_len = -1
            
        avg_lengths.append(avg_len)
        
    return {"avg_token_length": avg_lengths}


def calculate_model_token_length(model_path: str) -> pd.DataFrame:
    """
    (Optimized) Calculates average token length using dataset.map() for parallel processing.
    """
    model_name = os.path.basename(model_path)
    len_json_path = os.path.join(model_path, f"global_step{length_step}", GEN_FILE)
    
    print(f"\nCalculating token length for model: {model_name} (Optimized)...")
    if not os.path.exists(len_json_path):
        print(f"    Error: Length data file not found: {len_json_path}")
        return pd.DataFrame()

    dataset = load_dataset('json', data_files=len_json_path, split='train', cache_dir=f'./.cache/{model_name}_len_{length_step}')
    
    # Use map for fast, parallelized processing
    processed_dataset = dataset.map(
        _calculate_batch_token_lengths,
        batched=True,
        num_proc=os.cpu_count(), # Use all available CPU cores
        desc="Processing token lengths"
    )
    processed_dataset = processed_dataset.filter(lambda x: x["avg_token_length"] > 0, num_proc=16)
    
    # Convert the processed dataset to pandas, keeping only relevant columns
    processed_dataset = processed_dataset.to_pandas()
    print(processed_dataset)
    df = processed_dataset[['uid', 'avg_token_length']]
    return df

def load_demonstration_length(dir_path: str) -> pd.DataFrame:
    json_path = os.path.join(dir_path, f"global_step128", "forward.json")
    dataset = load_dataset("json", data_files=json_path)["train"]
    def get_demo_len(example):
        example["demo_len"] = len(example["token_entropy"])
        return example
    dataset = dataset.map(function=get_demo_len, num_proc=16)
    df = dataset.to_pandas()
    grouped_df = df.groupby("uid", as_index=False).agg(
        demo_len_list=("demo_len", list)
    )
    grouped_df['avg_token_length'] = grouped_df['demo_len_list'].apply(np.mean)
    print(grouped_df)
    df = grouped_df[['uid', 'avg_token_length']]
    return df


def main():
    """
    Main function to orchestrate data processing and plotting.
    """
    # --- Step 1: Calculate baseline correctness from Prefix-RFT ---
    prefix_rft_path = os.path.join(ROOT_DIR, Prefix_RFT_DIR)
    rft_path = os.path.join(ROOT_DIR, RFT_DIR)
    correctness_df = calculate_avg_correctness_across_steps(
        # prefix_rft_path,
        rft_path
    )
    if correctness_df.empty:
        print("Fatal Error: Could not calculate baseline correctness. Aborting.")
        return

    # --- Step 2: Prepare for plotting ---
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(14, 8))

    models_to_plot = {
        "Demonstrations": {"path": os.path.join(ROOT_DIR, SFT_DIR), "color": "black"},
        "RFT": {"path": os.path.join(ROOT_DIR, RFT_DIR), "color": "blue"},
        "SFT": {"path": os.path.join(ROOT_DIR, SFT_DIR), "color": "green"},
        "Prefix_RFT": {"path": prefix_rft_path, "color": "red"},
    }

    # --- Step 3: Process each model's length and plot against the baseline ---
    for model_name, details in models_to_plot.items():
        if model_name == "Demonstrations":
            length_df = load_demonstration_length(details['path'])
        else:
            length_df = calculate_model_token_length(details['path'])
        if length_df.empty:
            print(f"Skipping plot for {model_name} due to missing length data.")
            continue
        
        merged_df = pd.merge(correctness_df, length_df, on='uid', how='inner')
        if merged_df.empty:
            print(f"Warning: No common UIDs for {model_name}.")
            continue
            
        x_data = merged_df['avg_correctness']
        y_data = merged_df['avg_token_length']

        lowess = sm.nonparametric.lowess(y_data, x_data, frac=0.2)

        ax.scatter(x_data, y_data, label=f'{model_name} Data', color=details['color'], alpha=0.25, s=20)
        ax.plot(lowess[:, 0], lowess[:, 1], label=f'{model_name} Trend (LOWESS)', color=details['color'], linewidth=2.5)

    # --- Step 4: Finalize and save the plot ---
    ax.set_title('Model Generation Length vs. Prefix-RFT Correctness', fontsize=16, pad=20)
    ax.set_xlabel('Average Correctness per Prompt (Calculated from Prefix-RFT, steps 128-192)', fontsize=12)
    ax.set_ylabel(f'Average Generation Length (Tokens from "{TOKENIZER.name_or_path.split("/")[-1]}", step 256)', fontsize=12)
    ax.legend(title="Methods", fontsize=10)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax.set_xlim(-0.05, 1.05)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_filename = os.path.join(f"/data/_figs/token_length@{}_vs_prefix_rft_correctness.png")
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f"\nPlot successfully saved to: {output_filename}")
    plt.show()


if __name__ == "__main__":
    main()