import pandas as pd
import os
import numpy as np
from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import time # 引入 time 模块来对比性能

# --- Constants (保持不变) ---
ROOT_DIR = "/mnt/jfs2/cth/085b13/_analysis_v2"
RFT_DIR = "rft_Qwen2.5-Math-1.5B_luffy_analysis_1p5b_16k_seed_42_sys_wo_format_num_empty_8_clip_0.8_25-06-21-Jun-06-1750459811"
SFT_DIR = "sft_Qwen2.5-Math-1.5B_luffy_analysis_1p5b_16k_seed_42_sys_wo_format_lr5e-5_25-06-23-Jun-06-1750677874"
Prefix_RFT_DIR = "rft_Qwen2.5-Math-1.5B_luffy_analysis_1p5b_16k_seed_42_sys_wo_format_num_empty_7_clip_0.8_25-06-21-Jun-06-1750458724"
GEN_FILE = "gen_t1.0_top_p_0.95_max_tokens8192.json"

OUTPUT_DIR = "./figs/"
CACHE_DIR = "./.cache/results_cache" # 为缓存结果创建一个目录

# --- 优化后的函数 ---

def calculate_prefix_rft_correctness(model_path: str) -> pd.DataFrame:
    """
    (Optimized) Calculates average correctness using vectorized pandas operations.
    """
    # 结果缓存逻辑
    model_name = os.path.basename(model_path)
    cache_file = os.path.join(CACHE_DIR, f"correctness_{model_name}.parquet")
    if os.path.exists(cache_file):
        print(f"Loading cached correctness for {model_name}...")
        return pd.read_parquet(cache_file)

    correctness_steps = [128, 160, 192, 224, 256]
    json_paths = [os.path.join(model_path, f"global_step{step}", GEN_FILE) for step in correctness_steps]
    existing_paths = [p for p in json_paths if os.path.exists(p)]

    if not existing_paths:
        print(f"Error: No correctness files found in {model_path}.")
        return pd.DataFrame()

    print(f"\nCalculating baseline correctness for {model_name} (Optimized)...")
    dataset = load_dataset(
        'json', data_files=existing_paths, split='train', 
        # cache_dir=f'./.cache/prefix_rft_corr_{model_name}'
    )
    df = dataset.to_pandas()

    # --- 核心优化点: 使用 explode 代替循环 ---
    # 1. 只选择需要的列
    df = df[['uid', 'analysis_corr']]
    # 2. "爆炸"列表列，将每个分数放到单独的行
    exploded_df = df.explode('analysis_corr').dropna()
    # 3. 确保数据类型正确
    exploded_df['analysis_corr'] = exploded_df['analysis_corr'].astype(float)
    # 4. 直接在整个DataFrame上进行groupby和mean计算，效率极高
    results_df = exploded_df.groupby('uid')['analysis_corr'].mean().reset_index()
    results_df = results_df.rename(columns={'analysis_corr': 'prerft_correctness'})
    
    # 保存结果到缓存
    os.makedirs(CACHE_DIR, exist_ok=True)
    results_df.to_parquet(cache_file)
    print(f"Saved correctness cache to {cache_file}")

    return results_df


def calculate_similarity(model_path: str, uid2demo: dict) -> pd.DataFrame:
    """
    (Optimized) Calculates similarity using vectorized pandas and parallelized TF-IDF.
    """
    sim_step = 256
    model_name = os.path.basename(model_path)
    
    # 结果缓存逻辑
    cache_file = os.path.join(CACHE_DIR, f"similarity_{model_name}_step{sim_step}.parquet")
    if os.path.exists(cache_file):
        print(f"Loading cached similarity for {model_name}...")
        return pd.read_parquet(cache_file)

    gen_path = os.path.join(model_path, f"global_step{sim_step}", GEN_FILE)
    if not os.path.exists(gen_path):
        print(f"Error: Generation file not found at {gen_path}")
        return pd.DataFrame()

    print(f"\nCalculating TF-IDF similarity for {model_name} (Optimized)...")
    dataset = load_dataset('json', data_files=gen_path, split='train', cache_dir=f'./.cache/{model_name}_sim_{sim_step}')
    gen_df = dataset.to_pandas()
    
    gen_text_column = 'analysis_gen'
    if gen_text_column not in gen_df.columns:
        raise KeyError(f"Column '{gen_text_column}' not found in {gen_path}.")

    # --- 核心优化点: 使用 explode 和 map 代替 iterrows ---
    print("  Vectorizing and preparing texts...")
    # 1. 选择需要的列
    df = gen_df[['uid', gen_text_column]]
    # 2. "爆炸"列表，每个generation一行
    exploded_df = df.explode(gen_text_column).dropna()
    exploded_df = exploded_df.rename(columns={gen_text_column: 'generation'})
    # 3. 使用高效的 .map() 方法添加demonstration列
    exploded_df['demonstration'] = exploded_df['uid'].map(uid2demo)
    # 4. 删除没有对应demonstration的行
    exploded_df = exploded_df.dropna(subset=['demonstration'])

    if exploded_df.empty:
        print("Warning: No matching data found to calculate similarity.")
        return pd.DataFrame()

    print(f"  Calculating TF-IDF on {len(exploded_df)} text pairs...")
    # --- 优化点: 利用多核CPU (n_jobs=-1 使用所有可用核心) ---
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words='english')
    
    # 在所有文本上拟合
    vectorizer.fit(pd.concat([exploded_df['generation'], exploded_df['demonstration']]))
    
    # 转换
    gen_vectors = vectorizer.transform(exploded_df['generation'])
    demo_vectors = vectorizer.transform(exploded_df['demonstration'])
    
    print("  Calculating cosine similarities...")
    individual_similarities = np.diag(cosine_similarity(gen_vectors, demo_vectors))
    exploded_df['similarity'] = individual_similarities
    
    print("  Aggregating results...")
    # 5. 高效地分组聚合
    agg_df = exploded_df.groupby('uid')['similarity'].agg(['mean', 'var']).reset_index()
    agg_df.rename(columns={'mean': 'similarity_mean', 'var': 'similarity_variance'}, inplace=True)
    agg_df['similarity_variance'].fillna(0, inplace=True)
    
    # 保存结果到缓存
    os.makedirs(CACHE_DIR, exist_ok=True)
    agg_df.to_parquet(cache_file)
    print(f"Saved similarity cache to {cache_file}")

    return agg_df

# `load_demonstrations` 函数无需改动，它已经足够快了
def load_demonstrations() -> (pd.DataFrame, dict):
    # ... (代码与之前版本相同)
    demos = []
    uid2demo = dict()
    data_files = "/mnt/jfs2/cth/085b13/_data/processed_dataset_new/luffy_analysis_1p5b_16k_seed_42_sys_wo_format/valid_2k.parquet"
    if not os.path.exists(data_files):
        raise FileNotFoundError(f"Demonstration file not found: {data_files}")
        
    print("Loading demonstrations...")
    valid_dataset = load_dataset("parquet", data_files=data_files)['train']
    for dp in valid_dataset: # 这个循环数据量不大，速度很快，无需修改
        demo_text = dp["demos"][0] if dp["demos"] else ""
        uid2demo[dp["uid"]] = demo_text
        
    print(f"Loaded {len(uid2demo)} demonstrations.")
    return pd.DataFrame(demos), uid2demo

# --- 主程序逻辑 (基本不变) ---
if __name__ == "__main__":
    start_time = time.time()
    
    # Step 1 & 2
    demos_df, uid2demo = load_demonstrations()
    rft_path = os.path.join(ROOT_DIR, RFT_DIR)
    correctness_df = calculate_prefix_rft_correctness(rft_path)
    
    if correctness_df.empty:
        print("Fatal Error: Could not calculate baseline correctness. Aborting.")
        exit()

    # Step 3
    prefix_rft_path = os.path.join(ROOT_DIR, Prefix_RFT_DIR)
    sft_path = os.path.join(ROOT_DIR, SFT_DIR)
    
    models_to_plot = {
        "RFT": {"path": rft_path, "color": "blue"},
        "SFT": {"path": sft_path, "color": "green"},
        "Prefix_RFT": {"path": prefix_rft_path, "color": "red"},
    }

    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(12, 8))

    # Step 4
    for model_name, details in models_to_plot.items():
        similarity_df = calculate_similarity(details["path"], uid2demo)
        
        if similarity_df.empty:
            print(f"Skipping {model_name} due to missing similarity data.")
            continue

        merged_df = pd.merge(correctness_df, similarity_df, on='uid')
        
        if merged_df.empty:
            print(f"Warning: No common UIDs for {model_name}.")
            continue

        print(f"Plotting for {model_name} ({len(merged_df)} data points)...")
        sns.regplot(
            ax=ax, x='prerft_correctness', y='similarity_mean',
            data=merged_df, color=details['color'], label=f'{model_name} (Trend)',
            scatter_kws={'alpha': 0.4, 's': 20}, line_kws={'linewidth': 2}
        )
        avg_variance = merged_df['similarity_variance'].mean()
        print(f"  - Average similarity variance for {model_name}: {avg_variance:.4f}")

    # Step 5
    ax.set_title("Mean TF-IDF Similarity to Demonstration vs. PreRFT Correctness", fontsize=16)
    ax.set_xlabel("PreRFT Correctness", fontsize=12)
    ax.set_ylabel("Mean TF-IDF Similarity with Demonstration", fontsize=12)
    ax.legend(title="Model", fontsize=10)
    plt.tight_layout()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = os.path.join(OUTPUT_DIR, "mean_similarity_vs_prerft_correctness.png")
    plt.savefig(output_path, dpi=300)
    
    end_time = time.time()
    print(f"\nTotal execution time: {end_time - start_time:.2f} seconds.")
    print(f"Plot saved successfully to {output_path}")
    plt.show()