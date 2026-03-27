import json
import os # 导入os库来获取CPU核心数
import torch
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from datasets import load_dataset, Dataset
from tqdm import tqdm
from functools import reduce
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import jieba

"""
这个脚本将完成以下任务：
1. 以高性能方式加载和合并三个模型的数据。
2. 使用 datasets.map 并行计算 TF-IDF 文本相似度。
3. 可视化相似度分布。
"""

# calculate_similarity_tfidf 函数保持不变，它已经是高效的
def calculate_similarity_tfidf(doc1, doc2):
    # 对于非常大的数据集，为避免重复加载模型，可以考虑将其作为全局变量或使用functools.lru_cache
    # 但在这里，每次调用都是独立的，所以保持原样
    doc1_segmented = " ".join(jieba.cut(doc1))
    doc2_segmented = " ".join(jieba.cut(doc2))
    vectorizer = TfidfVectorizer(ngram_range=(1, 2))
    corpus = [doc1_segmented, doc2_segmented]
    tfidf_matrix = vectorizer.fit_transform(corpus)
    similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
    return similarity


# ===================================================================
# --- 新的、将被.map()调用的函数 ---
# ===================================================================
def calculate_similarity_for_example(example):
    """
    处理一个数据样本（一行），计算所有需要的相似度。
    """
    # 从样本中获取rollout列表
    prefix_texts = example['analysis_gen_Prefix-RFT']
    sft_texts = example['analysis_gen_SFT']
    rft_texts = example['analysis_gen_RFT']
    
    # 合并成大文档
    prefix_doc = " ".join(prefix_texts)
    sft_doc = " ".join(sft_texts)
    rft_doc = " ".join(rft_texts)
    
    # 计算相似度
    sim_sft = calculate_similarity_tfidf(prefix_doc, sft_doc)
    sim_rft = calculate_similarity_tfidf(prefix_doc, rft_doc)
    
    # 以字典形式返回新列
    return {
        "similarity_prefix_vs_sft": sim_sft,
        "similarity_prefix_vs_rft": sim_rft
    }


if __name__ == "__main__":
    import argparse
    # ... (argparse 和路径设置部分保持不变) ...
    parser = argparse.ArgumentParser()
    parser.add_argument("--step", "-s", default=512, type=int)
    args = parser.parse_args()
    step = args.step
    
    base_dir = "/mnt/jfs2/cth/085b13//_analysis_v2"
    rft_tag = "rft_Qwen2.5-Math-1.5B_luffy_analysis_1p5b_16k_seed_42_sys_wo_format_num_empty"
    sft_tag = "sft_Qwen2.5-Math-1.5B_luffy_analysis_1p5b_16k_seed_42_sys_wo_format"
    gen_config = "gen_t1.0_top_p_0.95_max_tokens8192"
    
    rl_exp_name = f"{rft_tag}_8_clip_0.8_25-06-21-Jun-06-1750459811"
    prefix_rft_exp_name = f"{rft_tag}_7_clip_0.8_25-06-21-Jun-06-1750458724"
    sft_exp_name = f"{sft_tag}_lr5e-5_25-06-23-Jun-06-1750677874"

    exp_names = {
        "RFT": rl_exp_name,
        "Prefix-RFT": prefix_rft_exp_name,
        "SFT": sft_exp_name 
    }
    gen_fpaths = dict()
    for k, v in exp_names.items():
        gen_fpaths[k] = f"{base_dir}/{v}/global_step{step}/{gen_config}.json"
    
    # --- 1. 数据加载与预处理 (在 datasets 中完成) ---
    print("正在加载数据集...")
    datasets_raw = {k: load_dataset("json", data_files=f)['train'] for k, f in gen_fpaths.items()}

    # 为保证对齐，按uid排序所有数据集
    print("正在排序以对齐数据集...")
    for k in datasets_raw:
        datasets_raw[k] = datasets_raw[k].sort('uid')

    # 将所有数据集的关键列合并到一个主数据集中
    # 我们以 Prefix-RFT 为基础，添加其他数据集的列
    combined_ds = datasets_raw['Prefix-RFT']
    combined_ds = combined_ds.rename_column('analysis_gen', 'analysis_gen_Prefix-RFT')

    # 添加其他模型的文本和奖励列
    combined_ds = combined_ds.add_column('analysis_gen_SFT', datasets_raw['SFT']['analysis_gen'])
    combined_ds = combined_ds.add_column('analysis_gen_RFT', datasets_raw['RFT']['analysis_gen'])

    # 计算并添加RFT和SFT的奖励
    def add_rewards(ds, key):
        return ds.map(lambda ex: {f'avg_reward_{key}': np.mean(ex['analysis_corr'])})

    rft_rewards = add_rewards(datasets_raw['RFT'], 'RFT')['avg_reward_RFT']
    sft_rewards = add_rewards(datasets_raw['SFT'], 'SFT')['avg_reward_SFT']
    prefix_rewards = add_rewards(datasets_raw['Prefix-RFT'], 'Prefix-RFT')['avg_reward_Prefix-RFT']

    combined_ds = combined_ds.add_column('avg_reward_RFT', rft_rewards)
    combined_ds = combined_ds.add_column('avg_reward_SFT', sft_rewards)
    combined_ds = combined_ds.add_column('avg_reward_Prefix-RFT', prefix_rewards)

    # --- 2. 使用 .map() 并行计算相似度 ---
    # 设置使用的CPU核心数，可以根据您的机器调整
    num_cores = os.cpu_count() // 2 if os.cpu_count() > 1 else 1
    print(f"开始使用 {num_cores} 个核心并行计算相似度...")
    
    processed_ds = combined_ds.map(
        calculate_similarity_for_example,
        num_proc=num_cores
    )

    # --- 3. 转换为Pandas并进行可视化 (代码不变) ---
    print("计算完成，转换为Pandas DataFrame进行可视化...")
    df_merged = processed_ds.to_pandas()

    # --- 1. 计算相似度差值 ---
    df_merged['similarity_difference'] = df_merged['similarity_prefix_vs_rft'] - df_merged['similarity_prefix_vs_sft']

    print("添加了相似度差值的数据预览:")
    # print(df_merged[['uid', 'similarity_prefix_vs_rft', 'similarity_prefix_vs_sft', 'similarity_difference']].head())
    print(df_merged)

    # # 检查不同方法之间的overlap
    # reward_cols = ['avg_reward_RFT', 'avg_reward_SFT', 'avg_reward_Prefix-RFT']
    # df_rewards = df_merged[reward_cols]

    # # --- 1b. 计算相关性矩阵 ---
    # correlation_matrix = df_rewards.corr()

    # print("三个奖励模型之间的相关性矩阵:")
    # print(correlation_matrix)

    # # --- 1c. 使用热力图进行可视化 ---
    # plt.figure(figsize=(8, 6))
    # sns.heatmap(
    #     correlation_matrix,
    #     annot=True,         # 在格子里显示数字
    #     cmap='coolwarm',    # 使用冷暖色调（正相关为暖色，负相关为冷色）
    #     fmt=".2f",          # 将数字格式化为两位小数
    #     linewidths=.5
    # )
    # plt.title('Correlation Matrix of Average Rewards', fontsize=16)
    # # --- 保存并显示 ---
    # output_filename = f'/data/corr_reward_different_method.png'
    # print(f"正在保存图形到 {output_filename}...")
    # plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    # plt.show()
    # plt.show()

    # # --- 可视化差值的分布 ---

    # # 创建奖励区间并获取X轴顺序 (与之前相同)
    df_merged['reward_prefix_rft_level'] = pd.qcut(
        df_merged['avg_reward_Prefix-RFT'], q=10, duplicates='drop'
    )
    x_axis_order = df_merged['reward_prefix_rft_level'].cat.categories.astype(str).tolist()
    # 将 reward_rft_level 列也转换为字符串，以便绘图
    df_merged['reward_prefix_rft_level'] = df_merged['reward_prefix_rft_level'].astype(str)


    # --- 使用 catplot 绘制小提琴图 ---
    # 这次我们不再需要melt数据，也不需要col分面，因为只画一个指标
    g = sns.catplot(
        data=df_merged,
        x='reward_prefix_rft_level',   # X轴是奖励等级
        y='similarity_difference', # Y轴是我们的新差值列
        kind='violin',
        order=x_axis_order,     # 保证X轴有序
        palette='coolwarm',     # 使用冷暖色调，很适合展示正负差异
        inner='quartile',
        height=6,
        aspect=2                # 图可以更宽一些
    )


    # --- 在 y=0 的位置添加一条水平参考线 ---
    # 这条线是解读差值的关键
    g.map(plt.axhline, y=0, color='black', linestyle='--', linewidth=2)


    # --- 美化图形 ---
    g.fig.suptitle('Distribution of Similarity Difference\n(Sim[Prefix vs RFT] - Sim[Prefix vs SFT])', y=1.05, fontsize=16)
    g.set_axis_labels("RFT Reward Level", "Similarity Difference")
    g.set_xticklabels(rotation=15, ha='right')
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    # --- 保存并显示 ---
    output_filename = f'/data/similarity_difference_distribution_step{step}.png'
    print(f"正在保存图形到 {output_filename}...")
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    plt.show()




    # # 数据准备部分的代码完全不变
    # df_merged['reward_rft_level'] = pd.qcut(
    #     df_merged['avg_reward_RFT'], q=5, duplicates='drop'
    # )
    # df_melted = df_merged.melt(
    #     id_vars=['reward_rft_level'],
    #     value_vars=['similarity_prefix_vs_sft', 'similarity_prefix_vs_rft'],
    #     var_name='comparison_type',
    #     value_name='similarity_score'
    # )
    # df_melted['comparison_type'] = df_melted['comparison_type'].str.replace('similarity_', '').str.replace('_', ' ').str.title()
    # # 将 reward_rft_level 从 Interval 对象转换为字符串，让标签更美观
    # df_melted['reward_rft_level'] = df_melted['reward_rft_level'].astype(str)


    # # ===================================================================
    # # --- 使用 seaborn.catplot 绘图 (已交换 x 和 col) ---
    # # ===================================================================
    # g = sns.catplot(
    #     data=df_melted,
    #     x='reward_rft_level',   # <-- 新的 X 轴：奖励等级
    #     y='similarity_score',   # Y 轴仍然是相似度分数
    #     col='comparison_type',    # <-- 新的分面：对比的类型
    #     kind='violin',
    #     palette='magma',          # 换个色板
    #     inner='quartile',
    #     height=6,                 # 稍微调高一点
    #     aspect=1.5                # 调宽一点以容纳X轴标签
    # )

    # # --- 美化图形 ---
    # # 设置总标题
    # g.fig.suptitle('Similarity Distribution Across RFT Reward Intervals', y=1.04, fontsize=18)

    # # 设置每个子图的标题模板，现在 {col_name} 会是 'Prefix Vs Rft' 或 'Prefix Vs Sft'
    # g.set_titles("Comparison: {col_name}", size=14)

    # # 设置坐标轴标签
    # g.set_axis_labels("RFT Reward Level", "TF-IDF Cosine Similarity Score")

    # # 旋转X轴标签以防重叠
    # g.set_xticklabels(rotation=25, ha='right')

    # # 调整布局
    # plt.tight_layout(rect=[0, 0, 1, 0.95])

    # # 保存并显示
    # output_filename = f'/data/tfidf_similarity_distribution_by_comparison_step{step}.png'
    # print(f"正在保存图形到 {output_filename}...")
    # plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    # plt.show()

    # print("脚本执行完毕。")