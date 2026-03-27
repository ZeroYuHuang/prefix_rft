import json
import torch
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from datasets import load_dataset, Dataset
from tqdm import tqdm
from functools import reduce
"""
This script is to visulize that given a training step:
load the forward result and reward result of sft, rft, and prefix-rft and to visulize it 
"""
def high_entropy_token_loss(example):
    loss = torch.tensor(example["loss"])
    token_entropy = torch.tensor(example["token_entropy"])
    loss_mask = torch.tensor(example["loss_mask"])
    response_losses = torch.masked_select(loss, loss_mask.bool())
    response_length = token_entropy.shape[0]
    k = int(response_length * 0.1)
    k = max(1, k) 
    _, top_k_indices = torch.topk(token_entropy, k)
    top_k_losses = torch.gather(response_losses, 0, top_k_indices)
    example['high_entropy_loss'] = top_k_losses.mean()
    return example

def get_demo_len(example):
    example["demo_len"] = len(example["token_entropy"])
    return example

def process_forward_data(
    dataset, 
    dist_measure, 
    uid2demo_corr, 
    uid2demo_len, 
    uid2train_demo_len,
    dataset_key
):
    if distance_measure == "high_entropy_loss":
        dataset = dataset.map(function=high_entropy_token_loss, num_proc=32, desc="calculating high entropy token loss")
 
    dataset = dataset.map(function=get_demo_len, num_proc=32, desc="getting demonstraion length")
    dataset = dataset.filter(lambda x: x["demo_len"] < 8192, num_proc=32, desc="Removing imcomplete sequences")

    def get_correctness_and_istrain(example):
        assert example["demo_len"] - 1 in uid2demo_len[example["uid"]]
        for c, dl in zip(
            uid2demo_corr[example["uid"]],
            uid2demo_len[example["uid"]]
        ):
            if dl == example["demo_len"] - 1:
                example["correctness"] = c
                break
        example["istrain"] = uid2train_demo_len[example["uid"]][0] == (example["demo_len"] - 1)
        return example

    dataset = dataset.map(function=get_correctness_and_istrain, num_proc=32, desc="Get correctness and istrain label")
    # dataset = dataset.filter(lambda x: x["istrain"], num_proc=32, desc="Only use training example")
    print(f"After processing: {dataset}")
    columns_to_remove = [
        col for col in ['token_log_p', 'token_entropy', 'perplexity', 'loss', 'loss_mask', 'final_loss']
        if col != dist_measure
    ]
    dataset = dataset.remove_columns(columns_to_remove)
    df = dataset.to_pandas()
    grouped_df = df.groupby("uid", as_index=False).agg(
        distance_measure=(dist_measure, list),
        demo_len_list=("demo_len", list)
    )
    grouped_df[f'distance_{dataset_key}'] = grouped_df['distance_measure'].apply(np.mean)
    grouped_df['demo_len'] = grouped_df['demo_len_list'].apply(np.mean)
    grouped_df.drop(columns=['distance_measure'], inplace=True)
    return grouped_df

def process_reward_data(dataset, dataset_key):
    def avgatn(example):
        example[f"avg_reward_{dataset_key}"] = np.mean(example["analysis_corr"])
        return example
    dataset = dataset.map(function=avgatn, num_proc=16)
    dataset = dataset.remove_columns(['data_source', 'prompt', 'ability', 'reward_model', 'extra_info', 'problem', 'analysis_gen', "analysis_corr"])
    df = dataset.to_pandas()
    return df

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--step", "-s", default=512, type=int)
    args = parser.parse_args()

    original_dataset = load_dataset(
        "parquet", 
        data_files="/mnt/jfs2/cth/085b13/_data/processed_dataset_new/luffy_analysis_1p5b_16k_seed_42_sys_wo_format/valid.parquet"
    )['train']
    uid2demo_corr = dict()
    uid2demo_len = dict()
    uid2train_demo_len = dict()
    for i in range(len(original_dataset)):
        d = original_dataset[i]
        uid2demo_corr[d["uid"]] = d["old_correctness"]
        uid2demo_len[d["uid"]] = d["old_demos_len"]
        uid2train_demo_len[d["uid"]] = d["demos_len"]
    # this script is to given a training step
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
    # distance_measure = "final_loss"
    distance_measure = "high_entropy_loss"
    
    forward_fpaths = dict()
    for k, v in exp_names.items():
        forward_fpaths[k] = f"{base_dir}/{v}/global_step{step}/forward.json"
    
    gen_fpaths = dict()
    for k, v in exp_names.items():
        gen_fpaths[k] = f"{base_dir}/{v}/global_step{step}/{gen_config}.json"

    dfs = []
    for k in exp_names.keys():
        df1 = process_forward_data(
            load_dataset("json", data_files=forward_fpaths[k])['train'],
            dist_measure=distance_measure,
            uid2demo_corr=uid2demo_corr, 
            uid2demo_len=uid2demo_len, 
            uid2train_demo_len=uid2train_demo_len,
            dataset_key=k
        )
        try:
            df2 = process_reward_data(load_dataset("json", data_files=gen_fpaths[k])['train'], dataset_key=k)
            df = pd.merge(df1, df2, on="uid", how="inner")
        except:
            df = df1
        print(df)
        dfs.append(df)
    merged_df = reduce(lambda left, right: pd.merge(left, right, on='uid', how='inner'), dfs)
    # 创建一个副本以进行修改
    cleaned_df = merged_df.copy()

    # 存储有差异的列名
    conflicting_cols = []

    # 遍历所有列，查找以 '_x' 结尾的列
    for col in cleaned_df.columns:
        if col.endswith('_x'):
            # 构建对应的 '_y' 列名
            base_name = col[:-2]
            col_x = col
            col_y = base_name + '_y'

            # 确保 '_y' 列也存在
            if col_y in cleaned_df.columns:
                print(f"--- 正在处理列: {base_name} ---")

                # 验证两列是否完全相同
                # .equals() 方法可以正确处理 NaN 值 (NaN 在不同位置也会导致 False)
                if cleaned_df[col_x].equals(cleaned_df[col_y]):
                    print(f"✅ 列 '{col_x}' 和 '{col_y}' 完全相同。正在清理...")
                    # 删除多余的 _y 列
                    cleaned_df.drop(columns=[col_y], inplace=True)
                    # 重命名 _x 列为原始名称
                    cleaned_df.rename(columns={col_x: base_name}, inplace=True)
                else:
                    print(f"⚠️ 列 '{col_x}' 和 '{col_y}' 存在差异。")
                    # 如果内容不同，我们将它们都保留，并记录下来以便后续手动处理
                    conflicting_cols.append(base_name)

    print("\n--- 清理完成 ---")
    print("清理后的 DataFrame：")
    print(cleaned_df)
    print(cleaned_df.columns)
    df = cleaned_df

    """
    # 假设 df 是您已经清理好列名的 DataFrame
    # （接续上一个回答中的步骤）

    # 1. 选取需要保留的标识列 (ID variables) 和需要转换的值列 (value variables)
    # --- 为 RFT 创建子集 ---
    df_rft = df[['distance_RFT', 'avg_reward_RFT']].copy()
    df_rft.rename(columns={
        'distance_RFT': 'distance',
        'avg_reward_RFT': 'avg_reward'
    }, inplace=True)
    df_rft['method_type'] = 'RFT'

    # --- 为 SFT 创建子集 ---
    df_sft = df[['distance_SFT', 'avg_reward_RFT']].copy()
    df_sft.rename(columns={
        'distance_SFT': 'distance',
        'avg_reward_RFT': 'avg_reward'
    }, inplace=True)
    df_sft['method_type'] = 'SFT'

    # --- 为 Prefix-RFT 创建子集 ---
    df_prefix = df[['distance_Prefix-RFT', 'avg_reward_RFT']].copy()
    df_prefix.rename(columns={
        'distance_Prefix-RFT': 'distance',
        # 'avg_reward_Prefix-RFT': 'avg_reward'，
        'avg_reward_RFT': 'avg_reward'
    }, inplace=True)
    df_prefix['method_type'] = 'Prefix-RFT'


    # --- 使用 pd.concat 将三个子集合并 ---
    df_long_paired = pd.concat([
        df_rft, 
        df_sft, 
        df_prefix
        ], ignore_index=True)

    print("重塑后的成对长格式数据 (头部):")
    print(df_long_paired.head())
    print("\n重塑后的成对长格式数据 (尾部):")
    print(df_long_paired.tail())
    # ===================================================================
    # --- 第0步：离群点处理 (使用IQR方法) ---
    # ===================================================================
    print(f"处理前的数据量: {df_long_paired.shape[0]} 行")
    Q1_dist = df_long_paired.groupby('method_type')['distance'].transform('quantile', 0.25)
    Q3_dist = df_long_paired.groupby('method_type')['distance'].transform('quantile', 0.75)
    IQR_dist = Q3_dist - Q1_dist
    lower_bound_dist = Q1_dist - 1.5 * IQR_dist
    upper_bound_dist = Q3_dist + 1.5 * IQR_dist

    Q1_reward = df_long_paired.groupby('method_type')['avg_reward'].transform('quantile', 0.25)
    Q3_reward = df_long_paired.groupby('method_type')['avg_reward'].transform('quantile', 0.75)
    IQR_reward = Q3_reward - Q1_reward
    lower_bound_reward = Q1_reward - 1.5 * IQR_reward
    upper_bound_reward = Q3_reward + 1.5 * IQR_reward

    is_inlier = (
        (df_long_paired['distance'] >= lower_bound_dist) &
        (df_long_paired['distance'] <= upper_bound_dist) &
        (df_long_paired['avg_reward'] >= lower_bound_reward) &
        (df_long_paired['avg_reward'] <= upper_bound_reward)
    )
    # 我们将使用这个 df_filtered 进行绘图
    df_filtered = df_long_paired[is_inlier].copy()
    print(f"处理后的数据量: {df_filtered.shape[0]} 行")
    df_long_paired = df_filtered

    # --- 1. 准备工作：获取颜色方案 ---
    # 创建一个调色板，确保KDE图和趋势线的颜色可以一一对应
    # 我们有三种方法，所以需要三种颜色
    palette = sns.color_palette(n_colors=3)

    # --- 2. 开始绘图 ---
    plt.figure(figsize=(12, 8)) # 稍微调大图形尺寸以容纳更多信息
    ax = plt.gca() # 获取当前的坐标轴，以便所有图形都画在上面

    # --- 3. 首先，绘制KDE背景图 ---
    # 这步和之前一样，它会建立基本的图层和图例
    sns.kdeplot(
        data=df_long_paired,
        x='distance',
        y='avg_reward',
        hue='method_type',
        palette=palette, # 应用我们的调色板
        fill=True,                  # 填充密度曲线下的区域
        alpha=0.5,                  # 为填充区域设置透明度
        linewidth=0.25,
        common_norm=False,
        ax=ax
    )

    # --- 4. 循环添加趋势线 ---
    # 遍历每一种方法类型和其对应的颜色
    method_types = df_long_paired['method_type'].unique()
    for i, method in enumerate(method_types):
        # 筛选出当前方法的数据子集
        subset = df_long_paired[df_long_paired['method_type'] == method]

        # 在该子集上绘制回归趋势线
        sns.regplot(
            data=subset,
            x='distance',
            y='avg_reward',
            scatter=False,      # 关键：不绘制散点
            ax=ax,              # 确保画在同一个坐标轴上
            color=palette[i],    # 使用与KDE图相同的颜色
            lowess=True
        )

    # --- 5. 美化图形 ---
    plt.title('KDE Plot with Regression Trend Lines', fontsize=16)
    plt.xlabel('Distance Value', fontsize=12)
    plt.ylabel('Average Reward Value', fontsize=12)
    # 图例已经由kdeplot自动生成，我们不需要额外操作
    plt.grid(True, linestyle='--', alpha=0.6)
    """
    # 使用 np.where 根据条件创建一个新列
    df['sft_reward_group'] = np.where(
        df['avg_reward_RFT'] > 0.5,
        'High SFT Reward (> 0.5)',
        'Low SFT Reward (<= 0.5)'
    )

    # --- 1b. 重新整理成长格式，并带上我们的新分组标签 ---
    # 注意：这次我们从 df 中提取列时，要包含 'sft_reward_group'
    df_rft = df[['sft_reward_group', 'distance_RFT', 'avg_reward_RFT']].rename(columns={'distance_RFT': 'distance', 'avg_reward_RFT': 'avg_reward'}).assign(method_type='RFT')
    df_sft = df[['sft_reward_group', 'distance_SFT', 'avg_reward_RFT']].rename(columns={'distance_SFT': 'distance', 'avg_reward_RFT': 'avg_reward'}).assign(method_type='SFT')
    df_prefix = df[['sft_reward_group', 'distance_Prefix-RFT', 'avg_reward_RFT']].rename(columns={'distance_Prefix-RFT': 'distance', 'avg_reward_RFT': 'avg_reward'}).assign(method_type='Prefix-RFT')

    # 将它们合并成一个最终用于绘图的 DataFrame
    df_final_simple = pd.concat([df_rft, df_sft, df_prefix], ignore_index=True)

    print("为FOR循环准备好的数据预览:")
    print(df_final_simple.head())

    # 定义我们要循环的组
    groups = ['Low SFT Reward (<= 0.5)', 'High SFT Reward (> 0.5)']
    # 定义颜色
    palette = sns.color_palette(n_colors=3)

    # --- 2a. 创建一个包含两个子图的画布 ---
    # figsize设置整个画布的大小，sharey=True让两个子图共享Y轴，便于比较
    fig, axes = plt.subplots(1, 2, figsize=(18, 8), sharey=True)

    # --- 2b. 开始FOR循环 ---
    for i, group_name in enumerate(groups):
        # 选择当前子图
        ax = axes[i]
        print(f"正在绘制子图: {group_name}")

        # 筛选出当前组的数据
        data_subset = df_final_simple[df_final_simple['sft_reward_group'] == group_name]

        # 在当前子图(ax)上绘制KDE图
        # 使用 hue='method_type' 会自动为三种方法应用不同颜色
        sns.kdeplot(
            data=data_subset,
            x='distance',
            y='avg_reward',
            hue='method_type',
            palette=palette,
            fill=True,
            linewidths=0,
            alpha=0.6,
            common_norm=False,
            ax=ax,
            legend=(i==0) # 只在第一个子图上显示图例
        )

        # 在当前子图(ax)上绘制LOESS趋势线
        # 我们需要对每种方法单独调用regplot
        for j, method in enumerate(data_subset['method_type'].unique()):
            method_data = data_subset[data_subset['method_type'] == method]
            sns.regplot(
                data=method_data,
                x='distance',
                y='avg_reward',
                scatter=False,
                lowess=True,
                ax=ax,
                color=palette[j]
            )

        # 设置每个子图的标题和网格
        ax.set_title(group_name, fontsize=14)
        ax.grid(True, linestyle='--', alpha=0.6)

    # --- 2c. 设置整个图形的标题和标签 ---
    fig.suptitle('Comparison of Plots by SFT Reward Level', fontsize=18)
    # 由于共享Y轴，我们只需要设置一次Y轴标签
    axes[0].set_ylabel('Average Reward Value', fontsize=12)
    axes[0].set_xlabel('Distance Value', fontsize=12)
    axes[1].set_xlabel('Distance Value', fontsize=12)


    # 调整布局，防止重叠
    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # 为总标题留出空间

    # 显示图形
    plt.show()
    fig_name = "test.png"
    print(f"Saving to /data/figures/dp/{fig_name}")
    plt.savefig(f'/data/_figures/dp/test.png')

    


