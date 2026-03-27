import json
import torch
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from datasets import load_dataset, Dataset
from tqdm import tqdm
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
    k = int(response_length * 0.2)
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
    uid2train_demo_len
):
    def high_entropy_token_loss(example):
        loss = torch.tensor(example["loss"])
        token_entropy = torch.tensor(example["token_entropy"])
        loss_mask = torch.tensor(example["loss_mask"])
        response_losses = torch.masked_select(loss, loss_mask.bool())
        response_length = token_entropy.shape[0]
        k = int(response_length * 0.2)
        k = max(1, k) 
        _, top_k_indices = torch.topk(token_entropy, k)
        top_k_losses = torch.gather(response_losses, 0, top_k_indices)
        example['high_entropy_loss'] = top_k_losses.mean()
        return example

    if distance_measure == "high_entropy_loss":
        dataset = dataset.map(function=high_entropy_token_loss, num_proc=32)
 
    dataset = dataset.map(function=get_demo_len, num_proc=32)
    dataset = dataset.filter(lambda x: x["demo_len"] < 8192, num_proc=32)

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

    dataset = dataset.map(function=get_correctness_and_istrain, num_proc=32)
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
    grouped_df['distance'] = grouped_df['distance_measure'].apply(np.mean)
    grouped_df['demo_len'] = grouped_df['demo_len_list'].apply(np.mean)
    return grouped_df

def process_reward_data(dataset):
    dataset = dataset.remove_columns(['data_source', 'prompt', 'ability', 'reward_model', 'extra_info', 'problem', 'analysis_gen'])
    def avgatn(example):
        example["performance"] = np.mean(example["analysis_corr"])
        return example
    
    # def transform_difficulty(example):
    #     level = None
    #     diff = example["difficulty"]
    #     if diff in [1,2]:
    #         level = "hard"
    #     if diff in [3,4,5,6]:
    #         level = "medium"
    #     if diff in [7,8]:
    #         level = "easy"
    #     example["difficulty"] = level
    #     return example
    dataset = dataset.map(function=avgatn, num_proc=16)
    # dataset = dataset.map(function=transform_difficulty, num_proc=16)
    print(set(dataset["difficulty"]))
    df = dataset.to_pandas()
    return df

if __name__ == "__main__":
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
    step = 128

    base_dir = "/mnt/jfs2/cth/085b13//_analysis"
    rft_tag = "rft_Qwen2.5-Math-1.5B_luffy_analysis_1p5b_16k_seed_42_sys_wo_format_num_empty"
    sft_tag = "sft_Qwen2.5-Math-1.5B_luffy_analysis_1p5b_16k_seed_42_sys_wo_format"
    gen_config = "gen_t0.6_top_p_0.95_max_tokens8192_4k"
    
    rl_exp_name = f"{rft_tag}_8_clip_0.8_25-06-21-Jun-06-1750459811"
    prefix_rft_exp_name = f"{rft_tag}_7_clip_0.8_25-06-21-Jun-06-1750458724"
    sft_exp_name = f"{sft_tag}_lr5e-5_25-06-19-Jun-06-1750280661"

    exp_names = {
        "RFT": rl_exp_name,
        "Prefix-RFT": prefix_rft_exp_name,
        "SFT": sft_exp_name 
    }
    distance_measure = "final_loss"
    # distance_measure = "high_entropy_loss"
    
    forward_fpaths = dict()
    for k, v in exp_names.items():
        forward_fpaths[k] = f"{base_dir}/{v}/global_step{step}/forward.json"
    
    gen_fpaths = dict()
    for k, v in exp_names.items():
        gen_fpaths[k] = f"{base_dir}/{v}/global_step{step}/{gen_config}.json"

    dfs = dict()
    for k in exp_names.keys():
        df1 = process_forward_data(
            load_dataset("json", data_files=forward_fpaths[k])['train'],
            dist_measure=distance_measure,
            uid2demo_corr=uid2corr, uid2demo_len=uid2demo_len, uid2train_demo_len=uid2train_demo_len
        )
        df2 = process_reward_data(load_dataset("json", data_files=gen_fpaths[k])['train'])
        df = pd.merge(df1, df2, on="uid", how="inner")
        dfs[k] = df

    print(dfs["Prefix-RFT"])
    difficulty_palette = {
        'hard': 'navy',
        'medium': 'royalblue',
        'easy': 'lightskyblue'
    }

    # 3. 创建图表
    #    设置图表的大小
    plt.figure(figsize=(12, 8))

    # 4. 绘制基础散点图
    #    ax对象用于后续在同一图上绘图
    # ax = sns.scatterplot(
    #     data=combined_df,
    #     x='distance',
    #     y='performance',
    #     hue='difficulty',
    #     style='source',
    #     palette=difficulty_palette,
    #     s=80
    # )
    ax = sns.kdeplot(
        data=combined_df,
        x='distance',
        y='performance',
        hue='source',           # 根据'difficulty'对数据进行分组和着色
        # hue='difficulty',
        # palette=difficulty_palette, # 使用我们定义的调色板
        fill=True,                  # 填充密度曲线下的区域
        alpha=0.5,                  # 为填充区域设置透明度
        linewidth=0.5
    )

    # 5. 循环遍历每个难度级别，并叠加回归趋势线
    # for diff_level in combined_df['difficulty'].unique():
    #     sns.regplot(
    #         data=dfs[2][dfs[2]['difficulty'] == diff_level],
    #         x='distance',
    #         y='performance',
    #         color=difficulty_palette[diff_level], # 确保趋势线颜色与散点颜色匹配
    #         scatter=False,                        # 不重复绘制散点
    #         ax=ax                                 # 在同一个ax对象上绘图
    #     )

    # 6. 美化图表，添加标题和标签
    # plt.title('Scatter Plot with Trend Lines by Difficulty and Source')
    plt.xlabel('SFT Loss')
    plt.ylabel('Avg@n')
    plt.grid(True)
    # # 使用seaborn的kdeplot函数
    # kde_plot = sns.kdeplot(
    #     data=combined_df,
    #     x='distance',
    #     y='performance',
    #     #hue='source',           # 根据'difficulty'对数据进行分组和着色
    #     # hue='difficulty',
    #     # palette=difficulty_palette, # 使用我们定义的调色板
    #     fill=True,                  # 填充密度曲线下的区域
    #     alpha=0.5,                  # 为填充区域设置透明度
    #     linewidth=0.5
    # )

    # 将图例移到图表外部
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)

    plt.tight_layout() # 调整布局

    # 保存图像
    fig_name = f"perf_vs_{distance_measure}_step{step}.png"
    print(f"Saving to /data/figures/dp/{fig_name}")
    plt.savefig(f'/data/_figures/dp/{fig_name}')

    


