import argparse
import os
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datasets import load_dataset


def analyze_convergence(gen_fpaths, steps, num_hardest=20, output_dir="."):
    """
    分析最难问题的收敛性。

    Args:
        gen_fpaths (list): 包含各个步骤结果的JSON文件路径列表。
        steps (list): 对应的训练步骤列表。
        num_hardest (int): 要分析的最难问题的数量。
        output_dir (str): 保存图表的输出目录。
    """
    # 步骤1: 加载并处理所有数据集，计算每个问题在每个步骤的平均正确率
    # 结构: {uid: {step: accuracy}}
    question_performance = defaultdict(dict)

    print("Loading and processing datasets...")
    for step, fpath in zip(steps, gen_fpaths):
        if not os.path.exists(fpath):
            print(f"Warning: File not found for step {step} at {fpath}. Skipping.")
            continue
            
        # 加载数据集，注意每个json是一个DatasetDict，我们取'train'部分
        dataset = load_dataset("json", data_files=fpath)['train']
        
        for record in dataset:
            uid = record.get('uid')
            # 确保 'analysis_corr' 存在且不为空
            corr_list = record.get('analysis_corr')
            
            if uid is None or not corr_list:
                continue

            # 计算平均正确率
            accuracy = np.mean(corr_list)
            question_performance[uid][step] = accuracy

    if not question_performance:
        print("Error: No data was loaded. Please check file paths and content.")
        return

    # 步骤2: 将数据转换为Pandas DataFrame以便于分析
    df = pd.DataFrame.from_dict(question_performance, orient='index')
    df = df.reindex(columns=steps) # 确保列的顺序与steps一致

    # 步骤3: 识别最难的问题
    # 计算每个问题在所有步骤的平均正确率作为“难度”指标
    df['mean_accuracy'] = df.mean(axis=1)
    
    # 按平均正确率升序排序，找到最难的N个问题
    hardest_questions_df = df.nsmallest(num_hardest, 'mean_accuracy')

    # 步骤4: 分析与报告收敛性
    # 计算最后两个步骤的性能变化
    last_step = steps[-1]
    previous_step = steps[-2]
    if last_step in hardest_questions_df.columns and previous_step in hardest_questions_df.columns:
        hardest_questions_df['convergence_diff'] = hardest_questions_df[last_step] - hardest_questions_df[previous_step]
    else:
        print(f"Warning: Columns for steps {previous_step} or {last_step} not found. Cannot calculate convergence diff.")
        hardest_questions_df['convergence_diff'] = 'N/A'


    print("\n" + "="*50)
    print(f"Top {num_hardest} Hardest Questions Analysis")
    print("="*50)
    print("These questions have the lowest average accuracy across all measured steps.")
    print("A small 'convergence_diff' (close to 0) suggests performance has stabilized.")
    
    # 打印结果表格，只显示相关列
    display_columns = steps + ['mean_accuracy', 'convergence_diff']
    print(hardest_questions_df[display_columns].to_string(float_format="%.4f"))


    # 步骤5: 可视化结果
    print(f"\nGenerating convergence plot for the {num_hardest} hardest questions...")
    
    # 获取用于绘图的数据
    plot_df = hardest_questions_df[steps].T # 转置DataFrame，方便绘图
    
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(12, 8))

    plot_df.plot(kind='line', marker='o', ax=ax, alpha=0.8)

    # 计算并绘制这些难题的平均性能曲线
    average_hardest_perf = hardest_questions_df[steps].mean()
    average_hardest_perf.plot(
        kind='line', 
        marker='s', 
        linewidth=4, 
        color='black', 
        label='Average of Hardest', 
        ax=ax
    )

    ax.set_title(f'Performance of Top {num_hardest} Hardest Questions Over Training Steps')
    ax.set_xlabel('Training Step')
    ax.set_ylabel('Average Correctness')
    ax.set_xticks(steps)
    ax.legend(title='Question UID', bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True)
    
    plt.tight_layout()
    plot_filename = os.path.join(output_dir, f'/data/hardest_{num_hardest}_questions_convergence.png')
    plt.savefig(plot_filename, dpi=300)
    print(f"Plot saved to: {plot_filename}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Analyze convergence of the hardest questions in an RL training run."
    )
    # 删除了不再需要的 --step 参数，因为我们处理的是一个步骤列表
    parser.add_argument(
        "--num_hardest", "-n", 
        default=300, 
        type=int,
        help="Number of hardest questions to analyze."
    )
    parser.add_argument(
        "--output_dir", "-o",
        default=".",
        type=str,
        help="Directory to save the output plot."
    )
    args = parser.parse_args()
    
    steps = [512, 544, 576, 608, 640]
    
    # 你的路径设置保持不变
    base_dir = "/mnt/jfs2/cth/085b13//_analysis_v2" 
    rft_tag = "rft_Qwen2.5-Math-1.5B_luffy_analysis_1p5b_16k_seed_42_sys_wo_format_num_empty" 
    gen_config = "gen_t1.0_top_p_0.95_max_tokens8192" 
    rl_exp_name = f"{rft_tag}_8_clip_0.8_25-06-21-Jun-06-1750459811" 
    
    gen_fpaths = [] 
    for step in steps: 
        gen_fpaths.append(f"{base_dir}/{rl_exp_name}/global_step{step}/{gen_config}.json") 

    # 确保输出目录存在
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 调用分析函数
    analyze_convergence(gen_fpaths, steps, args.num_hardest, args.output_dir)