import os
import re
import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Any

# 引入绘图和科学计算库
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba

# ! 依赖项: SciPy (可选, 但推荐)
try:
    # 尝试从 SciPy 导入样条插值函数
    from scipy.interpolate import make_interp_spline
    SCIPY_AVAILABLE = True
except ImportError:
    # 如果导入失败，则打印警告并设置标志位
    print("警告: 未找到 scipy 库。趋势线将使用直线代替平滑曲线。")
    print("为了获得最佳绘图效果，请在您的环境中运行: pip install scipy")
    SCIPY_AVAILABLE = False

# --- 1. 全局绘图设置 ---
# 设置全局字体为 Times New Roman
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']


def calculate_mean_loss(file_path: str) -> Optional[Dict[str, float]]:
    """从 forward.json 文件加载数据并计算 mean_loss 和 std_loss。"""
    # (此函数代码无变化)
    if not os.path.exists(file_path): return None
    try:
        from datasets import load_dataset
        dataset = load_dataset("json", data_files=file_path, split='train', cache_dir=f"./.cache/ds_{os.path.basename(file_path)}")
        losses = dataset["final_loss"]
        return {"loss_mean": np.mean(losses), "loss_std": np.std(losses)}
    except Exception as e:
        print(f"处理文件 {file_path} 时出错: {e}")
        return None


def calculate_mean_reward(file_path: Optional[str]) -> Optional[Dict[str, float]]:
    """从 gen_...json 文件加载数据并计算 mean_reward。"""
    # (此函数代码无变化)
    if file_path is None or not os.path.exists(file_path): return None
    try:
        from datasets import load_dataset
        dataset = load_dataset("json", data_files=file_path, split='train', cache_dir=f"./.cache/ds_{os.path.basename(file_path)}")
        def avgatn(example):
            example["avg"] = np.mean(example["analysis_corr"])
            example["best"] = np.max(example["analysis_corr"])
            return example
        dataset = dataset.map(function=avgatn, num_proc=4)
        avg = dataset["avg"]
        return {"avg_reward": np.mean(avg), "avg_reward_std": np.std(avg), "best_reward": np.mean(dataset["best"])}
    except Exception as e:
        print(f"处理文件 {file_path} 时出错: {e}")
        return None

def find_gen_json_file(directory: str) -> Optional[str]:
    """在目录中查找以 'gen' 开头并以 '.json' 结尾的文件。"""
    # (此函数代码无变化)
    if not os.path.isdir(directory): return None
    for filename in os.listdir(directory):
        if filename.startswith("gen") and filename.endswith(".json"):
            return os.path.join(directory, filename)
    return None


def plot_enhanced_training_trajectory(df: pd.DataFrame, output_path: str):
    """
    【V9 最终版】根据处理好的 DataFrame 绘制多层次轨迹图。
    - 趋势箭头: 从最初始的点指向最末尾的点，展示总体趋势。
    - 标注优化: 智能定位标注框，避免遮挡数据点。
    """
    print("\n正在生成 V9 最终版的轨迹图...")

    df_plot = df.dropna(subset=['loss_mean', 'step', 'avg_reward', 'best_reward']).copy()
    if df_plot.empty:
        print("没有足够的数据来生成图表。")
        return

    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(32, 14), sharex=True)
    
    unique_models = sorted(df_plot['model_name'].unique())
    palette = sns.color_palette("muted", n_colors=len(unique_models))
    color_map = dict(zip(unique_models, palette))

    plot_configs = [
        {
            'ax': axes[0], 'y_col': 'avg_reward', 
            # 'title': 'Avg@16 vs. Loss on Demonstrations'
        },
        {
            'ax': axes[1], 'y_col': 'best_reward', 
            # 'title': 'Best@16 vs. Loss Trajectory'
        }
    ]

    for config in plot_configs:
        ax = config['ax']
        y_col = config['y_col']
        
        for model_name in unique_models:
            model_df = df_plot[df_plot['model_name'] == model_name].sort_values('step')
            if model_df.shape[0] < 2:
                if not model_df.empty:
                     ax.scatter(model_df['loss_mean'], model_df[y_col], color=color_map[model_name], s=600, ec='black', marker='*', zorder=5)
                continue

            x_data = model_df['loss_mean']
            y_data = model_df[y_col]
            color = color_map[model_name]

            # 1. 用细微的点线连接真实的每一个数据点
            ax.plot(x_data, y_data, color=color, lw=2.0, alpha=0.7, linestyle=':', zorder=3)
            
            # 2. 【V9 核心修改】绘制一个从起点到终点的、代表总体趋势的大箭头
            x_start, y_start = x_data.iloc[0], y_data.iloc[0]
            x_final, y_final = x_data.iloc[-1], y_data.iloc[-1]
            
            ax.annotate("",
                        xy=(x_final, y_final), 
                        xytext=(x_start, y_start),
                        arrowprops=dict(
                            arrowstyle="-|>,head_width=0.6,head_length=1.2", # 定义箭头样式和大小
                            connectionstyle="arc3,rad=-0.3", # 弧度
                            # connectionstyle="arc3", # 直线连接
                            linestyle="--",
                            lw=20.0,
                            color=color,
                            alpha=0.3
                        ),
                        zorder=2)

            # 3. 叠加上原始数据点
            ax.scatter(x_data, y_data, color=color, s=300, ec='black', zorder=4, alpha=0.9)
            
            # 4. 突出终点的大星星
            ax.scatter(x_final, y_final, color='gold', s=1000, ec='black', marker='*', zorder=5)

            # 5. 智能定位标注，避免遮挡
            label_text = f" {model_name}\n Final: ({x_final:.2f}, {y_final:.2f})"
            
            # 标注位置的判断，仍然依据最后一步的移动方向，以获得最佳局部避让效果
            x_prev, y_prev = x_data.iloc[-2], y_data.iloc[-2]
            dx = x_final - x_prev
            dy = y_final - y_prev
            
            ha = 'right' if dx > 0 else 'left'
            va = 'top' if dy > 0 else 'bottom'
            
            ax.text(x_final+0.05, y_final+0.05, label_text, 
                    color='black',
                    fontsize=24,
                    fontweight='bold',
                    ha="right", va="top",
                    bbox=dict(boxstyle="round,pad=0.5", fc=to_rgba(color, 0.25), ec="none"),
                    zorder=10)

        # --- 美化子图 ---
        # ax.set_title(config['title'], fontsize=24, pad=20, weight='bold')
        ax.set_xlabel('Loss on Demontrations', fontsize=24, weight='bold')
        ax.tick_params(axis='both', which='major', labelsize=16)
        ax.invert_xaxis()

    axes[0].set_ylabel('Avg@16', fontsize=32, weight='bold')
    axes[1].set_ylabel('Best@16', fontsize=32, weight='bold')

    sns.despine(fig)
    #fig.suptitle('Overall Training Trajectory Analysis', fontsize=28, weight='bold', y=1.02)
    # fig.text(0.5, 0.96, 'Dashed arrow: overall trend from start to finish. Dotted line: actual steps. ★: final performance.',
    #          ha='center', va='center', fontsize=16, color='gray')
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"图表已成功保存到: {output_path}")
    plt.show()


def main():
    """
    主函数，用于遍历文件夹，提取数据，生成 DataFrame 并绘图。
    """
    CONFIG = {
        "root_dir": "/mnt/jfs2/cth/085b13/_analysis_v2",
        "model_map": {
            "rft_Qwen2.5-Math-1.5B_luffy_analysis_1p5b_16k_seed_42_sys_wo_format_num_empty_7_clip_0.8_25-06-21-Jun-06-1750458724": "Prefix-RFT",
            "rft_Qwen2.5-Math-1.5B_luffy_analysis_1p5b_16k_seed_42_sys_wo_format_num_empty_8_clip_0.8_25-06-21-Jun-06-1750459811": "RFT",
            "sft_Qwen2.5-Math-1.5B_luffy_analysis_1p5b_16k_seed_42_sys_wo_format_lr5e-5_25-06-23-Jun-06-1750677874": "SFT"
        },
        "output_csv_path": "experiment_results_final.csv",
        "output_image_path": "/data/_figs/training_trajectory.pdf"
    }
    
    all_results: List[Dict[str, Any]] = []

    root_dir = CONFIG["root_dir"]
    if not os.path.isdir(root_dir):
        print(f"错误: 根目录 '{root_dir}' 不存在。请检查路径。")
        return

    for model_dir_name, display_name in CONFIG["model_map"].items():
        model_path = os.path.join(root_dir, model_dir_name)
        if not os.path.isdir(model_path):
            print(f"警告: 未找到模型目录: {model_path}")
            continue
        
        print(f"\n正在处理模型: {display_name}")
        
        for step_dir_name in sorted(os.listdir(model_path)):
            if not step_dir_name.startswith("global_step"): continue
                
            step_match = re.search(r'global_step(\d+)', step_dir_name)
            if not step_match: continue
            
            step = int(step_match.group(1))
            step_path = os.path.join(model_path, step_dir_name)
            
            loss_metrics = calculate_mean_loss(os.path.join(step_path, "forward.json"))
            reward_metrics = calculate_mean_reward(find_gen_json_file(step_path))
            
            result = {"model_name": display_name, "step": step}
            if loss_metrics: result.update(loss_metrics)
            if reward_metrics: result.update(reward_metrics)
            
            if len(result) > 2: all_results.append(result)

    if not all_results:
        print("\n未找到任何结果。请检查 root_dir 和文件夹结构。")
        return

    df = pd.DataFrame(all_results)
    df_sorted = df.sort_values(by=["model_name", "step"]).reset_index(drop=True)

    print("\n--- 最终结果 ---")
    print(df_sorted)

    df_sorted.to_csv(CONFIG["output_csv_path"], index=False)
    print(f"\n结果数据已保存到: {CONFIG['output_csv_path']}")

    # 调用最终修复版的绘图函数
    plot_enhanced_training_trajectory(df_sorted, CONFIG["output_image_path"])


if __name__ == "__main__":
    main()