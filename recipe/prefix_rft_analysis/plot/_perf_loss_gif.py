import os
import re
import io
import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Any
from collections import defaultdict

# 引入核心库
from datasets import load_dataset
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba

# ! 依赖项: imageio
try:
    import imageio
    IMAGEIO_AVAILABLE = True
except ImportError:
    print("警告: 未找到 imageio 库。无法生成GIF。")
    print("请在您的环境中运行: pip install imageio")
    IMAGEIO_AVAILABLE = False
    
# --- 1. 全局绘图设置 (不变) ---
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
plt.rcParams['axes.titlesize'] = 20
plt.rcParams['axes.labelsize'] = 18
plt.rcParams['xtick.labelsize'] = 14
plt.rcParams['ytick.labelsize'] = 14
plt.rcParams['legend.fontsize'] = 12

# (calculate_mean_loss, calculate_mean_reward, etc. functions are unchanged)
def calculate_mean_loss(file_path: str, selected_uids: Optional[list] = None) -> Optional[dict]:
    if not os.path.exists(file_path): return None
    try:
        dataset = load_dataset("json", data_files=file_path, split='train', cache_dir=f"./.cache/ds_{os.path.basename(file_path)}")
        if selected_uids: dataset = dataset.filter(lambda x: x["uid"] in selected_uids, num_proc=os.cpu_count())
        if len(dataset) == 0: return None
        return {"loss_mean": np.mean(dataset["final_loss"]), "loss_std": np.std(dataset["final_loss"])}
    except Exception as e: return None

def calculate_mean_reward(file_path: str, selected_uids: Optional[list] = None) -> Optional[dict]:
    if file_path is None or not os.path.exists(file_path): return None
    try:
        dataset = load_dataset("json", data_files=file_path, split='train', cache_dir=f"./.cache/ds_{os.path.basename(file_path)}")
        if selected_uids: dataset = dataset.filter(lambda x: x["uid"] in selected_uids, num_proc=os.cpu_count())
        if len(dataset) == 0: return None
        def avgatn(example):
            example["avg"] = np.mean(example["analysis_corr"]); example["best"] = np.max(example["analysis_corr"]); return example
        dataset = dataset.map(function=avgatn, num_proc=os.cpu_count())
        return {"avg_reward": np.mean(dataset["avg"]), "avg_reward_std": np.std(dataset["avg"]), "best_reward": np.mean(dataset["best"])}
    except Exception as e: return None

def collect_hard_prompt_uids(fpaths: Dict[int, str], num_hardest: int = 256) -> list:
    question_performance = defaultdict(dict)
    print("正在提取最困难的样本子集...")
    for step, fpath in fpaths.items():
        dataset = load_dataset("json", data_files=fpath, split='train')['train']
        for record in dataset:
            uid, corr_list = record.get('uid'), record.get('analysis_corr')
            if uid is None or not corr_list: continue
            question_performance[uid][step] = np.mean(corr_list)
    df = pd.DataFrame.from_dict(question_performance, orient='index')
    df = df.reindex(columns=sorted(fpaths.keys())); df["difficulty"] = df.mean(axis=1)
    hardest_questions_df = df.nsmallest(num_hardest, 'difficulty')
    print(f"已识别出 {len(hardest_questions_df)} 个最困难的样本。")
    return list(hardest_questions_df.index)

def find_gen_json_file(directory: str) -> Optional[str]:
    if not os.path.isdir(directory): return None
    for filename in sorted(os.listdir(directory)):
        if filename.startswith("gen") and filename.endswith(".json"): return os.path.join(directory, filename)
    return None

# ## <-- 新增辅助函数: 在图像帧上添加顶部标题和文字 (修正版) -->
def add_header_to_frame(np_frame, suptitle_text, conclusion_text=""):
    """
    在一个numpy图像帧上重新绘制并添加标题和结论文字。
    (V2 - 修复了 reshape bug)
    """
    height, width, _ = np_frame.shape
    dpi = 100 
    fig_size = (width / dpi, height / dpi)

    fig, ax = plt.subplots(figsize=fig_size, dpi=dpi)
    ax.imshow(np_frame)
    ax.axis('off')

    if suptitle_text != "":
        print("Adding subtitle")
        fig.suptitle(suptitle_text, fontsize=32, weight='bold', y=0.98)

    if conclusion_text != "":
        print("Adding concultion text")
        fig.text(0.5, 0.92, conclusion_text, 
                 ha='center', va='center', 
                 fontsize=42, 
                 color='firebrick', 
                 weight='bold')

    # ## <-- 核心修正区域 -->

    # 1. 确保画布内容已绘制
    fig.canvas.draw()
    
    # 2. 从画布获取4通道的RGBA缓冲
    rgba_buffer = fig.canvas.buffer_rgba()
    
    # 3. 将缓冲转换为numpy数组，其形状将是 (height, width, 4)
    np_frame_rgba = np.asarray(rgba_buffer)
    
    # 4. 我们只需要RGB通道，所以切片去掉最后一个Alpha通道
    np_frame_rgb = np_frame_rgba[:, :, :3]

    # ## <-- 修正结束 -->
    
    plt.close(fig) 
    
    return np_frame_rgb

# 建议的 plot_gif_frame_v5 函数 (V5.1 - 紧凑版)
def plot_gif_frame_v5(df: pd.DataFrame, current_step: int, color_map: dict, plot_configs: list, global_xlim: tuple, bounds_per_plot: dict):
    """
    V5.1 - 紧凑版:
    - 减小了 figsize 以降低整体的空旷感。
    - 使用 fig.subplots_adjust() 精确控制子图间距，使其更紧凑。
    """
    # 【修改1】减小 figsize，让整体布局更紧凑
    fig, axes = plt.subplots(2, 2, figsize=(18, 15)) # 从 (24, 20) 调整为 (18, 15)

    for ax, config in zip(axes.flatten(), plot_configs):
        # --- 内部绘图逻辑保持不变 ---
        subset, y_col = config['subset'], config['y_col']
        subset_df = df[df['subset'] == subset]
        for model_name, model_df in subset_df.groupby('model_name'):
            history_df = model_df[model_df['step'] <= current_step].sort_values('step')
            if history_df.empty: continue
            x_data, y_data = history_df['loss_mean'], history_df[y_col]
            color = color_map[model_name]
            ax.plot(x_data, y_data, color=color, alpha=0.5, linestyle='--', marker='o', markersize=6, zorder=2)
            current_point = history_df[history_df['step'] == current_step]
            if not current_point.empty:
                x_val, y_val = current_point['loss_mean'].iloc[0], current_point[y_col].iloc[0]
                ax.scatter(x_val, y_val, color=color, s=500, ec='black', marker='*', zorder=10)
                label_text = f" {model_name}\n ({x_val:.3f}, {y_val:.3f})"
                ax.text(x_val, y_val, label_text, color='black', fontsize=16, fontweight='bold', ha='left', va='center', bbox=dict(boxstyle="round,pad=0.4", fc=to_rgba(color, 0.35), ec="none"), zorder=11)
        ax.set_xlabel('Loss on Demonstrations', weight='bold'); ax.set_ylabel(config['ylabel'], weight='bold', fontsize=22)
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)
        ax.set_xlim(global_xlim); ax.invert_xaxis()
        plot_key = (subset, y_col)
        if plot_key in bounds_per_plot: ax.set_ylim(bounds_per_plot[plot_key]['ylim'])

    # 【修改2】使用 subplots_adjust 替换 tight_layout
    # wspace 控制子图的水平间距，hspace 控制垂直间距。值越小，间距越小。
    # left, right, bottom, top 控制整个图表相对于画布边缘的距离。
    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.08, top=0.93, wspace=0.2, hspace=0.3)
    
    # 原来的代码:
    # plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    img_buffer = io.BytesIO()
    plt.savefig(img_buffer, format='png', dpi=100)
    img_buffer.seek(0)
    plt.close(fig)
    return imageio.imread(img_buffer)


def create_training_dynamics_gif_v7(df: pd.DataFrame, output_path: str):
    """
    【V7 演示版】GIF生成主控函数:
    - 结尾的结论文字以“打字机”效果逐词呈现。
    """
    if not IMAGEIO_AVAILABLE:
        print("ImageIO 未安装，无法生成GIF。"); return
        
    print("\n正在准备生成GIF动画 (V7)...")
    df_plot = df.dropna(subset=['loss_mean', 'step', 'avg_reward', 'best_reward']).copy()
    if df_plot.empty:
        print("没有足够的数据来生成GIF。"); return
        
    unique_models = sorted(df_plot['model_name'].unique())
    palette = sns.color_palette("muted", n_colors=len(unique_models))
    color_map = dict(zip(unique_models, palette))
    plot_configs = [ {'subset': 'Normal', 'y_col': 'avg_reward', 'ylabel': 'Avg@16'}, {'subset': 'Normal', 'y_col': 'best_reward', 'ylabel': 'Best@16'}, {'subset': 'Hard', 'y_col': 'avg_reward', 'ylabel': 'Avg@16 (Hard Set)'}, {'subset': 'Hard', 'y_col': 'best_reward', 'ylabel': 'Best@16 (Hard Set)'} ]
    
    print("正在计算坐标轴边界...")
    x_min_all, x_max_all = df_plot['loss_mean'].min(), df_plot['loss_mean'].max()
    
    # 【修改3】大幅减少X轴的 padding，这是让图紧凑的关键
    x_pad_all = (x_max_all - x_min_all) * 0.1 # 从 0.25 减小到 0.1
    # 原来的代码:
    # x_pad_all = (x_max_all - x_min_all) * 0.25
    
    global_xlim = (x_max_all + x_pad_all, x_min_all - x_pad_all)
    bounds_per_plot = {}
    for config in plot_configs:
        subset, y_col = config['subset'], config['y_col']
        plot_key = (subset, y_col)
        subset_df = df_plot[df_plot['subset'] == subset]
        if subset_df.empty: continue
        y_min, y_max = subset_df[y_col].min(), subset_df[y_col].max()
        y_pad = (y_max - y_min) * 0.1 # Y轴也可以适当增加一点padding，让星标不至于顶到边框
        bounds_per_plot[plot_key] = {'ylim': (y_min - y_pad, y_max + y_pad)}
        
    all_steps = sorted(df_plot['step'].unique())
    frames = []

    print(f"\n将为 {len(all_steps)} 个时间步生成帧...")
    for i, step in enumerate(all_steps):
        print(f"  - 正在渲染第 {i+1}/{len(all_steps)} 帧 (Step: {step})...")
        
        # 1. 调用更新后的绘图函数
        base_frame = plot_gif_frame_v5(df_plot, step, color_map, plot_configs, global_xlim, bounds_per_plot)
        
        # 2. 添加标题的逻辑不变
        final_frame = add_header_to_frame(base_frame, f'Training Dynamics at Step: {step}')
        frames.append(final_frame)
    
    if not frames:
        print("没有成功渲染任何帧，无法生成GIF。"); return

    # --- “打字机”效果和GIF保存逻辑保持不变 ---
    frame_duration = 1.0
    text_to_animate = "Prefix-RFT effectively blends SFT and RFT!"
    words = text_to_animate.split()
    text_stages = [" ".join(words[:i+1]) for i in range(len(words))]
    print(f"\n正在为GIF结尾添加“打字机”动画...")
    if frames:
        last_chart_frame = frames[-1]
        for stage_text in text_stages:
            frame_with_text = add_header_to_frame(last_chart_frame, "", stage_text)
            frames.append(frame_with_text)
            frames.append(frame_with_text)
        final_hold_seconds = 3
        num_final_hold_frames = int(final_hold_seconds / frame_duration)
        last_text_frame = frames[-1]
        for _ in range(num_final_hold_frames):
            frames.append(last_text_frame)

    print(f"所有帧渲染完毕，正在合成为GIF文件... (总帧数: {len(frames)})")
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        
    imageio.mimsave(output_path, frames, duration=frame_duration, loop=0)
    print(f"GIF动画已成功保存到: {output_path}")

def main():
    CONFIG = {
        # ... (CONFIG 内容与之前版本相同) ...
        "root_dir": "/mnt/jfs2/cth/085b13/_analysis_v2",
        "rft_exp_for_hardset": "rft_Qwen2.5-Math-1.5B_luffy_analysis_1p5b_16k_seed_42_sys_wo_format_num_empty_8_clip_0.8_25-06-21-Jun-06-1750459811",
        "rft_steps_for_hardset": [512, 544, 576, 608, 640],
        "num_hardest_samples": 256,
        "model_map": {
            "rft_Qwen2.5-Math-1.5B_luffy_analysis_1p5b_16k_seed_42_sys_wo_format_num_empty_7_clip_0.8_25-06-21-Jun-06-1750458724": "Prefix-RFT",
            "rft_Qwen2.5-Math-1.5B_luffy_analysis_1p5b_16k_seed_42_sys_wo_format_num_empty_8_clip_0.8_25-06-21-Jun-06-1750459811": "RFT",
            "sft_Qwen2.5-Math-1.5B_luffy_analysis_1p5b_16k_seed_42_sys_wo_format_lr5e-5_25-06-23-Jun-06-1750677874": "SFT"
        },
        "output_csv_path": "experiment_results_full_hard_and_normal.csv",
        "output_gif_path": "/data/_figs/training_dynamics_v7_presentation.gif"
    }

    if os.path.exists(CONFIG["output_csv_path"]):
        print(f"发现已存在的CSV文件: {CONFIG['output_csv_path']}，正在直接加载...")
        df_final = pd.read_csv(CONFIG["output_csv_path"])
    else:
        print("未发现CSV文件，开始执行数据收集和处理流程...")
        # ... (数据收集代码块不变) ...
        rft_ckpt_paths = { step: os.path.join(CONFIG["root_dir"], CONFIG["rft_exp_for_hardset"], f"global_step{step}", "gen_t1.0_top_p_0.95_max_tokens8192.json") for step in CONFIG["rft_steps_for_hardset"] }
        rft_ckpt_paths = {k: v for k, v in rft_ckpt_paths.items() if os.path.exists(v)}
        if not rft_ckpt_paths:
            hardset_uids = []
        else:
            hardset_uids = collect_hard_prompt_uids(rft_ckpt_paths, num_hardest=CONFIG["num_hardest_samples"])
        all_results = []
        subsets_to_process = {"Hard": hardset_uids, "Normal": None}
        for model_dir_name, display_name in CONFIG["model_map"].items():
            model_path = os.path.join(CONFIG["root_dir"], model_dir_name)
            if not os.path.isdir(model_path): continue
            print(f"\n正在处理模型: {display_name}")
            for step_dir_name in sorted(os.listdir(model_path)):
                if not step_dir_name.startswith("global_step"): continue
                step_match = re.search(r'global_step(\d+)', step_dir_name)
                if not step_match: continue
                step, step_path = int(step_match.group(1)), os.path.join(model_path, step_dir_name)
                forward_json_path, gen_json_path = os.path.join(step_path, "forward.json"), find_gen_json_file(step_path)
                for subset_name, uids in subsets_to_process.items():
                    if subset_name == "Hard" and not uids: continue
                    loss_metrics, reward_metrics = calculate_mean_loss(forward_json_path, selected_uids=uids), calculate_mean_reward(gen_json_path, selected_uids=uids)
                    result = {"model_name": display_name, "step": step, "subset": subset_name}
                    if loss_metrics: result.update(loss_metrics)
                    if reward_metrics: result.update(reward_metrics)
                    if len(result) > 3: all_results.append(result)
        if not all_results: print("\n未找到任何结果。"); return
        df_final = pd.DataFrame(all_results)
        df_final = df_final.sort_values(by=["subset", "model_name", "step"]).reset_index(drop=True)
        df_final.to_csv(CONFIG["output_csv_path"], index=False)
        print(f"\n结果数据已保存到: {CONFIG['output_csv_path']}")

    if df_final is not None and not df_final.empty:
        # 调用最终版的V7函数
        create_training_dynamics_gif_v7(df_final, CONFIG["output_gif_path"])
    else:
        print("没有数据可用于绘图。")


if __name__ == "__main__":
    main()