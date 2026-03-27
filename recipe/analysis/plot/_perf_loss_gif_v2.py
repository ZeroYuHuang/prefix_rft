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
from matplotlib.patches import FancyArrowPatch #  <-- 【新】引入箭头补丁

# ! 依赖项: imageio
try:
    import imageio
    IMAGEIO_AVAILABLE = True
except ImportError:
    print("警告: 未找到 imageio 库。无法生成GIF。")
    print("请在您的环境中运行: pip install imageio")
    IMAGEIO_AVAILABLE = False

# --- 1. 全局绘图设置 ---
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
plt.rcParams['axes.titlesize'] = 20
plt.rcParams['axes.labelsize'] = 18
plt.rcParams['xtick.labelsize'] = 14
plt.rcParams['ytick.labelsize'] = 14
plt.rcParams['legend.fontsize'] = 12

# --- 2. 数据处理辅助函数 (无改动) ---
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

# --- 3. 绘图辅助函数 ---
def add_header_to_frame(np_frame, suptitle_text, conclusion_text=""):
    height, width, _ = np_frame.shape
    dpi = 100
    fig_size = (width / dpi, height / dpi)
    fig, ax = plt.subplots(figsize=fig_size, dpi=dpi)
    ax.imshow(np_frame)
    ax.axis('off')
    if suptitle_text != "":
        fig.suptitle(suptitle_text, fontsize=32, weight='bold', y=0.98)
    if conclusion_text != "":
        fig.text(0.5, 0.92, conclusion_text, ha='center', va='center', fontsize=42, color='firebrick', weight='bold')
    fig.canvas.draw()
    rgba_buffer = fig.canvas.buffer_rgba()
    np_frame_rgba = np.asarray(rgba_buffer)
    np_frame_rgb = np_frame_rgba[:, :, :3]
    plt.close(fig)
    return np_frame_rgb

def plot_gif_frame_unified(df: pd.DataFrame, current_step: int, color_map: dict, plot_configs: list, global_xlim: tuple, bounds_per_plot: dict, title_text: str):
    fig, axes = plt.subplots(2, 2, figsize=(18, 16)) # 增加了一点垂直尺寸
    fig.suptitle(title_text, fontsize=32, weight='bold', y=0.98)

    # --- 绘图循环，但内部不再有箭头 ---
    for ax, config in zip(axes.flatten(), plot_configs):
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
                ax.text(x_val, y_val, label_text, color='black', fontsize=13, fontweight='bold', ha='left', va='center', bbox=dict(boxstyle="round,pad=0.4", fc=to_rgba(color, 0.35), ec="none"), zorder=11)
        ax.set_xlabel('Loss on Demonstrations', weight='bold'); ax.set_ylabel(config['ylabel'], weight='bold', fontsize=22)
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)
        ax.set_xlim(global_xlim); ax.invert_xaxis()
        plot_key = (subset, y_col)
        if plot_key in bounds_per_plot: ax.set_ylim(bounds_per_plot[plot_key]['ylim'])

    # --- 调整子图布局，为全局箭头和文字留出空间 ---
    fig.subplots_adjust(left=0.1, right=0.95, bottom=0.1, top=0.9)

    # ==================== 【全新方案】在整个画布上添加清晰的全局箭头 ====================
    # 定义一个显眼的箭头样式
    arrow_style = dict(arrowstyle="Simple, tail_width=3, head_width=10, head_length=12",
                       color='gray',
                       lw=1)

    # Y轴方向: "Higher is Better"
    # 添加文字
    fig.text(0.03, 0.5, "Higher is Better",
             ha='center', va='center', rotation='vertical',
             fontsize=20, color='black', weight='bold')
    # 添加箭头 (使用 figure-level 坐标)
    y_arrow = FancyArrowPatch((0.04, 0.3), (0.04, 0.7),
                              transform=fig.transFigure,
                              clip_on=False, # 允许在画布的任何地方绘制
                              **arrow_style)
    fig.patches.append(y_arrow)


    # X轴方向: "Lower is Better" (注意图中Loss是反向的，所以箭头向右)
    # 添加文字
    fig.text(0.5, 0.02, "Lower is Better",
             ha='center', va='center',
             fontsize=20, color='black', weight='bold')
    # 添加箭头
    x_arrow = FancyArrowPatch((0.7, 0.04), (0.3, 0.04),
                              transform=fig.transFigure,
                              clip_on=False,
                              **arrow_style)
    fig.patches.append(x_arrow)
    # =================================================================================

    img_buffer = io.BytesIO()
    plt.savefig(img_buffer, format='png', dpi=100)
    img_buffer.seek(0)
    frame = imageio.imread(img_buffer)
    plt.close(fig)

    if frame.shape[2] == 4:
        frame = frame[:, :, :3]

    return frame

# --- 4. GIF 生成主函数 (无改动) ---
def create_training_dynamics_gif_v7(df: pd.DataFrame, output_path: str):
    if not IMAGEIO_AVAILABLE:
        print("ImageIO 未安装，无法生成GIF。"); return
    print("\n正在准备生成GIF动画 (最终修正版)...")
    df_plot = df.dropna(subset=['loss_mean', 'step', 'avg_reward', 'best_reward']).copy()
    if df_plot.empty:
        print("没有足够的数据来生成GIF。"); return

    unique_models = sorted(df_plot['model_name'].unique())
    palette = sns.color_palette("muted", n_colors=len(unique_models))
    color_map = dict(zip(unique_models, palette))
    plot_configs = [ {'subset': 'Normal', 'y_col': 'avg_reward', 'ylabel': 'Avg@16'}, {'subset': 'Normal', 'y_col': 'best_reward', 'ylabel': 'Best@16'}, {'subset': 'Hard', 'y_col': 'avg_reward', 'ylabel': 'Avg@16 (Hard Set)'}, {'subset': 'Hard', 'y_col': 'best_reward', 'ylabel': 'Best@16 (Hard Set)'} ]
    print("正在计算坐标轴边界...")
    x_min_all, x_max_all = df_plot['loss_mean'].min(), df_plot['loss_mean'].max()
    x_pad_all = (x_max_all - x_min_all) * 0.1
    global_xlim = (x_max_all + x_pad_all, x_min_all - x_pad_all)
    bounds_per_plot = {}
    for config in plot_configs:
        subset, y_col = config['subset'], config['y_col']
        plot_key = (subset, y_col)
        subset_df = df_plot[df_plot['subset'] == subset]
        if subset_df.empty: continue
        y_min, y_max = subset_df[y_col].min(), subset_df[y_col].max()
        y_pad = (y_max - y_min) * 0.1
        bounds_per_plot[plot_key] = {'ylim': (y_min - y_pad, y_max + y_pad)}

    all_steps = sorted(df_plot['step'].unique())
    frames = []

    print(f"\n将为 {len(all_steps)} 个时间步生成帧...")
    for i, step in enumerate(all_steps):
        print(f"  - 正在渲染第 {i+1}/{len(all_steps)} 帧 (Step: {step})...")
        title = f'Training Dynamics at Step: {step}'
        frame = plot_gif_frame_unified(
            df_plot, step, color_map, plot_configs, global_xlim, bounds_per_plot, title_text=title
        )
        frames.append(frame)
    if not frames:
        print("没有成功渲染任何帧，无法生成GIF。"); return

    frame_duration = 1.0
    # text_to_animate = "Prefix-RFT effectively blends SFT and RFT!"
    # words = text_to_animate.split()
    # text_stages = [" ".join(words[:i+1]) for i in range(len(words))]
    # print(f"\n正在为GIF结尾添加“打字机”动画...")
    # if frames:
    #     last_chart_frame = frames[-1]
    #     for stage_text in text_stages:
    #         frame_with_text = add_header_to_frame(last_chart_frame, suptitle_text="", conclusion_text=stage_text)
    #         frames.append(frame_with_text)
    #         frames.append(frame_with_text)
    #     final_hold_seconds = 3
    #     num_final_hold_frames = int(final_hold_seconds / frame_duration)
    #     last_text_frame = frames[-1]
    #     for _ in range(num_final_hold_frames):
    #         frames.append(last_text_frame)

    print("\n正在检查并统一所有帧的尺寸...")
    if not frames:
        print("错误: 没有可供处理的帧。")
        return

    target_shape = frames[0].shape
    if len(target_shape) != 3 or target_shape[2] != 3:
        print(f"错误: 标准帧的尺寸 {target_shape} 不是有效的RGB图像。")
        return

    standardized_frames = []
    for i, frame in enumerate(frames):
        current_frame = frame
        if len(current_frame.shape) != 3 or current_frame.shape[2] == 4:
            current_frame = current_frame[:, :, :3]

        if current_frame.shape != target_shape:
            print(f"  - 警告: 第 {i+1} 帧的尺寸 {current_frame.shape} 与标准尺寸 {target_shape} 不匹配。正在进行裁剪...")
            h, w, _ = target_shape
            standardized_frames.append(current_frame[:h, :w, :])
        else:
            standardized_frames.append(current_frame)

    print("所有帧尺寸已统一。")

    print(f"所有帧渲染完毕，正在合成为GIF文件... (总帧数: {len(standardized_frames)})")
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    imageio.mimsave(output_path, standardized_frames, duration=frame_duration, loop=0)
    print(f"GIF动画已成功保存到: {output_path}")

# --- 5. 主执行逻辑 (无改动) ---
def main():
    CONFIG = {
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
        "output_gif_path": "/data/_figs/training_dynamics_final_annotated.gif"
    }

    if os.path.exists(CONFIG["output_csv_path"]):
        print(f"发现已存在的CSV文件: {CONFIG['output_csv_path']}，正在直接加载...")
        df_final = pd.read_csv(CONFIG["output_csv_path"])
    else:
        print("未发现CSV文件，开始执行数据收集和处理流程...")
        rft_ckpt_paths = { step: os.path.join(CONFIG["root_dir"], CONFIG["rft_exp_for_hardset"], f"global_step{step}", "gen_t1.0_top_p_0.95_max_tokens8192.json") for step in CONFIG["rft_steps_for_hardset"] }
        rft_ckpt_paths = {k: v for k, v in rft_ckpt_paths.items() if os.path.exists(v)}
        if not rft_ckpt_paths:
            hardset_uids, subsets_to_process = [], {"Normal": None}
            print("警告: 未找到用于定义困难集的RFT检查点文件，将跳过困难集分析。")
        else:
            hardset_uids = collect_hard_prompt_uids(rft_ckpt_paths, num_hardest=CONFIG["num_hardest_samples"])
            subsets_to_process = {"Normal": None, "Hard": hardset_uids}

        all_results = []
        for model_dir_name, display_name in CONFIG["model_map"].items():
            model_path = os.path.join(CONFIG["root_dir"], model_dir_name)
            if not os.path.isdir(model_path):
                print(f"警告: 找不到模型目录 {model_path}")
                continue

            print(f"\n正在处理模型: {display_name}")
            step_dirs = [d for d in os.listdir(model_path) if d.startswith("global_step")]
            step_dirs.sort(key=lambda x: int(re.search(r'(\d+)', x).group(1)))

            for step_dir_name in step_dirs:
                step_match = re.search(r'global_step(\d+)', step_dir_name)
                if not step_match: continue

                step = int(step_match.group(1))
                step_path = os.path.join(model_path, step_dir_name)
                forward_json_path = os.path.join(step_path, "forward.json")
                gen_json_path = find_gen_json_file(step_path)

                print(f"  - 处理步骤 {step}...")
                for subset_name, uids in subsets_to_process.items():
                    loss_metrics = calculate_mean_loss(forward_json_path, selected_uids=uids)
                    reward_metrics = calculate_mean_reward(gen_json_path, selected_uids=uids)
                    result = {"model_name": display_name, "step": step, "subset": subset_name}
                    if loss_metrics: result.update(loss_metrics)
                    if reward_metrics: result.update(reward_metrics)
                    if len(result) > 3: all_results.append(result)

        if not all_results:
            print("\n错误: 未能收集到任何有效数据，程序终止。")
            return

        df_final = pd.DataFrame(all_results)
        df_final = df_final.sort_values(by=["subset", "model_name", "step"]).reset_index(drop=True)
        df_final.to_csv(CONFIG["output_csv_path"], index=False)
        print(f"\n结果数据已保存到: {CONFIG['output_csv_path']}")

    if df_final is not None and not df_final.empty:
        create_training_dynamics_gif_v7(df_final, CONFIG["output_gif_path"])
    else:
        print("没有数据可用于绘图。")


if __name__ == "__main__":
    main()