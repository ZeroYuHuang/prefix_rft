import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.patches as mpatches

# --- 1. 全局绘图设置 ---
# 设置全局字体为 Times New Roman
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
# 设置Seaborn主题
sns.set_theme(style="white")

# --- 2. 数据定义 ---
qwen_data = {
    "SFT": 31.9,
    "RFT": 30.0,
    "ReLIFT": 34.4,
    "LUFFY": 38.0,
    "Prefix-RFT": 41.0
}

llama_data = {
    "SFT": 5.9,
    "RFT": 9.6,
    "LUFFY": 13.2,
    "Prefix-RFT": 14.5
}

# --- 3. 创建统一的颜色映射和Y轴限制 ---
# **修正**: 定义图例的显示顺序
# 1. 使用左侧图表的顺序作为基准
legend_order = list(qwen_data.keys())

# 2. 找到所有方法，并将只存在于右侧图表的方法添加到末尾 (为了代码稳健性)
all_methods_set = set(qwen_data.keys()) | set(llama_data.keys())
remaining_methods = sorted([m for m in all_methods_set if m not in legend_order])
final_legend_order = legend_order + remaining_methods

# 3. 根据最终的顺序来创建调色板和颜色映射
palette = sns.color_palette("Set2", n_colors=len(final_legend_order))
color_map = {method: color for method, color in zip(final_legend_order, palette)}

all_scores = list(qwen_data.values()) + list(llama_data.values())
global_max_score = max(all_scores)
y_axis_limit = global_max_score * 1.15

# --- 4. 创建图形和子图 ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 5))

# --- 5. 绘制子图 ---

# 左侧子图: Qwen1.5B 结果
methods1 = list(qwen_data.keys())
scores1 = list(qwen_data.values())
bar_colors1 = [color_map[method] for method in methods1]

ax1.bar(methods1, scores1, color=bar_colors1, width=0.6)
ax1.set_title('Qwen2.5-Math-1.5B', fontsize=16, weight='bold')
ax1.set_ylim(0, y_axis_limit)
ax1.set_xticklabels([])
ax1.tick_params(axis='x', length=0)
ax1.grid(False)

max_score1 = max(scores1)
for bar in ax1.patches:
    yval = bar.get_height()
    font_weight = 'bold' if yval == max_score1 else 'normal'
    ax1.text(bar.get_x() + bar.get_width()/2.0, yval + 0.5, f'{yval:.1f}',
             ha='center', va='bottom', fontsize=13, weight=font_weight)

# 右侧子图: Llama3 8B 结果
methods2 = list(llama_data.keys())
scores2 = list(llama_data.values())
bar_colors2 = [color_map[method] for method in methods2]

ax2.bar(methods2, scores2, color=bar_colors2, width=0.6)
ax2.set_title('LLaMA-3.1-8B', fontsize=16, weight='bold')
ax2.set_ylim(0, y_axis_limit)
ax2.set_xticklabels([])
ax2.tick_params(axis='x', length=0)
ax2.grid(False)

max_score2 = max(scores2)
for bar in ax2.patches:
    yval = bar.get_height()
    font_weight = 'bold' if yval == max_score2 else 'normal'
    ax2.text(bar.get_x() + bar.get_width()/2.0, yval + 0.2, f'{yval:.1f}',
             ha='center', va='bottom', fontsize=13, weight=font_weight)

# --- 6. 创建并添加共享图例 ---
# **修正**: 根据我们新定义的 'final_legend_order' 来创建图例
legend_patches = [mpatches.Patch(color=color_map[method], label=method) for method in final_legend_order]
fig.legend(handles=legend_patches,
           loc='upper center',
           bbox_to_anchor=(0.5, 0.99),
           ncol=len(final_legend_order),
           fontsize=16,
           frameon=False,
           # **优化**: 调整间距使图例更紧凑
           columnspacing=0.8,  # 减小列间距 (默认值 2.0)
           handletextpad=0.5   # 减小颜色块和文字的间距 (默认值 0.8)
          )

# --- 7. 最终布局与展示 ---
plt.tight_layout(rect=[0, 0, 1, 0.93])

# 保存为高分辨率PDF (请确保目录存在)
plt.savefig("/data/_figs/more_models_fixed.pdf", dpi=300, bbox_inches='tight')

plt.show()