from datasets import load_dataset

for_data = load_dataset(
    "json",
    data_files='/mnt/jfs2/cth/085b13/_analysis_v2/rft_Qwen2.5-Math-1.5B_luffy_analysis_1p5b_16k_seed_42_sys_wo_format_num_empty_7_clip_0.2_25-06-21-Jun-06-1750458680/global_step128/forward.json'
)
print(for_data)

gen_data = load_dataset(
    "json",
    data_files='/mnt/jfs2/cth/085b13/_analysis_v2/rft_Qwen2.5-Math-1.5B_luffy_analysis_1p5b_16k_seed_42_sys_wo_format_num_empty_7_clip_0.2_25-06-21-Jun-06-1750458680/global_step128/gen_t1.0_top_p_0.95_max_tokens8192.json'
)
print(gen_data)