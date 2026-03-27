import numpy as np
from datasets import load_dataset

if __name__ == "__main__":
    base_dir = "/mnt/jfs2/cth/085b13//_analysis"
    # exp_name = "sft_Qwen2.5-Math-1.5B_luffy_analysis_1p5b_16k_seed_42_sys_wo_format_lr1e-5_25-06-21-Jun-06-1750458477"
    exp_name = "sft_Qwen2.5-Math-1.5B_luffy_analysis_1p5b_16k_seed_42_sys_wo_format_lr5e-5_25-06-19-Jun-06-1750280661"
    step = 640
    gen_config = "gen_t0.6_top_p_0.95_max_tokens8192"
    fpath = f"{base_dir}/{exp_name}/global_step{step}/gen_t0.6_top_p_0.95_max_tokens8192.json"
    dataset = load_dataset("json", data_files=fpath)['train']
    print(dataset)
    print(len(dataset[2]["analysis_gen"]))
    rewards = [np.mean(dataset[i]["analysis_corr"]) for i in range(len(dataset))]
    avgat16 = np.mean(rewards)
    passat16 = np.mean([1 if r > 0 else 0 for r in rewards])
    majat16 = np.mean([1 if r > 0.5 else (0.5 if r==0.5 else 0 ) for r in rewards])
    print(avgat16)
    print(passat16)
    print(majat16)