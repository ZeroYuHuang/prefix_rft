import json
exp_name = "sft_Qwen2.5-Math-1.5B_luffy_analysis_1p5b_16k_seed_42_sys_wo_format_lr_25-06-19-Jun-06-1750280661"
step = 256
fpath_forward = f"/mnt/jfs2/cth/085b13/_analysis/{exp_name}/global_step{step}/forward.json"
forward_data = []
with open(fpath_forward, 'r') as f:
    for line in f:
        # Skip empty lines that might be at the end of the file
        if line.strip(): 
            forward_data.append(json.loads(line))

# with open(fpath_forward, 'r', encoding='utf-8') as f:
#     forward_data = json.load(f)

print(forward_data[0]["final_loss"])
print(forward_data[0]["perplexity"])