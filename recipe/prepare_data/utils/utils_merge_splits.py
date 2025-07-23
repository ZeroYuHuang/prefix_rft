import os
import json
import sys
from tqdm import tqdm
folder_path="/mnt/jfs/tianhao/085b13/cth-dev3/_data/raw_dataset/math/Qwen2.5-Math-1.5B-Instruct"
files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]

prefix = "data_split_0.0_1.0"
files = [os.path.join(folder_path, f)  for f in files if f.startswith(prefix)]
# print(files)
# print(len(files))

# load the first file:
merged_data = {}
for fp in files:
    with open(fp, "r", encoding='utf-8') as f:
        patience = 0
        for line in tqdm(f):
            data = json.loads(line.strip())
            idx = data.get("idx")
            # print(f"The new data idx is {idx}")
            if idx not in merged_data:
                merged_data[idx]={k: v for k, v in data.items() if k not in ["model_solution", "correct"]}
                merged_data[idx]["model_solution"] = []
                merged_data[idx]["correct"] = []
            for k, v in data.items():
                if k not in ["model_solution", "correct"]:
                    # print(k, v, merged_data[idx][k])
                    assert v == merged_data[idx][k], f"{k} is not the same"
                # elif k == "model_solution":
                #     assert v not in merged_data[idx][k], f"
                #     # if v in merged_data[idx][k]:
                #         # print(v)
                #         # print(merged_data[idx][k])
                #         # exit()
            if data["model_solution"] not in merged_data[idx]["model_solution"]:
                merged_data[idx]["model_solution"].append(data["model_solution"])
                merged_data[idx]["correct"].append(data["correct"])
            else:
                # print(f"duplicate answer: {fp}")
                patience += 1
            
            if patience >= 20:
                print(f"duplicate answer: {fp}")
                break

output_file=os.path.join(folder_path, f"{prefix}.jsonl")
with open(output_file, 'w', encoding='utf-8') as f:
    for question_id, dp in merged_data.items():
        f.write(json.dumps(dp, ensure_ascii=False) + '\n')


