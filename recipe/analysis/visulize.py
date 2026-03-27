import numpy as np
from datasets import load_dataset

dataset = load_dataset(
    "json",
    data_files="/data/gen_t1.0_top_p_0.95_max_tokens8192.jsonl"
)['train']
print(dataset)
for i in range(len(dataset)):
    print(np.mean(dataset[i]["analysis_corr"]))