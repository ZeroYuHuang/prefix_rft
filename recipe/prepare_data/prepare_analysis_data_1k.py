from datasets import load_dataset

HOME_DIR = "/mnt/jfs2/cth/085b13"
parquet_file = f"{HOME_DIR}/_data/processed_dataset_new/luffy_analysis_1p5b_16k_seed_42_sys_wo_format/valid.parquet"

dataset = load_dataset('parquet', data_files=parquet_file)["train"]
print(dataset)

dataset = dataset.shuffle(seed=42)
dataset = dataset.select(range(2048))

levels =  set(dataset["difficulty"])
for level in levels:
    print(
        len(dataset.filter(lambda example: example['difficulty'] == level)), 
        level
    )
dataset.to_parquet(f"{HOME_DIR}/_data/processed_dataset_new/luffy_analysis_1p5b_16k_seed_42_sys_wo_format/valid_2k.parquet")