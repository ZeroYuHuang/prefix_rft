import os
from datasets import load_dataset, concatenate_datasets

if __name__ == "__main__":
    # arc_c: 1172
    # gpqa: 198
    # mmlu_pro: 12032
    # valid_all: 6023
    # dataset = load_dataset("/data/reference-codebase/LUFFY/data/valid.all.parquet", split="train")
    dataset = load_dataset("parquet", data_files="/home/ma-user/work/hzy-dev/LUFFY/data/valid.all.parquet")['train']
    print(dataset)
    print(dataset[0])
    HOME_DIR = "/home/ma-user/work/hzy-dev/"

    def change_prompt(example, id):
        example["prompt"] = [
            {"content": example["prompt"][1]["content"], "role": "user"}
        ]
        return example

    dataset = dataset.map(function=change_prompt, with_indices=True, num_proc=16)
    print(dataset[0]["prompt"])
    save_dir = f"{HOME_DIR}/_data/processed_dataset_new/test_math_luffy"
    print(f"Save to {save_dir}/test.parquet")
    dataset.to_parquet(os.path.join(save_dir, f'test.parquet'))