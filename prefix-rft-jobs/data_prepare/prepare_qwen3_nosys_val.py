import os
import datasets
from datasets import load_dataset

if __name__ == "__main__":
    data_path = "/home/ma-user/work/hzy-dev/_data/processed_dataset_new/luffy_test_no_sys/test.parquet"
    dataset = load_dataset("parquet", data_files=data_path)
    print(datasets.__version__)
    # HOME_DIR = "/home/ma-user/work/hzy-dev"
    # save_dir = f"{HOME_DIR}/_data/processed_dataset_new/luffy_test_no_sys"
    # # dataset = load_dataset("parquet", data_files="/home/ma-user/work/hzy-dev/_data/processed_dataset_new/luffy_test_sys_wo_format/test.parquet")['train']
    # dataset = load_dataset("parquet", data_files="/home/ma-user/work/hzy-dev/_data/processed_dataset_new/luffy_test_no_sys/test.parquet")["train"]
    # # print(dataset[0])
    # print(dataset[0].keys())
    # print(dataset[0]["prompt"])
    # postfix = "Let's think step by step and output the final answer within \\boxed{}"


    # def delete_sys_prompt(example):
    #     raw_prompt = example["prompt"][1]['content']
    #     example["prompt"] = [
    #         {"content": raw_prompt + postfix, "role": "user"}
    #     ]
    #     return example
    
    # dataset = dataset.map(function=delete_sys_prompt, num_proc=16)
    # print(dataset)
    # print(f"Save to {save_dir}/test.parquet")
    # dataset.to_parquet(os.path.join(save_dir, f'test.parquet'))

