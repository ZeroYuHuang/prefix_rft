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

    def filter_aime(example):
        return "aime" in example['data_source']
    
    dataset = dataset.filter(filter_aime, num_proc=16)
    print(f"After filtering, dataset size: {len(dataset)}")
    # repeat dataset for 4 times
    dataset = concatenate_datasets([dataset] * 16)
    print(f"After repeating, dataset size: {len(dataset)}")
    # shuffle dataset
    # dataset = dataset.shuffle(seed=42)
    sys_prompt_wo_format = 'Your task is to follow a systematic, thorough reasoning process before providing the final solution. This involves analyzing, summarizing, exploring, reassessing, and refining your thought process through multiple iterations. Structure your response into two sections: Thought and Solution. Each thought should include detailed analysis, brainstorming, verification, and refinement of ideas. In the Solution section, provide the final, logical, and accurate answer, clearly derived from the exploration in the Thought section. Include the answer in \\boxed{} for closed-form results like multiple choices or mathematical solutions'
    inst_after_prompt = '\nPlease reason step by step, and put your final answer within \\boxed{}'
    # # exit()
    HOME_DIR = "/home/ma-user/work/hzy-dev/"

    def change_prompt(example, id):
        example["prompt"] = [
            {"content": sys_prompt_wo_format, "role": "system"},
            {"content": example["prompt"][1]["content"] + inst_after_prompt, "role": "user"}
        ]
        return example

    dataset = dataset.map(function=change_prompt, with_indices=True, num_proc=16)
    print(dataset[0]["prompt"])
    save_dir = f"{HOME_DIR}/_data/processed_dataset_new/aime2425_512"
    print(f"Save to {save_dir}/test.parquet")
    dataset.to_parquet(os.path.join(save_dir, f'test.parquet'))