import os
from datasets import load_dataset

if __name__ == "__main__":
    # arc_c: 1172
    # gpqa: 198
    # mmlu_pro: 12032
    # valid_all: 6023
    # dataset = load_dataset("/data/reference-codebase/LUFFY/data/valid.all.parquet", split="train")
    dataset = load_dataset("parquet", data_files="/data/code/LUFFY/data/valid.all.parquet")['train']
    print(dataset)
    print(dataset[0])
    sys_prompt_wo_format = 'Your task is to follow a systematic, thorough reasoning process before providing the final solution. This involves analyzing, summarizing, exploring, reassessing, and refining your thought process through multiple iterations. Structure your response into two sections: Thought and Solution. Each thought should include detailed analysis, brainstorming, verification, and refinement of ideas. In the Solution section, provide the final, logical, and accurate answer, clearly derived from the exploration in the Thought section. Include the answer in \\boxed{} for closed-form results like multiple choices or mathematical solutions'
    inst_after_prompt = '\nPlease reason step by step, and put your final answer within \\boxed{}'
    # exit()
    HOME_DIR = "/mnt/jfs2/cth/085b13"

    def change_prompt(example, id):
        example["prompt"] = [
            # {"content": sys_prompt_wo_format, "role": "system"},
            {"content": example["prompt"][1]["content"] + inst_after_prompt, "role": "user"}
        ]
        return example

    dataset = dataset.map(function=change_prompt, with_indices=True, num_proc=16)
    print(dataset[0]["prompt"])
    save_dir = f"{HOME_DIR}/_data/processed_dataset_new/luffy_test_beyond8020"
    print(f"Save to {save_dir}/test.parquet")
    dataset.to_parquet(os.path.join(save_dir, f'test.parquet'))