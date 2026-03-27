import os
from datasets import load_dataset, concatenate_datasets

if __name__ == "__main__":
    arc_c = load_dataset("parquet", data_files="/data/code/LUFFY/data/valid.arc_c.parquet")['train']
    print(arc_c)
    gpqa = load_dataset("parquet", data_files="/data/code/LUFFY/data/valid.gpqa.parquet")['train']
    print(gpqa)
    mmlu_pro = load_dataset("parquet", data_files="/data/code/LUFFY/data/valid.mmlu_pro.parquet")['train']
    list_of_datasets = [arc_c, gpqa, mmlu_pro]
    dataset = concatenate_datasets(list_of_datasets)
    # dataset = gpqa
    sys_prompt_wo_format = 'Your task is to follow a systematic, thorough reasoning process before providing the final solution. This involves analyzing, summarizing, exploring, reassessing, and refining your thought process through multiple iterations. Structure your response into two sections: Thought and Solution. Each thought should include detailed analysis, brainstorming, verification, and refinement of ideas. In the Solution section, provide the final, logical, and accurate answer, clearly derived from the exploration in the Thought section. Include the answer in \\boxed{} for closed-form results like multiple choices or mathematical solutions'
    # exit()
    HOME_DIR = "/mnt/jfs2/cth/085b13"

    def change_sys_prompt(example, id):
        example["prompt"] = [
            # {"content": sys_prompt_wo_format, "role": "system"},
            {"content": example["prompt"][1]["content"], "role": "user"}
        ]
        return example

    # dataset = dataset.map(function=change_sys_prompt, with_indices=True, num_proc=16)
    # save_dir = f"{HOME_DIR}/_data/processed_dataset_new/luffy_test_ood_sys_wo_format"
    save_dir = f"{HOME_DIR}/_data/processed_dataset_new/luffy_test_ood_raw"
    print(f"Save to {save_dir}/test.parquet")
    print(dataset)
    dataset.to_parquet(os.path.join(save_dir, f'test.parquet'))