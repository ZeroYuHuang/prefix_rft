import os
from datasets import load_dataset
from transformers import AutoTokenizer

if __name__ == "__main__":
    dataset = load_dataset("Elliott/Openr1-Math-46k-8192", split="train")
    HOME_DIR = "/mnt/jfs2/cth/085b13"
    save_dir = f"{HOME_DIR}/_data/processed_dataset_new/luffy_train_easy_sys_wo_format"
    tok = AutoTokenizer.from_pretrained(f"{HOME_DIR}/_models/Llama-3.1-8B")

    sys_prompt_wo_format = 'Your task is to follow a systematic, thorough reasoning process before providing the final solution. This involves analyzing, summarizing, exploring, reassessing, and refining your thought process through multiple iterations. Structure your response into two sections: Thought and Solution. Each thought should include detailed analysis, brainstorming, verification, and refinement of ideas. In the Solution section, provide the final, logical, and accurate answer, clearly derived from the exploration in the Thought section. Include the answer in \\boxed{} for closed-form results like multiple choices or mathematical solutions'

    def process_fn(example, idx):
        target = example["target"]
        response = target[0]['content']
        raw_prompt = example["prompt"][1]['content']
        example["prompt"] = [
            {"content": sys_prompt_wo_format, "role": "system"},
            {"content": raw_prompt, "role": "user"}
        ]
        demos = [target[0]['content']]
        example["demos"] = demos
        example["demos_len"] = [len(tok.tokenize(d)) for d in demos]
        return example

    def filter_by_demos_len(example):
        demos_len = example["demos_len"]
        for dl in demos_len:
            if dl > 2048:
                return False
        return True
            

    dataset = dataset.map(function=process_fn, with_indices=True, num_proc=16)
    dataset = dataset.filter(filter_by_demos_len)
    print(dataset)
    # print(dataset[0])
    print(f"Save to {save_dir}/train.parquet")
    dataset.to_parquet(os.path.join(save_dir, f'train.parquet'))
