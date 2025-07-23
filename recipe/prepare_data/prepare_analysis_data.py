import os
import uuid
from datasets import load_dataset

if __name__ == "__main__":
    seed, size = 42, 0.1
    HOME_DIR = "/mnt/jfs2/cth/085b13"
    save_dir = f"{HOME_DIR}/_data/processed_dataset_new/luffy_train_subset_{size}_seed_{seed}_sys_wo_format"
    sys_prompt_wo_format = 'Your task is to follow a systematic, thorough reasoning process before providing the final solution. This involves analyzing, summarizing, exploring, reassessing, and refining your thought process through multiple iterations. Structure your response into two sections: Thought and Solution. Each thought should include detailed analysis, brainstorming, verification, and refinement of ideas. In the Solution section, provide the final, logical, and accurate answer, clearly derived from the exploration in the Thought section. Include the answer in \\boxed{} for closed-form results like multiple choices or mathematical solutions'
    # for each run of analysis, we need to run
    # 1. SFT Training on {"prompt", "response"}
    # 2. RL Training on {"prompt", "reward_model"}
    # 3. Prefix-RFT Training on {"prompt", "reward_model", "demos"}
    # 4. inference evaluation using {"prompt", "reward_model"} on each trained checkpoint
    # 5. inference to calculate log_p, entropy, and ppl on {"prompt", "response"} on each trained checkpoint
    dataset = load_dataset("Elliott/Openr1-Math-46k-8192", split="train")
    print(dataset)
    """
    Dataset({
        features: ['data_source', 'prompt', 'target', 'ability', 'reward_model', 'extra_info'],
        num_rows: 45792
    })
    """
    # shuffle the dataset
    dataset = dataset.shuffle(seed=seed)

    # select a subset
    subset_size = int(len(dataset) * size)
    dataset = dataset.select(range(subset_size))

    def change_sys_prompt(example):
        raw_prompt = example["prompt"][1]['content']
        example["prompt"] = [
            {"content": sys_prompt_wo_format, "role": "system"},
            {"content": raw_prompt, "role": "user"}
        ]
        return example
    dataset = dataset.map(function=change_sys_prompt, num_proc=16)

    def add_demos_response_and_uid(example):
        target = example["target"]
        response = target[0]['content']
        demos = [response]
        example["demos"] = demos
        example["response"] = response
        example["uid"] = str(uuid.uuid4())
        return example
    dataset = dataset.map(function=add_demos_response_and_uid, num_proc=16)

    print(dataset)
    print(f"Save to {save_dir}/train.parquet")
    dataset.to_parquet(os.path.join(save_dir, f'train.parquet'))

    
