import os
from datasets import load_dataset
# from recipe.prefix_rft_v2.custom_rewards.math_luffy import rfn
# from recipe.prefix_rft_v2.custom_rewards.math_openr1 import rfn
from custom_rewards.math_oat import rfn_after_think
from math_verify import parse, verify

def labeling_responses(responses: list[str], golden_answer: str):
    predict_answers = list(map(parse, responses))
    golden_answers = list(map(parse, ["$" + golden_answer + "$"] * len(responses)))
    labels = list(map(verify, golden_answers, predict_answers))
    return labels, predict_answers, golden_answers

if __name__ == "__main__":
    dataset = load_dataset("Elliott/Openr1-Math-46k-8192", split="train")
    HOME_DIR = "/mnt/jfs2/cth/085b13"
    # save_dir = f"{HOME_DIR}/_data/processed_dataset_new/luffy_train_unfilter_sys_wo_format"
    save_dir = f"{HOME_DIR}/_data/processed_dataset_new/luffy_train_beyond8020"
    # print(dataset)
    # print(dataset[0])
    # exit()

    sys_prompt_wo_format = 'Your task is to follow a systematic, thorough reasoning process before providing the final solution. This involves analyzing, summarizing, exploring, reassessing, and refining your thought process through multiple iterations. Structure your response into two sections: Thought and Solution. Each thought should include detailed analysis, brainstorming, verification, and refinement of ideas. In the Solution section, provide the final, logical, and accurate answer, clearly derived from the exploration in the Thought section. Include the answer in \\boxed{} for closed-form results like multiple choices or mathematical solutions'
    inst_after_prompt = '\nPlease reason step by step, and put your final answer within \\boxed{}'
 
    def make_map_fn():

        def _process_fn(example, idx):
            target = example["target"]
            assert len(target) == 1
            response = target[0]['content']
            gt = example["reward_model"]['ground_truth']
            # is_correct = rfn_after_think(None, gt, response_str=response)["score"]
            # if is_correct == 0 or is_correct == False:
            #     demos = []
            # else:
            raw_prompt = example["prompt"][1]['content']
            example["prompt"] = [
                # {"content": sys_prompt_wo_format, "role": "system"},
                {"content": raw_prompt + inst_after_prompt, "role": "user"}
            ]
            demos = [target[0]['content']]
            example["demos"] = demos
            return example
        return _process_fn

    dataset = dataset.map(function=make_map_fn(), with_indices=True, num_proc=16)
    dataset = dataset.filter(lambda x: len(x['demos']) != 0)
    print(dataset)
    print(dataset[0]["prompt"])
    print(f"Save to {save_dir}/train.parquet")
    dataset.to_parquet(os.path.join(save_dir, f'train.parquet'))

