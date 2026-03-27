import os
import json
import re
import datasets
import argparse

from datasets import load_dataset, Dataset
from latex2sympy2_extended import NormalizationConfig
from math_verify import LatexExtractionConfig, parse, verify
from utils.prompts import *

def math_verify(gold_parsed, answer_parsed):
    return verify(gold=gold_parsed, target=answer_parsed)

EXTRACTION_CONFIG = [
    LatexExtractionConfig(
        normalization_config=NormalizationConfig(
            nits=False,
            malformed_operators=False,
            basic_latex=True,
            equations=True,
            boxed="all",
            units=True,
        ),
        # Ensures that boxed is tried first
        boxed_match_priority=0,
        try_extract_without_anchor=False,
    )
]

def or1_rwd_func(solution_str, ground_truth, extract=True):
    ground_truth = f"The final answer is ${ground_truth}$"  # TODO 这里需要确定一下如何让模型更好的parse出来，需要filterout不能parse出来的example
    gold_parsed = parse(
        ground_truth,
        extraction_mode="first_match",
        # extraction_config=[LatexExtractionConfig()],
    )
    generation = solution_str

    if len(gold_parsed) != 0:
        # We calculate the answer to be provided in correct latex (no malformed operators)
        answer_parsed = parse(
            generation,
            extraction_config=EXTRACTION_CONFIG,
            extraction_mode="first_match",
        )
        # Reward 1 if the content is the same as the ground truth, 0 otherwise
        try:
            reward = math_verify(gold_parsed, answer_parsed)
            reward = float(reward)
        except Exception as e:
            print(f"verify failed: {e}")
            print(gold_parsed, answer_parsed)
            reward = 0.0
    else:
        # if the gold solution is not parseable
        reward = 0.0
        answer_parsed = None
        print("Failed to parse gold solution")
    return reward


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt_style", type=str, default="qwen_math_cot")
    parser.add_argument("--rejection_sampling", default=False, action="store_true")
    args = parser.parse_args()

    system_prompt = PROMPT_STYLE_MAPPLING["system_prompt"][args.prompt_style]
    user_prompt = PROMPT_STYLE_MAPPLING["user_prompt"][args.prompt_style]
    print(system_prompt, user_prompt)

    if args.rejection_sampling:
        save_dir = f"/mnt/jfs/tianhao/085b13/cth-dev3/_data/processed_dataset_new/math_lvl3to5_8k_rjs_{args.prompt_style}"
    else:
        save_dir = f"/mnt/jfs/tianhao/085b13/cth-dev3/_data/processed_dataset_new/math_lvl3to5_8k_{args.prompt_style}"
    os.makedirs(save_dir, exist_ok=True)

    data_files = f"/mnt/jfs/tianhao/085b13/cth-dev3/_data/raw_dataset/math_lvl3to5_8k_Qwen2.5-Math-7B-Oat-Zero.jsonl"
    # dataset = datasets.load_dataset('json', data_files=data_files)
    with open(data_files, 'r', encoding='utf-8') as f:
        dataset = [json.loads(line.strip()) for line in f]
    dataset = Dataset.from_list(dataset)
    # Dataset({                                                                                                                                                                                                                                                                                  
    #     features: ['gt_answer', 'subject', 'level', 'problem', 'idx', 'model_solution', 'answers'],                                                                                                                                                                                            
    #     num_rows: 8522                                                                                                                                                                                                                                                                         
    # })  
    print(dataset)
    print(len(dataset[0]["model_solution"]))
    def parse_gold(example):
        gt = example['gt_answer']
        gt = f"The final answer is ${gt}$"
        gold_parsed = parse(gt, extraction_mode="first_match")
        return len(gold_parsed) != 0

    # make sure the gold is parsable
    dataset = dataset.filter(lambda example: parse_gold(example), num_proc=16)
    print(dataset)

    num_no_demo_data = 0

    def make_map_fn():

        def _process_fn(example, idx):
            global num_no_demo_data
            question = example.pop("problem")
            gt = example.pop("gt_answer")
            generations = example.pop("model_solution")

            demos = []
            for response in generations:
                is_correct = or1_rwd_func(response, gt, extract=args.prompt_style=='dsk_nat')
                if args.rejection_sampling:
                    if is_correct:
                        demos.append(response)
                else:
                    demos.append(response)
            messages = []
            if system_prompt != None:
                messages.append({"role": "system", "content": system_prompt})
            messages.append(
                {"role": "user", "content": user_prompt.render(prompt=question)}
            )

            if len(demos) == 0:
                num_no_demo_data += 1
            else:
                num_no_demo_data = 0
            data = {
                "data_source": "math",
                "prompt": messages,
                "demos": demos,
                "ability": "math",
                "reward_model": {
                    "style": "rule",
                    "ground_truth": gt  # This is the origianl solution, need to parse
                },
                "extra_info": {
                    'split': "openr1",
                    'index': idx
                },
            }
            return data

        return _process_fn

    dataset = dataset.map(function=make_map_fn(), with_indices=True, num_proc=16)
    # dataset = dataset.filter(lambda example: len(example["demos"]) > 0, num_proc=16)
    print(dataset)
    print(f"{num_no_demo_data} donot have demonstration data.")
    print(f"save to {save_dir}/train.parquet")
    dataset.to_parquet(os.path.join(save_dir, f'train.parquet'))
