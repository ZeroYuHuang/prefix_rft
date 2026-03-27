import os
import re
import datasets
import argparse
from jinja2 import Template
from utils.prompts import *
from latex2sympy2_extended import NormalizationConfig
from math_verify import LatexExtractionConfig, parse, verify

HOME_DIR = "/mnt/jfs/tianhao/085b13/cth-dev3"

def extract_final_answer_after_think(response):
    pattern = r"<think>.*?<\/think>(.*)"
    matches = re.search(pattern, response, re.DOTALL)
    if matches:
        return matches.group(1).strip()
    else:
        return ""

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
    )
    
    if extract:
        generation = extract_final_answer_after_think(solution_str)
    else:
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
            reward = float(verify(gold_parsed, answer_parsed))
        except Exception as e:
            print(f"verify failed: {e}")
            reward = 0.0
    else:
        # if the gold solution is not parseable
        reward = 0.0
        answer_parsed = None
        print("Failed to parse gold solution")
    return reward

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt_style", type=str, default="dsk_nat", choices=["dsk_nat", "qwen_math_cot"])  # orz and dsk that includes <answer> tags is decrepted
    parser.add_argument("--rejection_sampling", default=False, action="store_true")  # if rejection sampling, only the data that is parsable and the answer is right is kept
    args = parser.parse_args() 

    system_prompt = PROMPT_STYLE_MAPPLING["system_prompt"][args.prompt_style]
    user_prompt = PROMPT_STYLE_MAPPLING["user_prompt"][args.prompt_style]

    if args.rejection_sampling:
        save_dir = f"{HOME_DIR}/_data/processed_dataset_new/openr1_default_filter_w_or1_rjs_{args.prompt_style}"
    else:
        save_dir = f"{HOME_DIR}/_data/processed_dataset_new/openr1_default_filter_w_or1_{args.prompt_style}"
    os.makedirs(save_dir, exist_ok=True)


    data_path = f"/mnt/jfs/tianhao/085b13/cth-dev3/_data/raw_dataset/OpenR1-Math-220k"
    dataset = datasets.load_dataset(data_path, "default")
    dataset = dataset['train']
    print(dataset)
    # DatasetDict({
    #     train: Dataset({
    #         features: [
    #               'problem', 'solution', 'answer', 'problem_type', 'question_type', 
    #               'source', 'uuid', 'is_reasoning_complete', 'generations', 'correctness_math_verify', 'correctness_llama', 
    #               'finish_reasons', 'correctness_count', 'messages'],
    #         num_rows: 93733
    #     })
    # })

    def parse_gold(gt):
        gt = f"The final answer is ${gt}$"
        gold_parsed = parse(gt, extraction_mode="first_match")
        return len(gold_parsed) != 0
    
    dataset = dataset.filter(lambda example: parse_gold(example["answer"]), num_proc=16)
    print(len(dataset))

    def make_map_fn():

        def _process_fn(example, idx):
            question = example.pop("problem")
            gt = example.pop("answer")
            generations = example.pop("generations")
            
            demos = []
            num_generations = len(generations)
            is_reward_valid = True
            for i in range(num_generations):
                response = generations[i]
                cot = extract_final_answer_after_think(response)
                is_correct = or1_rwd_func(response, gt, extract=args.prompt_style=='dsk_nat')

                if args.prompt_style == 'dsk_nat' and cot == "":
                    continue
                
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

            data = {
                "data_source": example.pop("source"),
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
                'is_reward_valid': is_reward_valid
            }
            return data

        return _process_fn
    dataset = dataset.map(function=make_map_fn(), with_indices=True, num_proc=64)
    dataset = dataset.filter(lambda example: len(example["demos"]) > 0, num_proc=16)
    print(dataset)
    print(f"save to {save_dir}/train.parquet")
    # dataset.to_parquet(os.path.join(save_dir, f'train.parquet'))
