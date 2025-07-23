import os
import re
import datasets
import argparse
from jinja2 import Template
from utils.prompts import *
from latex2sympy2_extended import NormalizationConfig
from math_verify import LatexExtractionConfig
from recipe.prefix_rft.custom_rewards.math_oat import rfn
from transformers import AutoTokenizer

HOME_DIR = "/mnt/jfs/tianhao/085b13/cth-dev3"

def extract_final_answer_after_think(response):
    pattern = r"<think>.*?<\/think>(.*)"
    matches = re.search(pattern, response, re.DOTALL)
    if matches:
        return matches.group(1).strip()
    else:
        return ""


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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt_style", type=str, default="luffy", choices=["dsk_nat", "qwen_math_cot", "luffy"])  # orz and dsk that includes <answer> tags is decrepted
    parser.add_argument("--rejection_sampling", default=False, action="store_true")  # if rejection sampling, only the data that is parsable and the answer is right is kept
    parser.add_argument("--max_length", default=16384, type=int) # we use qwen tokenizer to measure the token length
    args = parser.parse_args() 

    system_prompt = PROMPT_STYLE_MAPPLING["system_prompt"][args.prompt_style]
    user_prompt = PROMPT_STYLE_MAPPLING["user_prompt"][args.prompt_style]

    print(system_prompt, user_prompt)

    tokenizer = AutoTokenizer.from_pretrained("/mnt/jfs/tianhao/085b13/cth-dev3/_models/Qwen2.5-Math-7B")

    if args.rejection_sampling:
        save_dir = f"{HOME_DIR}/_data/processed_dataset_new/openr1_default_filter_w_oat_rjs_{args.prompt_style}_{args.max_length}"
    else:
        save_dir = f"{HOME_DIR}/_data/processed_dataset_new/openr1_default_filter_w_oat_{args.prompt_style}_{args.max_length}"
    os.makedirs(save_dir, exist_ok=True)


    data_path = f"/mnt/jfs/tianhao/085b13/cth-dev3/_data/raw_dataset/OpenR1-Math-220k"
    dataset = datasets.load_dataset(data_path, "default", num_proc=16)
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

    def make_map_fn():

        def _process_fn(example, idx):
            question = example.pop("problem")
            gt = example.pop("answer")
            generations = example.pop("generations")
            corr_math_verify = example.pop('correctness_math_verify')

            assert len(generations) == len(corr_math_verify)
            
            demos = []
            num_generations = len(generations)
            is_reward_valid = True
            for i in range(num_generations):
                response = generations[i]
                # first do length fitering
                response_len = tokenizer(response, return_tensors='pt')['input_ids'].shape[-1]
                if response_len > args.max_length:
                    # print(response_len)
                    continue
                
                cot = extract_final_answer_after_think(response)
                rfn_output = rfn(response, gt)
                is_correct_my_rfn = rfn_output['score']
                is_correct_dataset = corr_math_verify[i]

                if args.prompt_style == 'dsk_nat' and cot == "":
                    continue
                
                if args.rejection_sampling:
                    if is_correct_my_rfn:
                        demos.append(response)
                    elif is_correct_dataset:
                        pass
                        # print(cot)
                        # print(gt)
                        # import pdb
                        # pdb.set_trace()
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
    dataset = dataset.map(function=make_map_fn(), with_indices=True, num_proc=16)
    dataset = dataset.filter(lambda example: len(example["demos"]) > 0, num_proc=16)
    print(dataset)
    print(f"save to {save_dir}/train.parquet")
    dataset.to_parquet(os.path.join(save_dir, f'train.parquet'))
