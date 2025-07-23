import os
import datasets
import argparse
from utils.prompts import *
from datasets import concatenate_datasets

HOME_DIR = "/mnt/jfs/tianhao/085b13/cth-dev3"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt_style", type=str, required=True, choices=["dsk_nat", "qwen_math_cot", "luffy"])
    args = parser.parse_args()

    system_prompt = PROMPT_STYLE_MAPPLING["system_prompt"][args.prompt_style]
    user_prompt = PROMPT_STYLE_MAPPLING["user_prompt"][args.prompt_style]

    print(system_prompt)
    print(user_prompt)

    test_file_paths = {
        # 'amc23': f"{HOME_DIR}/_data/raw_dataset/qwen_eval/amc23/test.jsonl", # id, problem, answer
        'amc': f"{HOME_DIR}/_data/raw_dataset/qwen_eval/amc/test_32.jsonl", # id, problem, answer
        # 'aime24': f'{HOME_DIR}/_data/raw_dataset/aime24/aime2024.json',
        # 'aime25': f'{HOME_DIR}/_data/raw_dataset/aime_2025/test.jsonl',  # id, problem, answer=solution, url, year
        'aime24@32': f'{HOME_DIR}/_data/raw_dataset/aime24/aime2024_32.jsonl',
        'aime25@32': f'{HOME_DIR}/_data/raw_dataset/aime_2025/test_32.jsonl',  # id, problem, answer=solution, url, year
        'math500': f'{HOME_DIR}/_data/raw_dataset/math500/math500.json',
        'gpqa_diamond': f'{HOME_DIR}/_data/raw_dataset/gpqa_diamond/gpqa_diamond.json',
        'minerva_math': f'{HOME_DIR}/_data/raw_dataset/qwen_eval/minerva_math/test.jsonl',  # problem, solution (cot with \\boxed, and idx 273 problem
        'olympiadbench': f'{HOME_DIR}/_data/raw_dataset/qwen_eval/olympiadbench/test.jsonl'  # id, question, ["final_answer"][0], 675 problems
    }

    save_dir = f"{HOME_DIR}/_data/processed_dataset_new/bmk_seven_32_{args.prompt_style}/"

    test_sets = dict()
    for test_name, test_path in test_file_paths.items():
        print(test_path)
        test_sets[test_name] = datasets.load_dataset("json", data_files=test_path)['train']
        print(len(test_sets[test_name]))

    def make_map_fn(data_source):

        def _process_fn(example, idx):
            if data_source in ['aime24@32', 'math500', 'gpqa_diamond']:
                question = example["prompt"][0]["value"]
                answer = example["final_answer"]
            elif data_source in ['amc', 'aime25@32']:
                question = example['problem']
                answer = example['answer']
            elif data_source == 'minerva_math':
                question = example['problem']
                answer = example['solution']
            elif data_source == 'olympiadbench':
                question = example['question']
                answer = example['final_answer'][0]
            else:
                print(data_source)
                raise NotImplementedError
            messages = []
            answer = str(answer)
            if system_prompt != None:
                messages.append({"role": "system", "content": system_prompt})
            messages.append(
                {"role": "user", "content": user_prompt.render(prompt=question)}
            )
            data = {
                "data_source": data_source,
                "prompt": messages,
                "reward_model": {
                    "style": "rule",
                    "ground_truth": answer
                },
                "extra_info": {
                    'split': "test",
                    'index': idx
                }
            }
            return data
        return _process_fn

    os.makedirs(save_dir, exist_ok=True)
    for data_source, dataset in test_sets.items():
        test_sets[data_source] = dataset.map(function=make_map_fn(data_source), with_indices=True)
        test_sets[data_source] = test_sets[data_source].select_columns(
            ["data_source", "prompt", "reward_model", "extra_info"]
        )
    merged_dataset = concatenate_datasets([t for t in test_sets.values()])
    print(merged_dataset)
    merged_dataset.to_parquet(os.path.join(save_dir, f'test.parquet'))
