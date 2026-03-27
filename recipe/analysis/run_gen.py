import os
import math
import json
import torch
import argparse
import uuid

from vllm import LLM, SamplingParams
from datasets import load_from_disk, load_dataset

from transformers import AutoTokenizer

from functools import partial

os.environ["NCCL_IGNORE_DISABLED_P2P"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "true"

# load datasets
def load_data(data_files: str):
    dataset = load_dataset("parquet", data_files=data_files)["train"]
    dataset.remove_columns(['demos', 'num_demo', 'demos_len', 'difficulty', 'old_demos', 'old_demos_len', 'old_correctness', 'response'])
    dataset = dataset.shuffle(seed=42)
    problems = [dataset[i] for i in range(len(dataset))]
    return problems, "prompt"


def generate(
    llm, sampling_params,
    question_list, args
):
    outputs = llm.generate(question_list, sampling_params=sampling_params, use_tqdm=True)
    completions = [output.outputs[0].text for output in outputs]
    return completions


def main(args):
    # read the data
    problems, problem_key = load_data(args.dataset)
    start_idx = math.ceil(len(problems) * args.split_start)
    end_idx = math.ceil(len(problems) * args.split_end)
    problems = problems[start_idx: end_idx]
    # for idx, p in enumerate(problems):
    #     p['idx'] = idx

    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"{args.seed}.jsonl")
    print("Problems Loaded")
    print(f"Generated results will save to {output_file}")

    llm = LLM(
        model=args.model,
        trust_remote_code=True,
        tensor_parallel_size=torch.cuda.device_count(),
        gpu_memory_utilization=0.9,
        swap_space=16,
        enforce_eager=True
    )
    sampling_params = SamplingParams(
        n=1, temperature=args.temperature,
        max_tokens=args.max_tokens, seed=args.seed
    )
    
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    templated_problems = [
        tokenizer.apply_chat_template(p[problem_key], add_generation_prompt=True, tokenize=False) 
        for p in problems
    ]
    print(len(templated_problems))
    completions = generate(llm, sampling_params, templated_problems, args)
    assert len(problems) == len(completions)
    with open(output_file, 'w', encoding='utf-8') as f:
        for p, completion in zip(problems, completions):
            p['analysis_gen'] = [completion]
            line = p
            keys_to_remove = ['demos', 'num_demo', 'demos_len', 'old_demos', 'old_demos_len', 'old_correctness', 'response']
            for ktr in keys_to_remove:
                p.pop(ktr)
            # dict_keys(
            # ['data_source', 'prompt', 'ability', 'reward_model', 'extra_info', 
            # 'problem', 'demos', 'num_demo', 'demos_len', 'difficulty', 'old_demos', 
            # 'old_demos_len', 'old_correctness', 'response', 'uid', 'analysis_gen'])
            f.write(json.dumps(line, ensure_ascii=False) + '\n')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", "-d", type=str, default="")
    parser.add_argument("--model", "-m", type=str, default="")
    parser.add_argument("--split_start", "-s", type=float, default=0.0)
    parser.add_argument("--split_end", "-e", type=float, default=1.0)
    parser.add_argument("--output_dir", "-o", type=str, required=True)
    
    # sampling parameters
    parser.add_argument("--temperature", "-t", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--max_tokens", type=int, default=8192)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    for k, v in vars(args).items():
        print(f"{k}: {v}")
    main(args)