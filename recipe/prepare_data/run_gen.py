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

"""
DeepSeek-R1-Distill-Qwen-1.5B
DeepSeek-R1-Distill-Qwen-7B
DeepSeek-R1-Distill-Qwen-32B
QwQ-32B
# LUFFY-Qwen-Math-7B-Zero
# Qwen2.5-Math-7B-Oat-Zero    
"""


# helper functions

def apply_template_oat_zero(question):
    return (
            "<|im_start|>system\nPlease reason step by step, and put your final answer within \\boxed{}.<|im_end|>\n<|im_start|>user\n"
            + question
            + "<|im_end|>\n<|im_start|>assistant\n"
        )

def apply_template_distill_r1(question, tokenizer):
    # print(question)
    question =  question + "\nPlease reason step by step, and put your final answer within \\boxed{}\n"
    messages = [{"role": "user", "content": question}]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

def apply_template_qwq_32b(question, tokenizer):
    # print(question)
    question =  question + "\nPlease reason step by step, and put your final answer within \\boxed{}\n"
    messages = [{"role": "user", "content": question}]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

def apply_template_qwen_math_base_STaR(question, tokenizer, ans):
    question = question + f"The answer to this question: {ans}. Based on the provided answer, Please provide the initial step towards resolving the question. This step may serve as a foundation but might not encompass the entire solution."
    messages = [{"role": "user", "content": question}]
    return tokenizer.apply_chat_template(messages)


def load_math_lvl3to5_8k():
    data_set = load_from_disk("/mnt/jfs/tianhao/085b13/cth-dev3/_data/raw_dataset/math_lvl3to5_8k/train")
    data_set = data_set.remove_columns(['input', 'answer', 'question', 'ground_truth_answer', 'target'])
    """
    Dataset({
        features: ['gt_answer', 'subject', 'level', 'problem'],
        num_rows: 8523
    })
    这个answer, gt_answer, ground_truth_answer, target全都一样
    """
    problems = [d for d in data_set]  # return a list
    return problems, "problem"


def load_luffy_train():
    data_set = load_dataset(
        "parquet", 
        data_files="/mnt/jfs2/cth/085b13/_data/processed_dataset_new/luffy_train_openr1/train.parquet"
    )['train']
    """
    Dataset({
        features: ['data_source', 'prompt', 'target', 'ability', 'reward_model', 'extra_info', 'demos'],
        num_rows: 45714
    })
    """
    def extract_raw_problems(example):
        example['problem'] = example['prompt'][1]["content"]
        return example
    data_set = data_set.map(extract_raw_problems, num_proc=16, desc="Extracting problems from messages", remove_columns=['demos', 'target'])
    print(data_set)
    problems = [d for d in data_set]  # return a list
    return problems, "problem"

# load datasets
def load_data(dataset_name: str):
    if dataset_name == 'math_lvl3to5_8k':
        return load_math_lvl3to5_8k()
    if dataset_name == 'luffy_train':
        return load_luffy_train()
    else:
        raise NotImplementedError

# apply_chat_template
def get_template_fn(model_name: str, tokenizer):
    if model_name == 'Qwen2.5-Math-7B-Oat-Zero':
        return apply_template_oat_zero
    if "DeepSeek-R1-Distill" in model_name:
        return partial(apply_template_distill_r1, tokenizer=tokenizer)
    if model_name == "QwQ-32B":
        return partial(apply_template_qwq_32b, tokenizer)


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

    model_name = args.model.split("/")[-1]
    output_dir = os.path.join(f"/mnt/jfs2/cth/085b13/_data/raw_dataset/distillation/{args.dataset}/{model_name}")
    
    
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"data_split_{args.split_start}_{args.split_end}_{str(uuid.uuid4())}" + ".jsonl")
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
        n=1,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        seed=args.seed
    )
    
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    template_fn = get_template_fn(model_name=model_name, tokenizer=tokenizer)
    templated_problems = [template_fn(p[problem_key]) for p in problems]
    completions = generate(llm, sampling_params, templated_problems, args)
    with open(output_file, 'w', encoding='utf-8') as f:
        for p, completion in zip(problems, completions):
            p['demos'] = ["<think>\n" + completion]
            line = p
            f.write(json.dumps(line, ensure_ascii=False) + '\n')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", "-d", type=str, default="")
    parser.add_argument("--model", "-m", type=str, default="")
    parser.add_argument("--split_start", "-s", type=float, default=0.0)
    parser.add_argument("--split_end", "-e", type=float, default=1.0)
    
    # sampling parameters
    parser.add_argument("--temperature", "-t", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--max_tokens", type=int, default=8192)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    for k, v in vars(args).items():
        print(f"{k}: {v}")
    main(args)