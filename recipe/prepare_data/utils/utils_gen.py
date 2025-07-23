import re
import json
from math_utils import (
    get_answer_str,
    is_equal
)

def generate_sample_batch(
    llm, 
    sampling_params,
    question_list, 
    args
):
    print(sampling_params)
    outputs = llm.generate(question_list, sampling_params=sampling_params, use_tqdm=True)
    completions = []
    for output in outputs:
        if args.is_thinking_model:
            completions.append("<think>\n" + output.outputs[0].text)
        else:
            completions.append(output.outputs[0].text)
    return completions

def write_jsonl_file(file_path, data):
    with open(file_path, 'w', encoding='utf-8') as f:
        for line in data:
            f.write(json.dumps(line, ensure_ascii=False) + '\n')

def read_jsonl_file(file_path):
    results = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            results.append(json.loads(line.strip()))
    return results

def check_format_and_correctness(response, gt, is_thinking_model):
    # for thinking model (distill R1 and QwQ32B), the output should be <think> ... </think> ANSWER
    if is_thinking_model:
        # the model should follow the thinking format
        pattern = r"<think>.*?<\/think>(.*)"
        matches = re.search(pattern, response, re.DOTALL)
        if matches:
            answer = matches.group(1).strip()
        else:
            return False, False, None, None, None
    else:
        answer = response

    # check if the model using the boxed to indicate the final answer
    ans = get_answer_str(answer)  # this func will return the original string if no boxed mathced
    if ans == answer:
        return False, False, answer, ans, None

    return True, is_equal(ans, get_answer_str(gt)), answer, ans, get_answer_str(gt)

def make_hf_chat(question, tokenizer, is_thinking_model, system_prompt=None):
    question = f"A math question: {question}.\nPlease reason step by step, and put your final answer within \boxed{{}}."
    msg = []
    if system_prompt:
        msg.append(system_prompt)
    msg.append({"role": "user", "content": question})
    if is_thinking_model:
        msg.append({"role": "assistant", "content": "<think>\n"})
    chat = tokenizer.apply_chat_template(
        msg, tokenize=False, 
        continue_final_message=is_thinking_model, 
        add_generation_prompt=(not is_thinking_model)
    )
    return chat