"""Reward function from OpenR1, this could be utilized for training on oprnr1 dataset"""

import re
import os
import warnings
from latex2sympy2_extended import NormalizationConfig
from math_verify import LatexExtractionConfig, parse, verify
# from concurrent.futures import ThreadPoolExecutor
# executor = ThreadPoolExecutor(max_workers=64)
# from func_timeout import func_timeout, FunctionTimedOut
# import timeout_decorator


def extract_final_answer_after_think(response):
    pattern = r"<think>.*?<\/think>(.*)"
    matches = re.search(pattern, response, re.DOTALL)
    if matches:
        return matches.group(1).strip()
    else:
        return ""

def extract_assistant_response(text: str, gen_start, gen_end):
    pattern = re.escape(gen_start) + r"(.*?)" + re.escape(gen_end)
    matches = re.findall(pattern, text, re.DOTALL)
    if matches:
        return matches[-1].strip()
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

# @timeout_decorator.timeout(3, use_signals=False)
def math_verify(gold_parsed, answer_parsed):
    return verify(gold=gold_parsed, target=answer_parsed)

def rfn_after_think_extract(solution_str: str, ground_truth: str, **kwargs):

    ground_truth = f"The final answer is ${ground_truth}$"  # TODO 这里需要确定一下如何让模型更好的parse出来，需要filterout不能parse出来的example
    gold_parsed = parse(
        ground_truth,
        extraction_mode="first_match",
        # extraction_config=[LatexExtractionConfig()],
    )
    
    GENERATION_START = os.getenv("GENERATION_START", None)
    GENERATION_END = os.getenv("GENERATION_END", None)
    assert GENERATION_START is not None and GENERATION_END is not None
    if GENERATION_START is None:
        print("Please set GENERATION_START as the assistant template used, set to default")
        GENERATION_START = "<|im_start|>assistant"
    if GENERATION_END is None:
        print("Please set GENERATION_START as the assistant template used, set to default")
        GENERATION_END = "<|endoftext|>"
    
    generation = extract_assistant_response(solution_str, gen_start=GENERATION_START, gen_end=GENERATION_END)
    generation = extract_final_answer_after_think(generation)
    generation = generation

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
            reward = 0.0
    else:
        # if the gold solution is not parseable
        reward = 0.0
        answer_parsed = None
        print("Failed to parse gold solution")
    return {
        "score": reward,
        "acc": reward
    }


def rfn_after_think(solution_str: str, ground_truth: str, **kwargs):

    ground_truth = f"The final answer is ${ground_truth}$"  # TODO 这里需要确定一下如何让模型更好的parse出来，需要filterout不能parse出来的example
    gold_parsed = parse(
        ground_truth,
        extraction_mode="first_match",
        # extraction_config=[LatexExtractionConfig()],
    )
    
    # GENERATION_START = os.getenv("GENERATION_START", None)
    # GENERATION_END = os.getenv("GENERATION_END", None)
    # assert GENERATION_START is not None and GENERATION_END is not None
    # if GENERATION_START is None:
    #     print("Please set GENERATION_START as the assistant template used, set to default")
    #     GENERATION_START = "<|im_start|>assistant"
    # if GENERATION_END is None:
    #     print("Please set GENERATION_START as the assistant template used, set to default")
    #     GENERATION_END = "<|endoftext|>"
    
    # generation = extract_assistant_response(solution_str, gen_start=GENERATION_START, gen_end=GENERATION_END)
    generation = kwargs['response_str']
    generation = extract_final_answer_after_think(generation)
    generation = generation

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
            reward = 0.0
    else:
        # if the gold solution is not parseable
        reward = 0.0
        answer_parsed = None
        print("Failed to parse gold solution")
    return {
        "score": reward,
        "acc": reward
    }

def rfn_after_think_step(solution_str: str, ground_truth: str, **kwargs):

    ground_truth = f"The final answer is ${ground_truth}$"  # TODO 这里需要确定一下如何让模型更好的parse出来，需要filterout不能parse出来的example
    gold_parsed = parse(
        ground_truth,
        extraction_mode="first_match",
        # extraction_config=[LatexExtractionConfig()],
    )
    
    generation = kwargs['response_str']
    cur_step = kwargs.get("cur_step", 0)
    extract_after_step_n = kwargs.get("extract_after_step_n", 0)
    print(f"The cur step is: {cur_step}, the thereshold step is: {extract_after_step_n}")
    if cur_step >= extract_after_step_n:
        generation = extract_final_answer_after_think(generation)
    generation = generation

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
            reward = 0.0
    else:
        # if the gold solution is not parseable
        reward = 0.0
        answer_parsed = None
        print("Failed to parse gold solution")
    return {
        "score": reward,
        "acc": reward
    }


def rfn(solution_str: str, ground_truth: str, **kwargs):
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
            reward = 0.0
    else:
        # if the gold solution is not parseable
        reward = 0.0
        answer_parsed = None
        print("Failed to parse gold solution")
    return {
        "score": reward,
        "acc": reward
    }

def rfn_val(solution_str: str, ground_truth: str, **kwargs):
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
            reward = 0.0
    else:
        # if the gold solution is not parseable
        reward = 0.0
        answer_parsed = None
        print("Failed to parse gold solution")
    return {
        "score": reward,
        "acc": reward
    }
