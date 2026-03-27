from jinja2 import Template
# orz style prompt
ORZ_SYSTEM_PROMPT = """\
A conversation between User and Assistant. The User asks a question, and the Assistant solves it. The Assistant first thinks about the reasoning process in the mind and then provides the User with the answer. \
The reasoning process is enclosed within <think> </think> and answer is enclosed within <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>."""

ORZ_USER_PROMPT = """\
You must put your answer inside <answer> </answer> tags, i.e., <answer> answer here </answer>. And your final answer will be extracted automatically by the \\boxed{} tag.
This is the problem:
{{prompt}}
"""

# Deepseek r1 style prompt
DSK_SYSTEM_PROMPT = """\
A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The Assistant first thinks about the reasoning process in the mind and then provides the user with the answer. \
The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>."""

DSK_USER_PROMPT = """{{prompt}}\n Please reason step by step, and put your final answer within \\boxed{}"""

# Deepseek r1 style prompt
DSK_NO_ANSWER_TAG_SYSTEM_PROMPT = """\
A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The Assistant first thinks about the reasoning process in the mind and then provides the user with the answer. \
The reasoning process should be enclosed within <think> </think> followed by your answer, i.e., <think> reasoning process here </think>answer here."""

DSK_USER_PROMPT = """{{prompt}}\n Please reason step by step, and put your final answer within \\boxed{}"""

# simple CoT style prompt
COT_USER_PROMPT = """{{prompt}}\nPlease reason step by step, and put your final answer within \\boxed{}"""

# Qwen math cot Prompt
QWEN_MATH_SYS_PROMPT = """Please reason step by step, and put your final answer within \\boxed{}"""
QWEN_MATH_USER_PROMPT = """{{prompt}}"""

# LUFFY system prompt
LUFFY_SYS_PROMPT = """Your task is to follow a systematic, thorough reasoning process before providing the final solution. \
    This involves analyzing, summarizing, exploring, reassessing, and refining your thought process through multiple iterations. \
    Structure your response into two sections: Thought and Solution. In the Thought section, present your reasoning using the format: “<think>\n thoughts </think>\n”. \
    Each thought should include detailed analysis, brainstorming, verification, and refinement of ideas. \
    After “</think>\n” in the Solution section, provide the final, logical, and accurate answer, clearly derived from the exploration in the Thought section. \
    If applicable, include the answer in \boxed{} for closed-form results like multiple choices or mathematical solutions."""
LUFFTY_USER_PROMPT = """{{prompt}}"""

PROMPT_STYLE_MAPPLING = {
    "system_prompt": {
        "orz": ORZ_SYSTEM_PROMPT,
        "dsk": DSK_SYSTEM_PROMPT,
        "dsk_nat": DSK_NO_ANSWER_TAG_SYSTEM_PROMPT,
        "cot": None,
        "qwen_math_cot": QWEN_MATH_SYS_PROMPT,
        "luffy": LUFFY_SYS_PROMPT
    },
    "user_prompt": {
        "orz": Template(ORZ_USER_PROMPT),
        "dsk": Template(DSK_USER_PROMPT),
        "dsk_nat": Template(DSK_USER_PROMPT),
        "cot": Template(COT_USER_PROMPT),
        "qwen_math_cot": Template(QWEN_MATH_USER_PROMPT),
        "luffy": Template(LUFFTY_USER_PROMPT)
    }
}