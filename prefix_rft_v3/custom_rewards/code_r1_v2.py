import re

from prefix_rft_v3.custom_rewards import code_r1 as base

_PYTHON_LANGS = {"", "py", "python", "python3"}
_CODE_BLOCK_PATTERN = re.compile(r"```(?P<lang>[\w.+-]+)?[ \t]*\n(?P<code>.*?)```", re.DOTALL)


def _extract_code(response_text):
    answer_text = base._extract_answer_block(response_text)
    code_blocks = []
    for match in _CODE_BLOCK_PATTERN.finditer(answer_text):
        lang = (match.group("lang") or "").strip().lower()
        code = match.group("code").strip()
        if code:
            code_blocks.append((lang, code))

    if not code_blocks and answer_text != (response_text or ""):
        for match in _CODE_BLOCK_PATTERN.finditer(response_text or ""):
            lang = (match.group("lang") or "").strip().lower()
            code = match.group("code").strip()
            if code:
                code_blocks.append((lang, code))

    if not code_blocks:
        return ""

    runnable_blocks = [code for lang, code in code_blocks if lang in _PYTHON_LANGS]
    if runnable_blocks:
        return runnable_blocks[-1]

    return code_blocks[-1][1]


def _score_code(solution_text, ground_truth, extra_info, timeout, format_reward, answer_reward):
    return base._score_code_with_extractor(
        solution_text=solution_text,
        ground_truth=ground_truth,
        extra_info=extra_info,
        timeout=timeout,
        format_reward=format_reward,
        answer_reward=answer_reward,
        extract_code_fn=_extract_code,
    )


def rfn(
    solution_str: str,
    ground_truth,
    data_source=None,
    response_str=None,
    extra_info=None,
    timeout=base._DEFAULT_TIMEOUT_SECONDS,
    format_reward=base._FORMAT_REWARD,
    answer_reward=base._ANSWER_REWARD,
    **kwargs,
):
    if not base._is_code_source(data_source):
        score = base._default_compute_score(
            data_source=data_source,
            solution_str=solution_str,
            ground_truth=ground_truth,
            extra_info=extra_info,
        )
        return base._build_non_code_result(score)

    return _score_code(
        solution_text=response_str or solution_str,
        ground_truth=ground_truth,
        extra_info=base._normalize_extra_info(extra_info),
        timeout=timeout,
        format_reward=format_reward,
        answer_reward=answer_reward,
    )


def rfn_val(*args, **kwargs):
    return rfn(*args, **kwargs)


def compute_score(*args, **kwargs):
    return rfn(*args, **kwargs)


def compute_score_val(*args, **kwargs):
    return rfn_val(*args, **kwargs)
