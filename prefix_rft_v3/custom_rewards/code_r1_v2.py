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

    if not code_blocks:
        return ""

    runnable_blocks = [code for lang, code in code_blocks if lang in _PYTHON_LANGS]
    if runnable_blocks:
        return runnable_blocks[-1]

    return code_blocks[-1][1]


def _score_code(solution_text, ground_truth, extra_info, timeout, format_reward, answer_reward):
    response_text = solution_text or ""
    if not base._validate_response_structure(response_text):
        return {
            "score": -(answer_reward + format_reward),
            "acc": 0.0,
            "formatted": 0.0,
            "passed": 0.0,
            "valid_ground_truth": 1.0,
        }

    solution_code = _extract_code(response_text)
    if not solution_code:
        return {
            "score": -(answer_reward + format_reward),
            "acc": 0.0,
            "formatted": 0.0,
            "passed": 0.0,
            "valid_ground_truth": 1.0,
        }

    try:
        ground_truth_obj = base._load_ground_truth(ground_truth)
    except Exception:
        return {
            "score": 0.0,
            "acc": 0.0,
            "formatted": 1.0,
            "passed": 0.0,
            "valid_ground_truth": 0.0,
        }

    try:
        if "functional" in ground_truth_obj:
            passed, _ = base._exec_functional(solution_code, ground_truth_obj["functional"], timeout=timeout)
        elif "pytest" in ground_truth_obj:
            passed, _ = base._exec_pytest(solution_code, ground_truth_obj["pytest"], timeout=timeout)
        elif "inputs" in ground_truth_obj and "outputs" in ground_truth_obj:
            passed, _ = base._exec_stdio(solution_code, ground_truth_obj["inputs"], ground_truth_obj["outputs"], timeout=timeout)
        else:
            return {
                "score": 0.0,
                "acc": 0.0,
                "formatted": 1.0,
                "passed": 0.0,
                "valid_ground_truth": 0.0,
            }
    except Exception as exc:
        prompt_text = extra_info.get("prompt") or extra_info.get("question") or ""
        print(f"code reward execution failed: {exc}")
        if prompt_text:
            print(prompt_text)
        passed = False

    reward = format_reward + answer_reward if passed else format_reward
    return {
        "score": reward,
        "acc": float(passed),
        "formatted": 1.0,
        "passed": float(passed),
        "valid_ground_truth": 1.0,
    }


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
        return {"score": float(score), "acc": float(score)}

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
