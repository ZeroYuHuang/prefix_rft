import re


_THINK_BLOCK_PATTERN = re.compile(r"<think>(.*?)</think>", re.DOTALL)
_ANSWER_BLOCK_PATTERN = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)


def _clean_block(text):
    return (text or "").strip()


def _wrap_reasoning_answer(reasoning_text, answer_text):
    reasoning = _clean_block(reasoning_text)
    answer = _clean_block(answer_text)
    return f"<think>\n{reasoning}\n</think>\n<answer>\n{answer}\n</answer>"


def normalize_code_response_text(response_text, reasoning_text=None):
    text = _clean_block(response_text)
    reasoning = _clean_block(reasoning_text)

    if not text and not reasoning:
        return text, False

    think_blocks = [match.group(1) for match in _THINK_BLOCK_PATTERN.finditer(text)]
    answer_blocks = [match.group(1) for match in _ANSWER_BLOCK_PATTERN.finditer(text)]

    if think_blocks and answer_blocks:
        normalized = _wrap_reasoning_answer(think_blocks[-1], answer_blocks[-1])
        return normalized, normalized != text

    if answer_blocks:
        normalized = _wrap_reasoning_answer(reasoning, answer_blocks[-1])
        return normalized, normalized != text

    if text:
        normalized = _wrap_reasoning_answer(reasoning, text)
        return normalized, normalized != text

    normalized = _wrap_reasoning_answer(reasoning, "")
    return normalized, True


def normalize_code_response_list(responses, reasoning_contents=None):
    normalized = []
    changed = False

    if reasoning_contents is None:
        reasoning_contents = []

    reasoning_contents = list(reasoning_contents)
    if len(reasoning_contents) < len(responses):
        reasoning_contents = reasoning_contents + [None] * (len(responses) - len(reasoning_contents))
    elif len(reasoning_contents) > len(responses):
        reasoning_contents = reasoning_contents[: len(responses)]

    for response_text, reasoning_text in zip(responses, reasoning_contents):
        normalized_text, item_changed = normalize_code_response_text(
            response_text=response_text,
            reasoning_text=reasoning_text,
        )
        normalized.append(normalized_text)
        changed = changed or item_changed

    return normalized, changed
