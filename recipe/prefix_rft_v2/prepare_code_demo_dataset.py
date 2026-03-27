import argparse
import math
import re
import shutil
from pathlib import Path

import numpy as np
import pandas as pd


VALID_RESPONSE_PATTERN = re.compile(r"<think>.*?</think>\s*<answer>.*?</answer>\s*$", re.DOTALL)
ANSWER_PATTERN = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)
CODE_PATTERN = re.compile(r"```(?:[\w.+-]+)?\s*\n(.*?)```", re.DOTALL)


def is_nan_scalar(value):
    return isinstance(value, float) and math.isnan(value)


def to_python_list(value):
    if value is None or is_nan_scalar(value):
        return []
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return [value]


def validate_response_structure(response_text):
    if not isinstance(response_text, str):
        return False
    return bool(VALID_RESPONSE_PATTERN.match(response_text.strip()))


def extract_answer_block(response_text):
    if not isinstance(response_text, str):
        return ""
    matches = list(ANSWER_PATTERN.finditer(response_text))
    if matches:
        return matches[-1].group(1).strip()
    return response_text


def extract_code(response_text):
    answer_text = extract_answer_block(response_text)
    code_blocks = CODE_PATTERN.findall(answer_text)
    if not code_blocks:
        return ""
    return "\n\n".join(block.strip() for block in code_blocks if block.strip()).strip()


def is_usable_demo(demo):
    if not isinstance(demo, dict):
        return False
    if not bool(demo.get("demo_pass", False)):
        return False
    if demo.get("status") not in (None, "success"):
        return False
    raw_response = demo.get("raw_response", "")
    return validate_response_structure(raw_response) and bool(extract_code(raw_response))


def demo_length(demo):
    token_len = demo.get("raw_response_token_len")
    if isinstance(token_len, (int, np.integer)):
        return int(token_len)
    if isinstance(token_len, float) and not math.isnan(token_len):
        return int(token_len)
    raw_response = demo.get("raw_response", "")
    return len(raw_response) if isinstance(raw_response, str) else 0


def build_selected_demo(row):
    original_demos = [demo for demo in to_python_list(row.get("demonstrations")) if isinstance(demo, dict)]
    if not original_demos:
        fallback_raw = row.get("demo_raw_response")
        if isinstance(fallback_raw, str) and fallback_raw:
            original_demos = [
                {
                    "candidate_index": 0,
                    "demo_code_token_len": row.get("demo_code_token_len"),
                    "demo_pass": bool(row.get("demo_pass", True)),
                    "raw_response": fallback_raw,
                    "raw_response_token_len": row.get("demo_token_len"),
                    "status": row.get("demo_verify_meta", {}).get("status", "success") if isinstance(row.get("demo_verify_meta"), dict) else "success",
                }
            ]

    usable_demos = [demo for demo in original_demos if is_usable_demo(demo)]
    if not usable_demos:
        return None, len(original_demos), 0

    selected_demo = max(
        usable_demos,
        key=lambda demo: (
            demo_length(demo),
            len(demo.get("raw_response", "")),
            -int(demo.get("candidate_index", 0) or 0),
        ),
    )
    return selected_demo, len(original_demos), len(usable_demos)


def line_count(text):
    if not isinstance(text, str) or not text:
        return 0
    return text.count("\n") + 1


def normalize_prompt(prompt):
    if isinstance(prompt, np.ndarray):
        return prompt.tolist()
    return prompt


def normalize_optional(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if is_nan_scalar(value):
        return None
    return value


def build_candidate_summary(selected_demo, demo_code):
    raw_response = selected_demo.get("raw_response", "")
    return {
        "candidate_index": 0,
        "demo_char_len": len(raw_response),
        "demo_code_char_len": len(demo_code),
        "demo_code_line_count": line_count(demo_code),
        "demo_line_count": line_count(raw_response),
        "demo_pass": True,
        "status": "success",
        "original_candidate_index": int(selected_demo.get("candidate_index", 0) or 0),
    }


def process_train_dataframe(df):
    processed_rows = []
    stats = {
        "rows_in": len(df),
        "rows_out": 0,
        "rows_dropped": 0,
        "demo_entries_total": 0,
        "demo_entries_usable": 0,
    }

    for _, row in df.iterrows():
        row_dict = {column: normalize_optional(row[column]) for column in df.columns}
        row_dict["prompt"] = normalize_prompt(row_dict.get("prompt"))

        selected_demo, total_demo_count, usable_demo_count = build_selected_demo(row_dict)
        stats["demo_entries_total"] += total_demo_count
        stats["demo_entries_usable"] += usable_demo_count

        if selected_demo is None:
            stats["rows_dropped"] += 1
            continue

        selected_demo = dict(selected_demo)
        original_candidate_index = int(selected_demo.get("candidate_index", 0) or 0)
        selected_raw_response = selected_demo.get("raw_response", "")
        selected_code = extract_code(selected_raw_response)
        selected_demo_token_len = selected_demo.get("raw_response_token_len")
        selected_demo_code_token_len = selected_demo.get("demo_code_token_len")

        selected_demo["original_candidate_index"] = original_candidate_index
        selected_demo["candidate_index"] = 0
        selected_demo["demo_pass"] = True
        selected_demo["status"] = "success"

        row_dict["demo_pass"] = True
        row_dict["demo_raw_response"] = selected_raw_response
        row_dict["demo_code"] = selected_code
        row_dict["demo_token_len"] = int(selected_demo_token_len) if selected_demo_token_len is not None and not is_nan_scalar(selected_demo_token_len) else len(selected_raw_response)
        row_dict["demo_char_len"] = len(selected_raw_response)
        row_dict["demo_line_count"] = line_count(selected_raw_response)
        row_dict["demo_code_token_len"] = int(selected_demo_code_token_len) if selected_demo_code_token_len is not None and not is_nan_scalar(selected_demo_code_token_len) else len(selected_code)
        row_dict["demo_code_char_len"] = len(selected_code)
        row_dict["demo_code_line_count"] = line_count(selected_code)
        row_dict["demonstrations"] = [selected_demo]
        row_dict["unpassed_demonstrations"] = []
        row_dict["candidate_summaries"] = [build_candidate_summary(selected_demo, selected_code)]
        row_dict["num_candidates"] = 1
        row_dict["num_passed_candidates"] = 1
        row_dict["passed_candidate_index"] = 0
        row_dict["demos"] = [selected_raw_response]
        row_dict["demos_corr"] = [True]
        row_dict["selected_demo_original_candidate_index"] = original_candidate_index
        row_dict["selected_demo_length"] = demo_length(selected_demo)
        row_dict["usable_demo_count"] = usable_demo_count
        row_dict["total_demo_count"] = total_demo_count

        demo_verify_meta = row_dict.get("demo_verify_meta")
        if not isinstance(demo_verify_meta, dict):
            demo_verify_meta = {}
        demo_verify_meta = dict(demo_verify_meta)
        demo_verify_meta["status"] = "success"
        demo_verify_meta["selection_rule"] = "longest_usable_by_current_reward_format"
        demo_verify_meta["usable_demo_count"] = usable_demo_count
        demo_verify_meta["total_demo_count"] = total_demo_count
        row_dict["demo_verify_meta"] = demo_verify_meta

        processed_rows.append(row_dict)

    stats["rows_out"] = len(processed_rows)
    return pd.DataFrame(processed_rows), stats


def process_test_dataframe(df):
    processed = df.copy()
    processed["prompt"] = processed["prompt"].apply(normalize_prompt)
    processed["demos"] = [[] for _ in range(len(processed))]
    processed["demos_corr"] = [[] for _ in range(len(processed))]
    return processed


def safe_write_parquet(df, destination: Path):
    local_tmp = Path("/tmp") / f"{destination.stem}.{destination.suffix.lstrip('.')}.tmp"
    if local_tmp.exists():
        local_tmp.unlink()
    df.to_parquet(local_tmp, index=False)
    if destination.exists():
        destination.unlink()
    shutil.copyfile(local_tmp, destination)
    local_tmp.unlink()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-in", required=True)
    parser.add_argument("--test-in", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading train dataset from {args.train_in}")
    train_df = pd.read_parquet(args.train_in)
    processed_train_df, stats = process_train_dataframe(train_df)

    train_out = out_dir / "train.parquet"
    safe_write_parquet(processed_train_df, train_out)
    print(f"Wrote processed train dataset to {train_out}")
    print("Train processing stats:")
    for key, value in stats.items():
        print(f"  {key}: {value}")

    print(f"Loading test dataset from {args.test_in}")
    test_df = pd.read_parquet(args.test_in)
    processed_test_df = process_test_dataframe(test_df)
    test_out = out_dir / "test.parquet"
    safe_write_parquet(processed_test_df, test_out)
    print(f"Wrote processed test dataset to {test_out}")


if __name__ == "__main__":
    main()
