import argparse
import json
from collections import defaultdict
from pathlib import Path

import pandas as pd

from prefix_rft_v3.response_normalization import normalize_code_response_list
from verl.utils.import_utils import load_extern_object


def _normalize_responses(value):
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if hasattr(value, "tolist"):
        converted = value.tolist()
        if isinstance(converted, list):
            return converted
        return [converted]
    return [value]


def _mean(values):
    return sum(values) / len(values) if values else 0.0


def _summarize_bucket(items):
    return {
        "count": len(items),
        "pass@1": _mean([item["acc"] for item in items]),
        "formatted_rate": _mean([item["formatted"] for item in items]),
        "mean_reward": _mean([item["score"] for item in items]),
        "valid_ground_truth_rate": _mean([item["valid_ground_truth"] for item in items]),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--response-key", default="responses")
    parser.add_argument("--reasoning-key", default="reasoning_contents")
    parser.add_argument(
        "--reward-path",
        default="pkg://prefix_rft_v3.custom_rewards.code_r1_v2",
    )
    parser.add_argument("--reward-name", default="compute_score")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    reward_fn = load_extern_object(module_path=args.reward_path, object_name=args.reward_name)

    df = pd.read_parquet(args.input_path)
    detail_rows = []
    by_dataset = defaultdict(list)
    by_data_source = defaultdict(list)
    normalized_rows = 0

    for row in df.to_dict(orient="records"):
        responses = _normalize_responses(row.get(args.response_key))
        reasoning_contents = _normalize_responses(row.get(args.reasoning_key))
        responses, changed = normalize_code_response_list(responses, reasoning_contents=reasoning_contents)
        normalized_rows += int(changed)
        if not responses:
            result = {
                "score": -1.1,
                "acc": 0.0,
                "formatted": 0.0,
                "passed": 0.0,
                "valid_ground_truth": 0.0,
            }
        else:
            response = responses[0]
            reward_model = row.get("reward_model") or {}
            extra_info = row.get("extra_info") or {}
            result = reward_fn(
                data_source=row.get("data_source"),
                solution_str=response,
                response_str=response,
                ground_truth=reward_model.get("ground_truth"),
                extra_info=extra_info,
            )

        dataset_name = (row.get("extra_info") or {}).get("dataset", "unknown")
        data_source = row.get("data_source", "unknown")

        detail = {
            "task_id": row.get("task_id") or ((row.get("extra_info") or {}).get("task_id")),
            "entry_point": ((row.get("extra_info") or {}).get("entry_point")),
            "dataset": dataset_name,
            "data_source": data_source,
            "score": float(result.get("score", 0.0)),
            "acc": float(result.get("acc", 0.0)),
            "formatted": float(result.get("formatted", 0.0)),
            "passed": float(result.get("passed", 0.0)),
            "valid_ground_truth": float(result.get("valid_ground_truth", 1.0)),
        }
        detail_rows.append(detail)
        by_dataset[dataset_name].append(detail)
        by_data_source[data_source].append(detail)

    overall = _summarize_bucket(detail_rows)
    by_dataset_summary = {name: _summarize_bucket(items) for name, items in sorted(by_dataset.items())}
    by_data_source_summary = {name: _summarize_bucket(items) for name, items in sorted(by_data_source.items())}

    metrics = {
        "overall": overall,
        "by_dataset": by_dataset_summary,
        "by_data_source": by_data_source_summary,
        "meta": {
            "input_path": str(args.input_path),
            "response_key": args.response_key,
            "reasoning_key": args.reasoning_key,
            "reward_path": args.reward_path,
            "reward_name": args.reward_name,
            "num_rows": len(detail_rows),
            "normalized_rows": normalized_rows,
        },
    }

    detail_df = pd.DataFrame(detail_rows)
    detail_path = output_dir / "metrics_detail.parquet"
    metrics_path = output_dir / "metrics.json"
    detail_df.to_parquet(detail_path, index=False)
    metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")

    print(json.dumps(metrics, indent=2, sort_keys=True))
    print(f"detail_path={detail_path}")
    print(f"metrics_path={metrics_path}")


if __name__ == "__main__":
    main()
