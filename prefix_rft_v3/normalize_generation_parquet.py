import argparse

import pandas as pd

from prefix_rft_v3.response_normalization import normalize_code_response_list


def _normalize_cell(value):
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-path", required=True)
    parser.add_argument("--output-path")
    parser.add_argument("--response-key", default="responses")
    parser.add_argument("--reasoning-key", default="reasoning_contents")
    parser.add_argument("--raw-response-key", default="responses_raw")
    args = parser.parse_args()

    output_path = args.output_path or args.input_path

    df = pd.read_parquet(args.input_path)
    if args.raw_response_key not in df.columns:
        df[args.raw_response_key] = df[args.response_key]

    reasoning_series = df[args.reasoning_key] if args.reasoning_key in df.columns else None

    normalized_responses = []
    changed_rows = 0
    for idx, raw_responses in enumerate(df[args.response_key].tolist()):
        responses = _normalize_cell(raw_responses)
        reasoning_contents = None
        if reasoning_series is not None:
            reasoning_contents = _normalize_cell(reasoning_series.iloc[idx])
        normalized, changed = normalize_code_response_list(responses, reasoning_contents=reasoning_contents)
        normalized_responses.append(normalized)
        changed_rows += int(changed)

    df[args.response_key] = normalized_responses
    df.to_parquet(output_path, index=False)

    print(f"normalized_rows={changed_rows}")
    print(f"output_path={output_path}")


if __name__ == "__main__":
    main()
