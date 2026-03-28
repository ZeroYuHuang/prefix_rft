import argparse
import random
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


def _build_messages(prompt, demo_raw_response):
    messages = list(prompt or [])
    messages.append({"role": "assistant", "content": demo_raw_response})
    return messages


def _row_schema():
    message_struct = pa.struct([("content", pa.string()), ("role", pa.string())])
    extra_info_struct = pa.struct(
        [
            ("dataset", pa.string()),
            ("index", pa.int64()),
            ("task_id", pa.string()),
            ("entry_point", pa.string()),
            ("prompt", pa.string()),
            ("reference", pa.string()),
            ("split", pa.string()),
        ]
    )
    return pa.schema(
        [
            ("messages", pa.list_(message_struct)),
            ("data_source", pa.string()),
            ("task_id", pa.string()),
            ("entry_point", pa.string()),
            ("extra_info", extra_info_struct),
        ]
    )


def _to_sft_rows(rows):
    sft_rows = []
    for row in rows:
        demo_raw_response = row.get("demo_raw_response")
        prompt = row.get("prompt")
        if not demo_raw_response or not prompt:
            continue
        sft_rows.append(
            {
                "messages": _build_messages(prompt, demo_raw_response),
                "data_source": row.get("data_source"),
                "task_id": row.get("task_id"),
                "entry_point": row.get("entry_point"),
                "extra_info": row.get("extra_info"),
            }
        )
    return sft_rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-path",
        default="/mnt/hdfs/haruna/home/byte_data_seed/hdd_hldy/user/zeyuhuang/datasets/prefix-rft-code/train.parquet",
    )
    parser.add_argument(
        "--train-output-path",
        default="/mnt/hdfs/haruna/home/byte_data_seed/hdd_hldy/user/zeyuhuang/datasets/prefix-rft-code/sft_train.parquet",
    )
    parser.add_argument(
        "--dev-output-path",
        default="/mnt/hdfs/haruna/home/byte_data_seed/hdd_hldy/user/zeyuhuang/datasets/prefix-rft-code/sft_dev.parquet",
    )
    parser.add_argument("--dev-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    input_path = Path(args.input_path)
    train_output_path = Path(args.train_output_path)
    dev_output_path = Path(args.dev_output_path)
    train_output_path.parent.mkdir(parents=True, exist_ok=True)
    dev_output_path.parent.mkdir(parents=True, exist_ok=True)

    table = pq.read_table(input_path)
    rows = table.to_pylist()
    sft_rows = _to_sft_rows(rows)

    if not sft_rows:
        raise RuntimeError("No valid SFT rows were built from the input parquet.")

    rng = random.Random(args.seed)
    rng.shuffle(sft_rows)

    dev_size = min(max(args.dev_size, 0), len(sft_rows))
    dev_rows = sft_rows[:dev_size]
    train_rows = sft_rows[dev_size:]

    schema = _row_schema()
    pq.write_table(pa.Table.from_pylist(train_rows, schema=schema), train_output_path)
    pq.write_table(pa.Table.from_pylist(dev_rows, schema=schema), dev_output_path)

    print(f"input_rows={len(rows)}")
    print(f"sft_rows={len(sft_rows)}")
    print(f"train_rows={len(train_rows)} -> {train_output_path}")
    print(f"dev_rows={len(dev_rows)} -> {dev_output_path}")


if __name__ == "__main__":
    main()
