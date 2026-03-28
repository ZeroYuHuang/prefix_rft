#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/.." && pwd)

DATA_ROOT="${DATA_ROOT:-/mnt/hdfs/haruna/home/byte_data_seed/hdd_hldy/user/zeyuhuang/datasets/prefix-rft-code}"
INPUT_PATH="${INPUT_PATH:-${DATA_ROOT}/train.parquet}"
TRAIN_OUTPUT_PATH="${TRAIN_OUTPUT_PATH:-${DATA_ROOT}/sft_train.parquet}"
DEV_OUTPUT_PATH="${DEV_OUTPUT_PATH:-${DATA_ROOT}/sft_dev.parquet}"
DEV_SIZE="${DEV_SIZE:-256}"
SEED="${SEED:-42}"

cd "${REPO_ROOT}"
python3 -m prefix_rft_v3.data_pre.build_sft_from_prefix_train \
    --input-path "${INPUT_PATH}" \
    --train-output-path "${TRAIN_OUTPUT_PATH}" \
    --dev-output-path "${DEV_OUTPUT_PATH}" \
    --dev-size "${DEV_SIZE}" \
    --seed "${SEED}" \
    "$@"
