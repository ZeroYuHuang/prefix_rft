#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

RESULT_ROOT="${RESULT_ROOT:-/mnt/hdfs/haruna/home/byte_data_seed/hdd_hldy/user/zeyuhuang/prft-results}"
RUN_NAME="${RUN_NAME:-qwen3-0p6b-code-offline-val-$(date "+%y-%m-%d-%H-%M-%S")}"
RUN_ROOT="${RESULT_ROOT}/${RUN_NAME}"

GEN_OUTPUT_PATH="${GEN_OUTPUT_PATH:-${RUN_ROOT}/val_generations.parquet}"
EVAL_OUTPUT_DIR="${EVAL_OUTPUT_DIR:-${RUN_ROOT}/offline_eval}"

mkdir -p "${RUN_ROOT}"

OUTPUT_PATH="${GEN_OUTPUT_PATH}" RUN_NAME="${RUN_NAME}" \
    "${SCRIPT_DIR}/code-qwen3-0p6b-offline-generate-val.sh"

INPUT_PATH="${GEN_OUTPUT_PATH}" OUTPUT_DIR="${EVAL_OUTPUT_DIR}" \
    "${SCRIPT_DIR}/code-qwen3-0p6b-offline-eval-val.sh"
