#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

RESULT_ROOT="${RESULT_ROOT:-/mnt/hdfs/haruna/home/byte_data_seed/hdd_hldy/user/zeyuhuang/prft-results}"
RUN_NAME="${RUN_NAME:-qwen3-1p7b-base-offline-valbeforetrain-like-$(date "+%y-%m-%d-%H-%M-%S")}"
RUN_ROOT="${RESULT_ROOT}/${RUN_NAME}"

GEN_OUTPUT_PATH="${GEN_OUTPUT_PATH:-${RUN_ROOT}/val_generations.parquet}"
EVAL_OUTPUT_DIR="${EVAL_OUTPUT_DIR:-${RUN_ROOT}/offline_eval_valbeforetrain_like}"

mkdir -p "${RUN_ROOT}"

OUTPUT_PATH="${GEN_OUTPUT_PATH}" RUN_NAME="${RUN_NAME}" \
    bash "${SCRIPT_DIR}/code-qwen3-1p7b-offline-generate-valbeforetrain-like.sh"

INPUT_PATH="${GEN_OUTPUT_PATH}" OUTPUT_DIR="${EVAL_OUTPUT_DIR}" \
    bash "${SCRIPT_DIR}/code-qwen3-1p7b-offline-eval-valbeforetrain-like.sh"
