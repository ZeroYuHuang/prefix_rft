#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

MODEL_PATH="${MODEL_PATH:-/mnt/hdfs/haruna/home/byte_data_seed/hdd_hldy/user/zeyuhuang/models/Qwen3-1.7B-Base}"
RUN_NAME="${RUN_NAME:-qwen3-1p7b-base-offline-valbeforetrain-like-$(date "+%y-%m-%d-%H-%M-%S")}"

MODEL_PATH="${MODEL_PATH}" \
RUN_NAME="${RUN_NAME}" \
TEMPERATURE="${TEMPERATURE:-0.6}" \
TOP_P="${TOP_P:-1.0}" \
N_SAMPLES="${N_SAMPLES:-1}" \
PROMPT_LENGTH="${PROMPT_LENGTH:-8192}" \
RESPONSE_LENGTH="${RESPONSE_LENGTH:-8192}" \
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-131072}" \
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.8}" \
exec bash "${SCRIPT_DIR}/code-qwen3-0p6b-offline-generate-val.sh" "$@"
