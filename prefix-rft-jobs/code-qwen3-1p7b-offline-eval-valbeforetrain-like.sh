#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

INPUT_PATH="${INPUT_PATH:-}"
OUTPUT_DIR="${OUTPUT_DIR:-}"

if [[ -z "${INPUT_PATH}" ]]; then
    echo "Set INPUT_PATH to a generations parquet before running val_before_train-like eval." >&2
    exit 1
fi

if [[ -z "${OUTPUT_DIR}" ]]; then
    OUTPUT_DIR="$(dirname -- "${INPUT_PATH}")/offline_eval_valbeforetrain_like"
fi

NORMALIZE_RESPONSES=0 \
CODER1_EXEC="${CODER1_EXEC:-firejail}" \
NUM_WORKERS="${NUM_WORKERS:-96}" \
PROGRESS_DESC="${PROGRESS_DESC:-val_before_train_like_eval}" \
SHOW_PROGRESS="${SHOW_PROGRESS:-1}" \
INPUT_PATH="${INPUT_PATH}" \
OUTPUT_DIR="${OUTPUT_DIR}" \
exec bash "${SCRIPT_DIR}/code-qwen3-1p7b-offline-eval-val.sh" "$@"
