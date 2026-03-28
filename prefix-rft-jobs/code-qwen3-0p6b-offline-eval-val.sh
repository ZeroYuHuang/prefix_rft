#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/.." && pwd)

INPUT_PATH="${INPUT_PATH:-}"
RESPONSE_KEY="${RESPONSE_KEY:-responses}"
REWARD_PATH="${REWARD_PATH:-pkg://prefix_rft_v3.custom_rewards.code_r1_v2}"
REWARD_NAME="${REWARD_NAME:-compute_score}"

run_cmd() {
    if [[ "${DRY_RUN:-0}" == "1" ]]; then
        printf '%q ' "$@"
        printf '\n'
        return 0
    fi
    "$@"
}

if [[ -z "${INPUT_PATH}" ]]; then
    echo "Set INPUT_PATH to a generations parquet before running offline eval." >&2
    exit 1
fi

OUTPUT_DIR="${OUTPUT_DIR:-$(dirname -- "${INPUT_PATH}")/offline_eval}"

cd "${REPO_ROOT}"
run_cmd python3 -m prefix_rft_v3.offline_code_eval \
    --input-path "${INPUT_PATH}" \
    --output-dir "${OUTPUT_DIR}" \
    --response-key "${RESPONSE_KEY}" \
    --reward-path "${REWARD_PATH}" \
    --reward-name "${REWARD_NAME}" \
    "$@"
