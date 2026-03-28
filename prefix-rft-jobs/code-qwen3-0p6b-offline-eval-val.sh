#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/.." && pwd)

INPUT_PATH="${INPUT_PATH:-}"
RESPONSE_KEY="${RESPONSE_KEY:-responses}"
REASONING_KEY="${REASONING_KEY:-reasoning_contents}"
REWARD_PATH="${REWARD_PATH:-pkg://prefix_rft_v3.custom_rewards.code_r1_v2}"
REWARD_NAME="${REWARD_NAME:-compute_score}"
LOCAL_STAGE_ROOT="${LOCAL_STAGE_ROOT:-/tmp/prefix-rft-offline-staging}"
CODER1_EXEC="${CODER1_EXEC:-firejail}"
NPROC=$(nproc)
DEFAULT_NUM_WORKERS=$(( NPROC > 96 ? 96 : NPROC ))
NUM_WORKERS="${NUM_WORKERS:-${DEFAULT_NUM_WORKERS}}"
PROGRESS_DESC="${PROGRESS_DESC:-offline_eval}"
SHOW_PROGRESS="${SHOW_PROGRESS:-1}"

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

if [[ -n "${LOCAL_OUTPUT_DIR:-}" ]]; then
    CREATED_LOCAL_STAGE_DIR=0
else
    LOCAL_OUTPUT_DIR="${LOCAL_STAGE_ROOT}/$(basename -- "$(dirname -- "${INPUT_PATH}")")/$(basename -- "${OUTPUT_DIR}")"
    CREATED_LOCAL_STAGE_DIR=1
fi

cleanup_local_stage() {
    if [[ "${DRY_RUN:-0}" == "1" ]]; then
        return
    fi
    if [[ "${KEEP_LOCAL_STAGE:-0}" == "1" ]]; then
        return
    fi
    if [[ "${CREATED_LOCAL_STAGE_DIR:-0}" == "1" && -d "${LOCAL_OUTPUT_DIR}" ]]; then
        rm -rf "${LOCAL_OUTPUT_DIR}"
    fi
}

trap cleanup_local_stage EXIT

mkdir -p "${LOCAL_OUTPUT_DIR}" "${OUTPUT_DIR}"

PROGRESS_ARGS=()
if [[ "${SHOW_PROGRESS}" == "0" ]]; then
    PROGRESS_ARGS+=(--no-progress)
fi

cd "${REPO_ROOT}"
run_cmd env CODER1_EXEC="${CODER1_EXEC}" python3 -m prefix_rft_v3.offline_code_eval \
    --input-path "${INPUT_PATH}" \
    --output-dir "${LOCAL_OUTPUT_DIR}" \
    --response-key "${RESPONSE_KEY}" \
    --reasoning-key "${REASONING_KEY}" \
    --num-workers "${NUM_WORKERS}" \
    --progress-desc "${PROGRESS_DESC}" \
    --reward-path "${REWARD_PATH}" \
    --reward-name "${REWARD_NAME}" \
    "${PROGRESS_ARGS[@]}" \
    "$@"

if [[ "${DRY_RUN:-0}" == "1" ]]; then
    printf 'cp -f %q %q\n' "${LOCAL_OUTPUT_DIR}/metrics.json" "${OUTPUT_DIR}/metrics.json"
    printf 'cp -f %q %q\n' "${LOCAL_OUTPUT_DIR}/metrics_detail.parquet" "${OUTPUT_DIR}/metrics_detail.parquet"
    exit 0
fi

cp -f "${LOCAL_OUTPUT_DIR}/metrics.json" "${OUTPUT_DIR}/metrics.json"
cp -f "${LOCAL_OUTPUT_DIR}/metrics_detail.parquet" "${OUTPUT_DIR}/metrics_detail.parquet"
echo "Copied offline eval outputs to ${OUTPUT_DIR}"
