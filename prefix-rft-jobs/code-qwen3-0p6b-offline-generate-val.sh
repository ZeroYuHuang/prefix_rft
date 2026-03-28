#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/.." && pwd)

BASE_MODEL_PATH="/mnt/hdfs/haruna/home/byte_data_seed/hdd_hldy/user/zeyuhuang/models/Qwen3-0.6B-Base"
DATA_ROOT="${DATA_ROOT:-/mnt/hdfs/haruna/home/byte_data_seed/hdd_hldy/user/zeyuhuang/datasets/prefix-rft-code}"
RESULT_ROOT="${RESULT_ROOT:-/mnt/hdfs/haruna/home/byte_data_seed/hdd_hldy/user/zeyuhuang/prft-results}"

RUN_NAME="${RUN_NAME:-qwen3-0p6b-code-offline-generate-val-$(date "+%y-%m-%d-%H-%M-%S")}"
RUN_ROOT="${RESULT_ROOT}/${RUN_NAME}"
INPUT_PATH="${INPUT_PATH:-${DATA_ROOT}/val.parquet}"
OUTPUT_PATH="${OUTPUT_PATH:-${RUN_ROOT}/val_generations.parquet}"
LOCAL_RAY_TMP="${LOCAL_RAY_TMP:-/tmp/prefix-rft-generate-ray}"
LOCAL_STAGE_ROOT="${LOCAL_STAGE_ROOT:-/tmp/prefix-rft-offline-staging}"

run_cmd() {
    if [[ "${DRY_RUN:-0}" == "1" ]]; then
        printf '%q ' "$@"
        printf '\n'
        return 0
    fi
    "$@"
}

resolve_model_path() {
    if [[ -n "${MODEL_PATH:-}" ]]; then
        printf '%s\n' "${MODEL_PATH}"
        return 0
    fi

    if [[ -z "${CHECKPOINT_ROOT:-}" ]]; then
        printf '%s\n' "${BASE_MODEL_PATH}"
        return 0
    fi

    local step_dir
    if [[ -n "${GLOBAL_STEP:-}" ]]; then
        step_dir="${CHECKPOINT_ROOT}/global_step_${GLOBAL_STEP}"
    else
        local candidates=()
        while IFS= read -r candidate; do
            candidates+=("${candidate}")
        done < <(compgen -G "${CHECKPOINT_ROOT}/global_step_*" || true)

        if [[ ${#candidates[@]} -eq 0 ]]; then
            echo "No checkpoint directories found under ${CHECKPOINT_ROOT}" >&2
            return 1
        fi

        IFS=$'\n' candidates=($(printf '%s\n' "${candidates[@]}" | sort -V))
        unset IFS
        step_dir="${candidates[$((${#candidates[@]} - 1))]}"
    fi

    local candidate_paths=(
        "${step_dir}/huggingface"
        "${step_dir}/actor/huggingface"
        "${step_dir}/actor"
        "${step_dir}"
    )

    local candidate_path
    for candidate_path in "${candidate_paths[@]}"; do
        if [[ -d "${candidate_path}" ]]; then
            printf '%s\n' "${candidate_path}"
            return 0
        fi
    done

    echo "Resolved checkpoint path does not exist: ${step_dir}" >&2
    return 1
}

if [[ -n "${N_GPUS_PER_NODE:-}" ]]; then
    DETECTED_GPUS="${N_GPUS_PER_NODE}"
elif [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    DETECTED_GPUS=$(python3 - <<'PY'
import os
print(len([x for x in os.environ["CUDA_VISIBLE_DEVICES"].split(",") if x.strip()]))
PY
)
else
    DETECTED_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
fi

if [[ "${DETECTED_GPUS}" -le 0 ]]; then
    if [[ "${DRY_RUN:-0}" == "1" ]]; then
        N_GPUS_PER_NODE=1
    else
        echo "No visible GPUs found. Set N_GPUS_PER_NODE explicitly if needed." >&2
        exit 1
    fi
else
    N_GPUS_PER_NODE="${DETECTED_GPUS}"
fi

MODEL_PATH="$(resolve_model_path)"

if [[ -n "${LOCAL_OUTPUT_PATH:-}" ]]; then
    CREATED_LOCAL_STAGE_DIR=0
else
    LOCAL_OUTPUT_PATH="${LOCAL_STAGE_ROOT}/${RUN_NAME}/$(basename -- "${OUTPUT_PATH}")"
    CREATED_LOCAL_STAGE_DIR=1
fi

LOCAL_OUTPUT_DIR=$(dirname -- "${LOCAL_OUTPUT_PATH}")

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

mkdir -p "${RUN_ROOT}" "$(dirname -- "${OUTPUT_PATH}")" "${LOCAL_RAY_TMP}" "${LOCAL_OUTPUT_DIR}"

export RAY_TEMP_DIR="${LOCAL_RAY_TMP}"
export TOKENIZERS_PARALLELISM=true
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1

GEN_ARGS=(
    "trainer.nnodes=1"
    "trainer.n_gpus_per_node=${N_GPUS_PER_NODE}"
    "actor_rollout_ref.model.path=${MODEL_PATH}"
    "actor_rollout_ref.model.trust_remote_code=True"
    "actor_rollout_ref.rollout.name=vllm"
    "actor_rollout_ref.rollout.tensor_model_parallel_size=${GEN_TP:-1}"
    "actor_rollout_ref.rollout.gpu_memory_utilization=${GPU_MEMORY_UTILIZATION:-0.65}"
    "actor_rollout_ref.rollout.temperature=${TEMPERATURE:-0.0}"
    "actor_rollout_ref.rollout.top_p=${TOP_P:-1.0}"
    "actor_rollout_ref.rollout.n=${N_SAMPLES:-1}"
    "actor_rollout_ref.rollout.prompt_length=${PROMPT_LENGTH:-2048}"
    "actor_rollout_ref.rollout.response_length=${RESPONSE_LENGTH:-4096}"
    "actor_rollout_ref.rollout.max_num_batched_tokens=${MAX_NUM_BATCHED_TOKENS:-32768}"
    "data.train_files=${INPUT_PATH}"
    "data.prompt_key=prompt"
    "+data.output_path=${LOCAL_OUTPUT_PATH}"
)

cd "${REPO_ROOT}"
run_cmd python3 -m verl.trainer.main_generation_server "${GEN_ARGS[@]}" "$@"

if [[ "${DRY_RUN:-0}" == "1" ]]; then
    printf 'cp -f %q %q\n' "${LOCAL_OUTPUT_PATH}" "${OUTPUT_PATH}"
    exit 0
fi

mkdir -p "$(dirname -- "${OUTPUT_PATH}")"
cp -f "${LOCAL_OUTPUT_PATH}" "${OUTPUT_PATH}"
echo "Copied generation output to ${OUTPUT_PATH}"
