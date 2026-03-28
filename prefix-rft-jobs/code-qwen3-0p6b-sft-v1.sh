#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/.." && pwd)

MODEL_PATH="${MODEL_PATH:-/mnt/hdfs/haruna/home/byte_data_seed/hdd_hldy/user/zeyuhuang/models/Qwen3-0.6B-Base}"
DATA_ROOT="${DATA_ROOT:-/mnt/hdfs/haruna/home/byte_data_seed/hdd_hldy/user/zeyuhuang/datasets/prefix-rft-code}"
RESULT_ROOT="${RESULT_ROOT:-/mnt/hdfs/haruna/home/byte_data_seed/hdd_hldy/user/zeyuhuang/prft-results}"

PREFIX_TRAIN_PATH="${PREFIX_TRAIN_PATH:-${DATA_ROOT}/train.parquet}"
TRAIN_FILES="${TRAIN_FILES:-${DATA_ROOT}/sft_train.parquet}"
VAL_FILES="${VAL_FILES:-${DATA_ROOT}/sft_dev.parquet}"
BUILD_SFT_DATA="${BUILD_SFT_DATA:-auto}"
SFT_DEV_SIZE="${SFT_DEV_SIZE:-256}"
SFT_SPLIT_SEED="${SFT_SPLIT_SEED:-42}"

RUN_NAME="${RUN_NAME:-qwen3-0p6b-code-sft-v1-$(date "+%y-%m-%d-%H-%M-%S")}"
RUN_ROOT="${RESULT_ROOT}/${RUN_NAME}"
CKPT_DIR="${RUN_ROOT}/checkpoints"
WANDB_RUN_DIR="${RUN_ROOT}/wandb"
LOGGER_LIST="${LOGGER_LIST:-[console,wandb]}"

run_cmd() {
    if [[ "${DRY_RUN:-0}" == "1" ]]; then
        printf '%q ' "$@"
        printf '\n'
        return 0
    fi
    "$@"
}

should_build_sft_data() {
    case "${BUILD_SFT_DATA}" in
        1|true|TRUE|True|yes|YES|on|ON)
            return 0
            ;;
        0|false|FALSE|False|no|NO|off|OFF)
            return 1
            ;;
        auto|AUTO|Auto)
            [[ ! -f "${TRAIN_FILES}" || ! -f "${VAL_FILES}" ]]
            return
            ;;
        *)
            echo "Unsupported BUILD_SFT_DATA value: ${BUILD_SFT_DATA}" >&2
            exit 1
            ;;
    esac
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

if should_build_sft_data; then
    cd "${REPO_ROOT}"
    run_cmd python3 -m prefix_rft_v3.data_pre.build_sft_from_prefix_train \
        --input-path "${PREFIX_TRAIN_PATH}" \
        --train-output-path "${TRAIN_FILES}" \
        --dev-output-path "${VAL_FILES}" \
        --dev-size "${SFT_DEV_SIZE}" \
        --seed "${SFT_SPLIT_SEED}"
fi

if [[ ! -f "${TRAIN_FILES}" || ! -f "${VAL_FILES}" ]]; then
    echo "Missing SFT parquet(s): ${TRAIN_FILES} ${VAL_FILES}" >&2
    echo "Set BUILD_SFT_DATA=1 or provide TRAIN_FILES/VAL_FILES explicitly." >&2
    exit 1
fi

mkdir -p "${CKPT_DIR}" "${WANDB_RUN_DIR}"

export WANDB_DIR="${WANDB_RUN_DIR}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export TOKENIZERS_PARALLELISM=true

SFT_ARGS=(
    "data.train_files=${TRAIN_FILES}"
    "data.val_files=${VAL_FILES}"
    "data.messages_key=messages"
    "data.train_batch_size=${TRAIN_BATCH_SIZE:-64}"
    "data.micro_batch_size_per_gpu=${MICRO_BATCH_SIZE_PER_GPU:-2}"
    "data.max_token_len_per_gpu=${MAX_TOKEN_LEN_PER_GPU:-24576}"
    "data.use_dynamic_bsz=${USE_DYNAMIC_BSZ:-True}"
    "data.pad_mode=${PAD_MODE:-no_padding}"
    "data.max_length=${MAX_LENGTH:-6144}"
    "data.truncation=${TRUNCATION:-error}"
    "data.num_workers=${DATA_NUM_WORKERS:-8}"
    "data.ignore_input_ids_mismatch=${IGNORE_INPUT_IDS_MISMATCH:-True}"
    "model.path=${MODEL_PATH}"
    "model.trust_remote_code=True"
    "model.enable_gradient_checkpointing=True"
    "model.use_remove_padding=True"
    "model.use_liger=${USE_LIGER:-False}"
    "engine.strategy=fsdp"
    "engine.fsdp_size=-1"
    "engine.param_offload=${ENGINE_PARAM_OFFLOAD:-false}"
    "engine.optimizer_offload=${ENGINE_OPTIMIZER_OFFLOAD:-false}"
    "engine.dtype=${ENGINE_DTYPE:-bfloat16}"
    "optim.lr=${LR:-1e-5}"
    "optim.weight_decay=${WEIGHT_DECAY:-0.01}"
    "trainer.default_local_dir=${CKPT_DIR}"
    "trainer.default_hdfs_dir=null"
    "trainer.project_name=prft-coder1"
    "trainer.experiment_name=${RUN_NAME}"
    "trainer.logger=${LOGGER_LIST}"
    "trainer.total_epochs=${TOTAL_EPOCHS:-2}"
    "trainer.save_freq=${SAVE_FREQ:-200}"
    "trainer.test_freq=${TEST_FREQ:-100}"
    "trainer.max_ckpt_to_keep=${MAX_CKPT_TO_KEEP:-3}"
    "trainer.resume_mode=${RESUME_MODE:-disable}"
    "trainer.seed=${TRAINER_SEED:-42}"
    "trainer.nnodes=1"
    "trainer.n_gpus_per_node=${N_GPUS_PER_NODE}"
    "checkpoint.save_contents=[model,optimizer,extra,hf_model]"
)

cd "${REPO_ROOT}"
run_cmd torchrun --standalone --nnodes=1 --nproc_per_node="${N_GPUS_PER_NODE}" \
    -m verl.trainer.sft_trainer "${SFT_ARGS[@]}" "$@"
