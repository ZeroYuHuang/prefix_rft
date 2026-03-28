#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/.." && pwd)

MODEL_PATH="/mnt/hdfs/haruna/home/byte_data_seed/hdd_hldy/user/zeyuhuang/models/Qwen3-0.6B-Base"
DATA_ROOT="/mnt/hdfs/haruna/home/byte_data_seed/hdd_hldy/user/zeyuhuang/datasets/prefix-rft-code"
RESULT_ROOT="/mnt/hdfs/haruna/home/byte_data_seed/hdd_hldy/user/zeyuhuang/prft-results"

RUN_NAME="${RUN_NAME:-qwen3-0p6b-code-prefix-rft-v3-$(date "+%y-%m-%d-%H-%M-%S")}" 
RUN_ROOT="${RESULT_ROOT}/${RUN_NAME}"
CKPT_DIR="${RUN_ROOT}/checkpoints"
WANDB_RUN_DIR="${RUN_ROOT}/wandb"
LOCAL_RAY_TMP="${LOCAL_RAY_TMP:-/tmp/prefix-rft-ray-v3}"
LOGGER_LIST="${LOGGER_LIST:-[console,wandb]}"

TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-1500}"
SAVE_FREQ="${SAVE_FREQ:-100}"
TEST_FREQ="${TEST_FREQ:-100}"

MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-8192}"
MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-8192}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-16384}"

TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-128}"
VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-64}"

ROLLOUT_N="${ROLLOUT_N:-1}"
NUM_PREFIX="${NUM_PREFIX:-8}"
NUM_EMPTY_PREFIX="${NUM_EMPTY_PREFIX:-7}"

TARGET_NORMALIZED_PPO_MINI_BATCH_SIZE_PER_GPU="${TARGET_NORMALIZED_PPO_MINI_BATCH_SIZE_PER_GPU:-64}"
PPO_MICRO_BATCH_SIZE_PER_GPU="${PPO_MICRO_BATCH_SIZE_PER_GPU:-64}"
PPO_MAX_TOKEN_LEN_PER_GPU="${PPO_MAX_TOKEN_LEN_PER_GPU:-65536}"
LOG_PROB_MAX_TOKEN_LEN_PER_GPU="${LOG_PROB_MAX_TOKEN_LEN_PER_GPU:-65536}"
ROLLOUT_MAX_BATCHED_TOKENS="${ROLLOUT_MAX_BATCHED_TOKENS:-131072}"
ROLLOUT_GPU_MEMORY_UTILIZATION="${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.75}"

PREFIX_HIGH_INIT="${PREFIX_HIGH_INIT:-0.95}"
PREFIX_LOW_INIT="${PREFIX_LOW_INIT:-0.95}"
PREFIX_LOW_TARGET="${PREFIX_LOW_TARGET:-0.5}"
PREFIX_LOW_WARMUP_RATIO="${PREFIX_LOW_WARMUP_RATIO:-0.0}"
PREFIX_LOW_DECAY_STEPS="${PREFIX_LOW_DECAY_STEPS:-${TOTAL_TRAINING_STEPS}}"

if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    N_GPUS_PER_NODE=$(python3 - <<'PY'
import os
print(len([x for x in os.environ["CUDA_VISIBLE_DEVICES"].split(",") if x.strip()]))
PY
)
else
    N_GPUS_PER_NODE=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
fi

PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-$((TARGET_NORMALIZED_PPO_MINI_BATCH_SIZE_PER_GPU * N_GPUS_PER_NODE / ROLLOUT_N))}"

mkdir -p "${CKPT_DIR}" "${WANDB_RUN_DIR}" "${LOCAL_RAY_TMP}"

export RAY_TEMP_DIR="${LOCAL_RAY_TMP}"
export WANDB_DIR="${WANDB_RUN_DIR}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1

TRAIN_ARGS=(
    "prefix_rft.enable=true"
    "prefix_rft.demo_ratio=1.0"
    "prefix_rft.only_keep_dp_with_demo=true"
    "prefix_rft.demo_key=demos"
    "prefix_rft.demo_corr_key=demos_corr"
    "prefix_rft.total_demo_n=1"
    "prefix_rft.min_prefix_len=32"
    "prefix_rft.num_prefix=${NUM_PREFIX}"
    "prefix_rft.num_empty_prefix=${NUM_EMPTY_PREFIX}"
    "prefix_rft.prefix_low_ctrl_type=cosine_decay"
    "prefix_rft.prefix_low_ctrl.kwargs.init=${PREFIX_LOW_INIT}"
    "prefix_rft.prefix_low_ctrl.kwargs.target=${PREFIX_LOW_TARGET}"
    "prefix_rft.prefix_low_ctrl.kwargs.n_steps=${PREFIX_LOW_DECAY_STEPS}"
    "prefix_rft.prefix_low_ctrl.kwargs.warmup_ratio=${PREFIX_LOW_WARMUP_RATIO}"
    "prefix_rft.prefix_high_ctrl_type=const"
    "prefix_rft.prefix_high_ctrl.kwargs.init=${PREFIX_HIGH_INIT}"
    "prefix_rft.prefix_sampler.kwargs.alpha=1.0"
    "prefix_rft.prefix_sampler.kwargs.beta=1.0"
    "prefix_rft.policy_loss_mode=prefix_v2"
    "prefix_rft.enable_clip=true"
    "data.train_files=${DATA_ROOT}/train.parquet"
    "data.val_files=${DATA_ROOT}/val.parquet"
    "data.max_prompt_length=${MAX_PROMPT_LENGTH}"
    "data.max_response_length=${MAX_RESPONSE_LENGTH}"
    "data.train_batch_size=${TRAIN_BATCH_SIZE}"
    "data.val_batch_size=${VAL_BATCH_SIZE}"
    "data.filter_overlong_prompts=true"
    "data.filter_overlong_prompts_workers=16"
    "data.trust_remote_code=true"
    "data.seed=42"
    "data.shuffle=true"
    "actor_rollout_ref.model.path=${MODEL_PATH}"
    "actor_rollout_ref.model.trust_remote_code=true"
    "actor_rollout_ref.model.use_remove_padding=true"
    "actor_rollout_ref.model.enable_gradient_checkpointing=true"
    "actor_rollout_ref.actor.optim.lr=1e-6"
    "actor_rollout_ref.actor.ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE}"
    "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${PPO_MICRO_BATCH_SIZE_PER_GPU}"
    "actor_rollout_ref.actor.ppo_epochs=1"
    "actor_rollout_ref.actor.use_dynamic_bsz=true"
    "actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${PPO_MAX_TOKEN_LEN_PER_GPU}"
    "actor_rollout_ref.actor.kl_loss_coef=0.0"
    "actor_rollout_ref.actor.entropy_coeff=0.001"
    "actor_rollout_ref.actor.fsdp_config.param_offload=false"
    "actor_rollout_ref.rollout.name=vllm"
    "actor_rollout_ref.rollout.tensor_model_parallel_size=1"
    "actor_rollout_ref.rollout.temperature=1.0"
    "actor_rollout_ref.rollout.gpu_memory_utilization=${ROLLOUT_GPU_MEMORY_UTILIZATION}"
    "actor_rollout_ref.rollout.n=${ROLLOUT_N}"
    "actor_rollout_ref.rollout.max_model_len=${MAX_MODEL_LEN}"
    "actor_rollout_ref.rollout.max_num_batched_tokens=${ROLLOUT_MAX_BATCHED_TOKENS}"
    "actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=true"
    "actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${LOG_PROB_MAX_TOKEN_LEN_PER_GPU}"
    "actor_rollout_ref.rollout.val_kwargs.do_sample=false"
    "actor_rollout_ref.rollout.val_kwargs.n=1"
    "actor_rollout_ref.rollout.val_kwargs.top_p=1.0"
    "actor_rollout_ref.rollout.val_kwargs.temperature=0.0"
    "actor_rollout_ref.ref.fsdp_config.param_offload=true"
    "reward.custom_reward_function.path=pkg://prefix_rft_v3.custom_rewards.code_r1_v2"
    "reward.custom_reward_function.name=compute_score"
    "algorithm.adv_estimator=grpo"
    "algorithm.norm_adv_by_std_in_grpo=false"
    "algorithm.kl_ctrl.kl_coef=0.0"
    "trainer.project_name=prft-coder1"
    "trainer.experiment_name=${RUN_NAME}"
    "trainer.logger=${LOGGER_LIST}"
    "trainer.log_val_generations=0"
    "trainer.nnodes=1"
    "trainer.n_gpus_per_node=${N_GPUS_PER_NODE}"
    "trainer.total_training_steps=${TOTAL_TRAINING_STEPS}"
    "trainer.save_freq=${SAVE_FREQ}"
    "trainer.test_freq=${TEST_FREQ}"
    "trainer.val_before_train=true"
    "trainer.default_local_dir=${CKPT_DIR}"
    "trainer.default_hdfs_dir=null"
    "trainer.max_actor_ckpt_to_keep=3"
    "trainer.max_critic_ckpt_to_keep=3"
    "hydra.run.dir=${RUN_ROOT}"
)

cd "${REPO_ROOT}"
python3 -m prefix_rft_v3.main "${TRAIN_ARGS[@]}" "$@"
