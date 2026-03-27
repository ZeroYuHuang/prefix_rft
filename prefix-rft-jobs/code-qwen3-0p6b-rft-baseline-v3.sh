#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/.." && pwd)

MODEL_PATH="/mnt/hdfs/haruna/home/byte_data_seed/hdd_hldy/user/zeyuhuang/models/Qwen3-0.6B-Base"
DATA_ROOT="/mnt/hdfs/haruna/home/byte_data_seed/hdd_hldy/user/zeyuhuang/datasets/prefix-rft-code"
RESULT_ROOT="/mnt/hdfs/haruna/home/byte_data_seed/hdd_hldy/user/zeyuhuang/prft-results"

RUN_NAME="${RUN_NAME:-qwen3-0p6b-code-rft-baseline-v3-$(date "+%y-%m-%d-%H-%M-%S")}"
RUN_ROOT="${RESULT_ROOT}/${RUN_NAME}"
CKPT_DIR="${RUN_ROOT}/checkpoints"
WANDB_RUN_DIR="${RUN_ROOT}/wandb"
LOCAL_RAY_TMP="${LOCAL_RAY_TMP:-/tmp/prefix-rft-ray-v3}"
LOGGER_LIST="${LOGGER_LIST:-[console,wandb]}"

if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    N_GPUS_PER_NODE=$(python3 - <<'PY'
import os
print(len([x for x in os.environ["CUDA_VISIBLE_DEVICES"].split(",") if x.strip()]))
PY
)
else
    N_GPUS_PER_NODE=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
fi

mkdir -p "${CKPT_DIR}" "${WANDB_RUN_DIR}" "${LOCAL_RAY_TMP}"

export RAY_TEMP_DIR="${LOCAL_RAY_TMP}"
export WANDB_DIR="${WANDB_RUN_DIR}"
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1

TRAIN_ARGS=(
    "data.train_files=${DATA_ROOT}/train.parquet"
    "data.val_files=${DATA_ROOT}/test.parquet"
    "data.max_prompt_length=2048"
    "data.max_response_length=4096"
    "data.train_batch_size=16"
    "data.val_batch_size=256"
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
    "actor_rollout_ref.actor.ppo_mini_batch_size=64"
    "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=64"
    "actor_rollout_ref.actor.ppo_epochs=1"
    "actor_rollout_ref.actor.use_dynamic_bsz=true"
    "actor_rollout_ref.actor.ppo_max_token_len_per_gpu=24576"
    "actor_rollout_ref.actor.kl_loss_coef=0.0"
    "actor_rollout_ref.actor.entropy_coeff=0.001"
    "actor_rollout_ref.actor.fsdp_config.param_offload=false"
    "actor_rollout_ref.rollout.name=vllm"
    "actor_rollout_ref.rollout.tensor_model_parallel_size=1"
    "actor_rollout_ref.rollout.temperature=1.0"
    "actor_rollout_ref.rollout.gpu_memory_utilization=0.65"
    "actor_rollout_ref.rollout.n=4"
    "actor_rollout_ref.rollout.max_num_batched_tokens=32768"
    "actor_rollout_ref.rollout.val_kwargs.do_sample=false"
    "actor_rollout_ref.rollout.val_kwargs.n=1"
    "actor_rollout_ref.rollout.val_kwargs.top_p=1.0"
    "actor_rollout_ref.rollout.val_kwargs.temperature=0.0"
    "actor_rollout_ref.ref.fsdp_config.param_offload=true"
    "reward.custom_reward_function.path=prefix_rft_v3/custom_rewards/code_r1.py"
    "reward.custom_reward_function.name=compute_score"
    "algorithm.adv_estimator=grpo"
    "algorithm.norm_adv_by_std_in_grpo=false"
    "algorithm.kl_ctrl.kl_coef=0.0"
    "trainer.project_name=prefix_rft_v3_code"
    "trainer.experiment_name=${RUN_NAME}"
    "trainer.logger=${LOGGER_LIST}"
    "trainer.log_val_generations=0"
    "trainer.nnodes=1"
    "trainer.n_gpus_per_node=${N_GPUS_PER_NODE}"
    "trainer.total_training_steps=1500"
    "trainer.save_freq=100"
    "trainer.test_freq=100"
    "trainer.val_before_train=true"
    "trainer.default_local_dir=${CKPT_DIR}"
    "trainer.default_hdfs_dir=null"
    "trainer.max_actor_ckpt_to_keep=3"
    "trainer.max_critic_ckpt_to_keep=3"
)

cd "${REPO_ROOT}"
python3 -m prefix_rft_v3.main "${TRAIN_ARGS[@]}" "$@"
