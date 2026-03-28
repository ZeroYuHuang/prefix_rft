#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

RUN_NAME="${RUN_NAME:-verify-qwen3-1p7b-prefix-rft-v3-$(date "+%y-%m-%d-%H-%M-%S")}"

export RUN_NAME
export WANDB_MODE="${WANDB_MODE:-offline}"
export LOGGER_LIST="${LOGGER_LIST:-[console,wandb]}"

bash "${SCRIPT_DIR}/code-qwen3-1p7b-prefix-rft-v3.sh" \
  trainer.experiment_name="${RUN_NAME}" \
  trainer.project_name="prft-coder1-verify" \
  trainer.total_epochs=1 \
  trainer.total_training_steps=5 \
  trainer.val_before_train=true \
  trainer.test_freq=1000 \
  trainer.save_freq=5 \
  data.max_prompt_length=3072 \
  data.max_response_length=3072 \
  data.train_max_samples=64 \
  data.val_max_samples=64 \
  data.val_batch_size=64 \
  "$@"
