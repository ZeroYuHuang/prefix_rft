set -e
export HOME=/home/ma-user/work/hzy-dev
date=$(date "+%y-%m-%d-%h-%m-%s")

# Data and reward function
train_data_tag=luffy_test_no_sys
test_data_tag=luffy_test_no_sys
# test_data_tag=luffy_test_ood
# test_data_tag=$1
# test_data_tag=test_math
# test_data_tag=luffy_test_raw
train_data=_data/processed_dataset_new/$train_data_tag
test_data=_data/processed_dataset_new/$test_data_tag
custom_rwd_fpath=math_oat
custom_rwd_fname=rfn

max_prompt_length=2500
max_response_length=13884
# model_dir=Qwen2.5-Math-7B_math_oat_rfn_luffy_train_unfilter_sys_wo_format_total_demo_8_grpo_low_cosine_decay_0.95_0.05_500_0.0_high_const_0.95____ent_clip_const_0.8___demo_ratio_1_25-08-27-Aug-08-1756281932
# tag=rft_baseline

# tag=rft_qwen3_1p7B
# model_dir=Qwen3-1.7B-Base_rft_25-11-24-Nov-11-1763957525
# global_step="global_step_500"
# model_path=$HOME/_checkpoints/prefix_rft_rebuttal/$model_dir/$global_step/actor/huggingface

# tag=prft_qwen3_1p7B
# model_dir=Qwen3-1.7B-Base_prft_25-11-27-Nov-11-1764233521
# global_step="global_step_600"
# model_path=$HOME/_checkpoints/prefix_rft_rebuttal/$model_dir/$global_step/actor/huggingface

tag=prft_ent_qwen3_1p7B
model_dir=Qwen3-1.7B-Base_prft_25-11-26-Nov-11-1764114220
global_step="global_step_500"
model_path=$HOME/_checkpoints/prefix_rft_rebuttal/$model_dir/$global_step/actor/huggingface

temp=0.3

# tag=sft_qwen3_1p7B
# model_dir=Qwen3-1.7B-Base_sft_baseline_luffy_train_subset_1.0_seed_42_sys_wo_format_25-11-19-Nov-11-1763511019
# global_step="global_step_1071"
# model_path=$HOME/_checkpoints/prefix_rft_rebuttal/$model_dir/$global_step

rollout_tp_size=1
vllm_gpu_util=0.9
val_before_train=True

proj_name=prefix_rft_eval
logger="['console']"

# Fake prefix_rft config to avoid errors
total_demo_n=8
min_prefix_len=50
num_prefix=8
num_empty_prefix=7
rollout_n=1

prefix_low_ctrl_type=cosine_decay
prefix_low_ctrl_init=0.95
prefix_low_ctrl_target=0.05
prefix_low_ctrl_n_steps=300
prefix_low_ctrl_warmup_ratio=0.0
prefix_low_ctrl_args="+data.prefix_low_ctrl_type=$prefix_low_ctrl_type \
    +data.prefix_low_ctrl.kwargs.init=$prefix_low_ctrl_init \
    +data.prefix_low_ctrl.kwargs.target=$prefix_low_ctrl_target \
    +data.prefix_low_ctrl.kwargs.n_steps=$prefix_low_ctrl_n_steps \
    +data.prefix_low_ctrl.kwargs.warmup_ratio=$prefix_low_ctrl_warmup_ratio"

prefix_high_ctrl_type=cosine_decay
prefix_high_ctrl_init=0.95
prefix_high_ctrl_target=0.05
prefix_high_ctrl_n_steps=250
prefix_high_ctrl_warmup_ratio=0.0
prefix_high_ctrl_args="+data.prefix_high_ctrl_type=$prefix_high_ctrl_type \
    +data.prefix_high_ctrl.kwargs.init=$prefix_high_ctrl_init \
    +data.prefix_high_ctrl.kwargs.target=$prefix_high_ctrl_target \
    +data.prefix_high_ctrl.kwargs.n_steps=$prefix_high_ctrl_n_steps \
    +data.prefix_high_ctrl.kwargs.warmup_ratio=$prefix_high_ctrl_warmup_ratio"

# Sampler args
alpha=1.0
beta=1.0
prefix_sampler_args="+data.prefix_sampler.kwargs.alpha=${alpha} +data.prefix_sampler.kwargs.beta=${beta}"
demo_ratio=1.0

# Combine all prefix args
prefix_ctrl_args="${prefix_low_ctrl_args} ${prefix_low_ctrl_wrapper_args} ${prefix_high_ctrl_args} ${prefix_high_ctrl_wrapper_args} ${prefix_sampler_args} \
    +data.min_prefix_len=$min_prefix_len \
    +data.num_prefix=$num_prefix \
    +data.num_empty_prefix=$num_empty_prefix"
# fake
reshaper=entropy
ent_mask_ratio_ctrl_type=const
ent_mask_ratio_ctrl_init=0.5
ent_mask_ratio_ctrl_args="+actor_rollout_ref.actor.off_adv_reshaper=$reshaper \
+actor_rollout_ref.actor.off_ent_mask_ratio_ctrl_type=$ent_mask_ratio_ctrl_type \
+actor_rollout_ref.actor.off_ent_mask_ratio_ctrl.kwargs.init=$ent_mask_ratio_ctrl_init \
+actor_rollout_ref.actor.off_ent_mask_ratio_ctrl.kwargs.target=$ent_mask_ratio_ctrl_target \
+actor_rollout_ref.actor.off_ent_mask_ratio_ctrl.kwargs.n_steps=$ent_mask_ratio_ctrl_n_steps \
+actor_rollout_ref.actor.off_ent_mask_ratio_ctrl.kwargs.warmup_ratio=$ent_mask_ratio_ctrl_warmup_ratio"

# exp_name
exp_name="eval_${tag}_${test_data_tag}_step_${global_step}_temp_${temp}_${date}"
export EXP_NAME=$exp_name
loss_ctrl_args="$prefix_ctrl_args $ent_mask_ratio_ctrl_args"

# Final training config
other_args="algorithm.adv_estimator=grpo \
    custom_reward_function.path=$HOME/prefix-rft/recipe/prefix_rft_v2/custom_rewards/$custom_rwd_fpath.py \
    custom_reward_function.name=$custom_rwd_fname \
    data.train_files=$HOME/$train_data/test.parquet \
    data.val_files=$HOME/$test_data/test.parquet \
    data.val_batch_size=8192 \
    data.filter_overlong_prompts=true \
    +data.demo_ratio=$demo_ratio \
    data.shuffle=true \
    data.filter_overlong_prompts_workers=16 \
    +data.filter_overlong_prompts_len=1024 \
    +data.seed=42 \
    +data.only_keep_dp_with_demo=$only_keep_dp_with_demo \
    data.max_prompt_length=$max_prompt_length \
    data.max_response_length=$max_response_length \
    +data.demos_key=demos \
    +data.total_demo_n=$total_demo_n \
    actor_rollout_ref.model.path=$model_path \
    actor_rollout_ref.model.use_remove_padding=true \
    +actor_rollout_ref.actor.replace_old_log_prob=${replace_old_log_prob} \
    actor_rollout_ref.actor.use_dynamic_bsz=true \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=$max_token_per_gpu \
    actor_rollout_ref.actor.kl_loss_coef=0.00 \
    +actor_rollout_ref.actor.enable_clip=$enable_clip \
    +actor_rollout_ref.actor.apply_kl_penalty=$apply_kl_penalty \
    actor_rollout_ref.actor.clip_ratio_low=${clip_ratio_low} \
    actor_rollout_ref.actor.clip_ratio_high=${clip_ratio_high} \
    +actor_rollout_ref.actor.clip_ratio_low_off=${clip_ratio_low_off} \
    +actor_rollout_ref.actor.clip_ratio_high_off=${clip_ratio_high_off} \
    actor_rollout_ref.actor.clip_ratio_c=${clip_ratio_c} \
    algorithm.kl_ctrl.kl_coef=$kl_penalty_coef \
    actor_rollout_ref.rollout.tensor_model_parallel_size=$rollout_tp_size \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.val_kwargs.temperature=$temp \
    actor_rollout_ref.rollout.val_kwargs.do_sample=true \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    actor_rollout_ref.rollout.val_kwargs.top_p=0.95 \
    actor_rollout_ref.rollout.gpu_memory_utilization=$vllm_gpu_util \
    actor_rollout_ref.rollout.n=$rollout_n \
    +actor_rollout_ref.rollout.truncate_length=$truncate_length \
    actor_rollout_ref.rollout.max_num_batched_tokens=32000 \
    trainer.val_before_train=$val_before_train \
    trainer.validation_data_dir=$HOME/_results/$exp_name/ \
    trainer.logger=$logger \
    trainer.log_val_generations=0 \
    trainer.project_name=$proj_name \
    trainer.experiment_name=$exp_name \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=20 \
    trainer.test_freq=$test_freq \
    trainer.total_training_steps=0 \
    trainer.total_epochs=0"

export TRAIN_CONFIG="$other_args $loss_ctrl_args"
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
export WANDB_DIR=$HOME/wandb_logs/
export WANDB_API_KEY=d0a97fe1ba84f9958aee7b38fef9ac05048af4e1
cd $HOME/prefix-rft
python3 -m recipe.prefix_rft_v2.main $TRAIN_CONFIG