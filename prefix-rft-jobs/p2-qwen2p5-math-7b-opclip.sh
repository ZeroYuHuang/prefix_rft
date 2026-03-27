# done!
#!/usr/bin/env bash
set -e

export RAY_TEMP_DIR="/home/ma-user/work/tmp"
export HOME=/home/ma-user/work/hzy-dev/
date=$(date "+%y-%m-%d-%h-%m-%s")

# Data and reward function
# train_data_tag=luffy_train_distill1p5b_sys_wo_format_all
# train_data_tag=luffy_train_distill1p5b_sys_wo_format_all_8k
# train_data_tag=luffy_train_distill1p5b_sys_wo_format_all_8k_correct
# train_data_tag=luffy_train_distill32b_sys_wo_format_all_8k_correct
train_data_tag=luffy_train_unfilter_sys_wo_format
test_data_tag=luffy_test_sys_wo_format
# train_data_tag=luffy_train_math_oat_unfilter
# test_data_tag=luffy_test
train_data=_data/processed_dataset_new/$train_data_tag
test_data=_data/processed_dataset_new/$test_data_tag
custom_rwd_fpath=math_oat
custom_rwd_fname=rfn

# Training related args
adv_estimator=grpo
norm_adv=false
loss_agg_mode=dr_grpo
max_prompt_length=8192
max_response_length=8192
truncate_length=8192
max_token_per_gpu=32768
model=Qwen2.5-Math-7B
actor_lr=1e-6
total_training_steps=2000
train_batch_size=128
ppo_mini_batch_size=64
ppo_epochs=1
apply_kl_penalty=false
kl_penalty_coef=0
rollout_tp_size=1
vllm_gpu_util=0.8
val_before_train=false
val_on_train=false

# Log
proj_name=prefix_rft_rebuttal
logger="['console','wandb']"
test_freq=1000

# Prefix config
total_demo_n=8
min_prefix_len=50
num_prefix=8
num_empty_prefix=8
rollout_n=1


# Prefix low controller
# prefix_low_ctrl_type=const
# prefix_low_ctrl_init=0.5
prefix_low_ctrl_type=cosine_decay
prefix_low_ctrl_init=0.95
prefix_low_ctrl_target=0.05
prefix_low_ctrl_n_steps=520
prefix_low_ctrl_warmup_ratio=0.0
prefix_low_ctrl_args="+data.prefix_low_ctrl_type=$prefix_low_ctrl_type \
    +data.prefix_low_ctrl.kwargs.init=$prefix_low_ctrl_init \
    +data.prefix_low_ctrl.kwargs.target=$prefix_low_ctrl_target \
    +data.prefix_low_ctrl.kwargs.n_steps=$prefix_low_ctrl_n_steps \
    +data.prefix_low_ctrl.kwargs.warmup_ratio=$prefix_low_ctrl_warmup_ratio"
prefix_low_ctrl_name="low_${prefix_low_ctrl_type}_${prefix_low_ctrl_init}_${prefix_low_ctrl_target}_${prefix_low_ctrl_n_steps}_${prefix_low_ctrl_warmup_ratio}"

# Prefix low controller wrapper
wrapper_type=delay
delay_steps=0
delay_val=0.0
if [[ "$wrapper_type" == "delay" ]]; then
    prefix_low_ctrl_wrapper_args="+data.prefix_low_ctrl_wrapper_type=delay \
    +data.prefix_low_ctrl_wrapper.kwargs.delay_steps=${delay_steps} \
    +data.prefix_low_ctrl_wrapper.kwargs.delay_val=${delay_val}"
fi

# Prefix high controller
prefix_high_ctrl_type=const
prefix_high_ctrl_init=0.95
# prefix_high_ctrl_type=cosine_decay
# prefix_high_ctrl_init=0.95
# prefix_high_ctrl_target=0.05
# prefix_high_ctrl_n_steps=250
# prefix_high_ctrl_warmup_ratio=0.0
prefix_high_ctrl_args="+data.prefix_high_ctrl_type=$prefix_high_ctrl_type \
    +data.prefix_high_ctrl.kwargs.init=$prefix_high_ctrl_init \
    +data.prefix_high_ctrl.kwargs.target=$prefix_high_ctrl_target \
    +data.prefix_high_ctrl.kwargs.n_steps=$prefix_high_ctrl_n_steps \
    +data.prefix_high_ctrl.kwargs.warmup_ratio=$prefix_high_ctrl_warmup_ratio"
prefix_high_ctrl_name="high_${prefix_high_ctrl_type}_${prefix_high_ctrl_init}_${prefix_high_ctrl_target}_${prefix_high_ctrl_n_steps}_${prefix_high_ctrl_warmup_ratio}"


# Wrapper args
# wrapper_type=delay
# delay_steps=500
# delay_val=0.95
# if [[ "$wrapper_type" == "delay" ]]; then
#     prefix_high_ctrl_wrapper_args="+data.prefix_high_ctrl_wrapper_type=delay \
#     +data.prefix_high_ctrl_wrapper.kwargs.delay_steps=${delay_steps} \
#     +data.prefix_high_ctrl_wrapper.kwargs.delay_val=${delay_val}"
# fi

# Sampler args
alpha=1.0
beta=1.0
prefix_sampler_args="+data.prefix_sampler.kwargs.alpha=${alpha} +data.prefix_sampler.kwargs.beta=${beta}"

# Combine all prefix args
prefix_ctrl_args="${prefix_low_ctrl_args} ${prefix_low_ctrl_wrapper_args} ${prefix_high_ctrl_args} ${prefix_high_ctrl_wrapper_args} ${prefix_sampler_args} \
    +data.min_prefix_len=$min_prefix_len \
    +data.num_prefix=$num_prefix \
    +data.num_empty_prefix=$num_empty_prefix "

# Advantage reshaping
# reshaper=identity
# reshaper=importance_sampling
# reshaper=luffy
# reshaper=\'relu,entropy\'
# reshaper=\'position,entropy\'
# reshaper=entropy_low
reshaper=entropy_op # hardcoded in the data base
export ON_POLICY_ENT_MASK_RATIO=0.8
# reshaper=\'entropy,length_on\'

ent_mask_ratio_ctrl_type=const
ent_mask_ratio_ctrl_init=1.0
# ent_mask_ratio_ctrl_type=cosine_decay 
# ent_mask_ratio_ctrl_init=0.95 
# ent_mask_ratio_ctrl_target=0.05 
# ent_mask_ratio_ctrl_n_steps=500 
# ent_mask_ratio_ctrl_warmup_ratio=0

ent_mask_ratio_ctrl_args="+actor_rollout_ref.actor.off_adv_reshaper=$reshaper \
+actor_rollout_ref.actor.off_ent_mask_ratio_ctrl_type=$ent_mask_ratio_ctrl_type \
+actor_rollout_ref.actor.off_ent_mask_ratio_ctrl.kwargs.init=$ent_mask_ratio_ctrl_init \
+actor_rollout_ref.actor.off_ent_mask_ratio_ctrl.kwargs.target=$ent_mask_ratio_ctrl_target \
+actor_rollout_ref.actor.off_ent_mask_ratio_ctrl.kwargs.n_steps=$ent_mask_ratio_ctrl_n_steps \
+actor_rollout_ref.actor.off_ent_mask_ratio_ctrl.kwargs.warmup_ratio=$ent_mask_ratio_ctrl_warmup_ratio"
ent_mask_ratio_ctrl_name="ent_clip_${ent_mask_ratio_ctrl_type}_${ent_mask_ratio_ctrl_init}_${ent_mask_ratio_ctrl_target}_${ent_mask_ratio_ctrl_n_steps}"

# Export flags
enable_clip=true
clip_ratio_low=0.2
clip_ratio_high=0.2
clip_ratio_c=3.0
clip_ratio_low_off=0.2
clip_ratio_high_off=0.2

entropy_coeff=0.001
entropy_mode=response_only
replace_old_log_prob=false

demo_ratio=1
only_keep_dp_with_demo=true

# Build exp name
other_name="${model}_onpolicy_ent_clip"
exp_name="${other_name}_${prefix_low_ctrl_name}_${prefix_high_ctrl_name}_${ent_mask_ratio_ctrl_name}_demo_ratio_${demo_ratio}_25-11-13-Nov-11-1763047495"
echo "$exp_name"
export EXP_NAME=$exp_name

# Dummy exports for undefined vars
loss_ctrl_args="$prefix_ctrl_args $ent_mask_ratio_ctrl_args"

# Final training config
other_args="algorithm.adv_estimator=$adv_estimator \
    algorithm.norm_adv_by_std_in_grpo=$norm_adv \
    custom_reward_function.path=$HOME/prefix-rft/recipe/prefix_rft_v2/custom_rewards/$custom_rwd_fpath.py \
    custom_reward_function.name=$custom_rwd_fname \
    data.train_files=$HOME/$train_data/train.parquet \
    data.val_files=$HOME/$test_data/test.parquet \
    data.filter_overlong_prompts=true \
    +data.demo_ratio=$demo_ratio \
    data.shuffle=true \
    data.filter_overlong_prompts_workers=16 \
    +data.filter_overlong_prompts_len=1024 \
    +data.seed=42 \
    +data.only_keep_dp_with_demo=$only_keep_dp_with_demo \
    data.train_batch_size=$train_batch_size \
    data.max_prompt_length=$max_prompt_length \
    data.max_response_length=$max_response_length \
    +data.demos_key=demos \
    +data.total_demo_n=$total_demo_n \
    actor_rollout_ref.model.path=/home/ma-user/work/dev/_huggingface/models/$model \
    actor_rollout_ref.model.use_remove_padding=true \
    actor_rollout_ref.actor.optim.lr=$actor_lr \
    actor_rollout_ref.actor.ppo_mini_batch_size=$ppo_mini_batch_size \
    actor_rollout_ref.actor.ppo_micro_batch_size=$ppo_mini_batch_size \
    actor_rollout_ref.actor.ppo_epochs=$ppo_epochs \
    actor_rollout_ref.actor.loss_agg_mode=$loss_agg_mode \
    +actor_rollout_ref.actor.entropy_mode=$entropy_mode \
    +actor_rollout_ref.actor.replace_old_log_prob=${replace_old_log_prob} \
    actor_rollout_ref.actor.use_dynamic_bsz=true \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=$max_token_per_gpu \
    actor_rollout_ref.actor.kl_loss_coef=0.00 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=1 \
    +actor_rollout_ref.actor.enable_clip=$enable_clip \
    +actor_rollout_ref.actor.apply_kl_penalty=$apply_kl_penalty \
    actor_rollout_ref.actor.clip_ratio_low=${clip_ratio_low} \
    actor_rollout_ref.actor.clip_ratio_high=${clip_ratio_high} \
    +actor_rollout_ref.actor.clip_ratio_low_off=${clip_ratio_low_off} \
    +actor_rollout_ref.actor.clip_ratio_high_off=${clip_ratio_high_off} \
    actor_rollout_ref.actor.clip_ratio_c=${clip_ratio_c} \
    algorithm.kl_ctrl.kl_coef=$kl_penalty_coef \
    actor_rollout_ref.model.enable_gradient_checkpointing=true \
    actor_rollout_ref.actor.fsdp_config.param_offload=false \
    actor_rollout_ref.rollout.tensor_model_parallel_size=$rollout_tp_size \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.6 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=true \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    actor_rollout_ref.rollout.val_kwargs.top_p=0.95 \
    +actor_rollout_ref.rollout.val_kwargs.max_tokens=12000 \
    actor_rollout_ref.rollout.gpu_memory_utilization=$vllm_gpu_util \
    actor_rollout_ref.rollout.n=$rollout_n \
    +actor_rollout_ref.rollout.truncate_length=$truncate_length \
    actor_rollout_ref.rollout.max_num_batched_tokens=32000 \
    actor_rollout_ref.ref.fsdp_config.param_offload=true \
    actor_rollout_ref.actor.entropy_coeff=$entropy_coeff \
    trainer.critic_warmup=0 \
    trainer.val_before_train=$val_before_train \
    trainer.logger=$logger \
    trainer.log_val_generations=0 \
    trainer.project_name=$proj_name \
    trainer.experiment_name=$exp_name \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=20 \
    trainer.test_freq=$test_freq \
    trainer.total_training_steps=$total_training_steps"

export TRAIN_CONFIG="$other_args $loss_ctrl_args"

export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
export WANDB_DIR=$HOME/wandb_logs/
export WANDB_API_KEY=d0a97fe1ba84f9958aee7b38fef9ac05048af4e1
cd $HOME/prefix-rft
python3 -m recipe.prefix_rft_v2.main $TRAIN_CONFIG