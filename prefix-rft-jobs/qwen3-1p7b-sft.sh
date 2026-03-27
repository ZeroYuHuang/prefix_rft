set -x

export HOME=/home/ma-user/work/hzy-dev
date=$(date "+%y-%m-%d-%h-%m-%s")

# train_data_tag=luffy_train_subset_0.1_seed_42_sys_wo_format
train_data_tag=luffy_train_subset_1.0_seed_42_sys_wo_format
train_data=$HOME/_data/processed_dataset_new/$train_data_tag

model=Qwen3-1.7B-Base
max_length=8192
proj_name=prefix_rft_rebuttal
experiment_name=${model}_sft_baseline_${train_data_tag}_${date}
logger="['console','wandb']"
total_epochs=3
lr=5e-5
warmup_steps_ratio=0.1

which conda
source /home/ma-user/work/hzy-dev/miniconda3/bin/activate prefix-rft

cd $HOME/prefix-rft

export WANDB_DIR=$HOME/wandb_logs/
export WANDB_API_KEY=d0a97fe1ba84f9958aee7b38fef9ac05048af4e1
/home/ma-user/work/hzy-dev/miniconda3/envs/prefix-rft/bin/torchrun --standalone --nnodes=1 --nproc_per_node=8 \
    -m recipe.sft.fsdp_sft_trainer \
    use_remove_padding=true \
    data.train_files=$train_data/train.parquet \
    data.truncation=right \
    data.val_files=$train_data/train.parquet \
    data.prompt_key=prompt \
    data.response_key=response \
    data.max_length=$max_length \
    data.train_batch_size=128 \
    data.micro_batch_size_per_gpu=2 \
    model.partial_pretrain=/home/ma-user/work/dev/_huggingface/models/$model \
    trainer.project_name=$proj_name \
    trainer.experiment_name=$experiment_name \
    trainer.total_epochs=$total_epochs \
    trainer.logger=$logger \
    trainer.n_gpus_per_node=8 \
    optim.lr=$lr \
    optim.warmup_steps_ratio=$warmup_steps_ratio \
    trainer.save_freq=357 \
    trainer.default_hdfs_dir=null $@