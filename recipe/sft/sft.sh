set -x

export HOME=/mnt/jfs2/cth/085b13/
date=$(date "+%y-%m-%d-%h-%m-%s")

train_data_tag=luffy_train_subset_0.1_seed_42_sys_wo_format
train_data=$HOME/_data/processed_dataset_new/$train_data_tag

model=Qwen2.5-Math-1.5B
max_length=8192
proj_name=prefix_rft_analysis
experiment_name=sft_${model}_${train_data_tag}_${date}
logger="['console','wandb']"
total_epochs=15


cd $HOME/train-verl-updated

torchrun --standalone --nnodes=1 --nproc_per_node=8 \
    -m recipe.sft.fsdp_sft_trainer \
    use_remove_padding=true \
    data.train_files=$train_data/train.parquet \
    data.truncation=right \
    data.val_files=$train_data/train.parquet \
    data.prompt_key=prompt \
    data.response_key=response \
    data.max_length=$max_length \
    data.train_batch_size=64 \
    data.micro_batch_size_per_gpu=2 \
    model.partial_pretrain=$HOME/_models/$model \
    trainer.project_name=$proj_name \
    trainer.experiment_name=$experiment_name \
    trainer.total_epochs=$total_epochs \
    trainer.logger=$logger \
    trainer.n_gpus_per_node=8 \
    trainer.default_hdfs_dir=null $@