cd /mnt/jfs2/cth/085b13/train-verl-updated/
export DATASET=luffy_train
export MODEL=/mnt/jfs/tianhao/085b13/cth-dev3/_models/DeepSeek-R1-Distill-Qwen-1.5B


# mkdir logs

# 获取当前时间戳并生成随机种子
generate_seed() {
    # 使用当前时间戳和 $RANDOM 生成一个随机数
    echo $(( $(date +%s%N) % 1000000 + $RANDOM ))
}

CUDA_VISIBLE_DEVICES=0 python3 -m recipe.prepare_data.run_gen -d $DATASET -m $MODEL --seed $(generate_seed) -t 0.6 --top_p 0.95 &

CUDA_VISIBLE_DEVICES=1 python3 -m recipe.prepare_data.run_gen -d $DATASET -m $MODEL --seed $(generate_seed) -t 0.6 --top_p 0.95 &

CUDA_VISIBLE_DEVICES=2 python3 -m recipe.prepare_data.run_gen -d $DATASET -m $MODEL --seed $(generate_seed) -t 0.6 --top_p 0.95 &

CUDA_VISIBLE_DEVICES=3 python3 -m recipe.prepare_data.run_gen -d $DATASET -m $MODEL --seed $(generate_seed) -t 0.6 --top_p 0.95 &

CUDA_VISIBLE_DEVICES=4 python3 -m recipe.prepare_data.run_gen -d $DATASET -m $MODEL --seed $(generate_seed) -t 0.6 --top_p 0.95 &

CUDA_VISIBLE_DEVICES=5 python3 -m recipe.prepare_data.run_gen -d $DATASET -m $MODEL --seed $(generate_seed) -t 0.6 --top_p 0.95 &

CUDA_VISIBLE_DEVICES=6 python3 -m recipe.prepare_data.run_gen -d $DATASET -m $MODEL --seed $(generate_seed) -t 0.6 --top_p 0.95 &

CUDA_VISIBLE_DEVICES=7 python3 -m recipe.prepare_data.run_gen -d $DATASET -m $MODEL --seed $(generate_seed) -t 0.6 --top_p 0.95 

wait

echo "All completions generated"