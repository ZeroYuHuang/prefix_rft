cd /mnt/jfs2/cth/085b13/train-verl-updated/

dataset=luffy
model=/mnt/jfs/tianhao/085b13/cth-dev3/_modelsDeepSeek-R1-Distill-Qwen-7B

# 获取当前时间戳并生成随机种子
generate_seed() {
    # 使用当前时间戳和 $RANDOM 生成一个随机数
    echo $(( $(date +%s%N) % 1000000 + $RANDOM ))
}

CUDA_VISIBLE_DEVICES=0 python3 -m recipe.prepare_data.run_gen -d $dataset -m $model --seed ${generate_seed}