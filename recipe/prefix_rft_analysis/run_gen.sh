cd /mnt/jfs2/cth/085b13/train-verl-updated/
# 获取当前时间戳并生成随机种子
generate_seed() {
    # 使用当前时间戳和 $RANDOM 生成一个随机数
    echo $(( $(date +%s%N) % 1000000 + $RANDOM ))
}

# step 1
s=0.0
e=1.0

CUDA_VISIBLE_DEVICES=0 python3 -m recipe.analysis.run_gen $GEN_CONFIG --seed $(generate_seed) -s $s -e $e &

CUDA_VISIBLE_DEVICES=1 python3 -m recipe.analysis.run_gen $GEN_CONFIG --seed $(generate_seed) -s $s -e $e &

CUDA_VISIBLE_DEVICES=2 python3 -m recipe.analysis.run_gen $GEN_CONFIG --seed $(generate_seed) -s $s -e $e &

CUDA_VISIBLE_DEVICES=3 python3 -m recipe.analysis.run_gen $GEN_CONFIG --seed $(generate_seed) -s $s -e $e &

CUDA_VISIBLE_DEVICES=4 python3 -m recipe.analysis.run_gen $GEN_CONFIG --seed $(generate_seed) -s $s -e $e &

CUDA_VISIBLE_DEVICES=5 python3 -m recipe.analysis.run_gen $GEN_CONFIG --seed $(generate_seed) -s $s -e $e &

CUDA_VISIBLE_DEVICES=6 python3 -m recipe.analysis.run_gen $GEN_CONFIG --seed $(generate_seed) -s $s -e $e &

CUDA_VISIBLE_DEVICES=7 python3 -m recipe.analysis.run_gen $GEN_CONFIG --seed $(generate_seed) -s $s -e $e 

wait

echo "2nd round"

s=0.0
e=1.0

CUDA_VISIBLE_DEVICES=0 python3 -m recipe.analysis.run_gen $GEN_CONFIG --seed $(generate_seed) -s $s -e $e &

CUDA_VISIBLE_DEVICES=1 python3 -m recipe.analysis.run_gen $GEN_CONFIG --seed $(generate_seed) -s $s -e $e &

CUDA_VISIBLE_DEVICES=2 python3 -m recipe.analysis.run_gen $GEN_CONFIG --seed $(generate_seed) -s $s -e $e &

CUDA_VISIBLE_DEVICES=3 python3 -m recipe.analysis.run_gen $GEN_CONFIG --seed $(generate_seed) -s $s -e $e &

CUDA_VISIBLE_DEVICES=4 python3 -m recipe.analysis.run_gen $GEN_CONFIG --seed $(generate_seed) -s $s -e $e &

CUDA_VISIBLE_DEVICES=5 python3 -m recipe.analysis.run_gen $GEN_CONFIG --seed $(generate_seed) -s $s -e $e &

CUDA_VISIBLE_DEVICES=6 python3 -m recipe.analysis.run_gen $GEN_CONFIG --seed $(generate_seed) -s $s -e $e &

CUDA_VISIBLE_DEVICES=7 python3 -m recipe.analysis.run_gen $GEN_CONFIG --seed $(generate_seed) -s $s -e $e 

wait

echo "16 completions generated"

python3 -m recipe.analysis.merge_and_verify \
    --input_folder=$OUTPUT_DIR \
    --output_file=$OUTPUT_FILE \
    --merged_key=analysis_gen

