#!/bin/bash

# --- 配置 ---

# 1. 设置你想要监视的文件夹路径
TARGET_DIR="/home/ma-user/work/hzy-dev/_checkpoints/prefix_rft_rebuttal/Qwen2.5-Math-7B_rft_baseline_25-11-13-Nov-11-1763032101/global_step_500"

# 2. 设置检查的间隔时间 (秒)
#    (避免CPU占用过高)
SLEEP_INTERVAL=180

# --- 脚本主逻辑 ---

echo "正在开始监视文件夹: $TARGET_DIR"

# 使用 while 循环来检查文件夹是否存在
# [ ! -d "$TARGET_DIR" ] 的意思是:
#   ! = "非" (Not)
#   -d = "是一个目录" (Directory)
# 整体含义: "只要 $TARGET_DIR 不是一个目录, 就继续循环"
while [ ! -d "$TARGET_DIR" ]; do
    echo "文件夹 $TARGET_DIR 尚不存在。 $SLEEP_INTERVAL 秒后重试..."
    sleep $SLEEP_INTERVAL
done

# 当循环结束时, 意味着文件夹已经存在
echo "----------------------------------------"
echo "检测到文件夹 $TARGET_DIR 已经存在！"
echo "正在执行命令..."

# --- 在这里替换成你想要执行的命令 ---

bash /home/ma-user/work/hzy-dev/prefix-rft-jobs/evals/prft.sh test_math

bash /home/ma-user/work/hzy-dev/prefix-rft-jobs/evals/prft.sh aime2425_512

bash /home/ma-user/work/hzy-dev/prefix-rft-jobs/evals/prft.sh aime2425_512

bash /home/ma-user/work/hzy-dev/prefix-rft-jobs/evals/prft.sh aime2425_512

bash /home/ma-user/work/hzy-dev/prefix-rft-jobs/evals/prft.sh aime2425_512

# --- 命令执行完毕 ---

echo "命令执行完毕。脚本退出。"

# 脚本执行完毕后会自动退出
exit 0