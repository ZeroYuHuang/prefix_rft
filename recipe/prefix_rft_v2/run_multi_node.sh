#!/usr/bin/env bash

set -e

cd $HOME/train-verl-updated
export WANDB_DIR=$HOME/wandb_logs/
mkdir -p $WANDB_DIR
export WANDB_API_KEY=d0a97fe1ba84f9958aee7b38fef9ac05048af4e1

echo "$EXP_NAME"

# 获取环境变量
MASTER_IP=${MASTER_ADDR:-"localhost"}
NODE_RANK=${NODE_RANK:-"-1"}
REDIS_PASSWORD=${REDIS_PASSWORD:-"5241590000000000"}
RAY_PORT=6379
EXPECTED_NODES=${NODE_COUNT:-1}  # 期望的总节点数（包括主节点）

# 日志函数
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] [$1] $2"
}

# 检查必要环境变量
if [[ -z "$MASTER_IP" || -z "$NODE_RANK" || -z "$REDIS_PASSWORD" ]]; then
    log "ERROR" "Environment variables MASTER_ADDR, NODE_RANK, and REDIS_PASSWORD must be set."
    exit 1
fi

# 清理 Ray 资源
log "INFO" "Cleaning up previous Ray resources on all nodes..."
ray stop --force || true  # 强制停止 Ray 进程，忽略错误
sleep 5
log "INFO" "Ray resources cleaned up successfully."

# 启动主节点
if [[ "$NODE_RANK" -eq 0 ]]; then
    log "INFO" "Starting Ray cluster on master node (IP: $MASTER_IP)..."
    ray start --head \
              --node-ip-address="$MASTER_IP" \
              --port="$RAY_PORT" \
              --dashboard-port 1234 \
              --dashboard-host=0.0.0.0
    echo "Showing dashboard log:"
    # cat /tmp/ray/session_latest/logs/dashboard.log
    if [ $? -ne 0 ]; then
        log "ERROR" "Failed to start Ray master node."
        exit 1
    fi
    log "INFO" "Ray master node started successfully."

    sleep 15

    # 等待所有节点加入
    log "INFO" "Waiting for all nodes to join the cluster (Expected $EXPECTED_NODES nodes)..."
    while true; do
        STATUS_OUTPUT=$(ray status --address="$MASTER_IP:$RAY_PORT" 2>/dev/null)
        if [[ $? -ne 0 ]]; then
            log "INFO" "Failed to get cluster status. Retrying in 5 seconds..."
            sleep 5
            continue
        fi

        # 解析节点数（匹配 "Active" 部分中的节点行）
        NUM_NODES=$(echo "$STATUS_OUTPUT" | awk '/Active:/,/Pending:/' | grep -c "node_")
        if [[ "$NUM_NODES" -ge "$EXPECTED_NODES" ]]; then
            log "INFO" "All nodes ($NUM_NODES/$EXPECTED_NODES) have joined the cluster."
            break
        fi
        log "INFO" "Current number of nodes: $NUM_NODES/$EXPECTED_NODES. Retrying in 5 seconds..."
        sleep 5
    done

    # 提交一个简单的 Ray 任务
    log "INFO" "Submitting Ray job to $MASTER_IP:1234"
    ray job submit \
        --address="http://$MASTER_IP:1234" \
        --runtime-env=$HOME/train-verl-updated/verl/trainer/runtime_env.yaml \
        -- python3 -m recipe.prefix_rft_v2.main $TRAIN_CONFIG
fi


# 工作节点等待主节点启动
if [[ "$NODE_RANK" -gt 0 ]]; then   
    # log "INFO" "Waiting for master node to be ready (MASTER_IP: $MASTER_IP)..."
    # while true; do
    #     ray status --address="$MASTER_IP:$RAY_PORT" > /dev/null 2>&1
    #     if [ $? -eq 0 ]; then
    #         break
    #     fi
    #     log "INFO" "Master node is not ready yet. Retrying in 5 seconds..."
    #     sleep 5
    # done
    sleep 15
    log "INFO" "Adding worker node to the cluster (MASTER_IP: $MASTER_IP)..."
    ray start --address="$MASTER_IP:$RAY_PORT"
    if [ $? -ne 0 ]; then
        log "ERROR" "Failed to start Ray worker node."
        exit 1
    fi
    log "INFO" "Ray worker node started successfully."
    sleep 30
    
    # 工作节点保持运行
    log "INFO" "Worker node is running and waiting for tasks..."
    while true; do
        # ray status --address="$MASTER_IP:$RAY_PORT"
        # curl http://${MASTER_IP}:1234
        sleep 60
    done
fi