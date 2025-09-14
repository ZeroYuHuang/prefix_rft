#!/usr/bin/env bash

set -e

cd $HOME/train-verl-updated
export WANDB_DIR=$HOME/wandb_logs/
mkdir -p $WANDB_DIR
export WANDB_API_KEY=YOUR_WANDB_API_KEY

echo "$EXP_NAME"

MASTER_IP=${MASTER_ADDR:-"localhost"}
NODE_RANK=${NODE_RANK:-"-1"}
REDIS_PASSWORD=${REDIS_PASSWORD:-"5241590000000000"}
RAY_PORT=6379
EXPECTED_NODES=${NODE_COUNT:-1}  # 期望的总节点数（包括主节点）

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] [$1] $2"
}

if [[ -z "$MASTER_IP" || -z "$NODE_RANK" || -z "$REDIS_PASSWORD" ]]; then
    log "ERROR" "Environment variables MASTER_ADDR, NODE_RANK, and REDIS_PASSWORD must be set."
    exit 1
fi

log "INFO" "Cleaning up previous Ray resources on all nodes..."
ray stop --force || true 
sleep 5
log "INFO" "Ray resources cleaned up successfully."

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

    log "INFO" "Waiting for all nodes to join the cluster (Expected $EXPECTED_NODES nodes)..."
    while true; do
        STATUS_OUTPUT=$(ray status --address="$MASTER_IP:$RAY_PORT" 2>/dev/null)
        if [[ $? -ne 0 ]]; then
            log "INFO" "Failed to get cluster status. Retrying in 5 seconds..."
            sleep 5
            continue
        fi

        NUM_NODES=$(echo "$STATUS_OUTPUT" | awk '/Active:/,/Pending:/' | grep -c "node_")
        if [[ "$NUM_NODES" -ge "$EXPECTED_NODES" ]]; then
            log "INFO" "All nodes ($NUM_NODES/$EXPECTED_NODES) have joined the cluster."
            break
        fi
        log "INFO" "Current number of nodes: $NUM_NODES/$EXPECTED_NODES. Retrying in 5 seconds..."
        sleep 5
    done

    log "INFO" "Submitting Ray job to $MASTER_IP:1234"
    ray job submit \
        --address="http://$MASTER_IP:1234" \
        --runtime-env=$HOME/train-verl-updated/verl/trainer/runtime_env.yaml \
        -- python3 -m recipe.prefix_rft_v2.main $TRAIN_CONFIG
fi


if [[ "$NODE_RANK" -gt 0 ]]; then   
    sleep 15
    log "INFO" "Adding worker node to the cluster (MASTER_IP: $MASTER_IP)..."
    ray start --address="$MASTER_IP:$RAY_PORT"
    if [ $? -ne 0 ]; then
        log "ERROR" "Failed to start Ray worker node."
        exit 1
    fi
    log "INFO" "Ray worker node started successfully."
    sleep 30
    
    log "INFO" "Worker node is running and waiting for tasks..."
    while true; do
        # ray status --address="$MASTER_IP:$RAY_PORT"
        # curl http://${MASTER_IP}:1234
        sleep 60
    done
fi