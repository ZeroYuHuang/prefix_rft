#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
CREDENTIAL_FILE="${WANDB_SYNC_ENV_FILE:-${SCRIPT_DIR}/.wandb_sync.env}"

if [[ -f "$CREDENTIAL_FILE" ]]; then
    set -a
    . "$CREDENTIAL_FILE"
    set +a
fi

usage() {
    cat <<'EOF'
Usage: scripts/sync_wandb_offline.sh <wandb_dir_or_run_root> [options]

Sync offline W&B runs to the cloud using WANDB_API_KEY authentication.

Arguments:
  <wandb_dir_or_run_root>   Either a run root like .../prft-results/<run_name>
                            or a direct wandb directory like .../<run_name>/wandb

Options:
  --entity <entity>         W&B entity/team name. Defaults to WANDB_ENTITY.
  --project <project>       W&B project name. Defaults to prft-coder1.
  --mark-synced             Mark synced runs locally. Enabled by default.
  --no-mark-synced          Do not mark synced runs locally.
  --clean                   Delete synced runs after upload.
  -h, --help                Show this help message.

Environment:
  WANDB_API_KEY             Required unless you have already configured auth.
  WANDB_ENTITY              Optional default entity.
  WANDB_PROJECT             Optional default project.
  WANDB_SYNC_ENV_FILE       Optional path to an env file to source before syncing.

Examples:
  export WANDB_API_KEY=xxx
  scripts/sync_wandb_offline.sh \
    /mnt/hdfs/haruna/home/byte_data_seed/hdd_hldy/user/zeyuhuang/prft-results/my-run

  scripts/sync_wandb_offline.sh \
    /mnt/hdfs/haruna/home/byte_data_seed/hdd_hldy/user/zeyuhuang/prft-results/my-run/wandb \
    --entity my-team --project prft-coder1
EOF
}

if [[ $# -lt 1 ]]; then
    usage
    exit 1
fi

TARGET_PATH=""
ENTITY="${WANDB_ENTITY:-}"
PROJECT="${WANDB_PROJECT:-prft-coder1}"
MARK_SYNCED=1
CLEAN=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --entity)
            ENTITY="$2"
            shift 2
            ;;
        --project)
            PROJECT="$2"
            shift 2
            ;;
        --mark-synced)
            MARK_SYNCED=1
            shift
            ;;
        --no-mark-synced)
            MARK_SYNCED=0
            shift
            ;;
        --clean)
            CLEAN=1
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        --*)
            echo "Unknown option: $1" >&2
            usage
            exit 1
            ;;
        *)
            if [[ -n "$TARGET_PATH" ]]; then
                echo "Unexpected extra argument: $1" >&2
                usage
                exit 1
            fi
            TARGET_PATH="$1"
            shift
            ;;
    esac
done

if [[ -z "$TARGET_PATH" ]]; then
    echo "Missing target path" >&2
    usage
    exit 1
fi

if [[ ! -d "$TARGET_PATH" ]]; then
    echo "Path does not exist: $TARGET_PATH" >&2
    exit 2
fi

if [[ -d "$TARGET_PATH/wandb" ]]; then
    TARGET_PATH="$TARGET_PATH/wandb"
fi

if ! command -v wandb >/dev/null 2>&1; then
    echo "wandb CLI is not installed or not on PATH" >&2
    exit 3
fi

if [[ -z "${WANDB_API_KEY:-}" ]]; then
    echo "WANDB_API_KEY is not set." >&2
    echo "This machine cannot write ~/.netrc reliably, so exporting WANDB_API_KEY is recommended." >&2
    exit 4
fi

shopt -s globstar nullglob
run_paths=()
for path in "$TARGET_PATH"/**/offline-run-* "$TARGET_PATH"/**/run-*; do
    if [[ -d "$path" ]]; then
        run_paths+=("$path")
    fi
done
shopt -u globstar nullglob

if [[ ${#run_paths[@]} -eq 0 ]]; then
    echo "No W&B run directories found under: $TARGET_PATH" >&2
    exit 5
fi

export WANDB_MODE=online
export WANDB_PROJECT="$PROJECT"
if [[ -n "$ENTITY" ]]; then
    export WANDB_ENTITY="$ENTITY"
fi
export HTTP_PROXY="${HTTP_PROXY:-http://sys-proxy-rd-relay.byted.org:8118}"
export http_proxy="${http_proxy:-http://sys-proxy-rd-relay.byted.org:8118}"
export HTTPS_PROXY="${HTTPS_PROXY:-http://sys-proxy-rd-relay.byted.org:8118}"
export https_proxy="${https_proxy:-http://sys-proxy-rd-relay.byted.org:8118}"
export NO_PROXY="${NO_PROXY:-localhost,.byted.org,byted.org,.bytedance.net,bytedance.net,.byteintl.net,.tiktok-row.net,.tiktok-row.org,127.0.0.1,127.0.0.0/8,169.254.0.0/16,100.64.0.0/10,172.16.0.0/12,192.168.0.0/16,10.0.0.0/8,::1,fe80::/10,fd00::/8}"
export no_proxy="${no_proxy:-$NO_PROXY}"

args=(
    sync
    --include-offline
    --no-include-online
    --project "$PROJECT"
)

if [[ -n "$ENTITY" ]]; then
    args+=(--entity "$ENTITY")
fi

if [[ $MARK_SYNCED -eq 1 ]]; then
    args+=(--mark-synced)
else
    args+=(--no-mark-synced)
fi

if [[ $CLEAN -eq 1 ]]; then
    args+=(--clean)
fi

for path in "${run_paths[@]}"; do
    args+=("$path")
done

echo "Syncing offline W&B runs"
echo "  target:  $TARGET_PATH"
echo "  project: $PROJECT"
echo "  entity:  ${ENTITY:-<unset>}"
echo "  runs:    ${#run_paths[@]}"
echo "  proxy:   ${https_proxy}"

wandb "${args[@]}"
