#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

"${SCRIPT_DIR}/code-qwen3-0p6b-offline-eval-val.sh" "$@"
