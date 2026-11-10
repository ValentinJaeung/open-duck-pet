#!/usr/bin/env bash
# 내 방 안에서 오리를 키보드로 걷게 합니다. 사용: bash scripts/walk_in_room.sh [onnx]
set -euo pipefail
R="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$R/third_party/Open_Duck_Playground"
JAX_PLATFORMS=cpu PYTHONPATH="$PWD" "$R/.venv/bin/python" "$R/sim/walk_in_room.py" "$@"
