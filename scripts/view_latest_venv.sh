#!/usr/bin/env bash
# 학습 중/후 가장 최근 .onnx 를 MuJoCo 뷰어로 봅니다. (학습과 동시에 실행 가능, CPU 사용)
# 사용법: bash scripts/view_latest_venv.sh [onnx 경로]   (생략하면 가장 최근 것)
# 조작: 화살표=이동, a/e=회전, h=고개 모드
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PLAYGROUND="$REPO_ROOT/third_party/Open_Duck_Playground"
ONNX="${1:-$(ls -t "$REPO_ROOT"/models/runs/*/*.onnx | head -1)}"
echo "정책: $ONNX"
cd "$PLAYGROUND"
JAX_PLATFORMS=cpu PYTHONPATH="$PLAYGROUND" "$REPO_ROOT/.venv/bin/python" playground/open_duck_mini_v2/mujoco_infer.py \
  -o "$(realpath "$ONNX")" --model_path playground/open_duck_mini_v2/xmls/scene_flat_terrain_backlash.xml
