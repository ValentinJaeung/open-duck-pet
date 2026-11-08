#!/usr/bin/env bash
# 학습된 정책(.onnx)을 MuJoCo 뷰어로 보면서 키보드로 조종합니다.
# 사용법:  bash scripts/view_policy.sh <정책.onnx> [scene xml]
# 조작:    화살표 = 앞/뒤/좌/우 이동,  a / e = 회전,  h = 고개 조종 모드 전환
set -euo pipefail

if [ $# -lt 1 ]; then
  echo "사용법: bash scripts/view_policy.sh <정책.onnx> [scene xml]" >&2
  exit 1
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PLAYGROUND="$REPO_ROOT/third_party/Open_Duck_Playground"
ONNX="$(realpath "$1")"
SCENE="${2:-playground/open_duck_mini_v2/xmls/scene_flat_terrain_backlash.xml}"

cd "$PLAYGROUND"
conda run --no-capture-output -n duck python playground/open_duck_mini_v2/mujoco_infer.py \
  -o "$ONNX" --model_path "$SCENE"
