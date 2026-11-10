#!/usr/bin/env bash
# 합성 데이터로 YOLO 학습 (.venv-vision). 먼저: MUJOCO_GL=glfw .venv-vision/bin/python sim/gen_dataset.py
# 사용법: bash scripts/train_yolo.sh [epochs]
set -euo pipefail
R="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$R/models/yolo"
# hsv_h=0: 물체를 색으로 구분하므로 색상(hue) 증강을 끔. flipud 등 기하 증강은 기본값
"$R/.venv-vision/bin/yolo" detect train model=yolov8n.pt data="$R/datasets/duck_objects/data.yaml" \
  epochs="${1:-40}" imgsz=320 batch=64 hsv_h=0.0 workers=8 project="$R/models/yolo" name=duck_objects exist_ok=True
