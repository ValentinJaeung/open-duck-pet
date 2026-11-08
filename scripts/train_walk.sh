#!/usr/bin/env bash
# 걷기 정책 학습 (PPO, MJX). RTX 3080 기준 3억 스텝에 약 2시간.
# 사용법:  bash scripts/train_walk.sh [스텝 수] [task]
#   예)   bash scripts/train_walk.sh 1000000          # 동작 확인용 짧은 학습
#         bash scripts/train_walk.sh                  # 본 학습 (3억 스텝)
# 결과:   models/runs/<날짜>/ 에 체크포인트, .onnx, TensorBoard 로그, train.log
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PLAYGROUND="$REPO_ROOT/third_party/Open_Duck_Playground"
STEPS="${1:-300000000}"
TASK="${2:-flat_terrain_backlash}"
RUN_DIR="$REPO_ROOT/models/runs/$(date +%Y%m%d_%H%M%S)_${TASK}"

mkdir -p "$RUN_DIR"
echo "task=$TASK steps=$STEPS"
echo "출력 폴더: $RUN_DIR"
echo "진행 상황: tail -f $RUN_DIR/train.log | grep STEP"
echo "그래프:    tensorboard --logdir=$REPO_ROOT/models/runs"

cd "$PLAYGROUND"   # runner 가 JAX 컴파일 캐시(.tmp)를 현재 폴더에 만듭니다
conda run --no-capture-output -n duck python playground/open_duck_mini_v2/runner.py \
  --task "$TASK" \
  --num_timesteps "$STEPS" \
  --output_dir "$RUN_DIR" 2>&1 | tee "$RUN_DIR/train.log"
