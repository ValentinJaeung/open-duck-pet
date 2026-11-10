#!/usr/bin/env bash
# conda 대신 uv venv(.venv)로 걷기 정책을 학습합니다. (train_walk.sh 와 동일 동작)
# 사용법: bash scripts/train_walk_venv.sh [스텝 수] [task] [이어할 체크포인트 경로]
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PLAYGROUND="$REPO_ROOT/third_party/Open_Duck_Playground"
STEPS="${1:-300000000}"
TASK="${2:-flat_terrain_backlash}"
RESTORE="${3:-}"
RUN_DIR="$REPO_ROOT/models/runs/$(date +%Y%m%d_%H%M%S)_${TASK}"
mkdir -p "$RUN_DIR"
export PYTHONPATH="$PLAYGROUND"
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.75}"
if ! "$REPO_ROOT/.venv/bin/python" -c 'import jax,sys; sys.exit(0 if any(d.platform=="gpu" for d in jax.devices()) else 1)' 2>/dev/null; then
  echo "[에러] JAX 가 GPU 를 못 찾습니다. nvidia-smi 확인 (docs/PROGRESS.md 참고)" >&2; exit 1
fi
echo "task=$TASK steps=$STEPS  출력: $RUN_DIR"
cd "$PLAYGROUND"
EXTRA=(); [ -n "$RESTORE" ] && EXTRA=(--restore_checkpoint_path "$RESTORE")
"$REPO_ROOT/.venv/bin/python" playground/open_duck_mini_v2/runner.py \
  --task "$TASK" --num_timesteps "$STEPS" --output_dir "$RUN_DIR" "${EXTRA[@]}" 2>&1 | tee "$RUN_DIR/train.log"
