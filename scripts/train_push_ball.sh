#!/usr/bin/env bash
# 공 밀기 상위 정책 학습 (계층형 RL: 고정된 걷기 정책 위에 속도 명령을 내는 정책). .venv(JAX GPU) 사용.
# 사용법: bash scripts/train_push_ball.sh [상위 스텝 수]   (기본 2천만, RTX 3080 에서 약 30~40분 + 컴파일 5분)
# 주의: Ollama 가 GPU 메모리(약 5.5GB)를 쓰고 있으면 먼저 내려야 합니다(자동으로 몇 분 뒤 내려감).
set -euo pipefail
R="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$R/third_party/Open_Duck_Playground"
export PYTHONPATH="$PWD" XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.7}"
exec "$R/.venv/bin/python" "$R/sim/rl/train_push_ball.py" "$@"
