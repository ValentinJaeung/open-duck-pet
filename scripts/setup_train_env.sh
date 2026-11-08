#!/usr/bin/env bash
# 학습/추론용 conda 환경(duck)을 만들고 필요한 환경 변수를 설정합니다.
# 사용법:  bash scripts/setup_train_env.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PLAYGROUND="$REPO_ROOT/third_party/Open_Duck_Playground"
ENV_NAME="duck"

if ! command -v conda >/dev/null 2>&1; then
  echo "[에러] conda 를 찾을 수 없습니다. Anaconda 또는 Miniforge 를 먼저 설치하세요." >&2
  exit 1
fi

# 서브모듈이 비어 있으면 받아오기
if [ ! -f "$PLAYGROUND/README.md" ]; then
  echo "[1/4] 서브모듈 받는 중..."
  git -C "$REPO_ROOT" submodule update --init --recursive
else
  echo "[1/4] 서브모듈 확인 완료"
fi

if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  echo "[2/4] conda 환경 '$ENV_NAME' 이 이미 있습니다. 패키지만 맞춰 설치합니다."
  conda run -n "$ENV_NAME" pip install -r "$REPO_ROOT/env/requirements-train.txt"
else
  echo "[2/4] conda 환경 '$ENV_NAME' 생성 중... (10분 이상 걸릴 수 있습니다)"
  (cd "$REPO_ROOT/env" && conda env create -f duck-train.yml)
fi

echo "[3/4] 환경 변수 설정"
# PYTHONPATH: Open_Duck_Playground 의 'playground' 패키지를 import 하기 위함
# XLA_PYTHON_CLIENT_MEM_FRACTION: JAX 가 미리 잡는 GPU 메모리 비율 (10GB GPU 기준)
conda env config vars set -n "$ENV_NAME" \
  PYTHONPATH="$PLAYGROUND" \
  XLA_PYTHON_CLIENT_MEM_FRACTION=0.75 >/dev/null

echo "[4/4] 동작 확인 (GPU 인식 + 오리 환경 한 스텝)"
cd "$PLAYGROUND"
# 주의: conda run 은 stdin(heredoc)을 넘기지 않으므로 python -c 로 실행합니다.
conda run -n "$ENV_NAME" python -c '
import jax
print("JAX devices:", jax.devices())
from playground.open_duck_mini_v2 import joystick
env = joystick.Joystick(task="flat_terrain_backlash")
s = jax.jit(env.reset)(jax.random.PRNGKey(0))
s = jax.jit(env.step)(s, jax.numpy.zeros(env.action_size))
print("환경 OK - action:", env.action_size, "obs:", env.observation_size["state"])
'

echo
echo "완료! 다음 명령으로 환경을 활성화하세요:  conda activate $ENV_NAME"
