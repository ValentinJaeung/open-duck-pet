#!/usr/bin/env bash
# 개방형 탐지 서버(OWLv2) 실행. 인식 노드가 모르는 물체 이름("yellow cup")을 찾을 때 이 서버를 부릅니다.
# 첫 실행에는 모델(약 600MB)을 내려받습니다. 사용법: bash scripts/run_ovd.sh [포트]
set -euo pipefail
R="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
exec "$R/.venv-vision/bin/python" "$R/sim/ovd_server.py" "${1:-8765}"
