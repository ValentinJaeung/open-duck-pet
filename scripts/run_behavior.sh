#!/usr/bin/env bash
# 자율 행동(심심함/에너지/외로움 + Behavior Tree) 실행. 브리지·기술·인식이 먼저 떠 있어야 합니다.
# 사용법: bash scripts/run_behavior.sh [--ros-args -p rate_scale:=5.0]   (rate_scale: 상태가 변하는 속도 배율, 시험용)
set -euo pipefail
R="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
set +u; source /opt/ros/humble/setup.bash; source "$R/ros2_ws/install/setup.bash"; set -u
export PYTHONPATH="$R/ros2_ws/src/duck_behavior:${PYTHONPATH:-}"
exec "$R/.venv-ros/bin/python" -m duck_behavior.behavior_node --ros-args -p use_sim_time:=true "$@"
