#!/usr/bin/env bash
# 인식 노드 실행 (YOLO ONNX + 3D 위치 + 의미 지도). 브리지(run_bridge.sh)가 먼저 떠 있어야 합니다.
set -euo pipefail
R="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
set +u; source /opt/ros/humble/setup.bash; source "$R/ros2_ws/install/setup.bash"; set -u
export DUCK_PET_ROOT="$R" JAX_PLATFORMS=cpu
export PYTHONPATH="$R/ros2_ws/src/duck_perception:${PYTHONPATH:-}"
exec "$R/.venv-ros/bin/python" -m duck_perception.perception_node --ros-args -p use_sim_time:=true "$@"
