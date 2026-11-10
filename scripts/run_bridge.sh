#!/usr/bin/env bash
# ROS2 브리지(방 + 오리 + 걷기 정책)를 실행합니다.
# 사용법: bash scripts/run_bridge.sh [--ros-args -p viewer:=false -p onnx:=/경로.onnx]
set -euo pipefail
R="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
set +u; source /opt/ros/humble/setup.bash; [ -f "$R/ros2_ws/install/setup.bash" ] && source "$R/ros2_ws/install/setup.bash"; set -u
export DUCK_PET_ROOT="$R" JAX_PLATFORMS=cpu MUJOCO_GL="${MUJOCO_GL:-glfw}"
export PYTHONPATH="$R/ros2_ws/src/duck_sim_bridge:$R/third_party/Open_Duck_Playground:${PYTHONPATH:-}"
exec "$R/.venv-ros/bin/python" -m duck_sim_bridge.bridge_node "$@"
