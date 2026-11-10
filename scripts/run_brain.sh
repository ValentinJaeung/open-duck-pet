#!/usr/bin/env bash
# 두뇌(LLM 도구 호출) 실행. Ollama 서버(localhost:11434)가 떠 있어야 합니다.
# 사용법: bash scripts/run_brain.sh [--ros-args -p model:=qwen3:8b]
set -euo pipefail
R="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
set +u; source /opt/ros/humble/setup.bash; source "$R/ros2_ws/install/setup.bash"; set -u
exec "$R/ros2_ws/install/duck_brain/lib/duck_brain/brain_node" --ros-args -p use_sim_time:=true "$@"
