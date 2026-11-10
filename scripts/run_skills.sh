#!/usr/bin/env bash
# 기술 Action 서버(turn/walk/look_at/감정 + walk_to)를 실행합니다. 먼저 scripts/run_bridge.sh 가 떠 있어야 합니다.
# (ros2 run 은 자식 프로세스를 따로 띄워 종료 때 남는 문제가 있어, 실행 파일을 직접 실행합니다)
set -euo pipefail
R="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
set +u; source /opt/ros/humble/setup.bash; source "$R/ros2_ws/install/setup.bash"; set -u
L="$R/ros2_ws/install/duck_skills/lib/duck_skills"
export DUCK_PET_ROOT="$R"
"$L/walk_to_node" --ros-args -p use_sim_time:=true &
WT=$!
"$L/push_ball_node" --ros-args -p use_sim_time:=true &
PB=$!; trap 'kill $WT $PB 2>/dev/null' EXIT INT TERM
"$L/skills_node" --ros-args -p use_sim_time:=true "$@"
