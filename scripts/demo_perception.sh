#!/usr/bin/env bash
# 3단계 데모: 오리가 제자리에서 돌며 방을 훑고, 본 물체를 지도에 기억하는 모습을 rviz2 로 봅니다.
#   터미널1: bash scripts/run_bridge.sh
#   터미널2: bash scripts/run_skills.sh
#   터미널3: bash scripts/run_perception.sh
#   터미널4: rviz2 -d rviz/perception.rviz        (화면에 지도가 안 보이면 Fixed Frame 이 odom 인지 확인)
#   터미널5: bash scripts/demo_perception.sh      (이 스크립트: 한 바퀴 돌기 → 앞으로 0.8m → 한 바퀴)
set -euo pipefail
R="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
set +u; source /opt/ros/humble/setup.bash; source "$R/ros2_ws/install/setup.bash"; set -u
turn() { ros2 action send_goal /turn duck_interfaces/action/Turn "{angle: $1}" >/dev/null; }
echo "▶ 제자리에서 한 바퀴"; for i in 1 2 3 4 5 6 7; do turn 1.0; done
echo "▶ 앞으로 0.8m";        ros2 action send_goal /walk duck_interfaces/action/Walk "{distance: 0.8}" >/dev/null
echo "▶ 다시 한 바퀴";       for i in 1 2 3 4 5 6 7; do turn 1.0; done
echo "끝. rviz2 지도에 빨간/초록/파란 구가 보이면 성공"
