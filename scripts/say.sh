#!/usr/bin/env bash
# 오리에게 말하기. 사용법: bash scripts/say.sh "go to the red block and look up"
# 인자 없이 실행하면 계속 입력받는 대화 모드. 'stop' 은 즉시 멈춤.
set -euo pipefail
R="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
set +u; source /opt/ros/humble/setup.bash; source "$R/ros2_ws/install/setup.bash"; set -u
send() { ros2 topic pub --once /user_command std_msgs/msg/String "{data: \"${1//\"/\\\"}\"}" >/dev/null; }
if [ $# -gt 0 ]; then send "$*"; exit 0; fi
echo "오리에게 말하세요 (Ctrl+C 종료, stop = 멈춤). 결과는 brain 터미널에 나옵니다."
while IFS= read -r -p "🦆> " line; do [ -n "$line" ] && send "$line"; done
