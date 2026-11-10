#!/usr/bin/env bash
# 감정 동작을 차례로 보여 줍니다. 먼저 다른 터미널 2개에서 브리지와 기술 서버를 켜 두세요.
#   터미널1: bash scripts/run_bridge.sh      (MuJoCo 뷰어 창이 뜸)
#   터미널2: bash scripts/run_skills.sh
#   터미널3: bash scripts/demo_emotions.sh
set -euo pipefail
R="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
set +u; source /opt/ros/humble/setup.bash; source "$R/ros2_ws/install/setup.bash"; set -u
send() { echo "▶ $1"; ros2 action send_goal "/$2" "duck_interfaces/action/$3" "$4" | grep -E "success|message|Goal (finished|rejected)" || true; sleep 1; }
send "고개 좌우 흔들기(wag_head) 3번" wag_head WagHead "{times: 3}"
send "기쁨(happy)"                    happy   Happy   "{intensity: 1.0}"
send "슬픔(sad)"                      sad     Sad     "{}"
send "춤(dance) 6초"                  dance   Dance   "{duration: 6.0}"
send "주위 둘러보기: 왼쪽"            look_at LookAt  "{yaw: 0.8, pitch: 0.0}"
send "주위 둘러보기: 오른쪽 위"        look_at LookAt  "{yaw: -0.8, pitch: 0.4}"
send "정면 복귀"                      look_at LookAt  "{yaw: 0.0, pitch: 0.0}"
echo "끝"
