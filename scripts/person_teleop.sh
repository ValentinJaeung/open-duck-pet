#!/usr/bin/env bash
# 시뮬레이션 안의 '사람'(키 약 1m 인형)을 키보드로 조종합니다. 오리가 이 사람을 보고 다가오는지 확인하는 용도.
# 이 터미널을 클릭하고:  i 앞으로, k 정지, j/l 왼쪽/오른쪽으로 회전, u/o 앞으로 가며 회전, , 뒤로
# (키를 누르는 동안만 움직이고, 떼면 0.5초 뒤 멈춤. 방 밖으로는 못 나감)
set -euo pipefail
set +u; source /opt/ros/humble/setup.bash; set -u
exec ros2 run teleop_twist_keyboard teleop_twist_keyboard --ros-args -p speed:=0.3 -p turn:=0.8 -r cmd_vel:=/person/cmd_vel
