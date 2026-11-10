#!/usr/bin/env bash
# ROS2 Humble 설치 (Ubuntu 22.04, apt). sudo 비밀번호가 필요하므로 터미널에서 직접 실행하세요.
# 사용법: bash scripts/setup_ros2_humble.sh
# 15~30분 걸리고 약 2GB를 받습니다. 이미 설치된 부분은 건너뜁니다.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [ "$(lsb_release -rs)" != "22.04" ]; then echo "[에러] Ubuntu 22.04 전용입니다." >&2; exit 1; fi

if [ ! -f /opt/ros/humble/setup.bash ]; then
  echo "[1/4] apt 저장소 등록"
  sudo apt update && sudo apt install -y software-properties-common curl
  sudo add-apt-repository -y universe
  sudo curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg
  echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" \
    | sudo tee /etc/apt/sources.list.d/ros2.list >/dev/null
  echo "[2/4] ros-humble-desktop 설치 (오래 걸림)"
  sudo apt update
  sudo apt install -y ros-humble-desktop ros-dev-tools python3-colcon-common-extensions \
    ros-humble-cv-bridge ros-humble-teleop-twist-keyboard
else
  echo "[1-2/4] ROS2 Humble 이미 설치됨"
fi

echo "[3/4] ~/.bashrc 에 ROS 환경 등록"
grep -q "/opt/ros/humble/setup.bash" ~/.bashrc || echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc

echo "[4/4] 오리 노드용 Python 3.10 환경(.venv-ros) 만들기"
cd "$REPO_ROOT"
[ -d .venv-ros ] || uv venv --python /usr/bin/python3 .venv-ros
uv pip install --python .venv-ros/bin/python -r env/requirements-ros.txt
echo "완료. 새 터미널을 열고 'ros2 --help' 가 되는지 확인하세요."
