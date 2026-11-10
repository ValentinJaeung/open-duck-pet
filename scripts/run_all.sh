#!/usr/bin/env bash
# 오리 전체(시뮬레이션+뷰어, 기술, 인식, 두뇌, 자율 행동)를 백그라운드로 한 번에 켭니다. 로그는 logs/ 에 저장됩니다.
# 사용법: bash scripts/run_all.sh            (MuJoCo 뷰어 창이 뜸)
#         VIEWER=false bash scripts/run_all.sh   (뷰어 없이)
#         RATE=6 bash scripts/run_all.sh     (심심함/외로움이 6배 빨리 변함: 구경하기 좋음. 기본 1)
#         OVD=false 로 개방형 탐지 서버를 끌 수 있음 (끄기: bash scripts/stop_all.sh, OVD 서버는 따로 kill)
# 명령:   bash scripts/say.sh "go to the red block"     사람 조종: bash scripts/person_teleop.sh
set -euo pipefail
R="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
mkdir -p "$R/logs"
if ! curl -s -m 2 localhost:11434/api/version >/dev/null; then
  echo "[알림] Ollama 서버가 꺼져 있습니다. 명령(LLM)을 쓰려면 다른 터미널에서 bash scripts/run_ollama.sh 를 켜세요. (자율 행동은 LLM 없이 동작)"
fi
bash "$R/scripts/stop_all.sh" >/dev/null 2>&1 || true
# 개방형 탐지(OWLv2) 서버: 학습하지 않은 물체 이름("yellow cup")을 찾을 때 필요. 이미 떠 있으면 그대로 둠(모델 로딩이 느려서 stop_all 은 끄지 않음)
if [ "${OVD:-true}" = "true" ] && [ -x "$R/.venv-vision/bin/python" ] && ! curl -s -m 2 localhost:8765/ >/dev/null; then
  nohup setsid bash "$R/scripts/run_ovd.sh" > "$R/logs/ovd.log" 2>&1 & echo "  시작: ovd (OWLv2, 로딩에 30~60초)"
fi
start() { local name="$1"; shift; nohup setsid "$@" > "$R/logs/$name.log" 2>&1 & echo "  시작: $name"; sleep "${DELAY:-1}"; }
start bridge     bash "$R/scripts/run_bridge.sh" --ros-args -p viewer:="${VIEWER:-true}" -p debug_pose:=false
DELAY=6 start skills bash "$R/scripts/run_skills.sh"
start perception bash "$R/scripts/run_perception.sh"
start brain      bash "$R/scripts/run_brain.sh"
start behavior   bash "$R/scripts/run_behavior.sh" --ros-args -p rate_scale:="${RATE:-1.0}"
echo "모두 시작함. 로그: $R/logs/   끄기: bash scripts/stop_all.sh"
