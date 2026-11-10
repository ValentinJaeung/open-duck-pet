# 진행 상황 기록 (이어서 작업하기용)

마지막 갱신: 2026-11-10 (각 단계 절의 괄호 시각이 그 절의 작성 시각. 2026-11-08 13:08 이후 일부 시각은 파일 수정 시각 기준 추정치)

## 현재 상태 요약 (이어서 하려면 여기부터)

| 단계 | 내용 | 상태 |
|---|---|---|
| 준비 | RL 걷기 정책 학습 (MJX+PPO, 3억 스텝, 86분) | ✅ 사용자가 뷰어에서 잘 걷는 것 확인 |
| 1 | 방 + 머리 카메라 + ROS2 브리지 (Humble) | ✅ rviz2 로 오리 시점 확인 |
| 2 | 기술 Action: turn/walk/look_at/happy/sad/wag_head/dance | ✅ (거리는 정확, 회전 오차 ±0.1~0.4 rad) |
| 3 | 인식: YOLO(합성 데이터) + 깊이 3D + 오도메트리 + 지도 | ✅ 물체 위치 오차 0.1~0.4 m |
| 4 | 언어→행동: 로컬 LLM(qwen3:8b) 도구 호출 + `walk_to` | ✅ 19/20, 새 문항 8/8, 전체 시연 성공 |
| 5 | 자율 행동: 심심함/에너지/외로움 + Behavior Tree + 사람 따라가기 | ✅ (쉬기 행동은 전체 시험에서 발동 못 시킴) |
| 6 | 새 RL 기술 | ⚠️ 앉기=불가(모델에 몸통 충돌 없음), 공 밀기=정답 관측 48% 이지만 **ROS 인식 연동 0/10**, 버튼/계단=미착수 |
| 8 | VLM 개방형 탐지(OWLv2): 학습하지 않은 물체 이름으로 찾아가기 | ✅ 같은 출발 6/6, 연달아 실행하면 3/6~6/6 (변동) |
| 7 | 음성 입력 | 📋 미착수 |

**알려진 큰 한계:** 공 밀기 ROS 연동 실패(병목은 인식/추정), 못 찾으면 탐험 없이 제자리에서 돌기만 함, 먼 작은 물체(2m+)는 320x240 에서 안 보임, 효과음 없음, 사용자가 직접 본 확인은 걷기·rviz·감정 일부뿐.

**다음 후보 (우선순위):** ① 물체를 못 찾으면 방의 다른 곳으로 이동하며 찾는 탐험 ② 공 밀기를 센서/구조 변경으로 재시도(아래를 보는 카메라, 더 나은 이동량 추정, 폐루프 시각 서보) ③ 몸통/머리 충돌 모델 추가 후 앉기·버튼 ④ 7단계 음성 ⑤ 쉬기 행동 실제 발동 확인

**실행 (저장소 폴더에서, 터미널마다 `cd /home/valen/projects/open-duck-pet`):**
```bash
bash scripts/run_ollama.sh            # LLM 서버(명령을 쓸 때만). 모델 qwen3:8b ~5GB
RATE=6 bash scripts/run_all.sh        # 브리지(뷰어)+기술+인식+두뇌+자율 행동 (+OWLv2 서버). RATE=상태 변화 배속
bash scripts/person_teleop.sh         # 시뮬레이션 속 사람을 키보드로 조종 (i 앞, j/l 회전, k 정지)
bash scripts/say.sh "go to the red block and look up"      # 오리에게 말하기 (인자 없으면 대화 모드)
bash scripts/stop_all.sh              # 끄기
```
새 물체(노란 컵, 주황 공, 보라색 상자)가 있는 방: `bash scripts/run_bridge.sh --ros-args -p novel_objects:=true`. 공 밀기 시험: 브리지에 `-p dynamic_objects:=true`.

**환경 3개:** `.venv`(Python 3.11, JAX GPU: RL 학습, `env/requirements-train.txt`), `.venv-ros`(3.10, ROS 노드: `env/requirements-ros.txt`), `.venv-vision`(3.11, PyTorch: YOLO 학습·OWLv2 서버: `env/requirements-vision.txt`). ROS2 Humble 은 `bash scripts/setup_ros2_humble.sh`(sudo). Ollama 는 `~/.local/ollama` 에 사용자 설치(sudo 불필요). 처음 한 번 `cd ros2_ws && source /opt/ros/humble/setup.bash && colcon build`.

### 커밋되지 않은 학습 산출물을 다시 만드는 방법 (`.gitignore` 로 제외됨)
| 산출물 | 위치 | 다시 만드는 법 | 소요 |
|---|---|---|---|
| 걷기 정책(ONNX) | `models/runs/<날짜>/*.onnx` | `bash scripts/train_walk_venv.sh` (이후 `sim/rl/low_policy.py` 의 `DEFAULT_ONNX` 경로를 새 파일로) | GPU 약 86분 |
| YOLO 학습 데이터 | `datasets/duck_objects/` | `MUJOCO_GL=glfw .venv-vision/bin/python sim/gen_dataset.py 6000 600` | 약 7분 |
| YOLO 모델 | `models/yolo/duck_objects.onnx` | `bash scripts/train_yolo.sh 40` 후 `yolo export ... format=onnx imgsz=320 opset=12`, 결과를 `models/yolo/duck_objects.onnx` 로 복사 | 약 4분 |
| 개방형 탐지 평가 이미지 | `datasets/ovd_eval/` | `MUJOCO_GL=glfw .venv-vision/bin/python sim/gen_ovd_eval.py 250` | 1분 |
| 공 밀기 정책 | `models/push/<날짜>/params_final.npz` | `bash scripts/train_push_ball.sh` (정답 관측 잔차 RL 기준은 `sim/rl/push_ball_env_v1_clean_obs.py`) | 약 40분 |
| OWLv2 가중치 | HuggingFace 캐시 | `bash scripts/run_ovd.sh` 첫 실행 때 자동(약 600MB) | |

## 머신
- Ubuntu 22.04.5, RTX 3080 (GA102), RAM 31GB, 16코어, conda 없음(uv 사용)
- 위치: /home/valen/projects/open-duck-pet

## 학습 기록
| 날짜 | 명령 | 결과 폴더 | 상태/메모 |
|---|---|---|---|
| 2026-11-08 | train_walk_venv.sh 1000000 | models/runs/20261108_113303_flat_terrain_backlash | 동작 확인 OK (ONNX 내보내기 포함, reward ~16) |
| 2026-11-08 | train_walk_venv.sh (3억) | models/runs/20261108_113921_flat_terrain_backlash | 완료 11:39~13:05 (약 86분, 3.0억 스텝). 최종 eval 보상 ≈ 271~315 (1.7억 스텝부터 정체). 최종 정책: 2026_11_08_130526_300482560.onnx |

## 참고
- 버전 올리지 말 것: docs/TROUBLESHOOTING.md
- 전체 로드맵: docs/ROADMAP.md

## 학습 시각화 방법
- 걷는 모습: `bash scripts/view_latest_venv.sh` (가장 최근 .onnx를 MuJoCo 뷰어로, 학습과 동시 실행 가능). 특정 파일은 인자로 지정.
- 그래프(보상 등): `.venv/bin/tensorboard --logdir models/runs` → http://localhost:6006
- 공부 팁: 초반(step 0) .onnx와 최근 .onnx를 번갈아 보며 비교

## 작업 로그 (시간순, KST, 2026-11-08)
| 시각 | 한 일 |
|---|---|
| 오전 | 저장소 clone, 서브모듈 초기화, uv로 .venv(Python 3.11) 생성 + 고정 버전 설치, CPU에서 환경 동작 확인 |
| 오전 | NVIDIA 드라이버 없음 확인 → 사용자가 직접 설치(595.99), JAX가 CudaDevice 인식 확인 |
| 11:33 | 1백만 스텝 동작 확인 학습 성공 (체크포인트 + ONNX 내보내기 OK) |
| 11:39 | 본 학습 3억 스텝 시작 (백그라운드). 폴더 `models/runs/20261108_113921_flat_terrain_backlash` |
| ~11:45 | view_latest_venv.sh 추가, 뷰어 실행 확인. 사용자 보고: 오리가 반응 없음 |
| ~11:55 | 원인 조사: 화면 없이 정책 시뮬레이션 → 2,100만 스텝 정책은 서 있기만 함(이동 1.5cm). 뷰어 자체는 정상 |
| 12:08 | GPU 100%/316W 확인 → 학습 속도 약 3.3M스텝/분, 설정으로 크게 빨라질 여지 적음 |
| ~12:15 | TensorBoard 실행 (http://localhost:6006) |
| 13:03 | 중간 보고: 2.79억 스텝, 보상 약 300에서 정체 |
| 13:05 | 학습 종료 (3.0048억 스텝, 최종 ONNX 저장) |
| 13:12 | 최종 평가: 보상 항목 변화 — alive 1078→11513, tracking_lin_vel 21→889, tracking_ang_vel 13→734, imitation -188→898 |
| 13:20 | 사용자가 뷰어에서 직접 확인: **잘 걷는다.** 간이 시뮬레이션의 4.7cm는 테스트 코드 쪽 문제로 판단(원인 미규명, 무시해도 됨) |
| 13:15 | 간이 시뮬레이션(앞 명령 0.15, 2000스텝): 이동 4.7cm. 훈련 지표(속도 추종 889)와 맞지 않아 **간이 테스트가 부정확할 가능성**. 뷰어에서 눈으로 확인 필요 |

| 13:08~13:15 | **1단계 진행(ROS2 제외 부분):** `sim/build_scene.py`(방 4x3m + 가구 + 물체 3개 + 오리 + 머리 카메라), `sim/render_check.py`(카메라 이미지 확인), `sim/walk_in_room.py` + `scripts/walk_in_room.sh`(방 안에서 키보드로 걷기). 카메라 방향은 정면 수평으로 보정(quat 0.7071 0 0 -0.7071, pos 0.02 0 -0.06). 방 안에서 정책이 넘어지지 않음 확인 |
| ~13:20 | 사용자가 ROS2 **Humble(apt)** 선택. `scripts/setup_ros2_humble.sh`(sudo 필요, 사용자가 직접 실행) 작성. sudo 없이 가능한 `.venv-ros`(시스템 Python 3.10 + mujoco/jax/brax/onnxruntime, numpy<2, `env/requirements-ros.txt`) 먼저 생성하고 방 모델 + 걷기 정책이 3.10에서 로드/추론되는 것 확인 |
| ~13:35 | 사용자가 Humble 설치 완료. 확인: ros2/rviz2/colcon/rclpy OK |
| 13:35~13:42 | **`duck_sim_bridge` 노드 작성** (`ros2_ws/src/duck_sim_bridge`, 실행: `bash scripts/run_bridge.sh`). 검증(뷰어 끔): /camera/image 10Hz, /joint_states 50Hz(실시간), /imu, /camera/depth, /camera_info 발행. `/cmd_vel` 0.15 로 8초 주행 → 머리 카메라 화면이 벽 쪽으로 이동(평균 픽셀 변화 36.8) 확인 |
| ~13:44 | 사용자가 rviz2에서 오리 시점 영상 확인 (1단계 핵심 결과물 달성) |

## 1단계 현황 (방 + ROS2 브리지)
- [x] 기본 도형으로 방 MJCF 작성 (치수는 임시값 4x3m. 실제 방 치수로 `sim/build_scene.py`의 ROOM/FURNITURE/OBJECTS 수정 필요)
- [x] 머리 RGB 카메라 추가 (깊이 카메라는 `mujoco.Renderer.enable_depth_rendering()`으로 가능, 아직 미적용)
- [ ] 방 안에서 눈으로 걷기 확인: `bash scripts/walk_in_room.sh` (사용자 확인 대기)
- [x] ROS2 Humble 설치 완료. 터미널에서 `bash scripts/setup_ros2_humble.sh` (sudo 비밀번호 입력, 15~30분). 끝나면 새 터미널에서 `ros2 --help` 확인. ROS 노드는 `.venv-ros`(Python 3.10)에서 실행, 학습은 `.venv`(3.11)
- [x] duck_sim_bridge 노드 (토픽: 구독 /cmd_vel, /head_cmd(4값) / 발행 /camera/image, /camera/depth, /camera/camera_info, /imu, /joint_states). 명령이 0.5초 안 오면 자동 정지. 정답 좌표는 발행하지 않음
- [x] rviz2에서 카메라 보기 확인됨 (키보드 조종 확인은 미보고)
  - 터미널1: `bash scripts/run_bridge.sh`
  - 터미널2: `source /opt/ros/humble/setup.bash && ros2 run teleop_twist_keyboard teleop_twist_keyboard --ros-args -p speed:=0.15 -p turn:=1.0` (i 앞, j/l 회전, k 정지)
  - 터미널3: `rviz2` → Add → By topic → /camera/image → Image
- [ ] (미룸) tf/오도메트리는 3단계(인식)에서

## 알려진 주의점
- 방 물체는 정책 추론 코드(`MJInferBase`)가 '움직이는 몸체=오리 1개'만 가정해서 walk_in_room 에서는 고정. 공 밀기 등은 6단계에서 자체 브리지로 해결
- 렌더링: `MUJOCO_GL=glfw` 사용 (egl 은 종료 때 에러). 스크래치에 `sim.py` 같은 이름을 두면 `sim` 패키지를 가림
- `pgrep -f runner.py` 는 자기 자신을 잡아 항상 "실행 중"으로 보임. 확인은 `ps aux | grep "[o]pen_duck_mini_v2/runner"`
- 키 입력: ↑ 앞, ↓ 뒤, ←→ 옆, Q/E 회전, H 고개 모드. 누르는 동안만 명령이 들어감
- train.log 에 STEP 줄은 일부만 나옴. 진행도는 체크포인트 파일명 숫자로 확인

## 측정 기록: 고개 / 걷기·회전 정확도 (세션 997b13, 13:45~13:53)
같은 시각에 다른 세션(fork)도 duck_skills / 브리지를 수정 중이라 **이 기록은 측정값만 남기고 코드는 건드리지 않음**.
- **걷기 정책은 고개 명령을 따라가도록 학습되지 않았음.** 학습 코드에서 고개 목표 덮어쓰기(`motor_targets.at[5:9].set(...)`)가 주석 처리되어 있고 고개 추종 보상도 없음. `/head_cmd`를 관측에만 넣으면 고개 관절이 전혀 안 움직임(실측). 정책 출력 뒤에 ctrl[5:9] 를 직접 덮어쓰면 명령각(0.5 → 0.49)을 정확히 따라감. 걷기에는 영향 없음(덮어쓰기 켜고/끄고 비교: 거의 동일)
- 고개 부호(운동학 계산): neck_pitch, head_pitch, head_yaw 모두 **양수 = 위 / 왼쪽**, 1 rad ≈ 카메라 1 rad. head_roll 은 화면만 기울임
- 걷기·회전 실측 (명령 대비): 앞 0.15 → 0.095~0.11 m/s 이고 오른쪽으로 약 -0.1 rad/s 휨. 뒤 -0.15 → 0.04~0.07 m/s(불안정). 회전 ±0.5 rad/s → 왼 0.07~0.41, 오른 0.25~0.31 rad/s 로 상태에 따라 크게 다름(시간만으로 각도 추정은 부정확)
- 폐루프 시험(IMU 자이로 적분으로 회전 각도를 재며 보정): 요청 +1.57 → 실제 +1.49, -1.57 → -1.42, +3.14 → +3.03, +0.50 → +0.37 rad. 직진 walk(0.5m) → 0.51 m(방향 -0.26 rad), walk(1.0m) → 1.06 m. 이 코드는 현재 다른 세션의 skills_node.py 로 덮어써진 상태(대화 기록에는 있음)
- 시각 정정: 이 문서의 13:08 이후 시각은 파일 수정 시각 기준 추정치 (이전 버전에서 14시대로 잘못 적었던 것을 정정)

## 2단계 현황: 기본 기술 (ROS2 Action) — 동작 확인 (21:16)
- 두 세션이 같은 파일을 동시에 수정하는 충돌이 있었음 → **한 세션만 쓰기로 결정**(사용자 선택 C). 현재 코드는 다른 세션이 만든 버전(`HeadlessDuck` + 시간 기반 보정 + 감정 동작)을 기준으로 하고, 중복 선언 버그 1건(debug_pose)만 수정함. 폐루프(IMU 자이로) turn/walk 는 코드 폐기, 측정 결과만 위 기록에 남김
- Action 7개: `turn`, `walk`, `look_at`, `happy`, `sad`, `wag_head`, `dance` (정의 `ros2_ws/src/duck_interfaces/action/`, 서버 `ros2_ws/src/duck_skills/`)
- 실행: 터미널1 `bash scripts/run_bridge.sh`, 터미널2 `bash scripts/run_skills.sh`, 터미널3 `source /opt/ros/humble/setup.bash && source ros2_ws/install/setup.bash && ros2 action send_goal /turn duck_interfaces/action/Turn "{angle: 1.57}" --feedback`
- 첫 빌드 필요: `cd ros2_ws && source /opt/ros/humble/setup.bash && colcon build`
- **정확도 시험 결과** (시뮬레이터 정답으로 측정, 기술 코드는 정답을 쓰지 않음):

| 호출 | 실제 결과 |
|---|---|
| turn(+1.57) / turn(-1.57) | +1.76 / -1.68 rad (약 +12%, +7% 오차) |
| turn(+3.14) | -2.74 rad (≡ +3.54, 약 0.4 rad 초과) |
| turn(+0.50) | +0.65 rad |
| walk(0.5) / walk(1.0) | 0.51 m / 1.02 m (방향이 -0.12~-0.15 rad 틀어짐) |
| look_at 3회 | 관절이 명령각을 따라감, 몸 이동 거의 없음 |
| happy/sad/wag_head/dance | 모두 성공, 넘어지지 않음(z≈0.17) |

- 결론: 거리는 잘 맞고, 회전은 ±0.1~0.4 rad(약 6~23°) 오차, 직진은 약 7°~8° 휨. 시간 기반 보정의 한계이며 3단계 오도메트리(IMU 적분, 카메라)로 개선 예정
- 아직 못 한 것: 감정 동작을 눈으로 보는 확인(사용자), 효과음, 정책을 바꾸면 `sim/calibrate.py`로 보정값 재측정

## 3단계 현황: 인식 (카메라만으로 물체 찾기) — 동작 확인 (21:43)
- 환경 추가: `.venv-vision`(Python 3.11, PyTorch + ultralytics + mujoco, 학습/데이터 생성용). ROS 노드는 `.venv-ros`(3.10)에서 YOLO를 **ONNX + onnxruntime** 으로 실행(PyTorch 불필요)
- 합성 데이터 `sim/gen_dataset.py`: 물체/오리 위치·고개 각도·조명·배경색 무작위, 방해 물체(다른 색 상자) 4개, 라벨은 MuJoCo 분할 렌더링에서 자동 생성(가장 큰 연결 덩어리만 사용, 분할 이미지에 같은 id 의 흩어진 픽셀이 섞여 나오는 문제 때문). 학습 5000장 + 검증 600장(약 6분), `datasets/duck_objects/`
- YOLOv8n 학습 `bash scripts/train_yolo.sh` (40 에폭, 3분 11초): 검증 mAP50 0.985 (red 0.985, green 0.985, blue 0.984). **색으로 구분하므로 hue 증강은 꺼야 함(hsv_h=0)**. 내보낸 모델 `models/yolo/duck_objects.onnx`
- 직접 만든 ONNX+numpy 추론 검증(검증 300장, conf 0.5): 재현율 97.2%, 정밀도 100%(오탐 0). 파란 침대를 파란 공으로 오탐하지 않음
- `duck_perception` 노드 (`bash scripts/run_perception.sh`): 탐지 → 깊이 중앙값으로 3D → 관절 각도 + 로봇 모델로 카메라 자세 → 몸통 좌표 → odom 좌표 → 의미 지도(`/semantic_map`), `/detections`, `/detections_image`, TF odom→base_link
- 오도메트리: 방향 = IMU 자이로 적분, 위치 = 명령 속도 x 0.7 적분(측정한 비율). 
- **발견한 버그 2건 (모두 수정)**
  1. `mujoco.mj_kinematics` 는 카메라 자세(`cam_xmat`)를 갱신하지 않음 → `mj_camlight` 도 호출해야 함. 안 하면 모든 물체가 몸통 위치(0,0,-0.15)로 찍힘
  2. **시뮬레이션이 실시간보다 느리게(약 0.82배) 돌면 벽시계 시각을 쓰는 자이로 적분이 22% 과대**. 브리지가 `/clock` 과 센서 메시지 시각을 시뮬레이션 시간으로 내보내고, 인식·기술 노드는 `use_sim_time:=true` 로 실행. 기술(`duck_skills`)의 시간 기준 보정값도 시뮬레이션 시간 기준으로 바꿈
- **정확도 (정답과 비교, 제자리 한 바퀴 + 0.8m 이동 + 한 바퀴 시험)**:

| 항목 | 결과 |
|---|---|
| 정지 상태 3D 위치(빨강 상자, 1m) | 정답 (0.96, 0.30) vs 추정 (0.94, 0.21) → 약 0.1m |
| 자이로 방향(시뮬레이션 시간 적용 후) | 정답 +2.97 rad vs +2.96 rad (turn 3회) |
| 오도메트리 위치 오차 | 제자리 회전 후 0.1~0.2m, 이동 후 0.05~0.10m |
| 지도에 기억한 물체 위치 오차 | red 0.24m, green 0.37m, blue 0.31m (종류당 1개로 병합, 관측 3회 미만은 숨김) |

- 한계: 제자리 회전 때 몸이 0.15~0.2m 실제로 이동하는데 오도메트리는 모름 → 지도 오차의 주원인. 지도는 "대략 어디쯤"용이고, 정밀 접근은 4단계에서 카메라 방향을 보며 보정(visual servoing)
- 봐야 할 것(사용자): rviz2 에서 `rviz2 -d rviz/perception.rviz` + `bash scripts/demo_perception.sh` (터미널 구성은 스크립트 상단 주석)

## 4단계 현황: 언어 → 행동 — 완료 (아래 '실제 LLM 결과' 참고. 이 줄 아래 '막힌 곳'은 해결됨) (2026-11-09 18:40)
- **막힌 곳:** Ollama(로컬 LLM 서버)를 제가 직접 받아 `~/.local/ollama` 에 풀어 두었으나(v0.40.15, GitHub 릴리스, sudo 불필요), 그 실행 파일을 돌리는 명령이 자동 권한 심사에서 거부됨 → **사용자가 직접 `bash scripts/run_ollama.sh` 실행** (또는 권한 허용) 필요. 모델은 qwen3:8b(약 5GB 다운로드)
- 만든 것
  - `duck_perception`: `/scene`(JSON) 추가 — 기억한 물체를 오리 기준 거리/방향으로 LLM 에 줌
  - `WalkTo.action` + `duck_skills/walk_to_node.py`: 카메라 시각 서보잉(방향 오차에 비례 회전, 정면일 때만 전진), 안 보이면 기억한 방향으로 돌며 찾기, 접근 중 고개 숙임(-0.35 rad), 물체가 화면 아래 가장자리에 닿으면 도착 판단
  - `duck_brain`(`planner.py` 순수 파이썬 LLM 계획기 + `brain_node.py` ROS 실행기): 도구 5개 `walk, turn, look, walk_to, emote`, 인자 검증/보정, 호출 결과를 LLM 에 돌려주는 반복(최대 6회), 'stop' 은 LLM 을 거치지 않고 즉시 취소
  - `scripts/run_ollama.sh`, `run_brain.sh`, `say.sh`(명령 보내기, 인자 없으면 대화 모드), `eval_planner.py`(LLM 계획 정확도 19문항 채점)
- **가짜 LLM 서버로 실제 시뮬레이션과 연결 검증 (정답 거리로 측정)**: walk_to red/green/blue → 멈춘 실제 거리 0.38 / 0.35 / 0.34 m (멈춤 기준 0.35 m), 8 / 14 / 17 초. green/blue 는 처음에 시야 밖이라 돌며 찾은 결과. 도구 호출 → Action 실행 → 결과를 LLM 에 반환 → 마무리 문장 흐름 확인
- **발견한 문제(수정):** `ros2 run` 으로 띄운 노드는 부모를 종료해도 자식 프로세스가 남아, 이전 실행의 두뇌/기술 노드와 중복되어 명령이 두 번 처리됨. 실행 파일을 직접 exec 하도록 run 스크립트 수정. 3단계 시험(duck_skills 를 `ros2 run` 으로 실행) 때도 남은 프로세스가 있었을 가능성은 확인하지 못함(결과는 일관됐음)
- 아직 안 한 것: 실제 LLM 으로 계획 정확도 측정(`python3 scripts/eval_planner.py`), `"go to the red block and look up"` 전체 시연
- 실행 순서(Ollama 준비 후): 터미널1 `bash scripts/run_ollama.sh`, 2 `run_bridge.sh`, 3 `run_skills.sh`, 4 `run_perception.sh`, 5 `run_brain.sh`, 6 `bash scripts/say.sh "go to the red block and look up"`

### 4단계 — 실제 LLM 결과 ($(date +%Y-%m-%d) 19:00)
- **권한 문제 해결:** 사용자가 선택지 (B) 를 고름 → `.claude/settings.local.json`(gitignore)에 좁은 허용 규칙 5개만 추가: `~/.local/ollama/bin/ollama` 의 serve/pull/run/list 와 `bash scripts/run_ollama.sh`. Ollama v0.40.15 서버 기동, `qwen3:8b`(5.2GB) 다운로드 완료. 서버는 계속 켜져 있음(GPU 약 5.5GB). 끄려면 `ps` 로 `ollama serve` 의 PID 를 찾아 kill
- **LLM 계획 정확도** (`python3 scripts/eval_planner.py`, 시뮬레이션 없이): 처음 19/20(95%), 틀린 문항은 "look to the left" 를 yaw -30(오른쪽)으로 호출. 프롬프트에 방향 예시 추가 후 20/20 — **단, 이 점수는 실패 문항을 보고 고친 것이라 부풀려져 있음.** 프롬프트에 없던 새 문항 8개(오른쪽 보기, 1/4 회전, 두 물체 순서대로, 위+왼쪽, 30cm 뒤로, 한국어 2개, 상식 질문 거절)는 8/8. 명령당 평균 0.4초(첫 호출은 모델 로딩 51초)
- **실제 LLM + 실제 시뮬레이션 전체 시연** (브리지+기술+인식+두뇌, 정답 거리로 측정):

| 명령 | LLM 호출 | 결과 |
|---|---|---|
| go to the red block and look up | walk_to(red) → look(0,30°) | 실제 0.37 m 에서 멈춤, 8초 |
| now go to the blue ball and be happy | walk_to(blue) → emote(happy) | 0.38 m, 12초 |
| walk to the green block | walk_to(green) | 0.35 m, **84초**(안 보여서 돌며 찾음) |
| 빨간 블록으로 돌아가 | walk_to(red) | 0.47 m, 7초 (화면 아래 가장자리 판정으로 일찍 멈춤) |

- 남은 문제: 안 보이는 물체 찾기가 느림(최대 84초, 제자리 회전 속도가 약 0.3 rad/s). 한 번 본 물체는 지도 방향으로 바로 돌게 했지만 /scene 힌트 활용은 단순함. 도착 거리는 0.35~0.47 m 로 편차 있음
- 아직 안 한 것: 사용자 눈 확인(`bash scripts/say.sh` 대화 모드), 도중 'stop' 취소 실사용 시험, 로드맵 4단계 항목 중 모두 완료

## 5단계 현황: 애완동물다운 자율 행동 — 동작 확인 (2026-11-09 19:25)
- 만든 것
  - `sim/build_scene.py`: 조종하는 **사람 인형**(키 약 1m, 몸통·팔·다리·머리 primitive, mocap 몸체라 충돌 없음). 브리지에 `/person/cmd_vel`(Twist) 추가, 방 안으로 제한. `scripts/person_teleop.sh` 로 키보드 조종
  - YOLO **4클래스**(red_block, green_block, blue_ball, person) 재학습: 6000+600장, 40에폭 3분 33초, 검증 mAP50 0.984 (person 0.993). ROS 쪽 ONNX+numpy 검증(400장, conf 0.5): person 재현율 0.915/정밀도 0.985, red 0.976, green 0.931, blue 0.992. (이전 3클래스 모델은 `models/yolo/duck_objects_3class.onnx` 로 보관)
  - 인식/기술/두뇌 수정: 사람은 지도에 저장하지 않고 탐지로만 사용(움직이므로), 키가 커서 화면 위/아래에 닿아도 깊이 인정, `walk_to(person)` 은 0.9m 에서 멈추고 고개를 숙이지 않음, `walk_to` 가 **마지막으로 본 방향**을 기억해 놓친 대상을 그쪽으로 먼저 돌며 찾음, 두뇌가 `/brain_busy` 발행, LLM 도구 `walk_to` 에 person 추가("come here")
  - **`duck_behavior`** (`state.py` 내부 상태 + `behavior_node.py` Behavior Tree, py_trees 2.6.0): 내부 상태 3개(심심함·에너지·외로움=사람에 대한 관심), 우선순위 ①사용자 명령 중 양보 ②에너지 부족 시 쉬기 ③사람이 보이면 (처음이면 반가워하고) 다가가기/곁에 머물기 ④방금 놓친 사람 쫓기 ⑤외로우면 사람 찾기(못 찾으면 45초 포기) ⑥심심하면 돌아다니기/물체 살펴보러 가기/두리번/춤(아주 심심하면 시무룩) ⑦그 외 가끔 두리번. **LLM 은 부르지 않음**(명령 때만 두뇌가 부름)
  - 스크립트: `run_behavior.sh`, `run_all.sh`(전체를 백그라운드로 한 번에, 로그는 `logs/`), `stop_all.sh`, `person_teleop.sh`
- **시험 결과 1 — 혼자 있을 때** (상태 변화 6배속, 150초): 외로움 0.6 에서 사람 찾기 → 못 찾아 포기 → 심심해서 돌아다니기/춤/물체 살펴보러 가기/두리번 번갈아. 70초에 LLM 명령("be happy")이 오자 **즉시 양보**, 명령 수행 후 약 10초 뒤 자율 행동 복귀. 오리 이동 범위 x[-0.2,1.0] y[-0.5,0.1]
- **시험 결과 2 — 사람이 걸어 다닐 때** (3배속, 정답 거리로 측정): 사람이 시야에 들어오자 반가워하며(happy) 다가가 곁에 머묾(외로움 0.52 → 0.00). 사람이 오리 뒤로 사라지면 **바로 '놓친 사람 쫓아가기'** 가 시작되어 약 4~6초 안에 다시 찾음(2회 모두). 사람이 움직이는 동안 오리-사람 거리 평균 1.1~1.2m(최소 0.99, 최대 1.5m), 멈춰 있을 때 0.84~1.4m. 사람 곁에 있는 중 LLM 명령 "go to the blue ball and be happy" → 즉시 양보 → 수행(0.37m 에 도착, 21초) → 끝나자 **스스로 사람에게 돌아옴**
- 알려진 한계 / 못 한 것
  - 에너지 부족 → **쉬기 행동은 단위 시험(상태 수식)만 했고 전체 시험에서는 발동시키지 못함** (에너지가 0.82 까지만 내려감). 실제 발동 확인 필요
  - 돌아다니기는 가구 위치를 **미리 알고 있는 방 배치(`AVOID` 상수)** 로 피함 → 센서 기반 장애물 회피가 아님. 오도메트리 오차가 쌓이면 방 밖/가구 쪽으로 갈 수 있음
  - 사람 인형은 물리 충돌이 없어 오리와 겹쳐 지나갈 수 있음
  - 경계 거리에서 '다가가기'/'머물기'가 1초 안에 번갈아 나타나는 짧은 떨림이 남아 있음(PERSON_CLOSE 1.45m 로 완화)
  - 효과음 없음, 사용자가 직접 눈으로 본 확인 대기
- 실행: `bash scripts/run_ollama.sh`(명령 쓸 때만) → `bash scripts/run_all.sh`(또는 `RATE=6 bash scripts/run_all.sh` 로 구경하기 좋게 빠르게) → `bash scripts/person_teleop.sh` 로 사람 조종 → `bash scripts/say.sh "go to the red block"` → `bash scripts/stop_all.sh`

## 6단계 시작: 새 RL 기술 — '앉기' 가능성 조사 (2026-11-09 19:51)
- 계획 결정: 6단계(앉기, 공 밀기, 계단, 버튼) 다음에 8단계 'VLM 개방형 탐지' 추가 (docs/ROADMAP.md 에 기록)
- 관련 논문(robotics.growbotics.ai/research, 사이트에 표시된 경로): **Eureka: Human-Level Reward Design via Coding Large Language Models** (`/research/papers/eureka-human-level-reward-design-coding-large-language-models`) — LLM 이 보상 함수 코드를 직접 쓰고 RL 결과로 개선 → 6단계 보상 설계에 응용 가능. 서 있기/앉기/계단/수영 등 직접 일치하는 논문은 없음
- **발견 (물리 시뮬레이션으로 측정):**
  1. 현재 로봇 모델에서 **바닥과 충돌하는 것은 발 2개뿐**. 몸통/머리/다리 몸체는 충돌 모델이 없음(눈에 보이기만 함) → 몸통을 땅에 내려놓는 앉기는 지금 모델로는 성립하지 않음. 머리/부리로 미는 동작도 불가능(머리 충돌 없음)
  2. 기본 서 있는 자세(home)가 이미 깊은 웅크림(무릎 약 78°). 다리만 더 접어도(엉덩이 pitch -0.63→-1.13, 무릎 1.37→1.57) 몸통 높이가 0.158→0.140m 로 **2cm 밖에 안 낮아지고**, 더 접으면 넘어짐(엉덩이 -1.03 이상 + 무릎 1.47 이하에서 전도)
  3. 몸통 바닥면이 지면에서 12.5cm 위(z=0.125m), 몸통 폭 ±5.5cm 인데 다리는 ±12.4cm 로 옆으로 튀어나옴
- 결론: '앉기'는 로봇 모델에 몸통 충돌 geom 을 추가하는 **모델 변경**이 필요함. 사용자에게 방향 확인 요청 중

## 6단계 진행: 공 밀기 (계층형 RL) — 학습 진행 중 (2026-11-09 21:20)
- 사용자 선택: '앉기'는 건너뛰고 공 밀기부터(모델 변경 필요 → 나중에 재논의). 8단계(VLM 개방형 탐지)는 6단계 뒤에 진행
- **설계: 계층형 RL.** 걷기 정책은 그대로 고정(JAX 로 복원, onnxruntime 과 오차 1.5e-6), 그 위에 상위 정책(PPO, 입력 9: 공/목표 상대위치·공 속도·직전 행동 → 출력 3: vx, vy, wz)을 학습. 상위 1스텝 = 걷기 정책 4번(0.08초), 에피소드 250스텝(20초). 보상: 공이 목표에 가까워진 만큼 x30, 공 뒤쪽 '미는 위치'(공에서 목표 반대 방향 0.2m)에 다가간 만큼 x4, 성공(0.15m) +15, 넘어짐 -5, 시간/떨림 약한 감점
- 코드: `sim/rl/` (`low_policy.py` 고정 걷기 정책, `make_xml.py` 공 장면, `push_ball_env.py` 환경, `train_push_ball.py`/`scripts/train_push_ball.sh` 학습, `export_hl.py` pkl→npz, `hl_policy.py` numpy 실행(Brax 공식 추론과 오차 7e-6), `eval_push_offline.py` 일반 MuJoCo 평가), ROS: `PushBall.action`, `duck_skills/push_ball_node.py`, LLM 도구 `push(object, target)`, 브리지 `/walk_mode`(native/override)와 `dynamic_objects` 파라미터
- **발견 1 (측정):** 컴파일 직후 첫 2번의 step 호출이 각각 재컴파일(약 40초)되어 처음엔 "초당 77 스텝"처럼 보였음 → 워밍업 3번 뒤 측정하면 원본 걷기 환경 62k 스텝/초, PushBall 상위 스텝 11.5k/초. 벤치마크는 워밍업 후에 잴 것
- **발견 2 (측정): 걷기 정책의 고개를 고정하고 관측을 가리는 배포 방식은 넘어짐을 늘림.** 3초 안에 넘어진 비율: 학습된 그대로 1.8%, 가리기 2~17%, 가리기+고개 고정 7.5~30%(회전 명령 1.0 에서 30%). → 공 밀기는 걷기 정책을 학습된 그대로 쓰고, 이 기술 실행 중에만 브리지를 `/walk_mode native` 로 전환(고개 카메라는 수평 근처)
- **발견 3 (측정): 첫 학습(2천만 스텝 중 1280만까지)에서 성공이 거의 0 이었던 원인 가설 → 확인.** 공 구름 마찰이 0.0001 이라 0.4m/s 로 찬 공이 6초에 1.69m 굴러감(제어 불가). 0.002 로 올리면 0.23m. 학습 환경·ROS 시뮬레이션 공을 모두 0.002 로 통일하고 처음부터 재학습. (첫 학습의 중간 정책은 공에 다가가 4초 만에 공-목표 0.72→0.39m 로 차는 행동까지는 했음)
- 해야 할 것: 재학습 결과 평가(MJX 학습 → 일반 MuJoCo 에서 `eval_push_offline.py` 로 성공률), ROS 전체 시험(`walk_mode native`, 인식 기반 추정으로 성공률), '카메라에 공이 안 보일 때' 내 이동량으로 공 위치를 추적하는 방식(Tracker)의 정확도

## 6단계 결과 정리: 공 밀기 (2026-11-09 22:57) — 학습/시뮬레이션 평가는 성공, ROS 인식 연동은 실패
- **세 번의 학습 시행과 교훈** (모두 `models/push/`, 각 약 35분)
  1. 순수 RL(공 구름 마찰 0.0001): 성공 ≈ 0. 원인 가설 확인 → 0.4m/s 로 찬 공이 6초에 1.69m 굴러감(제어 불가). 구름 마찰 0.002 로 수정(0.23m)
  2. 순수 RL + 마찰 수정, 3. 순수 RL + 커리큘럼(난이도 u 혼합) + 보상 조정: 둘 다 불안정(진행량 ≈ 0 또는 음수, 보상 -38 까지 붕괴). 로봇이 크게 움직이면 넘어짐 감점이 큰데 공을 맞히는 우연한 경험은 드물어 '가만히 있는' 쪽으로 학습되는 것으로 추정
  4. **잔차 RL** (사람이 설계한 제어기 `baseline_push.py` + 정책 보정 x0.5, 커리큘럼 유지): 성공 35%(제어기 단독, MJX) → 2천만 스텝에서 약 56~66%(MJX 평가)
- **일반 MuJoCo(CPU) 평가, 같은 48판, 정답 위치를 관측으로 사용** (`sim/rl/eval_push_offline.py`, 학습은 MJX 였으므로 엔진이 달라도 통하는지 확인 겸):

| | 제어기 단독 | 제어기 + RL 잔차 |
|---|---|---|
| 0.15m 안에 성공 | 9/48 (19%) | **23/48 (48%)** |
| 공이 목표에 가장 가까웠던 거리(중앙값) | 0.43 m | **0.16 m** |
| 어려운 배치(u>0.67) 성공 | 1/19 | 6/19 |
| 넘어짐 | 0 | 0 |

  제어기 단독은 성공 기준 0.3m 에서도 38%(24판, 별도 측정). MJX→MuJoCo 전이는 성공(MJX 평가 약 53% ↔ CPU 48~50%)
- **ROS 연동 (인식 결과로 동작, 정답은 측정에만 사용): 6번 시도 중 성공 0번.** 공(1.2,1.0)→빨간 상자(1.0,0.3), 80초 제한: ①제어기 단독 1번: 공 이동 1.89m 로 오히려 목표에서 멀어짐(0.73→1.57m), ②RL 3번(첫 3번, 이동량 예측을 명령값 기준): 공을 건드리지 못함 2번, 0.73→0.43m 로 가까워졌으나 성공 기준 미달 1번, ③자이로 회전 보정 후 3번: 접촉 없음 2번, 공을 2.88m 밀어 0.73→2.24m 1번
- **실패 원인 분석:** 정책이 아니라 **상태 추정**. 걷기 정책을 학습된 그대로 쓰므로 고개가 수평 근처라 공이 발 앞 약 0.45m 안으로 들어오면 카메라에서 안 보이고, 그동안은 이동량 예측(오차 누적)에 의존. 로그에서 오리가 공 주변 0.4~0.8m 를 맴도는 동안 추정값이 고정(0.91)되어 있었음. 상위 정책은 정답 관측으로 학습했기 때문에 노이즈/가림에 약함
- **개선 아이디어(미실시):** (a) 관측 노이즈와 '공이 보이지 않는 구간'을 학습에 포함(관측 가림 증강), (b) 고개 카메라를 아래로 향하게 하되 걷기 안정성 유지(현재 고개를 고정하면 넘어짐 7~30%), (c) 카메라 여러 대/넓은 화각, (d) 공의 마지막 위치를 이동량 대신 특징 기반 오도메트리로 갱신
- **6단계 나머지 항목 상태:** 앉기 = 현재 로봇 모델에 몸통 충돌이 없어 불가(위 조사). 버튼 누르기 = 머리/부리 충돌 모델이 없어 모델 변경 필요(측정: 충돌 가능한 geom 은 발 2개뿐). 계단 오르기 = 아직 조사 안 함(지형 모델 + 새 학습 필요, 발밑 높이 인식도 필요)
- 코드 위치: `sim/rl/` (`low_policy`, `make_xml`, `push_ball_env`, `train_push_ball`, `export_hl`, `hl_policy`, `baseline_push`, `eval_push_offline`), `scripts/train_push_ball.sh`, ROS `duck_skills/push_ball_node.py` + `PushBall.action`, LLM 도구 `push(object, target)`(LLM 통합은 실행 시험 안 함), 브리지 `/walk_mode`·`dynamic_objects`·`/debug/object_poses`

## 8단계 현황: VLM 개방형 탐지 — 동작 확인 (2026-11-09 23:36)
- **계획 결정(사용자):** 6단계 다음에 8단계 진행. 관련 논문(사이트에 표시된 경로): **RT-2**(`/research/papers/rt-2-vision-language-action-models`: 인터넷 규모 비전-언어 모델 지식을 로봇 제어에 사용), **OmniManip**(`/research/papers/omnimanip-general-robotic-manipulation-object-centric-interaction-primitives`: VLM 으로 고수준 추론 + 3D 기하로 정밀 조작 → 이 프로젝트의 'VLM 이 무엇을, 깊이가 어디를' 분업과 같은 구도). 개방형 탐지 자체를 다룬 논문은 사이트에 없음
- **후보 모델 비교** (`sim/bench_ovd.py`, 평가 이미지 `sim/gen_ovd_eval.py` 250장 중 150장, 새 물체 3종 + 기존 4종, 같은 이미지/IoU>=0.5):

| 모델 | 새 물체 재현율 / 정밀도 | 기존 물체 | 속도 | GPU 메모리 |
|---|---|---|---|---|
| **OWLv2 base** (선택) | 0.665 / 0.947 (주황 공 1.00/0.98, 보라 상자 0.87/0.91, 노란 컵 0.15/1.00) | 0.673 / 0.882 (사람 0.00) | 133 ms | 0.81 GB |
| OWL-ViT base | 0.540 / 0.524 | 0.704 / 0.689 (사람 0.71) | 23 ms | 0.65 GB |
| Grounding DINO tiny | 0.210 / 0.808 (라벨 매핑 방식 영향 가능) | 0.171 / 0.510 | 157 ms | 1.85 GB |

- **측정 중 발견:** (1) 새 '컵'이 처음엔 손잡이 없는 원기둥이라 모델이 컵으로 안 보고 "yellow cylinder" 35/40, "yellow object" 35/40 vs "yellow cup" 8/40 → 시험 물체를 컵답게(손잡이+어두운 입구) 고침. (2) 카메라 320x240 에서 먼 물체는 10픽셀 안팎이라 한계. (3) 같은 물체도 표현에 따라 점수가 크게 다름(orange ball 중앙값 0.72 vs orange sphere 0.52), 정탐/오탐 점수가 겹쳐 한 임계값으로 해결 불가 → '가까이서 확인' 방식 채택. (4) OWLv2 는 우리 인형 사람을 못 찾음(0.00), YOLO 는 0.92 → **하이브리드: YOLO 는 아는 4종류, OWLv2 는 YOLO 가 모르는 이름만**
- **구현:** `sim/ovd_server.py`(OWLv2 HTTP 서버, `scripts/run_ovd.sh`, 포트 8765, GPU 0.62GB. ROS 는 Python 3.10 이라 Ollama 처럼 HTTP 로 분리), 인식 노드: `/ovd_prompts` 로 질문 목록을 받아 2Hz 로 최근 프레임을 서버에 보내고 같은 깊이 파이프라인으로 3D 위치(별도 스레드, YOLO 인식은 안 막힘). 두뇌/LLM: 도구 `walk_to(object)` 가 아무 이름이나 받음(YOLO 가 모르는 이름이면 질문 등록), `walk_to` 가 임의 라벨 허용. 개방형 대상은 **가까이서 점수 >=0.25 로 확인**되어야 '도착' 인정, 화면 폭 45% 넘는 큰 상자(침대 등)는 무시, 확인 안 된 후보에는 '시야에서 사라지면 도착' 규칙 미적용(오탐 방지). 브리지 `novel_objects:=true` 로 새 물체 3개(노란 컵, 주황 공, 보라색 상자) 추가
- **종단 시험 (실제 LLM + OWLv2 + 시뮬레이션, 정답 거리로 측정; 모든 명령을 같은 출발 위치에서, 명령마다 새로 시작):**

| 명령 | LLM 호출 | 실제 거리 | 비고 |
|---|---|---|---|
| go to the orange ball | walk_to(orange ball) | 0.27 m | 목표 0.35 m |
| go to the purple box | walk_to(purple box) | 0.63 m | '시야에서 사라짐' 도착 |
| go to the yellow cup | walk_to(yellow cup) | 0.33 m | |
| go to the orange sphere (동의어) | walk_to(orange sphere) | 0.31 m | |
| go to the purple cube (동의어) | walk_to(purple cube) | 0.63 m | '시야에서 사라짐' 도착 |
| go to the yellow mug (동의어) | walk_to(yellow mug) | 0.29 m | |

  6/6 도착(0.65m 이내), 그중 4개는 목표(0.35m)±0.07m. **대조:** OWLv2 서버를 끄면 "go to the orange ball" 이 실패("찾지 못함", 37초, 실제 0.94m). **허위 탐지 시험:** 방에 없는 "pink elephant" → 찾지 못했다고 정직하게 보고(실제로 물체를 지어내지 않음)
- **재현성 주의 (정직한 기록):** 명령을 연달아 보내면 직전 명령이 끝난 위치에서 시작해서 결과가 달라짐. 같은 코드로 6개 연속 실행 시 3/6~6/6 사이에서 변동(먼 작은 물체는 보이지 않아 한 바퀴 돌고 실패, 노란 컵과 'mug' 표현이 특히 불안정). 앞서 한 번 3~5초 만에 실패한 4건은 이전 실행의 노드가 남아 있었던 것으로 추정(가설, 확인 못 함) → 시험 스크립트를 kill -9 + 노드 0개 확인 후 시작하도록 바꿈
- 한계: 작은 물체는 약 2m 이상에서 안 보임(320x240), 못 찾으면 제자리에서 돌기만 하고 **탐험(다른 곳으로 이동)은 안 함**, 표현에 따라 점수 차이 큼(예: mug), 먼 거리 오탐은 '가까이서 확인'으로만 거름, 사람은 OWLv2 가 못 찾으므로 YOLO 의존
- 실행: `bash scripts/run_ollama.sh`, `bash scripts/run_all.sh`(OWLv2 서버 자동 시작) 후 새 물체가 있는 방은 `bash scripts/run_bridge.sh --ros-args -p novel_objects:=true`(run_all 은 기본 방에서 시작하므로 이 옵션은 브리지를 따로 켤 때), `bash scripts/say.sh "go to the yellow cup"`

## 6단계 공 밀기 재학습 결과 (2026-11-10 00:34) — 개선 실패. 원인은 정책이 아니라 상태 추정
- **시도:** ROS 실패 원인(상태 추정)을 학습에 반영. 학습 환경이 정답 대신 **추정값**만 보게 함(`sim/rl/push_ball_env.py`, 이전 버전은 `push_ball_env_v1_clean_obs.py`): ROS 노드 `Tracker` 와 같은 수식(보이면 측정과 0.5:0.5, 안 보이면 이동량x0.7 + 자이로 회전량으로 예측), 보임 조건(화각 ±40°, 0.45~3m, 10% 누락), 거리 비례 측정 오차(0.03+0.08d). 보상은 정답으로 계산. 잔차 RL + 커리큘럼 유지, 2천만 스텝 약 40분
- **결과 (같은 48판, '추정값 관측' 조건의 CPU MuJoCo 평가, 성공 0.15m):**

| 정책 | 성공 | 공이 목표에 가장 가까웠던 거리(중앙값) |
|---|---|---|
| 제어기 단독 | 13/48 (27%) | 0.30 m |
| 이전 잔차 RL(정답 관측으로 학습) | **19/48 (40%)** | 0.23 m |
| 새 잔차 RL(추정값 관측으로 학습) | 6/48 (**12%**) | 0.37 m |

  새 정책은 MJX 학습 중 성공률도 21~32% 로 시작값(제어기 단독 약 18%)에서 크게 오르지 않았음. 잡음/가림을 넣자 **오히려 제어기 단독보다 나빠짐.** (참고: 이전 정책은 정답 관측으로 평가하면 48%)
- **ROS 연동(공 1.2,1.0 → 빨간 상자 1.0,0.3, 80초 제한):** 새 정책 **0/4**(접촉 없음 2, 공을 2.18m 밀어 멀어짐 1, **허위 성공 1**: 노드가 성공 보고했으나 정답상 공은 0.00m 움직임). 이전 정책 0/6(앞 기록). → 합계 ROS 성공 0/10
- **ROS 추정 오차 측정** (푸시 노드가 `/push_debug` 로 추정값을 내보내고 정답과 비교, 70초, 862개 표본):

| | 오차 중앙값 / 90% | 평균 거리 | 비율 |
|---|---|---|---|
| 공이 보일 때 | 0.14 m / 0.37 m | 1.03 m | 28% |
| 공이 안 보일 때(예측만) | **0.51 m / 1.57 m** | 0.42 m | 72% |
| 목표가 보일 때 | 0.31 m / 0.66 m | 0.96 m | **6%** |
| 목표가 안 보일 때 | 0.33 m / 1.25 m | 0.88 m | 94% |

  오프라인 모델의 같은 지표는 안 보일 때 0.12 m / 0.52 m(이전 정책, 20초 에피소드) → **ROS 의 예측 오차가 모델의 약 4배.** 공이 안 보일 때 오차(0.51m)가 공까지의 거리(0.42m)보다 큼 = 가까이서 밀 때 공이 어디 있는지 사실상 모름. 원인 후보: (a) ROS 시험은 80초 동안 헤매 오차가 계속 쌓임(오프라인 에피소드는 20초/성공 시 종료), (b) 명령 대비 실제 이동 응답·검출 지연 등 모델에 없는 오차. 어느 쪽인지는 **확인하지 못함**
- **고침:** 허위 성공 방지 — 푸시 노드는 공과 목표가 둘 다 '지금 카메라에 보여' 측정으로 확인될 때만 성공을 선언(못 보면 계속 시도하다 시간 초과). 기본 정책은 오프라인에서 더 좋았던 이전 정책(`models/push/20261109_215957`)으로 지정
- **결론 (정직하게):** 공 밀기는 정답 위치가 주어지는 시뮬레이션(정답 관측 48%)에서는 동작하지만, 카메라 인식으로 돌리는 ROS 에서는 **현재 구조로 성공하지 못함(0/10)**. 학습을 더 하거나 잡음을 더 넣는 것은 효과가 없었음(12%). 병목은 인식/추정: 걷기 정책을 학습된 그대로 쓰면 고개가 수평이라 발 앞 0.45m 안은 안 보이고 목표는 94% 안 보임. 현실적인 다음 수단은 정책 학습이 아니라 **센서/구조 변경**(아래를 향하는 두 번째 카메라나 넓은 화각, 다리 오도메트리 등 더 나은 이동량 추정, 공 뒤로 돌아가기를 `walk_to` 처럼 폐루프 시각 서보로 분해)
