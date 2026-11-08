# open-duck-pet 🦆

[Open Duck Mini V2](https://github.com/apirrone/Open_Duck_Mini) 로봇을 MuJoCo 시뮬레이션 안의 **애완동물**로 만드는 개인 프로젝트입니다.

자연어(키보드 → 음성)로 명령하면 오리가 카메라로 환경을 인식하고 행동합니다.
명령이 없을 때는 스스로 돌아다니고, 다가오고, 심심해합니다.

```
"go to the red block and look up"
   → 로컬 LLM이 기술 선택 → ROS2 Action → RL 걷기 정책 → MuJoCo 속 오리가 움직임
```

## 현재 상태

| 단계 | 내용 | 상태 |
|---|---|---|
| 준비 | RL 걷기 정책 학습 (MJX + PPO, 3억 스텝) | ✅ 완료 (WSL2) |
| 0 | Ubuntu 환경 구축 | ⏳ 진행 예정 |
| 1~7 | 방 모델링, ROS2, 인식, LLM, 자율 행동, 새 기술, 음성 | 📋 [로드맵](docs/ROADMAP.md) |

## 빠른 시작 (Ubuntu 24.04 + NVIDIA GPU)

```bash
git clone --recursive https://github.com/ValentinJaeung/open-duck-pet.git
cd open-duck-pet

bash scripts/setup_train_env.sh      # conda 환경 'duck' 생성 + 동작 확인
conda activate duck

bash scripts/train_walk.sh 1000000   # 짧은 동작 확인 학습
bash scripts/train_walk.sh           # 본 학습 (RTX 3080 기준 약 2시간)

bash scripts/view_policy.sh models/runs/<폴더>/<파일>.onnx   # 뷰어로 보기
```

드라이버 설치부터 자세한 순서는 [docs/SETUP_UBUNTU.md](docs/SETUP_UBUNTU.md)에 있습니다.

## 폴더 구조

```
open-duck-pet/
├── docs/
│   ├── ROADMAP.md            # 전체 계획과 단계별 체크리스트
│   ├── SETUP_UBUNTU.md       # Ubuntu 설치 가이드
│   └── TROUBLESHOOTING.md    # 겪었던 문제와 해결법
├── env/
│   ├── duck-train.yml        # 학습용 conda 환경 (conda-forge)
│   └── requirements-train.txt  # 검증된 버전 고정 목록
├── scripts/
│   ├── setup_train_env.sh    # 환경 생성 + 환경 변수 + 동작 확인
│   ├── train_walk.sh         # 걷기 정책 학습
│   └── view_policy.sh        # 학습된 정책을 뷰어로 보기
├── ros2_ws/                  # ROS2 패키지 (1단계부터)
├── models/                   # 학습 결과 (커밋 안 함)
└── third_party/              # 원본 저장소 (git 서브모듈, 커밋 고정)
    ├── Open_Duck_Playground/ # RL 학습 코드 (MuJoCo Playground 기반)
    └── Open_Duck_Mini/       # 로봇 하드웨어/문서 (sim2real 때 참고)
```

## 버전 주의

Open_Duck_Playground는 2025년 8월 코드라 최신 패키지와 호환되지 않습니다.
`env/requirements-train.txt`의 버전(jax 0.6.2, mujoco 3.3.4, brax 0.12.4, playground 0.0.5)을 **올리지 마세요.**
이유는 [TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md)에 있습니다.

## 크레딧

- [Open Duck Mini](https://github.com/apirrone/Open_Duck_Mini), [Open Duck Playground](https://github.com/apirrone/Open_Duck_Playground): Antoine Pirrone 외
- [MuJoCo Playground](https://github.com/google-deepmind/mujoco_playground), [Brax](https://github.com/google/brax)
