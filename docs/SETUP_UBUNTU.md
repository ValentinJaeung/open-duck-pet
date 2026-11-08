# Ubuntu 설치 가이드

듀얼 부팅 Ubuntu에서 이 프로젝트를 처음부터 세팅하는 순서입니다.
기준 환경은 **Ubuntu 24.04 LTS + NVIDIA RTX 3080(10GB)** 입니다.
ROS2 Jazzy가 24.04용이라 이 버전을 권장합니다.

---

## 1. NVIDIA 드라이버

```bash
sudo apt update
sudo ubuntu-drivers install      # 권장 드라이버 자동 설치
sudo reboot
nvidia-smi                       # GPU 이름과 드라이버 버전이 나오면 성공
```

> **Secure Boot가 켜져 있으면** 드라이버 설치 중 비밀번호를 설정하라는 화면이 나옵니다.
> 재부팅할 때 파란 화면(MOK 관리)이 뜨면 `Enroll MOK`를 선택하고 그 비밀번호를 입력하세요.
> 이 과정을 건너뛰면 드라이버가 로드되지 않습니다.

CUDA Toolkit은 따로 설치하지 않아도 됩니다. JAX가 필요한 CUDA 라이브러리를 pip 패키지로 함께 설치합니다.

## 2. 기본 도구

```bash
sudo apt install -y git curl build-essential
# GitHub CLI (선택. push 할 때 로그인이 편해짐)
sudo apt install -y gh && gh auth login
```

## 3. conda

이미 Anaconda를 쓰고 있다면 Ubuntu에도 Anaconda를 설치하면 됩니다.

```bash
# Anaconda (https://www.anaconda.com/download 에서 Linux 설치 파일 받기)
bash ~/Downloads/Anaconda3-*-Linux-x86_64.sh
# 설치가 끝나면 터미널을 새로 열기
conda --version
```

> 이 프로젝트의 환경 파일(`env/duck-train.yml`)은 **conda-forge 채널만** 씁니다.
> 그래서 Anaconda 기본 채널 약관(ToS) 동의 없이도 환경을 만들 수 있습니다.
> 더 가벼운 대안으로 [Miniforge](https://github.com/conda-forge/miniforge)도 있습니다.

## 4. 저장소 받기

**`--recursive`를 꼭 붙이세요.** 서브모듈(Open_Duck_Playground, Open_Duck_Mini)까지 함께 받습니다.

```bash
mkdir -p ~/Projects && cd ~/Projects
git clone --recursive https://github.com/ValentinJaeung/open-duck-pet.git
cd open-duck-pet
```

`--recursive` 없이 받았다면 다음을 실행하세요.

```bash
git submodule update --init --recursive
```

## 5. 학습 환경 만들기

```bash
bash scripts/setup_train_env.sh
```

이 스크립트가 하는 일은 다음과 같습니다.
1. 서브모듈 확인
2. conda 환경 `duck` 생성 (Python 3.11 + 검증된 버전 고정 패키지, 10분 이상 걸림)
3. 환경 변수 설정
   - `PYTHONPATH`: Open_Duck_Playground 코드를 import 할 수 있게 함
   - `XLA_PYTHON_CLIENT_MEM_FRACTION=0.75`: JAX가 미리 잡는 GPU 메모리 비율
4. GPU 인식 확인 + 오리 환경 한 스텝 실행

마지막에 `JAX devices: [CudaDevice(id=0)]`와 `환경 OK`가 나오면 성공입니다.

```bash
conda activate duck
```

## 6. 걷기 정책 학습

```bash
# (1) 동작 확인용 짧은 학습 (몇 분)
bash scripts/train_walk.sh 1000000

# (2) 본 학습 (3억 스텝, RTX 3080 기준 약 2시간)
bash scripts/train_walk.sh
```

- 결과는 `models/runs/<날짜>_flat_terrain_backlash/`에 저장됩니다.
- 처음 몇 분은 JIT 컴파일 때문에 아무 출력이 없을 수 있습니다. 정상입니다.
- 학습 그래프는 다른 터미널에서 다음 명령으로 보고, 브라우저에서 http://localhost:6006 을 엽니다.
  ```bash
  conda activate duck
  tensorboard --logdir=models/runs
  ```

## 7. 학습된 정책 보기

```bash
bash scripts/view_policy.sh models/runs/<폴더>/<날짜>_<스텝>.onnx
```

MuJoCo 뷰어 창이 뜨면 다음 키로 조종합니다.
- 화살표: 이동
- `a` / `e`: 회전
- `h`: 고개 조종 모드 전환

---

## 다음: ROS2 (1단계 시작할 때)

> 아래는 1단계에서 설정할 예정입니다. **아직 이 저장소에서 검증하지 않았습니다.**

ROS2는 Ubuntu 시스템 Python(3.12)에 맞춰 빌드되어 있어서 conda의 Python 3.11 환경에서는 `import rclpy`가 안 됩니다.
그래서 **RoboStack**(conda-forge로 ROS2를 설치하는 프로젝트)으로 **실행용 환경을 따로** 만들 계획입니다.

| 환경 | 용도 |
|---|---|
| `duck` | RL 학습 (MJX/JAX, GPU 집중 사용) |
| `duck_ros` | 실행: ROS2 Jazzy + MuJoCo + ONNX 정책 + YOLO + LLM 연결 |

문제가 생기면 [TROUBLESHOOTING.md](TROUBLESHOOTING.md)를 보세요.
