# 문제 해결 기록

실제로 겪은 문제와 해결 방법입니다. 새로 겪는 문제도 여기에 계속 추가합니다.

---

### `ModuleNotFoundError: No module named 'playground'`

**원인 1:** `duck` 환경이 아니라 `(base)`에서 실행했습니다.
```bash
conda activate duck      # 프롬프트가 (duck) 으로 바뀌는지 확인
echo $PYTHONPATH         # .../third_party/Open_Duck_Playground 가 나와야 함
```

**원인 2:** 환경 변수를 바꾼 뒤 환경을 다시 활성화하지 않았습니다.
```bash
conda deactivate && conda activate duck
```

---

### `ModuleNotFoundError: No module named 'mujoco_playground._src.collision'`

**원인:** `playground`(MuJoCo Playground) 0.1 이상에서 이 모듈이 없어졌습니다.
Open_Duck_Playground 코드는 2025년 8월 기준입니다.

**해결:** `playground==0.0.5`로 고정합니다 (`env/requirements-train.txt`에 이미 반영).

---

### `AttributeError: jax.device_put_replicated is deprecated`

**원인:** JAX 0.10에서 이 함수가 제거됐는데 Brax PPO 학습 코드가 아직 사용합니다.

**해결:** 2025년 중반 버전 조합으로 고정합니다 (`env/requirements-train.txt`에 이미 반영).

| 패키지 | 버전 |
|---|---|
| jax / jaxlib (cuda12) | 0.6.2 |
| mujoco / mujoco-mjx | 3.3.4 |
| brax | 0.12.4 |
| flax | 0.10.7 |
| playground | 0.0.5 |

> **패키지를 업그레이드하지 마세요.** `pip install -U ...`로 하나만 올려도 위 에러들이 다시 납니다.

---

### `CUDA_ERROR_OUT_OF_MEMORY: Failed to allocate device memory of 8.50GiB`

**원인:** JAX는 시작할 때 GPU 메모리를 정해진 비율만큼 미리 잡습니다.
화면 출력 등이 이미 GPU 메모리를 쓰고 있으면 그만큼을 확보하지 못합니다.
WSL2에서는 Windows 화면 출력이 약 1.4GB를 쓰고 있어서 85%가 실패했습니다.

**해결:** 비율을 낮춥니다.
```bash
conda env config vars set -n duck XLA_PYTHON_CLIENT_MEM_FRACTION=0.70
conda deactivate && conda activate duck
```

학습 중에 메모리가 부족하면 병렬 환경 수(`num_envs`, 기본 8192)를 줄여야 합니다.
이 값은 `third_party/Open_Duck_Playground/playground/common/runner.py`의 PPO 설정에서 정해집니다.

---

### conda 환경 생성 시 `CondaToSNonInteractiveError`

**원인:** Anaconda 기본 채널(`repo.anaconda.com`)은 약관 동의가 필요합니다.

**해결:** 이 저장소의 `env/duck-train.yml`은 conda-forge만 쓰므로 그대로 쓰면 됩니다.
직접 환경을 만들 때는 다음처럼 채널을 지정하세요.
```bash
conda create -n 이름 --override-channels -c conda-forge python=3.11
```

---

### 경고: 무시해도 되는 것들

- `oneDNN custom operations are on` (TensorFlow 안내)
- `overflow encountered in cast`
- `All configs were filtered out because none of them sufficiently match the hints` (XLA 자동 튜닝)
- `Delay kernel timed out: measured time has sub-optimal accuracy`

---

### 학습 스텝이 지정한 값보다 더 진행됨

정상입니다. Brax는 한 번 반복할 때 `num_envs × unroll_length`(8192 × 20 = 16만) 스텝씩 진행합니다.
평가 횟수(`num_evals=15`)에 맞추느라 지정한 값보다 조금 더 돌 수 있습니다.
