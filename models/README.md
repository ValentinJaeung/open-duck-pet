# models/

학습 결과물이 저장되는 폴더입니다. **저장소에는 커밋하지 않습니다** (`.gitignore` 처리).

`scripts/train_walk.sh` 를 실행하면 `models/runs/<날짜>_<task>/` 아래에 다음이 생깁니다.

| 파일 | 설명 |
|---|---|
| `<날짜>_<스텝>/` | Orbax 체크포인트 (학습 이어하기용, `--restore_checkpoint_path`) |
| `<날짜>_<스텝>.onnx` | 추론용 정책. `scripts/view_policy.sh` 나 ROS2 노드에서 사용 |
| `events.out.tfevents.*` | TensorBoard 로그 |
| `train.log` | 콘솔 출력 |

## 참고 기록 (WSL2 + RTX 3080, 2026-11-06)

`flat_terrain_backlash`, 3억 스텝, 약 2시간.

| 지표 | 시작 | 끝 |
|---|---|---|
| eval/episode_reward | 14 | 약 270~300 (1.5억 스텝 이후 정체) |
| reward/imitation | -187 | 919 |
| reward/tracking_lin_vel | 21 | 851 |
| reward/tracking_ang_vel | 14 | 715 |
| cost/action_rate | 194 | 374 (동작 떨림 가능성, 확인 필요) |
