"""걷기/회전 시간 보정값.

이 걷기 정책은 속도 명령을 그대로 따르지 않는다(명령 0.15 m/s -> 실제 약 0.11 m/s, 한쪽으로 휘어짐,
뒤로는 못 걸음, 왼/오른쪽 회전 속도가 다름). 그래서 "속도 x 시간" 계산 대신 시뮬레이션에서 실측한
관계식  시간 = 양 / 속도 + 오프셋  을 쓴다. (오프셋 = 멈춘 상태에서 출발할 때 가속에 드는 시간)

측정 방법: sim/calibrate.py (정답 위치를 읽는 오프라인 측정. 기술 실행 중에는 정답 위치를 쓰지 않음)
측정 대상 정책: models/runs/20261108_113921_flat_terrain_backlash/2026_11_08_130526_300482560.onnx
다시 학습하거나 정책을 바꾸면 sim/calibrate.py 를 돌려 이 값을 갱신해야 한다.
"""
# 직진: 명령 vx=0.15 에 wz=+0.1 을 더해야 똑바로 감 (안 더하면 오른쪽으로 휘어 -0.16 rad/s)
WALK_VX = 0.15
WALK_WZ_BIAS = 0.10
WALK_SPEED = 0.1126      # m/s
WALK_OFFSET = 0.25       # s
WALK_MIN, WALK_MAX = 0.05, 5.0   # m

# 제자리 회전: wz=+-1.0 (최대). 왼쪽(+)이 오른쪽(-)보다 느림
TURN_WZ = 1.0
TURN_LEFT_RATE, TURN_LEFT_OFFSET = 0.563, 0.26      # rad/s, s
TURN_RIGHT_RATE, TURN_RIGHT_OFFSET = 0.757, 0.14
TURN_MAX = 4 * 3.141592653589793   # 2바퀴

# 고개 (bridge /head_cmd 와 같은 부호: yaw +왼쪽, pitch +위. 시뮬레이션으로 확인함)
HEAD_YAW_RANGE = (-1.5, 1.5)
HEAD_PITCH_RANGE = (-0.78, 0.78)    # head_pitch 만 사용. 목(neck_pitch)은 쓰지 않음
GAZE_SPEED = 1.5                    # rad/s 고개 이동 속도
