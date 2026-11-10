# 카메라로 본 물체 하나
string label              # red_block, green_block, blue_ball
float32 score             # 탐지 신뢰도 0~1
float32 u                 # 상자 중심 (픽셀)
float32 v
float32 width             # 상자 크기 (픽셀)
float32 height
float32 depth             # 깊이 카메라로 잰 거리(m, 카메라 앞 방향). 못 재면 0
geometry_msgs/Point base_xyz   # 오리 몸통 기준 위치 (x 앞, y 왼쪽, z 위, m). depth 가 0 이면 0
geometry_msgs/Point odom_xyz   # 오도메트리 좌표계(시작 위치 기준) 위치 추정치
