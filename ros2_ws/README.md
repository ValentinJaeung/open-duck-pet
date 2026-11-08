# ros2_ws/

ROS2 워크스페이스입니다. 1단계부터 여기에 패키지를 만듭니다. 아직 비어 있습니다.

예정된 패키지 (자세한 내용은 [docs/ROADMAP.md](../docs/ROADMAP.md)):

| 패키지 | 단계 | 역할 |
|---|---|---|
| `duck_sim_bridge` | 1 | MuJoCo 실행 + 카메라/IMU/관절 상태 발행, `/cmd_vel`·`/head_cmd` 구독 → RL 정책 |
| `duck_skills` | 2 | `turn`, `walk`, `look_at`, `happy`, `dance` 등 ROS2 Action 서버 |
| `duck_perception` | 3 | YOLO 물체 탐지 + 깊이 → 물체 3D 위치, 의미 지도 |
| `duck_brain` | 4 | 로컬 LLM(Ollama) 도구 호출 → 기술 Action 실행 |
| `duck_behavior` | 5 | Behavior Tree, 내부 상태(심심함/에너지) |
| `duck_voice` | 7 | faster-whisper 음성 인식 |
