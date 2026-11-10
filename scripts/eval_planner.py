"""LLM 계획 정확도 평가 (시뮬레이션/ROS 없이). Ollama 서버가 떠 있어야 한다.
사용: python3 scripts/eval_planner.py [모델명]
각 명령에 대해 호출한 도구 순서/인자가 기대와 맞는지(허용 오차 포함) 채점하고, 틀린 것을 보여 준다.
"""
import os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "ros2_ws", "src", "duck_brain"))
from duck_brain.planner import Planner, run_agent

SCENE = {"objects": [{"label": "red_block", "distance_m": 1.0, "bearing_deg": 17, "visible_now": True},
                     {"label": "blue_ball", "distance_m": 1.6, "bearing_deg": 40, "visible_now": False}]}
near = lambda a, b, tol: abs(a - b) <= tol
# (명령, 검사 함수(호출 리스트) -> bool)
CASES = [
    ("go to the red block and look up", lambda c: [x[0] for x in c] == ["walk_to", "look"] and c[0][1]["object"] == "red_block" and c[1][1]["pitch_deg"] > 10),
    ("go to the blue ball", lambda c: [x[0] for x in c] == ["walk_to"] and c[0][1]["object"] == "blue_ball"),
    ("walk to the green block", lambda c: [x[0] for x in c] == ["walk_to"] and c[0][1]["object"] == "green_block"),
    ("walk forward 1 meter", lambda c: [x[0] for x in c] == ["walk"] and near(c[0][1]["distance_m"], 1.0, 0.1)),
    ("go back half a meter", lambda c: [x[0] for x in c] == ["walk"] and near(c[0][1]["distance_m"], -0.5, 0.1)),
    ("turn left 90 degrees", lambda c: [x[0] for x in c] == ["turn"] and near(c[0][1]["angle_deg"], 90, 5)),
    ("turn right 45 degrees", lambda c: [x[0] for x in c] == ["turn"] and near(c[0][1]["angle_deg"], -45, 5)),
    ("do a full spin", lambda c: [x[0] for x in c] == ["turn"] and near(abs(c[0][1]["angle_deg"]), 360, 5)),
    ("look to the left", lambda c: [x[0] for x in c] == ["look"] and c[0][1]["yaw_deg"] > 20),
    ("look down", lambda c: [x[0] for x in c] == ["look"] and c[0][1]["pitch_deg"] < -10),
    ("look straight ahead", lambda c: [x[0] for x in c] == ["look"] and near(c[0][1]["yaw_deg"], 0, 3) and near(c[0][1]["pitch_deg"], 0, 3)),
    ("be happy!", lambda c: [x[0] for x in c] == ["emote"] and c[0][1]["emotion"] == "happy"),
    ("dance for me", lambda c: [x[0] for x in c] == ["emote"] and c[0][1]["emotion"] == "dance"),
    ("you did something bad, feel sad", lambda c: [x[0] for x in c] == ["emote"] and c[0][1]["emotion"] == "sad"),
    ("turn around then walk forward 2 meters", lambda c: [x[0] for x in c] == ["turn", "walk"] and near(abs(c[0][1]["angle_deg"]), 180, 10) and near(c[1][1]["distance_m"], 2.0, 0.1)),
    ("go to the red block and be happy", lambda c: [x[0] for x in c] == ["walk_to", "emote"] and c[1][1]["emotion"] == "happy"),
    ("빨간 블록으로 가서 위를 봐", lambda c: [x[0] for x in c] == ["walk_to", "look"] and c[0][1]["object"] == "red_block" and c[1][1]["pitch_deg"] > 10),
    ("파란 공 쪽으로 가", lambda c: [x[0] for x in c] == ["walk_to"] and c[0][1]["object"] == "blue_ball"),
    ("fly to the moon", lambda c: len(c) == 0),                      # 불가능한 요청: 도구 호출 없이 거절해야 함
    ("go to the yellow cube", lambda c: len(c) == 0 or all(x[0] != "walk_to" or x[1]["object"] in ("red_block", "green_block", "blue_ball") for x in c)),
]

def main():
    model = sys.argv[1] if len(sys.argv) > 1 else "qwen3:8b"
    pl = Planner(model)
    ok = 0; total_t = 0.0
    for cmd, check in CASES:
        t = time.time()
        reply, calls = run_agent(pl, cmd, SCENE, lambda n, a: "ok: done", log=lambda *_: None)
        dt = time.time() - t; total_t += dt
        good = check([(n, a) for n, a, _ in calls]); ok += good
        print(f"{'✅' if good else '❌'} {cmd!r:48s} {dt:4.1f}s → {[(n, a) for n, a, _ in calls]}" + ("" if good else f"  | reply: {reply[:60]!r}"))
    print(f"\n정답 {ok}/{len(CASES)} ({100 * ok / len(CASES):.0f}%), 명령당 평균 {total_t / len(CASES):.1f}s, 모델 {model}")

if __name__ == "__main__":
    main()
