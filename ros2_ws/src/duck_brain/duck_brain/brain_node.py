"""duck_brain: 자연어 명령 → (로컬 LLM 도구 호출) → ROS2 Action 실행.

입력:  /user_command (std_msgs/String)  예) "go to the red block and look up"   ← scripts/say.sh
       /scene        (std_msgs/String)  인식 노드가 내보내는 기억한 물체 목록(JSON)
출력:  /brain_log    (std_msgs/String)  진행 상황(어떤 도구를 호출했고 결과가 어땠는지, 최종 답변)
"""
import json
import queue
import re
import threading
import time

import rclpy
from rclpy.action import ActionClient
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from std_msgs.msg import Bool, String

from duck_interfaces.action import Dance, Happy, LookAt, PushBall, Sad, Turn, WagHead, Walk, WalkTo
from duck_brain.planner import Planner, run_agent

YOLO_CLASSES = {"red_block", "green_block", "blue_ball", "person"}
STOP_WORDS = {"stop", "halt", "cancel", "멈춰", "그만", "정지"}
ACTION_TIMEOUT = 330.0     # s (실제 시간). walk_to 는 탐험하면 최대 220 시뮬레이션 초(시뮬레이션이 실시간의 약 0.8배)


class Brain(Node):
    def __init__(self):
        super().__init__("duck_brain")
        self.declare_parameter("model", "qwen3:8b")
        self.declare_parameter("host", "http://localhost:11434")
        self.planner = Planner(self.get_parameter("model").value, self.get_parameter("host").value)
        cb = ReentrantCallbackGroup()
        self.scene = {}
        self.create_subscription(String, "/scene", lambda m: self._set_scene(m), 1, callback_group=cb)
        self.create_subscription(String, "/user_command", self.on_command, 10, callback_group=cb)
        self.pub_log = self.create_publisher(String, "/brain_log", 10)
        self.pub_ovd = self.create_publisher(String, "/ovd_prompts", 5)     # YOLO 가 모르는 물체 이름을 인식 노드(OWLv2)에 전달
        self.ovd_active = []
        self.pub_busy = self.create_publisher(Bool, "/brain_busy", 10)     # 명령 처리 중이면 자율 행동은 양보
        self.busy_n = 0
        self.acts = {
            "walk": ActionClient(self, Walk, "walk", callback_group=cb),
            "turn": ActionClient(self, Turn, "turn", callback_group=cb),
            "look": ActionClient(self, LookAt, "look_at", callback_group=cb),
            "walk_to": ActionClient(self, WalkTo, "walk_to", callback_group=cb),
            "push": ActionClient(self, PushBall, "push_ball", callback_group=cb),
            "happy": ActionClient(self, Happy, "happy", callback_group=cb),
            "sad": ActionClient(self, Sad, "sad", callback_group=cb),
            "wag_head": ActionClient(self, WagHead, "wag_head", callback_group=cb),
            "dance": ActionClient(self, Dance, "dance", callback_group=cb),
        }
        self.q = queue.Queue()
        self.cur_gh = None
        self.stop_flag = False
        threading.Thread(target=self.worker, daemon=True).start()
        self.log(f"두뇌 준비 (모델 {self.planner.model}). /user_command 로 명령을 보내세요.")

    def _set_scene(self, m):
        try:
            self.scene = json.loads(m.data)
        except Exception:
            pass

    def log(self, text):
        self.get_logger().info(text)
        self.pub_log.publish(String(data=text))

    # ---- 명령 접수 ----
    def on_command(self, msg):
        cmd = msg.data.strip()
        if not cmd:
            return
        if cmd.lower() in STOP_WORDS:                       # 멈춤은 LLM 을 거치지 않고 즉시 처리
            self.stop_flag = True
            while not self.q.empty():
                self.q.get_nowait()
            if self.cur_gh is not None:
                self.cur_gh.cancel_goal_async()
            self.pub_busy.publish(Bool(data=False))
            self.log("■ 멈춤: 실행 중인 동작을 취소하고 대기열을 비웠습니다.")
            return
        self.pub_busy.publish(Bool(data=True))                # LLM 응답을 기다리는 동안에도 자율 행동을 멈추게 함
        self.q.put(cmd)
        self.log(f"명령 접수: {cmd!r}" + (f" (대기열 {self.q.qsize()})" if self.q.qsize() > 1 else ""))

    def worker(self):
        while True:
            cmd = self.q.get()
            self.stop_flag = False
            self.pub_busy.publish(Bool(data=True))
            t0 = time.time()
            try:
                reply, done = run_agent(self.planner, cmd, self.scene, self.execute, log=self.log)
                self.log(f"💬 {reply}   ({len(done)}개 동작, {time.time() - t0:.1f}s)")
            except Exception as e:          # LLM 서버가 꺼져 있는 경우 등
                self.log(f"⚠ 명령 처리 실패: {type(e).__name__}: {e}")
            if self.q.empty():
                self.pub_busy.publish(Bool(data=False))

    # ---- 도구 → Action ----
    def execute(self, name, a):
        if self.stop_flag:
            return "failed: stopped by user"
        if name == "walk":
            return self.call("walk", Walk.Goal(distance=float(a["distance_m"])))
        if name == "turn":
            return self.call("turn", Turn.Goal(angle=float(a["angle_deg"]) * 0.017453292519943295))
        if name == "look":
            return self.call("look", LookAt.Goal(yaw=float(a["yaw_deg"]) * 0.017453292519943295,
                                                  pitch=float(a["pitch_deg"]) * 0.017453292519943295))
        if name == "walk_to":
            label = re.sub(r"[^a-z0-9]+", "_", a["object"].lower()).strip("_")
            if label not in YOLO_CLASSES:                       # YOLO 가 모르는 이름 → 개방형 탐지에 질문 등록(최근 5개 유지)
                self.ovd_active = [x for x in self.ovd_active if x != label][-4:] + [label]
                self.pub_ovd.publish(String(data=json.dumps(self.ovd_active)))
                self.log(f"개방형 탐지(OWLv2)에 '{label.replace('_', ' ')}' 질문 등록")
            return self.call("walk_to", WalkTo.Goal(target=label, stop_distance=0.0))
        if name == "push":
            return self.call("push", PushBall.Goal(ball=a["object"], goal=a["target"]))
        if name == "emote":
            e = a["emotion"]
            goal = {"happy": Happy.Goal, "sad": Sad.Goal, "wag_head": WagHead.Goal, "dance": Dance.Goal}[e]()
            if e == "happy":
                goal.intensity = 1.0
            elif e == "wag_head":
                goal.times = 3
            elif e == "dance":
                goal.duration = 6.0
            return self.call(e, goal)
        return f"failed: unknown tool {name}"

    def call(self, key, goal):
        cl = self.acts[key]
        if not cl.wait_for_server(timeout_sec=3.0):
            return f"failed: action server '{key}' is not running"
        ev, out = threading.Event(), {}

        def on_result(f):
            out["r"] = f.result().result
            out["status"] = f.result().status
            ev.set()

        def on_goal(f):
            gh = f.result()
            self.cur_gh = gh
            if not gh.accepted:
                out["rejected"] = True
                ev.set()
                return
            gh.get_result_async().add_done_callback(on_result)

        cl.send_goal_async(goal).add_done_callback(on_goal)
        if not ev.wait(ACTION_TIMEOUT):
            if self.cur_gh is not None:
                self.cur_gh.cancel_goal_async()
            return "failed: timeout"
        self.cur_gh = None
        if out.get("rejected"):
            return "failed: goal rejected"
        r = out["r"]
        return ("ok: " if r.success else "failed: ") + (r.message or "")


def main():
    rclpy.init()
    node = Brain()
    ex = MultiThreadedExecutor(num_threads=6)
    ex.add_node(node)
    try:
        ex.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
