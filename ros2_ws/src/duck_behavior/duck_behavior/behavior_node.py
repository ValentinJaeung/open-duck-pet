"""duck_behavior: 명령이 없을 때 혼자 노는 애완동물 행동 (Behavior Tree + 내부 상태).

우선순위 (위가 높음, 매 틱마다 위에서부터 다시 판단):
  1. 사용자 명령 중(LLM 두뇌가 일하는 중)  → 양보하고 가만히 있기
  2. 에너지 부족                           → 고개를 숙이고 쉬기(에너지가 0.7 이 될 때까지)
  3. 사람이 보임                           → 처음 만나면 반가워하고, 멀면 다가가고, 가까우면 곁에 머물기
  4. 사람이 방금 시야에서 사라짐           → 마지막으로 본 쪽으로 돌며 바로 쫓아가기
  4'. 외로움이 큼 (사람이 안 보임)         → 사람을 찾으러 돌아다니기(찾지 못하면 한동안 포기)
  5. 심심함이 큼                           → 돌아다니기 / 물체 살펴보러 가기 / 둘러보기 / 춤 중 하나
  6. 그 외                                 → 가끔 고개를 갸웃거리며 두리번
LLM 은 명령이 들어올 때만 두뇌 노드가 부르므로, 이 노드는 LLM 을 전혀 쓰지 않는다.

입력: /detections(사람 탐지), /scene(기억한 물체), /tf(오도메트리), /brain_busy, /user_command
출력: /pet_state(JSON, 1Hz), /pet_log(행동이 바뀔 때마다 한 줄)
"""
import json
import math
import random

import py_trees
import rclpy
from py_trees.common import Status
from rclpy.action import ActionClient
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from std_msgs.msg import Bool, String
from tf2_msgs.msg import TFMessage

from duck_interfaces.action import Dance, Happy, LookAt, Sad, Turn, WagHead, Walk, WalkTo
from duck_interfaces.msg import DetectedObjectArray
from duck_behavior.state import InternalState

DEG = math.pi / 180.0
PERSON_SEEN_FOR = 2.0       # s: 이 안에 본 사람은 '지금 보임'
PERSON_CLOSE = 1.45         # m: 이보다 멀면 다가가기 시작(다가가기는 walk_to 기본 정지 거리 0.9m 에서 끝남 → 떨림 방지)
USER_COOLDOWN = 4.0         # s: 사용자 명령이 끝난 뒤 자율 행동을 다시 시작하기까지 기다림
CHASE_WINDOW = 12.0         # s: 사람이 시야에서 사라진 지 이 시간 안이면 바로 쫓아감(마지막으로 본 쪽으로 돌며 찾기)
FIND_PERSON_GIVEUP = 45.0   # s: 사람을 못 찾으면 이 시간 동안 다시 찾지 않음
# 방 안 안전한 돌아다니기 영역(미리 알고 있는 방 배치: 가구를 피하기 위함)
WANDER_X, WANDER_Y = 1.5, 1.0
AVOID = [(-1.9, -0.7, 0.3, 1.5), (0.9, 1.9, -1.4, -0.3)]   # (xmin, xmax, ymin, ymax): 침대, 책상 주변


def clip(v, lo, hi):
    return max(lo, min(hi, v))


def norm_angle(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


class Cond(py_trees.behaviour.Behaviour):
    """조건 확인. fn() 이 True 면 SUCCESS"""
    def __init__(self, name, fn):
        super().__init__(name)
        self.fn = fn

    def update(self):
        return Status.SUCCESS if self.fn() else Status.FAILURE


class Yield(py_trees.behaviour.Behaviour):
    """아무것도 안 하고 RUNNING (사용자 명령 중 양보)"""
    def __init__(self, name, pet):
        super().__init__(name)
        self.pet = pet

    def initialise(self):
        self.pet.log("사용자 명령 중 → 자율 행동 양보")

    def update(self):
        return Status.RUNNING


class Activity(py_trees.behaviour.Behaviour):
    """여러 단계로 이루어진 행동 하나. plan() 이 단계 목록을 돌려준다.
    단계: ('act', 키, goal)=ROS Action 실행 / ('wait', fn)=fn() 이 True 가 될 때까지 대기 / ('call', fn)=즉시 실행
    위 우선순위가 이 행동을 끊으면(terminate INVALID) 실행 중이던 Action 을 취소한다."""
    def __init__(self, name, pet, plan, on_success=None, on_fail=None, ignore_step_fail=False):
        super().__init__(name)
        self.pet, self.plan, self.on_success, self.on_fail, self.ignore = pet, plan, on_success, on_fail, ignore_step_fail
        self.steps, self.idx, self.gh, self.result = [], 0, None, None

    def initialise(self):
        self.steps = self.plan() or []
        self.idx, self.gh, self.result, self.t_step = 0, None, None, self.pet.now_s()
        self.pet.set_activity(self.name)
        self._start_step()

    def _start_step(self):
        self.gh, self.result = None, None
        while self.idx < len(self.steps):
            kind = self.steps[self.idx][0]
            if kind == "call":
                self.steps[self.idx][1]()
                self.idx += 1
                continue
            if kind == "act":
                _, key, goal = self.steps[self.idx]
                cl = self.pet.acts[key]
                if not cl.server_is_ready():
                    self.result = "no_server"
                    return
                self.pet.moving = key in ("walk", "turn", "walk_to")
                cl.send_goal_async(goal).add_done_callback(self._on_goal)
            self.t_step = self.pet.now_s()
            return

    def _on_goal(self, fut):
        gh = fut.result()
        if not gh.accepted:
            self.result = "rejected"
            return
        self.gh = gh
        gh.get_result_async().add_done_callback(lambda f: setattr(self, "result", f.result().result))

    def update(self):
        if self.idx >= len(self.steps):
            return self._finish(True)
        kind = self.steps[self.idx][0]
        if kind == "wait":
            if self.steps[self.idx][1]():
                self.idx += 1
                self._start_step()
            return self._after_advance()
        if self.result is None:
            return Status.RUNNING
        ok = self.result not in ("no_server", "rejected") and getattr(self.result, "success", False)
        self.pet.moving = False
        if not ok and not self.ignore:
            return self._finish(False)
        self.idx += 1
        self._start_step()
        return self._after_advance()

    def _after_advance(self):
        return self._finish(True) if self.idx >= len(self.steps) else Status.RUNNING

    def _finish(self, ok):
        self.pet.moving = False
        cb = self.on_success if ok else self.on_fail
        if cb:
            cb()
        return Status.SUCCESS if ok else Status.FAILURE

    def terminate(self, new_status):
        self.pet.moving = False
        if new_status == Status.INVALID and self.gh is not None and self.result is None:
            self.gh.cancel_goal_async()


class Pet(Node):
    def __init__(self):
        super().__init__("duck_behavior")
        self.declare_parameter("rate_scale", 1.0)        # 내부 상태가 변하는 속도 배율(시험할 때 키움)
        self.declare_parameter("tick_hz", 4.0)
        self.declare_parameter("seed", 0)
        self.declare_parameter("enabled", True)
        self.rng = random.Random(self.get_parameter("seed").value)
        self.state = InternalState(rate_scale=self.get_parameter("rate_scale").value)
        cb = ReentrantCallbackGroup()
        self.acts = {k: ActionClient(self, t, n, callback_group=cb) for k, (t, n) in {
            "walk": (Walk, "walk"), "turn": (Turn, "turn"), "look": (LookAt, "look_at"), "walk_to": (WalkTo, "walk_to"),
            "happy": (Happy, "happy"), "sad": (Sad, "sad"), "wag": (WagHead, "wag_head"), "dance": (Dance, "dance")}.items()}
        self.create_subscription(DetectedObjectArray, "/detections", self.on_det, 5, callback_group=cb)
        self.create_subscription(String, "/scene", self.on_scene, 1, callback_group=cb)
        self.create_subscription(TFMessage, "/tf", self.on_tf, 10, callback_group=cb)
        self.create_subscription(Bool, "/brain_busy", self.on_busy, 10, callback_group=cb)
        self.create_subscription(String, "/user_command", self.on_user_command, 10, callback_group=cb)
        self.pub_state = self.create_publisher(String, "/pet_state", 1)
        self.pub_log = self.create_publisher(String, "/pet_log", 20)

        self.person_seen_t = None      # 사람을 마지막으로 본 시각(시뮬레이션 초)
        self.person_dist = None
        self.person_absent_since = 0.0
        self.greet_pending = True      # 사람이 (다시) 나타나면 반갑게 인사
        self.scene = []
        self.pose = None               # odom 에서의 (x, y, yaw)
        self.brain_busy = False
        self.hold_until = -1.0
        self.moving = False
        self.resting = False
        self.find_blocked_until = -1.0
        self.chase_blocked_until = -1.0
        self.activity = "시작"
        self.next_idle = 0.0
        self.last_t = None
        self.accompany_next = 0.0
        self.tree = py_trees.trees.BehaviourTree(self.build_tree())
        self.create_timer(1.0 / self.get_parameter("tick_hz").value, self.tick, callback_group=cb)
        self.create_timer(1.0, self.publish_state, callback_group=cb)
        self.log("자율 행동 시작 (심심함/에너지/외로움 상태로 스스로 행동합니다)")

    # ---------------- 입력 ----------------
    def now_s(self):
        return self.get_clock().now().nanoseconds * 1e-9

    def log(self, text):
        line = f"[{self.now_s():6.1f}s] {text}   (심심 {self.state.boredom:.2f} 에너지 {self.state.energy:.2f} 외로움 {self.state.social:.2f})"
        self.get_logger().info(line)
        self.pub_log.publish(String(data=line))

    def set_activity(self, name):
        self.activity = name
        self.log(f"행동: {name}")

    def on_det(self, msg):
        t = self.now_s()
        for o in msg.objects:
            if o.label == "person" and o.score >= 0.5:
                if self.person_seen_t is None or t - self.person_seen_t > PERSON_SEEN_FOR:
                    if t - self.person_absent_since > 10.0 or self.person_seen_t is None:
                        self.greet_pending = True               # 한동안 못 보다가 다시 만남
                self.person_seen_t = t
                if o.depth > 0:
                    self.person_dist = math.hypot(o.base_xyz.x, o.base_xyz.y)

    def on_scene(self, msg):
        try:
            self.scene = json.loads(msg.data)["objects"]
        except Exception:
            pass

    def on_tf(self, msg):
        for tr in msg.transforms:
            if tr.child_frame_id == "base_link":
                q = tr.transform.rotation
                self.pose = (tr.transform.translation.x, tr.transform.translation.y, 2 * math.atan2(q.z, q.w))

    def on_busy(self, msg):
        self.brain_busy = msg.data
        if not msg.data:
            self.hold_until = self.now_s() + USER_COOLDOWN

    def on_user_command(self, msg):
        self.hold_until = self.now_s() + USER_COOLDOWN + 6.0

    # ---------------- 판단 ----------------
    def user_active(self):
        return self.brain_busy or self.now_s() < self.hold_until

    def person_visible(self):
        return self.person_seen_t is not None and self.now_s() - self.person_seen_t < PERSON_SEEN_FOR

    def person_just_lost(self):
        t = self.now_s()
        return (self.person_seen_t is not None and not self.person_visible()
                and t - self.person_seen_t < CHASE_WINDOW and t > self.chase_blocked_until)

    def person_far(self):
        return self.person_dist is None or self.person_dist > PERSON_CLOSE

    # ---------------- 계획(행동 단계) ----------------
    def plan_greet_and_approach(self):
        steps = []
        if self.greet_pending:
            steps.append(("call", lambda: setattr(self, "greet_pending", False)))
            steps.append(("act", "happy", Happy.Goal(intensity=1.0)))
        if self.person_far():
            steps.append(("act", "walk_to", WalkTo.Goal(target="person", stop_distance=0.0)))
        return steps

    def plan_accompany(self):
        """사람 곁에 머무름: 사람이 보이는 동안 계속 RUNNING. 가끔 꼬리(고개) 흔들기"""
        def stay():
            if not self.person_visible() or self.person_far():
                return True
            if self.now_s() > self.accompany_next:
                self.accompany_next = self.now_s() + self.rng.uniform(15, 30)
                self.state.play(0.05)
            return False
        return [("wait", stay)]

    def plan_find_person(self):
        return [("act", "walk_to", WalkTo.Goal(target="person", stop_distance=0.0))]

    def chase_failed(self):
        self.chase_blocked_until = self.now_s() + 20.0
        self.log("놓친 사람을 못 찾음 → 잠시 포기")

    def find_failed(self):
        self.find_blocked_until = self.now_s() + FIND_PERSON_GIVEUP
        self.log("사람을 찾지 못함 → 한동안 포기")

    def plan_rest(self):
        self.resting = True
        return [("act", "look", LookAt.Goal(yaw=0.0, pitch=-0.6)),
                ("wait", lambda: not self.state.needs_rest(True) or self.user_active() or self.person_visible()),
                ("call", lambda: setattr(self, "resting", False)),
                ("act", "look", LookAt.Goal(yaw=0.0, pitch=0.0))]

    def stop_rest(self):
        self.resting = False

    def wander_point(self):
        for _ in range(40):
            x, y = self.rng.uniform(-WANDER_X, WANDER_X), self.rng.uniform(-WANDER_Y, WANDER_Y)
            if any(a <= x <= b and c <= y <= d for a, b, c, d in AVOID):
                continue
            if self.pose is None or 0.5 < math.hypot(x - self.pose[0], y - self.pose[1]) < 1.6:
                return x, y
        return None

    def plan_wander(self):
        p = self.wander_point()
        if p is None or self.pose is None:
            return [("act", "walk", Walk.Goal(distance=0.4))]
        dx, dy = p[0] - self.pose[0], p[1] - self.pose[1]
        turn = norm_angle(math.atan2(dy, dx) - self.pose[2])
        steps = []
        if abs(turn) > 0.2:
            steps.append(("act", "turn", Turn.Goal(angle=float(turn))))
        steps.append(("act", "walk", Walk.Goal(distance=float(min(math.hypot(dx, dy), 1.0)))))
        return steps

    def plan_investigate(self):
        objs = [o["label"] for o in self.scene if o["label"] != "person"]
        if not objs:
            return self.plan_look_around()
        return [("act", "walk_to", WalkTo.Goal(target=self.rng.choice(objs), stop_distance=0.0)),
                ("act", "wag", WagHead.Goal(times=2))]

    def plan_look_around(self):
        return [("act", "look", LookAt.Goal(yaw=0.9, pitch=0.1)), ("act", "look", LookAt.Goal(yaw=-0.9, pitch=0.1)),
                ("act", "look", LookAt.Goal(yaw=0.0, pitch=0.0))]

    def plan_dance(self):
        return [("act", "dance", Dance.Goal(duration=6.0))]

    def plan_sulk(self):
        return [("act", "sad", Sad.Goal())]

    def plan_bored(self):
        """심심함이 클 때 놀이 하나를 고른다 (가중치 무작위, 심심함이 아주 크면 춤/시무룩 추가)"""
        b = self.state.boredom
        choices = [("돌아다니기", self.plan_wander, 0.15, 4), ("물체 살펴보러 가기", self.plan_investigate, 0.25, 3),
                   ("두리번거리기", self.plan_look_around, 0.1, 2)]
        if b > 0.8:
            choices += [("춤추기", self.plan_dance, 0.45, 2), ("시무룩해지기", self.plan_sulk, 0.1, 1)]
        name, plan, effect, _ = self.rng.choices(choices, weights=[c[3] for c in choices])[0]
        self.log(f"심심해서 → {name}")
        self.state.play(0.0)
        steps = plan()
        steps.append(("call", lambda: self.state.play(effect)))
        return steps

    def plan_idle(self):
        yaw, pitch = self.rng.uniform(-0.5, 0.5), self.rng.uniform(-0.1, 0.25)
        self.next_idle = self.now_s() + self.rng.uniform(6, 12)
        return [("act", "look", LookAt.Goal(yaw=yaw, pitch=pitch)), ("wait", lambda: self.now_s() > self.next_idle - 3.0),
                ("act", "look", LookAt.Goal(yaw=0.0, pitch=0.0))]

    # ---------------- 트리 ----------------
    def build_tree(self):
        S = py_trees.composites.Sequence
        return py_trees.composites.Selector("애완동물", memory=False, children=[
            S("사용자 명령", memory=False, children=[Cond("명령중?", self.user_active), Yield("양보", self)]),
            S("쉬기", memory=False, children=[
                Cond("에너지 부족?", lambda: self.state.needs_rest(self.resting)),
                Activity("쉬기", self, self.plan_rest, on_success=self.stop_rest, on_fail=self.stop_rest)]),
            S("사람 만남", memory=False, children=[
                Cond("사람 보임?", self.person_visible),
                py_trees.composites.Selector("다가가기/머물기", memory=False, children=[
                    S("다가가기", memory=False, children=[
                        Cond("멀거나 인사 전?", lambda: self.person_far() or self.greet_pending),
                        Activity("사람에게 다가가기", self, self.plan_greet_and_approach)]),
                    Activity("사람 곁에 머물기", self, self.plan_accompany)])]),
            S("사람 쫓기", memory=False, children=[
                Cond("방금 놓침?", self.person_just_lost),
                Activity("놓친 사람 쫓아가기", self, self.plan_find_person, on_fail=self.chase_failed)]),
            S("사람 찾기", memory=False, children=[
                Cond("외로움?", lambda: self.state.is_lonely() and self.now_s() > self.find_blocked_until),
                Activity("사람 찾으러 가기", self, self.plan_find_person, on_fail=self.find_failed)]),
            S("놀이", memory=False, children=[
                Cond("심심함?", self.state.is_bored),
                Activity("놀이", self, self.plan_bored)]),
            Activity("두리번", self, self.plan_idle),
        ])

    # ---------------- 주기 실행 ----------------
    def tick(self):
        if not self.get_parameter("enabled").value:
            return
        t = self.now_s()
        if t <= 0.0:
            return                                          # 시뮬레이션 시계가 아직 안 옴
        if self.last_t is not None:
            self.state.update(t - self.last_t, moving=self.moving, resting=self.resting,
                              person_visible=self.person_visible(), user_active=self.user_active())
        self.last_t = t
        self.tree.tick()

    def publish_state(self):
        d = self.state.as_dict()
        d.update({"activity": self.activity, "person_visible": self.person_visible(),
                  "person_distance": None if self.person_dist is None else round(self.person_dist, 2),
                  "user_active": self.user_active()})
        self.pub_state.publish(String(data=json.dumps(d, ensure_ascii=False)))


def main():
    rclpy.init()
    node = Pet()
    ex = MultiThreadedExecutor(num_threads=4)
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
