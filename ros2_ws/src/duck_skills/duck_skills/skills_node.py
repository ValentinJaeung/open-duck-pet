"""오리 기술 Action 서버.

기술은 새로 학습한 것이 아니라, 이미 있는 걷기 정책에 /cmd_vel 과 /head_cmd 를 시간에 맞춰 보내는 "조종사"다.
  turn(angle)      제자리 회전.  angle 라디안, +왼쪽
  walk(distance)   직진.         distance 미터 (뒤로는 못 감)
  look_at(yaw, pitch) 고개만 움직여 보기. yaw +왼쪽, pitch +위
  happy / sad / wag_head / dance   고개 + 몸 흔들기 키프레임 동작
한 번에 하나만 실행한다: 새 목표가 오면 실행 중이던 기술은 중단(abort)된다.

주의: 거리/각도는 정답 위치를 쓰지 않고 "시간"으로 추정한다(calibration.py 의 실측 관계식).
그래서 오차가 있다. 오도메트리가 생기는 3단계 이후 개선한다.
"""
import math
import threading
import time

import rclpy
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from geometry_msgs.msg import Twist
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray

from duck_interfaces.action import Dance, Happy, LookAt, Sad, Turn, WagHead, Walk
from duck_skills import calibration as C

RATE_HZ = 20.0
HEAD_JOINTS = ["neck_pitch", "head_pitch", "head_yaw", "head_roll"]


def clip(v, lo, hi):
    return max(lo, min(hi, v))


def smoothstep(x):
    x = clip(x, 0.0, 1.0)
    return x * x * (3 - 2 * x)


def envelope(t, total, ramp):
    """시작/끝에서 0 -> 1 -> 0 으로 부드럽게. 동작이 끝나면 고개가 원래 자리로 돌아오게 함"""
    if total <= 0:
        return 0.0
    r = min(ramp, total / 2)
    return smoothstep(min(t / r, (total - t) / r))


# ---- 감정/춤 키프레임: t(초) -> (고개 오프셋 [neck, pitch, yaw, roll], 속도 명령 [vx, vy, wz]) ----
def pose_happy(t, total, k):
    e = envelope(t, total, 0.5)
    head = [0.0, 0.45 * k * e, 0.25 * k * e * math.sin(2 * math.pi * t / 1.4), 0.3 * k * e * math.sin(2 * math.pi * t / 0.7)]
    return head, [0.0, 0.0, 1.0 * k * e * math.sin(2 * math.pi * t / 1.0)]


def pose_sad(t, total, k):
    e = envelope(t, total, 1.2)
    head = [0.0, -0.7 * k * e, 0.35 * k * e * math.sin(2 * math.pi * t / 3.0), 0.0]
    return head, [0.0, 0.0, 0.0]


def pose_wag(t, total, k):
    e = envelope(t, total, 0.15)
    return [0.0, 0.0, 0.6 * e * math.sin(2 * math.pi * t / 0.9), 0.0], [0.0, 0.0, 0.0]


def pose_dance(t, total, k):
    e = envelope(t, total, 0.6)
    head = [0.0, 0.35 * e * math.sin(2 * math.pi * t / 1.0 + math.pi / 2), 0.5 * e * math.sin(2 * math.pi * t / 2.0),
            0.3 * e * math.sin(2 * math.pi * t / 1.0)]
    return head, [0.0, 0.0, 1.0 * e * math.sin(2 * math.pi * t / 2.0)]


class DuckSkills(Node):
    def __init__(self):
        super().__init__("duck_skills")
        self.pub_vel = self.create_publisher(Twist, "/cmd_vel", 10)
        self.pub_head = self.create_publisher(Float64MultiArray, "/head_cmd", 10)
        self.head_cmd = [0.0, 0.0, 0.0, 0.0]     # 마지막으로 보낸 고개 명령(= 현재 고개 목표)
        self._head_synced = False
        self.create_subscription(JointState, "/joint_states", self.on_joints, 10)
        self._current = None
        self._lock = threading.Lock()
        g = ReentrantCallbackGroup()
        for name, typ, fn in [("turn", Turn, self.exec_turn), ("walk", Walk, self.exec_walk),
                              ("look_at", LookAt, self.exec_look_at), ("happy", Happy, self.exec_happy),
                              ("sad", Sad, self.exec_sad), ("wag_head", WagHead, self.exec_wag),
                              ("dance", Dance, self.exec_dance)]:
            ActionServer(self, typ, name, execute_callback=fn, callback_group=g,
                         goal_callback=lambda goal: GoalResponse.ACCEPT,
                         cancel_callback=lambda gh: CancelResponse.ACCEPT,
                         handle_accepted_callback=self.on_accepted)
        self.get_logger().info("기술 준비 완료: turn walk look_at happy sad wag_head dance")

    # ---- 공통 ----
    def on_joints(self, msg):
        if self._head_synced:
            return
        pos = dict(zip(msg.name, msg.position))
        if all(n in pos for n in HEAD_JOINTS):
            self.head_cmd = [float(pos[n]) for n in HEAD_JOINTS]
            self._head_synced = True

    def on_accepted(self, gh):
        with self._lock:
            self._current = gh       # 이전 기술은 실행 루프에서 이걸 보고 스스로 중단함
        gh.execute()

    def send_vel(self, vx=0.0, vy=0.0, wz=0.0):
        m = Twist()
        m.linear.x, m.linear.y, m.angular.z = float(vx), float(vy), float(wz)
        self.pub_vel.publish(m)

    def send_head(self, h):
        h = [float(h[0]), clip(h[1], *C.HEAD_PITCH_RANGE), clip(h[2], *C.HEAD_YAW_RANGE), clip(h[3], -0.5, 0.5)]
        self.head_cmd = h
        m = Float64MultiArray()
        m.data = h
        self.pub_head.publish(m)

    def stop_body(self, gh):
        if self._current is gh or self._current is None:   # 다른 기술이 이어받았다면 건드리지 않음
            for _ in range(3):
                self.send_vel()

    def now_s(self):
        """노드 시계(초). use_sim_time 이면 시뮬레이션 시간 → 시뮬레이션이 느려도 보정값(시간 기준)이 맞는다."""
        return self.get_clock().now().nanoseconds * 1e-9

    def run(self, gh, total, step_fn, result):
        """total 초 동안 20Hz 로 step_fn(t) 를 호출. 취소/대체되면 중단. result: 결과 객체"""
        t0 = self.now_s()
        period = 1.0 / RATE_HZ
        while True:
            t = self.now_s() - t0
            if gh.is_cancel_requested:
                self.stop_body(gh)
                gh.canceled()
                result.success, result.message = False, "취소됨"
                return result
            if self._current is not gh:
                gh.abort()
                result.success, result.message = False, "다른 기술이 시작되어 중단됨"
                return result
            if t >= total:
                break
            step_fn(t)
            self._publish_feedback(gh, min(1.0, t / total) if total > 0 else 1.0)
            time.sleep(period)
        self.stop_body(gh)
        elapsed = self.now_s() - t0
        gh.succeed()
        result.success, result.message = True, "완료"
        if hasattr(result, "duration"):
            result.duration = float(elapsed)
        return result

    _FEEDBACK = {}

    def _publish_feedback(self, gh, progress):
        cls = self._FEEDBACK.get(type(gh.request))
        if cls is not None:
            fb = cls()
            fb.progress = float(progress)
            gh.publish_feedback(fb)

    def fail(self, gh, result, msg):
        gh.abort()
        result.success, result.message = False, msg
        return result

    # ---- 기술 ----
    def exec_turn(self, gh):
        r = Turn.Result()
        a = float(gh.request.angle)
        if not math.isfinite(a) or abs(a) > C.TURN_MAX:
            return self.fail(gh, r, f"각도는 +-{C.TURN_MAX:.1f} rad 이내여야 함")
        if abs(a) < 1e-3:
            gh.succeed(); r.success, r.message = True, "이미 그 방향"; return r
        left = a > 0
        rate, off = (C.TURN_LEFT_RATE, C.TURN_LEFT_OFFSET) if left else (C.TURN_RIGHT_RATE, C.TURN_RIGHT_OFFSET)
        wz = C.TURN_WZ if left else -C.TURN_WZ
        total = abs(a) / rate + off
        return self.run(gh, total, lambda t: self.send_vel(wz=wz), r)

    def exec_walk(self, gh):
        r = Walk.Result()
        d = float(gh.request.distance)
        if not math.isfinite(d):
            return self.fail(gh, r, "거리가 숫자가 아님")
        if d < 0:
            return self.fail(gh, r, "뒤로 걷기는 지원하지 않음(이 정책은 뒤로 못 감). turn(3.14) 후 walk 하세요")
        if d < C.WALK_MIN or d > C.WALK_MAX:
            return self.fail(gh, r, f"거리는 {C.WALK_MIN}~{C.WALK_MAX} m 이내여야 함")
        total = d / C.WALK_SPEED + C.WALK_OFFSET
        return self.run(gh, total, lambda t: self.send_vel(vx=C.WALK_VX, wz=C.WALK_WZ_BIAS), r)

    def exec_look_at(self, gh):
        r = LookAt.Result()
        yaw, pitch = float(gh.request.yaw), float(gh.request.pitch)
        if not (math.isfinite(yaw) and math.isfinite(pitch)):
            return self.fail(gh, r, "각도가 숫자가 아님")
        cy, cp = clip(yaw, *C.HEAD_YAW_RANGE), clip(pitch, *C.HEAD_PITCH_RANGE)
        note = "" if (cy, cp) == (yaw, pitch) else f" (범위를 넘어 yaw={cy:.2f}, pitch={cp:.2f} 로 제한됨)"
        start = list(self.head_cmd)
        total = max(0.3, max(abs(cy - start[2]), abs(cp - start[1])) / C.GAZE_SPEED) + 0.3   # +0.3초: 도착 후 안정

        def step(t):
            s = smoothstep(t / (total - 0.3))
            self.send_head([start[0], start[1] + (cp - start[1]) * s, start[2] + (cy - start[2]) * s, start[3]])
        out = self.run(gh, total, step, r)
        if out.success:
            out.message = "완료" + note
        return out

    def _emote(self, gh, result, total, pose_fn, k):
        start = list(self.head_cmd)

        def step(t):
            off, vel = pose_fn(t, total, k)
            self.send_head([start[i] + off[i] for i in range(4)])
            self.send_vel(*vel)
        return self.run(gh, total, step, result)

    def exec_happy(self, gh):
        k = gh.request.intensity if gh.request.intensity > 0 else 1.0
        return self._emote(gh, Happy.Result(), 3.5, pose_happy, clip(k, 0.1, 1.0))

    def exec_sad(self, gh):
        k = gh.request.intensity if gh.request.intensity > 0 else 1.0
        return self._emote(gh, Sad.Result(), 4.5, pose_sad, clip(k, 0.1, 1.0))

    def exec_wag(self, gh):
        n = gh.request.times if gh.request.times > 0 else 3
        n = min(n, 20)
        return self._emote(gh, WagHead.Result(), n * 0.9, pose_wag, 1.0)

    def exec_dance(self, gh):
        d = gh.request.duration if gh.request.duration > 0 else 6.0
        return self._emote(gh, Dance.Result(), clip(d, 1.0, 60.0), pose_dance, 1.0)


DuckSkills._FEEDBACK = {Turn.Goal: Turn.Feedback, Walk.Goal: Walk.Feedback, LookAt.Goal: LookAt.Feedback,
                        Happy.Goal: Happy.Feedback, Sad.Goal: Sad.Feedback, WagHead.Goal: WagHead.Feedback,
                        Dance.Goal: Dance.Feedback}


def main():
    rclpy.init()
    node = DuckSkills()
    ex = MultiThreadedExecutor(num_threads=8)
    ex.add_node(node)
    try:
        ex.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.send_vel()
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
