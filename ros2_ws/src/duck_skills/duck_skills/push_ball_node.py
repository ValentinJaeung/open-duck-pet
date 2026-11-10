"""push_ball(ball, goal): 공을 목표 물체 쪽으로 미는 Action 서버.

RL 로 학습한 상위 정책(sim/rl, 입력 9차원 → 속도 명령 3개)을 numpy 로 12.5Hz 로 실행하고, 그 명령을 /cmd_vel 로 보낸다.
상위 정책은 걷기 정책을 '학습된 그대로(고개 포함)' 쓰는 조건으로 학습됐으므로, 이 기술을 실행하는 동안 /walk_mode 를 native 로 바꾼다.
입력은 센서 기반 추정만 쓴다: 공/목표의 몸통 좌표(카메라+깊이), 안 보일 땐 지도 기억과 내 이동량(명령 속도)으로 갱신.
"""
import glob
import json
import math
import os
import sys
import time

import numpy as np
import rclpy
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Imu
from std_msgs.msg import String

from duck_interfaces.action import PushBall
from duck_interfaces.msg import DetectedObjectArray

ROOT = os.environ.get("DUCK_PET_ROOT", os.path.abspath(os.path.join(os.path.dirname(__file__), *([".."] * 4))))
sys.path.insert(0, ROOT)
from sim.rl.baseline_push import BaselinePush  # noqa: E402
from sim.rl.hl_policy import HLPolicy, make_obs  # noqa: E402

PERIOD = 0.08                  # s: 학습에서 상위 스텝 1번 = 걷기 정책 4번 = 0.08s
SUCCESS_DIST = 0.15            # m (추정치 기준)
FRESH = 0.4                    # s: 이 안에 본 탐지만 '지금 보임'
FWD_GAIN, YAW_GAIN = 0.7, 0.6  # 명령 대비 실제 이동/회전 비율(측정 약 0.7 / 0.3~0.8): 안 보일 때 내 이동량 추정용
TIMEOUT = 120.0
SEARCH_WZ = 0.8


def clip(v, lo, hi):
    return max(lo, min(hi, v))


class Tracker:
    """몸통 좌표에서의 2D 위치 추적: 보이면 측정값으로 갱신, 안 보이면 내 이동(속도 명령)만큼 반대로 이동시켜 예측"""
    def __init__(self):
        self.pos = None
        self.t_seen = -1e9
        self.vel = np.zeros(2)

    def predict(self, vx, vy, dth, dt):
        """dth: IMU 자이로로 잰 이 틱 동안의 실제 회전량(rad). 이동량은 명령 속도 x 실측 비율로 추정"""
        if self.pos is None:
            return
        p = self.pos - np.array([vx * FWD_GAIN, vy * FWD_GAIN]) * dt          # 내가 앞으로 가면 물체는 뒤로
        c, s = math.cos(-dth), math.sin(-dth)                                  # 내가 왼쪽으로 돌면 물체는 오른쪽으로
        self.pos = np.array([c * p[0] - s * p[1], s * p[0] + c * p[1]])

    def update(self, meas, t, dt):
        if self.pos is not None and dt > 0:
            self.vel = 0.6 * self.vel + 0.4 * (meas - self.pos) / dt if t - self.t_seen < 1.0 else np.zeros(2)
            self.vel = np.clip(self.vel, -1.0, 1.0)
        self.pos = meas.copy() if self.pos is None else 0.5 * self.pos + 0.5 * meas
        self.t_seen = t


class PushBallServer(Node):
    def __init__(self):
        super().__init__("duck_push_ball")
        cb = ReentrantCallbackGroup()
        # 기본 정책: 오프라인 평가(추정값 관측, 48판)에서 가장 좋았던 정답 관측 학습 잔차 정책(40%). 잡음/가림 학습 정책(20261109_234428)은 12% 로 더 나빴음
        preferred = os.path.join(ROOT, "models", "push", "20261109_215957", "params_final.npz")
        default = [preferred] if os.path.exists(preferred) else sorted(glob.glob(os.path.join(ROOT, "models", "push", "*", "params_final.npz")))
        # policy: "baseline"=사람이 설계한 제어기만, 파일 경로(.npz)=RL 정책. residual=True 면 제어기+정책 보정(잔차 RL)
        self.declare_parameter("policy", default[-1] if default else "baseline")
        self.declare_parameter("residual", True)
        self.policy = None
        self.pub_vel = self.create_publisher(Twist, "/cmd_vel", 10)
        self.pub_mode = self.create_publisher(String, "/walk_mode", 1)
        self.pub_dbg = self.create_publisher(String, "/push_debug", 5)      # 측정 전용: 추정값과 '지금 보이는지' (정답과 비교해 추정 오차를 잼)
        self.create_subscription(DetectedObjectArray, "/detections", self.on_det, 5, callback_group=cb)
        self.create_subscription(String, "/scene", self.on_scene, 1, callback_group=cb)
        self.create_subscription(Imu, "/imu", self.on_imu, 50, callback_group=cb)
        self.yaw, self._imu_t = 0.0, None        # IMU 자이로 적분으로 추정한 방향(rad)
        self.det, self.scene = {}, {}
        self._current = None
        self.srv = ActionServer(self, PushBall, "push_ball", self.execute, callback_group=cb,
                                goal_callback=lambda g: GoalResponse.ACCEPT, cancel_callback=lambda g: CancelResponse.ACCEPT)
        self.get_logger().info("push_ball 준비 (정책: %s)" % (self.get_parameter("policy").value or "없음"))

    def now_s(self):
        return self.get_clock().now().nanoseconds * 1e-9

    def on_det(self, msg):
        t = self.now_s()
        for o in msg.objects:
            if o.depth > 0 and (o.label not in self.det or o.score >= self.det[o.label][1].score or t - self.det[o.label][0] > 0.3):
                self.det[o.label] = (t, o)

    def on_imu(self, m):
        t = m.header.stamp.sec + m.header.stamp.nanosec * 1e-9
        if self._imu_t is not None and 0 < t - self._imu_t < 0.5:
            self.yaw += m.angular_velocity.z * (t - self._imu_t)
        self._imu_t = t

    def on_scene(self, msg):
        try:
            self.scene = {o["label"]: o for o in json.loads(msg.data)["objects"]}
        except Exception:
            pass

    def mode(self, native):
        self.pub_mode.publish(String(data="native" if native else "override"))

    def vel(self, vx=0.0, wz=0.0, vy=0.0):
        m = Twist()
        m.linear.x, m.linear.y, m.angular.z = float(vx), float(vy), float(wz)
        self.pub_vel.publish(m)

    def measure(self, label):
        """label 의 몸통 좌표 xy 측정값(신선한 탐지) 또는 None"""
        item = self.det.get(label)
        if item is not None and self.now_s() - item[0] < FRESH:
            return np.array([item[1].base_xyz.x, item[1].base_xyz.y])
        return None

    def from_scene(self, label):
        o = self.scene.get(label)
        if o is None:
            return None
        b = math.radians(o["bearing_deg"])
        return np.array([o["distance_m"] * math.cos(b), o["distance_m"] * math.sin(b)])

    def execute(self, gh):
        self._current = gh
        res, fb = PushBall.Result(), PushBall.Feedback()
        res.final_distance = -1.0
        ball_l, goal_l = gh.request.ball, gh.request.goal
        if self.policy is None and not self._load():
            gh.abort(); res.success, res.message = False, "상위 정책을 불러오지 못함(policy 파라미터 확인)"; return res
        known = ("red_block", "green_block", "blue_ball", "person")
        if ball_l not in known or goal_l not in known or ball_l == goal_l:
            gh.abort(); res.success, res.message = False, f"잘못된 대상 (가능: {', '.join(known)}, 공과 목표는 달라야 함)"; return res
        ball, goal = Tracker(), Tracker()
        self.mode(True)
        t0, last_t, reason, msg, search_t0 = self.now_s(), self.now_s(), None, "", None
        last_yaw = self.yaw
        last_cmd, last_act = (0.0, 0.0, 0.0), np.zeros(3)
        self.det.pop(ball_l, None); self.det.pop(goal_l, None)
        while True:
            now = self.now_s(); dt = max(now - last_t, 1e-3); last_t = now
            if gh.is_cancel_requested:
                reason = "canceled"; break
            if self._current is not gh:
                reason = "preempted"; break
            if now - t0 > TIMEOUT:
                reason = "시간 초과"; break
            dth, last_yaw = self.yaw - last_yaw, self.yaw
            vis_flags = []
            for trk, lab in ((ball, ball_l), (goal, goal_l)):
                trk.predict(last_cmd[0], last_cmd[1], dth, dt)
                m = self.measure(lab)
                vis_flags.append(m is not None)
                if m is not None:
                    trk.update(m, now, dt)
                elif trk.pos is None:                       # 한 번도 못 봤으면 지도 기억으로 시작
                    mm = self.from_scene(lab)
                    if mm is not None:
                        trk.pos, trk.t_seen = mm, now - 5.0
            if ball.pos is None or goal.pos is None:
                if search_t0 is None:
                    search_t0 = now
                if now - search_t0 > 30.0:
                    reason = f"{ball_l if ball.pos is None else goal_l} 을(를) 찾지 못함"; break
                self.vel(0.0, SEARCH_WZ); last_cmd = (0.0, 0.0, SEARCH_WZ)
                fb.state, fb.ball_goal_distance = "searching", -1.0
            else:
                search_t0 = None
                bg = float(np.linalg.norm(goal.pos - ball.pos))
                res.final_distance = bg
                if bg < SUCCESS_DIST:
                    # 추정값만으로 성공을 선언하면 허위 성공이 생김(측정: 공이 안 움직였는데 '성공' 보고). 공과 목표가 둘 다
                    # 지금 카메라에 보여 측정으로 확인될 때만 성공으로 인정한다. 못 보면 계속 시도하다 시간 초과로 끝남
                    if vis_flags[0] and vis_flags[1]:
                        msg = f"{ball_l} 을(를) {goal_l} 근처로 밀었음 (측정 확인, 거리 {bg:.2f} m)"; break
                    fb.state = "unverified"
                obs = make_obs(ball.pos, goal.pos, ball.vel, last_act)
                cmd, last_act = self.policy.command(obs)
                self.vel(cmd[0], cmd[2], cmd[1]); last_cmd = (float(cmd[0]), float(cmd[1]), float(cmd[2]))
                fb.state, fb.ball_goal_distance = "pushing", bg
            if ball.pos is not None and goal.pos is not None:
                self.pub_dbg.publish(String(data=json.dumps({"ball": ball.pos.tolist(), "goal": goal.pos.tolist(), "ball_vis": vis_flags[0], "goal_vis": vis_flags[1], "cmd": list(last_cmd)})))
            gh.publish_feedback(fb)
            time.sleep(PERIOD)
        for _ in range(3):
            self.vel(0.0, 0.0); time.sleep(0.02)
        if reason != "preempted":
            self.mode(False)
        if reason == "canceled":
            gh.canceled(); res.success, res.message = False, "취소됨"
        elif reason == "preempted":
            gh.abort(); res.success, res.message = False, "다른 기술이 시작되어 중단됨"
        elif reason is not None:
            gh.abort(); res.success, res.message = False, reason
        else:
            gh.succeed(); res.success, res.message = True, msg
        return res

    def _load(self):
        try:
            pth = self.get_parameter("policy").value
            self.policy = BaselinePush() if pth == "baseline" else HLPolicy(pth, residual=self.get_parameter("residual").value)
            self.get_logger().info(f"상위 정책 로드: {pth}")
            return True
        except Exception as e:
            self.get_logger().error(f"정책 로드 실패: {e}")
            return False


def main():
    rclpy.init()
    node = PushBallServer()
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
