"""walk_to(target): 카메라로 물체 방향을 계속 보며 접근하는 Action 서버.

시계 방향 제어: 물체가 보이면 방향 오차에 비례해 회전, 정면에 가까울 때만 전진(시각 서보잉).
안 보이면 /scene(기억한 물체) 방향 힌트로 돌며 찾고, 못 찾으면 실패를 돌려준다.
정답 위치는 쓰지 않는다: /detections(카메라+깊이)와 /scene(지도)만 사용.
바닥 물체는 가까워지면 화면 아래로 벗어나므로 접근하는 동안 고개를 숙인다.
"""
import json
import math
import time

import rclpy
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from geometry_msgs.msg import Twist
from sensor_msgs.msg import CameraInfo
from std_msgs.msg import Float64MultiArray, String

from duck_interfaces.action import WalkTo
from duck_interfaces.msg import DetectedObjectArray

RATE_HZ = 10.0
PERSON_STOP = 0.9           # 사람 앞에서 멈출 거리(m)
YOLO_LABELS = ("red_block", "green_block", "blue_ball", "person")
OPEN_MAX_WIDTH = 0.45        # 개방형 탐지 상자 폭이 화면 폭의 이 비율을 넘으면 큰 배경으로 보고 무시
OPEN_CONFIRM_SCORE = 0.25    # 개방형 탐지로 '도착'을 인정하려면 필요한 점수(가까이서)
DEFAULT_STOP = 0.35          # m, 몸통 중심에서 물체 중심까지
HEAD_PITCH_DOWN = -0.35      # rad, 접근 중 고개 숙임 (아래 -)
MAX_VX, MAX_WZ = 0.15, 1.0
SEARCH_WZ = 0.8              # 찾을 때 회전 명령
SEARCH_TIMEOUT = 30.0        # s (시뮬레이션 시간). 실제 회전 속도가 약 0.3~0.5 rad/s 라 한 바퀴 이상
TOTAL_TIMEOUT = 90.0
FRESH = 0.6                  # s, 이 안에 본 탐지만 '지금 보임'으로 인정


def clip(v, lo, hi):
    return max(lo, min(hi, v))


class WalkToServer(Node):
    def __init__(self):
        super().__init__("duck_walk_to")
        cb = ReentrantCallbackGroup()
        self.pub_vel = self.create_publisher(Twist, "/cmd_vel", 10)
        self.pub_head = self.create_publisher(Float64MultiArray, "/head_cmd", 10)
        self.create_subscription(DetectedObjectArray, "/detections", self.on_det, 5, callback_group=cb)
        self.create_subscription(String, "/scene", self.on_scene, 1, callback_group=cb)
        self.create_subscription(CameraInfo, "/camera/camera_info", self.on_info, 1, callback_group=cb)
        self.det = {}                 # label -> (수신 시각(s), DetectedObject)
        self.scene = {}               # label -> {distance_m, bearing_deg}
        self.last_seen = {}           # label -> (본 시각, 방향 rad)
        self.cam = None               # (fx, cx, width, height)
        self.last_score = 1.0
        self._current = None
        self.srv = ActionServer(self, WalkTo, "walk_to", self.execute, callback_group=cb,
                                goal_callback=lambda g: GoalResponse.ACCEPT,
                                cancel_callback=lambda g: CancelResponse.ACCEPT)
        self.get_logger().info("walk_to 준비")

    def now_s(self):
        return self.get_clock().now().nanoseconds * 1e-9

    def on_info(self, m):
        self.cam = (m.k[0], m.k[2], m.width, m.height)

    def on_det(self, msg):
        t = self.now_s()
        best = {}
        for o in msg.objects:                       # 같은 종류가 여러 개면 점수가 가장 높은 것
            if o.label not in best or o.score > best[o.label].score:
                best[o.label] = o
        for label, o in best.items():
            self.det[label] = (t, o)
            # 마지막으로 본 방향(왼쪽 +)을 기억: 시야에서 놓쳤을 때 그쪽으로 먼저 돌며 찾기 위함
            if o.depth > 0:
                self.last_seen[label] = (t, math.atan2(o.base_xyz.y, o.base_xyz.x))
            elif self.cam is not None:
                self.last_seen[label] = (t, math.atan2(self.cam[1] - o.u, self.cam[0]))

    def on_scene(self, msg):
        try:
            self.scene = {o["label"]: o for o in json.loads(msg.data)["objects"]}
        except Exception:
            pass

    def vel(self, vx=0.0, wz=0.0):
        m = Twist()
        m.linear.x, m.angular.z = float(vx), float(wz)
        self.pub_vel.publish(m)

    def head(self, pitch):
        m = Float64MultiArray()
        m.data = [0.0, float(pitch), 0.0, 0.0]
        self.pub_head.publish(m)

    def target_obs(self, label):
        """지금 보이는 탐지(신선한 것)를 (bearing[rad], distance[m] 또는 None, 가까움 여부) 로. 없으면 None"""
        item = self.det.get(label)
        if item is None or self.now_s() - item[0] > FRESH:
            return None
        o = item[1]
        openvocab = label not in YOLO_LABELS
        if openvocab and self.cam is not None and o.width > OPEN_MAX_WIDTH * self.cam[2]:
            return None                        # 화면을 가득 채우는 큰 상자는 배경(침대, 벽 등)을 잘못 잡은 것으로 보고 무시
        near_bottom = self.cam is not None and (o.v + o.height / 2) >= self.cam[3] - 3
        self.last_score = o.score
        if o.depth > 0:
            b = math.atan2(o.base_xyz.y, o.base_xyz.x)
            return b, math.hypot(o.base_xyz.x, o.base_xyz.y), near_bottom
        if self.cam is None:
            return None
        fx, cx = self.cam[0], self.cam[1]
        return math.atan2(cx - o.u, fx), None, near_bottom          # 깊이 없음: 픽셀로 방향만

    def execute(self, gh):
        self._current = gh
        label = gh.request.target
        is_person = label == "person"
        stop = gh.request.stop_distance if gh.request.stop_distance > 0 else (PERSON_STOP if is_person else DEFAULT_STOP)
        res, fb = WalkTo.Result(), WalkTo.Feedback()
        res.final_distance = -1.0
        if not label.strip():
            gh.abort(); res.success, res.message = False, "찾을 물체 이름이 비어 있음"; return res
        # 라벨은 YOLO 가 아는 4종류 외에도 개방형 탐지(OWLv2)가 내는 임의 이름(예: yellow_cup)을 허용한다
        t0 = self.now_s()
        self.det.pop(label, None)
        last_seen, last_dist, last_bearing = None, None, None
        search_t0, sign, unconfirmed, confirmed = None, 1.0, 0, (label in YOLO_LABELS)     # 개방형 대상은 가까이서 높은 점수로 확인돼야 confirmed
        reason, msg = None, ""
        period = 1.0 / RATE_HZ
        while True:
            now = self.now_s()
            if gh.is_cancel_requested:
                reason = "canceled"; break
            if self._current is not gh:
                reason = "preempted"; break
            if now - t0 > TOTAL_TIMEOUT:
                reason = "시간 초과"; break
            self.head(0.0 if is_person else HEAD_PITCH_DOWN)     # 사람은 키가 커서 고개를 숙이지 않음
            obs = self.target_obs(label)
            if obs is not None:
                search_t0 = None
                bearing, dist, near_bottom = obs
                last_seen, last_bearing = now, bearing
                if dist is not None:
                    last_dist = dist
                if label not in YOLO_LABELS and self.last_score >= OPEN_CONFIRM_SCORE and dist is not None and dist < 1.2:
                    confirmed = True
                arrived = (dist is not None and dist <= stop) or (not is_person and near_bottom and (dist is None or dist < 0.8))
                if arrived and label not in YOLO_LABELS and self.last_score < OPEN_CONFIRM_SCORE:
                    arrived = False                    # 개방형 탐지는 멀리서 오탐이 있을 수 있음: 가까이서 점수가 충분히 높아야 '도착' 인정
                    unconfirmed += 1
                    if unconfirmed > 25:               # 약 2.5초 가까이서도 확인이 안 되면 오탐으로 보고 포기
                        reason = f"{label} 후보를 가까이서 확인하지 못함(점수 {self.last_score:.2f} < {OPEN_CONFIRM_SCORE}). 다른 물체일 수 있음"; break
                if arrived:
                    msg = f"{label} 앞에 도착 (거리 약 {last_dist if last_dist is not None else stop:.2f} m)"
                    break
                wz = clip(1.5 * bearing, -MAX_WZ, MAX_WZ)
                if abs(bearing) > 0.08:
                    wz = math.copysign(max(abs(wz), 0.4), bearing)      # 정책이 작은 회전 명령을 무시하므로 최소값 부여
                vx = MAX_VX * max(0.0, math.cos(bearing)) if abs(bearing) < 0.45 else 0.0
                self.vel(vx, wz)
                fb.state, fb.distance, fb.bearing_deg = "approaching", float(dist if dist is not None else -1.0), float(math.degrees(bearing))
            else:
                if not is_person and confirmed and last_seen is not None and now - last_seen < 1.5 and last_dist is not None and last_dist < 0.8:
                    msg = f"{label} 바로 앞에서 시야에서 벗어남 → 도착으로 간주 (마지막 거리 {last_dist:.2f} m)"
                    break
                # 안 보임: 기억한 방향 쪽으로, 없으면 마지막으로 본 쪽/왼쪽으로 돌며 찾기
                if search_t0 is None:
                    search_t0 = now
                    hint = self.scene.get(label)
                    mem = self.last_seen.get(label)
                    if last_bearing is not None:                      # 이 요청 중에 본 방향이 가장 정확
                        ang = last_bearing
                    elif mem is not None and now - mem[0] < 20.0:     # 최근 20초 안에 본 방향
                        ang = mem[1]
                    elif hint is not None:                            # 지도에 기억한 방향
                        ang = math.radians(hint["bearing_deg"])
                    else:
                        ang = 1e-3                                    # 단서 없음: 왼쪽부터
                    sign = math.copysign(1.0, ang or 1e-3)
                    self.get_logger().info(f"{label} 안 보임 → 찾는 중 (방향 {'왼쪽' if sign > 0 else '오른쪽'})")
                if now - search_t0 > SEARCH_TIMEOUT:
                    self.get_logger().warn(f"탐색 시간 초과: now={now:.1f} search_t0={search_t0:.1f} t0={t0:.1f}")
                    reason = f"{label} 을(를) 찾지 못함 (제자리에서 한 바퀴 이상 돌았음)"; break
                self.vel(0.0, sign * SEARCH_WZ)
                fb.state, fb.distance, fb.bearing_deg = "searching", -1.0, 0.0
            gh.publish_feedback(fb)
            time.sleep(period)
        for _ in range(3):
            self.vel(0.0, 0.0); time.sleep(0.02)
        if reason != "preempted":
            self.head(0.0)
        res.final_distance = float(last_dist if last_dist is not None else -1.0)
        if reason == "canceled":
            gh.canceled(); res.success, res.message = False, "취소됨"
        elif reason == "preempted":
            gh.abort(); res.success, res.message = False, "다른 기술이 시작되어 중단됨"
        elif reason is not None:
            gh.abort(); res.success, res.message = False, reason
        else:
            gh.succeed(); res.success, res.message = True, msg
        return res


def main():
    rclpy.init()
    node = WalkToServer()
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
