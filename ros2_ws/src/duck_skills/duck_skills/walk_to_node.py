"""walk_to(target): 카메라로 물체 방향을 계속 보며 접근하는 Action 서버.

시계 방향 제어: 물체가 보이면 방향 오차에 비례해 회전, 정면에 가까울 때만 전진(시각 서보잉).
안 보이면 /scene(기억한 물체) 방향 힌트 쪽으로 제자리에서 한 바퀴 돌며 찾는다.
그래도 없으면 **탐험**: 돌면서 모은 /depth_scan(깊이로 만든 장애물 거리)으로 갈 수 있는 방향을 고르고,
이미 둘러본 곳에서 먼 '다음 관측 지점'으로 걸어가 다시 한 바퀴 돈다(최대 max_viewpoints 곳).
정답 위치는 쓰지 않는다: /detections(카메라+깊이), /scene(지도), /depth_scan, TF odom→base_link(오도메트리)만 사용.
바닥 물체는 가까워지면 화면 아래로 벗어나므로 접근하는 동안 고개를 숙인다.
"""
import json
import math
import time

import numpy as np
import rclpy
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from geometry_msgs.msg import Twist
from sensor_msgs.msg import CameraInfo, LaserScan
from std_msgs.msg import Float64MultiArray, String
from tf2_msgs.msg import TFMessage

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
TOTAL_TIMEOUT = 90.0         # 탐험을 끈 경우
FRESH = 0.6                  # s, 이 안에 본 탐지만 '지금 보임'으로 인정
# --- 탐험 ---
SPIN_ANGLE = 2 * math.pi + 0.3   # rad, 자이로(오도메트리 방향)로 잰 회전량이 이만큼이면 '한 바퀴 다 봄'
SPIN_TIMEOUT = 40.0          # s, 회전이 느려도 이 시간이면 한 바퀴로 침
SHORT_SPIN = 1.4             # rad, 잠깐 본 후보를 놓쳤을 때 그쪽으로 더 돌아보는 양
N_BINS = 36                  # 방향 칸(10°) 마다 장애물 거리 기록
MIN_CLEAR = 0.75             # m, 이보다 트인 방향만 후보
SAFE = 0.4                   # m, 장애물 앞 이만큼은 남기고 멈춤
LEG_MIN, LEG_MAX = 0.4, 1.0  # m, 한 번에 이동하는 거리
STOP_CLEAR = 0.32            # m, 이동 중 정면(±20°) 장애물이 이보다 가까우면 멈춤
EXPLORE_TIMEOUT = 220.0      # s, 탐험을 켠 경우 전체 제한


def clip(v, lo, hi):
    return max(lo, min(hi, v))


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


class Explorer:
    """목표 하나를 찾는 동안의 탐험 상태: spin(한 바퀴) → turn_to(다음 방향) → go(이동) → spin ..."""

    def __init__(self, sign):
        self.phase, self.sign = "spin", sign
        self.visited = []              # 한 바퀴 돌았던 지점(odom)
        self.blocked = []              # (지점, 방향): 가려다 바로 막힌 방향. 같은 자리에서 다시 고르지 않음
        self.spins = 0
        self.last_stamp = None
        self.reset_spin()

    def reset_spin(self, short=False):
        """short=True: 탐험 중 후보를 잠깐 보고 놓친 경우. 놓친 쪽으로 조금만 돌고, 이 자리에서 모은 스캔은 그대로 다시 씀"""
        self.phase, self.spun, self.prev_yaw, self.t_phase = "spin", 0.0, None, None
        self.short = short
        self.spin_limit = SHORT_SPIN if short else SPIN_ANGLE
        if not short:
            self.bins = [[] for _ in range(N_BINS)]

    def record(self, pose, scan):
        """스캔을 odom 방향 칸에 기록 (각 칸 = 그 방향으로 얼마나 트였나)"""
        angles, ranges = scan
        for a, r in zip(angles, ranges):
            if np.isfinite(r):
                k = int(round(wrap(pose[2] + a) / (2 * math.pi / N_BINS))) % N_BINS
                self.bins[k].append(float(r))

    def choose(self, pose, hint_heading):
        """다음 관측 지점: 트인 방향 중 '이미 본 곳에서 먼' 점. 지도 힌트가 있으면 그쪽을 더 선호.
        없으면 기준을 낮춰(덜 트여도, 막혔던 방향도) 한 번 더. 그래도 없으면 None"""
        return self._choose(pose, hint_heading, MIN_CLEAR, True) or self._choose(pose, hint_heading, 0.55, False)

    def _choose(self, pose, hint_heading, min_clear, avoid_blocked):
        best, best_s = None, -1e9
        for k, rs in enumerate(self.bins):
            if len(rs) < 3:
                continue
            c = float(np.percentile(rs, 25))          # 칸 안 관측의 하위 25%: 잡음에 덜 민감하면서 보수적
            if c < min_clear:
                continue
            th = k * 2 * math.pi / N_BINS
            if avoid_blocked and any(math.hypot(pose[0] - bp[0], pose[1] - bp[1]) < 0.3 and abs(wrap(th - bt)) < 0.45 for bp, bt in self.blocked):
                continue
            leg = clip(c - SAFE, min(LEG_MIN, c - 0.3), LEG_MAX)
            p = (pose[0] + leg * math.cos(th), pose[1] + leg * math.sin(th))
            novelty = min(math.hypot(p[0] - v[0], p[1] - v[1]) for v in self.visited)
            s = novelty + 0.3 * leg
            if hint_heading is not None:
                s += 0.8 * math.cos(wrap(th - hint_heading))
            if s > best_s:
                best, best_s = (th, leg, p), s
        return best


class WalkToServer(Node):
    def __init__(self):
        super().__init__("duck_walk_to")
        cb = ReentrantCallbackGroup()
        self.pub_vel = self.create_publisher(Twist, "/cmd_vel", 10)
        self.pub_head = self.create_publisher(Float64MultiArray, "/head_cmd", 10)
        self.create_subscription(DetectedObjectArray, "/detections", self.on_det, 5, callback_group=cb)
        self.create_subscription(String, "/scene", self.on_scene, 1, callback_group=cb)
        self.create_subscription(CameraInfo, "/camera/camera_info", self.on_info, 1, callback_group=cb)
        self.create_subscription(LaserScan, "/depth_scan", self.on_scan, 5, callback_group=cb)
        self.create_subscription(TFMessage, "/tf", self.on_tf, 20, callback_group=cb)
        self.declare_parameter("explore", True)          # False: 예전처럼 제자리에서만 돌며 찾기
        self.declare_parameter("max_viewpoints", 4)      # 처음 자리 말고 더 둘러볼 지점 수
        self.pose = None              # odom 에서 (x, y, yaw) — 인식 노드의 오도메트리
        self.pose_hist = []           # [(시각, pose)] 최근 3초
        self.scan = None              # (수신 시각, 각도 배열, 거리 배열)
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

    def on_tf(self, msg):
        for tr in msg.transforms:
            if tr.child_frame_id == "base_link" and tr.header.frame_id == "odom":
                q = tr.transform.rotation
                self.pose = (tr.transform.translation.x, tr.transform.translation.y, 2 * math.atan2(q.z, q.w))
                t = tr.header.stamp.sec + tr.header.stamp.nanosec * 1e-9
                self.pose_hist.append((t, self.pose))
                if len(self.pose_hist) > 60:            # 20Hz x 3초
                    self.pose_hist.pop(0)

    def pose_at(self, t):
        """시각 t 의 자세(가장 가까운 기록). 회전 중에 스캔을 방향 칸에 넣을 때 카메라→인식 지연만큼 방향이 밀리지 않게 함"""
        if not self.pose_hist:
            return self.pose
        return min(self.pose_hist, key=lambda e: abs(e[0] - t))[1]

    def on_scan(self, m):
        a = m.angle_min + m.angle_increment * np.arange(len(m.ranges))
        stamp = m.header.stamp.sec + m.header.stamp.nanosec * 1e-9
        self.scan = (self.now_s(), a, np.asarray(m.ranges, dtype=float), stamp)

    def fresh_scan(self):
        return None if self.scan is None or self.now_s() - self.scan[0] > 0.4 else self.scan[1:3]

    def front_clear(self, half=math.radians(20)):
        s = self.fresh_scan()
        if s is None:
            return None
        sel = np.abs(s[0]) <= half
        r = s[1][sel]
        r = r[np.isfinite(r)]
        # 정면 칸이 전부 nan = 깊이 점이 하나도 안 잡힘(벽에 코를 박을 만큼 가까울 때 생김) → 막힌 것으로 본다
        return float(r.min()) if len(r) else 0.0

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

    def explore_step(self, ex, label, now, max_vp, fb):
        """탐험 한 주기(0.1초). 실패로 끝내야 하면 이유 문자열, 계속하면 None"""
        pose, log = self.pose, self.get_logger()
        fb.distance, fb.bearing_deg = -1.0, 0.0
        if ex.phase == "spin":
            if ex.t_phase is None:
                ex.t_phase, ex.prev_yaw = now, pose[2]
                # 같은 자리에서 다시 도는 경우(후보를 잠깐 보고 놓침 등)는 새 관측 지점으로 세지 않음
                if not ex.visited or min(math.hypot(pose[0] - v[0], pose[1] - v[1]) for v in ex.visited) > 0.3:
                    ex.visited.append(pose[:2])
            ex.spun += abs(wrap(pose[2] - ex.prev_yaw))
            ex.prev_yaw = pose[2]
            scan = self.fresh_scan()
            if scan is not None and self.scan[3] != ex.last_stamp:      # 같은 스캔을 여러 번 넣지 않음
                ex.last_stamp = self.scan[3]
                ex.record(self.pose_at(self.scan[3]), scan)
            if ex.spun < ex.spin_limit and now - ex.t_phase < SPIN_TIMEOUT:
                self.vel(0.0, ex.sign * SEARCH_WZ)
                fb.state = "searching"
                return None
            done = len(ex.visited)
            if not ex.short:
                ex.spins += 1
            if done > max_vp or ex.spins > max_vp + 3:
                return f"{label} 을(를) 찾지 못함 ({done}곳에서 한 바퀴씩 둘러봄)"
            hint = self.scene.get(label)
            hint_heading = pose[2] + math.radians(hint["bearing_deg"]) if hint is not None else None
            ex.target = ex.choose(pose, hint_heading)
            if ex.target is None:
                return f"{label} 을(를) 찾지 못함 (이 자리에서 한 바퀴 둘러봤고, 더 갈 수 있는 트인 곳이 없음)"
            th, leg, p = ex.target
            log.info(f"[탐험] {done}번째 자리에서 못 찾음 (회전 {math.degrees(ex.spun):.0f}°) → 방향 {math.degrees(th):.0f}° 로 {leg:.2f} m 이동"
                     f" (odom {pose[0]:.2f},{pose[1]:.2f} → {p[0]:.2f},{p[1]:.2f}{', 지도 힌트 있음' if hint else ''})")
            ex.phase, ex.t_phase = "turn_to", now
        fb.state = "exploring"
        th, leg, p = ex.target
        if ex.phase == "turn_to":
            err = wrap(th - pose[2])
            if abs(err) > 0.15 and now - ex.t_phase < 12.0:
                self.vel(0.0, math.copysign(clip(1.5 * abs(err), 0.4, MAX_WZ), err))
                return None
            ex.phase, ex.t_phase, ex.start = "go", now, pose[:2]
            log.info(f"[탐험] 방향 맞춤 끝 (남은 오차 {math.degrees(err):.0f}°) → 이동")
        # go: 다음 관측 지점으로 직진하며 방향 보정. 정면 장애물이 가까우면 그 자리에서 멈춤
        dx, dy = p[0] - pose[0], p[1] - pose[1]
        b = wrap(math.atan2(dy, dx) - pose[2])
        fc = self.front_clear()
        moved = math.hypot(pose[0] - ex.start[0], pose[1] - ex.start[1])
        forward = abs(b) < 0.7                           # 방향이 크게 틀어졌으면 제자리에서 돌기만 함(장애물 검사 불필요)
        why = None
        if math.hypot(dx, dy) < 0.1 or moved >= leg:
            why = "도착"
        elif forward and fc is not None and fc < STOP_CLEAR:
            s = self.fresh_scan()
            detail = " ".join(f"{math.degrees(a):+.0f}:{r:.2f}" for a, r in zip(*s) if abs(a) <= math.radians(22)) if s else "스캔 없음"
            why = f"정면 장애물 {fc:.2f} m [{detail}]"
        elif now - ex.t_phase > leg / 0.06 + 6.0:
            why = "이동 시간 초과"
        if why:
            log.info(f"[탐험] 이동 끝({why}, {moved:.2f} m) → 다시 한 바퀴")
            if moved < 0.15:
                ex.blocked.append((ex.start, th))
            self.vel(0.0, 0.0)
            ex.reset_spin()
            return None
        if forward:                                      # 부드러운 보정: 세게 꺾으면 정책 회전 지연 때문에 좌우로 흔들림
            self.vel(MAX_VX * max(0.4, math.cos(b)), clip(1.0 * b, -0.5, 0.5))
        else:
            self.vel(0.0, math.copysign(clip(1.5 * abs(b), 0.4, MAX_WZ), b))
        return None

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
        explore = bool(self.get_parameter("explore").value)
        max_vp = int(self.get_parameter("max_viewpoints").value)
        total_timeout = EXPLORE_TIMEOUT if explore else TOTAL_TIMEOUT
        ex = None                                       # Explorer: 처음 안 보일 때 만듦
        while True:
            now = self.now_s()
            if gh.is_cancel_requested:
                reason = "canceled"; break
            if self._current is not gh:
                reason = "preempted"; break
            if now - t0 > total_timeout:
                reason = f"시간 초과 ({total_timeout:.0f}초)" + (f", {len(ex.visited)}곳에서 둘러봄" if ex else ""); break
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
                    if ex is None:
                        ex = Explorer(sign)
                    else:                               # 탐험 중 잠깐 봤다가 다시 놓침: 놓친 쪽으로 조금 더 돌아본 뒤 탐험 계속
                        ex.sign = sign
                        far = not ex.visited or min(math.hypot(self.pose[0] - v[0], self.pose[1] - v[1]) for v in ex.visited) > 0.3 if self.pose else True
                        ex.reset_spin(short=not far)    # 새 자리(이동 중에 봄)면 한 바퀴 전체
                if not explore or self.pose is None:    # 탐험 끔(또는 오도메트리 없음): 제자리에서만 찾기
                    if now - search_t0 > SEARCH_TIMEOUT:
                        reason = f"{label} 을(를) 찾지 못함 (제자리에서 한 바퀴 이상 돌았음)"; break
                    self.vel(0.0, sign * SEARCH_WZ)
                    fb.state, fb.distance, fb.bearing_deg = "searching", -1.0, 0.0
                else:
                    reason = self.explore_step(ex, label, now, max_vp, fb)
                    if reason:
                        break
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
