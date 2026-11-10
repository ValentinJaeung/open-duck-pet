"""인식 노드: 머리 카메라 → 물체 탐지(YOLO) → 3D 위치(깊이) → 오도메트리 → 의미 지도.

입력(센서만 사용. 시뮬레이터 정답은 쓰지 않음):
  /camera/image, /camera/depth, /camera/camera_info, /joint_states(고개 각도), /imu(자이로), /cmd_vel(이동 추정)
출력:
  /detections      (duck_interfaces/DetectedObjectArray) 이번 프레임에서 본 물체
  /semantic_map    (visualization_msgs/MarkerArray)      지금까지 기억한 물체 위치 (rviz2, Fixed Frame = odom)
  /detections_image(sensor_msgs/Image)                    상자를 그린 카메라 화면
  TF odom → base_link                                    추정한 오리 위치

오도메트리(추정): 방향 = IMU 자이로 적분, 위치 = 명령 속도 x 실측 비율(0.7)을 적분. 오차가 쌓이는 단순한 방식이다.
카메라 위치: 로봇 자신의 관절 각도 + 로봇 모델(운동학)로 계산한다. (로봇이 자기 몸 구조를 아는 것이라 정답 정보가 아님)
"""
import base64
import json
import math
import os
import re
import sys
import threading
import time
import urllib.request

import numpy as np
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Point, TransformStamped, Twist
from sensor_msgs.msg import CameraInfo, Image, Imu, JointState, LaserScan
from std_msgs.msg import String
from tf2_ros import TransformBroadcaster
from visualization_msgs.msg import Marker, MarkerArray
from duck_interfaces.msg import DetectedObject, DetectedObjectArray
from PIL import Image as PILImage, ImageDraw


def norm_label(text):
    """'Yellow Cup' → 'yellow_cup' (탐지 라벨의 표준형)"""
    return re.sub(r"[^a-z0-9]+", "_", str(text).lower()).strip("_")


def find_root():
    r = os.environ.get("DUCK_PET_ROOT")
    if r:
        return r
    d = os.path.dirname(os.path.abspath(__file__))
    while d != "/":
        if os.path.exists(os.path.join(d, "sim", "build_scene.py")):
            return d
        d = os.path.dirname(d)
    raise RuntimeError("저장소 루트를 못 찾음. DUCK_PET_ROOT 설정 필요")


ROOT = find_root()
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
import mujoco  # noqa: E402
from sim.build_scene import build  # noqa: E402
from duck_perception.yolo_onnx import YoloOnnx  # noqa: E402

CLASSES = ["red_block", "green_block", "blue_ball", "person"]
COLORS = {"red_block": (1.0, 0.1, 0.1), "green_block": (0.1, 0.8, 0.2), "blue_ball": (0.2, 0.4, 1.0), "person": (1.0, 0.8, 0.1)}
FWD_GAIN, BACK_GAIN = 0.7, 0.35     # 명령 속도 대비 실제 속도 비율 (측정: 0.15 명령 → 약 0.105 m/s)
PERSON_HALF_DEPTH = 0.08             # 사람은 몸 두께 절반 정도를 더함
HALF_SIZE = 0.045                    # 물체 반 크기(m): 상자/공 반지름이 약 0.04~0.05
MIN_OBS = 3                          # 지도에 표시하려면 최소 관측 횟수
CMD_TIMEOUT = 0.5                    # 브리지와 같이, 이 시간 동안 명령이 없으면 정지로 본다
# 깊이 → 가상 레이저 스캔(/depth_scan): 몸통 기준 방향별 '가장 가까운 장애물' 거리. 탐험할 때 갈 수 있는 쪽을 고르는 데 씀
FLOOR_Z = -0.155                     # 몸통 좌표에서 바닥 높이(m). 서 있을 때 몸통 높이 약 0.155~0.158
OBST_MIN_H, OBST_MAX_H = 0.04, 0.6   # 바닥 위 이 높이 사이의 점을 장애물로 봄 (바닥 자체와 천장 쪽 제외)
SCAN_HALF, SCAN_STEP = math.radians(50), math.radians(2)
SCAN_MAX = 3.0                       # 장애물이 안 보이는 방향의 거리(m)
SELF_R = 0.12                        # 몸통 중심에서 이 안의 점은 자기 몸(다리/발)으로 보고 무시


class Perception(Node):
    def __init__(self):
        super().__init__("duck_perception")
        self.declare_parameter("model", os.path.join(ROOT, "models", "yolo", "duck_objects.onnx"))
        self.declare_parameter("conf", 0.5)
        self.yolo = YoloOnnx(self.get_parameter("model").value, CLASSES, conf=self.get_parameter("conf").value)
        # 로봇 몸 구조(운동학)용 모델. 관절 각도로 카메라 위치를 계산할 때만 쓴다.
        self.model = build(dynamic_objects=False)
        self.data = mujoco.MjData(self.model)
        mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
        self.cam_id, self.base_id = self.model.camera("head_cam").id, self.model.body("base").id
        self.qadr = {self.model.joint(i).name: self.model.jnt_qposadr[i] for i in range(self.model.njnt)}

        self.K = None
        self.depth = None
        self.joints = {}
        self.x = self.y = self.yaw = 0.0                 # odom 에서의 추정 자세
        self.cmd = (0.0, 0.0, 0.0, None)                  # vx, vy, wz, 받은 시각
        self.last_imu_t = None
        self.last_odom_t = self.get_clock().now()
        self.landmarks = []                               # {label, pos(np3), n, seen(시각)}

        self.create_subscription(CameraInfo, "/camera/camera_info", lambda m: setattr(self, "K", m), 1)
        self.create_subscription(Image, "/camera/depth", self.on_depth, 1)
        self.create_subscription(Image, "/camera/image", self.on_image, 1)
        self.create_subscription(JointState, "/joint_states", self.on_joints, 10)
        self.create_subscription(Imu, "/imu", self.on_imu, 50)
        self.create_subscription(Twist, "/cmd_vel", self.on_cmd, 10)
        self.pub_det = self.create_publisher(DetectedObjectArray, "/detections", 5)
        self.pub_map = self.create_publisher(MarkerArray, "/semantic_map", 1)
        self.pub_img = self.create_publisher(Image, "/detections_image", 1)
        self.pub_scene = self.create_publisher(String, "/scene", 1)
        self.pub_scan = self.create_publisher(LaserScan, "/depth_scan", 5)
        # --- 개방형 탐지(8단계): YOLO 가 모르는 이름은 OWLv2 서버에 물어본다. 물어볼 이름은 /ovd_prompts(JSON 문자열 목록)로 받음
        self.declare_parameter("ovd_url", "http://127.0.0.1:8765")
        self.declare_parameter("ovd_hz", 2.0)
        self.declare_parameter("ovd_threshold", 0.12)
        self.ovd_prompts, self.ovd_vis, self.last, self.ovd_err_t = [], [], None, 0.0
        self.kin_lock = threading.Lock()
        self.create_subscription(String, "/ovd_prompts", self.on_ovd_prompts, 5)
        threading.Thread(target=self.ovd_loop, daemon=True).start()
        self.tf = TransformBroadcaster(self)
        self.create_timer(0.5, self.publish_scene)
        self.create_timer(0.05, self.odom_tick)
        self.get_logger().info("인식 노드 준비 (YOLO: %s)" % self.get_parameter("model").value)

    # ---- 센서 입력 ----
    def on_depth(self, m):
        self.depth = np.frombuffer(m.data, np.float32).reshape(m.height, m.width)

    def on_joints(self, m):
        self.joints = dict(zip(m.name, m.position))

    def on_cmd(self, m):
        self.cmd = (m.linear.x, m.linear.y, m.angular.z, self.get_clock().now())

    def on_imu(self, m):
        t = m.header.stamp.sec + m.header.stamp.nanosec * 1e-9
        if self.last_imu_t is not None and 0 < t - self.last_imu_t < 0.5:
            self.yaw += m.angular_velocity.z * (t - self.last_imu_t)        # 자이로 적분 = 방향
        self.last_imu_t = t

    def odom_tick(self):
        now = self.get_clock().now()
        dt = (now - self.last_odom_t).nanoseconds * 1e-9
        self.last_odom_t = now
        vx, vy, _, t = self.cmd
        if t is not None and (now - t).nanoseconds * 1e-9 < CMD_TIMEOUT:
            v = vx * (FWD_GAIN if vx >= 0 else BACK_GAIN)
            self.x += (v * math.cos(self.yaw) - vy * math.sin(self.yaw)) * dt
            self.y += (v * math.sin(self.yaw) + vy * math.cos(self.yaw)) * dt
        tf = TransformStamped()
        tf.header.stamp, tf.header.frame_id, tf.child_frame_id = now.to_msg(), "odom", "base_link"
        tf.transform.translation.x, tf.transform.translation.y = self.x, self.y
        tf.transform.rotation.z, tf.transform.rotation.w = math.sin(self.yaw / 2), math.cos(self.yaw / 2)
        self.tf.sendTransform(tf)

    # ---- 카메라 → 3D ----
    def now_s(self):
        return self.get_clock().now().nanoseconds * 1e-9

    def on_ovd_prompts(self, msg):
        try:
            ps = [norm_label(p).replace("_", " ") for p in json.loads(msg.data)]
        except Exception:
            return
        self.ovd_prompts = [p for p in dict.fromkeys(ps) if p and norm_label(p) not in CLASSES][:5]       # YOLO 가 아는 것은 제외, 최대 5개
        self.get_logger().info(f"개방형 탐지 질문: {self.ovd_prompts}")

    def ovd_loop(self):
        """질문(prompts)이 있으면 최근 프레임을 OWLv2 서버로 보내 결과를 /detections 로 낸다 (별도 스레드: YOLO 인식을 막지 않음)"""
        while True:
            time.sleep(1.0 / self.get_parameter("ovd_hz").value)
            prompts, last = list(self.ovd_prompts), self.last
            if not prompts or last is None or self.K is None or not self.joints:
                continue
            rgb, depth, header = last
            try:
                req = {"width": rgb.shape[1], "height": rgb.shape[0], "rgb_b64": base64.b64encode(rgb.tobytes()).decode(),
                       "prompts": prompts, "threshold": self.get_parameter("ovd_threshold").value}
                r = json.loads(urllib.request.urlopen(urllib.request.Request(
                    self.get_parameter("ovd_url").value + "/detect", json.dumps(req).encode(), {"Content-Type": "application/json"}), timeout=3).read())
            except Exception as e:
                if self.now_s() - self.ovd_err_t > 10:
                    self.ovd_err_t = self.now_s()
                    self.get_logger().warn(f"개방형 탐지 서버 응답 없음 ({type(e).__name__}). scripts/run_ovd.sh 를 켜세요")
                continue
            with self.kin_lock:
                Rcb, pcb = self.cam_in_base()
            f, cx, cy = self.K.k[0], self.K.k[2], self.K.k[5]
            out, vis, now = DetectedObjectArray(), [], self.get_clock().now()
            out.header = header
            out.header.frame_id = "base_link"
            for d in r.get("detections", []):
                label = norm_label(d["prompt"])
                out.objects.append(self.make_obj(label, d["score"], d["box"], rgb.shape[1], rgb.shape[0], depth, Rcb, pcb, f, cx, cy, now))
                vis.append((label, d["score"], tuple(d["box"]), self.now_s()))
            self.ovd_vis = vis
            if out.objects:
                self.pub_det.publish(out)

    def cam_in_base(self):
        """로봇 관절 각도로 계산한 카메라의 몸통 기준 자세 (R: 3x3, p: 3). 몸통 x 앞, y 왼쪽, z 위."""
        for n, v in self.joints.items():
            if n in self.qadr:
                self.data.qpos[self.qadr[n]] = v
        mujoco.mj_kinematics(self.model, self.data)
        mujoco.mj_camlight(self.model, self.data)       # 카메라 자세(cam_xpos/cam_xmat)는 이 함수가 갱신함
        Rb = self.data.xmat[self.base_id].reshape(3, 3); pb = self.data.xpos[self.base_id]
        Rc = self.data.cam_xmat[self.cam_id].reshape(3, 3); pc = self.data.cam_xpos[self.cam_id]
        return Rb.T @ Rc, Rb.T @ (pc - pb)

    def make_obj(self, label, score, box, W, Ht, depth_img, Rcb, pcb, f, cx, cy, now):
        """탐지 상자 하나 → DetectedObject (깊이 중앙값으로 3D 위치). YOLO 와 개방형(OWLv2) 탐지가 함께 쓴다"""
        x0, y0, x1, y1 = box
        o = DetectedObject()
        o.label, o.score = label, float(score)
        o.u, o.v, o.width, o.height = (x0 + x1) / 2, (y0 + y1) / 2, x1 - x0, y1 - y0
        # 상자 중앙 절반 영역의 깊이 중앙값 (배경/가장자리 섞임을 줄임)
        w4, h4 = (x1 - x0) / 4, (y1 - y0) / 4
        patch = depth_img[int(max(0, o.v - h4)):int(o.v + h4) + 1, int(max(0, o.u - w4)):int(o.u + w4) + 1]
        patch = patch[patch > 0.05]
        # 키 큰 사람은 화면 위/아래에 닿는 게 정상이라 좌우만 본다. 작은 물체는 4면 모두(잘린 상자는 깊이가 부정확)
        touches = x0 < 2 or x1 > W - 2 or (label != "person" and (y0 < 2 or y1 > Ht - 2))
        if len(patch) and not touches:
            z = float(np.median(patch))
            if 0.2 < z < 3.5:
                o.depth = z
                z += PERSON_HALF_DEPTH if label == "person" else HALF_SIZE                    # 깊이는 물체 표면까지의 거리 → 중심까지 반 크기만큼 더함
                # 카메라 좌표(MuJoCo: x 오른쪽, y 위, -z 앞) → 몸통 → odom
                pc = np.array([(o.u - cx) / f * z, -(o.v - cy) / f * z, -z])
                pb = Rcb @ pc + pcb
                o.base_xyz = Point(x=float(pb[0]), y=float(pb[1]), z=float(pb[2]))
                c, sn = math.cos(self.yaw), math.sin(self.yaw)
                po = np.array([self.x + c * pb[0] - sn * pb[1], self.y + sn * pb[0] + c * pb[1], pb[2]])
                o.odom_xyz = Point(x=float(po[0]), y=float(po[1]), z=float(po[2]))
                if label != "person":                  # 사람은 움직이므로 지도에 기억하지 않고 탐지로만 알림
                    self.update_map(label, po, now, z)
        return o

    def on_image(self, m):
        if self.K is None or self.depth is None or not self.joints:
            return
        rgb = np.frombuffer(m.data, np.uint8).reshape(m.height, m.width, 3)
        self.last = (rgb.copy(), self.depth.copy(), m.header)           # 개방형 탐지 스레드가 쓸 최근 프레임
        dets = self.yolo(rgb)
        f, cx, cy = self.K.k[0], self.K.k[2], self.K.k[5]
        with self.kin_lock:
            Rcb, pcb = self.cam_in_base()
        out = DetectedObjectArray()
        out.header = m.header
        out.header.frame_id = "base_link"
        now = self.get_clock().now()
        for label, score, x0, y0, x1, y1 in dets:
            o = self.make_obj(label, score, (x0, y0, x1, y1), m.width, m.height, self.depth, Rcb, pcb, f, cx, cy, now)
            out.objects.append(o)
        self.pub_det.publish(out)
        self.publish_scan(m.header, self.depth, Rcb, pcb, f, cx, cy)
        self.publish_annotated(m, dets + [(l, sc, *bx) for (l, sc, bx, t) in self.ovd_vis if self.now_s() - t < 1.5])
        self.publish_map(now)

    def publish_scan(self, header, depth_img, Rcb, pcb, f, cx, cy, stride=2):
        """깊이 영상 → 몸통 기준 가상 레이저 스캔. 각 방향 칸: 장애물 최소 거리, 장애물 없이 보이기만 하면 SCAN_MAX, 안 보이면 nan"""
        d = depth_img[::stride, ::stride]
        vs, us = np.mgrid[0:depth_img.shape[0]:stride, 0:depth_img.shape[1]:stride]
        ok = (d > 0.05) & (d < 6.0)
        z, u, v = d[ok], us[ok], vs[ok]
        pc = np.stack([(u - cx) / f * z, -(v - cy) / f * z, -z], axis=1)
        pb = pc @ Rcb.T + pcb                                   # 몸통 좌표 (x 앞, y 왼쪽, z 위)
        r = np.hypot(pb[:, 0], pb[:, 1])
        ang = np.arctan2(pb[:, 1], pb[:, 0])
        h = pb[:, 2] - FLOOR_Z
        # 걸을 때 몸통이 1~2° 만 기울어도 1m 밖 바닥이 3~4cm 높아 보임 → 바닥 후보 점으로 평면 z = ax + by + c 를 매 프레임 맞춤
        for _ in range(2):
            fl = (np.abs(h) < 0.05) & (r < 2.5)
            if fl.sum() < 200:
                break
            A = np.stack([pb[fl, 0], pb[fl, 1], np.ones(fl.sum())], axis=1)
            coef = np.linalg.lstsq(A, pb[fl, 2], rcond=None)[0]
            h = pb[:, 2] - (pb[:, 0] * coef[0] + pb[:, 1] * coef[1] + coef[2])
        n = int(round(2 * SCAN_HALF / SCAN_STEP)) + 1
        idx = np.round((ang + SCAN_HALF) / SCAN_STEP).astype(int)
        keep = (idx >= 0) & (idx < n) & (r > SELF_R)
        ranges = np.full(n, np.nan, np.float32)
        seen = np.bincount(idx[keep], minlength=n) > 0
        ranges[seen] = SCAN_MAX
        obst = keep & (h > OBST_MIN_H) & (h < OBST_MAX_H)
        if obst.any():
            mins = np.full(n, np.inf)
            np.minimum.at(mins, idx[obst], r[obst])
            hit = np.isfinite(mins)
            ranges[hit] = np.minimum(mins[hit], SCAN_MAX)
        s = LaserScan()
        s.header = header
        s.header.frame_id = "base_link"
        s.angle_min, s.angle_max, s.angle_increment = -SCAN_HALF, SCAN_HALF, SCAN_STEP
        s.range_min, s.range_max = float(SELF_R), float(SCAN_MAX)
        s.ranges = ranges.tolist()
        self.pub_scan.publish(s)

    # ---- 의미 지도 ----
    def update_map(self, label, pos, now, depth):
        # 같은 종류이고 gate 안이면 같은 물체로 본다. 깊이 오차와 오도메트리 방향 오차는 멀수록 커지므로 gate 도 거리에 비례
        best, bd = None, 0.3 + 0.2 * depth
        for lm in self.landmarks:
            d = np.linalg.norm(lm["pos"][:2] - pos[:2])
            if lm["label"] == label and d < bd:
                best, bd = lm, d
        if best is None:
            self.landmarks.append({"label": label, "pos": pos.copy(), "n": 1, "seen": now})
        else:
            best["n"] += 1
            a = max(1.0 / best["n"], 0.15)             # 새 관측을 조금씩 반영(방 배치가 바뀌어도 따라감)
            best["pos"] = (1 - a) * best["pos"] + a * pos
            best["seen"] = now

    def publish_scene(self):
        """기억한 물체를 '오리 기준 거리/방향' 으로 정리해 /scene(JSON 문자열)로 발행. LLM 에게 줄 '지금 아는 물체 목록'."""
        now = self.get_clock().now()
        objs = []
        for lm in self.landmarks:
            if lm["n"] < MIN_OBS:
                continue
            dx, dy = lm["pos"][0] - self.x, lm["pos"][1] - self.y
            b = math.atan2(dy, dx) - self.yaw
            b = (b + math.pi) % (2 * math.pi) - math.pi
            objs.append({"label": lm["label"], "distance_m": round(math.hypot(dx, dy), 2),
                         "bearing_deg": round(math.degrees(b)),            # 0=정면, +왼쪽, -오른쪽
                         "visible_now": (now - lm["seen"]).nanoseconds * 1e-9 < 1.0})
        self.pub_scene.publish(String(data=json.dumps({"objects": objs}, ensure_ascii=False)))

    def publish_map(self, now):
        arr = MarkerArray()
        d = Marker(); d.action = Marker.DELETEALL; arr.markers.append(d)
        for i, lm in enumerate(self.landmarks):
            if lm["n"] < MIN_OBS:                       # 한두 번 본 것은 오탐/먼 거리 노이즈일 수 있어 숨김
                continue
            r, g, b = COLORS.get(lm["label"], (0.9, 0.5, 0.9))
            mk = Marker()
            mk.header.frame_id, mk.header.stamp = "odom", now.to_msg()
            mk.ns, mk.id, mk.type, mk.action = "objects", i, Marker.SPHERE, Marker.ADD
            mk.pose.position.x, mk.pose.position.y, mk.pose.position.z = map(float, lm["pos"])
            mk.pose.orientation.w = 1.0
            mk.scale.x = mk.scale.y = mk.scale.z = 0.12
            mk.color.r, mk.color.g, mk.color.b, mk.color.a = r, g, b, 1.0
            arr.markers.append(mk)
            tx = Marker()
            tx.header = mk.header
            tx.ns, tx.id, tx.type, tx.action = "labels", i, Marker.TEXT_VIEW_FACING, Marker.ADD
            tx.pose.position.x, tx.pose.position.y, tx.pose.position.z = lm["pos"][0], lm["pos"][1], lm["pos"][2] + 0.15
            tx.pose.orientation.w = 1.0
            tx.scale.z = 0.08
            tx.color.r = tx.color.g = tx.color.b = tx.color.a = 1.0
            tx.text = f"{lm['label']} ({lm['n']})"
            arr.markers.append(tx)
        self.pub_map.publish(arr)

    def publish_annotated(self, m, dets):
        im = PILImage.fromarray(np.frombuffer(m.data, np.uint8).reshape(m.height, m.width, 3).copy())
        d = ImageDraw.Draw(im)
        for label, score, x0, y0, x1, y1 in dets:
            col = tuple(int(255 * c) for c in COLORS.get(label, (0.9, 0.5, 0.9)))
            d.rectangle([x0, y0, x1, y1], outline=col, width=2)
            d.text((x0 + 2, max(0, y0 - 11)), f"{label} {score:.2f}", fill=(255, 255, 255))
        msg = Image()
        msg.header, msg.height, msg.width, msg.encoding, msg.step = m.header, m.height, m.width, "rgb8", m.width * 3
        msg.data = np.asarray(im).tobytes()
        self.pub_img.publish(msg)


def main():
    rclpy.init()
    node = Perception()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
