"""MuJoCo 방 시뮬레이션 + 걷기 정책(ONNX) ROS2 브리지.

구독:  /cmd_vel   (geometry_msgs/Twist)        x,y 이동 속도(m/s), angular.z 회전(rad/s). 허용 범위로 잘림 (이 정책은 뒤로는 거의 못 걸음)
       /head_cmd  (std_msgs/Float64MultiArray) 고개 관절 목표각 [neck_pitch, head_pitch, head_yaw, head_roll] (rad).
                                              위/왼쪽이 양수. 기본 자세는 [0,0,0,0]
발행:  /camera/image (rgb8), /camera/depth (32FC1, m), /camera/camera_info  — 머리 카메라
       /imu (sensor_msgs/Imu), /joint_states (sensor_msgs/JointState)
설계 원칙: 발행하는 것은 센서 값(카메라, IMU, 관절)뿐. 시뮬레이터의 정답 좌표는 내보내지 않는다.
실물 로봇으로 옮길 때는 이 노드를 실물 로봇 노드로 바꾸면 된다.
"""
import glob
import json
import math
import os
import sys

import numpy as np
import rclpy
from rclpy.node import Node
from builtin_interfaces.msg import Time
from geometry_msgs.msg import PoseStamped, Twist
from sensor_msgs.msg import CameraInfo, Image, Imu, JointState
from rosgraph_msgs.msg import Clock
from std_msgs.msg import Float64MultiArray, String


def find_repo_root():
    r = os.environ.get("DUCK_PET_ROOT")
    if r:
        return r
    d = os.path.dirname(os.path.abspath(__file__))
    while d != "/":
        if os.path.exists(os.path.join(d, "sim", "build_scene.py")):
            return d
        d = os.path.dirname(d)
    raise RuntimeError("저장소 루트를 못 찾음. DUCK_PET_ROOT 환경 변수를 설정하세요.")


ROOT = find_repo_root()
PLAYGROUND = os.path.join(ROOT, "third_party", "Open_Duck_Playground")
for p in (ROOT, PLAYGROUND):
    if p not in sys.path:
        sys.path.insert(0, p)

import mujoco  # noqa: E402
from sim.build_scene import PERSON_START, ROOM  # noqa: E402
from sim.headless import HeadlessDuck  # noqa: E402  (정책 실행 + 고개 직접 제어 + 관측 가리기)


class DuckSimBridge(Node):
    def __init__(self):
        super().__init__("duck_sim_bridge")
        latest = max(glob.glob(f"{ROOT}/models/runs/*/*.onnx"), key=os.path.getmtime)
        self.declare_parameter("onnx", latest)
        self.declare_parameter("viewer", True)          # MuJoCo 뷰어 창 띄우기
        self.declare_parameter("dynamic_objects", False)   # True: 공/상자가 밀리는 물체(공 밀기 시험)
        self.declare_parameter("novel_objects", False)   # True: 학습하지 않은 새 물체(노란 컵, 주황 공, 보라색 상자) 추가 — 8단계 개방형 탐지 시험
        self.declare_parameter("camera_hz", 10.0)
        self.declare_parameter("debug_pose", False)     # 측정 전용: 정답 위치를 /debug/base_pose 로 발행. 기술/인식 코드에서 쓰지 말 것
        self.declare_parameter("cam_width", 320)
        self.declare_parameter("cam_height", 240)
        onnx = os.path.realpath(self.get_parameter("onnx").value)
        self.get_logger().info(f"정책: {onnx}")

        self.core = HeadlessDuck(onnx, dynamic_objects=self.get_parameter("dynamic_objects").value, novel=self.get_parameter("novel_objects").value)
        self.m = self.core.m
        self.vel_cmd = [0.0, 0.0, 0.0]            # vx, vy, wz
        self.head_cmd = [0.0, 0.0, 0.0, 0.0]      # neck_pitch, head_pitch, head_yaw, head_roll

        w, h = self.get_parameter("cam_width").value, self.get_parameter("cam_height").value
        room = self.core.model
        self.rgb_r = mujoco.Renderer(room, h, w)
        self.depth_r = mujoco.Renderer(room, h, w)
        self.depth_r.enable_depth_rendering()
        self.cam_w, self.cam_h = w, h
        self.fovy = float(room.cam_fovy[room.camera("head_cam").id])

        self.viewer = None
        if self.get_parameter("viewer").value:
            from mujoco import viewer as mj_viewer
            self.viewer = mj_viewer.launch_passive(self.m.model, self.m.data,
                                                       show_left_ui=False, show_right_ui=False)

        self.create_subscription(Twist, "/cmd_vel", self.on_cmd_vel, 10)
        self.create_subscription(Float64MultiArray, "/head_cmd", self.on_head_cmd, 10)
        self.pub_img = self.create_publisher(Image, "/camera/image", 1)
        self.pub_depth = self.create_publisher(Image, "/camera/depth", 1)
        self.pub_info = self.create_publisher(CameraInfo, "/camera/camera_info", 1)
        self.pub_imu = self.create_publisher(Imu, "/imu", 10)
        self.pub_js = self.create_publisher(JointState, "/joint_states", 10)
        self.pub_clock = self.create_publisher(Clock, "/clock", 10)

        self.pub_dbg = self.create_publisher(PoseStamped, "/debug/base_pose", 10) if self.get_parameter("debug_pose").value else None
        # 사용자가 조종하는 사람(인형). /person/cmd_vel(Twist): x 앞뒤(m/s), angular.z 회전(rad/s). 방 안으로 제한
        self.person = list(PERSON_START)                    # x, y, yaw
        self.person_cmd = [0.0, 0.0, 0.0, None]             # vx, vy, wz, 받은 시각
        self.create_subscription(Twist, "/person/cmd_vel", self.on_person_cmd, 10)
        self.create_subscription(String, "/walk_mode", self.on_walk_mode, 10)   # "native": 걷기 정책을 학습된 그대로(고개 포함) / "override": 고개를 명령값으로 고정(기본)
        self.pub_obj_dbg = self.create_publisher(String, "/debug/object_poses", 5) if self.get_parameter("debug_pose").value else None   # 측정 전용
        self.pub_person_dbg = self.create_publisher(PoseStamped, "/debug/person_pose", 10) if self.get_parameter("debug_pose").value else None
        self.counter = 0
        self.last_cmd_time = self.get_clock().now()
        # 정책 1번(50Hz) = 물리 10스텝(500Hz). 실시간 속도로 돌림
        self.create_timer(self.m.sim_dt * self.m.decimation, self.tick)
        hz = self.get_parameter("camera_hz").value
        self.create_timer(1.0 / hz, self.publish_camera)
        self.get_logger().info("준비 완료. /cmd_vel 로 조종하세요.")

    # ---- 입력 ----
    def on_cmd_vel(self, msg):
        # 범위 자르기는 core.step 이 함
        self.vel_cmd = [msg.linear.x, msg.linear.y, msg.angular.z]
        self.last_cmd_time = self.get_clock().now()

    def on_head_cmd(self, msg):
        if len(msg.data) != 4:
            self.get_logger().warn("/head_cmd 는 값 4개 [neck_pitch, head_pitch, head_yaw, head_roll] 필요")
            return
        self.head_cmd = [float(v) for v in msg.data]   # 고개는 계속 유지됨(속도 명령과 달리 자동 해제 없음)

    # ---- 시뮬레이션 ----
    def on_person_cmd(self, msg):
        self.person_cmd = [msg.linear.x, msg.linear.y, msg.angular.z, self.get_clock().now()]

    def on_walk_mode(self, msg):
        self.core.native_head = (msg.data.strip() == "native")
        self.get_logger().info(f"걷기 모드: {'native(고개도 정책이)' if self.core.native_head else 'override(고개 명령값)'}")

    def update_person(self):
        dt = self.m.sim_dt * self.m.decimation
        vx, vy, wz, t = self.person_cmd
        if t is None or (self.get_clock().now() - t).nanoseconds > 0.5e9:      # 키를 떼면 멈춤
            vx = vy = wz = 0.0
        x, y, yaw = self.person
        yaw += wz * dt
        x += (vx * math.cos(yaw) - vy * math.sin(yaw)) * dt
        y += (vx * math.sin(yaw) + vy * math.cos(yaw)) * dt
        x, y = max(-ROOM["x"] / 2 + 0.2, min(ROOM["x"] / 2 - 0.2, x)), max(-ROOM["y"] / 2 + 0.2, min(ROOM["y"] / 2 - 0.2, y))
        self.person = [x, y, yaw]
        d = self.core.data
        d.mocap_pos[0] = [x, y, 0.0]
        d.mocap_quat[0] = [math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)]

    def sim_time(self):
        """시뮬레이션 시간(builtin_interfaces/Time). 시뮬레이션이 실시간보다 느려도 센서 시각이 정확하도록 이것을 쓴다."""
        t = float(self.core.data.time)
        return Time(sec=int(t), nanosec=int((t - int(t)) * 1e9))

    def tick(self):
        # 속도 명령이 0.5초 이상 안 오면 정지 (키보드는 누르는 동안만 명령이 오기 때문)
        if (self.get_clock().now() - self.last_cmd_time).nanoseconds > 0.5e9:
            self.vel_cmd = [0.0, 0.0, 0.0]
        vx, vy, wz = self.vel_cmd
        self.core.step(vx=vx, vy=vy, wz=wz, head=self.head_cmd)   # 정책 1번(0.02s) = 물리 10스텝
        self.update_person()
        if self.viewer is not None:
            if self.viewer.is_running():
                self.viewer.sync()
            else:
                self.get_logger().info("뷰어 창이 닫혀 종료합니다.")
                raise KeyboardInterrupt
        self.pub_clock.publish(Clock(clock=self.sim_time()))   # 다른 노드가 use_sim_time 으로 시뮬레이션 시간을 쓰게 함
        self.publish_fast()
        if self.pub_dbg is not None:
            x, y, yaw = self.core.pose()
            pm = PoseStamped()
            pm.header.stamp, pm.header.frame_id = self.sim_time(), "world"
            pm.pose.position.x, pm.pose.position.y = x, y
            pm.pose.orientation.z, pm.pose.orientation.w = math.sin(yaw / 2), math.cos(yaw / 2)
            self.pub_dbg.publish(pm)
            if self.pub_obj_dbg is not None and self.m.model.nbody > 0:
                d = self.m.data
                objs = {n: [round(float(v), 3) for v in d.xpos[self.m.model.body(n).id][:2]] for n in ("blue_ball", "red_block", "green_block")}
                self.pub_obj_dbg.publish(String(data=json.dumps(objs)))
            if self.pub_person_dbg is not None:
                pp = PoseStamped()
                pp.header = pm.header
                pp.pose.position.x, pp.pose.position.y = self.person[0], self.person[1]
                pp.pose.orientation.z, pp.pose.orientation.w = math.sin(self.person[2] / 2), math.cos(self.person[2] / 2)
                self.pub_person_dbg.publish(pp)

    def publish_fast(self):
        m, now = self.m, self.sim_time()
        d = m.data
        imu = Imu()
        imu.header.stamp, imu.header.frame_id = now, "imu"
        g = d.sensordata[m.gyro_addr:m.gyro_addr + 3]
        acc = d.sensordata[m.accelerometer_addr:m.accelerometer_addr + 3]
        imu.angular_velocity.x, imu.angular_velocity.y, imu.angular_velocity.z = map(float, g)
        imu.linear_acceleration.x, imu.linear_acceleration.y, imu.linear_acceleration.z = map(float, acc)
        imu.orientation_covariance[0] = -1.0  # 방향 값 없음(실물 IMU 도 필터 없이는 없음)
        self.pub_imu.publish(imu)
        js = JointState()
        js.header.stamp = now
        js.name = list(m.actuator_names)
        js.position = [float(d.qpos[int(a)]) for a in m.actuator_joint_qpos_addr]
        js.velocity = [float(d.qvel[int(a)]) for a in m.actuator_qvel_addr]
        self.pub_js.publish(js)
        if self.pub_dbg is not None:
            ps = PoseStamped()
            ps.header.stamp, ps.header.frame_id = now, "world"
            q = d.qpos[m._floating_base_qpos_addr:m._floating_base_qpos_addr + 7]
            ps.pose.position.x, ps.pose.position.y, ps.pose.position.z = map(float, q[:3])
            ps.pose.orientation.w, ps.pose.orientation.x, ps.pose.orientation.y, ps.pose.orientation.z = map(float, q[3:7])
            self.pub_dbg.publish(ps)

    def publish_camera(self):
        now = self.sim_time()
        d, model = self.m.data, self.m.model
        self.rgb_r.update_scene(d, camera="head_cam")
        rgb = np.ascontiguousarray(self.rgb_r.render())
        self.depth_r.update_scene(d, camera="head_cam")
        depth = np.ascontiguousarray(self.depth_r.render().astype(np.float32))
        depth[~np.isfinite(depth)] = 0.0
        for pub, arr, enc, step in ((self.pub_img, rgb, "rgb8", self.cam_w * 3),
                                    (self.pub_depth, depth, "32FC1", self.cam_w * 4)):
            msg = Image()
            msg.header.stamp, msg.header.frame_id = now, "head_cam"
            msg.height, msg.width, msg.encoding, msg.step = self.cam_h, self.cam_w, enc, step
            msg.data = arr.tobytes()
            pub.publish(msg)
        f = 0.5 * self.cam_h / np.tan(np.radians(self.fovy) / 2)
        info = CameraInfo()
        info.header.stamp, info.header.frame_id = now, "head_cam"
        info.height, info.width = self.cam_h, self.cam_w
        info.k = [f, 0.0, self.cam_w / 2, 0.0, f, self.cam_h / 2, 0.0, 0.0, 1.0]
        info.p = [f, 0.0, self.cam_w / 2, 0.0, 0.0, f, self.cam_h / 2, 0.0, 0.0, 0.0, 1.0, 0.0]
        self.pub_info.publish(info)


def main():
    rclpy.init()
    node = DuckSimBridge()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
