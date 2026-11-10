"""ROS 없이 같은 정책/방 모델을 돌려 측정하는 도구 (보정·검증용).
주의: 여기서 읽는 위치/방향은 시뮬레이터 정답값이다. 오리의 기술(duck_skills) 안에서는 쓰지 않고,
기술의 속도 보정값을 구하거나 오차를 재는 데만 쓴다.
"""
import glob, os, sys
import numpy as np
import mujoco

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PLAYGROUND = os.path.join(ROOT, "third_party", "Open_Duck_Playground")
for p in (ROOT, PLAYGROUND):
    if p not in sys.path:
        sys.path.insert(0, p)
from sim.build_scene import build  # noqa: E402
from playground.open_duck_mini_v2.mujoco_infer import MjInfer, USE_MOTOR_SPEED_LIMITS  # noqa: E402

BASE = os.path.join(PLAYGROUND, "playground", "open_duck_mini_v2")


class HeadlessDuck:
    def __init__(self, onnx=None, dynamic_objects=False, novel=False):
        onnx = onnx or max(glob.glob(f"{ROOT}/models/runs/*/*.onnx"), key=os.path.getmtime)
        room = build(dynamic_objects=dynamic_objects, novel=novel)       # True 면 공/상자가 밀리는 물체(공 밀기 등)
        orig = mujoco.MjModel.from_xml_string
        mujoco.MjModel.from_xml_string = staticmethod(lambda *a, **k: room)
        try:
            self.m = MjInfer(os.path.join(BASE, "xmls/scene_flat_terrain_backlash.xml"),
                             os.path.join(BASE, "data/polynomial_coefficients.pkl"), os.path.realpath(onnx), False)
        finally:
            mujoco.MjModel.from_xml_string = orig
        self.model, self.data = self.m.model, self.m.data
        self.cam_id = self.model.camera("head_cam").id
        self.base_id = self.model.body("base").id
        self.head_target = np.zeros(4)
        self.native_head = False   # True: 걷기 정책을 학습된 그대로 사용(고개도 정책이 움직임, 관측 가리기 없음). 공 밀기 상위 정책이 이 방식으로 학습됨
        self.obs_head = "mask"  # cmd=그대로, zero=명령 칸만 0, mask=고개 관련 관측 전부를 기본 자세로 가림(걷기 유지에 필요)

    def step(self, vx=0.0, vy=0.0, wz=0.0, head=(0, 0, 0, 0)):
        """정책 1스텝(0.02s). 명령은 bridge_node 와 같은 범위로 자름."""
        m = self.m
        m.commands[0] = float(np.clip(vx, *m.COMMANDS_RANGE_X))
        m.commands[1] = float(np.clip(vy, *m.COMMANDS_RANGE_Y))
        m.commands[2] = float(np.clip(wz, *m.COMMANDS_RANGE_THETA))
        rng = [m.NECK_PITCH_RANGE, m.HEAD_PITCH_RANGE, m.HEAD_YAW_RANGE, m.HEAD_ROLL_RANGE]
        for i, (v, r) in enumerate(zip(head, rng)):
            m.commands[3 + i] = float(np.clip(v, *r))
        for _ in range(m.decimation):
            mujoco.mj_step(self.model, self.data)
        m.imitation_i = (m.imitation_i + m.phase_frequency_factor) % m.PRM.nb_steps_in_period
        a = m.imitation_i / m.PRM.nb_steps_in_period * 2 * np.pi
        m.imitation_phase = np.array([np.cos(a), np.sin(a)])
        obs_cmd = list(m.commands)
        if self.native_head:
            obs_cmd[3:7] = [0.0] * 4
        if self.obs_head == "zero":
            obs_cmd[3:7] = [0.0] * 4
        obs = m.get_obs(self.data, obs_cmd)
        if self.obs_head == "mask" and not self.native_head:
            # 정책이 고개를 항상 기본 자세로 보게 함. 관측 레이아웃(101): gyro3 acc3 cmd7 관절각14 관절속도14 행동3x14 목표14 접촉2 위상2
            obs = obs.copy()
            obs[9:13] = 0.0                                   # 명령의 고개 4칸
            obs[13 + 5:13 + 9] = 0.0                          # 고개 관절 각도(기본 대비)
            obs[27 + 5:27 + 9] = 0.0                          # 고개 관절 속도
            obs[83 + 5:83 + 9] = m.default_actuator[5:9]      # 직전 모터 목표
        action = m.policy.infer(obs)
        m.last_last_last_action = m.last_last_action.copy()
        m.last_last_action = m.last_action.copy()
        m.last_action = action.copy()
        m.motor_targets = m.default_actuator + action * m.action_scale
        if USE_MOTOR_SPEED_LIMITS:
            dv = m.max_motor_velocity * m.sim_dt * m.decimation
            m.motor_targets = np.clip(m.motor_targets, m.prev_motor_targets - dv, m.prev_motor_targets + dv)
            m.prev_motor_targets = m.motor_targets.copy()
        # 고개 4관절(액추에이터 5~8)은 정책이 아니라 우리가 직접 지정한다.
        # 이 정책은 고개 명령을 따르도록 학습되지 않았음(관측값으로만 들어감). 속도 제한은 다른 관절과 동일.
        if self.native_head:                       # 고개도 정책 출력 그대로 (위에서 속도 제한 적용됨)
            self.data.ctrl = m.motor_targets.copy()
            return
        dv = m.max_motor_velocity * m.sim_dt * m.decimation
        head_cmd = np.array(m.commands[3:7])
        self.head_target = np.clip(head_cmd, self.head_target - dv, self.head_target + dv)  # 직전 "고개 목표"에서 출발
        m.motor_targets[5:9] = self.head_target
        m.prev_motor_targets = m.motor_targets.copy()
        self.data.ctrl = m.motor_targets.copy()

    # --- 정답값(측정 전용) ---
    def pose(self):
        """(x, y, yaw[rad]) 몸통 기준"""
        R = self.data.xmat[self.base_id].reshape(3, 3)
        return float(self.data.xpos[self.base_id][0]), float(self.data.xpos[self.base_id][1]), float(np.arctan2(R[1, 0], R[0, 0]))

    def cam_dir_body(self):
        """카메라가 보는 방향을 몸통 좌표계의 (yaw, pitch)로. yaw 양수=왼쪽, pitch 양수=위"""
        Rc = self.data.cam_xmat[self.cam_id].reshape(3, 3)
        fwd_w = -Rc[:, 2]
        Rb = self.data.xmat[self.base_id].reshape(3, 3)
        f = Rb.T @ fwd_w
        return float(np.arctan2(f[1], f[0])), float(np.arcsin(np.clip(f[2], -1, 1)))
