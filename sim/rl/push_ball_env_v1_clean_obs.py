"""공 밀기 환경 (계층형 RL): 상위 정책이 속도 명령(vx, vy, wz)을 내면, 고정된 걷기 정책이 걷는다.

  상위 정책 관측(9): 공 상대위치(2), 목표 상대위치(2), 공 속도(2), 직전 상위 행동(3)  — 모두 로봇 몸통 좌표
  상위 정책 행동(3): [-1,1] → (vx 0.15, vy 0.2, wz 1.0 m/s·rad/s) 속도 명령
  보상: 공이 목표에 가까워진 만큼(진행), 공 뒤쪽 '미는 위치'에 다가간 만큼, 성공 보너스, 넘어짐 감점, 시간/떨림 약한 감점
원본 Playground 코드는 수정하지 않고 상속해서 확장한다.
"""
import os
import sys

import jax
import jax.numpy as jp
import numpy as np
from ml_collections import config_dict
from mujoco import mjx
from mujoco.mjx._src import math
from mujoco_playground._src import mjx_env

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
for p in (ROOT, os.path.join(ROOT, "third_party", "Open_Duck_Playground")):
    if p not in sys.path:
        sys.path.insert(0, p)

from playground.open_duck_mini_v2 import base as duck_base  # noqa: E402
from playground.open_duck_mini_v2 import joystick  # noqa: E402
from sim.rl.low_policy import load_walk_policy, walk_policy  # noqa: E402
from sim.rl.make_xml import make as make_xml  # noqa: E402

CMD_SCALE = jp.array([0.15, 0.2, 1.0])       # 걷기 정책이 학습한 명령 범위
SUCCESS_DIST = 0.15                          # 공이 목표에서 이 거리 안이면 성공(m)
PUSH_OFFSET = 0.2                            # 공 뒤쪽 '미는 위치'까지 거리(m)
ARENA = 2.5                                  # 공이 이 거리 밖으로 나가면 실패


def baseline_action(b, g):
    """sim/rl/baseline_push.py 의 JAX 버전(벡터화 가능). b, g: 로봇 몸통 좌표의 공/목표 (2,). 반환: 정규화 행동 (3,)"""
    bg = g - b
    dist = jp.linalg.norm(bg) + 1e-6
    u = bg / dist
    rb = -b
    along, lateral = rb @ u, u[0] * rb[1] - u[1] * rb[0]
    behind_ok = (along < -0.08) & (jp.abs(lateral) < 0.5 * (-along) + 0.06) & (jp.linalg.norm(b) < 0.8)
    p = b - 0.22 * u
    t = jp.clip((b @ p) / (p @ p + 1e-9), 0.0, 1.0)
    near_path = (jp.linalg.norm(b - t * p) < 0.18) & (jp.linalg.norm(p) > 0.15)
    side = jp.where(u[0] * b[1] - u[1] * b[0] < 0, 1.0, -1.0)
    detour = b - 0.1 * u + jp.array([-u[1], u[0]]) * side * 0.32
    target = jp.where(behind_ok, b + 0.12 * u, jp.where(near_path, detour, p))
    h = jp.arctan2(target[1], target[0])
    wz = jp.clip(2.0 * h, -1.0, 1.0)
    vx = jp.where(jp.abs(h) < 0.7, jp.maximum(0.0, jp.cos(h)), 0.0)
    vx = vx * jp.where(behind_ok, jp.clip(dist / 0.5, 0.35, 1.0), 1.0)
    return jp.array([vx, 0.0, wz])


def default_config():
    cfg = joystick.default_config()
    cfg.episode_length = 250                 # 상위 스텝 수(1스텝 = 걷기 정책 K번 = 0.08s) → 20초
    cfg.push_config.enable = False           # 학습 초기엔 로봇을 밀치는 외란을 끔
    cfg.hl = config_dict.create(residual=True, res_scale=0.5, k_low=4, progress=30.0, approach=4.0, success=20.0, fall=-2.0, time=-0.01, jitter=-0.01)
    return cfg


class PushBall(joystick.Joystick):
    def __init__(self, config=None, config_overrides=None):
        config = config or default_config()
        duck_base.OpenDuckMiniV2Env.__init__(self, xml_path=make_xml(), config=config, config_overrides=config_overrides)
        # 공의 자유 관절이 'backlash 관절'로 잘못 분류되지 않게 제외
        keep = [i for i, n in enumerate(self.backlash_joint_names) if n != "ball_free"]
        self.backlash_joint_names = [self.backlash_joint_names[i] for i in keep]
        self.backlash_joint_ids = [self.backlash_joint_ids[i] for i in keep]
        self.backlash_joint_qpos_addr = [self.backlash_joint_qpos_addr[i] for i in keep]
        self._post_init()
        self._low = load_walk_policy()
        self._k_low = int(config.hl.k_low)
        jid = self._mj_model.joint("ball_free").id
        self._ball_q = int(self._mj_model.jnt_qposadr[jid])
        self._ball_v = int(self._mj_model.jnt_dofadr[jid])

    @property
    def action_size(self):
        return 3                    # 상위 정책의 행동: 속도 명령 (vx, vy, wz). 걷기 정책의 14 는 내부에서만 씀

    # ---------- 관측 ----------
    def _yaw(self, qpos):
        q = qpos[self._floating_base_qpos_addr + 3: self._floating_base_qpos_addr + 7]
        return jp.arctan2(2 * (q[0] * q[3] + q[1] * q[2]), 1 - 2 * (q[2] ** 2 + q[3] ** 2))

    def _rel(self, vec_world_xy, yaw):
        c, s = jp.cos(yaw), jp.sin(yaw)
        return jp.array([c * vec_world_xy[0] + s * vec_world_xy[1], -s * vec_world_xy[0] + c * vec_world_xy[1]])

    def _hl_obs(self, data, goal, last_act):
        b = self._floating_base_qpos_addr
        base_xy, yaw = data.qpos[b:b + 2], self._yaw(data.qpos)
        ball_xy, ball_v = data.qpos[self._ball_q:self._ball_q + 2], data.qvel[self._ball_v:self._ball_v + 2]
        o = jp.concatenate([self._rel(ball_xy - base_xy, yaw) / 1.5, self._rel(goal - base_xy, yaw) / 2.0,
                            self._rel(ball_v, yaw), last_act])
        return {"state": o, "privileged_state": o}

    def _push_pos(self, ball_xy, goal):
        d = goal - ball_xy
        return ball_xy - PUSH_OFFSET * d / (jp.linalg.norm(d) + 1e-6)

    # ---------- reset ----------
    def reset(self, rng):
        state = super().reset(rng)
        k1, k2, k3, k4 = jax.random.split(jax.random.fold_in(state.info["rng"], 7), 4)
        qpos = state.data.qpos
        b = self._floating_base_qpos_addr
        base_xy, yaw = qpos[b:b + 2], self._yaw(qpos)
        # 커리큘럼: 에피소드마다 난이도 u~U(0,1). u 가 작으면 '공이 정면 가까이 + 목표가 일직선 앞쪽'(걷기만 해도 밀림),
        # u 가 크면 공이 360° 어디서든, 목표도 어느 방향이든. 항상 쉬운 배치가 섞여 있어 탐색이 가능하다.
        ku, kb = jax.random.split(k1)
        u = jax.random.uniform(ku)
        d = jax.random.uniform(kb, minval=0.25, maxval=0.25 + 0.75 * u)
        a = jax.random.uniform(k2, minval=-1.0, maxval=1.0) * (0.4 + (jp.pi - 0.4) * u)
        ball_xy = base_xy + d * jp.array([jp.cos(yaw + a), jp.sin(yaw + a)])
        gd = jax.random.uniform(k3, minval=0.3, maxval=0.3 + 0.9 * u)
        ga = (yaw + a) + jax.random.uniform(k4, minval=-1.0, maxval=1.0) * (0.3 + (jp.pi - 0.3) * u)   # 로봇→공 방향을 기준으로 목표 방향
        goal = ball_xy + gd * jp.array([jp.cos(ga), jp.sin(ga)])
        qpos = qpos.at[self._ball_q:self._ball_q + 7].set(jp.array([ball_xy[0], ball_xy[1], 0.05, 1.0, 0.0, 0.0, 0.0]))
        data = mjx.forward(self.mjx_model, state.data.replace(qpos=qpos))
        info = dict(state.info)
        info.update(goal=goal, hl_last=jp.zeros(3), low_obs=state.obs["state"],
                    prev_ball_goal=jp.linalg.norm(goal - ball_xy), prev_push=jp.linalg.norm(self._push_pos(ball_xy, goal) - base_xy))
        metrics = dict(state.metrics)
        metrics.update(ball_goal_dist=jp.zeros(()), success=jp.zeros(()), fell=jp.zeros(()), progress=jp.zeros(()))   # step 의 지표와 키가 같아야 함(Brax)
        return state.replace(data=data, info=info, metrics=metrics, obs=self._hl_obs(data, goal, jp.zeros(3)))

    # ---------- step ----------
    def step(self, state, action):
        act = jp.clip(action, -1.0, 1.0)
        if self._config.hl.residual:       # 잔차 RL: 사람이 설계한 제어기 + 정책이 학습한 보정
            o = state.obs["state"]
            act = jp.clip(baseline_action(o[0:2] * 1.5, o[2:4] * 2.0) + self._config.hl.res_scale * act, -1.0, 1.0)
        cmd = jp.concatenate([act * CMD_SCALE, jp.zeros(4)])
        s, done_any = state, jp.zeros(())
        for _ in range(self._k_low):
            s.info["command"] = cmd
            # 걷기 정책을 '학습된 그대로' 사용(고개 포함). 고개를 고정/가리면 넘어짐이 1.8% → 7~30% 로 늘어남(측정)
            low_a = walk_policy(self._low, s.info["low_obs"])
            s = joystick.Joystick.step(self, s, low_a)
            s.info["low_obs"] = s.obs["state"]
            done_any = jp.maximum(done_any, s.done)
        data, info = s.data, s.info
        b = self._floating_base_qpos_addr
        base_xy = data.qpos[b:b + 2]
        ball_xy = data.qpos[self._ball_q:self._ball_q + 2]
        goal = info["goal"]
        bg = jp.linalg.norm(goal - ball_xy)
        push = jp.linalg.norm(self._push_pos(ball_xy, goal) - base_xy)
        w = self._config.hl
        success = bg < SUCCESS_DIST
        out = jp.linalg.norm(ball_xy) > ARENA
        reward = (w.progress * (info["prev_ball_goal"] - bg)
                  + w.approach * jp.clip(info["prev_push"] - push, -0.2, 0.2) * (bg > SUCCESS_DIST)
                  + w.success * success + w.fall * (done_any > 0) + w.time
                  + w.jitter * jp.sum((act - info["hl_last"]) ** 2))
        prog = info["prev_ball_goal"] - bg
        info["prev_ball_goal"], info["prev_push"], info["hl_last"] = bg, push, act
        done = jp.maximum(done_any, jp.maximum(success.astype(jp.float32), out.astype(jp.float32)))
        s.metrics["ball_goal_dist"] = bg
        s.metrics["success"] = success.astype(jp.float32)
        s.metrics["fell"] = done_any
        s.metrics["progress"] = prog                                    # 에피소드 합 = 공이 목표에 가까워진 총량(m)
        return s.replace(obs=self._hl_obs(data, goal, act), reward=reward, done=done)
