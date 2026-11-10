"""공 밀기 상위 정책을 '일반 MuJoCo(CPU) 시뮬레이션'에서 평가한다. (학습은 MJX 였으므로 엔진이 달라도 통하는지 확인)
정답 위치(공/목표)를 관측으로 직접 넣는다 — 인식 오차를 빼고 정책 자체의 성능을 보기 위함.
사용: .venv-ros/bin/python sim/rl/eval_push_offline.py <params.pkl> [에피소드 수]
"""
import math
import os
import sys

import mujoco
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)
from sim.headless import HeadlessDuck  # noqa: E402
from sim.rl.hl_policy import HLPolicy, make_obs  # noqa: E402

STATS = []
NOISY = os.environ.get("PUSH_NOISY", "0") == "1"     # 1: 정책이 정답 대신 '추정값'(카메라 오차+가림+이동량 예측)을 본다 = ROS 와 같은 조건
SUCCESS, K_LOW, MAX_HL = float(os.environ.get("PUSH_SUCCESS", 0.15)), 4, 250


def track(est, vel, true_rel, cmd_xy, dyaw, rng, dt=0.08, fwd_gain=0.7, sigma0=0.03, sigma1=0.08, drop=0.1, vmin=0.45, vmax=3.0, fov=0.7):
    """학습 환경(push_ball_env._track)·ROS 노드(Tracker)와 같은 수식의 numpy 판"""
    p = est - cmd_xy * fwd_gain * dt
    c, sn = math.cos(-dyaw), math.sin(-dyaw)
    pred = np.array([c * p[0] - sn * p[1], sn * p[0] + c * p[1]])
    d = float(np.linalg.norm(true_rel))
    visible = (vmin < d < vmax) and abs(math.atan2(true_rel[1], true_rel[0])) < fov and rng.random() > drop
    meas = true_rel + rng.normal(size=2) * (sigma0 + sigma1 * d)
    if visible:
        out = np.clip(0.5 * pred + 0.5 * meas, -3, 3), np.clip(0.6 * vel + 0.4 * (meas - pred) / dt, -1, 1)
    else:
        out = np.clip(pred, -3, 3), np.zeros(2)
    STATS.append((d, float(np.linalg.norm(out[0] - true_rel)), bool(visible)))     # 추정 오차 통계(ROS 와 비교용)
    return out


def episode(policy, rng, difficulty=None):
    d = HeadlessDuck(dynamic_objects=True)
    d.native_head = True
    m, data = d.model, d.data
    jid = m.joint("blue_ball_free").id
    bq, bv = m.jnt_qposadr[jid], m.jnt_dofadr[jid]
    # 다른 물체는 멀리 치움
    for n in ("red_block_free", "green_block_free"):
        j = m.joint(n).id; data.qpos[m.jnt_qposadr[j]:m.jnt_qposadr[j] + 3] = [5, 5, 0.05]
    # 학습과 같은 배치: 로봇 정면 기준 무작위 방향 0.3~1.0m 에 공, 공에서 0.5~1.2m 무작위 방향에 목표
    yaw0 = rng.uniform(-math.pi, math.pi)
    q = np.zeros(4); mujoco.mju_euler2Quat(q, np.array([0, 0, yaw0]), "xyz"); data.qpos[3:7] = q
    # 학습과 같은 난이도 분포: u 가 작으면 공이 정면 가까이 + 목표가 일직선 앞쪽, 크면 360° 어디서든
    u = rng.uniform(0, 1) if difficulty is None else difficulty
    dist = rng.uniform(0.25, 0.25 + 0.75 * u); a = rng.uniform(-1, 1) * (0.4 + (math.pi - 0.4) * u)
    ball = np.array([dist * math.cos(yaw0 + a), dist * math.sin(yaw0 + a)])
    data.qpos[bq:bq + 3] = [ball[0], ball[1], 0.05]
    gd = rng.uniform(0.3, 0.3 + 0.9 * u); ga = (yaw0 + a) + rng.uniform(-1, 1) * (0.3 + (math.pi - 0.3) * u)
    goal = ball + gd * np.array([math.cos(ga), math.sin(ga)])
    mujoco.mj_forward(m, data)
    last = np.zeros(3); fell = False; best = 9.0
    if NOISY:                                              # 처음엔 탐색으로 둘 다 본 상태(측정 오차만)
        qw, qx, qy, qz = data.qpos[3:7]; y0 = math.atan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz)); c0, s0 = math.cos(y0), math.sin(y0)
        rot0 = lambda v: np.array([c0 * v[0] + s0 * v[1], -s0 * v[0] + c0 * v[1]])
        b_est = rot0(ball - data.qpos[0:2]); g_est = rot0(goal - data.qpos[0:2])
        b_est = b_est + rng.normal(size=2) * (0.03 + 0.08 * np.linalg.norm(b_est)); g_est = g_est + rng.normal(size=2) * (0.03 + 0.08 * np.linalg.norm(g_est))
        v_est, prev_yaw = np.zeros(2), y0
    for t in range(MAX_HL):
        base = data.qpos[0:2].copy(); qw, qx, qy, qz = data.qpos[3:7]
        yaw = math.atan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz)); c, s = math.cos(yaw), math.sin(yaw)
        rot = lambda v: np.array([c * v[0] + s * v[1], -s * v[0] + c * v[1]])
        ball = data.qpos[bq:bq + 2].copy()
        obs = make_obs(b_est, g_est, v_est, last) if NOISY else make_obs(rot(ball - base), rot(goal - base), rot(data.qvel[bv:bv + 2]), last)
        cmd, last = policy.command(obs)
        for _ in range(K_LOW):
            d.step(vx=cmd[0], vy=cmd[1], wz=cmd[2])
        if NOISY:                                          # 추정 갱신 (정답은 '카메라가 본 것'을 만드는 데만 사용)
            qw, qx, qy, qz = data.qpos[3:7]; yaw1 = math.atan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))
            c1, s1 = math.cos(yaw1), math.sin(yaw1); rot1 = lambda v: np.array([c1 * v[0] + s1 * v[1], -s1 * v[0] + c1 * v[1]])
            dyaw = (yaw1 - prev_yaw + math.pi) % (2 * math.pi) - math.pi; prev_yaw = yaw1
            b1 = data.qpos[0:2]
            b_est, v_est = track(b_est, v_est, rot1(data.qpos[bq:bq + 2] - b1), cmd[0:2], dyaw, rng)
            g_est, _ = track(g_est, np.zeros(2), rot1(goal - b1), cmd[0:2], dyaw, rng)
        if data.xmat[d.base_id].reshape(3, 3)[2, 2] < 0.3:      # 몸통이 많이 기울면 넘어짐
            fell = True; break
        best = min(best, float(np.linalg.norm(data.qpos[bq:bq + 2] - goal)))
        if np.linalg.norm(data.qpos[bq:bq + 2] - goal) < SUCCESS:
            return True, False, t + 1, float(np.linalg.norm(data.qpos[bq:bq + 2] - goal)), u, best
        if np.linalg.norm(data.qpos[bq:bq + 2]) > 2.5: break
    return False, fell, t + 1, float(np.linalg.norm(data.qpos[bq:bq + 2] - goal)), u, best


if __name__ == "__main__":
    if sys.argv[1] == "baseline":
        from sim.rl.baseline_push import BaselinePush
        policy = BaselinePush()
    else:
        policy = HLPolicy(sys.argv[1], residual=os.environ.get("PUSH_RESIDUAL", "1") == "1")
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 20
    rng = np.random.default_rng(0); res = []
    for i in range(n):
        ok, fell, steps, dist, u, best = episode(policy, rng); res.append((ok, fell, steps, dist, u, best))
        print(f"  에피소드 {i + 1:2d} (난이도 {u:.2f}): {'성공' if ok else ('넘어짐' if fell else '실패')}  {steps:3d} 스텝({steps * 0.08:4.1f}s)  마지막 공-목표 거리 {dist:.2f}m, 최소 {best:.2f}m", flush=True)
    if STATS:
        A = np.array(STATS, dtype=float); v = A[:, 2] > 0.5
        for lab, m in (("보일 때", v), ("안 보일 때(예측만)", ~v)):
            if m.sum() > 3: print(f"  [오프라인 모델의 추정 오차] {lab:14s} {m.sum():5d}개 ({100 * m.mean():.0f}%) 오차 중앙값 {np.median(A[m, 1]):.2f} m, 90% {np.percentile(A[m, 1], 90):.2f} m, 평균 거리 {A[m, 0].mean():.2f} m")
    ok = sum(r[0] for r in res); fl = sum(r[1] for r in res)
    print(f"\n전체 성공 {ok}/{n} ({100 * ok / n:.0f}%), 넘어짐 {fl}, 성공 시 평균 {np.mean([r[2] for r in res if r[0]] or [0]) * 0.08:.1f}초")
    bests = np.array([r[5] for r in res]); print("  에피소드 중 공이 목표에 가장 가까웠던 거리: 중앙값 %.2f m, <0.15m %d회, <0.25m %d회, <0.35m %d회 (/%d)" % (np.median(bests), (bests < 0.15).sum(), (bests < 0.25).sum(), (bests < 0.35).sum(), n))
    for lo, hi, name in ((0, 0.34, "쉬움(u<0.34)"), (0.34, 0.67, "중간"), (0.67, 1.01, "어려움(u>0.67)")):
        g = [r for r in res if lo <= r[4] < hi]
        if g: print(f"  {name:14s} 성공 {sum(r[0] for r in g)}/{len(g)}")
