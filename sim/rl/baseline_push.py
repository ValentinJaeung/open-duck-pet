"""사람이 설계한 공 밀기 제어기 (RL 이 아님). 상위 정책과 같은 입출력(관측 9 → 행동 3)을 가진다.
전략: 로봇이 '공 뒤쪽(목표 반대편)' 원뿔 안에 있으면 공을 목표 방향으로 걷어 밀고,
      아니면 공 뒤쪽 '미는 위치'로 돌아간다(공이 가는 길을 막으면 옆으로 돌아서). 잔차 RL 의 기준선으로도 쓴다.
"""
import math

import numpy as np

CMD_SCALE = np.array([0.15, 0.2, 1.0])


def _go_to(target, vmax=1.0):
    """로봇 좌표 target 쪽으로: 방향 오차가 크면 제자리 회전, 작으면 전진하며 회전. 반환: 정규화 행동 [-1,1]x3"""
    h = math.atan2(target[1], target[0])
    wz = float(np.clip(2.0 * h, -1.0, 1.0))
    vx = vmax * max(0.0, math.cos(h)) if abs(h) < 0.7 else 0.0
    return np.array([vx, 0.0, wz])


class BaselinePush:
    def __init__(self, behind=0.22, cone=0.5, beyond=0.12, detour=0.32):
        self.behind, self.cone, self.beyond, self.detour = behind, cone, beyond, detour
        self.mode = "approach"

    def action(self, obs):
        b, g = np.asarray(obs[0:2]) * 1.5, np.asarray(obs[2:4]) * 2.0      # 몸통 좌표의 공/목표
        bg = g - b
        dist = np.linalg.norm(bg)
        if dist < 1e-3:
            return np.zeros(3)
        u = bg / dist                                                       # 공을 보낼 방향
        rb = -b                                                             # 공에서 본 로봇 위치
        along, lateral = float(rb @ u), float(u[0] * rb[1] - u[1] * rb[0])  # 밀기 방향 성분(음수=공 뒤쪽), 옆 편차
        behind_ok = along < -0.08 and abs(lateral) < self.cone * (-along) + 0.06 and np.linalg.norm(b) < 0.8
        if behind_ok:
            self.mode = "push"
            a = _go_to(b + self.beyond * u)                                 # 공 중심을 지나 목표 방향으로 걸음
            a[0] *= float(np.clip(dist / 0.5, 0.35, 1.0))                   # 목표 가까이에서는 살살(공이 지나쳐 굴러가는 것 방지)
            return a
        self.mode = "approach"
        p = b - self.behind * u                                             # 미는 위치(공 뒤쪽)
        # 로봇→p 경로가 공 가까이를 지나가면(공을 엉뚱한 쪽으로 칠 위험) 공 옆으로 우회
        seg = p
        t = float(np.clip((b @ seg) / (seg @ seg + 1e-9), 0, 1))
        if np.linalg.norm(b - t * seg) < 0.18 and np.linalg.norm(p) > 0.15:
            side = 1.0 if (u[0] * b[1] - u[1] * b[0]) < 0 else -1.0       # 로봇이 있는 쪽으로 돈다
            perp = np.array([-u[1], u[0]]) * side
            p = b - 0.1 * u + perp * self.detour
        return _go_to(p)

    # HLPolicy 와 같은 인터페이스 (eval_push_offline 에서 교체해서 쓰기 위함)
    def command(self, obs):
        a = self.action(obs)
        return a * CMD_SCALE, a
