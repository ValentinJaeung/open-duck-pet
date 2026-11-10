"""학습된 상위 정책(Brax PPO 가 저장한 pkl)을 numpy 로 실행. ROS(Python 3.10) 환경에서 JAX 없이 쓴다.
구조: (obs - mean)/std → [Dense → swish] x3 → Dense(6) → 앞 3개 tanh = 속도 명령 정규화 값 (결정론적)
"""
import pickle

import numpy as np

CMD_SCALE = np.array([0.15, 0.2, 1.0])        # push_ball_env.CMD_SCALE 과 같아야 함


class HLPolicy:
    def __init__(self, path, residual=False):
        """residual=True: 사람이 설계한 제어기(baseline_push) + res_scale * 정책 (잔차 RL 로 학습한 경우)
        path: export_hl.py 가 만든 .npz (순수 numpy. ROS 환경에서 사용). 학습이 저장한 .pkl 은 Brax 가 있어야 읽힘"""
        self.residual, self.res_scale, self._base = residual, 0.5, None
        if path.endswith(".pkl"):
            stats, pol, _ = pickle.load(open(path, "rb"))
            self.mean, self.std = np.asarray(stats.mean["state"]), np.asarray(stats.std["state"])
            p = pol["params"]
            self.W = [np.asarray(p[f"hidden_{i}"]["kernel"]) for i in range(4)]
            self.b = [np.asarray(p[f"hidden_{i}"]["bias"]) for i in range(4)]
        else:
            z = np.load(path)
            self.mean, self.std = z["mean"], z["std"]
            self.W, self.b = [z[f"W{i}"] for i in range(4)], [z[f"b{i}"] for i in range(4)]

    def __call__(self, obs):
        x = (np.asarray(obs, dtype=np.float64) - self.mean) / self.std
        for i in range(3):
            x = x @ self.W[i] + self.b[i]
            x = x * (1.0 / (1.0 + np.exp(-x)))      # swish
        x = x @ self.W[3] + self.b[3]
        return np.tanh(x[:3])

    def command(self, obs):
        a = self(obs)
        if self.residual:
            if self._base is None:
                from sim.rl.baseline_push import BaselinePush
                self._base = BaselinePush()
            a = np.clip(self._base.action(obs) + self.res_scale * a, -1.0, 1.0)
        return a * CMD_SCALE, a


def make_obs(ball_rel, goal_rel, ball_vel_rel, last_act):
    """학습 환경(push_ball_env._hl_obs)과 같은 9차원 관측. 모두 로봇 몸통 좌표(x 앞, y 왼쪽)"""
    return np.concatenate([np.asarray(ball_rel) / 1.5, np.asarray(goal_rel) / 2.0, np.asarray(ball_vel_rel), np.asarray(last_act)])
