"""공 밀기 상위 정책 학습 (Brax PPO). 사용: bash scripts/train_push_ball.sh [상위 스텝 수]"""
import functools
import os
import pickle
import sys
import time

import jax
import numpy as np
from brax.training.agents.ppo import networks as ppo_networks
from brax.training.agents.ppo import train as ppo
from mujoco_playground import wrapper

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "third_party", "Open_Duck_Playground"))
from sim.rl.push_ball_env import PushBall  # noqa: E402

steps = int(float(sys.argv[1])) if len(sys.argv) > 1 else 20_000_000
out = sys.argv[2] if len(sys.argv) > 2 else os.path.join(ROOT, "models", "push", time.strftime("%Y%m%d_%H%M%S"))
os.makedirs(out, exist_ok=True)
env, eval_env = PushBall(), PushBall()
t0 = time.time()


def progress(step, metrics):
    keys = ("eval/episode_reward", "eval/episode_success", "eval/episode_progress", "eval/episode_fell", "eval/avg_episode_length")
    row = " ".join(f"{k.split('/')[-1]}={float(metrics[k]):.3f}" for k in keys if k in metrics)
    print(f"[{time.time() - t0:6.0f}s] 스텝 {step:>10,}  {row}", flush=True)


def save(step, make_policy, params, *a):
    with open(os.path.join(out, f"params_{step}.pkl"), "wb") as f:
        pickle.dump(jax.tree_util.tree_map(np.asarray, params), f)


train = functools.partial(
    ppo.train, num_timesteps=steps, num_evals=20, episode_length=250, num_envs=2048, batch_size=512,
    num_minibatches=4, unroll_length=10, num_updates_per_batch=4, discounting=0.98, learning_rate=3e-4,
    entropy_cost=1e-3, reward_scaling=1.0, normalize_observations=True, action_repeat=1, seed=0,
    network_factory=functools.partial(ppo_networks.make_ppo_networks, policy_hidden_layer_sizes=(128, 128, 64),
                                      value_hidden_layer_sizes=(256, 256, 128), policy_obs_key="state", value_obs_key="privileged_state"),
    progress_fn=progress, policy_params_fn=save)
print(f"학습 시작: {steps:,} 상위 스텝, 출력 {out}", flush=True)
_, params, _ = train(environment=env, eval_env=eval_env, wrap_env_fn=wrapper.wrap_for_brax_training)
with open(os.path.join(out, "params_final.pkl"), "wb") as f:
    pickle.dump(jax.tree_util.tree_map(np.asarray, params), f)
import subprocess
subprocess.run([sys.executable, os.path.join(ROOT, "sim", "rl", "export_hl.py"), os.path.join(out, "params_final.pkl")])
print("완료", flush=True)
