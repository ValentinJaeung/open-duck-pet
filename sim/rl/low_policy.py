"""학습된 걷기 정책(ONNX)을 JAX 로 복원한 고정 저수준 정책. 계층형 RL(공 밀기 등)에서 상위 정책 아래에 그대로 쓴다.
구조(ONNX 그래프 확인): (obs - mean) * (1/std) → [Dense → swish] x3 → Dense → 앞 14개(평균) → tanh
"""
import os
import jax
import jax.numpy as jp
import numpy as np
import onnx
from onnx import numpy_helper

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DEFAULT_ONNX = os.path.join(_ROOT, "models/runs/20261108_113921_flat_terrain_backlash/2026_11_08_130526_300482560.onnx")


def load_walk_policy(path=DEFAULT_ONNX):
    m = onnx.load(path)
    w = {t.name: numpy_helper.to_array(t) for t in m.graph.initializer}
    get = lambda key: next(v for k, v in w.items() if key in k)
    p = {"mean": get("sub/ReadVariableOp"), "inv_std": get("truediv_recip")}
    for i in range(4):
        p[f"W{i}"] = get(f"hidden_{i}_1/Cast")
        p[f"b{i}"] = get(f"hidden_{i}_1/BiasAdd")
    return {k: jp.asarray(v) for k, v in p.items()}


def walk_policy(params, obs):
    """obs: (..., 101) → 행동 (..., 14) (-1~1). 결정론적."""
    x = (obs - params["mean"]) * params["inv_std"]
    for i in range(3):
        x = x @ params[f"W{i}"] + params[f"b{i}"]
        x = x * jax.nn.sigmoid(x)
    x = x @ params["W3"] + params["b3"]
    return jp.tanh(x[..., :14])
