"""학습이 저장한 params_*.pkl (Brax 필요) → 순수 numpy .npz (ROS 환경에서 사용).
사용: .venv/bin/python sim/rl/export_hl.py models/push/<폴더>/params_final.pkl   → 같은 이름의 .npz 생성
"""
import pickle
import sys

import numpy as np

src = sys.argv[1]
stats, pol, _ = pickle.load(open(src, "rb"))
p = pol["params"]
d = {"mean": np.asarray(stats.mean["state"]), "std": np.asarray(stats.std["state"])}
for i in range(4):
    d[f"W{i}"], d[f"b{i}"] = np.asarray(p[f"hidden_{i}"]["kernel"]), np.asarray(p[f"hidden_{i}"]["bias"])
out = src.rsplit(".", 1)[0] + ".npz"
np.savez(out, **d)
print("저장:", out, {k: v.shape for k, v in d.items()})
