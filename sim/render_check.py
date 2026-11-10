"""머리 카메라 + 위에서 본 전체 장면을 PNG 로 저장해 카메라 방향을 확인합니다.
사용: MUJOCO_GL=egl python sim/render_check.py [출력폴더] [pos x y z] [quat w x y z]
"""
import os, sys
import numpy as np, mujoco
from PIL import Image
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from sim.build_scene import build

out = sys.argv[1] if len(sys.argv) > 1 else "."
cam = {}
if len(sys.argv) >= 5: cam["pos"] = tuple(map(float, sys.argv[2:5]))
if len(sys.argv) >= 9: cam["quat"] = tuple(map(float, sys.argv[5:9]))
m = build(cam); d = mujoco.MjData(m)
mujoco.mj_resetDataKeyframe(m, d, 0); mujoco.mj_forward(m, d)
r = mujoco.Renderer(m, 480, 640)
r.update_scene(d, camera="head_cam"); Image.fromarray(r.render()).save(f"{out}/head_cam.png")
free = mujoco.MjvCamera(); free.lookat[:] = [0, 0, 0.2]; free.distance = 4.5; free.azimuth = 200; free.elevation = -45
r.update_scene(d, camera=free); Image.fromarray(r.render()).save(f"{out}/overview.png")
print("saved", out)
