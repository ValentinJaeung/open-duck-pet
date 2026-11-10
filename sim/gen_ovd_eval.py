"""개방형 탐지 평가용 이미지 생성: 학습에 없던 새 물체(노란 컵, 주황 공, 보라색 상자) + 기존 물체 + 사람.
정답 상자는 분할 렌더링에서 자동 계산. 사용: MUJOCO_GL=glfw .venv-vision/bin/python sim/gen_ovd_eval.py [장수]
"""
import json, math, os, sys
import numpy as np
import mujoco
from PIL import Image
from scipy import ndimage

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
from sim.build_scene import build, ROOM, FURNITURE, OBJECTS, NOVEL_OBJECTS, PERSON_PARTS  # noqa: E402

N = int(sys.argv[1]) if len(sys.argv) > 1 else 250
OUT = os.path.join(ROOT, "datasets", "ovd_eval")
W, H = 320, 240
PROMPT = {"red_block": "red block", "green_block": "green block", "blue_ball": "blue ball", "yellow_cup": "yellow cup",
          "orange_ball": "orange ball", "purple_box": "purple box", "person": "person"}
rng = np.random.default_rng(1)
model = build(dynamic_objects=False, novel=True)
data = mujoco.MjData(model)
rgb_r, seg_r = mujoco.Renderer(model, H, W), mujoco.Renderer(model, H, W)
seg_r.enable_segmentation_rendering()
known = [o[0] for o in OBJECTS]; novel = [o[0] for o in NOVEL_OBJECTS]
body = {n: model.body(n).id for n in known + novel}
geom = {n: model.geom(n).id for n in known + novel}
person_geoms = [model.geom(p[0]).id for p in PERSON_PARTS] + [model.geom("p_head").id]
half = {n: (0.0 if n in novel else model.geom_size[geom[n]][0]) for n in known + novel}      # 새 물체는 몸체 원점이 바닥
base_light = (model.vis.headlight.diffuse.copy(), model.vis.headlight.ambient.copy())
q_addr = {n: model.jnt_qposadr[model.joint(n).id] for n in ("neck_pitch", "head_pitch", "head_yaw", "head_roll")}
rects = [(x - sx - 0.1, x + sx + 0.1, y - sy - 0.1, y + sy + 0.1) for _, x, y, sx, sy, _, _ in FURNITURE]
X, Y = ROOM["x"] / 2 - 0.15, ROOM["y"] / 2 - 0.15
HFOV = 2 * math.atan(math.tan(math.radians(35)) * W / H)
free = lambda x, y: not any(a <= x <= b and c <= y <= d for a, b, c, d in rects)


def rand_free(m=0.15):
    for _ in range(100):
        x, y = rng.uniform(-(X + 0.15 - m), X + 0.15 - m), rng.uniform(-(Y + 0.15 - m), Y + 0.15 - m)
        if free(x, y): return x, y
    return 0.0, 0.0


def sample():
    model.vis.headlight.diffuse[:] = base_light[0] * rng.uniform(0.6, 1.0)
    placed, targets = [], []
    for n in known + novel:                                       # 모든 물체를 무작위 위치에(서로 25cm 이상 떨어지게)
        for _ in range(50):
            x, y = rand_free()
            if all(math.hypot(x - a, y - b) > 0.25 for a, b in placed): break
        placed.append((x, y)); targets.append((x, y))
        if n in body: model.body_pos[body[n]] = [x, y, half[n]]
        else: model.geom_pos[geom[n]][:] = [x, y, half[n]]
    px, py = rand_free(0.3); placed.append((px, py)); targets.append((px, py))
    hy = rng.uniform(-0.8, 0.8)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    pyaw = rng.uniform(-math.pi, math.pi)
    data.mocap_pos[0], data.mocap_quat[0] = [px, py, 0.0], [math.cos(pyaw / 2), 0, 0, math.sin(pyaw / 2)]
    data.qpos[q_addr["head_yaw"]] = hy; data.qpos[q_addr["head_pitch"]] = rng.uniform(-0.2, 0.3)
    tx, ty = targets[rng.integers(len(targets))]
    for _ in range(100):
        ang, d = rng.uniform(-math.pi, math.pi), rng.uniform(0.5, 2.0)
        dx, dy = tx + d * math.cos(ang), ty + d * math.sin(ang)
        if abs(dx) < X - 0.15 and abs(dy) < Y - 0.15 and free(dx, dy): break
    dyaw = math.atan2(ty - dy, tx - dx) - hy + rng.uniform(-0.3, 0.3) * HFOV
    data.qpos[0], data.qpos[1] = dx, dy
    q = np.zeros(4); mujoco.mju_euler2Quat(q, np.array([0, 0, dyaw]), "xyz"); data.qpos[3:7] = q
    mujoco.mj_forward(model, data)


os.makedirs(f"{OUT}/images", exist_ok=True)
labels = {}
counts = {k: 0 for k in PROMPT}
for i in range(N):
    sample()
    rgb_r.update_scene(data, camera="head_cam"); img = rgb_r.render().copy()
    seg_r.update_scene(data, camera="head_cam"); seg = seg_r.render(); ids, types = seg[..., 0], seg[..., 1]
    objs = []
    for n in PROMPT:
        gs = person_geoms if n == "person" else [g for g in range(model.ngeom) if model.geom_bodyid[g] == body[n]] if n in novel else [geom[n]]
        mask = (types == mujoco.mjtObj.mjOBJ_GEOM) & np.isin(ids, gs)
        lab, k = ndimage.label(mask)
        if k > 1: mask = lab == (1 + int(np.argmax(ndimage.sum(mask, lab, range(1, k + 1)))))
        ys, xs = np.nonzero(mask)
        if len(xs) >= (40 if n == "person" else 20):                # 너무 작거나 거의 가려진 것은 정답에서 뺌
            objs.append({"name": n, "prompt": PROMPT[n], "box": [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]}); counts[n] += 1
    Image.fromarray(img).save(f"{OUT}/images/{i:04d}.jpg", quality=95)
    labels[f"{i:04d}.jpg"] = objs
json.dump({"prompts": PROMPT, "labels": labels}, open(f"{OUT}/labels.json", "w"), ensure_ascii=False)
print("완료", N, "장, 물체별 등장 횟수", counts)
