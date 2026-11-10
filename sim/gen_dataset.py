"""합성 학습 데이터 생성: 방/물체/오리 위치, 조명, 색, 고개 각도를 무작위로 바꿔 머리 카메라 이미지를 렌더링하고
YOLO 라벨(물체 상자)을 자동으로 만든다. 라벨은 MuJoCo 분할(segmentation) 렌더링에서 얻으므로 정확하다.
(시뮬레이터 정답 정보는 '학습 데이터를 만들 때만' 쓴다. 로드맵의 설계 원칙)

사용: MUJOCO_GL=glfw .venv-vision/bin/python sim/gen_dataset.py [train 개수] [val 개수]
"""
import os, sys, math
import numpy as np
import mujoco
from PIL import Image
from scipy import ndimage

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, ROOT)
from sim.build_scene import build, ROOM, FURNITURE, OBJECTS, PERSON_PARTS  # noqa: E402

OBJ_CLASSES = [o[0] for o in OBJECTS]        # red_block, green_block, blue_ball
CLASSES = OBJ_CLASSES + ["person"]            # 3 = person (색이 아니라 모양으로 구분)
W, H = 320, 240                              # 브리지 카메라와 같은 해상도
N_DISTRACT = 4
OUT = os.path.join(ROOT, "datasets", "duck_objects")
rng = np.random.default_rng(0)

model = build(dynamic_objects=False, n_distractors=N_DISTRACT)
data = mujoco.MjData(model)
rgb_r = mujoco.Renderer(model, H, W)
seg_r = mujoco.Renderer(model, H, W)
seg_r.enable_segmentation_rendering()

gid = lambda n: model.geom(n).id
bid = lambda n: model.body(n).id
obj_geom = {n: gid(n) for n in OBJ_CLASSES}
person_geoms = [gid(p[0]) for p in PERSON_PARTS] + [gid("p_head")]
person_base_rgba = {g: model.geom_rgba[g].copy() for g in person_geoms}
obj_body = {n: bid(n) for n in CLASSES}
dis_body = [bid(f"distractor_{i}") for i in range(N_DISTRACT)]
dis_geom = [gid(f"distractor_{i}") for i in range(N_DISTRACT)]
base_rgba = model.geom_rgba.copy()
base_light = (model.vis.headlight.diffuse.copy(), model.vis.headlight.ambient.copy())
q_addr = {n: model.jnt_qposadr[model.joint(n).id] for n in ("neck_pitch", "head_pitch", "head_yaw", "head_roll")}
bg_geoms = [g for g in range(model.ngeom) if model.geom(g).name in
            ("floor", "wall_n", "wall_s", "wall_e", "wall_w", "table", "bed")]
FOVY = math.radians(70)
HFOV = 2 * math.atan(math.tan(FOVY / 2) * W / H)

# 가구(통과 불가 영역) 사각형
rects = [(x - sx - 0.1, x + sx + 0.1, y - sy - 0.1, y + sy + 0.1) for _, x, y, sx, sy, _, _ in FURNITURE]
def blocked(x, y): return any(a <= x <= b and c <= y <= d for a, b, c, d in rects)
X, Y = ROOM["x"] / 2 - 0.15, ROOM["y"] / 2 - 0.15


def rand_free(margin=0.15):
    for _ in range(100):
        x, y = rng.uniform(-(X + 0.15 - margin), X + 0.15 - margin), rng.uniform(-(Y + 0.15 - margin), Y + 0.15 - margin)
        if not blocked(x, y):
            return x, y
    return 0.0, 0.0


def jitter(rgba, amt):
    c = np.clip(np.array(rgba[:3]) * rng.uniform(1 - amt, 1 + amt) + rng.uniform(-amt, amt, 3) * 0.5, 0, 1)
    return [*c, 1.0]


def randomize():
    # 배경 색, 조명 (너무 밝으면 하얗게 날아가므로 낮은 범위)
    for g in bg_geoms:
        model.geom_rgba[g] = jitter(base_rgba[g], 0.35)
    model.vis.headlight.diffuse[:] = base_light[0] * rng.uniform(0.4, 1.0)
    model.vis.headlight.ambient[:] = np.clip(base_light[1] * rng.uniform(0.4, 1.2), 0, 1)
    # 1) 물체를 먼저 놓기 (서로 0.2m 이상 떨어지게)
    placed = []
    def place(body_id, geom_id, half, color):
        for _ in range(50):
            x, y = rand_free()
            if all(math.hypot(x - a, y - b) > 0.2 for a, b in placed): break
        placed.append((x, y)); model.body_pos[body_id] = [x, y, half]; model.geom_rgba[geom_id] = color
    targets = []
    for n, shape, _, _, s, rgba in OBJECTS:
        s2 = s * rng.uniform(0.8, 1.3)
        model.geom_size[obj_geom[n]][:3] = [s2, s2, s2] if shape == "box" else [s2, 0, 0]
        place(obj_body[n], obj_geom[n], s2, jitter(rgba, 0.15)); targets.append(placed[-1])
    # 사람: 70% 는 방 안 어딘가에 서 있음. 옷 색/방향 무작위
    has_person = rng.random() < 0.7
    if has_person:
        px, py = rand_free(0.3)
        for _ in range(30):
            if all(math.hypot(px - a, py - b) > 0.3 for a, b in placed): break
            px, py = rand_free(0.3)
        placed.append((px, py)); targets.append((px, py))
        pyaw = rng.uniform(-math.pi, math.pi)
        person_pose = ([px, py, 0.0], [math.cos(pyaw / 2), 0, 0, math.sin(pyaw / 2)])
        shirt = [*rng.uniform(0.05, 1.0, 3), 1.0]; pants = [*rng.uniform(0.05, 0.8, 3), 1.0]; skin = [*jitter([0.9, 0.75, 0.6], 0.2)[:3], 1.0]
        for (n, *_), g in zip(PERSON_PARTS, person_geoms):
            model.geom_rgba[g] = pants if n.startswith("p_leg") else shirt
        model.geom_rgba[person_geoms[-1]] = skin
    else:
        person_pose = ([0, 0, -5], [1, 0, 0, 0])
    for i in range(N_DISTRACT):
        if rng.random() < 0.7:
            s2 = rng.uniform(0.03, 0.08); model.geom_size[dis_geom[i]][:3] = [s2] * 3
            col = rng.choice([[0.9, 0.85, 0.1], [0.6, 0.2, 0.7], [0.5, 0.5, 0.5], [0.95, 0.5, 0.1], [0.1, 0.7, 0.7]])
            place(dis_body[i], dis_geom[i], s2, [*col, 1])
        else:
            model.body_pos[dis_body[i]] = [0, 0, -5]
    # 2) 고개 각도, 3) 오리 위치/방향: 75% 는 물체 하나를 시야 안에 두고, 나머지는 무작위(물체 없는 장면 포함)
    hy = rng.uniform(-1.0, 1.0)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    data.mocap_pos[0], data.mocap_quat[0] = person_pose      # 키프레임 리셋이 mocap 도 초기화하므로 그 뒤에 설정
    data.qpos[q_addr["head_yaw"]] = hy
    data.qpos[q_addr["head_pitch"]] = rng.uniform(-0.3, 0.4)
    data.qpos[q_addr["neck_pitch"]] = rng.uniform(-0.1, 0.4)
    data.qpos[q_addr["head_roll"]] = rng.normal(0, 0.1)
    if rng.random() < 0.75:
        tx, ty = targets[rng.integers(len(targets))]
        for _ in range(100):
            ang, d = rng.uniform(-math.pi, math.pi), rng.uniform(0.4, 2.2)
            dx, dy = tx + d * math.cos(ang), ty + d * math.sin(ang)
            if abs(dx) < X - 0.15 and abs(dy) < Y - 0.15 and not blocked(dx, dy): break
        bearing = math.atan2(ty - dy, tx - dx)
        dyaw = bearing - hy + rng.uniform(-0.4, 0.4) * HFOV
    else:
        dx, dy = rand_free(0.4); dyaw = rng.uniform(-math.pi, math.pi)
    data.qpos[0], data.qpos[1] = dx, dy
    r, p = rng.normal(0, 0.03, 2)                           # 걷는 중 몸통 흔들림
    q = np.zeros(4); mujoco.mju_euler2Quat(q, np.array([r, p, dyaw]), "xyz"); data.qpos[3:7] = q
    mujoco.mj_forward(model, data)


def make_sample():
    randomize()
    rgb_r.update_scene(data, camera="head_cam"); img = rgb_r.render().copy()
    seg_r.update_scene(data, camera="head_cam"); seg = seg_r.render()
    ids, types = seg[..., 0], seg[..., 1]
    lines = []
    for ci, n in enumerate(CLASSES):
        if n == "person":
            mask = (types == mujoco.mjtObj.mjOBJ_GEOM) & np.isin(ids, person_geoms)
        else:
            mask = (types == mujoco.mjtObj.mjOBJ_GEOM) & (ids == obj_geom[n])
        lab, nlab = ndimage.label(mask)           # 분할 렌더링에 같은 id 의 흩어진 픽셀이 섞여 나옴 → 가장 큰 덩어리만 사용
        if nlab > 1:
            mask = lab == (1 + int(np.argmax(ndimage.sum(mask, lab, range(1, nlab + 1)))))
        ys, xs = np.nonzero(mask)
        if len(xs) >= (40 if n == "person" else 12):             # 너무 작게 보이는 건 라벨 안 붙임
            x0, x1, y0, y1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
            lines.append(f"{ci} {(x0 + x1) / 2 / W:.6f} {(y0 + y1) / 2 / H:.6f} {(x1 - x0) / W:.6f} {(y1 - y0) / H:.6f}")
    return img, lines


def main():
    n_train = int(sys.argv[1]) if len(sys.argv) > 1 else 4000
    n_val = int(sys.argv[2]) if len(sys.argv) > 2 else 500
    for split, n in (("train", n_train), ("val", n_val)):
        os.makedirs(f"{OUT}/images/{split}", exist_ok=True); os.makedirs(f"{OUT}/labels/{split}", exist_ok=True)
        counts = np.zeros(len(CLASSES), int); empty = 0
        for i in range(n):
            img, lines = make_sample()
            Image.fromarray(img).save(f"{OUT}/images/{split}/{i:05d}.jpg", quality=95)
            open(f"{OUT}/labels/{split}/{i:05d}.txt", "w").write("\n".join(lines))
            for l in lines: counts[int(l.split()[0])] += 1
            empty += not lines
        print(f"{split}: {n}장, 물체 수 {dict(zip(CLASSES, counts.tolist()))}, 물체 없는 이미지 {empty}장", flush=True)
    open(f"{OUT}/data.yaml", "w").write(f"path: {OUT}\ntrain: images/train\nval: images/val\nnames:\n" +
                                       "".join(f"  {i}: {n}\n" for i, n in enumerate(CLASSES)))
    print("완료:", OUT)


if __name__ == "__main__":
    main()
