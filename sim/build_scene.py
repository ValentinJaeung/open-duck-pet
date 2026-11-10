"""방 + 오리 + 머리 카메라를 합친 MuJoCo 모델을 만듭니다. (원본 third_party 는 수정하지 않음)

사용:
    from sim.build_scene import build
    model = build()                     # mujoco.MjModel
방 치수/물체는 아래 ROOM, FURNITURE, OBJECTS 만 고치면 됩니다. (단위: 미터)
"""
import os
import mujoco

HERE = os.path.dirname(os.path.abspath(__file__))
XMLS = os.path.join(HERE, "..", "third_party", "Open_Duck_Playground",
                    "playground", "open_duck_mini_v2", "xmls")

# --- 내 방 (임시 값: 줄자로 재서 바꾸세요) ---------------------------------
ROOM = dict(x=4.0, y=3.0, wall_h=1.0)           # 가로(x) 세로(y) 벽 높이
# (이름, 중심 x, y, 크기 x/2, y/2, z/2, 색 rgba)  — 바닥(z=0) 위에 놓임
FURNITURE = [
    ("table", 1.4, -0.9, 0.40, 0.25, 0.20, (0.55, 0.38, 0.22, 1)),
    ("bed",  -1.3,  0.8, 0.60, 0.50, 0.15, (0.40, 0.50, 0.70, 1)),
]
# 오리가 찾을 물체 (움직일 수 있음): (이름, 모양, x, y, 크기, 색)
OBJECTS = [
    ("red_block",   "box",    1.0,  0.3, 0.05, (0.9, 0.1, 0.1, 1)),
    ("green_block", "box",    0.6, -0.6, 0.04, (0.1, 0.7, 0.2, 1)),
    ("blue_ball",   "sphere", 1.2,  1.0, 0.05, (0.1, 0.3, 0.9, 1)),
]
# 학습하지 않은 새 물체 (8단계 개방형 탐지 시험용. build(novel=True) 일 때만 추가됨): (이름, 모양, x, y, 크기(반지름, 반높이), 색)
NOVEL_OBJECTS = [
    ("yellow_cup", "cylinder", 0.3, 0.9, (0.035, 0.045), (0.95, 0.85, 0.1, 1)),
    ("orange_ball", "sphere", -0.5, -0.7, (0.045, 0.0), (0.95, 0.5, 0.05, 1)),
    ("purple_box", "box", 1.6, 0.4, (0.05, 0.05), (0.55, 0.2, 0.7, 1)),
]
# 사용자가 조종하는 '사람' (mocap 몸체: 물리 충돌 없이 위치만 지정). 키 약 1m 인형
PERSON_START = (-1.4, -0.9, 0.0)            # x, y, yaw(rad)
PERSON_PARTS = [                            # (이름, 중심 x,y,z, 반 크기 x,y,z, 색 rgba)
    ("p_legL", 0, 0.07, 0.2, 0.05, 0.05, 0.2, (0.15, 0.2, 0.45, 1)),
    ("p_legR", 0, -0.07, 0.2, 0.05, 0.05, 0.2, (0.15, 0.2, 0.45, 1)),
    ("p_torso", 0, 0, 0.62, 0.08, 0.14, 0.22, (0.95, 0.55, 0.1, 1)),
    ("p_armL", 0, 0.2, 0.62, 0.035, 0.035, 0.18, (0.95, 0.55, 0.1, 1)),
    ("p_armR", 0, -0.2, 0.62, 0.035, 0.035, 0.18, (0.95, 0.55, 0.1, 1)),
]
PERSON_HEAD = ("p_head", 0, 0, 0.97, 0.09, (0.9, 0.75, 0.6, 1))   # 구
# 머리 카메라: 머리 몸체 기준 위치/방향/시야각 (실제 방향은 render_check.py 로 확인 후 조정)
# quat: 머리 몸체 기준. 로봇 정면(+x)을 수평으로 바라보도록 계산한 값
CAM = dict(pos=(0.02, 0.0, -0.06), quat=(0.7071, 0.0, 0.0, -0.7071), fovy=70)


def build(cam=None, dynamic_objects=True, n_distractors=0, person=True, novel=False):
    c = {**CAM, **(cam or {})}
    spec = mujoco.MjSpec.from_file(os.path.join(XMLS, "scene_flat_terrain_backlash.xml"))
    # 기존 평평한 바닥 제거 (방이 바닥을 가짐)
    for b in list(spec.bodies):
        if b.name == "floor":
            spec.delete(b)

    # 머리 카메라
    cam_obj = spec.body("head_assembly").add_camera(name="head_cam", pos=list(c["pos"]), fovy=c["fovy"])
    cam_obj.quat = list(c["quat"])

    spec.visual.global_.offwidth = 1280
    spec.visual.global_.offheight = 720
    w = spec.worldbody
    w.add_light(pos=[0, 0, 2.5], dir=[0, 0, -1], type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL, diffuse=[0.5, 0.5, 0.5])

    # 바닥 (원래 학습 환경과 같은 마찰/접촉 설정)
    # 몸체 이름이 "floor" 여야 함: 걷기 정책의 발 접촉 확인 코드가 이 이름을 찾음
    w.add_body(name="floor").add_geom(name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[ROOM["x"] / 2, ROOM["y"] / 2, 0.05],
               rgba=[0.75, 0.63, 0.47, 1], friction=[0.6, 0.005, 0.0001], condim=3, contype=1, conaffinity=0, priority=1)
    # 벽 4개
    X, Y, H, t = ROOM["x"] / 2, ROOM["y"] / 2, ROOM["wall_h"] / 2, 0.05
    for n, pos, size in [("wall_n", (0, Y + t, H), (X + 2 * t, t, H)), ("wall_s", (0, -Y - t, H), (X + 2 * t, t, H)),
                         ("wall_e", (X + t, 0, H), (t, Y, H)), ("wall_w", (-X - t, 0, H), (t, Y, H))]:
        w.add_geom(name=n, type=mujoco.mjtGeom.mjGEOM_BOX, pos=list(pos), size=list(size), rgba=[0.93, 0.93, 0.9, 1])
    # 가구 (고정)
    for n, x, y, sx, sy, sz, rgba in FURNITURE:
        w.add_geom(name=n, type=mujoco.mjtGeom.mjGEOM_BOX, pos=[x, y, sz], size=[sx, sy, sz], rgba=list(rgba))
    # 물체. dynamic_objects=False 면 고정(정책 추론 스크립트 호환용). 오리 다음에 추가되므로 키프레임 qpos 뒤에 7개씩 이어붙임
    extra_qpos = []
    for n, shape, x, y, s, rgba in OBJECTS:
        b = w.add_body(name=n, pos=[x, y, s])
        if dynamic_objects:
            b.add_freejoint(name=f"{n}_free")
        if shape == "box":
            b.add_geom(name=n, type=mujoco.mjtGeom.mjGEOM_BOX, size=[s, s, s], rgba=list(rgba), mass=0.05, friction=[0.6, 0.005, 0.0001])
        else:
            b.add_geom(name=n, type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[s, 0, 0], rgba=list(rgba), mass=0.05,
                       friction=[0.6, 0.005, 0.002], condim=6)   # 공 밀기 학습(sim/rl/make_xml.py)과 같은 접촉 설정
        if dynamic_objects:
            extra_qpos += [x, y, s, 1, 0, 0, 0]
    # 새 물체: 고정 물체로 놓음(탐지 시험용). 크기: cylinder(반지름, 반높이), sphere(반지름), box(한 변 반)
    if novel:
        for n, shape, x, y, (r, hh), rgba in NOVEL_OBJECTS:
            nb = w.add_body(name=n, pos=[x, y, 0])                     # 몸체 하나에 부품 여러 개 → 통째로 옮길 수 있음
            if shape == "cylinder":                                    # 컵: 몸통 + 손잡이 + 어두운 입구
                nb.add_geom(name=n, type=mujoco.mjtGeom.mjGEOM_CYLINDER, pos=[0, 0, hh], size=[r, hh, 0], rgba=list(rgba))
                nb.add_geom(name=n + "_handle", type=mujoco.mjtGeom.mjGEOM_BOX, pos=[r + 0.012, 0, hh], size=[0.014, 0.006, 0.028], rgba=list(rgba))
                nb.add_geom(name=n + "_inside", type=mujoco.mjtGeom.mjGEOM_CYLINDER, pos=[0, 0, 2 * hh + 0.0005], size=[r * 0.85, 0.001, 0], rgba=[0.25, 0.15, 0.05, 1])
            elif shape == "sphere":
                nb.add_geom(name=n, type=mujoco.mjtGeom.mjGEOM_SPHERE, pos=[0, 0, r], size=[r, 0, 0], rgba=list(rgba))
            else:
                nb.add_geom(name=n, type=mujoco.mjtGeom.mjGEOM_BOX, pos=[0, 0, r], size=[r, r, r], rgba=list(rgba))
    # 사람 (mocap). 처음엔 방 구석에 둠. 위치는 data.mocap_pos / mocap_quat 로 옮긴다
    if person:
        pb = w.add_body(name="person", pos=[PERSON_START[0], PERSON_START[1], 0], mocap=True)
        for n, x, y, z, sx, sy, sz, rgba in PERSON_PARTS:
            pb.add_geom(name=n, type=mujoco.mjtGeom.mjGEOM_BOX, pos=[x, y, z], size=[sx, sy, sz], rgba=list(rgba), contype=0, conaffinity=0)
        n, x, y, z, r, rgba = PERSON_HEAD
        pb.add_geom(name=n, type=mujoco.mjtGeom.mjGEOM_SPHERE, pos=[x, y, z], size=[r, 0, 0], rgba=list(rgba), contype=0, conaffinity=0)
    # 방해 물체(다른 색 상자): 탐지 학습용. 평소엔 0개. 처음엔 방 밖에 숨겨 두고 데이터 생성기가 옮김
    for i in range(n_distractors):
        b = w.add_body(name=f"distractor_{i}", pos=[0, 0, -5])
        b.add_geom(name=f"distractor_{i}", type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.05, 0.05, 0.05],
                   rgba=[0.5, 0.5, 0.5, 1], contype=0, conaffinity=0)
    for k in spec.keys:
        k.qpos = list(k.qpos) + extra_qpos
    return spec.compile()


if __name__ == "__main__":
    m = build()
    print("OK  nq=%d nu=%d cameras=%d" % (m.nq, m.nu, m.ncam))
