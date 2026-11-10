"""공 밀기 학습용 장면(XML) 생성: 원본 오리 장면 + 공 하나(자유 관절). third_party 는 수정하지 않는다.
생성 결과 sim/rl/scene_push_ball.xml 을 Playground 환경이 읽는다(메시는 Playground 의 assets 에서 이름으로 찾음).
"""
import os
import mujoco

HERE = os.path.dirname(os.path.abspath(__file__))
XMLS = os.path.join(HERE, "..", "..", "third_party", "Open_Duck_Playground", "playground", "open_duck_mini_v2", "xmls")
BALL_R, BALL_MASS = 0.05, 0.05
ROLLING = 0.002          # 구름 마찰. 0.0001 이면 0.4m/s 로 찬 공이 6초에 1.7m 굴러가 제어 불가(측정) → 0.002 면 0.23m


def make(out=os.path.join(HERE, "scene_push_ball.xml")):
    spec = mujoco.MjSpec.from_file(os.path.join(XMLS, "scene_flat_terrain_backlash.xml"))
    ball = spec.worldbody.add_body(name="ball", pos=[0.5, 0.0, BALL_R])
    ball.add_freejoint(name="ball_free")
    ball.add_geom(name="ball_geom", type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[BALL_R, 0, 0], mass=BALL_MASS,
                  rgba=[0.1, 0.3, 0.9, 1], friction=[0.6, 0.005, ROLLING], condim=6)
    for k in spec.keys:
        k.qpos = list(k.qpos) + [0.5, 0.0, BALL_R, 1, 0, 0, 0]
    xml = spec.to_xml()
    open(out, "w").write(xml)
    return out


if __name__ == "__main__":
    print(make())
