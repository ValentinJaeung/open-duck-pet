"""방 안에서 학습된 걷기 정책을 키보드로 조종합니다. (MuJoCo 뷰어)
사용: bash scripts/walk_in_room.sh [onnx]
조작: 화면 클릭 후 ↑↓←→ 이동, Q/E 회전, H 고개 모드. 누르는 동안만 명령이 들어갑니다.
참고: 기존 추론 코드가 '움직이는 물체 1개(오리)'만 가정해서, 방 물체는 고정으로 둡니다.
"""
import os, sys, glob
import mujoco
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.join(HERE, "..")
sys.path.insert(0, ROOT)
from sim.build_scene import build
from playground.open_duck_mini_v2 import mujoco_infer

onnx = sys.argv[1] if len(sys.argv) > 1 else max(glob.glob(f"{ROOT}/models/runs/*/*.onnx"), key=os.path.getmtime)
print("정책:", onnx)
room_model = build(dynamic_objects=False)
mujoco.MjModel.from_xml_string = staticmethod(lambda *a, **k: room_model)   # 방 모델을 끼워 넣기
base = os.path.join(ROOT, "third_party/Open_Duck_Playground/playground/open_duck_mini_v2")
# model_path 는 사용되지 않지만 Path.read_text() 가 호출되므로 존재하는 파일을 넘김
mujoco_infer.MjInfer(os.path.join(base, "xmls/scene_flat_terrain_backlash.xml"),
                     os.path.join(base, "data/polynomial_coefficients.pkl"), os.path.realpath(onnx), False).run()
