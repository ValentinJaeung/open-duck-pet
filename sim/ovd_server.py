"""개방형 탐지 서버 (OWLv2). ROS(Python 3.10)와 PyTorch 환경이 달라서 Ollama 처럼 HTTP 로 연결한다.
요청:  POST /detect  본문 = JSON {"width":W,"height":H,"rgb_b64":"<RGB 원시 바이트 base64>","prompts":["yellow cup",...],"threshold":0.12}
응답:  {"detections":[{"prompt":"yellow cup","score":0.31,"box":[x0,y0,x1,y1]}, ...]}   문장마다 점수가 가장 높은 상자 1개(임계값 이상일 때만)
실행:  .venv-vision/bin/python sim/ovd_server.py [포트 8765]     (scripts/run_ovd.sh)
"""
import base64
import json
import sys
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

import numpy as np
import torch
from PIL import Image
from transformers import Owlv2ForObjectDetection, Owlv2Processor

NAME = "google/owlv2-base-patch16-ensemble"
PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8765
print("모델 로딩:", NAME, flush=True)
proc = Owlv2Processor.from_pretrained(NAME)
model = Owlv2ForObjectDetection.from_pretrained(NAME).to("cuda").eval()


@torch.no_grad()
def detect(img, prompts, th):
    w, h = img.size
    texts = [f"a photo of a {p}" for p in prompts]
    inp = proc(text=[texts], images=img, return_tensors="pt").to("cuda")
    out = model(**inp)
    s = max(w, h)                                    # OWLv2 는 정사각형으로 패딩해서 처리 → 박스는 패딩된 정사각형 좌표
    r = proc.post_process_grounded_object_detection(out, threshold=0.03, target_sizes=torch.tensor([[s, s]]).cuda(), text_labels=[texts])[0]
    best = {}
    for sc, lb, b in zip(r["scores"], r["labels"], r["boxes"]):
        i = int(lb); sc = float(sc)
        if sc >= th and (i not in best or sc > best[i][0]):
            best[i] = (sc, [max(0.0, float(b[0])), max(0.0, float(b[1])), min(float(w), float(b[2])), min(float(h), float(b[3]))])
    return [{"prompt": prompts[i], "score": round(v[0], 4), "box": [round(x, 1) for x in v[1]]} for i, v in best.items()]


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_GET(self):                                # 상태 확인용
        self.send_response(200); self.end_headers(); self.wfile.write(b'{"ok":true,"model":"%s"}' % NAME.encode())

    def do_POST(self):
        t0 = time.time()
        try:
            req = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            arr = np.frombuffer(base64.b64decode(req["rgb_b64"]), np.uint8).reshape(req["height"], req["width"], 3)
            res = detect(Image.fromarray(arr), req["prompts"], float(req.get("threshold", 0.12)))
            body = json.dumps({"detections": res, "ms": round((time.time() - t0) * 1000)}).encode()
            self.send_response(200)
        except Exception as e:
            body = json.dumps({"error": f"{type(e).__name__}: {e}"}).encode()
            self.send_response(500)
        self.send_header("Content-Type", "application/json"); self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body)


detect(Image.new("RGB", (320, 240)), ["warmup"], 0.5)          # 워밍업
print(f"준비 완료: http://127.0.0.1:{PORT}  (GPU 메모리 {torch.cuda.memory_allocated() / 1e9:.2f} GB)", flush=True)
HTTPServer(("127.0.0.1", PORT), H).serve_forever()
