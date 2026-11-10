"""개방형 탐지 모델 비교. 사용: .venv-vision/bin/python sim/bench_ovd.py <owlv2|owlvit|gdino> [이미지 수]
같은 이미지에서 7개 문장(새 물체 3 + 기존 4)을 한꺼번에 질문하고, 정답(IoU>=0.5)과 비교한다.
"""
import json, os, sys, time
import numpy as np
import torch
from PIL import Image

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
D = os.path.join(ROOT, "datasets", "ovd_eval")
meta = json.load(open(f"{D}/labels.json")); PROMPTS = meta["prompts"]; LAB = meta["labels"]
names = list(PROMPTS); texts = [PROMPTS[n] for n in names]
which = sys.argv[1]; N = int(sys.argv[2]) if len(sys.argv) > 2 else 120
dev = "cuda"
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection, Owlv2ForObjectDetection, OwlViTForObjectDetection, Owlv2Processor, OwlViTProcessor

if which == "owlv2":
    P = Owlv2Processor.from_pretrained("google/owlv2-base-patch16-ensemble"); M = Owlv2ForObjectDetection.from_pretrained("google/owlv2-base-patch16-ensemble").to(dev).eval()
elif which == "owlvit":
    P = OwlViTProcessor.from_pretrained("google/owlvit-base-patch32"); M = OwlViTForObjectDetection.from_pretrained("google/owlvit-base-patch32").to(dev).eval()
else:
    P = AutoProcessor.from_pretrained("IDEA-Research/grounding-dino-tiny"); M = AutoModelForZeroShotObjectDetection.from_pretrained("IDEA-Research/grounding-dino-tiny").to(dev).eval()
params = sum(p.numel() for p in M.parameters()) / 1e6


@torch.no_grad()
def detect(img):
    """반환: [(문장 인덱스, 점수, [x0,y0,x1,y1])]"""
    w, h = img.size
    if which.startswith("owl"):
        prompts = [f"a photo of a {t}" for t in texts]
        inp = P(text=[prompts], images=img, return_tensors="pt").to(dev)
        out = M(**inp)
        s = max(w, h) if which == "owlv2" else None
        ts = torch.tensor([[s, s]] if s else [[h, w]]).to(dev)
        r = P.post_process_grounded_object_detection(out, threshold=0.03, target_sizes=ts, text_labels=[prompts])[0]
        return [(int(l), float(sc), [float(v) for v in b]) for sc, l, b in zip(r["scores"], r["labels"], r["boxes"])]
    inp = P(images=img, text=". ".join(texts) + ".", return_tensors="pt").to(dev)
    out = M(**inp)
    r = P.post_process_grounded_object_detection(out, inp.input_ids, threshold=0.05, text_threshold=0.05, target_sizes=[(h, w)])[0]
    res = []
    labs = r.get("text_labels", r.get("labels"))
    for sc, lb, b in zip(r["scores"], labs, r["boxes"]):
        lb = str(lb).strip()
        idx = [i for i, t in enumerate(texts) if lb == t] or [i for i, t in enumerate(texts) if lb and (lb in t or t in lb)]
        if len(idx) == 1: res.append((idx[0], float(sc), [float(v) for v in b]))      # 문장이 모호하면(여러 개에 걸침) 버림
    return res


def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))
    return ix / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - ix + 1e-9)


files = sorted(LAB)[:N]
dets = {}
img0 = Image.open(f"{D}/images/{files[0]}").convert("RGB")
for _ in range(3): detect(img0)                                   # 워밍업
torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize(); t0 = time.time()
for f in files:
    dets[f] = detect(Image.open(f"{D}/images/{f}").convert("RGB"))
torch.cuda.synchronize(); ms = (time.time() - t0) / len(files) * 1000
vram = torch.cuda.max_memory_allocated() / 1e9


def score(th, keys):
    """문장별 (맞춤, 놓침, 오탐). 한 문장당 한 이미지에서 점수가 가장 높은 상자 하나만 후보로 본다"""
    tp = fn = fp = 0
    for f in files:
        gt = {o["name"]: o["box"] for o in LAB[f]}
        for i, n in enumerate(names):
            if n not in keys: continue
            c = [d for d in dets[f] if d[0] == i and d[1] >= th]
            best = max(c, key=lambda d: d[1]) if c else None
            if n in gt:
                if best and iou(best[2], gt[n]) >= 0.5: tp += 1
                else: fn += 1; fp += 1 if best else 0
            elif best: fp += 1
    return tp, fn, fp


print(f"\n=== {which}: 파라미터 {params:.0f}M, {len(files)}장, 속도 {ms:.0f} ms/장, GPU 메모리 최대 {vram:.2f} GB")
novel = {"yellow_cup", "orange_ball", "purple_box"}
for group, keys in (("새 물체(학습 안 한 것)", novel), ("기존 물체", set(names) - novel)):
    best = max(((th,) + score(th, keys) for th in np.arange(0.03, 0.8, 0.01)), key=lambda r: 2 * r[1] / (2 * r[1] + r[2] + r[3] + 1e-9))
    th, tp, fn, fp = best
    print(f"  {group:18s} 최적 임계값 {th:.2f}: 재현율 {tp / max(1, tp + fn):.3f}  정밀도 {tp / max(1, tp + fp):.3f}  (맞춤 {tp}, 놓침 {fn}, 오탐 {fp})")
    for n in sorted(keys, key=names.index):
        tp, fn, fp = score(th, {n})
        print(f"      {n:12s} 재현율 {tp / max(1, tp + fn):.2f} 정밀도 {tp / max(1, tp + fp):.2f}")
