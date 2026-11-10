"""YOLOv8 ONNX 추론 (onnxruntime + numpy 만 사용). ROS(Python 3.10) 환경에서 PyTorch 없이 돌리기 위함."""
import numpy as np
import onnxruntime as ort


class YoloOnnx:
    def __init__(self, path, names, size=320, conf=0.4, iou=0.5):
        self.sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
        self.inp = self.sess.get_inputs()[0].name
        self.names, self.size, self.conf, self.iou = names, size, conf, iou

    def __call__(self, rgb):
        """rgb: (H,W,3) uint8 → [(label, score, x0, y0, x1, y1)] 원본 픽셀 좌표"""
        h, w = rgb.shape[:2]
        s = self.size / max(h, w)
        nh, nw = round(h * s), round(w * s)
        # 레터박스: 학습 때와 같이 가운데 정렬 + 회색(114) 패딩
        canvas = np.full((self.size, self.size, 3), 114, np.uint8)
        top, left = (self.size - nh) // 2, (self.size - nw) // 2
        ys = (np.arange(nh) / s).astype(int).clip(0, h - 1)
        xs = (np.arange(nw) / s).astype(int).clip(0, w - 1)
        canvas[top:top + nh, left:left + nw] = rgb[ys][:, xs]
        x = canvas.transpose(2, 0, 1)[None].astype(np.float32) / 255.0
        out = self.sess.run(None, {self.inp: x})[0][0]          # (4+nc, N)
        boxes, scores = out[:4].T, out[4:].T
        cls = scores.argmax(1); conf = scores[np.arange(len(cls)), cls]
        keep = conf >= self.conf
        boxes, cls, conf = boxes[keep], cls[keep], conf[keep]
        if len(boxes) == 0:
            return []
        xyxy = np.stack([boxes[:, 0] - boxes[:, 2] / 2, boxes[:, 1] - boxes[:, 3] / 2,
                         boxes[:, 0] + boxes[:, 2] / 2, boxes[:, 1] + boxes[:, 3] / 2], 1)
        res = []
        for c in np.unique(cls):                                   # 클래스별 NMS
            idx = np.nonzero(cls == c)[0]
            for i in self._nms(xyxy[idx], conf[idx]):
                x0, y0, x1, y1 = xyxy[idx[i]]
                res.append((self.names[int(c)], float(conf[idx[i]]),
                            float(np.clip((x0 - left) / s, 0, w)), float(np.clip((y0 - top) / s, 0, h)),
                            float(np.clip((x1 - left) / s, 0, w)), float(np.clip((y1 - top) / s, 0, h))))
        return res

    def _nms(self, b, sc):
        order, keep = sc.argsort()[::-1], []
        while len(order):
            i = order[0]; keep.append(i)
            xx0 = np.maximum(b[i, 0], b[order[1:], 0]); yy0 = np.maximum(b[i, 1], b[order[1:], 1])
            xx1 = np.minimum(b[i, 2], b[order[1:], 2]); yy1 = np.minimum(b[i, 3], b[order[1:], 3])
            inter = np.clip(xx1 - xx0, 0, None) * np.clip(yy1 - yy0, 0, None)
            area = lambda q: (q[..., 2] - q[..., 0]) * (q[..., 3] - q[..., 1])
            iou = inter / (area(b[i]) + area(b[order[1:]]) - inter + 1e-9)
            order = order[1:][iou < self.iou]
        return keep
