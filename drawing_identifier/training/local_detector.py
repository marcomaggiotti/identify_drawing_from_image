"""Inference wrapper for the locally trained detector (YOLO segmentation via ultralytics)."""

from __future__ import annotations

from pathlib import Path

import numpy as np


class LocalDetector:
    """Detects shapes / lines / arrows / text with weights from ``drawid train-detector``."""

    def __init__(self, weights: str | Path, conf: float = 0.35, imgsz: int = 1024, device: str | None = None):
        try:
            from ultralytics import YOLO
        except ImportError as e:  # pragma: no cover - optional dependency
            raise RuntimeError("the local detector needs ultralytics: pip install 'drawing-identifier[yolo]'") from e
        if not Path(weights).exists():
            raise FileNotFoundError(f"detector weights not found: {weights}")
        self.model = YOLO(str(weights))
        self.conf = conf
        self.imgsz = imgsz
        self.device = device
        self.names = self.model.names

    def detect(self, image_bgr: np.ndarray) -> list[dict]:
        kw = {"conf": self.conf, "imgsz": self.imgsz, "verbose": False}
        if self.device:
            kw["device"] = self.device
        res = self.model.predict(image_bgr, **kw)[0]
        out: list[dict] = []
        if res.boxes is None:
            return out
        boxes = res.boxes.xyxy.cpu().numpy()
        cls = res.boxes.cls.cpu().numpy().astype(int)
        conf = res.boxes.conf.cpu().numpy()
        polys = res.masks.xy if getattr(res, "masks", None) is not None else [None] * len(boxes)
        for b, c, p, poly in zip(boxes, cls, conf, polys):
            out.append(
                {
                    "class": self.names[int(c)],
                    "conf": float(p),
                    "bbox": tuple(float(v) for v in b),
                    "polygon": [(float(x), float(y)) for x, y in poly] if poly is not None and len(poly) >= 3 else [],
                }
            )
        return out
