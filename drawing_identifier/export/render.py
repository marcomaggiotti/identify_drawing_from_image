"""Overlay rendering (also used as the 'set-of-marks' image shown to the verifier VLM)."""

from __future__ import annotations

import colorsys

import cv2
import numpy as np

from ..schema import DiagramGraph, ShapeType


def _color(i: int) -> tuple[int, int, int]:
    h = (i * 0.618033988749895) % 1.0
    r, g, b = colorsys.hsv_to_rgb(h, 0.85, 0.85)
    return int(b * 255), int(g * 255), int(r * 255)


def _label(img, text, org, color, scale=0.5):
    x, y = int(org[0]), int(org[1])
    (tw, th), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
    x = max(0, min(img.shape[1] - tw - 2, x))
    y = max(th + 2, min(img.shape[0] - 2, y))
    cv2.rectangle(img, (x - 1, y - th - 2), (x + tw + 1, y + base - 1), (255, 255, 255), -1)
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)


def render_overlay(image: np.ndarray, g: DiagramGraph, show_text: bool = True, fade: float = 0.45) -> np.ndarray:
    """Draw shapes (S#), connections (C#) and texts (T#) on a faded copy of the image."""
    base = image.copy()
    if base.ndim == 2:
        base = cv2.cvtColor(base, cv2.COLOR_GRAY2BGR)
    out = cv2.addWeighted(base, 1 - fade, np.full_like(base, 255), fade, 0)
    scale = max(0.4, min(1.0, max(out.shape[:2]) / 2000))
    thick = max(2, int(round(max(out.shape[:2]) / 700)))
    for i, s in enumerate(g.shapes):
        col = _color(i)
        if len(s.polygon) >= 3:
            pts = np.round(np.asarray(s.polygon)).astype(np.int32).reshape(-1, 1, 2)
            style = cv2.LINE_AA
            cv2.polylines(out, [pts], True, col, thick if s.type != ShapeType.REGION else max(1, thick - 1), style)
        x0, y0, _, _ = s.bbox.xyxy
        _label(out, f"{s.id} {s.type.value}", (x0, y0 - 2), col, scale)
    for c in g.connections:
        col = (0, 0, 200) if c.heavy else (200, 0, 120)
        if len(c.path) >= 2:
            pts = np.round(np.asarray(c.path)).astype(np.int32).reshape(-1, 1, 2)
            cv2.polylines(out, [pts], False, col, thick, cv2.LINE_AA)
        for ep in c.endpoints:
            cv2.circle(out, (int(ep.point[0]), int(ep.point[1])), thick + 2, (0, 140, 255) if ep.is_head else col, -1)
        if c.path:
            mid = c.path[len(c.path) // 2]
            _label(out, c.id, (mid[0] + 4, mid[1] - 4), col, scale)
    for t in g.texts:
        x0, y0, x1, y1 = (int(round(v)) for v in t.bbox.xyxy)
        cv2.rectangle(out, (x0, y0), (x1, y1), (0, 150, 0), 1)
        lab = t.id
        if show_text and t.text:
            lab += f' "{t.text[:24]}"'
        _label(out, lab, (x0, y1 + int(14 * scale) + 2), (0, 120, 0), scale * 0.9)
    return out
