"""Small geometry / raster helpers shared by the vision modules."""

from __future__ import annotations

from functools import lru_cache
from typing import Iterable, Sequence

import cv2
import numpy as np


@lru_cache(maxsize=64)
def disk(radius: int) -> np.ndarray:
    r = max(0, int(radius))
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))


def odd(n: float, minimum: int = 1) -> int:
    k = max(minimum, int(round(n)))
    return k if k % 2 == 1 else k + 1


def as_contour(points: Sequence[Sequence[float]]) -> np.ndarray:
    return np.asarray(points, dtype=np.float32).reshape(-1, 1, 2)


def rasterize_polygon(points: Sequence[Sequence[float]], shape: tuple[int, int], offset=(0, 0), scale: float = 1.0) -> np.ndarray:
    mask = np.zeros(shape, np.uint8)
    if len(points) < 3:
        return mask.astype(bool)
    pts = (np.asarray(points, np.float64) - np.asarray(offset, np.float64)) * scale
    cv2.fillPoly(mask, [np.round(pts).astype(np.int32).reshape(-1, 1, 2)], 1)
    return mask.astype(bool)


def mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return float(inter) / float(union) if union else 0.0


def polygon_area(points: Sequence[Sequence[float]]) -> float:
    if len(points) < 3:
        return 0.0
    p = np.asarray(points, np.float64)
    x, y = p[:, 0], p[:, 1]
    return float(0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def point_in_polygon(pt: Sequence[float], contour: np.ndarray) -> bool:
    return cv2.pointPolygonTest(contour, (float(pt[0]), float(pt[1])), False) >= 0


def signed_distance(pt: Sequence[float], contour: np.ndarray) -> float:
    """>0 inside, <0 outside (pixels)."""
    return float(cv2.pointPolygonTest(contour, (float(pt[0]), float(pt[1])), True))


def fraction_inside(points: Iterable[Sequence[float]], contour: np.ndarray) -> float:
    pts = list(points)
    if not pts:
        return 0.0
    return sum(point_in_polygon(p, contour) for p in pts) / len(pts)


def simplify(points: Sequence[Sequence[float]], eps: float, closed: bool) -> list[tuple[float, float]]:
    if len(points) < 3:
        return [(float(x), float(y)) for x, y in points]
    approx = cv2.approxPolyDP(as_contour(points), max(eps, 0.5), closed)
    return [(round(float(p[0][0]), 1), round(float(p[0][1]), 1)) for p in approx]


def resample_polyline(points: Sequence[Sequence[float]], step: float) -> np.ndarray:
    p = np.asarray(points, np.float64)
    if len(p) < 2:
        return p
    seg = np.linalg.norm(np.diff(p, axis=0), axis=1)
    s = np.concatenate([[0], np.cumsum(seg)])
    if s[-1] == 0:
        return p[:1]
    n = max(2, int(s[-1] / max(step, 1e-6)) + 1)
    t = np.linspace(0, s[-1], n)
    return np.stack([np.interp(t, s, p[:, 0]), np.interp(t, s, p[:, 1])], axis=1)


def polyline_length(points: Sequence[Sequence[float]]) -> float:
    p = np.asarray(points, np.float64)
    if len(p) < 2:
        return 0.0
    return float(np.linalg.norm(np.diff(p, axis=0), axis=1).sum())


def bbox_of_mask(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def neighbour_count(skel: np.ndarray) -> np.ndarray:
    k = np.ones((3, 3), np.float32)
    k[1, 1] = 0
    n = cv2.filter2D(skel.astype(np.float32), -1, k, borderType=cv2.BORDER_CONSTANT)
    return np.where(skel, n, 0).astype(np.int32)
