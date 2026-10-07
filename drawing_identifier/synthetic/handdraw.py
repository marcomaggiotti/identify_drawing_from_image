"""Primitives that look drawn by hand: wobbly outlines, pressure-varying strokes, curved connectors."""

from __future__ import annotations

import math

import cv2
import numpy as np

from ..schema import ShapeType


def smooth_noise(t: np.ndarray, rng: np.random.Generator, octaves: int = 3, base_freq: float = 1.0, periodic: bool = True) -> np.ndarray:
    """Sum of random sinusoids evaluated at ``t`` in [0, 1] -> roughly unit amplitude."""
    out = np.zeros_like(t, dtype=float)
    amp = 1.0
    for o in range(octaves):
        f = base_freq * (2 ** o) * (1 if periodic else rng.uniform(0.7, 1.3))
        if periodic:
            f = max(1, round(f))
        out += amp * np.sin(2 * math.pi * f * t + rng.uniform(0, 2 * math.pi))
        amp *= 0.5
    return out / 1.75


def _rot(points: np.ndarray, angle_deg: float, center) -> np.ndarray:
    a = math.radians(angle_deg)
    R = np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])
    return (points - center) @ R.T + center


def _densify(poly: np.ndarray, step: float, closed: bool = True) -> np.ndarray:
    pts = np.vstack([poly, poly[:1]]) if closed else poly
    out = []
    for a, b in zip(pts[:-1], pts[1:]):
        n = max(2, int(np.linalg.norm(b - a) / step))
        t = np.linspace(0, 1, n, endpoint=False)[:, None]
        out.append(a + (b - a) * t)
    if not closed:
        out.append(pts[-1:])
    return np.vstack(out)


def _wobble_closed(pts: np.ndarray, rng, amp: float) -> np.ndarray:
    """Displace a closed densified path along its normals with smooth noise."""
    n = len(pts)
    t = np.linspace(0, 1, n, endpoint=False)
    d = smooth_noise(t, rng, octaves=3, base_freq=rng.uniform(2, 4)) * amp
    tang = np.roll(pts, -1, axis=0) - np.roll(pts, 1, axis=0)
    norm = np.stack([-tang[:, 1], tang[:, 0]], axis=1)
    norm /= np.linalg.norm(norm, axis=1, keepdims=True) + 1e-9
    return pts + norm * d[:, None]


def clean_outline(shape_type: ShapeType, cx: float, cy: float, w: float, h: float, angle: float = 0.0, n: int = 160) -> np.ndarray:
    """Ideal (closed) outline of a primitive, centred at (cx, cy)."""
    c = np.array([cx, cy])
    if shape_type in (ShapeType.ELLIPSE, ShapeType.CIRCLE, ShapeType.SCRIBBLE):
        t = np.linspace(0, 2 * math.pi, n, endpoint=False)
        pts = np.stack([cx + w / 2 * np.cos(t), cy + h / 2 * np.sin(t)], axis=1)
        return _rot(pts, angle, c)
    if shape_type == ShapeType.RECTANGLE:
        poly = np.array([[cx - w / 2, cy - h / 2], [cx + w / 2, cy - h / 2], [cx + w / 2, cy + h / 2], [cx - w / 2, cy + h / 2]])
    elif shape_type == ShapeType.ROUNDED_RECTANGLE:
        r = min(w, h) * 0.22
        pts = []
        for ox, oy, a0 in ((w / 2 - r, h / 2 - r, 0), (-w / 2 + r, h / 2 - r, 90), (-w / 2 + r, -h / 2 + r, 180), (w / 2 - r, -h / 2 + r, 270)):
            for k in range(10):
                a = math.radians(a0 + 90 * k / 9)
                pts.append((cx + ox + r * math.cos(a), cy + oy + r * math.sin(a)))
        poly = np.array(pts)
    elif shape_type == ShapeType.TRIANGLE:
        poly = np.array([[cx, cy - h / 2], [cx + w / 2, cy + h / 2], [cx - w / 2, cy + h / 2]])
    elif shape_type == ShapeType.DIAMOND:
        poly = np.array([[cx, cy - h / 2], [cx + w / 2, cy], [cx, cy + h / 2], [cx - w / 2, cy]])
    elif shape_type == ShapeType.POLYGON:
        k = 6
        t = np.linspace(0, 2 * math.pi, k, endpoint=False) - math.pi / 2
        poly = np.stack([cx + w / 2 * np.cos(t), cy + h / 2 * np.sin(t)], axis=1)
    else:
        raise ValueError(shape_type)
    poly = _rot(poly, angle, c)
    return _densify(poly, max(2.0, (w + h) / n))


def hand_outline(shape_type: ShapeType, cx: float, cy: float, w: float, h: float, rng: np.random.Generator, angle: float = 0.0, wobble: float = 0.012) -> tuple[np.ndarray, np.ndarray]:
    """Returns (stroke path as drawn, closed annotation polygon).

    The stroke starts at a random point, may overshoot or stop short of where it
    began (pen lift), and wanders slightly - like a quickly drawn oval.
    """
    clean = clean_outline(shape_type, cx, cy, w, h, angle)
    scale = (w + h) / 2
    amp = wobble * scale * rng.uniform(0.5, 1.5)
    if shape_type in (ShapeType.RECTANGLE, ShapeType.TRIANGLE, ShapeType.DIAMOND, ShapeType.POLYGON):
        amp *= 0.6
    wob = _wobble_closed(clean, rng, amp)
    n = len(wob)
    start = rng.integers(0, n)
    wob = np.roll(wob, -start, axis=0)
    annotation = wob.copy()
    # overshoot (continue past the start) or leave a small gap
    mode = rng.random()
    if mode < 0.45:
        extra = int(n * rng.uniform(0.02, 0.12))
        drift = np.linspace(0, 1, extra)[:, None] * rng.normal(0, 0.01 * scale, 2)
        path = np.vstack([wob, wob[:extra] + drift])
    elif mode < 0.65:
        cut = int(n * rng.uniform(0.0, 0.015))
        path = wob[: n - cut] if cut else np.vstack([wob, wob[:1]])
    else:
        path = np.vstack([wob, wob[:1]])
    return path, annotation


def bezier(p0, p1, rng: np.random.Generator, bend: float = 0.25, n: int = 60, wobble: float = 0.01) -> np.ndarray:
    """Curved hand-drawn connector from p0 to p1."""
    p0 = np.asarray(p0, float)
    p1 = np.asarray(p1, float)
    d = p1 - p0
    L = np.linalg.norm(d) + 1e-9
    nrm = np.array([-d[1], d[0]]) / L
    c1 = p0 + d * rng.uniform(0.2, 0.4) + nrm * L * rng.uniform(-bend, bend)
    c2 = p0 + d * rng.uniform(0.6, 0.8) + nrm * L * rng.uniform(-bend, bend)
    t = np.linspace(0, 1, max(8, n))[:, None]
    pts = (1 - t) ** 3 * p0 + 3 * (1 - t) ** 2 * t * c1 + 3 * (1 - t) * t**2 * c2 + t**3 * p1
    if wobble > 0:
        tt = t.ravel()
        off = smooth_noise(tt, rng, octaves=2, base_freq=rng.uniform(1, 3), periodic=False) * wobble * L
        off *= np.sin(math.pi * tt)  # keep the ends where they were aimed
        pts = pts + nrm * off[:, None]
    return pts


def elbow(p0, p1, rng: np.random.Generator) -> np.ndarray:
    """Right-angled connector (like the heavy lines in nested-rectangle graphs)."""
    p0 = np.asarray(p0, float)
    p1 = np.asarray(p1, float)
    if rng.random() < 0.5:
        mid = np.array([p1[0], p0[1]])
    else:
        mid = np.array([p0[0], p1[1]])
    pts = np.vstack([np.linspace(p0, mid, 20), np.linspace(mid, p1, 20)[1:]])
    return pts + rng.normal(0, 0.6, pts.shape)


def draw_stroke(img: np.ndarray, pts: np.ndarray, width: float, color, rng: np.random.Generator, taper: bool = True, pressure: float = 0.25) -> None:
    """Draw a polyline with smoothly varying width (pen pressure) and tapered ends."""
    pts = np.asarray(pts, float)
    if len(pts) < 2:
        return
    n = len(pts)
    t = np.linspace(0, 1, n)
    wv = width * (1 + pressure * smooth_noise(t, rng, octaves=2, base_freq=rng.uniform(1, 3), periodic=False))
    if taper and n > 6:
        k = max(2, n // 12)
        ramp = np.ones(n)
        ramp[:k] = np.linspace(0.55, 1, k)
        ramp[-k:] = np.linspace(1, 0.55, k)
        wv *= ramp
    wv = np.maximum(1, wv)
    ip = np.round(pts * 4).astype(np.int32)  # 2 fractional bits for sub-pixel accuracy
    for i in range(n - 1):
        th = int(round((wv[i] + wv[i + 1]) / 2))
        cv2.line(img, tuple(ip[i]), tuple(ip[i + 1]), color, max(1, th), cv2.LINE_AA, shift=2)


def scribble(center, size, rng: np.random.Generator, n: int = 14) -> np.ndarray:
    """Loopy back-and-forth scribbling used to cross something out."""
    cx, cy = center
    w, h = size
    t = np.linspace(0, 1, 60 * n)
    f1, f2 = rng.uniform(3, 7) * n / 6, rng.uniform(5, 11) * n / 6
    x = cx + w / 2 * (0.7 * np.sin(2 * math.pi * f1 * t + rng.uniform(0, 6)) + 0.3 * smooth_noise(t, rng, 3, 2, periodic=False))
    y = cy + h / 2 * (0.7 * np.sin(2 * math.pi * f2 * t + rng.uniform(0, 6)) + 0.3 * smooth_noise(t, rng, 3, 2, periodic=False))
    return np.stack([x, y], axis=1)


def boundary_point_towards(outline: np.ndarray, center, target) -> np.ndarray:
    """Point of a closed outline in the direction of ``target`` as seen from ``center``."""
    c = np.asarray(center, float)
    d = np.asarray(target, float) - c
    ang = math.atan2(d[1], d[0])
    v = outline - c
    a = np.arctan2(v[:, 1], v[:, 0])
    diff = np.abs((a - ang + math.pi) % (2 * math.pi) - math.pi)
    return outline[int(np.argmin(diff))]
