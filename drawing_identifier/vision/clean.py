"""Remove running prose from a manuscript page, keep only the diagrams.

Ported from the author's ``clean_eg.py`` and made scale-aware (the original
pixel constants were tuned on 826x1273 Houghton cards; here they are
fractions of the page diagonal or multiples of the pen width):

1. **cut ovals**: seal the ink with a closing, flood the background from
   the border; unreached white pockets are enclosed areas. A pocket counts as
   a cut when it is big enough and holds a little ink (the words floating
   inside); empty pockets are cursive loops.
2. **lines of identity**: strokes much heavier than the pen (distance
   transform), kept only near a cut.
3. cut and line seeds are grown into **diagram regions**.
4. ink is kept **by connected component**: any blob that reaches a diagram
   region is kept whole (outline, interior words, words hanging on lines),
   prose sharing a bounding box is not.

``reach`` trades cleanliness for completeness: larger values recover more
attached words but may grab neighbouring prose.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class CleanParams:
    seal_frac: float = 0.006  # closing that seals hand-drawn rings (x page diagonal)
    min_oval_frac: float = 0.0198  # sqrt(min pocket area) (x diagonal)
    max_oval_area_frac: float = 0.25  # reject page-sized pockets
    min_oval_w_frac: float = 0.020
    min_oval_h_frac: float = 0.012
    min_inside_ink: float = 0.05  # a cut encloses *some* ink
    thick_scale: float = 1.3  # heavy line: half-width >= thick_scale x pen width
    line_link_frac: float = 0.024  # heavy strokes kept only this close to a cut
    cluster_frac: float = 0.024  # dilation merging one figure's parts
    min_cluster_frac: float = 0.029  # sqrt(min region area) (x diagonal)
    reach_frac: float = 0.012  # cleanliness <-> completeness knob
    ignore_blue: bool = True  # archival stamps / blue pencil are not part of a graph
    stamp_corner: tuple[float, float] | None = None  # (h, w) fractions of a top-left stamp to ignore


def _k(n: float) -> np.ndarray:
    n = max(3, int(round(n)))
    return np.ones((n, n), np.uint8)


def keep_mask(ink: np.ndarray, bgr: np.ndarray | None = None, stroke_width: float = 2.0, p: CleanParams | None = None) -> tuple[np.ndarray, dict]:
    """(bool mask of ink to keep, debug layers) for a bool ``ink`` mask."""
    p = p or CleanParams()
    H, W = ink.shape
    d = float(np.hypot(H, W))
    ink8 = ink.astype(np.uint8) * 255
    blue = np.zeros_like(ink8)
    if bgr is not None and bgr.ndim == 3 and p.ignore_blue:
        b, g, r = (bgr[:, :, i].astype(np.int16) for i in range(3))
        blue = (((b - r) > 20) & ((b - g) > 10) & (b > 80)).astype(np.uint8) * 255
    stamp = np.zeros_like(ink8)
    if p.stamp_corner:
        stamp[: int(H * p.stamp_corner[0]), : int(W * p.stamp_corner[1])] = 255

    # 1. cut ovals: enclosed pockets that hold a little ink
    seal = max(3, int(round(p.seal_frac * d)) | 1)
    sealed = cv2.morphologyEx(ink8, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (seal, seal)), iterations=2)
    flood = sealed.copy()
    cv2.floodFill(flood, np.zeros((H + 2, W + 2), np.uint8), (0, 0), 255)
    holes = cv2.bitwise_and(cv2.bitwise_not(sealed), cv2.bitwise_not(flood))
    n, lab, st, _ = cv2.connectedComponentsWithStats(holes, 8)
    oval = np.zeros_like(ink8)
    min_area = (p.min_oval_frac * d) ** 2
    for i in range(1, n):
        x, y, w, h, a = st[i]
        if not (min_area < a < p.max_oval_area_frac * H * W):
            continue
        if w < p.min_oval_w_frac * d or h < p.min_oval_h_frac * d:
            continue
        if (ink8[y : y + h, x : x + w] > 0).mean() < p.min_inside_ink:
            continue  # empty pocket: a cursive loop, not a cut
        if (blue[y : y + h, x : x + w] > 0).mean() > 0.20 or (stamp[y : y + h, x : x + w] > 0).mean() > 0.30:
            continue
        oval[y : y + h, x : x + w][lab[y : y + h, x : x + w] == i] = 255

    # 2. heavy lines of identity near a cut
    dist = cv2.distanceTransform(ink8, cv2.DIST_L2, 5)
    thick = ((dist >= max(2.0, p.thick_scale * stroke_width)) * 255).astype(np.uint8)
    thick[(blue > 0) | (stamp > 0)] = 0
    thick = cv2.bitwise_and(thick, cv2.dilate(oval, _k(p.line_link_frac * d)))

    # 3. diagram regions
    c = p.cluster_frac * d
    grown = cv2.morphologyEx(cv2.dilate(cv2.bitwise_or(oval, thick), _k(c)), cv2.MORPH_CLOSE, _k(c + 8))
    nc, labc, stc, _ = cv2.connectedComponentsWithStats(grown, 8)
    diagram = np.zeros_like(ink8)
    for i in range(1, nc):
        if stc[i, cv2.CC_STAT_AREA] > (p.min_cluster_frac * d) ** 2:
            diagram[labc == i] = 255
    reach = cv2.dilate(diagram, _k(p.reach_frac * d)) > 0

    # 4. keep whole ink components that touch a diagram region
    ni, labi = cv2.connectedComponents(ink8, connectivity=8)
    touched = np.zeros(ni, bool)
    touched[np.unique(labi[reach & (ink8 > 0)])] = True
    touched[0] = False
    keep = touched[labi]
    return keep, {"oval_seed": oval > 0, "thick": thick > 0, "diagram": diagram > 0}


def clean_page(bgr: np.ndarray, ink: np.ndarray, stroke_width: float = 2.0, p: CleanParams | None = None) -> tuple[np.ndarray, np.ndarray, dict]:
    """(page with only the diagram ink on white paper, keep mask, debug layers)."""
    keep, dbg = keep_mask(ink, bgr, stroke_width, p)
    out = np.full_like(bgr, 255)
    out[keep] = bgr[keep]
    return out, keep, dbg


def debug_overlay(bgr: np.ndarray, dbg: dict) -> np.ndarray:
    vis = bgr.copy()
    vis[dbg["diagram"]] = (0.55 * vis[dbg["diagram"]] + np.array([0, 0, 115])).astype(np.uint8)
    vis[dbg["oval_seed"]] = (0, 0, 255)
    vis[dbg["thick"]] = (0, 200, 0)
    return vis
