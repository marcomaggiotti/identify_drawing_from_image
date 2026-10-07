"""Group the remaining ink (after removing outlines and connectors) into text items."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class TextBlob:
    bbox: tuple[int, int, int, int]  # x0, y0, x1, y1
    ink_pixels: int
    n_components: int


def estimate_text_height(text_mask: np.ndarray, sw: float) -> float:
    n, _, stats, _ = cv2.connectedComponentsWithStats(text_mask.astype(np.uint8), connectivity=8)
    if n <= 1:
        return max(8.0, 6 * sw)
    hs = stats[1:, cv2.CC_STAT_HEIGHT].astype(float)
    hs = hs[(hs >= 2 * sw) & (hs <= 40 * sw)]
    if len(hs) == 0:
        return max(8.0, 6 * sw)
    return float(np.clip(np.percentile(hs, 60), 3 * sw, 30 * sw))


def group_text(text_mask: np.ndarray, barrier: np.ndarray | None, sw: float, text_h: float | None = None) -> tuple[list[TextBlob], float]:
    """Merge letters into words/short lines without crossing shape outlines (``barrier``)."""
    th = text_h or estimate_text_height(text_mask, sw)
    m = text_mask.astype(np.uint8)
    n0, lab0, st0, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    # drop specks that are not dots of i/j or punctuation next to letters
    keep = st0[:, cv2.CC_STAT_AREA] >= max(3, int(0.6 * sw * sw))
    keep[0] = False
    m = keep[lab0].astype(np.uint8)

    kx = max(3, int(round(1.3 * th)))
    ky = max(1, int(round(0.35 * th)))
    smear = cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_RECT, (kx, ky)))
    if barrier is not None:
        smear[barrier] = 0
        smear |= m  # never cut letters themselves
    n, lab, stats, _ = cv2.connectedComponentsWithStats(smear, connectivity=8)
    blobs: list[TextBlob] = []
    min_ink = max(6, int(0.8 * sw * th))
    mb = m.astype(bool)
    for i in range(1, n):
        bx, by, bw, bh, _ = stats[i]
        region = (lab[by : by + bh, bx : bx + bw] == i) & mb[by : by + bh, bx : bx + bw]
        cnt = int(region.sum())
        if cnt < min_ink:
            continue
        ys, xs = np.nonzero(region)
        x0, y0, x1, y1 = int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1
        if (x1 - x0) < 0.4 * th and (y1 - y0) < 0.4 * th:
            continue
        sub_n = cv2.connectedComponents(region[y0:y1, x0:x1].astype(np.uint8), connectivity=8)[0] - 1
        blobs.append(TextBlob((x0 + bx, y0 + by, x1 + bx, y1 + by), cnt, sub_n))
    return blobs, th
