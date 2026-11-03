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


_CRAFT = None


def craft_regions(bgr: np.ndarray, pad: int = 4) -> list[tuple[int, int, int, int]]:
    """Word boxes from the CRAFT detector (via EasyOCR, detection only).

    Morphology cannot split or join cursive reliably; CRAFT finds joined
    handwritten words. Thresholds are looser than EasyOCR's defaults because
    ink on aged paper is faint (from the author's find_text_regions_craft.py).
    """
    global _CRAFT
    if _CRAFT is None:
        import easyocr

        try:
            import torch

            gpu = torch.cuda.is_available()
        except Exception:
            gpu = False
        _CRAFT = easyocr.Reader(["en"], gpu=gpu, verbose=False)
    h, w = bgr.shape[:2]
    horizontal, free = _CRAFT.detect(bgr[:, :, ::-1], text_threshold=0.4, low_text=0.3, link_threshold=0.3, add_margin=0.15, width_ths=0.7)
    boxes = [(b[0], b[2], b[1], b[3]) for b in (horizontal[0] if horizontal else [])]
    for quad in free[0] if free else []:
        xs = [q[0] for q in quad]
        ys = [q[1] for q in quad]
        boxes.append((min(xs), min(ys), max(xs), max(ys)))
    out = []
    for x0, y0, x1, y1 in boxes:
        x0, y0 = max(0, int(x0) - pad), max(0, int(y0) - pad)
        x1, y1 = min(w, int(x1) + pad), min(h, int(y1) + pad)
        if x1 > x0 and y1 > y0:
            out.append((x0, y0, x1, y1))
    return out
