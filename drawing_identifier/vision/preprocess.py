"""Page extraction, orientation, illumination flattening and ink binarisation.

Designed for scans/microfilm of notebooks: bright page on a dark frame,
two-page spreads with a gutter, ruled paper, stains, pages rotated by 90 degrees.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np
from PIL import Image, ImageOps
from skimage.morphology import skeletonize

from .geometry import disk, odd

ROTATIONS = {
    0: None,
    90: cv2.ROTATE_90_CLOCKWISE,
    180: cv2.ROTATE_180,
    270: cv2.ROTATE_90_COUNTERCLOCKWISE,
}


def load_image(path: str) -> np.ndarray:
    """Load any PIL-readable file (jpg/png/webp/tiff...) as a BGR uint8 array."""
    with Image.open(path) as im:
        im = ImageOps.exif_transpose(im).convert("RGB")
        arr = np.asarray(im)
    return np.ascontiguousarray(arr[:, :, ::-1])


def rotate(img: np.ndarray, degrees_cw: int) -> np.ndarray:
    code = ROTATIONS[int(degrees_cw) % 360]
    return img if code is None else cv2.rotate(img, code)


def ink_gray(bgr: np.ndarray) -> np.ndarray:
    """Grey image where ink is dark.

    Uses the max channel (HSV value): coloured ruled lines (light blue, red margins)
    stay bright, while black/brown ink stays dark.
    """
    if bgr.ndim == 2:
        return bgr.copy()
    return bgr.max(axis=2)


def find_page(gray: np.ndarray) -> tuple[int, int, int, int] | None:
    """Bounding box (x, y, w, h) of the bright page on a dark scanner frame, or None."""
    h, w = gray.shape
    s = 600.0 / max(h, w)
    small = cv2.resize(gray, (max(1, int(w * s)), max(1, int(h * s))), interpolation=cv2.INTER_AREA)
    blur = cv2.GaussianBlur(small, (0, 0), 3)
    thr, bright = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    bright = cv2.morphologyEx(bright, cv2.MORPH_OPEN, disk(4))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(bright, 8)
    if n <= 1:
        return None
    i = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    x, y, bw, bh, area = stats[i]
    frac = (bw * bh) / float(small.shape[0] * small.shape[1])
    inside = blur[lab == i].mean()
    outside = blur[lab != i].mean() if (lab != i).any() else inside
    if not (0.12 < frac < 0.93) or inside - outside < 40:
        return None
    inset = 0.006 * max(bw, bh)
    x0 = (x + inset) / s
    y0 = (y + inset) / s
    x1 = (x + bw - inset) / s
    y1 = (y + bh - inset) / s
    return int(x0), int(y0), int(x1 - x0), int(y1 - y0)


def flatten_background(gray: np.ndarray) -> np.ndarray:
    """Divide by an estimate of the paper background (removes stains, shading, vignetting)."""
    h, w = gray.shape
    k = odd(max(15, min(h, w) / 45))
    bg = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, disk(k // 2))
    bg = cv2.medianBlur(bg, min(k, 255))
    bg = np.maximum(bg.astype(np.float32), 1.0)
    norm = np.clip(gray.astype(np.float32) * 255.0 / bg, 0, 255)
    return norm.astype(np.uint8)


def binarize(norm: np.ndarray) -> np.ndarray:
    """Ink mask (bool) from a background-flattened grey image."""
    otsu, _ = cv2.threshold(norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    t = float(np.clip(otsu, 140, 215))
    ink = norm < t
    # hysteresis: keep faint pixels connected to clearly dark ones
    strong = norm < t * 0.8
    n, lab = cv2.connectedComponents(ink.astype(np.uint8), connectivity=8)
    keep = np.zeros(n, bool)
    keep[np.unique(lab[strong])] = True
    keep[0] = False
    return keep[lab]


def remove_specks(ink: np.ndarray, min_area: int) -> np.ndarray:
    n, lab, stats, _ = cv2.connectedComponentsWithStats(ink.astype(np.uint8), connectivity=8)
    keep = stats[:, cv2.CC_STAT_AREA] >= min_area
    keep[0] = False
    return keep[lab]


def remove_ruled_lines(ink: np.ndarray, frac: float = 0.33, max_thickness: float = 4.0) -> tuple[np.ndarray, np.ndarray]:
    """Remove long, perfectly straight, thin horizontal/vertical rules (ruled paper, gutters, frames).

    Heavy hand-drawn lines (thicker than ``max_thickness``) are kept, and strokes that
    cross a removed rule are re-joined so shapes drawn over ruled paper stay closed.
    """
    h, w = ink.shape
    u8 = ink.astype(np.uint8)
    out = ink.copy()
    rules = np.zeros_like(ink)
    t = max(1, int(round(max_thickness)))
    for horizontal in (True, False):
        k = (max(10, int(w * frac)), 1) if horizontal else (1, max(10, int(h * frac)))
        line = cv2.morphologyEx(u8, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, k)) > 0
        if not line.any():
            continue
        # thickness test: a thin rule disappears when opened across its width
        across = (1, t + 1) if horizontal else (t + 1, 1)
        thick = cv2.morphologyEx(u8, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, across)) > 0
        line &= ~thick
        line = cv2.dilate(line.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
        line &= ink
        removed = out & ~line
        bridge = (1, 2 * t + 5) if horizontal else (2 * t + 5, 1)
        rejoin = cv2.morphologyEx(removed.astype(np.uint8), cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, bridge)) > 0
        out = removed | (rejoin & line)
        rules |= line & ~out
    return out, rules


def stroke_width(ink: np.ndarray) -> tuple[float, float]:
    """(median stroke width, 90th percentile stroke width) in pixels."""
    if ink.sum() < 20:
        return 2.0, 3.0
    dist = cv2.distanceTransform(ink.astype(np.uint8), cv2.DIST_L2, 5)
    skel = skeletonize(ink)
    vals = dist[skel]
    if len(vals) == 0:
        return 2.0, 3.0
    med = max(1.5, 2.0 * float(np.median(vals)) - 1.0)
    p90 = max(med, 2.0 * float(np.percentile(vals, 90)) - 1.0)
    return med, p90


def _small_components(ink: np.ndarray, max_dim: int) -> np.ndarray:
    n, lab, stats, _ = cv2.connectedComponentsWithStats(ink.astype(np.uint8), connectivity=8)
    keep = (stats[:, cv2.CC_STAT_WIDTH] < max_dim) & (stats[:, cv2.CC_STAT_HEIGHT] < max_dim)
    keep[0] = False
    return keep[lab]


def text_direction_ratio(ink: np.ndarray, sw: float) -> float:
    """Ink mass of vertically elongated word blobs / horizontally elongated ones.

    Words written left-to-right form wide blobs once letters are merged, so a
    ratio well above 1 means the page is lying on its side.
    """
    text = _small_components(ink, max_dim=int(max(25, 18 * sw)))
    r = max(2, int(round(2.5 * sw)))
    blobs = cv2.dilate(text.astype(np.uint8), disk(r))
    n, _, st, _ = cv2.connectedComponentsWithStats(blobs, connectivity=8)
    if n <= 1:
        return 1.0
    w = st[1:, cv2.CC_STAT_WIDTH].astype(float)
    h = st[1:, cv2.CC_STAT_HEIGHT].astype(float)
    a = st[1:, cv2.CC_STAT_AREA].astype(float)
    hor = (a * (w > 2 * h)).sum()
    ver = (a * (h > 2 * w)).sum()
    return float((ver + 1.0) / (hor + 1.0))


def estimate_rotation(ink: np.ndarray, sw: float) -> tuple[int, float]:
    """Classical orientation guess: (clockwise rotation to apply, confidence 0..1).

    Only distinguishes horizontal from vertical writing; telling 90 from 270 (or
    0 from 180) needs reading the text, which the preprocessing agent asks the VLM
    for. Without a VLM a vertical page is turned 90 degrees clockwise, the usual
    microfilm convention (both rotated examples in examples/images need it).
    """
    ratio = text_direction_ratio(ink, sw)
    if ratio > 2.0:
        return 90, float(min(1.0, np.log(ratio) / np.log(20)))
    return 0, float(min(1.0, np.log(1.0 / max(ratio, 1e-6)) / np.log(20)))


@dataclass
class Preprocessed:
    image: np.ndarray  # BGR working image
    gray: np.ndarray  # background-flattened grey (ink dark)
    ink: np.ndarray  # bool ink mask, rules removed
    stroke_width: float
    heavy_width: float
    rotation: int
    crop: tuple[int, int, int, int] | None
    scale: float
    original_size: tuple[int, int]
    rules: np.ndarray | None = None
    notes: list[str] = field(default_factory=list)

    @property
    def size(self) -> tuple[int, int]:
        h, w = self.ink.shape
        return w, h


def preprocess(
    bgr: np.ndarray,
    max_side: int = 1800,
    crop_page: bool = True,
    rotation: int | None = None,
    remove_rules: bool = True,
) -> Preprocessed:
    """Full preprocessing. ``rotation=None`` means estimate it classically."""
    oh, ow = bgr.shape[:2]
    notes: list[str] = []
    gray0 = ink_gray(bgr)
    crop = find_page(gray0) if crop_page else None
    if crop is not None:
        x, y, w, h = crop
        bgr = bgr[y : y + h, x : x + w]
        notes.append(f"cropped page {crop} from {ow}x{oh} frame")
    h, w = bgr.shape[:2]
    scale = 1.0
    if max(h, w) > max_side:
        scale = max_side / float(max(h, w))
        bgr = cv2.resize(bgr, (int(round(w * scale)), int(round(h * scale))), interpolation=cv2.INTER_AREA)
    elif max(h, w) < 900:
        scale = 900 / float(max(h, w))
        bgr = cv2.resize(bgr, (int(round(w * scale)), int(round(h * scale))), interpolation=cv2.INTER_CUBIC)

    def _ink_of(img):
        g = flatten_background(ink_gray(img))
        ink = binarize(g)
        ink = remove_specks(ink, 6)
        rules = None
        if remove_rules:
            ink, rules = remove_ruled_lines(ink, max_thickness=max(2.0, 1.5 * stroke_width(ink)[0]))
        return g, ink, rules

    gray, ink, rules = _ink_of(bgr)
    sw, heavy = stroke_width(ink)
    if rotation is None:
        rotation, conf = estimate_rotation(ink, sw)
        if rotation:
            notes.append(f"estimated rotation {rotation} deg (confidence {conf:.2f})")
    if rotation:
        bgr = rotate(bgr, rotation)
        gray = rotate(gray, rotation)
        ink = rotate(ink.astype(np.uint8), rotation).astype(bool)
        if rules is not None:
            rules = rotate(rules.astype(np.uint8), rotation).astype(bool)
    ink = remove_specks(ink, max(6, int(0.8 * sw * sw)))
    return Preprocessed(
        image=bgr,
        gray=gray,
        ink=ink,
        stroke_width=sw,
        heavy_width=heavy,
        rotation=int(rotation or 0),
        crop=crop,
        scale=scale,
        original_size=(ow, oh),
        rules=rules,
        notes=notes,
    )
