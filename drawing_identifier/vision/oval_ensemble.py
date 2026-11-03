"""Ensemble oval ("cut") detector: many diverse detection traces, one consensus.

Ported from the author's ``detect_ovals_spiral_v6`` (SPIRAL-style
sequential / parallel / aggregative detection), fixed and adapted to this
package:

* **sequential** - one *trace* = binarise -> bridge or fill -> ellipse fit,
  under one configuration;
* **parallel** - two trace *families* with decorrelated errors run over the
  same page: ``raw`` (flat-field illumination, adaptive / Sauvola / top-hat
  binarisers, tone-adaptive kernel shapes) and ``clean`` (one high-threshold
  mask, no opening, structural diversity only);
* **aggregative** - all candidates are clustered by ellipse IoU; a cluster's
  confidence combines how many traces found it, how well they fit an ellipse
  and whether *both* families agree; a solo path keeps a single very good fit
  with strong boundary ink support.

The aggregation is an :class:`AggPolicy` (a flat parameter vector) that
:func:`tune_policy` random-searches against ground-truth ellipses - the
synthetic generator provides those for free (``drawid tune-ovals``).
"""

from __future__ import annotations

import json
import math
import random
from dataclasses import asdict, dataclass, field
from typing import Callable

import cv2
import numpy as np

TONE_DARK, TONE_MID, TONE_BRIGHT = "dark", "mid", "bright"


# ---------------------------------------------------------------- low level
def _odd(x: float) -> int:
    x = max(3, int(round(x)))
    return x if x % 2 == 1 else x + 1


def fill_holes(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    ff = mask.copy()
    cv2.floodFill(ff, np.zeros((h + 2, w + 2), np.uint8), (0, 0), 255)
    return cv2.bitwise_or(mask, cv2.bitwise_not(ff))


def classify_tone(gray: np.ndarray, dark_below: int = 200, bright_at: int = 250) -> str:
    med = float(np.median(gray))
    return TONE_DARK if med < dark_below else (TONE_BRIGHT if med >= bright_at else TONE_MID)


def kernel(frac: float, diag: float, tone: str | None = None) -> np.ndarray:
    """Scale-relative structuring element; the page tone picks the shape only.

    dark pages binarise to thin strokes that a fat element would eat (CROSS),
    bright pages carry bold strokes that take the most aggressive one (RECT).
    """
    k = _odd(frac * diag)
    shape = cv2.MORPH_CROSS if tone == TONE_DARK else (cv2.MORPH_RECT if tone == TONE_BRIGHT else cv2.MORPH_ELLIPSE)
    return cv2.getStructuringElement(shape, (k, k))


def illum_flatfield(gray: np.ndarray, sigma_frac: float = 0.06) -> np.ndarray:
    g = gray.astype(np.float32)
    sigma = max(10.0, sigma_frac * max(gray.shape))
    bg = cv2.GaussianBlur(g, (0, 0), sigmaX=sigma, sigmaY=sigma)
    return np.clip((g / (bg + 1e-3)) * 128.0, 0, 255).astype(np.uint8)


def bin_adaptive(flat: np.ndarray, diag: float, block_frac: float = 0.027, C: int = 12) -> np.ndarray:
    return cv2.adaptiveThreshold(flat, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, _odd(block_frac * diag), C)


def bin_sauvola(flat: np.ndarray, diag: float, window_frac: float = 0.05, k: float = 0.2, R: float = 128.0) -> np.ndarray:
    img = flat.astype(np.float32)
    w = _odd(window_frac * diag)
    mean = cv2.boxFilter(img, -1, (w, w), normalize=True, borderType=cv2.BORDER_REFLECT)
    sq = cv2.boxFilter(img * img, -1, (w, w), normalize=True, borderType=cv2.BORDER_REFLECT)
    std = np.sqrt(np.maximum(sq - mean * mean, 0.0))
    return (img < mean * (1.0 + k * ((std / R) - 1.0))).astype(np.uint8) * 255


def bin_tophat(gray: np.ndarray, diag: float, tone: str, kernel_frac: float = 0.02) -> np.ndarray:
    bg = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel(kernel_frac, diag, tone))
    _, bw = cv2.threshold(cv2.subtract(bg, gray), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return bw


def _drop_specks(bw: np.ndarray, min_area: int) -> np.ndarray:
    n, lab, stats, _ = cv2.connectedComponentsWithStats(bw, 8)
    keep = stats[:, cv2.CC_STAT_AREA] >= min_area
    keep[0] = False
    return (keep[lab] * 255).astype(np.uint8)


def _border(bw: np.ndarray, m: int) -> np.ndarray:
    bw[:m] = 0
    bw[-m:] = 0
    bw[:, :m] = 0
    bw[:, -m:] = 0
    return bw


def postprocess_raw(bw: np.ndarray, diag: float, tone: str, min_blob_frac: float = 3e-5) -> np.ndarray:
    """Grid-suppression opening (CROSS on dark pages), speck removal, wide border mask."""
    open_k = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3)) if tone == TONE_DARK else np.ones((3, 3), np.uint8)
    bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, open_k)
    return _border(_drop_specks(bw, max(25, int(min_blob_frac * bw.size))), max(6, int(0.015 * diag)))


def binarize_clean(gray: np.ndarray, thresh: int = 200) -> np.ndarray:
    _, bw = cv2.threshold(gray, thresh, 255, cv2.THRESH_BINARY_INV)
    return bw


def postprocess_clean(bw: np.ndarray, diag: float, min_blob_frac: float = 2e-5) -> np.ndarray:
    """No opening (protects ~2 px strokes), speck removal, narrow border mask."""
    return _border(_drop_specks(bw, max(16, int(min_blob_frac * bw.size))), max(4, int(0.008 * diag)))


# ---------------------------------------------------------------- traces
@dataclass
class Trace:
    name: str
    family: str  # "raw" | "clean"
    make_bw: Callable[[np.ndarray, float, str], np.ndarray]
    mode: str = "bridge"  # "bridge" | "fillblob"
    bridge_frac: float = 0.005
    close_frac: float = 0.009
    inlier_thresh: float = 0.50
    min_area_frac: float = 0.0008
    max_area_frac: float = 0.35
    max_aspect: float = 4.0
    hole_frac_min: float = 0.06


def raw_family() -> list[Trace]:
    ff = illum_flatfield

    def mk(fn):
        return lambda g, d, t: postprocess_raw(fn(g, d, t), d, t)

    return [
        Trace("raw:strict-adaptive", "raw", mk(lambda g, d, t: bin_adaptive(ff(g), d, C=12)), "bridge", bridge_frac=0.007, inlier_thresh=0.62, max_aspect=3.0, min_area_frac=0.0013),
        Trace("raw:permissive-adapt", "raw", mk(lambda g, d, t: bin_adaptive(ff(g), d, C=7)), "bridge", bridge_frac=0.010, inlier_thresh=0.48, max_aspect=3.2, min_area_frac=0.0013),
        Trace("raw:faint-bigbridge", "raw", mk(lambda g, d, t: bin_adaptive(ff(g), d, C=8)), "bridge", bridge_frac=0.015, close_frac=0.020, inlier_thresh=0.44, max_aspect=3.2, min_area_frac=0.0013),
        Trace("raw:sauvola", "raw", mk(lambda g, d, t: bin_sauvola(ff(g), d)), "bridge", bridge_frac=0.009, inlier_thresh=0.50, max_aspect=3.0, min_area_frac=0.0013),
        Trace("raw:fill-mid", "raw", mk(lambda g, d, t: bin_adaptive(ff(g), d, C=9)), "fillblob", close_frac=0.016, inlier_thresh=0.40, max_aspect=7.5, min_area_frac=0.0013),
        Trace("raw:fill-tophat", "raw", mk(lambda g, d, t: bin_tophat(g, d, t)), "fillblob", close_frac=0.012, inlier_thresh=0.42, max_aspect=7.0, min_area_frac=0.0013),
    ]


def clean_family(thresh: int = 200) -> list[Trace]:
    def mk():
        return lambda g, d, t: postprocess_clean(binarize_clean(g, thresh), d)

    return [
        Trace("clean:bridge-tight", "clean", mk(), "bridge", bridge_frac=0.003, close_frac=0.006, inlier_thresh=0.55),
        Trace("clean:bridge-mid", "clean", mk(), "bridge", bridge_frac=0.006, close_frac=0.011, inlier_thresh=0.50),
        Trace("clean:bridge-wide", "clean", mk(), "bridge", bridge_frac=0.010, close_frac=0.017, inlier_thresh=0.45),
        Trace("clean:fill-tight", "clean", mk(), "fillblob", close_frac=0.005, inlier_thresh=0.45),
        Trace("clean:fill-mid", "clean", mk(), "fillblob", close_frac=0.009, inlier_thresh=0.42),
        Trace("clean:fill-wide", "clean", mk(), "fillblob", close_frac=0.014, inlier_thresh=0.40, max_aspect=5.0),
    ]


@dataclass
class Candidate:
    center: tuple[float, float]
    axes: tuple[float, float]  # full axis lengths (cv2.fitEllipse convention)
    angle: float
    area: float
    inlier: float
    cover: float
    aspect: float
    trace: str
    family: str


def _ellipse_inlier(contour: np.ndarray, tol: float = 0.18):
    if len(contour) < 15:
        return None
    (cx, cy), (MA, ma), ang = cv2.fitEllipse(contour)
    if MA <= 0 or ma <= 0:
        return None
    th = math.radians(ang)
    ct, st = math.cos(th), math.sin(th)
    pts = contour.reshape(-1, 2).astype(np.float32)
    dx, dy = pts[:, 0] - cx, pts[:, 1] - cy
    xr, yr = ct * dx + st * dy, -st * dx + ct * dy
    r = np.sqrt((xr / (MA / 2)) ** 2 + (yr / (ma / 2)) ** 2)
    return (cx, cy), (MA, ma), ang, float(np.mean(np.abs(r - 1.0) < tol))


def _cand(c, lo, hi, tr: Trace, cover_lo, cover_hi) -> Candidate | None:
    a = cv2.contourArea(c)
    if a < lo or a > hi:
        return None
    res = _ellipse_inlier(c)
    if res is None:
        return None
    (cx, cy), (MA, ma), ang, inl = res
    aspect = max(MA, ma) / max(min(MA, ma), 1.0)
    ell = math.pi * (MA / 2) * (ma / 2)
    cover = a / ell if ell > 0 else 0.0
    if inl >= tr.inlier_thresh and aspect <= tr.max_aspect and cover_lo < cover < cover_hi:
        return Candidate((float(cx), float(cy)), (float(MA), float(ma)), float(ang), float(a), inl, float(cover), float(aspect), tr.name, tr.family)
    return None


def run_trace(gray: np.ndarray, diag: float, tone: str, tr: Trace) -> list[Candidate]:
    area_tot = gray.shape[0] * gray.shape[1]
    lo, hi = tr.min_area_frac * area_tot, tr.max_area_frac * area_tot
    bw = tr.make_bw(gray, diag, tone)

    def kget(f):
        return kernel(f, diag, tone) if tr.family == "raw" else cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (_odd(f * diag),) * 2)

    out: list[Candidate] = []
    if tr.mode == "bridge":
        bridged = cv2.dilate(bw, kget(tr.bridge_frac), iterations=1)
        bridged = cv2.morphologyEx(bridged, cv2.MORPH_CLOSE, kget(tr.close_frac))
        cnts, _ = cv2.findContours(bridged, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        for c in cnts:
            cand = _cand(c, lo, hi, tr, 0.4, 1.8)
            if cand:
                out.append(cand)
    else:
        closed = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, kget(tr.close_frac))
        filled = cv2.morphologyEx(fill_holes(closed), cv2.MORPH_OPEN, kernel(tr.close_frac * 0.6, diag, None))
        n, lab, stats, _ = cv2.connectedComponentsWithStats(filled, 8)
        ink = closed > 0
        for i in range(1, n):
            x, y, w, h, area = stats[i]
            if area < lo:
                continue
            comp = lab[y : y + h, x : x + w] == i
            hole_frac = 1.0 - np.count_nonzero(ink[y : y + h, x : x + w] & comp) / max(int(area), 1)
            if hole_frac < tr.hole_frac_min:
                continue  # a solid blob (ink stain, punch hole) is not a cut
            cs, _ = cv2.findContours(comp.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            if cs:
                cand = _cand(max(cs, key=cv2.contourArea) + np.array([x, y]), lo, hi, tr, 0.55, 1.4)
                if cand:
                    out.append(cand)
    return out


# ---------------------------------------------------------------- aggregation
@dataclass
class AggPolicy:
    w_agree: float = 0.45
    w_fit: float = 0.45
    family_bonus: float = 0.15  # both families saw it -> extra confidence
    conf_thr: float = 0.42
    min_agree: int = 2
    agree_sat: int = 3
    overlap_min: float = 0.45
    bsupport_min: float = 0.15
    solo_inlier: float = 0.62
    solo_bsupport: float = 0.55

    FIELDS = ("w_agree", "w_fit", "family_bonus", "conf_thr", "min_agree", "agree_sat", "overlap_min", "bsupport_min", "solo_inlier", "solo_bsupport")

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def load(cls, path: str) -> "AggPolicy":
        with open(path, encoding="utf-8") as f:
            return cls(**{k: v for k, v in json.load(f).items() if k in cls.FIELDS})


SEARCH_SPACE = dict(
    w_agree=(0.2, 0.6),
    w_fit=(0.2, 0.6),
    family_bonus=(0.0, 0.3),
    conf_thr=(0.30, 0.60),
    min_agree=(1, 3),
    agree_sat=(2, 4),
    overlap_min=(0.30, 0.60),
    bsupport_min=(0.05, 0.30),
    solo_inlier=(0.50, 0.80),
    solo_bsupport=(0.40, 0.75),
)


@dataclass
class OvalDetection:
    center: tuple[float, float]
    axes: tuple[float, float]
    angle: float
    area: float
    confidence: float
    agreement: int
    families: list[str]
    traces: list[str]
    external: bool = True
    mean_inlier: float = 0.0
    bsupport: float = 0.0

    def polygon(self, n: int = 72, offset=(0.0, 0.0)) -> list[tuple[float, float]]:
        (cx, cy), (MA, ma) = self.center, self.axes
        th = math.radians(self.angle)
        ct, st = math.cos(th), math.sin(th)
        pts = []
        for i in range(n):
            t = 2 * math.pi * i / n
            ex, ey = MA / 2 * math.cos(t), ma / 2 * math.sin(t)
            pts.append((round(cx + ex * ct - ey * st + offset[0], 1), round(cy + ex * st + ey * ct + offset[1], 1)))
        return pts


def _raster(center, axes, angle, ox, oy, sc, grid):
    m = np.zeros((grid, grid), np.uint8)
    cv2.ellipse(m, (int((center[0] - ox) * sc), int((center[1] - oy) * sc)), (max(1, int(axes[0] / 2 * sc)), max(1, int(axes[1] / 2 * sc))), angle, 0, 360, 255, -1)
    return m > 0


def ellipse_iou(a, b, grid: int = 64) -> float:
    """IoU of two ellipses given as objects with center/axes/angle."""
    rad = max(max(a.axes), max(b.axes))
    ox = min(a.center[0], b.center[0]) - rad
    oy = min(a.center[1], b.center[1]) - rad
    span = max(abs(a.center[0] - b.center[0]), abs(a.center[1] - b.center[1])) + 2 * rad
    sc = grid / (span + 1e-3)
    A = _raster(a.center, a.axes, a.angle, ox, oy, sc, grid)
    B = _raster(b.center, b.axes, b.angle, ox, oy, sc, grid)
    return float(np.logical_and(A, B).sum() / max(np.logical_or(A, B).sum(), 1))


def _point_in_ellipse(p, det, scale: float = 1.0) -> bool:
    th = math.radians(det.angle)
    ct, st = math.cos(th), math.sin(th)
    dx, dy = p[0] - det.center[0], p[1] - det.center[1]
    xr, yr = ct * dx + st * dy, -st * dx + ct * dy
    return (xr / (scale * det.axes[0] / 2)) ** 2 + (yr / (scale * det.axes[1] / 2)) ** 2 < 1.0


def boundary_support(center, axes, angle, ink_dil: np.ndarray, n: int = 180) -> float:
    """Fraction of the ellipse perimeter lying on (dilated) ink."""
    (cx, cy), (MA, ma) = center, axes
    th = math.radians(angle)
    ct, st = math.cos(th), math.sin(th)
    t = np.linspace(0, 2 * math.pi, n, endpoint=False)
    ex, ey = MA / 2 * np.cos(t), ma / 2 * np.sin(t)
    xs = np.round(cx + ex * ct - ey * st).astype(int)
    ys = np.round(cy + ex * st + ey * ct).astype(int)
    H, W = ink_dil.shape
    ok = (xs >= 0) & (xs < W) & (ys >= 0) & (ys < H)
    return float(ink_dil[ys[ok], xs[ok]].astype(bool).sum() / n)


def aggregate(cands: list[Candidate], pol: AggPolicy, ink_dil: np.ndarray | None = None) -> list[OvalDetection]:
    cands = sorted(cands, key=lambda c: -c.inlier)
    clusters: list[list[Candidate]] = []
    for c in cands:
        for cl in clusters:
            if ellipse_iou(c, cl[0]) >= pol.overlap_min:
                cl.append(c)
                break
        else:
            clusters.append([c])
    dets: list[OvalDetection] = []
    for cl in clusters:
        traces = sorted({m.trace for m in cl})
        families = sorted({m.family for m in cl})
        agree = len(traces)
        mean_inl = float(np.mean([m.inlier for m in cl]))
        conf = pol.w_agree * min(1.0, agree / max(pol.agree_sat, 1)) + pol.w_fit * mean_inl + (pol.family_bonus if len(families) > 1 else 0.0)
        rep = max(cl, key=lambda m: m.inlier)
        bs = 0.0
        if ink_dil is not None:
            # a bridged blob's contours run just outside / inside the stroke; the cluster's median
            # axes sit on the stroke itself, so take the better of the two
            med = (float(np.median([m.axes[0] for m in cl])), float(np.median([m.axes[1] for m in cl])))
            bs = max(boundary_support(rep.center, rep.axes, rep.angle, ink_dil), boundary_support(rep.center, med, rep.angle, ink_dil))
        consensus_ok = agree >= pol.min_agree and conf >= pol.conf_thr
        solo_ok = rep.inlier >= pol.solo_inlier and bs >= pol.solo_bsupport
        if not (consensus_ok or solo_ok):
            continue
        if ink_dil is not None and bs < pol.bsupport_min:
            continue
        dets.append(OvalDetection(rep.center, rep.axes, rep.angle, rep.area, round(min(conf, 1.0), 3), agree, families, traces, mean_inlier=round(mean_inl, 3), bsupport=round(bs, 3)))
    dets.sort(key=lambda d: -d.area)
    externals: list[OvalDetection] = []
    for d in dets:
        if any(_point_in_ellipse(d.center, e, scale=0.75) for e in externals):
            d.external = False
        else:
            externals.append(d)
    return dets


# ---------------------------------------------------------------- page level
@dataclass
class EnsembleResult:
    detections: list[OvalDetection]
    candidates: list[Candidate] = field(default_factory=list)
    per_trace: dict[str, int] = field(default_factory=dict)
    tone: str = TONE_MID


def to_gray(img: np.ndarray) -> np.ndarray:
    return img if img.ndim == 2 else cv2.cvtColor(img[:, :, :3], cv2.COLOR_BGR2GRAY)


def collect_candidates(gray: np.ndarray, families: str = "both") -> tuple[list[Candidate], dict[str, int], np.ndarray, str]:
    """All traces' candidates plus the shared ink-support mask."""
    h, w = gray.shape
    diag = float(np.hypot(w, h))
    tone = classify_tone(gray)
    traces: list[Trace] = []
    if families in ("raw", "both"):
        traces += raw_family()
    if families in ("clean", "both"):
        traces += clean_family()
    cands: list[Candidate] = []
    per: dict[str, int] = {}
    for tr in traces:
        cs = run_trace(gray, diag, tone, tr)
        per[tr.name] = len(cs)
        cands.extend(cs)
    # ink support: union of both families' notion of ink, one reference for every candidate
    ink = cv2.bitwise_or(postprocess_clean(binarize_clean(gray), diag), postprocess_raw(bin_adaptive(illum_flatfield(gray), diag, C=10), diag, tone))
    ink_dil = cv2.dilate((ink > 0).astype(np.uint8), np.ones((7, 7), np.uint8))
    return cands, per, ink_dil, tone


def detect_ovals(img: np.ndarray, policy: AggPolicy | None = None, families: str = "both") -> EnsembleResult:
    """Run every trace on ``img`` (BGR or grey, already cropped to the page) and aggregate."""
    gray = to_gray(img)
    cands, per, ink_dil, tone = collect_candidates(gray, families)
    return EnsembleResult(aggregate(cands, policy or AggPolicy(), ink_dil), cands, per, tone)


# ---------------------------------------------------------------- tuning
def reward(dets: list[OvalDetection], gt: list[dict], iou_match: float = 0.5, externals_only: bool = False) -> tuple[float, dict]:
    """sum(IoU of matches) - 0.5 FP - 0.5 FN. ``gt``: [{"center":[x,y],"axes":[A,a],"angle":deg}]"""

    class _G:
        def __init__(self, g):
            self.center, self.axes, self.angle = tuple(g["center"]), tuple(g["axes"]), float(g.get("angle", 0.0))

    gts = [_G(g) for g in gt]
    pool = [d for d in dets if d.external] if externals_only else list(dets)
    used: set[int] = set()
    tp, iou_sum = 0, 0.0
    for d in pool:
        best_j, best = -1, 0.0
        for j, g in enumerate(gts):
            if j not in used:
                i = ellipse_iou(d, g)
                if i > best:
                    best, best_j = i, j
        if best_j >= 0 and best >= iou_match:
            used.add(best_j)
            tp += 1
            iou_sum += best
    fp, fn = len(pool) - tp, len(gts) - tp
    return iou_sum - 0.5 * fp - 0.5 * fn, dict(tp=tp, fp=fp, fn=fn, mean_iou=iou_sum / tp if tp else 0.0)


def sample_policy(rng: random.Random) -> AggPolicy:
    kw = {}
    for f in AggPolicy.FIELDS:
        lo, hi = SEARCH_SPACE[f]
        kw[f] = rng.randint(int(lo), int(hi)) if f in ("min_agree", "agree_sat") else rng.uniform(lo, hi)
    return AggPolicy(**kw)


def tune_policy(pages: list[tuple[np.ndarray, list[dict]]], trials: int = 200, seed: int = 0, families: str = "both", progress: bool = True) -> tuple[AggPolicy, float, dict]:
    """Random-search the aggregation policy. Detection is policy-independent, so
    candidates are computed once per page and only the aggregation is re-run."""
    cache = []
    for img, gt in pages:
        cands, _, ink_dil, _ = collect_candidates(to_gray(img), families)
        cache.append((cands, ink_dil, gt))
    rng = random.Random(seed)
    best_pol, best_r, best_stats = AggPolicy(), -1e9, {}
    for trial in range(trials):
        pol = AggPolicy() if trial == 0 else sample_policy(rng)
        total, agg = 0.0, dict(tp=0, fp=0, fn=0)
        for cands, ink_dil, gt in cache:
            r, st = reward(aggregate(list(cands), pol, ink_dil), gt)
            total += r
            for k in agg:
                agg[k] += st[k]
        if total > best_r:
            best_pol, best_r, best_stats = pol, total, agg
            if progress:
                print(f"  trial {trial:4d}: reward {total:8.3f}  tp={agg['tp']} fp={agg['fp']} fn={agg['fn']}  <- best", flush=True)
    return best_pol, best_r, best_stats


CUT_TYPES = ("ellipse", "circle", "rounded_rectangle", "rectangle", "polygon", "diamond", "triangle")


def ellipse_gt_from_graph(graph, types: tuple[str, ...] = CUT_TYPES) -> list[dict]:
    """Ground-truth ellipses (center/axes/angle) fitted to a DiagramGraph's closed outlines ("cuts")."""
    out = []
    for s in graph.shapes:
        if s.type.value in types and len(s.polygon) >= 5:
            (cx, cy), (MA, ma), ang = cv2.fitEllipse(np.asarray(s.polygon, np.float32))
            out.append({"center": [cx, cy], "axes": [MA, ma], "angle": ang})
    return out
