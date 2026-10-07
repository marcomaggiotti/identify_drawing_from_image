"""Closed-shape detection from an ink mask.

Idea: every closed hand-drawn curve encloses one or more *holes* (background
regions not connected to the page exterior).  Each hole is turned into a
region at the stroke centre line, fitted against primitives (ellipse,
rectangle, rounded rectangle, triangle, diamond, polygon) and classified.

Lines crossing a shape split its interior into several holes; adjacent sibling
holes are therefore also merged (up to ``max_merge``) and the union is kept
when it is a better primitive than its parts.  Overlapping shapes (Euler/Venn
style) come out naturally because one hole can belong to several unions.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field

import cv2
import numpy as np
from scipy import ndimage as ndi
from scipy.spatial import cKDTree
from skimage.morphology import skeletonize

from ..schema import ShapeType
from .geometry import disk, neighbour_count, odd


@dataclass
class ShapeFit:
    type: ShapeType
    score: float
    scores: dict[str, float]
    attributes: dict = field(default_factory=dict)


@dataclass
class ShapeCandidate:
    holes: frozenset[int]
    roi: tuple[int, int, int, int]  # x0, y0, x1, y1
    mask: np.ndarray  # bool region inside roi (stroke centre line)
    contour: np.ndarray  # (N, 2) float, full-image coordinates
    fit: ShapeFit
    area: float

    @property
    def type(self) -> ShapeType:
        return self.fit.type

    @property
    def bbox_xyxy(self) -> tuple[float, float, float, float]:
        c = self.contour
        return float(c[:, 0].min()), float(c[:, 1].min()), float(c[:, 0].max()), float(c[:, 1].max())


# ----------------------------------------------------------------- primitives
def _iou(a: np.ndarray, b: np.ndarray) -> float:
    inter = np.count_nonzero(a & b)
    union = np.count_nonzero(a | b)
    return inter / union if union else 0.0


def _draw(shape: tuple[int, int], fn) -> np.ndarray:
    m = np.zeros(shape, np.uint8)
    fn(m)
    return m.astype(bool)


def _rounded_rect_poly(cx, cy, w, h, angle_deg, r, n_arc=8) -> np.ndarray:
    r = max(0.0, min(r, w / 2, h / 2))
    pts = []
    corners = [(w / 2 - r, h / 2 - r, 0), (-w / 2 + r, h / 2 - r, 90), (-w / 2 + r, -h / 2 + r, 180), (w / 2 - r, -h / 2 + r, 270)]
    for ox, oy, a0 in corners:
        for k in range(n_arc + 1):
            a = math.radians(a0 + 90 * k / n_arc)
            pts.append((ox + r * math.cos(a), oy + r * math.sin(a)))
    pts = np.asarray(pts)
    t = math.radians(angle_deg)
    rot = np.array([[math.cos(t), -math.sin(t)], [math.sin(t), math.cos(t)]])
    return pts @ rot.T + np.array([cx, cy])


def fit_primitives(mask: np.ndarray) -> ShapeFit:
    """Classify a filled region mask by IoU against fitted primitives."""
    m = mask.astype(np.uint8)
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cnts:
        return ShapeFit(ShapeType.UNKNOWN, 0.0, {})
    c = max(cnts, key=cv2.contourArea)
    if len(c) < 6 or cv2.contourArea(c) < 20:
        return ShapeFit(ShapeType.UNKNOWN, 0.0, {})
    sh = mask.shape
    mb = mask.astype(bool)
    scores: dict[str, float] = {}
    attrs: dict = {}

    # ellipse from second moments of the filled region (robust to wobbly outlines)
    ys, xs = np.nonzero(mb)
    cx, cy = xs.mean(), ys.mean()
    cov = np.cov(np.stack([xs - cx, ys - cy]))
    evals, evecs = np.linalg.eigh(cov)
    evals = np.maximum(evals, 1e-6)
    a = 2.0 * math.sqrt(evals[1])
    b = 2.0 * math.sqrt(evals[0])
    ang = math.degrees(math.atan2(evecs[1, 1], evecs[0, 1]))
    ell = _draw(sh, lambda im: cv2.ellipse(im, (round(cx), round(cy)), (max(1, round(a)), max(1, round(b))), ang, 0, 360, 1, -1))
    scores["ellipse"] = _iou(mb, ell)
    attrs["axis_ratio"] = round(b / a, 3) if a > 0 else 1.0
    attrs["angle"] = round(ang, 1)

    # rectangles: orientation from the minimum-area rectangle, size from robust extents
    (rcx, rcy), (rw, rh), rang = cv2.minAreaRect(c)
    t = math.radians(rang)
    u = (xs - rcx) * math.cos(t) + (ys - rcy) * math.sin(t)
    v = -(xs - rcx) * math.sin(t) + (ys - rcy) * math.cos(t)
    u0, u1 = np.percentile(u, [0.5, 99.5])
    v0, v1 = np.percentile(v, [0.5, 99.5])
    W, H = (u1 - u0) + 1, (v1 - v0) + 1
    ccx = rcx + ((u0 + u1) / 2) * math.cos(t) - ((v0 + v1) / 2) * math.sin(t)
    ccy = rcy + ((u0 + u1) / 2) * math.sin(t) + ((v0 + v1) / 2) * math.cos(t)
    best_rr = 0.0
    for frac in (0.0, 0.15, 0.3):
        poly = _rounded_rect_poly(ccx, ccy, W, H, rang, frac * min(W, H))
        pm = _draw(sh, lambda im: cv2.fillPoly(im, [np.round(poly).astype(np.int32)], 1))
        iou = _iou(mb, pm)
        if frac == 0.0:
            scores["rectangle"] = iou
        else:
            best_rr = max(best_rr, iou)
    scores["rounded_rectangle"] = best_rr

    # triangle
    hull = cv2.convexHull(c)
    try:
        _, tri = cv2.minEnclosingTriangle(hull.astype(np.float32))
        if tri is not None:
            tm = _draw(sh, lambda im: cv2.fillPoly(im, [np.round(tri.reshape(-1, 2)).astype(np.int32)], 1))
            scores["triangle"] = _iou(mb, tm)
    except cv2.error:
        pass

    # diamond: vertices at the midpoints of the axis-aligned bounding box sides
    x0, x1 = xs.min(), xs.max() + 1
    y0, y1 = ys.min(), ys.max() + 1
    mx, my = (x0 + x1) / 2, (y0 + y1) / 2
    dpoly = np.array([[mx, y0], [x1, my], [mx, y1], [x0, my]])
    dm = _draw(sh, lambda im: cv2.fillPoly(im, [np.round(dpoly).astype(np.int32)], 1))
    scores["diamond"] = _iou(mb, dm)

    # generic polygon (5..8 corners)
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.025 * peri, True)
    k = len(approx)
    if 5 <= k <= 8:
        pm = _draw(sh, lambda im: cv2.fillPoly(im, [approx], 1))
        scores["polygon"] = _iou(mb, pm) - 0.06  # any blob fits a 5-8 gon; demand a clearly better fit
        attrs["vertices"] = k

    # small priors: prefer the simpler explanation on near-ties
    prior = {"ellipse": 0.012, "rectangle": 0.006, "rounded_rectangle": 0.0, "triangle": 0.0, "diamond": 0.0, "polygon": 0.0}
    if abs(((rang % 90) + 90) % 90 - 45) < 12:
        prior["diamond"] = 0.015  # a square standing on a corner reads as a diamond
    best = max(scores, key=lambda n: scores[n] + prior.get(n, 0))
    st = ShapeType(best)
    if st == ShapeType.ELLIPSE and attrs["axis_ratio"] >= 0.85:
        st = ShapeType.CIRCLE
    hull_area = cv2.contourArea(hull)
    attrs["solidity"] = round(float(cv2.contourArea(c) / hull_area), 3) if hull_area > 0 else 0.0
    return ShapeFit(st, float(scores[best]), {k2: round(v2, 4) for k2, v2 in scores.items()}, attrs)


def _cand_iou(a: "ShapeCandidate", b: "ShapeCandidate") -> float:
    ax0, ay0, ax1, ay1 = a.roi
    bx0, by0, bx1, by1 = b.roi
    x0, y0, x1, y1 = min(ax0, bx0), min(ay0, by0), max(ax1, bx1), max(ay1, by1)
    if min(ax1, bx1) <= max(ax0, bx0) or min(ay1, by1) <= max(ay0, by0):
        return 0.0
    ma = np.zeros((y1 - y0, x1 - x0), bool)
    mb = np.zeros_like(ma)
    ma[ay0 - y0 : ay1 - y0, ax0 - x0 : ax1 - x0] = a.mask
    mb[by0 - y0 : by1 - y0, bx0 - x0 : bx1 - x0] = b.mask
    return _iou(ma, mb)


# ------------------------------------------------------------------- detector
@dataclass
class ShapeDetectorParams:
    stroke_width: float = 2.0
    heavy_width: float = 4.0
    min_shape_scale: float = 7.0
    gap_close_scale: float = 2.0
    bridge_gap_scale: float = 8.0
    fit_threshold: float = 0.86
    min_region_score: float = 0.5
    keep_subregions: bool = False
    max_merge: int = 3
    max_holes: int = 400


def _bridge_radius(sw: float, heavy: float) -> int:
    return int(math.ceil(max(heavy, 2.5 * sw) / 2.0)) + 2


def _region_from_holes(hole_mask: np.ndarray, sw: float, heavy: float) -> np.ndarray:
    """Hole(s) -> filled region whose boundary runs along the stroke centre line.

    Dilating by about half the heavy-line width bridges lines that enter the
    shape (slits), filling recovers nested content, eroding brings the outline back.
    """
    r = _bridge_radius(sw, heavy)
    m = cv2.dilate(hole_mask.astype(np.uint8), disk(r))
    m = ndi.binary_fill_holes(m).astype(np.uint8)
    back = max(0, r - int(math.ceil(sw / 2.0)))
    if back:
        m = cv2.erode(m, disk(back))
    return m.astype(bool)


def _largest_component(mask: np.ndarray) -> np.ndarray:
    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    if n <= 1:
        return mask.astype(bool)
    i = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return lab == i


def _make_candidate(region: np.ndarray, roi, holes=frozenset()) -> ShapeCandidate | None:
    if not region.any():
        return None
    x0, y0 = roi[0], roi[1]
    fit = fit_primitives(region)
    cnts, _ = cv2.findContours(region.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    c = max(cnts, key=cv2.contourArea).reshape(-1, 2).astype(np.float64) + [x0, y0]
    return ShapeCandidate(frozenset(holes), tuple(int(v) for v in roi), region, c, fit, float(region.sum()))


def _component_candidates(closed: np.ndarray, sw: float, heavy: float, min_dim: float, thr: float) -> list[ShapeCandidate]:
    """Closed curves whose interior is cluttered (text touching the outline, many crossing lines).

    Each ink component is filled; thin protrusions (lines leaving the shape, text
    written across the outline) are removed by an opening, and what remains is fitted.
    """
    H, W = closed.shape
    n, lab, stats, _ = cv2.connectedComponentsWithStats(closed, connectivity=8)
    ro = int(math.ceil(max(heavy, 2.0 * sw))) + 1
    pad = ro + 3
    out = []
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if min(w, h) < 1.2 * min_dim:
            continue
        x0, y0, x1, y1 = max(0, x - pad), max(0, y - pad), min(W, x + w + pad), min(H, y + h + pad)
        comp = lab[y0:y1, x0:x1] == i
        filled = ndi.binary_fill_holes(comp)
        enclosed = filled.sum() - comp.sum()
        if enclosed < max(0.6 * comp.sum(), 0.5 * min_dim * min_dim):
            continue  # encloses little: a word or a line, not an outline
        opened = cv2.morphologyEx(filled.astype(np.uint8), cv2.MORPH_OPEN, disk(ro)).astype(bool)
        if not opened.any():
            continue
        region = _largest_component(opened)
        back = int(math.ceil(sw / 2.0))
        region = cv2.erode(region.astype(np.uint8), disk(back)).astype(bool) if back else region
        cand = _make_candidate(region, (x0, y0, x1, y1))
        if cand is not None and cand.fit.score >= thr:
            cand.fit.attributes["from_component"] = True
            out.append(cand)
    return out


def _boundary_overlap(inner: ShapeCandidate, outer: ShapeCandidate, tol: float) -> float:
    """Fraction of ``inner``'s outline lying on ``outer``'s outline."""
    oc = outer.contour.astype(np.float32).reshape(-1, 1, 2)
    pts = inner.contour[:: max(1, len(inner.contour) // 120)]
    near = sum(abs(cv2.pointPolygonTest(oc, (float(px), float(py)), True)) <= tol for px, py in pts)
    return near / max(1, len(pts))


def stroke_endpoints(ink: np.ndarray, sw: float, min_extent: float) -> list[tuple[tuple[int, int], tuple[float, float]]]:
    """Skeleton end points of long strokes with their outgoing direction (unit vector)."""
    n, lab, stats, _ = cv2.connectedComponentsWithStats(ink.astype(np.uint8), connectivity=8)
    big = np.zeros(n, bool)
    big[1:] = np.maximum(stats[1:, cv2.CC_STAT_WIDTH], stats[1:, cv2.CC_STAT_HEIGHT]) >= min_extent
    skel = skeletonize(big[lab])
    deg = neighbour_count(skel)
    ys, xs = np.nonzero(deg == 1)
    out = []
    walk = max(4, int(3 * sw))
    for x, y in zip(xs, ys):
        # walk back along the skeleton to estimate the stroke direction
        path = [(x, y)]
        prev = None
        cur = (x, y)
        for _ in range(walk):
            cx, cy = cur
            nxt = None
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    if dx == 0 and dy == 0:
                        continue
                    px, py = cx + dx, cy + dy
                    if 0 <= px < skel.shape[1] and 0 <= py < skel.shape[0] and skel[py, px] and (px, py) != prev and (px, py) not in path:
                        nxt = (px, py)
                        break
                if nxt:
                    break
            if nxt is None:
                break
            prev, cur = cur, nxt
            path.append(cur)
        if len(path) < 3:
            continue
        bx, by = path[-1]
        d = np.array([x - bx, y - by], float)
        nd = np.linalg.norm(d)
        if nd > 0:
            out.append(((int(x), int(y)), (float(d[0] / nd), float(d[1] / nd))))
    return out


def bridge_gaps(ink: np.ndarray, sw: float, max_gap: float, min_extent: float) -> np.ndarray:
    """Join pen lifts: connect pairs of stroke end points that point at each other."""
    eps = stroke_endpoints(ink, sw, min_extent)
    if len(eps) < 2:
        return ink
    pts = np.array([e[0] for e in eps], float)
    tree = cKDTree(pts)
    pairs = sorted(tree.query_pairs(max_gap), key=lambda ij: np.linalg.norm(pts[ij[0]] - pts[ij[1]]))
    used: set[int] = set()
    out = ink.astype(np.uint8).copy()
    t = max(1, int(round(sw)))
    for i, j in pairs:
        if i in used or j in used:
            continue
        v = pts[j] - pts[i]
        dist = np.linalg.norm(v)
        if dist < 1:
            continue
        v /= dist
        if np.dot(v, eps[i][1]) < 0.4 or np.dot(-v, eps[j][1]) < 0.4:
            continue
        cv2.line(out, eps[i][0], eps[j][0], 1, t)
        used.update((i, j))
    return out.astype(bool)


def detect_shapes(ink: np.ndarray, params: ShapeDetectorParams) -> list[ShapeCandidate]:
    H, W = ink.shape
    sw = max(1.0, params.stroke_width)
    heavy = max(sw, params.heavy_width)
    min_dim = params.min_shape_scale * sw
    if params.bridge_gap_scale > 0:
        ink = bridge_gaps(ink, sw, params.bridge_gap_scale * sw, min_extent=1.5 * min_dim)
    k = odd(params.gap_close_scale * sw, 3)
    closed = cv2.morphologyEx(ink.astype(np.uint8), cv2.MORPH_CLOSE, disk(k // 2))
    bg = (closed == 0).astype(np.uint8)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(bg, connectivity=4)

    min_area = 0.5 * min_dim * min_dim
    holes: dict[int, tuple[int, int, int, int]] = {}
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if x == 0 or y == 0 or x + w >= W or y + h >= H:
            continue  # touches the border: page exterior
        if min(w, h) < min_dim or area < min_area:
            continue
        if w > 0.92 * W and h > 0.92 * H:
            continue  # frame around the whole page
        holes[i] = (x, y, x + w, y + h)
    if len(holes) > params.max_holes:
        keep = sorted(holes, key=lambda i: -stats[i, cv2.CC_STAT_AREA])[: params.max_holes]
        holes = {i: holes[i] for i in keep}

    pad = _bridge_radius(sw, heavy) + 3

    def roi_of(ids) -> tuple[int, int, int, int]:
        x0 = min(holes[i][0] for i in ids) - pad
        y0 = min(holes[i][1] for i in ids) - pad
        x1 = max(holes[i][2] for i in ids) + pad
        y1 = max(holes[i][3] for i in ids) + pad
        return max(0, x0), max(0, y0), min(W, x1), min(H, y1)

    cache: dict[frozenset, ShapeCandidate | None] = {}

    def candidate(ids: frozenset) -> ShapeCandidate | None:
        if ids not in cache:
            x0, y0, x1, y1 = roi_of(ids)
            hm = np.isin(lab[y0:y1, x0:x1], list(ids))
            cache[ids] = _make_candidate(_region_from_holes(hm, sw, heavy), (x0, y0, x1, y1), ids)
        return cache[ids]

    singles = {i: candidate(frozenset([i])) for i in holes}
    singles = {i: c for i, c in singles.items() if c is not None}

    # adjacency between sibling holes separated by a thin stroke
    reach = int(math.ceil(max(sw, heavy))) + 2
    adj: dict[int, set[int]] = {i: set() for i in singles}
    for i in singles:
        x0, y0, x1, y1 = holes[i]
        X0, Y0, X1, Y1 = max(0, x0 - reach), max(0, y0 - reach), min(W, x1 + reach), min(H, y1 + reach)
        sub = lab[Y0:Y1, X0:X1]
        dil = cv2.dilate((sub == i).astype(np.uint8), disk(reach)).astype(bool)
        for j in np.unique(sub[dil]):
            j = int(j)
            if j != i and j in singles:
                adj[i].add(j)

    def nested(i: int, j: int) -> bool:
        """True if hole j lies inside the region of hole i (or vice versa)."""
        for a, b in ((i, j), (j, i)):
            ca = singles[a]
            ys, xs = np.nonzero(lab[holes[b][1] : holes[b][3], holes[b][0] : holes[b][2]] == b)
            if len(xs) == 0:
                continue
            px, py = xs[len(xs) // 2] + holes[b][0], ys[len(ys) // 2] + holes[b][1]
            rx0, ry0, rx1, ry1 = ca.roi
            if rx0 <= px < rx1 and ry0 <= py < ry1 and ca.mask[py - ry0, px - rx0]:
                return True
        return False

    for i in list(adj):
        adj[i] = {j for j in adj[i] if not nested(i, j)}

    # enumerate connected subsets of size 2..max_merge
    subsets: set[frozenset] = set()
    frontier = {frozenset([i]) for i in singles}
    for _ in range(max(0, params.max_merge - 1)):
        nxt = set()
        for s_ in frontier:
            for i in s_:
                for j in adj[i]:
                    if j not in s_:
                        nxt.add(s_ | {j})
        nxt -= subsets
        subsets |= nxt
        frontier = nxt
        if len(subsets) > 4000:
            break

    thr = params.fit_threshold
    cands: list[ShapeCandidate] = list(singles.values())
    for s_ in subsets:
        c = candidate(s_)
        if c is None:
            continue
        parts_best = max(singles[i].fit.score for i in s_)
        if c.fit.score >= thr and c.fit.score >= parts_best - 0.01:
            cands.append(c)

    good = [c for c in cands if c.fit.score >= thr]
    good.sort(key=lambda c: (-len(c.holes), -c.fit.score))
    accepted: list[ShapeCandidate] = []
    for c in good:
        subsumed = any(c.holes < a.holes and a.fit.score >= c.fit.score - 0.03 for a in accepted)
        if subsumed or any(_cand_iou(c, a) > 0.9 for a in accepted):
            continue
        accepted.append(c)

    # outlines with cluttered interiors that the hole analysis could not assemble
    for c in _component_candidates(closed, sw, heavy, min_dim, thr):
        if any(_cand_iou(c, a) > 0.85 for a in accepted):
            continue
        # union of overlapping shapes that were already found (Venn diagrams)
        x0, y0, x1, y1 = c.roi
        union = np.zeros(c.mask.shape, bool)
        pieces = []  # accepted candidates lying (almost) entirely inside c
        for a in accepted:
            ax0, ay0, ax1, ay1 = a.roi
            ix0, iy0, ix1, iy1 = max(x0, ax0), max(y0, ay0), min(x1, ax1), min(y1, ay1)
            if ix1 > ix0 and iy1 > iy0:
                part = a.mask[iy0 - ay0 : iy1 - ay0, ix0 - ax0 : ix1 - ax0]
                union[iy0 - y0 : iy1 - y0, ix0 - x0 : ix1 - x0] |= part
                inside = np.count_nonzero(part & c.mask[iy0 - y0 : iy1 - y0, ix0 - x0 : ix1 - x0])
                if inside >= 0.9 * max(1, np.count_nonzero(a.mask)):
                    pieces.append(a)
        if union.any() and _iou(union, c.mask) > 0.8:
            # genuine Venn pieces fit better than their union; partial pieces of one outline (cut into
            # more parts than max_merge) fit worse than the whole outline, which then replaces them
            if pieces and c.fit.score > max(a.fit.score for a in pieces) + 0.02:
                accepted = [a for a in accepted if not any(a is p for p in pieces)]
            else:
                continue
        accepted.append(c)

    covered = set().union(*[a.holes for a in accepted]) if accepted else set()
    tol = 2.0 * sw + 2.0
    out = list(accepted)
    for i, c in singles.items():
        if any(c is a for a in accepted):
            continue
        subdivision = i in covered
        if not subdivision:
            if c.fit.score < params.min_region_score:
                continue  # thin sliver between strokes, not a meaningful area
            # area cut out of a shape by lines/text: shares a good part of its outline
            for a in accepted:
                if a.fit.attributes.get("from_component") and _cand_iou(c, a) > 0 and _boundary_overlap(c, a, tol) > 0.25:
                    subdivision = True
                    break
        if subdivision and not params.keep_subregions:
            continue
        attrs = {**c.fit.attributes, **({"subdivision": True} if subdivision else {})}
        c.fit = ShapeFit(ShapeType.REGION, c.fit.score, c.fit.scores, attrs)
        if any(_cand_iou(c, a) > 0.9 for a in out):
            continue
        out.append(c)

    # ink density inside each shape: heavily scribbled interiors are crossed out
    for c in out:
        x0, y0, x1, y1 = c.roi
        inner = cv2.erode(c.mask.astype(np.uint8), disk(int(math.ceil(1.5 * sw)) + 1)).astype(bool)
        if inner.any():
            dens = float(ink[y0:y1, x0:x1][inner].mean())
            c.fit.attributes["ink_density"] = round(dens, 3)
    return out
