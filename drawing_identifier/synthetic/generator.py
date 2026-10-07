"""Synthetic hand-drawn diagram pages with exact ground truth.

Every page is generated from a random *scene graph* mimicking the example
manuscripts: nested ovals/boxes ("cuts"), heavy lines of identity joining
labels, shape-to-shape lines and arrows, Venn-like overlaps, labels inside and
next to shapes, paragraphs of handwriting with crossed-out words, scribbles,
ruled paper, two-page notebook spreads, stains, microfilm frames, rotations and
scanner noise.  The ground truth is a ``DiagramGraph`` in final image pixels.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np

from ..schema import (
    BBox,
    Connection,
    ConnectionType,
    DiagramGraph,
    Endpoint,
    ImageInfo,
    Shape,
    ShapeType,
    TextItem,
    TextPlacement,
)
from ..vision.topology import build_relations, place_texts, shape_pairs_overlap
from .handdraw import bezier, boundary_point_towards, draw_stroke, elbow, hand_outline, scribble
from .text import TextPatch, list_fonts, random_label, random_sentence, render_text, strike_through


@dataclass
class SynthConfig:
    width_range: tuple[int, int] = (1100, 1700)
    height_range: tuple[int, int] = (850, 1500)
    output_max_side: int = 1600
    n_diagrams: tuple[int, int] = (1, 4)
    max_depth: int = 4
    shape_weights: dict[str, float] = field(
        default_factory=lambda: {
            "ellipse": 0.40,
            "circle": 0.10,
            "rounded_rectangle": 0.15,
            "rectangle": 0.13,
            "triangle": 0.08,
            "diamond": 0.09,
            "polygon": 0.05,
        }
    )
    stroke_width: tuple[float, float] = (1.8, 4.0)
    heavy_factor: tuple[float, float] = (2.0, 3.2)
    text_height: tuple[float, float] = (13.0, 26.0)
    p_venn: float = 0.15
    p_near_label: float = 0.35
    p_line_of_identity: float = 0.7
    p_shape_edge: float = 0.35
    p_arrow: float = 0.35
    p_branch: float = 0.15
    p_dangling: float = 0.15
    p_paragraphs: float = 0.8
    p_crossed_line: float = 0.08
    p_scribble: float = 0.12
    p_two_page: float = 0.3
    p_ruled: float = 0.35
    p_margin: float = 0.5
    p_microfilm: float = 0.45
    p_grayscale: float = 0.3
    p_rotate90: float = 0.1
    max_skew_deg: float = 2.0
    p_stains: float = 0.5
    p_bleed_through: float = 0.25
    fonts_dir: str | None = "assets/fonts"  # scripts/download_fonts.py; falls back to OpenCV fonts


@dataclass
class _Shape:
    id: str
    type: ShapeType
    path: np.ndarray
    polygon: np.ndarray
    center: tuple[float, float]
    size: tuple[float, float]
    parent: str | None
    depth: int
    diagram: int
    crossed_out: bool = False


@dataclass
class _Text:
    id: str
    text: str
    box: tuple[int, int, int, int]
    patch: np.ndarray
    kind: str  # label | para
    diagram: int | None = None
    crossed_out: bool = False


@dataclass
class _Conn:
    id: str
    type: ConnectionType
    path: np.ndarray
    ends: list[tuple[str | None, np.ndarray, str, bool]]  # ref, point, attachment, is_head
    heavy: bool
    width: float
    extra_paths: list[np.ndarray] = field(default_factory=list)


class SceneBuilder:
    def __init__(self, cfg: SynthConfig, rng: np.random.Generator):
        self.cfg = cfg
        self.rng = rng
        self.W = int(rng.integers(*cfg.width_range))
        self.H = int(rng.integers(*cfg.height_range))
        self.two_page = rng.random() < cfg.p_two_page
        if self.two_page:
            self.W = int(self.W * 1.35)
        self.sw = float(rng.uniform(*cfg.stroke_width))
        self.th = float(rng.uniform(*cfg.text_height))
        self.occ = np.zeros((self.H, self.W), bool)  # where text may not go
        self.diag_occ = np.zeros((self.H, self.W), bool)  # diagram areas
        self.shapes: list[_Shape] = []
        self.texts: list[_Text] = []
        self.conns: list[_Conn] = []
        self.scribbles: list[tuple[str, np.ndarray]] = []  # (crossed-out shape id, scribble path)
        self.fonts = list_fonts(cfg.fonts_dir)
        self._n = {"S": 0, "T": 0, "C": 0}
        self.margin = int(0.04 * min(self.W, self.H))

    # ------------------------------------------------------------- utilities
    def _id(self, p: str) -> str:
        self._n[p] += 1
        return f"{p}{self._n[p]}"

    def _pick_type(self) -> ShapeType:
        names = list(self.cfg.shape_weights)
        w = np.array([self.cfg.shape_weights[n] for n in names], float)
        return ShapeType(names[int(self.rng.choice(len(names), p=w / w.sum()))])

    def _gutter_ok(self, box) -> bool:
        if not self.two_page:
            return True
        g = self.W / 2
        pad = 0.03 * self.W
        return box[2] < g - pad or box[0] > g + pad

    def _mark_outline(self, pts: np.ndarray, pad: float) -> None:
        cv2.polylines(self.occ.view(np.uint8), [np.round(pts).astype(np.int32).reshape(-1, 1, 2)], False, 1, max(1, int(2 * pad)))

    def _free(self, box, mask=None) -> bool:
        x0, y0, x1, y1 = (int(round(v)) for v in box)
        if x0 < self.margin * 0.5 or y0 < self.margin * 0.5 or x1 > self.W - self.margin * 0.5 or y1 > self.H - self.margin * 0.5:
            return False
        m = self.occ if mask is None else mask
        return not m[y0:y1, x0:x1].any()

    def _make_text(self, s: str, height: float | None = None) -> TextPatch:
        return render_text(s, height or self.th * self.rng.uniform(0.85, 1.15), self.rng, self.fonts, pen=self.sw)

    def _place_patch(self, patch: TextPatch, box, kind: str, diagram: int | None, tries: int = 30, crossed: bool = False) -> _Text | None:
        """Place a text patch somewhere inside ``box`` (x0,y0,x1,y1) on free space."""
        h, w = patch.mask.shape
        x0, y0, x1, y1 = box
        if x1 - x0 < w + 2 or y1 - y0 < h + 2:
            return None
        pad = max(2, int(0.25 * self.th))
        for _ in range(tries):
            px = self.rng.uniform(x0, x1 - w)
            py = self.rng.uniform(y0, y1 - h)
            b = (px, py, px + w, py + h)
            if self._free((b[0] - pad, b[1] - pad, b[2] + pad, b[3] + pad)):
                return self._commit_text(patch, b, kind, diagram, crossed)
        return None

    def _commit_text(self, patch: TextPatch, b, kind: str, diagram: int | None, crossed: bool = False) -> _Text:
        x0, y0 = int(round(b[0])), int(round(b[1]))
        h, w = patch.mask.shape
        t = _Text(self._id("T"), patch.text, (x0, y0, x0 + w, y0 + h), patch.mask, kind, diagram, crossed)
        pad = max(2, int(0.25 * self.th))
        self.occ[max(0, y0 - pad) : y0 + h + pad, max(0, x0 - pad) : x0 + w + pad] = True
        self.texts.append(t)
        return t

    # ------------------------------------------------------------- diagrams
    def _inner_box(self, st: ShapeType, cx, cy, w, h):
        f = {
            ShapeType.ELLIPSE: (0.66, 0.62),
            ShapeType.CIRCLE: (0.64, 0.64),
            ShapeType.ROUNDED_RECTANGLE: (0.84, 0.78),
            ShapeType.RECTANGLE: (0.86, 0.80),
            ShapeType.TRIANGLE: (0.42, 0.34),
            ShapeType.DIAMOND: (0.46, 0.44),
            ShapeType.POLYGON: (0.66, 0.6),
        }[st]
        iw, ih = w * f[0], h * f[1]
        oy = h * 0.18 if st == ShapeType.TRIANGLE else 0.0
        return (cx - iw / 2, cy - ih / 2 + oy, cx + iw / 2, cy + ih / 2 + oy)

    def build_container(self, box, depth: int, parent: str | None, diagram: int) -> _Shape:
        rng = self.rng
        st = self._pick_type()
        x0, y0, x1, y1 = box
        w, h = x1 - x0, y1 - y0
        if st == ShapeType.CIRCLE or (st in (ShapeType.DIAMOND, ShapeType.TRIANGLE, ShapeType.POLYGON) and rng.random() < 0.5):
            d = min(w, h)
            w = h = d
        elif st == ShapeType.ELLIPSE and abs(w - h) < 0.12 * max(w, h):
            h *= 0.75  # keep ellipses visibly elongated
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        angle = float(rng.normal(0, 6)) if st == ShapeType.ELLIPSE else float(rng.normal(0, 1.5))
        path, poly = hand_outline(st, cx, cy, w, h, rng, angle=angle)
        sh = _Shape(self._id("S"), st, path, poly, (cx, cy), (w, h), parent, depth, diagram)
        self.shapes.append(sh)
        self._mark_outline(path, 0.45 * self.th + self.sw)
        inner = self._inner_box(st, cx, cy, w, h)
        self._fill(inner, depth, sh.id, diagram)
        return sh

    def _fill(self, inner, depth: int, parent: str, diagram: int) -> None:
        rng = self.rng
        iw, ih = inner[2] - inner[0], inner[3] - inner[1]
        min_child = 4.5 * self.th
        r = rng.random()
        can_nest = depth + 1 < self.cfg.max_depth and min(iw, ih) > min_child
        if can_nest and r < 0.33:
            f = rng.uniform(0.72, 0.9)
            cx, cy = (inner[0] + inner[2]) / 2, (inner[1] + inner[3]) / 2
            self.build_container((cx - iw * f / 2, cy - ih * f / 2, cx + iw * f / 2, cy + ih * f / 2), depth + 1, parent, diagram)
            return
        if can_nest and r < 0.7:
            k = int(rng.integers(2, 4))
            horizontal = iw >= ih
            cells = []
            for i in range(k):
                if horizontal:
                    cells.append((inner[0] + iw * i / k, inner[1], inner[0] + iw * (i + 1) / k, inner[3]))
                else:
                    cells.append((inner[0], inner[1] + ih * i / k, inner[2], inner[1] + ih * (i + 1) / k))
            for c in cells:
                cw, ch = c[2] - c[0], c[3] - c[1]
                g = 0.08
                cell = (c[0] + cw * g, c[1] + ch * g, c[2] - cw * g, c[3] - ch * g)
                if min(cw, ch) > min_child and rng.random() < 0.6:
                    self.build_container(cell, depth + 1, parent, diagram)
                else:
                    self._labels(cell, int(rng.integers(1, 3)), diagram)
            return
        self._labels(inner, int(rng.integers(1, 4)), diagram)

    def _labels(self, box, n: int, diagram: int) -> list[_Text]:
        out = []
        for _ in range(n):
            label = random_label(self.rng)
            for attempt in range(3):
                patch = self._make_text(label, self.th * self.rng.uniform(1.0, 1.45))
                t = self._place_patch(patch, box, "label", diagram)
                if t is not None:
                    out.append(t)
                    break
                label = random_label(self.rng) if attempt == 0 else self.rng.choice(["A", "B", "p", "q", "r", "C"])
        return out

    def _near_label(self, sh: _Shape, diagram: int) -> _Text | None:
        patch = self._make_text(str(self.rng.choice(["A", "B", "C", "D", "E", "O", "I", "M", "a", "b", "p", "q", "r"])))
        h, w = patch.mask.shape
        cx, cy = sh.center
        for _ in range(16):
            ang = self.rng.uniform(0, 2 * math.pi)
            far = np.array([cx + math.cos(ang) * 10 * max(sh.size), cy + math.sin(ang) * 10 * max(sh.size)])
            bp = boundary_point_towards(sh.polygon, sh.center, far)
            d = self.rng.uniform(0.35, 0.9) * self.th + 0.5 * max(w, h)
            px = bp[0] + math.cos(ang) * d - w / 2
            py = bp[1] + math.sin(ang) * d - h / 2
            b = (px, py, px + w, py + h)
            if self._free((b[0] - 2, b[1] - 2, b[2] + 2, b[3] + 2)) and self._gutter_ok(b):
                return self._commit_text(patch, b, "label", diagram)
        return None

    def build_venn(self, box, diagram: int) -> None:
        rng = self.rng
        x0, y0, x1, y1 = box
        w, h = x1 - x0, y1 - y0
        st = ShapeType.CIRCLE if rng.random() < 0.5 else ShapeType.ELLIPSE
        k = 1.0 if st == ShapeType.CIRCLE else 1.15  # half-width / r
        # the pair spans 2 * (dx + k r) <= 2 r (0.85 + k) wide: keep it inside the box
        r = min(w / (2 * (0.85 + k)), h / (2 * (1.0 if st == ShapeType.CIRCLE else 0.8)) * 0.95)
        cy = (y0 + y1) / 2
        cx = (x0 + x1) / 2
        dx = r * rng.uniform(0.55, 0.85)
        ids = []
        for sx in (-1, 1):
            ww, hh = (2 * r, 2 * r) if st == ShapeType.CIRCLE else (2 * r * 1.15, 2 * r * 0.8)
            path, poly = hand_outline(st, cx + sx * dx, cy, ww, hh, rng)
            sh = _Shape(self._id("S"), st, path, poly, (cx + sx * dx, cy), (ww, hh), None, 0, diagram)
            self.shapes.append(sh)
            self._mark_outline(path, 0.4 * self.th + self.sw)
            ids.append(sh)
        for px in (cx - dx - 0.45 * r, cx, cx + dx + 0.45 * r):
            bw = 0.5 * r
            self._labels((px - bw / 2, cy - 0.35 * r, px + bw / 2, cy + 0.35 * r), 1, diagram)

    def place_diagrams(self) -> None:
        rng = self.rng
        n = int(rng.integers(self.cfg.n_diagrams[0], self.cfg.n_diagrams[1] + 1))
        boxes = []
        for d in range(n):
            for _ in range(40):
                w = rng.uniform(0.22, 0.55) * self.W * (0.75 if self.two_page else 1.0)
                h = rng.uniform(0.6, 1.3) * w * rng.uniform(0.6, 1.0)
                h = min(h, 0.6 * self.H)
                x = rng.uniform(self.margin, self.W - self.margin - w)
                y = rng.uniform(self.margin, self.H - self.margin - h)
                b = (x, y, x + w, y + h)
                if not self._gutter_ok(b) or any(_overlap(b, o, 0.04 * self.W) for o in boxes):
                    continue
                boxes.append(b)
                break
        for d, b in enumerate(boxes):
            self.diag_occ[int(b[1]) : int(b[3]), int(b[0]) : int(b[2])] = True
            if rng.random() < self.cfg.p_venn:
                self.build_venn(b, d)
            else:
                self.build_container(b, 0, None, d)
            for sh in [s for s in self.shapes if s.diagram == d]:
                if rng.random() < self.cfg.p_near_label * (0.6 if sh.depth else 1.0):
                    self._near_label(sh, d)
            self._diagram_connections(d)
        self._inter_diagram_arrows()

    # ------------------------------------------------------------ connectors
    def _text_anchor(self, t: _Text, toward) -> np.ndarray:
        x0, y0, x1, y1 = t.box
        c = np.array([(x0 + x1) / 2, (y0 + y1) / 2])
        d = np.asarray(toward, float) - c
        if abs(d[0]) * (y1 - y0) > abs(d[1]) * (x1 - x0):
            p = np.array([x1 + 2 if d[0] > 0 else x0 - 2, c[1]])
        else:
            p = np.array([c[0], y1 + 2 if d[1] > 0 else y0 - 2])
        return p + self.rng.normal(0, 1.0, 2)

    def _diagram_connections(self, d: int) -> None:
        rng = self.rng
        labels = [t for t in self.texts if t.diagram == d and t.kind == "label"]
        shapes = [s for s in self.shapes if s.diagram == d]
        used: set[str] = set()
        if len(labels) >= 2 and rng.random() < self.cfg.p_line_of_identity:
            for _ in range(int(rng.integers(1, min(4, len(labels)) + 1))):
                a, b = rng.choice(len(labels), 2, replace=False)
                ta, tb = labels[int(a)], labels[int(b)]
                if ta.id in used and tb.id in used:
                    continue
                ca = np.array([(ta.box[0] + ta.box[2]) / 2, (ta.box[1] + ta.box[3]) / 2])
                cb = np.array([(tb.box[0] + tb.box[2]) / 2, (tb.box[1] + tb.box[3]) / 2])
                pa, pb = self._text_anchor(ta, cb), self._text_anchor(tb, ca)
                if np.linalg.norm(pb - pa) < 2.5 * self.th:
                    continue
                path = elbow(pa, pb, rng) if rng.random() < 0.25 else bezier(pa, pb, rng, bend=0.3)
                heavy = rng.random() < 0.65
                conn = _Conn(self._id("C"), ConnectionType.LINE, path, [(ta.id, pa, "text", False), (tb.id, pb, "text", False)], heavy, self._width(heavy))
                if rng.random() < self.cfg.p_branch:
                    others = [t for t in labels if t.id not in (ta.id, tb.id)]
                    if others:
                        tc = others[int(rng.integers(0, len(others)))]
                        mid = path[len(path) // 2]
                        pc = self._text_anchor(tc, mid)
                        conn.extra_paths.append(bezier(mid, pc, rng, bend=0.2))
                        conn.ends.append((tc.id, pc, "text", False))
                self.conns.append(conn)
                used.update({ta.id, tb.id})
        if labels and rng.random() < self.cfg.p_dangling:
            t = labels[int(rng.integers(0, len(labels)))]
            c = np.array([(t.box[0] + t.box[2]) / 2, (t.box[1] + t.box[3]) / 2])
            ang = rng.uniform(0, 2 * math.pi)
            far = c + np.array([math.cos(ang), math.sin(ang)]) * rng.uniform(3, 7) * self.th
            far = np.clip(far, self.margin, [self.W - self.margin, self.H - self.margin])
            p0 = self._text_anchor(t, far)
            heavy = rng.random() < 0.7
            self.conns.append(_Conn(self._id("C"), ConnectionType.LINE, bezier(p0, far, rng, 0.15), [(t.id, p0, "text", False), (None, far, "free", False)], heavy, self._width(heavy)))
        # shape -> shape edges between shapes that are not nested in one another
        if len(shapes) >= 2 and rng.random() < self.cfg.p_shape_edge:
            anc = {s.id: self._ancestors(s) for s in shapes}
            pairs = [(a, b) for i, a in enumerate(shapes) for b in shapes[i + 1 :] if a.id not in anc[b.id] and b.id not in anc[a.id]]
            if pairs:
                a, b = pairs[int(rng.integers(0, len(pairs)))]
                pa = boundary_point_towards(a.polygon, a.center, b.center)
                pb = boundary_point_towards(b.polygon, b.center, a.center)
                if np.linalg.norm(pb - pa) > 2 * self.th:
                    arrow = rng.random() < self.cfg.p_arrow
                    self.conns.append(
                        _Conn(
                            self._id("C"),
                            ConnectionType.ARROW if arrow else ConnectionType.LINE,
                            bezier(pa, pb, rng, 0.2),
                            [(a.id, pa, "boundary", False), (b.id, pb, "boundary", arrow)],
                            False,
                            self._width(False),
                        )
                    )

    def _inter_diagram_arrows(self) -> None:
        roots = [s for s in self.shapes if s.parent is None]
        if len({s.diagram for s in roots}) < 2 or self.rng.random() > 0.3:
            return
        a, b = self.rng.choice(len(roots), 2, replace=False)
        sa, sb = roots[int(a)], roots[int(b)]
        if sa.diagram == sb.diagram:
            return
        pa = boundary_point_towards(sa.polygon, sa.center, sb.center)
        pb = boundary_point_towards(sb.polygon, sb.center, sa.center)
        d = pb - pa
        L = np.linalg.norm(d)
        if L < 3 * self.th:
            return
        pa = pa + d / L * 0.6 * self.th
        pb = pb - d / L * 0.6 * self.th
        self.conns.append(
            _Conn(self._id("C"), ConnectionType.ARROW, bezier(pa, pb, self.rng, 0.25), [(sa.id, pa, "boundary", False), (sb.id, pb, "boundary", True)], False, self._width(False))
        )

    def _ancestors(self, s: _Shape) -> set[str]:
        by = {x.id: x for x in self.shapes}
        out, p = set(), s.parent
        while p:
            out.add(p)
            p = by[p].parent
        return out

    def _width(self, heavy: bool) -> float:
        return self.sw * (self.rng.uniform(*self.cfg.heavy_factor) if heavy else self.rng.uniform(0.9, 1.15))

    # ------------------------------------------------------------ distractors
    def place_paragraphs(self) -> None:
        rng = self.rng
        if rng.random() > self.cfg.p_paragraphs:
            return
        for _ in range(int(rng.integers(1, 5))):
            lh = self.th * rng.uniform(1.9, 2.6)
            n_lines = int(rng.integers(1, 7))
            w = rng.uniform(0.25, 0.55) * self.W
            h = n_lines * lh
            for _ in range(25):
                x = rng.uniform(self.margin, self.W - self.margin - w)
                y = rng.uniform(self.margin, self.H - self.margin - h)
                b = (x, y, x + w, y + h)
                if not self._gutter_ok(b) or self.diag_occ[int(y) : int(y + h), int(x) : int(x + w)].any() or not self._free(b):
                    continue
                for i in range(n_lines):
                    sentence = random_sentence(rng)
                    patch = self._make_text(sentence)
                    crossed = rng.random() < self.cfg.p_crossed_line
                    m = strike_through(patch.mask, rng) if crossed else patch.mask
                    ph, pw = m.shape
                    if pw > self.W - self.margin - x:
                        continue
                    ly = y + i * lh + rng.normal(0, 0.1 * self.th)
                    lx = x + rng.normal(0, 0.3 * self.th) + (rng.uniform(0, 3) * self.th if rng.random() < 0.3 else 0)
                    box = (lx, ly, lx + pw, ly + ph)
                    if self._free(box):
                        self._commit_text(TextPatch(m, sentence), box, "para", None, crossed)
                self.diag_occ[int(y) : int(y + h), int(x) : int(x + w)] = True
                break

    def place_scribbles(self) -> None:
        if self.rng.random() > self.cfg.p_scribble or not self.shapes:
            return
        cand = [s for s in self.shapes if s.depth > 0] or self.shapes
        s = cand[int(self.rng.integers(0, len(cand)))]
        w, h = s.size
        self.scribbles.append((s.id, scribble(s.center, (0.55 * w, 0.5 * h), self.rng, n=int(self.rng.integers(8, 20)))))
        s.crossed_out = True


def _overlap(a, b, pad: float) -> bool:
    return not (a[2] + pad < b[0] or b[2] + pad < a[0] or a[3] + pad < b[1] or b[3] + pad < a[1])


# ============================================================== rendering
def _paper(W: int, H: int, rng: np.random.Generator) -> np.ndarray:
    base = np.array(
        [
            (245, 245, 245),
            (215, 232, 242),
            (190, 214, 230),
            (225, 228, 228),
            (205, 225, 238),
        ][int(rng.integers(0, 5))],
        np.float32,
    )
    img = np.ones((H, W, 3), np.float32) * base
    low = cv2.resize(rng.normal(0, 1, (6, 6)).astype(np.float32), (W, H), interpolation=cv2.INTER_CUBIC)
    img += (low * rng.uniform(3, 12))[:, :, None]
    img += rng.normal(0, rng.uniform(1, 4), (H, W, 1)).astype(np.float32)
    return img


def _rules(img: np.ndarray, rng: np.random.Generator, margin: bool) -> None:
    H, W = img.shape[:2]
    gap = rng.uniform(24, 38)
    color = np.array([200, 170, 140], np.float32) * rng.uniform(0.9, 1.05)  # light blue (BGR)
    y = rng.uniform(40, 90)
    while y < H:
        cv2.line(img, (0, int(y)), (W, int(y)), color.tolist(), 1, cv2.LINE_AA)
        y += gap
    if margin:
        x = rng.uniform(0.1, 0.2) * W
        for dx in (0, 4):
            cv2.line(img, (int(x + dx), 0), (int(x + dx), H), (150, 150, 225), 1, cv2.LINE_AA)


def _stains(img: np.ndarray, rng: np.random.Generator) -> None:
    H, W = img.shape[:2]
    for _ in range(int(rng.integers(1, 6))):
        m = np.zeros((H, W), np.float32)
        c = (int(rng.uniform(0, W)), int(rng.uniform(0, H)))
        ax = (int(rng.uniform(0.03, 0.15) * W), int(rng.uniform(0.03, 0.12) * H))
        cv2.ellipse(m, c, ax, rng.uniform(0, 180), 0, 360, 1.0, -1)
        m = cv2.GaussianBlur(m, (0, 0), rng.uniform(10, 40))
        img -= (m * rng.uniform(15, 55))[:, :, None] * np.array([1.0, 1.1, 1.0], np.float32)


def render_scene(sc: SceneBuilder) -> np.ndarray:
    rng = sc.rng
    cfg = sc.cfg
    W, H = sc.W, sc.H
    img = _paper(W, H, rng)
    if rng.random() < cfg.p_ruled:
        _rules(img, rng, rng.random() < cfg.p_margin)
    if rng.random() < cfg.p_bleed_through:
        ghost = np.zeros((H, W), np.uint8)
        for _ in range(int(rng.integers(3, 10))):
            p = render_text(random_sentence(rng), sc.th, rng)
            m = cv2.flip(p.mask, 1)
            x, y = int(rng.uniform(0, max(1, W - m.shape[1]))), int(rng.uniform(0, max(1, H - m.shape[0])))
            sub = ghost[y : y + m.shape[0], x : x + m.shape[1]]
            np.maximum(sub, m[: sub.shape[0], : sub.shape[1]], out=sub)
        img -= (cv2.GaussianBlur(ghost, (0, 0), 1.5).astype(np.float32) / 255.0 * rng.uniform(12, 35))[:, :, None]
    if sc.two_page:
        g = int(W / 2)
        band = np.zeros((H, W), np.float32)
        band[:, g - 3 : g + 3] = 1
        band = cv2.GaussianBlur(band, (0, 0), rng.uniform(4, 14))
        img -= (band * rng.uniform(40, 90))[:, :, None]
        cv2.line(img, (g, 0), (g, H), (90, 90, 90), 1, cv2.LINE_AA)
    if rng.random() < cfg.p_stains:
        _stains(img, rng)

    ink = np.zeros((H, W), np.uint8)
    for s in sc.shapes:
        draw_stroke(ink, s.path, sc.sw * rng.uniform(0.9, 1.1), 255, rng)
    for c in sc.conns:
        draw_stroke(ink, c.path, c.width, 255, rng, taper=not c.heavy)
        for p in c.extra_paths:
            draw_stroke(ink, p, c.width, 255, rng, taper=False)
        if c.type == ConnectionType.ARROW:
            tip = c.path[-1]
            back = c.path[max(0, len(c.path) - 6)]
            d = tip - back
            d /= np.linalg.norm(d) + 1e-9
            L = sc.th * rng.uniform(0.6, 1.0)
            for sgn in (-1, 1):
                a = math.radians(25 * sgn)
                R = np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])
                barb = tip - (R @ d) * L
                draw_stroke(ink, np.linspace(tip, barb, 6), c.width, 255, rng, taper=False)
    for _, scr in sc.scribbles:
        draw_stroke(ink, scr, sc.sw * 0.9, 255, rng, taper=False)
    for t in sc.texts:
        x0, y0, x1, y1 = t.box
        sub = ink[y0:y1, x0:x1]
        m = t.patch[: sub.shape[0], : sub.shape[1]]
        np.maximum(sub, m, out=sub)
    ink_color = np.array([(25, 25, 25), (30, 40, 60), (70, 40, 30), (15, 15, 20)][int(rng.integers(0, 4))], np.float32)
    a = (ink.astype(np.float32) / 255.0)[:, :, None] * rng.uniform(0.82, 0.98)
    img = img * (1 - a) + ink_color * a
    return np.clip(img, 0, 255).astype(np.uint8)


def _degrade(img: np.ndarray, rng: np.random.Generator, cfg: SynthConfig) -> tuple[np.ndarray, np.ndarray, dict]:
    """Scanner / microfilm effects. Returns (image, 3x3 transform page->image, info)."""
    H, W = img.shape[:2]
    info: dict[str, Any] = {}
    M = np.eye(3)
    microfilm = rng.random() < cfg.p_microfilm
    skew = float(rng.uniform(-cfg.max_skew_deg, cfg.max_skew_deg))
    if microfilm:
        mx = int(W * rng.uniform(0.02, 0.15))
        my = int(H * rng.uniform(0.1, 0.45))
        CW, CH = W + 2 * mx, H + 2 * my
        A = cv2.getRotationMatrix2D((W / 2, H / 2), skew, 1.0)
        A[:, 2] += (mx, my)
        frame = int(rng.uniform(15, 60))
        out = cv2.warpAffine(img, A, (CW, CH), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(frame, frame, frame))
        vig = cv2.GaussianBlur(rng.normal(0, 1, (5, 5)).astype(np.float32), (0, 0), 1)
        vig = cv2.resize(vig, (CW, CH), interpolation=cv2.INTER_CUBIC)
        out = np.clip(out.astype(np.float32) + vig[:, :, None] * 12, 0, 255).astype(np.uint8)
        info["microfilm"] = True
    else:
        A = cv2.getRotationMatrix2D((W / 2, H / 2), skew, 1.0)
        out = cv2.warpAffine(img, A, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    M = np.vstack([A, [0, 0, 1]]) @ M
    if rng.random() < cfg.p_rotate90:
        oh, ow = out.shape[:2]
        if rng.random() < 0.7:  # page lying on its side, text reading bottom-to-top (like the examples)
            out = cv2.rotate(out, cv2.ROTATE_90_COUNTERCLOCKWISE)
            R = np.array([[0, 1, 0], [-1, 0, ow - 1], [0, 0, 1]], float)
            info["rotated"] = 270
        else:
            out = cv2.rotate(out, cv2.ROTATE_90_CLOCKWISE)
            R = np.array([[0, -1, oh - 1], [1, 0, 0], [0, 0, 1]], float)
            info["rotated"] = 90
        M = R @ M
    oh, ow = out.shape[:2]
    s = min(1.0, cfg.output_max_side / max(oh, ow))
    if s < 1.0:
        out = cv2.resize(out, (int(round(ow * s)), int(round(oh * s))), interpolation=cv2.INTER_AREA)
        M = np.diag([s, s, 1.0]) @ M
    # photometric
    if microfilm or rng.random() < cfg.p_grayscale:
        g = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY)
        if microfilm and rng.random() < 0.5:
            # high-contrast film
            g = np.clip((g.astype(np.float32) - 90) * rng.uniform(1.5, 2.5) + 128, 0, 255).astype(np.uint8)
        out = cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)
    gamma = rng.uniform(0.8, 1.25)
    out = np.clip(255.0 * (out / 255.0) ** gamma, 0, 255).astype(np.uint8)
    if rng.random() < 0.6:
        out = cv2.GaussianBlur(out, (0, 0), rng.uniform(0.4, 1.1))
    noise = rng.normal(0, rng.uniform(2, 9), out.shape).astype(np.float32)
    out = np.clip(out.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    if rng.random() < 0.7:
        ok, enc = cv2.imencode(".jpg", out, [cv2.IMWRITE_JPEG_QUALITY, int(rng.integers(35, 92))])
        out = cv2.imdecode(enc, cv2.IMREAD_COLOR)
    return out, M, info


def _tx(M: np.ndarray, pts) -> np.ndarray:
    p = np.asarray(pts, float).reshape(-1, 2)
    h = np.hstack([p, np.ones((len(p), 1))]) @ M.T
    return h[:, :2]


def _tx_box(M: np.ndarray, box) -> BBox:
    x0, y0, x1, y1 = box
    return BBox.from_points([tuple(p) for p in _tx(M, [(x0, y0), (x1, y0), (x1, y1), (x0, y1)])]).round()


def _pl(pts: np.ndarray, step: int = 1) -> list[tuple[float, float]]:
    return [(round(float(x), 1), round(float(y), 1)) for x, y in pts[::step]]


def build_ground_truth(sc: SceneBuilder, M: np.ndarray, size: tuple[int, int], info: dict) -> DiagramGraph:
    w, h = size
    scale = math.sqrt(abs(np.linalg.det(M[:2, :2])))
    g = DiagramGraph(image=ImageInfo(width=w, height=h, original_width=w, original_height=h, stroke_width=round(sc.sw * scale, 2)))
    g.is_drawing = True
    g.drawing_confidence = 1.0
    for s in sc.shapes:
        poly = _tx(M, s.polygon)
        step = max(1, len(poly) // 90)
        g.shapes.append(
            Shape(
                id=s.id,
                type=s.type,
                bbox=BBox.from_points(poly).round(),
                polygon=_pl(poly, step),
                parent_id=s.parent,
                depth=s.depth,
                crossed_out=s.crossed_out,
                source=["synthetic"],
            )
        )
    depth = {s.id: s.depth for s in sc.shapes}
    for i, (host, scr) in enumerate(sc.scribbles, 1):
        pts = _tx(M, scr)
        hull = cv2.convexHull(pts.astype(np.float32)).reshape(-1, 2)
        g.shapes.append(
            Shape(
                id=f"S{len(sc.shapes) + i}",
                type=ShapeType.SCRIBBLE,
                bbox=BBox.from_points(pts).round(),
                polygon=_pl(hull),
                parent_id=host,
                depth=depth.get(host, 0) + 1,
                source=["synthetic"],
                crossed_out=True,
            )
        )
    for t in sc.texts:
        src = ["synthetic", "para"] if t.kind == "para" else ["synthetic", "label"]
        g.texts.append(TextItem(id=t.id, text=t.text, bbox=_tx_box(M, t.box), crossed_out=t.crossed_out, source=src, placement=TextPlacement.FREE))
    for c in sc.conns:
        path = _tx(M, c.path)
        eps = []
        for ref, pt, att, head in c.ends:
            p = _tx(M, [pt])[0]
            is_text = ref is not None and ref.startswith("T")
            eps.append(
                Endpoint(
                    point=(round(float(p[0]), 1), round(float(p[1]), 1)),
                    text_id=ref if is_text else None,
                    shape_id=ref if (ref and not is_text) else None,
                    attachment=att,
                    is_head=head,
                )
            )
        attrs = {"width_px": round(c.width * scale, 2)}
        if c.extra_paths:
            attrs["branches"] = [_pl(_tx(M, p), 3) for p in c.extra_paths]
        g.connections.append(
            Connection(id=c.id, type=c.type, path=_pl(path, max(1, len(path) // 30)), endpoints=eps, heavy=c.heavy, source=["synthetic"], attributes=attrs)
        )
    th = sc.th * scale
    place_texts(g, th)
    for t in g.texts:
        src = next(x for x in sc.texts if x.id == t.id)
        if src.kind == "para":
            t.placement = TextPlacement.FREE if not t.inside_shape_id else t.placement
    overlaps, touches = shape_pairs_overlap(g, sc.sw * scale)
    build_relations(g, overlaps, touches)
    g.notes.append(f"synthetic: text_height={th:.1f}px stroke={sc.sw * scale:.2f}px two_page={sc.two_page} {info}")
    g.image.content_rotation = int(info.get("rotated", 0))
    g.summary = _summary(g)
    return g


def _summary(g: DiagramGraph) -> str:
    """A factual one-sentence description (also the 'summary' target for VLM fine-tuning)."""
    kinds: dict[str, int] = {}
    for s in g.shapes:
        if s.type != ShapeType.SCRIBBLE:
            kinds[s.type.value] = kinds.get(s.type.value, 0) + 1
    parts = [f"{n} {k.replace('_', ' ')}{'s' if n > 1 else ''}" for k, n in sorted(kinds.items())]
    depth = max((s.depth for s in g.shapes), default=0)
    labels = [t.text for t in g.texts if "label" in t.source and t.text][:6]
    out = "Hand drawing with " + (", ".join(parts) if parts else "no closed shapes")
    if depth:
        out += f", nested {depth + 1} levels deep"
    if g.connections:
        heavy = sum(1 for c in g.connections if c.heavy)
        out += f", {len(g.connections)} connecting line{'s' if len(g.connections) > 1 else ''}" + (f" ({heavy} heavy)" if heavy else "")
    if labels:
        out += "; labels include " + ", ".join(f'"{x}"' for x in labels)
    return out + "."


def generate_sample(seed: int, cfg: SynthConfig | None = None) -> tuple[np.ndarray, DiagramGraph]:
    """One synthetic page: (BGR image, ground-truth DiagramGraph)."""
    cfg = cfg or SynthConfig()
    rng = np.random.default_rng(seed)
    sc = SceneBuilder(cfg, rng)
    sc.place_diagrams()
    sc.place_paragraphs()
    sc.place_scribbles()
    page = render_scene(sc)
    img, M, info = _degrade(page, rng, cfg)
    g = build_ground_truth(sc, M, (img.shape[1], img.shape[0]), info)
    return img, g
