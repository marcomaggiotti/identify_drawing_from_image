"""Metrics against ground truth (e.g. the synthetic validation set) and VLM benchmarking."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from scipy.optimize import linear_sum_assignment

from .schema import DiagramGraph, ShapeType
from .vision.geometry import rasterize_polygon

IGNORED_SHAPES = {ShapeType.REGION, ShapeType.SCRIBBLE, ShapeType.UNKNOWN}
TYPE_FAMILY = {
    "circle": "round",
    "ellipse": "round",
    "rectangle": "box",
    "rounded_rectangle": "box",
    "triangle": "triangle",
    "diamond": "diamond",
    "polygon": "polygon",
}


# ------------------------------------------------------------ coordinates
def map_to_original(g: DiagramGraph) -> DiagramGraph:
    """Map a prediction from working-image pixels back to the original file's pixels."""
    info = g.image
    rot = int(info.rotation or 0) % 360
    w, h = info.width, info.height
    W0, H0 = (h, w) if rot in (90, 270) else (w, h)  # size before rotation
    s = info.scale or 1.0
    ox, oy = (info.crop.x, info.crop.y) if info.crop else (0.0, 0.0)

    def f(p):
        x, y = p
        if rot == 90:
            x, y = y, H0 - 1 - x
        elif rot == 180:
            x, y = W0 - 1 - x, H0 - 1 - y
        elif rot == 270:
            x, y = W0 - 1 - y, x
        return (round(x / s + ox, 1), round(y / s + oy, 1))

    out = g.model_copy(deep=True)
    from .schema import BBox

    def fb(b):
        x0, y0, x1, y1 = b.xyxy
        return BBox.from_points([f((x0, y0)), f((x1, y1)), f((x0, y1)), f((x1, y0))])

    for sh in out.shapes:
        sh.polygon = [f(p) for p in sh.polygon]
        sh.bbox = fb(sh.bbox)
    for t in out.texts:
        t.bbox = fb(t.bbox)
    for c in out.connections:
        c.path = [f(p) for p in c.path]
        for ep in c.endpoints:
            ep.point = f(ep.point)
    out.image.width, out.image.height = info.original_width or w, info.original_height or h
    out.image.rotation, out.image.scale, out.image.crop = 0, 1.0, None
    return out


# ------------------------------------------------------------ matching
def _shape_masks(g: DiagramGraph, shapes, scale: float, shp) -> list[np.ndarray]:
    return [rasterize_polygon(s.polygon, shp, scale=scale) if len(s.polygon) >= 3 else _box_mask(s.bbox, shp, scale) for s in shapes]


def _box_mask(b, shp, scale):
    m = np.zeros(shp, bool)
    x0, y0, x1, y1 = (int(round(v * scale)) for v in b.xyxy)
    m[max(0, y0) : y1, max(0, x0) : x1] = True
    return m


def _iou_matrix(ma: list[np.ndarray], mb: list[np.ndarray]) -> np.ndarray:
    M = np.zeros((len(ma), len(mb)))
    if not ma or not mb:
        return M
    A = np.stack([m.ravel() for m in ma]).astype(np.float32)
    B = np.stack([m.ravel() for m in mb]).astype(np.float32)
    inter = A @ B.T
    union = A.sum(1)[:, None] + B.sum(1)[None, :] - inter
    return np.where(union > 0, inter / np.maximum(union, 1), 0)


def _match(iou: np.ndarray, thr: float) -> list[tuple[int, int, float]]:
    if iou.size == 0:
        return []
    r, c = linear_sum_assignment(-iou)
    return [(int(i), int(j), float(iou[i, j])) for i, j in zip(r, c) if iou[i, j] >= thr]


def _box_iou_matrix(a, b) -> np.ndarray:
    """IoU of text boxes; a small label whose centre falls in a slightly larger box also matches."""
    M = np.zeros((len(a), len(b)))
    for i, x in enumerate(a):
        for j, y in enumerate(b):
            v = x.bbox.iou(y.bbox)
            if v < 0.3 and y.bbox.contains_point(x.bbox.center) and y.bbox.area <= 6 * max(x.bbox.area, 1):
                v = 0.3
            M[i, j] = v
    return M


def cer(a: str, b: str) -> float:
    a, b = a.strip(), b.strip()
    if not a:
        return 0.0 if not b else 1.0
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1] / len(a)


@dataclass
class Counts:
    tp: int = 0
    fp: int = 0
    fn: int = 0

    def add(self, tp, fp, fn):
        self.tp += tp
        self.fp += fp
        self.fn += fn

    def prf(self) -> dict[str, float]:
        p = self.tp / (self.tp + self.fp) if self.tp + self.fp else 0.0
        r = self.tp / (self.tp + self.fn) if self.tp + self.fn else 0.0
        f = 2 * p * r / (p + r) if p + r else 0.0
        return {"precision": round(p, 4), "recall": round(r, 4), "f1": round(f, 4)}


@dataclass
class Evaluation:
    shapes: Counts = field(default_factory=Counts)
    containment: Counts = field(default_factory=Counts)
    texts: Counts = field(default_factory=Counts)
    connections: Counts = field(default_factory=Counts)
    type_correct: int = 0
    family_correct: int = 0
    matched_shapes: int = 0
    placement_correct: int = 0
    placement_total: int = 0
    cer_sum: float = 0.0
    cer_n: int = 0
    exact_text: int = 0
    images: int = 0
    seconds: float = 0.0

    def add_image(self, pred: DiagramGraph, gt: DiagramGraph, shape_iou: float = 0.5, text_iou: float = 0.3) -> None:
        self.images += 1
        W, H = max(1, gt.image.width), max(1, gt.image.height)
        scale = min(1.0, 600.0 / max(W, H))
        shp = (int(H * scale) + 2, int(W * scale) + 2)
        gs = [s for s in gt.shapes if s.type not in IGNORED_SHAPES]
        ps = [s for s in pred.shapes if s.type not in IGNORED_SHAPES]
        m = _match(_iou_matrix(_shape_masks(gt, gs, scale, shp), _shape_masks(pred, ps, scale, shp)), shape_iou)
        self.shapes.add(len(m), len(ps) - len(m), len(gs) - len(m))
        smap = {ps[j].id: gs[i].id for i, j, _ in m}  # pred id -> gt id
        for i, j, _ in m:
            self.matched_shapes += 1
            if gs[i].type == ps[j].type:
                self.type_correct += 1
            if TYPE_FAMILY.get(gs[i].type.value) == TYPE_FAMILY.get(ps[j].type.value):
                self.family_correct += 1
        # containment among matched shapes (direct parent pairs)
        matched_gt = set(smap.values())
        gt_pairs = {(s.id, s.parent_id) for s in gs if s.parent_id and s.id in matched_gt and s.parent_id in matched_gt}
        pr_pairs = {(smap[s.id], smap[s.parent_id]) for s in ps if s.parent_id in smap and s.id in smap}
        self.containment.add(len(gt_pairs & pr_pairs), len(pr_pairs - gt_pairs), len(gt_pairs - pr_pairs))
        # texts
        tm = _match(_box_iou_matrix(gt.texts, pred.texts), text_iou)
        self.texts.add(len(tm), len(pred.texts) - len(tm), len(gt.texts) - len(tm))
        tmap = {pred.texts[j].id: gt.texts[i].id for i, j, _ in tm}
        for i, j, _ in tm:
            gtt, prt = gt.texts[i], pred.texts[j]
            if gtt.text and prt.text:
                c = cer(gtt.text, prt.text)
                self.cer_sum += c
                self.cer_n += 1
                self.exact_text += int(gtt.text.strip().lower() == prt.text.strip().lower())
            if gtt.inside_shape_id and gtt.inside_shape_id in matched_gt:
                self.placement_total += 1
                if prt.inside_shape_id and smap.get(prt.inside_shape_id) == gtt.inside_shape_id:
                    self.placement_correct += 1
        # connections: unordered pairs of attached elements, in gt ids
        idmap = {**smap, **tmap}

        def pairs(g: DiagramGraph, mapping: dict | None) -> set[frozenset]:
            out = set()
            for c in g.connections:
                ids = c.attached_ids()
                if mapping is not None:
                    ids = [mapping[x] for x in ids if x in mapping]
                for a in range(len(ids)):
                    for b in range(a + 1, len(ids)):
                        if ids[a] != ids[b]:
                            out.add(frozenset((ids[a], ids[b])))
            return out

        gp = pairs(gt, None)
        pp = pairs(pred, idmap)
        self.connections.add(len(gp & pp), len(pp - gp), len(gp - pp))

    def report(self) -> dict[str, Any]:
        return {
            "images": self.images,
            "shapes": self.shapes.prf(),
            "shape_type_accuracy": round(self.type_correct / self.matched_shapes, 4) if self.matched_shapes else None,
            "shape_family_accuracy": round(self.family_correct / self.matched_shapes, 4) if self.matched_shapes else None,
            "containment": self.containment.prf(),
            "texts": self.texts.prf(),
            "text_cer": round(self.cer_sum / self.cer_n, 4) if self.cer_n else None,
            "text_exact": round(self.exact_text / self.cer_n, 4) if self.cer_n else None,
            "text_inside_accuracy": round(self.placement_correct / self.placement_total, 4) if self.placement_total else None,
            "connections": self.connections.prf(),
            "seconds_per_image": round(self.seconds / self.images, 2) if self.images else None,
        }


def evaluate_dataset(data_dir: str | Path, orchestrator, split: str = "val", limit: int | None = None, save_dir: str | Path | None = None, progress: bool = True) -> dict[str, Any]:
    """Run the orchestrator on ``images/<split>`` and compare with ``annotations/<split>``."""
    data_dir = Path(data_dir)
    ann = sorted((data_dir / "annotations" / split).glob("*.json"))
    if limit:
        ann = ann[:limit]
    ev = Evaluation()
    for k, a in enumerate(ann, 1):
        gt = DiagramGraph.from_json(a.read_text(encoding="utf-8"))
        img = data_dir / "images" / split / (a.stem + ".jpg")
        t0 = time.time()
        res = orchestrator.analyze(str(img))
        ev.seconds += time.time() - t0
        pred = map_to_original(res.graph)
        ev.add_image(pred, gt)
        if save_dir:
            res.save(save_dir, a.stem, include_trace=True)
        if progress:
            print(f"  [{k}/{len(ann)}] {a.stem}: {res.graph.stats()['shapes']} shapes / gt {len(gt.shapes)}", flush=True)
    return ev.report()


def report_markdown(rows: Iterable[tuple[str, dict]]) -> str:
    head = "| system | shapes P/R/F1 | type acc | family acc | containment F1 | texts F1 | text CER | inside acc | connections F1 | s/img |\n|---|---|---|---|---|---|---|---|---|---|"
    lines = [head]
    for name, r in rows:
        s = r["shapes"]
        lines.append(
            f"| {name} | {s['precision']:.2f}/{s['recall']:.2f}/{s['f1']:.2f} | {_f(r['shape_type_accuracy'])} | {_f(r['shape_family_accuracy'])} | "
            f"{r['containment']['f1']:.2f} | {r['texts']['f1']:.2f} | {_f(r['text_cer'])} | {_f(r['text_inside_accuracy'])} | {r['connections']['f1']:.2f} | {_f(r['seconds_per_image'])} |"
        )
    return "\n".join(lines)


def _f(v) -> str:
    return "-" if v is None else f"{v:.2f}"


def save_report(path: str | Path, rows: list[tuple[str, dict]]) -> None:
    Path(path).write_text(json.dumps({name: r for name, r in rows}, indent=2), encoding="utf-8")
