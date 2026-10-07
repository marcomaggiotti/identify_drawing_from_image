"""Perception agents: preprocessing, triage, shapes, connections, text layout, topology."""

from __future__ import annotations

import logging
from typing import Any

import cv2
import numpy as np

from ..prompts import ORIENTATION_PROMPT, SYSTEM_PROMPT, TRIAGE_PROMPT
from ..schema import BBox, Connection, ConnectionType, Endpoint, ImageInfo, Shape, ShapeType, TextItem
from ..vision.geometry import simplify
from ..vision.lines import LineParams, extract_lines, shape_ring_mask
from ..vision.preprocess import preprocess, rotate
from ..vision.shapes import ShapeDetectorParams, detect_shapes
from ..vision.text_layout import estimate_text_height, group_text
from ..vision.topology import update_topology
from .base import Agent, AgentReport, AnalysisContext

log = logging.getLogger(__name__)


# --------------------------------------------------------------------------- preprocess
class PreprocessAgent(Agent):
    name = "preprocess"
    description = "Crop the page from the scanner frame, fix orientation (asks the VLM when unsure), flatten the background and binarise the ink."
    provides = ("preprocessed",)
    uses_vlm = True

    def run(self, ctx: AnalysisContext, rotation: int | None = None, **_: Any) -> AgentReport:
        cfg = self.config.preprocess
        fixed: int | None = rotation
        if fixed is None and str(cfg.orientation).lower() not in ("auto",):
            fixed = 0 if str(cfg.orientation).lower() == "none" else int(cfg.orientation)
        prep = preprocess(ctx.original, cfg.max_side, cfg.crop_page, fixed, cfg.remove_rules)
        how = "fixed" if fixed is not None else "classical"
        if fixed is None and self.vlm is not None and cfg.orientation_use_vlm:
            choice = self._ask_orientation(prep.image)
            if choice is not None and choice != 0:
                new_rot = (prep.rotation + choice) % 360
                prep = preprocess(ctx.original, cfg.max_side, cfg.crop_page, new_rot, cfg.remove_rules)
                how = "vlm"
            elif choice == 0:
                how = "classical+vlm"
        ctx.prep = prep
        h, w = prep.ink.shape
        ow, oh = prep.original_size
        crop = BBox(x=prep.crop[0], y=prep.crop[1], w=prep.crop[2], h=prep.crop[3]) if prep.crop else None
        ctx.graph.image = ImageInfo(
            path=ctx.source,
            original_width=ow,
            original_height=oh,
            width=w,
            height=h,
            rotation=prep.rotation,
            crop=crop,
            scale=round(prep.scale, 5),
            stroke_width=round(prep.stroke_width, 2),
        )
        ctx.graph.notes.extend(prep.notes)
        return AgentReport(
            self.name,
            f"working image {w}x{h}, rotation {prep.rotation} ({how}), stroke width {prep.stroke_width:.1f}px",
            {"rotation": prep.rotation, "orientation_method": how, "stroke_width": prep.stroke_width},
        )

    def _ask_orientation(self, img: np.ndarray) -> int | None:
        """Show the page as-is and upside-down side by side; returns extra rotation (0 or 180)."""
        h, w = img.shape[:2]
        s = 700.0 / max(h, w)
        small = cv2.resize(img, (max(1, int(w * s)), max(1, int(h * s))), interpolation=cv2.INTER_AREA)
        tiles = []
        for i, r in enumerate((0, 180), 1):
            t = rotate(small, r).copy()
            t = cv2.copyMakeBorder(t, 60, 10, 10, 10, cv2.BORDER_CONSTANT, value=(255, 255, 255))
            cv2.putText(t, str(i), (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.6, (0, 0, 255), 3)
            tiles.append(t)
        hh = max(t.shape[0] for t in tiles)
        tiles = [cv2.copyMakeBorder(t, 0, hh - t.shape[0], 0, 0, cv2.BORDER_CONSTANT, value=(255, 255, 255)) for t in tiles]
        sheet = cv2.hconcat(tiles)
        prompt = ORIENTATION_PROMPT.format(n=2) + "\n(The two candidates are shown side by side in one image, labelled 1 and 2 in red.)"
        try:
            ans = self.vlm.generate_json(prompt, [sheet], system=SYSTEM_PROMPT, max_tokens=200)
            k = int(ans.get("upright", 1))
            return {1: 0, 2: 180}.get(k)
        except Exception as e:  # VLM trouble must never break the pipeline
            log.warning("orientation VLM call failed: %s", e)
            return None


# --------------------------------------------------------------------------- triage
class TriageAgent(Agent):
    name = "triage"
    description = "Decide whether the image is a hand drawing/diagram (vs a photo or pure text)."
    requires = ("preprocessed",)
    provides = ("triaged",)
    uses_vlm = True

    def run(self, ctx: AnalysisContext, **_: Any) -> AgentReport:
        prep = ctx.prep
        ink_frac = float(prep.ink.mean())
        hsv = cv2.cvtColor(prep.image, cv2.COLOR_BGR2HSV)
        sat = float(hsv[:, :, 1].mean())
        # long strokes = drawing structure
        n, _, stats, _ = cv2.connectedComponentsWithStats(prep.ink.astype(np.uint8), connectivity=8)
        big = int((np.maximum(stats[1:, 2], stats[1:, 3]) > 12 * prep.stroke_width).sum()) if n > 1 else 0
        score = 0.0
        score += 0.4 if 0.003 < ink_frac < 0.25 else 0.0
        score += 0.3 if sat < 70 else 0.0
        score += 0.3 if big >= 1 else 0.0
        is_drawing = score >= 0.6
        conf = score
        kind = "hand_drawn_diagram" if is_drawing else "other"
        desc = None
        method = "classical"
        if self.vlm is not None:
            try:
                ans = self.vlm.generate_json(TRIAGE_PROMPT, [prep.image], system=SYSTEM_PROMPT, max_tokens=300)
                is_drawing = bool(ans.get("is_drawing", is_drawing))
                conf = float(ans.get("confidence", conf))
                kind = str(ans.get("kind", kind))
                desc = ans.get("description")
                method = "vlm"
            except Exception as e:
                log.warning("triage VLM call failed: %s", e)
        ctx.graph.is_drawing = is_drawing
        ctx.graph.drawing_confidence = round(conf, 3)
        if desc and not ctx.graph.summary:
            ctx.graph.summary = desc
        if not is_drawing and self.config.orchestrator.stop_if_not_drawing:
            ctx.facts.add("stop")
        return AgentReport(
            self.name,
            f"is_drawing={is_drawing} ({kind}, conf {conf:.2f}, {method})",
            {"is_drawing": is_drawing, "kind": kind, "ink_fraction": round(ink_frac, 4), "saturation": round(sat, 1)},
        )


# --------------------------------------------------------------------------- shapes
def _iou_xyxy(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


class ShapeDetectionAgent(Agent):
    name = "shapes"
    description = "Find closed hand-drawn shapes and classify them (ellipse, circle, rectangle, rounded rectangle, triangle, diamond, polygon, irregular region) with classical CV and, if configured, the locally trained detector."
    requires = ("preprocessed",)
    provides = ("shapes",)

    def __init__(self, config, vlm=None):
        super().__init__(config, vlm)
        self._local = None

    def _local_detector(self):
        cfg = self.config.shapes
        if "yolo" not in cfg.detector or not cfg.weights:
            return None
        if self._local is None:
            from ..training.local_detector import LocalDetector

            self._local = LocalDetector(cfg.weights, conf=cfg.yolo_conf)
        return self._local

    def run(self, ctx: AnalysisContext, mode: str = "replace", relax: bool = False, **overrides: Any) -> AgentReport:
        cfg = self.config.shapes
        prep = ctx.prep
        params = ShapeDetectorParams(
            stroke_width=prep.stroke_width,
            heavy_width=prep.heavy_width,
            min_shape_scale=float(overrides.get("min_shape_scale", cfg.min_shape_scale)),
            gap_close_scale=float(overrides.get("gap_close_scale", cfg.gap_close_scale)),
            fit_threshold=float(overrides.get("fit_threshold", cfg.fit_threshold)),
            keep_subregions=cfg.keep_subregions,
            max_merge=cfg.max_merge,
        )
        if relax:
            params.min_shape_scale *= 0.7
            params.gap_close_scale *= 1.6
            params.bridge_gap_scale *= 1.5
            params.fit_threshold -= 0.04
        shapes: list[Shape] = []
        if cfg.detector in ("cv", "cv+yolo") or self._local_detector() is None:
            for c in detect_shapes(prep.ink, params):
                x0, y0, x1, y1 = c.bbox_xyxy
                attrs = {"fit_scores": c.fit.scores, **{k: v for k, v in c.fit.attributes.items()}}
                shapes.append(
                    Shape(
                        id="",
                        type=c.type,
                        bbox=BBox.from_xyxy(x0, y0, x1, y1).round(),
                        polygon=simplify(c.contour.tolist(), max(1.0, 0.6 * prep.stroke_width), closed=True),
                        confidence=round(float(min(1.0, c.fit.score)), 3),
                        source=["cv"],
                        attributes=attrs,
                    )
                )
        n_local = 0
        det = self._local_detector()
        if det is not None:
            local = [d for d in det.detect(prep.image) if d["class"] in {t.value for t in ShapeType}]
            n_local = len(local)
            for d in local:
                box = d["bbox"]
                match = max(shapes, key=lambda s: _iou_xyxy(s.bbox.xyxy, box), default=None)
                if match is not None and _iou_xyxy(match.bbox.xyxy, box) >= 0.5:
                    if "yolo" not in match.source:
                        match.source.append("yolo")
                    if match.type in (ShapeType.REGION, ShapeType.POLYGON, ShapeType.UNKNOWN) or d["conf"] > match.confidence:
                        match.attributes["cv_type"] = match.type.value
                        match.type = ShapeType(d["class"])
                    match.confidence = round(max(match.confidence, d["conf"]), 3)
                else:
                    poly = d.get("polygon") or []
                    b = BBox.from_xyxy(*box)
                    if len(poly) < 3:
                        from ..prompts import bbox_polygon

                        poly = bbox_polygon(b, ShapeType(d["class"]))
                    shapes.append(
                        Shape(id="", type=ShapeType(d["class"]), bbox=b.round(), polygon=[tuple(p) for p in poly], confidence=round(d["conf"], 3), source=["yolo"])
                    )
        g = ctx.graph
        if mode == "merge":
            added = 0
            for s in shapes:
                if all(_iou_xyxy(s.bbox.xyxy, o.bbox.xyxy) < 0.5 for o in g.shapes):
                    s.id = g.next_id("S")
                    g.shapes.append(s)
                    added += 1
            summary = f"re-scan ({'relaxed' if relax else 'default'}) added {added} shape(s)"
        else:
            keep = [s for s in g.shapes if "cv" not in s.source and "yolo" not in s.source]
            g.shapes = keep
            for s in shapes:
                s.id = g.next_id("S")
                g.shapes.append(s)
            summary = f"{len(shapes)} shape(s): " + ", ".join(
                f"{n} {t}" for t, n in sorted(_count(s.type.value for s in shapes).items())
            )
        if n_local:
            summary += f" (local detector: {n_local} detections)"
        return AgentReport(self.name, summary, {"n_shapes": len(g.shapes)})


def _count(items) -> dict:
    out: dict = {}
    for i in items:
        out[i] = out.get(i, 0) + 1
    return out


# --------------------------------------------------------------------------- connections
class ConnectionAgent(Agent):
    name = "connections"
    description = "Trace lines, arrows and heavy lines of identity between shapes and labels (erasing outlines first, re-joining lines cut at outlines)."
    requires = ("shapes",)
    provides = ("connections",)

    def run(self, ctx: AnalysisContext, **_: Any) -> AgentReport:
        prep = ctx.prep
        cfg = self.config.connections
        contours = [np.asarray(s.polygon) for s in ctx.graph.shapes if len(s.polygon) >= 3]
        ring = shape_ring_mask(prep.ink.shape, contours, prep.stroke_width)
        th = estimate_text_height(prep.ink & ~ring, prep.stroke_width)
        lines, text_mask, conn_mask = extract_lines(
            prep.ink,
            ring,
            LineParams(
                stroke_width=prep.stroke_width,
                heavy_ratio=cfg.heavy_ratio,
                min_line_scale=cfg.min_line_scale,
                min_length_px=2.5 * th,
            ),
        )
        ctx.artifacts.update({"ring": ring, "text_mask": text_mask, "connector_mask": conn_mask})
        g = ctx.graph
        g.connections = [c for c in g.connections if "cv" not in c.source]
        for ln in lines:
            heads = set(ln.heads)
            eps = [Endpoint(point=(round(p[0], 1), round(p[1], 1)), is_head=p in heads) for p in ln.endpoints]
            g.connections.append(
                Connection(
                    id=g.next_id("C"),
                    type=ConnectionType.ARROW if heads else ConnectionType.LINE,
                    path=[(round(x, 1), round(y, 1)) for x, y in ln.path],
                    endpoints=eps,
                    heavy=ln.heavy,
                    confidence=0.8 if ln.pieces == 1 else 0.7,
                    source=["cv"],
                    attributes={"length_px": round(ln.length, 1), "width_px": round(ln.width, 2), "pieces": ln.pieces},
                )
            )
        n_heavy = sum(1 for ln in lines if ln.heavy)
        n_arrow = sum(1 for ln in lines if ln.heads)
        return AgentReport(self.name, f"{len(lines)} connector(s): {n_heavy} heavy, {n_arrow} arrow(s)", {"n_connections": len(lines)})


# --------------------------------------------------------------------------- text layout
class TextLayoutAgent(Agent):
    name = "text_layout"
    description = "Group the remaining ink into words / text lines and locate them (does not read them)."
    requires = ("connections",)
    provides = ("text_regions",)

    def run(self, ctx: AnalysisContext, **_: Any) -> AgentReport:
        text_mask = ctx.artifacts.get("text_mask")
        if text_mask is None:
            text_mask = ctx.prep.ink
        blobs, th = group_text(text_mask, ctx.artifacts.get("ring"), ctx.prep.stroke_width)
        ctx.text_height = th
        g = ctx.graph
        g.texts = [t for t in g.texts if "cv" not in t.source]
        for b in blobs:
            x0, y0, x1, y1 = b.bbox
            box = BBox.from_xyxy(x0, y0, x1, y1)
            # an existing (VLM) text at the same spot keeps its transcription
            if any(t.bbox.iou(box) > 0.5 for t in g.texts):
                continue
            g.texts.append(TextItem(id=g.next_id("T"), bbox=box, confidence=0.5, source=["cv"]))
        return AgentReport(self.name, f"{len(blobs)} text region(s), text height ~{th:.0f}px", {"n_texts": len(blobs), "text_height": th})


# --------------------------------------------------------------------------- topology
class TopologyAgent(Agent):
    name = "topology"
    description = "Compute nesting (which shape contains which), overlaps/touching shapes, which text is inside/near which shape, and what each line end is attached to."
    requires = ("shapes",)
    provides = ("topology",)

    def run(self, ctx: AnalysisContext, **_: Any) -> AgentReport:
        update_topology(
            ctx.graph,
            ctx.sw,
            ctx.th,
            near_scale=self.config.text.near_scale,
            attach_scale=self.config.connections.attach_scale,
        )
        g = ctx.graph
        n_contains = sum(1 for r in g.relations if r.type.value == "contains")
        n_conn = sum(1 for r in g.relations if r.type.value == "connected")
        return AgentReport(
            self.name,
            f"{len(g.relations)} relation(s): {n_contains} contains, {n_conn} connected; max depth {g.stats()['max_depth']}",
            {"n_relations": len(g.relations)},
        )
