"""Spatial reasoning on a DiagramGraph: nesting, overlaps, text placement, line attachment."""

from __future__ import annotations

import math

import cv2
import numpy as np
from scipy.spatial import cKDTree

from ..schema import (
    DiagramGraph,
    Relation,
    RelationType,
    ShapeType,
    TextPlacement,
)
from .geometry import as_contour, polygon_area, rasterize_polygon, resample_polyline


def _contour(points) -> np.ndarray:
    return as_contour(points)


def _shape_area(s) -> float:
    return polygon_area(s.polygon) if len(s.polygon) >= 3 else s.bbox.area


def compute_containment(g: DiagramGraph, min_inside: float = 0.92) -> None:
    """Direct parent of every shape = smallest shape that contains (almost) all of its outline."""
    shapes = sorted(g.shapes, key=_shape_area)
    contours = {s.id: _contour(s.polygon) for s in shapes if len(s.polygon) >= 3}
    areas = {s.id: _shape_area(s) for s in shapes}
    for s in shapes:
        s.parent_id = None
        pts = s.polygon if len(s.polygon) >= 3 else [s.bbox.center]
        sample = pts[:: max(1, len(pts) // 40)]
        for cand in shapes:
            if cand.id == s.id or areas[cand.id] <= areas[s.id] * 1.05 or cand.id not in contours:
                continue
            c = contours[cand.id]
            inside = sum(cv2.pointPolygonTest(c, (float(x), float(y)), False) >= 0 for x, y in sample) / len(sample)
            if inside >= min_inside:
                s.parent_id = cand.id
                break  # shapes sorted by area: first hit is the smallest container
    by_id = {s.id: s for s in g.shapes}
    for s in g.shapes:
        d = 0
        p = s.parent_id
        seen = set()
        while p and p not in seen:
            seen.add(p)
            d += 1
            p = by_id[p].parent_id if p in by_id else None
        s.depth = d


def shape_pairs_overlap(g: DiagramGraph, sw: float) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    """(overlapping pairs, touching pairs) among shapes not nested in each other."""
    W = max(1, g.image.width)
    H = max(1, g.image.height)
    scale = min(1.0, 700.0 / max(W, H))
    shp = (int(H * scale) + 2, int(W * scale) + 2)
    masks = {s.id: rasterize_polygon(s.polygon, shp, scale=scale) for s in g.shapes if len(s.polygon) >= 3}
    ancestors: dict[str, set[str]] = {}
    by_id = {s.id: s for s in g.shapes}
    for s in g.shapes:
        a, p = set(), s.parent_id
        while p and p not in a:
            a.add(p)
            p = by_id[p].parent_id if p in by_id else None
        ancestors[s.id] = a
    # densified outlines: simplified polygons have few vertices along long straight edges
    step = max(1.0, sw)
    dense = {s.id: resample_polyline(list(s.polygon) + [s.polygon[0]], step) for s in g.shapes if len(s.polygon) >= 3}
    trees = {sid: cKDTree(p) for sid, p in dense.items()}
    overlaps, touches = [], []
    ids = list(masks)
    for i, a in enumerate(ids):
        for b in ids[i + 1 :]:
            if a in ancestors[b] or b in ancestors[a]:
                continue
            sa, sb = by_id[a], by_id[b]
            if sa.bbox.expand(3 * sw).iou(sb.bbox.expand(3 * sw)) <= 0:
                continue
            inter = np.logical_and(masks[a], masks[b]).sum()
            smaller = min(masks[a].sum(), masks[b].sum())
            if smaller and inter / smaller > 0.03:
                overlaps.append((a, b))
                continue
            d, _ = trees[a].query(dense[b], k=1)
            if len(d) and float(d.min()) <= 2.0 * sw + 1:
                touches.append((a, b))
    return overlaps, touches


def place_texts(g: DiagramGraph, text_h: float, near_scale: float = 2.5) -> None:
    """inside = innermost shape containing the text centre; near = shapes whose outline is close.

    Scribbles are marks over a shape, not containers, so they never hold text."""
    contours = {s.id: _contour(s.polygon) for s in g.shapes if len(s.polygon) >= 3 and s.type != ShapeType.SCRIBBLE}
    areas = {s.id: _shape_area(s) for s in g.shapes}
    near_d = near_scale * text_h
    for s in g.shapes:
        s.text_inside, s.text_near = [], []
    for t in g.texts:
        cx, cy = t.bbox.center
        containing = [sid for sid, c in contours.items() if cv2.pointPolygonTest(c, (cx, cy), False) >= 0]
        t.inside_shape_id = min(containing, key=lambda sid: areas[sid]) if containing else None
        t.near_shape_ids = []
        half = 0.5 * math.hypot(t.bbox.w, t.bbox.h)
        for sid, c in contours.items():
            if sid in containing:
                continue
            d = -cv2.pointPolygonTest(c, (cx, cy), True) - half
            if d <= near_d:
                t.near_shape_ids.append(sid)
        t.near_shape_ids.sort(key=lambda sid: -cv2.pointPolygonTest(contours[sid], (cx, cy), True))
        t.near_shape_ids = t.near_shape_ids[:3]
        if t.placement != TextPlacement.ON_LINE or t.inside_shape_id:
            t.placement = (
                TextPlacement.INSIDE if t.inside_shape_id else (TextPlacement.NEAR if t.near_shape_ids else TextPlacement.FREE)
            )
    by_id = {s.id: s for s in g.shapes}
    for t in g.texts:
        if t.inside_shape_id in by_id:
            by_id[t.inside_shape_id].text_inside.append(t.id)
        for sid in t.near_shape_ids:
            if sid in by_id:
                by_id[sid].text_near.append(t.id)


def _text_boxes(g: DiagramGraph) -> np.ndarray:
    return np.array([t.bbox.xyxy for t in g.texts], float).reshape(-1, 4)


def _box_distance(boxes: np.ndarray, px: float, py: float) -> np.ndarray:
    dx = np.maximum.reduce([boxes[:, 0] - px, np.zeros(len(boxes)), px - boxes[:, 2]])
    dy = np.maximum.reduce([boxes[:, 1] - py, np.zeros(len(boxes)), py - boxes[:, 3]])
    return np.hypot(dx, dy)


def attach_connections(g: DiagramGraph, sw: float, text_h: float, attach_scale: float = 3.0) -> None:
    """Decide what each connector end touches (text label, shape outline, shape interior)."""
    contours = {s.id: _contour(s.polygon) for s in g.shapes if len(s.polygon) >= 3}
    areas = {s.id: _shape_area(s) for s in g.shapes}
    sboxes = {s.id: s.bbox for s in g.shapes if s.id in contours}
    tol = attach_scale * sw + 2
    text_tol = max(tol, 0.6 * text_h)
    tboxes = _text_boxes(g)
    tids = [t.id for t in g.texts]
    for c in g.connections:
        if c.attributes.get("locked_ends"):
            continue  # ends were stated by a VLM, keep them
        for ep in c.endpoints:
            px, py = ep.point
            best_t = None
            if len(tids):
                d = _box_distance(tboxes, px, py)
                k = int(np.argmin(d))
                if d[k] <= text_tol:
                    best_t = tids[k]
            near = [sid for sid, b in sboxes.items() if b.contains_point((px, py), pad=tol)]
            on_boundary = [sid for sid in near if abs(cv2.pointPolygonTest(contours[sid], (px, py), True)) <= tol]
            inside = [sid for sid in near if cv2.pointPolygonTest(contours[sid], (px, py), False) >= 0]
            ep.text_id = best_t
            if on_boundary:
                ep.shape_id = min(on_boundary, key=lambda sid: areas[sid])
                ep.attachment = "text" if best_t else "boundary"
            elif inside:
                ep.shape_id = min(inside, key=lambda sid: areas[sid])
                ep.attachment = "text" if best_t else "inside"
            else:
                ep.shape_id = None
                ep.attachment = "text" if best_t else "free"
        # which outlines does the line cross?
        c.crosses = []
        if len(c.path) >= 2:
            cb = c.bbox
            pts = resample_polyline(c.path, max(2.0, sw))
            ends = np.array([ep.point for ep in c.endpoints], float).reshape(-1, 2)
            for sid, b in sboxes.items():
                if b.expand(tol).iou(cb.expand(1)) <= 0:
                    continue
                sd = np.array([cv2.pointPolygonTest(contours[sid], (float(x), float(y)), False) for x, y in pts])
                sign = sd > 0
                flips = np.nonzero(sign[1:] != sign[:-1])[0]
                for k in flips:
                    if len(ends) == 0 or np.min(np.hypot(*(ends - pts[k]).T)) > tol * 1.5:
                        c.crosses.append(sid)
                        break


def label_connections(g: DiagramGraph, text_h: float) -> None:
    """Texts written next to a connector (not at its ends) become its labels."""
    for c in g.connections:
        c.label_ids = []
    if not g.connections:
        return
    end_texts = {ep.text_id for c in g.connections for ep in c.endpoints if ep.text_id}
    paths = [(c, resample_polyline(c.path, 3.0), c.bbox.expand(1.5 * text_h)) for c in g.connections if len(c.path) >= 2]
    for t in g.texts:
        t.connection_id = None
        if t.id in end_texts:
            continue
        cx, cy = t.bbox.center
        best = None
        for c, pts, box in paths:
            if not box.contains_point((cx, cy), pad=0.5 * max(t.bbox.w, t.bbox.h)):
                continue
            d = float(np.min(np.hypot(pts[:, 0] - cx, pts[:, 1] - cy))) - 0.5 * max(t.bbox.w, t.bbox.h)
            if d <= 0.8 * text_h and (best is None or d < best[0]):
                best = (d, c)
        if best is not None:
            c = best[1]
            t.connection_id = c.id
            c.label_ids.append(t.id)
            if not t.inside_shape_id and not t.near_shape_ids:
                t.placement = TextPlacement.ON_LINE


def prune_connections(g: DiagramGraph, text_h: float) -> int:
    """Drop thin geometric strokes that connect nothing (underlines, strike-throughs, long letter strokes)."""
    keep, dropped = [], 0
    for c in g.connections:
        if "cv" in c.source and len(c.source) == 1 and not c.heavy and c.type.value == "line":
            ids = set(c.attached_ids())
            on_shape = any(ep.attachment == "boundary" for ep in c.endpoints)
            length = float(c.attributes.get("length_px", 0.0))
            if len(ids) < 2 and not on_shape and not c.crosses and length < 12 * text_h:
                dropped += 1
                continue
        keep.append(c)
    g.connections = keep
    return dropped


def build_relations(g: DiagramGraph, overlaps, touches) -> None:
    rel: dict[tuple, Relation] = {}

    def add(r: Relation):
        rel.setdefault(r.key(), r)

    for s in g.shapes:
        if s.parent_id:
            add(Relation(type=RelationType.CONTAINS, subject=s.parent_id, object=s.id))
    for t in g.texts:
        if t.inside_shape_id:
            add(Relation(type=RelationType.CONTAINS, subject=t.inside_shape_id, object=t.id))
        for sid in t.near_shape_ids:
            add(Relation(type=RelationType.NEAR, subject=t.id, object=sid))
        if t.connection_id:
            add(Relation(type=RelationType.LABELS, subject=t.id, object=t.connection_id))
    for a, b in overlaps:
        add(Relation(type=RelationType.OVERLAPS, subject=a, object=b))
    for a, b in touches:
        add(Relation(type=RelationType.TOUCHES, subject=a, object=b))
    for c in g.connections:
        ids = c.attached_ids()
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                add(Relation(type=RelationType.CONNECTED, subject=ids[i], object=ids[j], via=c.id))
    g.relations = list(rel.values())


def update_topology(g: DiagramGraph, sw: float, text_h: float, near_scale: float = 2.5, attach_scale: float = 3.0) -> None:
    compute_containment(g)
    place_texts(g, text_h, near_scale)
    attach_connections(g, sw, text_h, attach_scale)
    prune_connections(g, text_h)
    label_connections(g, text_h)
    overlaps, touches = shape_pairs_overlap(g, sw)
    build_relations(g, overlaps, touches)
    # crossed-out shapes: densely scribbled interior
    for s in g.shapes:
        dens = s.attributes.get("ink_density")
        if dens is not None and dens > 0.33 and s.type != ShapeType.REGION:
            s.crossed_out = True
