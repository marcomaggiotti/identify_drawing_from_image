"""Prompts and the compact JSON format exchanged with VLMs.

The same format is used (a) by the VLM analyst agent at inference time and
(b) as the target of the synthetic fine-tuning dataset, so a local VLM
fine-tuned with ``drawid finetune-vlm`` answers exactly what the agent expects.
"""

from __future__ import annotations

import math
from typing import Any

from .schema import (
    BBox,
    Connection,
    ConnectionType,
    DiagramGraph,
    Endpoint,
    Shape,
    ShapeType,
    TextItem,
    TextPlacement,
)

SHAPE_TYPES = [t.value for t in ShapeType if t not in (ShapeType.UNKNOWN,)]

SYSTEM_PROMPT = (
    "You are an expert at reading hand-drawn diagrams in scanned manuscripts and notebooks "
    "(logic graphs, existential graphs, Venn/Euler diagrams, flowcharts, sketches). "
    "You are precise, you never invent elements that are not visible, and you answer with strict JSON."
)

ANALYSIS_PROMPT = """Analyse the hand drawing in this image.

Identify:
1. every closed hand-drawn SHAPE (type one of: {types}), its bounding box and which shape directly contains it (nesting);
2. every TEXT written inside or near a shape (also labels on lines). Transcribe it exactly, keep single letters and subscripts (e.g. "r", "A1", "loves");
3. every CONNECTION: a drawn line or arrow joining shapes/texts (heavy lines of identity count). Give the ids of what each end touches.

Coordinates: {coords}
Ignore paragraphs of running prose unless they are inside/next to a shape; ignore ruled paper lines and page borders.

Return ONLY this JSON:
{{"is_drawing": true,
 "shapes": [{{"id": "S1", "type": "ellipse", "bbox": [x0, y0, x1, y1], "parent": null, "crossed_out": false}}],
 "texts": [{{"id": "T1", "text": "loves", "bbox": [x0, y0, x1, y1], "inside": "S1", "near": [], "crossed_out": false}}],
 "connections": [{{"id": "C1", "type": "line", "heavy": true, "ends": ["T1", "S2"], "path": [[x, y], [x, y]]}}],
 "summary": "one sentence describing the diagram"}}"""


COORDS = {
    "xyxy_1000": "bbox = [x0, y0, x1, y1] as integers normalised to 0-1000 of the image width/height; path points are [x, y] on the same scale.",
    "yxyx_1000": "bbox = [y0, x0, y1, x1] as integers normalised to 0-1000 of the image height/width; path points are [y, x] on the same scale.",
    "xyxy_pixels": "bbox = [x0, y0, x1, y1] in pixels of the image you are given; path points are [x, y] pixels.",
}


def analysis_prompt(bbox_format: str = "xyxy_1000") -> str:
    """The analysis request; the JSON example keeps xyxy order, the text states the coordinate convention."""
    return ANALYSIS_PROMPT.format(types=", ".join(SHAPE_TYPES), coords=COORDS.get(bbox_format, COORDS["xyxy_1000"]))


TRIAGE_PROMPT = """Look at this image and decide whether it contains a drawing or diagram made by hand
(sketch, diagram with shapes/lines, possibly with handwritten text) as opposed to a photograph or pure text.
Return ONLY JSON: {"is_drawing": true|false, "kind": "hand_drawn_diagram|sketch|printed_diagram|handwritten_text|photo|other", "confidence": 0.0-1.0, "description": "short"}"""

ORIENTATION_PROMPT = """These {n} images are the same page rotated by different angles (labelled 1..{n} in order).
In which one is the handwriting upright and readable left-to-right?
Return ONLY JSON: {{"upright": <number 1..{n}>, "confidence": 0.0-1.0}}"""

TEXT_READ_PROMPT = """Each of the {n} images is a crop of handwritten text from a scanned manuscript (crop ids: {ids}, in order).
Transcribe each crop exactly as written (keep math symbols, single letters, subscripts as e.g. A1, primes as A').
If a word is struck through, still transcribe it and mark crossed_out. If a crop has no text, use "".
Return ONLY JSON: {{"items": [{{"id": "<crop id>", "text": "...", "crossed_out": false, "confidence": 0.0-1.0}}]}}"""

VERIFY_PROMPT = """You are checking an automatic analysis of a hand drawing.
Image 1: the original drawing. Image 2: the same drawing with the detections overlaid:
shapes are outlined and labelled S<n>, connections are drawn as polylines labelled C<n>, texts are boxed and labelled T<n>.

Current analysis (JSON):
{analysis}

Find mistakes and return ONLY a JSON object (no comments) with these keys; use [] or {{}} when nothing changes:
- "remove_shapes": ids of detections that are not real shapes (letters, page border...), e.g. ["S3"]
- "retype_shapes": wrong shape types, e.g. {{"S2": "rectangle"}}
- "add_shapes": clearly visible shapes that were missed, e.g. [{{"type": "ellipse", "bbox": [x0, y0, x1, y1]}}] with bbox normalised 0-1000
- "text_corrections": wrong or missing transcriptions, e.g. {{"T4": "loves"}}
- "remove_texts": e.g. ["T9"]
- "remove_connections": e.g. ["C2"]
- "add_connections": e.g. [{{"ends": ["T1", "S2"], "type": "line"}}]
- "rerun": optionally ["shapes"] if many shapes were missed, ["text_reader"] if most text is wrong
- "is_complete": true when the analysis is essentially right, otherwise false
- "issues": short description of each problem
- "summary": one or two sentences describing what the diagram shows
Example: {{"remove_shapes": [], "retype_shapes": {{}}, "add_shapes": [], "text_corrections": {{}}, "remove_texts": [], "remove_connections": [], "add_connections": [], "rerun": [], "is_complete": true, "issues": [], "summary": "..."}}"""

PLANNER_PROMPT = """You coordinate a team of agents that analyse a hand drawing.
Goal: find every shape (and its type), how shapes are nested/connected, and what is written inside and near them.

Agents you can call (only those marked ready can run now):
{agents}

Current state:
{state}

Recent history:
{history}

Choose the next agent to run, or "finish" when the analysis is complete and verified.
Return ONLY JSON: {{"action": "<agent name or finish>", "params": {{}}, "reason": "short"}}"""


# ----------------------------------------------------------------- coordinates
def to_norm_box(b: BBox, w: int, h: int) -> list[int]:
    x0, y0, x1, y1 = b.xyxy
    return [
        int(round(1000 * x0 / max(w, 1))),
        int(round(1000 * y0 / max(h, 1))),
        int(round(1000 * x1 / max(w, 1))),
        int(round(1000 * y1 / max(h, 1))),
    ]


def _scale_for(fmt: str, w: int, h: int, sent: tuple[int, int] | None) -> tuple[float, float]:
    """Multipliers from the model's coordinate space to analysis-image pixels."""
    if fmt == "xyxy_pixels":
        if sent and sent[0] and sent[1]:
            return w / float(sent[0]), h / float(sent[1])  # the model saw a resized copy
        return 1.0, 1.0
    return w / 1000.0, h / 1000.0


def from_model_point(p: Any, w: int, h: int, fmt: str = "xyxy_1000", sent: tuple[int, int] | None = None) -> tuple[float, float] | None:
    try:
        a, b = float(p[0]), float(p[1])
    except Exception:
        return None
    x, y = (b, a) if fmt == "yxyx_1000" else (a, b)
    sx, sy = _scale_for(fmt, w, h, sent)
    return (min(max(0.0, x * sx), w), min(max(0.0, y * sy), h))


def from_model_box(box: Any, w: int, h: int, fmt: str = "xyxy_1000", sent: tuple[int, int] | None = None) -> BBox | None:
    try:
        vals = [float(v) for v in box][:4]
    except Exception:
        return None
    if len(vals) != 4:
        return None
    if fmt == "yxyx_1000":
        y0, x0, y1, x1 = vals
    else:
        x0, y0, x1, y1 = vals
    if fmt != "xyxy_pixels" and max(vals) <= 1.0:
        # tolerate models that answer 0..1 fractions
        x0, y0, x1, y1 = x0 * 1000, y0 * 1000, x1 * 1000, y1 * 1000
    sx, sy = _scale_for(fmt, w, h, sent)
    x0, x1, y0, y1 = x0 * sx, x1 * sx, y0 * sy, y1 * sy
    b = BBox.from_xyxy(max(0, x0), max(0, y0), min(w, x1), min(h, y1))
    return b if b.w > 1 and b.h > 1 else None


def _ref(v: Any) -> str | None:
    """Element reference from a model answer (ids may come back as ints)."""
    if v is None or v is False or isinstance(v, (dict, list)):
        return None
    s = str(v).strip()
    return s or None


def _as_bool(v: Any, default: bool | None = False) -> bool | None:
    if isinstance(v, bool):
        return v
    if v is None:
        return default
    t = str(v).strip().lower()
    if t in ("true", "yes", "y", "1"):
        return True
    if t in ("false", "no", "n", "0", ""):
        return False if t else default
    return default


def bbox_polygon(b: BBox, shape_type: ShapeType, n: int = 48) -> list[tuple[float, float]]:
    """Approximate outline for a shape known only by its bounding box."""
    cx, cy = b.center
    if shape_type in (ShapeType.ELLIPSE, ShapeType.CIRCLE, ShapeType.REGION, ShapeType.UNKNOWN, ShapeType.SCRIBBLE):
        return [
            (cx + b.w / 2 * math.cos(2 * math.pi * i / n), cy + b.h / 2 * math.sin(2 * math.pi * i / n))
            for i in range(n)
        ]
    x0, y0, x1, y1 = b.xyxy
    if shape_type == ShapeType.TRIANGLE:
        return [(cx, y0), (x1, y1), (x0, y1)]
    if shape_type == ShapeType.DIAMOND:
        return [(cx, y0), (x1, cy), (cx, y1), (x0, cy)]
    return [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]


def _shape_type(value: Any) -> ShapeType:
    v = str(value or "unknown").lower().strip().replace(" ", "_").replace("-", "_")
    aliases = {
        "oval": "ellipse",
        "cut": "ellipse",
        "square": "rectangle",
        "box": "rectangle",
        "rect": "rectangle",
        "roundrect": "rounded_rectangle",
        "rounded_rect": "rounded_rectangle",
        "rhombus": "diamond",
        "loop": "region",
        "closed_curve": "region",
        "blob": "region",
    }
    v = aliases.get(v, v)
    try:
        return ShapeType(v)
    except ValueError:
        return ShapeType.UNKNOWN


# ------------------------------------------------------------- graph <-> JSON
def graph_to_vlm_target(g: DiagramGraph, w: int | None = None, h: int | None = None, max_path_points: int = 6) -> dict:
    w = w or g.image.width
    h = h or g.image.height
    shapes = [
        {
            "id": s.id,
            "type": s.type.value,
            "bbox": to_norm_box(s.bbox, w, h),
            "parent": s.parent_id,
            "crossed_out": s.crossed_out,
        }
        for s in g.shapes
    ]
    texts = [
        {
            "id": t.id,
            "text": t.text or "",
            "bbox": to_norm_box(t.bbox, w, h),
            "inside": t.inside_shape_id,
            "near": list(t.near_shape_ids),
            "crossed_out": t.crossed_out,
        }
        for t in g.texts
    ]
    conns = []
    for c in g.connections:
        pts = c.path or [ep.point for ep in c.endpoints]
        if len(pts) > max_path_points:
            step = (len(pts) - 1) / (max_path_points - 1)
            pts = [pts[round(i * step)] for i in range(max_path_points)]
        ends = [ep.text_id or ep.shape_id for ep in c.endpoints]
        conns.append(
            {
                "id": c.id,
                "type": c.type.value,
                "heavy": c.heavy,
                "ends": ends,
                "path": [[int(round(1000 * x / w)), int(round(1000 * y / h))] for x, y in pts],
            }
        )
    return {
        "is_drawing": True if g.is_drawing is None else g.is_drawing,
        "shapes": shapes,
        "texts": texts,
        "connections": conns,
        "summary": g.summary or "",
    }


def parse_vlm_graph(
    data: dict,
    w: int,
    h: int,
    bbox_format: str = "xyxy_1000",
    source: str = "vlm",
    sent_size: tuple[int, int] | None = None,
) -> DiagramGraph:
    """Turn the model's JSON (see ANALYSIS_PROMPT) into a DiagramGraph in pixel coordinates.

    ``sent_size`` is the size of the image the model actually saw (needed for pixel bbox formats).
    """
    g = DiagramGraph()
    g.image.width, g.image.height = w, h
    if not isinstance(data, dict):
        return g
    g.is_drawing = _as_bool(data.get("is_drawing"), None)
    g.summary = data.get("summary") or None
    ids: set[str] = set()
    for i, raw in enumerate(data.get("shapes") or [], 1):
        if not isinstance(raw, dict):
            continue
        b = from_model_box(raw.get("bbox"), w, h, bbox_format, sent_size)
        if b is None:
            continue
        st = _shape_type(raw.get("type"))
        sid = str(raw.get("id") or f"S{i}")
        ids.add(sid)
        g.shapes.append(
            Shape(
                id=sid,
                type=st,
                bbox=b,
                polygon=bbox_polygon(b, st),
                confidence=0.6,
                parent_id=_ref(raw.get("parent")),
                crossed_out=bool(_as_bool(raw.get("crossed_out"), False)),
                source=[source],
            )
        )
    for i, raw in enumerate(data.get("texts") or [], 1):
        if not isinstance(raw, dict):
            continue
        b = from_model_box(raw.get("bbox"), w, h, bbox_format, sent_size)
        if b is None:
            continue
        inside = _ref(raw.get("inside"))
        near_raw = raw.get("near") or []
        near = [r for r in (_ref(x) for x in (near_raw if isinstance(near_raw, list) else [near_raw])) if r]
        placement = TextPlacement.INSIDE if inside else (TextPlacement.NEAR if near else TextPlacement.FREE)
        tid = str(raw.get("id") or f"T{i}")
        ids.add(tid)
        g.texts.append(
            TextItem(
                id=tid,
                text=str(raw.get("text") or "") or None,
                bbox=b,
                placement=placement,
                inside_shape_id=inside,
                near_shape_ids=near,
                crossed_out=bool(_as_bool(raw.get("crossed_out"), False)),
                confidence=0.6,
                source=[source],
            )
        )
    shape_ids = {s.id for s in g.shapes}
    text_ids = {t.id for t in g.texts}
    for i, raw in enumerate(data.get("connections") or [], 1):
        if not isinstance(raw, dict):
            continue
        path = []
        for p in raw.get("path") or []:
            q = from_model_point(p, w, h, bbox_format, sent_size)
            if q is not None:
                path.append(q)
        ends_raw = raw.get("ends") or []
        ends = list(ends_raw) if isinstance(ends_raw, list) else []
        eps = []
        for j, ref in enumerate(ends):
            pt = path[0] if (j == 0 and path) else (path[-1] if path else (0.0, 0.0))
            ref = _ref(ref)
            eps.append(
                Endpoint(
                    point=pt,
                    shape_id=ref if ref in shape_ids else None,
                    text_id=ref if ref in text_ids else None,
                    attachment="text" if ref in text_ids else ("boundary" if ref in shape_ids else "free"),
                    is_head=(str(raw.get("type")) == "arrow" and j == len(ends) - 1),
                )
            )
        ctype = ConnectionType.ARROW if str(raw.get("type", "line")).lower() == "arrow" else ConnectionType.LINE
        g.connections.append(
            Connection(
                id=str(raw.get("id") or f"C{i}"),
                type=ctype,
                path=path,
                endpoints=eps,
                heavy=bool(_as_bool(raw.get("heavy"), False)),
                confidence=0.6,
                source=[source],
            )
        )
    # drop dangling parent references
    for s in g.shapes:
        if s.parent_id not in shape_ids:
            s.parent_id = None
    for t in g.texts:
        if t.inside_shape_id not in shape_ids:
            t.inside_shape_id = None
        t.near_shape_ids = [x for x in t.near_shape_ids if x in shape_ids]
    return g
