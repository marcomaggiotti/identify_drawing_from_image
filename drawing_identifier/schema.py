"""Data model shared by every agent, the synthetic generator and the exporters.

All coordinates are pixels in the *working image* (the page after cropping,
rotation and rescaling done by the preprocessing agent).  ``ImageInfo`` keeps
the transform so results can be mapped back to the original file.
"""

from __future__ import annotations

import json
from enum import Enum
from typing import Any, Iterable

from pydantic import BaseModel, Field

Point = tuple[float, float]


class ShapeType(str, Enum):
    ELLIPSE = "ellipse"
    CIRCLE = "circle"
    RECTANGLE = "rectangle"
    ROUNDED_RECTANGLE = "rounded_rectangle"
    TRIANGLE = "triangle"
    DIAMOND = "diamond"
    POLYGON = "polygon"
    REGION = "region"  # closed area bounded by strokes, but not a regular shape
    SCRIBBLE = "scribble"  # scribbled-out area
    UNKNOWN = "unknown"


REGULAR_SHAPES = {
    ShapeType.ELLIPSE,
    ShapeType.CIRCLE,
    ShapeType.RECTANGLE,
    ShapeType.ROUNDED_RECTANGLE,
    ShapeType.TRIANGLE,
    ShapeType.DIAMOND,
    ShapeType.POLYGON,
}


class ConnectionType(str, Enum):
    LINE = "line"
    ARROW = "arrow"


class TextPlacement(str, Enum):
    INSIDE = "inside"  # written inside a shape
    NEAR = "near"  # written outside, but close to a shape boundary
    ON_LINE = "on_line"  # label of a connector
    FREE = "free"  # free text (paragraphs, notes)


class RelationType(str, Enum):
    CONTAINS = "contains"  # subject shape directly contains object (shape or text)
    OVERLAPS = "overlaps"  # two shapes intersect without containment
    TOUCHES = "touches"  # boundaries touch / are tangent
    CONNECTED = "connected"  # a connector joins subject and object
    NEAR = "near"  # text written close to a shape
    LABELS = "labels"  # text labels a connector


class BBox(BaseModel):
    x: float
    y: float
    w: float
    h: float

    @classmethod
    def from_xyxy(cls, x0: float, y0: float, x1: float, y1: float) -> "BBox":
        return cls(x=min(x0, x1), y=min(y0, y1), w=abs(x1 - x0), h=abs(y1 - y0))

    @classmethod
    def from_points(cls, pts: Iterable[Point]) -> "BBox":
        pts = list(pts)
        if not pts:
            return cls(x=0, y=0, w=0, h=0)
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        return cls.from_xyxy(min(xs), min(ys), max(xs), max(ys))

    @property
    def xyxy(self) -> tuple[float, float, float, float]:
        return (self.x, self.y, self.x + self.w, self.y + self.h)

    @property
    def center(self) -> Point:
        return (self.x + self.w / 2.0, self.y + self.h / 2.0)

    @property
    def area(self) -> float:
        return max(self.w, 0.0) * max(self.h, 0.0)

    def iou(self, other: "BBox") -> float:
        ax0, ay0, ax1, ay1 = self.xyxy
        bx0, by0, bx1, by1 = other.xyxy
        iw = max(0.0, min(ax1, bx1) - max(ax0, bx0))
        ih = max(0.0, min(ay1, by1) - max(ay0, by0))
        inter = iw * ih
        union = self.area + other.area - inter
        return inter / union if union > 0 else 0.0

    def contains_point(self, p: Point, pad: float = 0.0) -> bool:
        x0, y0, x1, y1 = self.xyxy
        return x0 - pad <= p[0] <= x1 + pad and y0 - pad <= p[1] <= y1 + pad

    def expand(self, pad: float) -> "BBox":
        return BBox(x=self.x - pad, y=self.y - pad, w=self.w + 2 * pad, h=self.h + 2 * pad)

    def round(self, nd: int = 1) -> "BBox":
        return BBox(x=round(self.x, nd), y=round(self.y, nd), w=round(self.w, nd), h=round(self.h, nd))


class Shape(BaseModel):
    id: str
    type: ShapeType = ShapeType.UNKNOWN
    bbox: BBox
    polygon: list[Point] = Field(default_factory=list, description="Outline at the stroke centre line")
    confidence: float = 1.0
    parent_id: str | None = None
    depth: int = 0
    text_inside: list[str] = Field(default_factory=list)
    text_near: list[str] = Field(default_factory=list)
    crossed_out: bool = False
    source: list[str] = Field(default_factory=list)
    attributes: dict[str, Any] = Field(default_factory=dict)


class Endpoint(BaseModel):
    point: Point
    shape_id: str | None = None
    text_id: str | None = None
    attachment: str = "free"  # boundary | inside | text | free
    is_head: bool = False  # arrow head end


class Connection(BaseModel):
    id: str
    type: ConnectionType = ConnectionType.LINE
    path: list[Point] = Field(default_factory=list)
    endpoints: list[Endpoint] = Field(default_factory=list)
    crosses: list[str] = Field(default_factory=list, description="Shapes whose boundary the line crosses")
    heavy: bool = False
    label_ids: list[str] = Field(default_factory=list)
    confidence: float = 1.0
    source: list[str] = Field(default_factory=list)
    attributes: dict[str, Any] = Field(default_factory=dict)

    def attached_ids(self) -> list[str]:
        out: list[str] = []
        for ep in self.endpoints:
            ref = ep.text_id or ep.shape_id
            if ref and ref not in out:
                out.append(ref)
        return out

    @property
    def bbox(self) -> BBox:
        return BBox.from_points(self.path or [ep.point for ep in self.endpoints])


class TextItem(BaseModel):
    id: str
    text: str | None = None
    bbox: BBox
    placement: TextPlacement = TextPlacement.FREE
    inside_shape_id: str | None = None
    near_shape_ids: list[str] = Field(default_factory=list)
    connection_id: str | None = None
    crossed_out: bool = False
    confidence: float = 1.0
    source: list[str] = Field(default_factory=list)


class Relation(BaseModel):
    type: RelationType
    subject: str
    object: str
    via: str | None = None
    confidence: float = 1.0

    def key(self) -> tuple[str, str, str]:
        return (self.type.value, self.subject, self.object)


class ImageInfo(BaseModel):
    path: str | None = None
    original_width: int = 0
    original_height: int = 0
    width: int = 0
    height: int = 0
    rotation: int = 0  # clockwise degrees applied to the original (after crop)
    crop: BBox | None = None  # page crop in original coordinates
    scale: float = 1.0  # working / original (after crop)
    stroke_width: float | None = None


class TraceEvent(BaseModel):
    step: int
    agent: str
    message: str
    data: dict[str, Any] = Field(default_factory=dict)
    seconds: float = 0.0


class DiagramGraph(BaseModel):
    image: ImageInfo = Field(default_factory=ImageInfo)
    is_drawing: bool | None = None
    drawing_confidence: float | None = None
    shapes: list[Shape] = Field(default_factory=list)
    connections: list[Connection] = Field(default_factory=list)
    texts: list[TextItem] = Field(default_factory=list)
    relations: list[Relation] = Field(default_factory=list)
    summary: str | None = None
    notes: list[str] = Field(default_factory=list)
    trace: list[TraceEvent] = Field(default_factory=list)

    # ------------------------------------------------------------------ lookup
    def shape(self, sid: str) -> Shape | None:
        return next((s for s in self.shapes if s.id == sid), None)

    def text(self, tid: str) -> TextItem | None:
        return next((t for t in self.texts if t.id == tid), None)

    def connection(self, cid: str) -> Connection | None:
        return next((c for c in self.connections if c.id == cid), None)

    def children(self, sid: str | None) -> list[Shape]:
        return [s for s in self.shapes if s.parent_id == sid]

    def roots(self) -> list[Shape]:
        return self.children(None)

    def label(self, ref: str) -> str:
        """Human readable label for a shape/text/connection id."""
        t = self.text(ref)
        if t is not None:
            return f'"{t.text}"' if t.text else ref
        s = self.shape(ref)
        if s is not None:
            return f"{s.type.value} {s.id}"
        return ref

    # --------------------------------------------------------------- id utils
    def next_id(self, prefix: str) -> str:
        existing = {s.id for s in self.shapes} | {c.id for c in self.connections} | {t.id for t in self.texts}
        i = 1
        while f"{prefix}{i}" in existing:
            i += 1
        return f"{prefix}{i}"

    def renumber(self) -> None:
        """Give stable, compact ids (S1.., C1.., T1..) ordered top-to-bottom, left-to-right."""
        def order(items, key):
            return sorted(items, key=lambda it: (round(key(it)[1] / 40), key(it)[0]))

        smap: dict[str, str] = {}
        for i, s in enumerate(order(self.shapes, lambda s: (s.bbox.x, s.bbox.y)), 1):
            smap[s.id] = f"S{i}"
        tmap: dict[str, str] = {}
        for i, t in enumerate(order(self.texts, lambda t: (t.bbox.x, t.bbox.y)), 1):
            tmap[t.id] = f"T{i}"
        cmap: dict[str, str] = {}
        for i, c in enumerate(order(self.connections, lambda c: (c.bbox.x, c.bbox.y)), 1):
            cmap[c.id] = f"C{i}"
        allmap = {**smap, **tmap, **cmap}

        def m(x):
            return allmap.get(x, x) if x else x

        for s in self.shapes:
            s.id = smap[s.id]
            s.parent_id = m(s.parent_id)
            s.text_inside = [m(x) for x in s.text_inside]
            s.text_near = [m(x) for x in s.text_near]
        for t in self.texts:
            t.id = tmap[t.id]
            t.inside_shape_id = m(t.inside_shape_id)
            t.near_shape_ids = [m(x) for x in t.near_shape_ids]
            t.connection_id = m(t.connection_id)
        for c in self.connections:
            c.id = cmap[c.id]
            c.crosses = [m(x) for x in c.crosses]
            c.label_ids = [m(x) for x in c.label_ids]
            for ep in c.endpoints:
                ep.shape_id = m(ep.shape_id)
                ep.text_id = m(ep.text_id)
        for r in self.relations:
            r.subject = m(r.subject)
            r.object = m(r.object)
            r.via = m(r.via)
        self.shapes.sort(key=lambda s: int(s.id[1:]))
        self.texts.sort(key=lambda t: int(t.id[1:]))
        self.connections.sort(key=lambda c: int(c.id[1:]))

    # ------------------------------------------------------------------- I/O
    def to_json(self, include_trace: bool = True, indent: int | None = 2) -> str:
        exclude = None if include_trace else {"trace"}
        return self.model_dump_json(indent=indent, exclude=exclude)

    @classmethod
    def from_json(cls, data: str | bytes | dict) -> "DiagramGraph":
        if isinstance(data, dict):
            return cls.model_validate(data)
        return cls.model_validate(json.loads(data))

    def stats(self) -> dict[str, Any]:
        by_type: dict[str, int] = {}
        for s in self.shapes:
            by_type[s.type.value] = by_type.get(s.type.value, 0) + 1
        return {
            "shapes": len(self.shapes),
            "shape_types": by_type,
            "connections": len(self.connections),
            "texts": len(self.texts),
            "texts_read": sum(1 for t in self.texts if t.text),
            "relations": len(self.relations),
            "max_depth": max((s.depth for s in self.shapes), default=0),
        }
