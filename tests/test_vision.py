import cv2
import numpy as np

from drawing_identifier.schema import RelationType, ShapeType
from drawing_identifier.vision.preprocess import estimate_rotation, preprocess
from drawing_identifier.vision.shapes import ShapeDetectorParams, detect_shapes, fit_primitives


def _mask(fn, shape=(300, 400)):
    m = np.zeros(shape, np.uint8)
    fn(m)
    return m.astype(bool)


def test_fit_primitives_types():
    assert fit_primitives(_mask(lambda m: cv2.ellipse(m, (200, 150), (150, 80), 10, 0, 360, 1, -1))).type == ShapeType.ELLIPSE
    assert fit_primitives(_mask(lambda m: cv2.circle(m, (200, 150), 100, 1, -1))).type == ShapeType.CIRCLE
    assert fit_primitives(_mask(lambda m: cv2.rectangle(m, (50, 50), (350, 250), 1, -1))).type == ShapeType.RECTANGLE
    tri = np.array([[200, 20], [360, 280], [40, 280]])
    assert fit_primitives(_mask(lambda m: cv2.fillPoly(m, [tri], 1))).type == ShapeType.TRIANGLE
    dia = np.array([[200, 20], [330, 150], [200, 280], [70, 150]])
    assert fit_primitives(_mask(lambda m: cv2.fillPoly(m, [dia], 1))).type == ShapeType.DIAMOND


def test_detect_shapes_on_clean_drawing(simple_drawing):
    p = preprocess(simple_drawing, rotation=0)
    cands = detect_shapes(p.ink, ShapeDetectorParams(stroke_width=p.stroke_width, heavy_width=p.heavy_width))
    types = sorted(c.type.value for c in cands if c.type != ShapeType.REGION)
    # 2 nested ellipses, rectangle (crossed by nothing), triangle, diamond, 2 overlapping circles
    assert types.count("ellipse") == 2
    assert types.count("circle") == 2
    assert "rectangle" in types and "triangle" in types and "diamond" in types


def test_pipeline_topology(simple_drawing, cv_config):
    from drawing_identifier import Orchestrator

    res = Orchestrator(cv_config).analyze(simple_drawing)
    g = res.graph
    rels = {(r.type, g.shape(r.subject).type.value if g.shape(r.subject) else r.subject) for r in g.relations}
    contains = [r for r in g.relations if r.type == RelationType.CONTAINS and g.shape(r.object)]
    assert any(g.shape(r.subject).type == ShapeType.ELLIPSE and g.shape(r.object).type == ShapeType.ELLIPSE for r in contains)
    assert any(r.type == RelationType.OVERLAPS for r in g.relations)
    # the heavy line connects the rectangle and the outer ellipse
    heavy = [c for c in g.connections if c.heavy]
    assert heavy, "heavy line not found"
    ends = {g.shape(e.shape_id).type for c in heavy for e in c.endpoints if e.shape_id}
    assert ShapeType.RECTANGLE in ends and ShapeType.ELLIPSE in ends
    # text inside the inner ellipse
    inner = min((s for s in g.shapes if s.type == ShapeType.ELLIPSE), key=lambda s: s.bbox.area)
    assert inner.text_inside, "text inside the inner ellipse not assigned"
    assert rels


def test_orientation_on_rotated_text():
    img = np.full((900, 1200), 255, np.uint8)
    for i in range(12):
        cv2.putText(img, "Every collection of things is r to an A", (40, 70 + i * 65), cv2.FONT_HERSHEY_SIMPLEX, 1.1, 0, 2)
    ink = img < 128
    assert estimate_rotation(ink, 2.0)[0] == 0
    rot = np.rot90(ink, k=1)  # page lying on its side
    assert estimate_rotation(rot, 2.0)[0] in (90, 270)
