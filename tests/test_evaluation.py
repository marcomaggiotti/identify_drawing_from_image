from drawing_identifier.evaluation import Evaluation, cer, map_to_original
from drawing_identifier.schema import BBox, DiagramGraph, ImageInfo, Shape, ShapeType
from drawing_identifier.synthetic import generate_sample


def test_identity_scores_perfect():
    _, g = generate_sample(5)
    ev = Evaluation()
    ev.add_image(g, g)
    r = ev.report()
    assert r["shapes"]["f1"] == 1.0 and r["texts"]["f1"] == 1.0 and r["connections"]["f1"] == 1.0
    assert r["text_cer"] == 0.0


def test_cer():
    assert cer("loves", "loves") == 0
    assert cer("loves", "lovez") == 0.2
    assert cer("", "") == 0


def _fwd(p, crop, scale, rot, size_before_rot):
    """Original -> working, as preprocessing does it."""
    x, y = (p[0] - crop[0]) * scale, (p[1] - crop[1]) * scale
    W0, H0 = size_before_rot
    if rot == 90:
        return (H0 - 1 - y, x)
    if rot == 180:
        return (W0 - 1 - x, H0 - 1 - y)
    if rot == 270:
        return (y, W0 - 1 - x)
    return (x, y)


def test_map_to_original_inverts_preprocessing():
    crop, scale, W0, H0 = (30, 50), 0.5, 400, 300
    for rot in (0, 90, 180, 270):
        w, h = (H0, W0) if rot in (90, 270) else (W0, H0)
        pts = [(100.0, 120.0), (500.0, 300.0)]
        wp = [_fwd(p, crop, scale, rot, (W0, H0)) for p in pts]
        g = DiagramGraph(image=ImageInfo(width=w, height=h, rotation=rot, scale=scale, crop=BBox(x=crop[0], y=crop[1], w=800, h=600)))
        g.shapes.append(Shape(id="S1", type=ShapeType.ELLIPSE, bbox=BBox.from_points(wp), polygon=wp))
        back = map_to_original(g).shapes[0].polygon
        for (bx, by), (ox, oy) in zip(back, pts):
            assert abs(bx - ox) < 0.6 and abs(by - oy) < 0.6, (rot, back, pts)
