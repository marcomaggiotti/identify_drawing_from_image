from drawing_identifier.schema import BBox, Connection, DiagramGraph, Endpoint, Shape, ShapeType, TextItem


def test_bbox_iou_and_points():
    a = BBox(x=0, y=0, w=10, h=10)
    b = BBox(x=5, y=5, w=10, h=10)
    assert abs(a.iou(b) - 25 / 175) < 1e-9
    assert a.contains_point((5, 5)) and not a.contains_point((11, 5))
    assert BBox.from_points([(3, 4), (1, 9)]).xyxy == (1, 4, 3, 9)


def test_renumber_and_roundtrip():
    g = DiagramGraph()
    g.shapes = [
        Shape(id="x2", type=ShapeType.ELLIPSE, bbox=BBox(x=500, y=10, w=50, h=50)),
        Shape(id="x1", type=ShapeType.CIRCLE, bbox=BBox(x=10, y=10, w=50, h=50), parent_id="x2"),
    ]
    g.texts = [TextItem(id="t9", text="A", bbox=BBox(x=20, y=20, w=5, h=5), inside_shape_id="x1")]
    g.connections = [Connection(id="c5", endpoints=[Endpoint(point=(0, 0), shape_id="x1"), Endpoint(point=(1, 1), text_id="t9")])]
    g.renumber()
    assert [s.id for s in g.shapes] == ["S1", "S2"]
    assert g.shape("S1").type == ShapeType.CIRCLE and g.shape("S1").parent_id == "S2"
    assert g.texts[0].id == "T1" and g.texts[0].inside_shape_id == "S1"
    assert g.connections[0].attached_ids() == ["S1", "T1"]
    g2 = DiagramGraph.from_json(g.to_json())
    assert g2.model_dump() == g.model_dump()
