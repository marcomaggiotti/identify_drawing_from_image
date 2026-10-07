"""Regression tests for defects found in the multi-agent review."""

import cv2
import numpy as np
import pytest
from skimage.morphology import skeletonize

from drawing_identifier import Orchestrator, load_config
from drawing_identifier.agents.orchestrator import normalize_array
from drawing_identifier.agents.reasoning import apply_corrections
from drawing_identifier.config import VLMProfile
from drawing_identifier.prompts import parse_vlm_graph
from drawing_identifier.schema import BBox, DiagramGraph, Shape, ShapeType
from drawing_identifier.vision.lines import analyse_skeleton
from drawing_identifier.vision.preprocess import load_image, preprocess
from drawing_identifier.vlm import MockBackend, extract_json


def test_integer_ids_and_string_booleans_from_vlm():
    g = parse_vlm_graph(
        {
            "is_drawing": "false",
            "shapes": [{"id": 1, "type": "oval", "bbox": [0, 0, 500, 500]}, {"id": 2, "type": "circle", "bbox": [100, 100, 200, 200], "parent": 1, "crossed_out": "false"}],
            "texts": [{"id": 3, "text": "a", "bbox": [120, 120, 150, 150], "inside": 1, "near": 2}],
            "connections": [{"ends": [3, 1], "path": [[1, 2], [3, 4]], "heavy": "true"}],
        },
        1000,
        800,
    )
    assert g.is_drawing is False
    assert g.shape("2").parent_id == "1" and g.shape("2").crossed_out is False
    assert g.texts[0].inside_shape_id == "1" and g.texts[0].near_shape_ids == ["2"]
    assert g.connections[0].heavy is True and g.connections[0].attached_ids() == ["3", "1"]


@pytest.mark.parametrize(
    "text,expected",
    [
        ('{"a": 1, // note\n "b": "http://x"}', {"a": 1, "b": "http://x"}),
        ('Readings: [{"id": "T1"}, {"id": "T2"}]', [{"id": "T1"}, {"id": "T2"}]),
    ],
)
def test_extract_json_outermost(text, expected):
    assert extract_json(text) == expected


def test_extract_json_truncated_raises():
    with pytest.raises(ValueError):
        extract_json('{"shapes": [{"id": "S1", "type": "ellipse"}, {"id": "S2"')


def test_pixel_bbox_format_uses_sent_size():
    g = parse_vlm_graph({"shapes": [{"id": "S1", "type": "ellipse", "bbox": [10, 20, 100, 200]}]}, 1000, 800, "xyxy_pixels", sent_size=(500, 400))
    assert g.shapes[0].bbox.xyxy == (20.0, 40.0, 200.0, 400.0)


def test_arrow_head_detected():
    img = np.zeros((300, 500), np.uint8)
    cv2.line(img, (30, 150), (450, 150), 1, 3)
    cv2.line(img, (450, 150), (420, 130), 1, 3)
    cv2.line(img, (450, 150), (420, 170), 1, 3)
    r = analyse_skeleton(skeletonize(img.astype(bool)), sw=3)
    assert len(r["heads"]) == 1 and r["heads"][0][0] > 400
    assert len(r["endpoints"]) == 2


def test_transparent_png_and_odd_arrays(tmp_path):
    a = np.zeros((300, 400, 4), np.uint8)
    cv2.circle(a, (200, 150), 100, (0, 0, 0, 255), 3)
    p = tmp_path / "t.png"
    cv2.imwrite(str(p), a)
    pre = preprocess(load_image(str(p)))
    assert 0 < pre.ink.mean() < 0.1
    assert normalize_array(a).shape == (300, 400, 3) and normalize_array(a).mean() > 200
    assert normalize_array(np.zeros((10, 10, 1), np.float32)).shape == (10, 10, 3)
    assert normalize_array(np.ones((10, 10), np.float64)).max() == 255


def test_malformed_verifier_answer_is_ignored():
    g = DiagramGraph()
    g.image.width, g.image.height = 100, 100
    g.shapes.append(Shape(id="S1", type=ShapeType.ELLIPSE, bbox=BBox(x=0, y=0, w=10, h=10)))
    ch = apply_corrections(g, {"retype_shapes": ["S1"], "text_corrections": "none", "remove_shapes": "S9", "add_connections": [{"ends": "x"}]})
    assert not any(ch.values()) and g.shapes[0].type == ShapeType.ELLIPSE


def test_ids_are_not_reused_after_removal():
    g = DiagramGraph()
    g.shapes.append(Shape(id=g.next_id("S"), bbox=BBox(x=0, y=0, w=1, h=1)))
    g.shapes.clear()
    assert g.next_id("S") == "S2"


def test_planner_does_not_loop_on_a_failing_agent(simple_drawing):
    def responder(prompt, images):
        if "You coordinate a team of agents" in prompt:
            return {"action": "vlm_analyst"}
        if prompt.startswith("Analyse the hand drawing"):
            return "not json at all"
        return {}

    cfg = load_config(overrides={"default_vlm": "none", "orchestrator": {"mode": "planner", "max_steps": 30}})
    res = Orchestrator(cfg, vlm=MockBackend(VLMProfile(provider="mock", model="m"), responder=responder)).analyze(simple_drawing)
    runs = [e.agent for e in res.graph.trace]
    assert runs.count("vlm_analyst") <= 3
    assert "text_reader" in runs and "verifier" in runs


def test_rerun_upstream_agent_invalidates_downstream_facts(simple_drawing, cv_config):
    orc = Orchestrator(cv_config)
    res = orc.analyze(simple_drawing)
    ctx = res.context
    orc._run(ctx, "shapes")
    assert "connections" not in ctx.facts and "topology" not in ctx.facts
    orc._run(ctx, "preprocess")
    assert ctx.facts == {"preprocessed"}
    assert not ctx.graph.shapes
