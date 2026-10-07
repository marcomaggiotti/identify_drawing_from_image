"""The agentic loop with a scripted VLM (no network)."""

import re

from drawing_identifier import Orchestrator, load_config
from drawing_identifier.config import VLMProfile
from drawing_identifier.schema import ShapeType
from drawing_identifier.vlm import MockBackend


class ScriptedVLM:
    """Answers like a cooperative VLM would, based on which agent is asking."""

    def __init__(self, planner_actions=None):
        self.planner_actions = list(planner_actions or [])
        self.seen = []

    def __call__(self, prompt, images):
        if "upright" in prompt:
            self.seen.append("orientation")
            return {"upright": 1, "confidence": 0.9}
        if "contains a drawing or diagram" in prompt:
            self.seen.append("triage")
            return {"is_drawing": True, "kind": "hand_drawn_diagram", "confidence": 0.95, "description": "logic diagram"}
        if prompt.startswith("Analyse the hand drawing"):
            self.seen.append("analyst")
            return {
                "is_drawing": True,
                "shapes": [{"id": "S1", "type": "ellipse", "bbox": [57, 120, 367, 410], "parent": None}],
                # a label the geometry could not see (far away from any ink)
                "texts": [{"id": "T1", "text": "extra", "bbox": [900, 900, 960, 940], "inside": None, "near": []}],
                "connections": [],
                "summary": "Nested ovals with letters and a heavy line.",
            }
        if "Transcribe each crop" in prompt or "sheet of" in prompt:
            self.seen.append("reader")
            ids = re.search(r"crop ids: ([^)]*)\)", prompt).group(1).split(", ")
            return {"items": [{"id": i, "text": f"w{i}", "confidence": 0.9} for i in ids]}
        if "checking an automatic analysis" in prompt:
            self.seen.append("verifier")
            ids = re.findall(r'"id":"(S\d+)","type":"rectangle"', prompt)
            return {
                "retype_shapes": {ids[0]: "rounded_rectangle"} if ids else {},
                "text_corrections": {"T1": "loves"},
                "is_complete": True,
                "issues": ["rectangle corners are rounded"],
                "summary": "Verified diagram.",
            }
        if "You coordinate a team of agents" in prompt:
            self.seen.append("planner")
            if self.planner_actions:
                return {"action": self.planner_actions.pop(0), "params": {}, "reason": "scripted"}
            return {"action": "finish", "reason": "done"}
        return {}


def _orc(responder, **over):
    cfg = load_config(overrides={"default_vlm": "none", **over})
    mock = MockBackend(VLMProfile(provider="mock", model="scripted"), responder=responder)
    return Orchestrator(cfg, vlm=mock), mock


def test_pipeline_with_vlm(simple_drawing):
    script = ScriptedVLM()
    orc, mock = _orc(script)
    res = orc.analyze(simple_drawing)
    g = res.graph
    assert {"orientation", "triage", "analyst", "reader", "verifier"} <= set(script.seen)
    assert g.is_drawing is True
    assert g.summary == "Verified diagram."
    # the VLM-only text was fused in, then everything got transcribed
    assert any("vlm" in t.source for t in g.texts)
    assert sum(1 for t in g.texts if t.text) >= len(g.texts) - 1
    # verifier corrections applied
    assert any(s.type == ShapeType.ROUNDED_RECTANGLE and s.attributes.get("previous_type") == "rectangle" for s in g.shapes)
    assert any(t.text == "loves" for t in g.texts)
    assert any("verifier" in e.agent for e in g.trace)
    assert any("VLMs used: mock:scripted" in n for n in g.notes)


def test_planner_mode_with_fallbacks(simple_drawing):
    # the planner asks for an agent that is not ready yet, then a bogus one, then finishes early
    script = ScriptedVLM(planner_actions=["topology", "does_not_exist", "preprocess", "shapes", "finish"])
    orc, _ = _orc(script, orchestrator={"mode": "planner", "max_steps": 25})
    res = orc.analyze(simple_drawing)
    agents = [e.agent for e in res.graph.trace]
    assert "planner" in script.seen
    assert agents.index("preprocess") < agents.index("shapes")
    assert "topology" in agents
    assert res.graph.shapes


def test_cv_only_still_works(simple_drawing, cv_config):
    res = Orchestrator(cv_config).analyze(simple_drawing)
    assert res.graph.shapes and res.graph.texts
    assert any("none (classical CV only)" in n for n in res.graph.notes)
    assert all(t.text is None for t in res.graph.texts)


def test_broken_vlm_does_not_break_pipeline(simple_drawing):
    def bad(prompt, images):
        return "I cannot comply."

    orc, _ = _orc(bad)
    res = orc.analyze(simple_drawing)
    assert res.graph.shapes
