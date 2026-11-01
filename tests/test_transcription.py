"""Transcription engines, the local-first consensus cascade and the segment benchmark."""

import json
import re

import cv2
import numpy as np
import pytest

from drawing_identifier.config import ConsensusConfig, VLMProfile, load_config
from drawing_identifier.transcription import (
    CallableTranscriber,
    TesseractTranscriber,
    VLMTranscriber,
    cer,
    crop_segment,
    diff_html,
    make_variants,
    normalize,
    select_engines,
    similarity,
    vlm_traits,
    wer,
)
from drawing_identifier.transcription.bench import consensus_of_rows, rows_to_csv, rows_to_json, run_segment_benchmark, score_rows
from drawing_identifier.transcription.consensus import ConsensusCascade, plan, rank_engines
from drawing_identifier.transcription.metrics import medoid
from drawing_identifier.vlm import MockBackend

IMG = np.full((40, 160, 3), 255, np.uint8)


def engine(eid, answer, kind="vlm", local=True, size=3.0, cost=0.0, log=None, conf=None):
    """Scripted engine; ``answer`` is a string or {segment_key: string}."""

    def fn(img):
        if log is not None:
            log.append(eid)
        a = answer(img) if callable(answer) else answer
        return (a, conf) if conf is not None else a

    return CallableTranscriber(eid, fn, kind=kind, local=local, params_b=size, cost_rank=cost)


# ------------------------------------------------------------------ metrics
def test_metrics():
    assert normalize("  A’  is  “r” ") == "A' is \"r"
    assert normalize("Loves", case_sensitive=False) == "loves"
    assert similarity("loves", "loves") == 1.0 and similarity("", "") == 1.0
    assert 0.79 < similarity("loves", "lovcs") < 0.81
    assert cer("loves", "lovcs") == pytest.approx(0.2)
    assert wer("some A is r", "some A is") == pytest.approx(0.25)
    h = diff_html("every", "evry B")
    assert "<del" in h and "<ins" in h
    assert medoid(["abc", "abd", "xyz"]) in (0, 1)


def test_crop_and_variants():
    img = np.zeros((100, 200, 3), np.uint8)
    seg = crop_segment(img, (150, 80, 50, 20), pad=0, rotation=90)  # corners in any order, rotated
    assert seg.shape[:2] == (100, 60)
    v = make_variants(np.full((20, 50, 3), 255, np.uint8))
    assert set(v) == {"original", "enhanced", "binarized"} and v["original"].shape[0] >= 64


# ------------------------------------------------------------------ ordering
def test_vlm_traits_and_tiers():
    claude = vlm_traits(VLMProfile(provider="anthropic", model="claude-opus-5-5"))
    haiku = vlm_traits(VLMProfile(provider="anthropic", model="claude-haiku-4-5-20251001"))
    assert claude[0] is False and claude[2] == 5 and haiku[2] == 1
    assert vlm_traits(VLMProfile(provider="ollama", model="qwen2.5vl:7b"))[:2] == (True, 7.0)
    assert vlm_traits(VLMProfile(provider="hf", model="Qwen/Qwen2.5-VL-3B-Instruct"))[:2] == (True, 3.0)
    assert vlm_traits(VLMProfile(provider="openai_compatible", model="x", base_url="http://localhost:8000/v1"))[0] is True
    assert vlm_traits(VLMProfile(provider="openai_compatible", model="x", base_url="https://openrouter.ai/api/v1"))[0] is False
    assert vlm_traits(VLMProfile(provider="anthropic", model="custom", local=True, params_b=2, cost_rank=0)) == (True, 2.0, 0.0)


def test_rank_local_first_then_size_then_cost():
    es = [
        engine("vlm:opus", "", local=False, size=100, cost=5),
        engine("vlm:qwen7b", "", size=7),
        engine("vlm:haiku", "", local=False, size=100, cost=1),
        engine("ocr", "", kind="ocr", size=0.02),
        engine("vlm:qwen3b", "", size=3),
    ]
    assert [e.id for e in rank_engines(es)] == ["ocr", "vlm:qwen3b", "vlm:qwen7b", "vlm:haiku", "vlm:opus"]
    assert [e.tier for e in rank_engines(es)] == [0, 1, 2, 3, 5]
    steps = plan(es, ConsensusConfig(allow_remote=False))
    assert [s[0].id for s in steps] == ["ocr", "vlm:qwen3b", "vlm:qwen7b"]


# ------------------------------------------------------------------ cascade
def test_cascade_stops_before_the_cloud_when_local_models_agree():
    log = []
    es = [
        engine("tesseract:psm7", "lovcs", kind="ocr", size=0.02, log=log),
        engine("easyocr:en", "l0ves", kind="ocr", size=0.1, log=log),
        engine("vlm:qwen3b", "loves", size=3, log=log),
        engine("vlm:qwen7b", "loves", size=7, log=log),
        engine("vlm:haiku", "loves", local=False, size=100, cost=1, log=log),
        engine("vlm:opus", "loves", local=False, size=100, cost=5, log=log),
    ]
    res = ConsensusCascade(es).run(IMG)
    assert res.status == "consensus" and res.text == "loves"
    assert "vlm:haiku" not in log and "vlm:opus" not in log
    assert res.remote_calls == 0 and set(res.skipped) == {"vlm:haiku", "vlm:opus"}
    assert res.tiers_run == [0, 1, 2]


def test_cascade_escalates_to_expensive_models_on_disagreement():
    log = []
    es = [
        engine("vlm:qwen3b", "reject", size=3, log=log),
        engine("vlm:qwen7b", "refects", size=7, log=log),
        engine("vlm:haiku", "rejects", local=False, size=100, cost=1, log=log),
        engine("vlm:opus", "rejects", local=False, size=100, cost=5, log=log),
    ]
    res = ConsensusCascade(es, ConsensusConfig(similarity=0.95)).run(IMG)
    assert log == ["vlm:qwen3b", "vlm:qwen7b", "vlm:haiku", "vlm:opus"]
    assert res.status == "consensus" and res.text == "rejects" and res.remote_calls == 2
    # never leave the machine when remote models are not allowed
    log.clear()
    res = ConsensusCascade(es, ConsensusConfig(similarity=0.95, allow_remote=False)).run(IMG)
    assert log == ["vlm:qwen3b", "vlm:qwen7b"] and res.status == "best_effort" and res.remote_calls == 0


def test_one_engine_cannot_agree_with_itself():
    # four Tesseract page modes reading the same thing are one witness, not four
    es = [engine(f"tesseract:psm{p}", "Some A", kind="ocr", size=0.02) for p in (6, 7, 8, 13)]
    res = ConsensusCascade(es).run(IMG)
    assert res.status == "best_effort" and res.independent == 1
    es.append(engine("easyocr:en", "Some A", kind="ocr", size=0.1))
    assert ConsensusCascade(es).run(IMG).status == "consensus"


def test_per_engine_escalation_and_errors_abstain():
    log = []

    def boom(img):
        raise RuntimeError("model offline")

    es = [
        CallableTranscriber("vlm:broken", boom, local=True, params_b=1),
        engine("vlm:a", "B", size=3, log=log),
        engine("vlm:b", "B", size=7, log=log),
        engine("vlm:c", "B", local=False, size=100, cost=3, log=log),
    ]
    res = ConsensusCascade(es, ConsensusConfig(per_engine=True)).run(IMG)
    assert res.status == "consensus" and res.text == "B" and "vlm:c" not in log
    assert any(v.error and "offline" in v.error for v in res.votes)


def test_batch_only_escalates_disputed_segments():
    cloud_saw = []

    def ocr(img):
        return "A" if img.shape[1] < 300 else "x"

    def small(img):
        return "A" if img.shape[1] < 300 else "y"

    def cloud(img):
        cloud_saw.append(img.shape[1])
        return "loves"

    es = [
        CallableTranscriber("easyocr:en", ocr, kind="ocr", params_b=0.1),
        CallableTranscriber("vlm:small", small, params_b=3),
        CallableTranscriber("vlm:big1", cloud, local=False, params_b=100, cost_rank=1),
        CallableTranscriber("vlm:big2", cloud, local=False, params_b=100, cost_rank=3),
    ]
    easy = np.full((64, 100, 3), 255, np.uint8)
    hard = np.full((64, 400, 3), 255, np.uint8)
    out = ConsensusCascade(es).run_batch({"easy": easy, "hard": hard})
    assert out["easy"].status == "consensus" and out["easy"].remote_calls == 0
    assert out["hard"].text == "loves" and out["hard"].remote_calls == 2
    assert len(cloud_saw) == 2 and all(w >= 400 for w in cloud_saw)


# ------------------------------------------------------------------ engines
def test_vlm_transcriber_batch_and_sheet_mode():
    seen = []

    def responder(prompt, images):
        seen.append((prompt, len(images)))
        ids = re.search(r"crop ids: ([^)]*)\)", prompt).group(1).split(", ")
        return {"items": [{"id": i, "text": f"w{i}", "confidence": 90, "crossed_out": "true"} for i in ids]}

    multi = VLMTranscriber(MockBackend(VLMProfile(provider="mock", model="m"), responder=responder), "m", batch_size=4)
    out = multi.read_batch({f"T{i}": IMG for i in range(6)})
    assert [out[f"T{i}"].text for i in range(6)] == [f"wT{i}" for i in range(6)]
    assert out["T0"].confidence == pytest.approx(0.9) and out["T0"].crossed_out is True
    assert [n for _, n in seen] == [4, 2]
    seen.clear()
    single = VLMTranscriber(MockBackend(VLMProfile(provider="mock", model="m", max_images_per_call=1), responder=responder), "s", batch_size=4)
    single.read_batch({f"T{i}": IMG for i in range(3)})
    assert len(seen) == 1 and seen[0][1] == 1 and "sheet" in seen[0][0]
    # a single object instead of {"items": [...]} is accepted for one crop
    one = VLMTranscriber(MockBackend(VLMProfile(provider="mock", model="m"), responder=lambda p, i: {"text": "loves"}), "o")
    assert one.read(IMG).text == "loves"


def test_select_engines():
    cfg = load_config()
    sel = select_engines(cfg, ["nope", "vlm:mock", "tesseract:psm7"])
    assert sel.unknown == ["nope"]
    assert "vlm:mock" in [e.id for e in sel.engines]
    auto = select_engines(cfg, "auto")
    assert all(not e.needs_download for e in auto.engines)


tesseract_ok = TesseractTranscriber(7).available()[0]


@pytest.mark.skipif(not tesseract_ok, reason="tesseract not installed")
def test_tesseract_reads_printed_text():
    img = np.full((80, 400, 3), 255, np.uint8)
    cv2.putText(img, "loves benefactress", (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 0), 2)
    r = TesseractTranscriber(7).read(make_variants(img)["binarized"])
    assert r.ok and "loves" in r.text


# ------------------------------------------------------------------ benchmark
def test_segment_benchmark_scores_and_exports():
    es = [
        engine("tesseract:psm7", "Some A is r", kind="ocr", size=0.02),
        engine("easyocr:en", "Sone A is r", kind="ocr", size=0.1),
        engine("vlm:q", "Some A is r", size=3),
    ]
    rows = run_segment_benchmark(IMG, es, ["original", "binarized"], reference="Some A is r")
    assert len(rows) == 6
    perfect = [r for r in rows if r.engine_id != "easyocr:en"]
    assert all(r.cer == 0 and r.exact for r in perfect)
    assert all(r.cer == pytest.approx(1 / 11) for r in rows if r.engine_id == "easyocr:en")
    cons = consensus_of_rows(rows)
    assert cons.status == "consensus" and cons.text == "Some A is r"
    rows = score_rows(rows, None)  # no reference: only agreement between engines
    assert all(r.cer is None and r.agreement is not None for r in rows)
    data = json.loads(rows_to_json(rows, cons, None, None, [0, 0, 10, 10]))
    assert len(data["results"]) == 6 and data["consensus_all"]["text"] == "Some A is r"
    assert rows_to_csv(rows).count("\n") == 7


# ------------------------------------------------------------------ pipeline
def test_text_reader_consensus_mode(simple_drawing, monkeypatch):
    from drawing_identifier import Orchestrator
    import drawing_identifier.transcription as tr

    log = []
    fake = tr.engines.EngineSelection(
        engines=[
            engine("ocr", "B", kind="ocr", size=0.05, log=log),
            engine("vlm:small", "B", size=3, log=log),
            engine("vlm:cloud", "B", local=False, size=100, cost=5, log=log),
        ]
    )
    monkeypatch.setattr(tr, "select_engines", lambda *a, **k: fake)
    cfg = load_config(overrides={"default_vlm": "none", "text": {"ocr": "consensus"}})
    res = Orchestrator(cfg).analyze(simple_drawing)
    texts = [t for t in res.graph.texts if "consensus" in t.source]
    assert texts and all(t.text == "B" for t in texts)
    assert texts[0].attributes["consensus"]["status"] == "consensus"
    assert "vlm:cloud" not in log  # settled locally, the expensive model was never called
    msg = next(e.message for e in res.graph.trace if e.agent == "text_reader")
    assert "settled locally" in msg and "0 remote" in msg


# ------------------------------------------------------------------ UI callbacks / CLI
def test_benchmark_tab_callbacks(tmp_path, monkeypatch):
    from drawing_identifier import ui_bench
    from drawing_identifier.synthetic import generate_dataset

    rgb = np.full((300, 400, 3), 255, np.uint8)
    pts, box = ui_bench.click(rgb, [], (350, 250))
    assert box is None and len(pts) == 1
    pts, box = ui_bench.click(rgb, pts, (50, 40))
    assert box == (50, 40, 350, 250)
    assert ui_bench.render_page(rgb, box, {"T1": {"box": (1, 1, 20, 20)}}, None).shape == rgb.shape
    assert ui_bench.segment(rgb, box, 90, 0).shape[:2] == (300, 210)
    assert len(ui_bench.variant_gallery(ui_bench.segment(rgb, box))) == 3

    fake = __import__("drawing_identifier.transcription.engines", fromlist=["EngineSelection"]).EngineSelection(
        engines=[engine("ocr", "Some", kind="ocr"), engine("vlm:a", "Some", size=3), engine("vlm:b", "Some", local=False, size=100, cost=5)]
    )
    monkeypatch.setattr(ui_bench, "select_engines", lambda *a, **k: fake)
    cfg = load_config()
    rows, cons, _ = ui_bench.compare(cfg, ui_bench.segment(rgb, box), ["ocr"], "", ["original"], "Some", ui_bench.consensus_params())
    assert cons.status == "consensus" and "Some" in ui_bench.summary_markdown(rows, cons, "Some")
    assert "<table" in ui_bench.diff_panel(rows, "Some", "reference")
    res, _, plan_lines = ui_bench.cascade(cfg, ui_bench.segment(rgb, box), [], "", ui_bench.consensus_params())
    md = ui_bench.cascade_markdown(res, plan_lines)
    assert "consensus" in md and "vlm:b" in md  # listed as not needed
    files = ui_bench.export(rows, cons, res, "Some", box)
    assert all(p.endswith((".json", ".csv")) for p in files)

    generate_dataset(tmp_path, 2, seed=3, formats=("ocr",), workers=1, progress=False)
    img, ref, msg = ui_bench.load_synthetic_sample(str(tmp_path), "train", 0)
    assert img is not None and ref and "sample 1/" in msg


def test_cli_engines_and_transcribe(tmp_path, capsys):
    from drawing_identifier.cli import main

    assert main(["engines"]) == 0
    assert "consensus order" in capsys.readouterr().out
    if not tesseract_ok:
        pytest.skip("tesseract not installed")
    img = np.full((80, 400, 3), 255, np.uint8)
    cv2.putText(img, "loves", (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 0, 0), 3)
    p = tmp_path / "seg.png"
    cv2.imwrite(str(p), img)
    out = tmp_path / "r.json"
    assert main(["transcribe", str(p), "--engines", "tesseract:psm7,tesseract:psm8", "--reference", "loves", "--json", str(out)]) == 0
    data = json.loads(out.read_text())
    assert len(data["results"]) == 2 and data["reference"] == "loves"
    assert main(["transcribe", str(p), "--consensus", "--engines", "tesseract:psm7", "--no-remote"]) == 0


def test_ui_builds_with_benchmark_tab():
    pytest.importorskip("gradio")
    from drawing_identifier.ui import build_app

    app = build_app()
    assert "Benchmark" in str(app.config)


def test_word_level_fusion_corrects_each_engines_slips():
    from drawing_identifier.transcription.consensus import fuse_words

    readings = [("either heis not rabbed by anybody", 0.6), ("either he is not robbed bY onybody", 0.6), ("either he is not robbed by anybody", 0.5)]
    assert fuse_words(readings) == "either he is not robbed by anybody"
    es = [engine(f"vlm:m{i}", t, size=3) for i, (t, _) in enumerate(readings)]  # one tier: all three vote
    res = ConsensusCascade(es, ConsensusConfig(similarity=0.8)).run(IMG)
    assert res.status == "consensus" and res.text == "either he is not robbed by anybody"
