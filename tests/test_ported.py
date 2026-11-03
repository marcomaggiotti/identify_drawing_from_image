"""Parts ported from the author's earlier scripts: oval ensemble, prose cleaner,
line/text discriminator, extra OCR engines, corrections dataset, index/search."""

import json
import os
import shutil

import cv2
import numpy as np
import pytest

from drawing_identifier.config import load_config


def _page_with_ovals():
    img = np.full((700, 900, 3), 250, np.uint8)
    cv2.ellipse(img, (250, 300), (180, 110), 5, 0, 360, (20, 20, 20), 4)
    cv2.ellipse(img, (250, 300), (80, 45), 5, 0, 360, (20, 20, 20), 4)
    cv2.ellipse(img, (680, 420), (130, 150), 0, 0, 360, (20, 20, 20), 4)
    cv2.putText(img, "loves", (210, 310), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (20, 20, 20), 2)
    return img


# ------------------------------------------------------------------ oval ensemble
def test_oval_ensemble_finds_drawn_ovals_with_cross_family_agreement():
    from drawing_identifier.vision.oval_ensemble import AggPolicy, detect_ovals, reward

    res = detect_ovals(_page_with_ovals())
    strong = [d for d in res.detections if d.agreement >= 3 and len(d.families) == 2]
    assert len(strong) >= 3
    gt = [
        {"center": [250, 300], "axes": [360, 220], "angle": 5},
        {"center": [250, 300], "axes": [160, 90], "angle": 5},
        {"center": [680, 420], "axes": [260, 300], "angle": 0},
    ]
    r, st = reward(strong, gt)
    assert st["tp"] == 3 and st["mean_iou"] > 0.85
    assert sum(1 for d in strong if not d.external) >= 1  # the inner oval is nested


def test_policy_tuning_and_roundtrip(tmp_path):
    from drawing_identifier.vision.oval_ensemble import AggPolicy, tune_policy

    gt = [{"center": [250, 300], "axes": [360, 220], "angle": 5}, {"center": [250, 300], "axes": [160, 90], "angle": 5}, {"center": [680, 420], "axes": [260, 300], "angle": 0}]
    pol, r, st = tune_policy([(_page_with_ovals(), gt)], trials=15, seed=1, progress=False)
    assert st["tp"] == 3 and st["fp"] <= 3
    p = tmp_path / "pol.json"
    pol.save(str(p))
    assert AggPolicy.load(str(p)) == pol


def test_spiral_fusion_in_the_pipeline():
    from drawing_identifier import Orchestrator

    cfg = load_config(overrides={"default_vlm": "none", "shapes": {"detector": "cv+spiral"}})
    res = Orchestrator(cfg).analyze(_page_with_ovals())
    assert any("spiral" in s.source for s in res.graph.shapes)
    msg = next(e.message for e in res.graph.trace if e.agent == "shapes")
    assert "oval ensemble" in msg


# ------------------------------------------------------------------ prose cleaner
def test_clean_keeps_diagrams_and_drops_far_prose():
    from drawing_identifier.vision.clean import clean_page
    from drawing_identifier.vision.preprocess import preprocess

    img = _page_with_ovals()
    for i in range(4):  # a paragraph far from any figure
        cv2.putText(img, "Every collection of A is r", (520, 60 + 40 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (20, 20, 20), 2)
    p = preprocess(img, rotation=0, crop_page=False)
    out, keep, dbg = clean_page(p.image, p.ink, p.stroke_width)
    h, w = keep.shape
    assert keep[: int(0.25 * h), int(0.55 * w) :].sum() < 0.1 * p.ink[: int(0.25 * h), int(0.55 * w) :].sum()  # prose gone
    assert keep[int(0.25 * h) :, : int(0.6 * w)].sum() > 0.8 * p.ink[int(0.25 * h) :, : int(0.6 * w)].sum()  # figure kept
    assert dbg["oval_seed"].any()


def test_runs_per_scanline_separates_lines_from_cursive():
    from drawing_identifier.vision.lines import runs_per_scanline

    line = np.zeros((20, 300), bool)
    line[8:12, :] = True
    word = np.zeros((60, 300), np.uint8)
    cv2.putText(word, "benefactress", (5, 45), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 1.4, 1, 2)
    assert runs_per_scanline(line) == pytest.approx(1.0)
    assert runs_per_scanline(word.astype(bool)) > 1.4


# ------------------------------------------------------------------ engines / corrections
def test_engine_ids_and_variants():
    from drawing_identifier.transcription import engine_from_id, make_variants

    cfg = load_config()
    t = engine_from_id("tesseract:psm7:eg@/models/tessdata", cfg)
    assert (t.psm, t.lang, t.tessdata_dir, t.tier) == (7, "eg", "/models/tessdata", 0)
    assert engine_from_id("smolvlm:500m", cfg).tier == 1  # a tiny local VLM sits in the small-local tier
    assert engine_from_id("trocr:models/trocr-eg", cfg).model == "models/trocr-eg"
    assert set(make_variants(np.full((30, 90, 3), 255, np.uint8))) == {"original", "enhanced", "binarized", "upscaled", "thickened"}
    cfg2 = load_config(overrides={"consensus": {"extra_engines": ["trocr:models/trocr-eg"]}})
    from drawing_identifier.transcription import all_engines

    assert "trocr:models/trocr-eg" in [e.id for e in all_engines(cfg2)]


def test_corrections_and_ocr_dataset(tmp_path):
    from drawing_identifier.synthetic import generate_dataset
    from drawing_identifier.transcription.corrections import build_ocr_dataset, load_manifest, save_correction

    crop = np.full((40, 120, 3), 255, np.uint8)
    cv2.putText(crop, "loves", (5, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)
    corr = tmp_path / "corr"
    save_correction(corr, crop, "lovs", {"box": [0, 0, 1, 1]})
    save_correction(corr, crop, "loves")  # same segment again: replaces the text
    rows = load_manifest(corr)
    assert len(rows) == 1 and rows[0]["text"] == "loves" and (corr / rows[0]["file"]).exists()
    synth = tmp_path / "syn"
    generate_dataset(synth, 3, seed=2, formats=("ocr",), workers=1, progress=False)
    info = build_ocr_dataset(tmp_path / "ocr", [str(synth)], [str(corr)], repeat_corrections=3)
    train = load_manifest(tmp_path / "ocr")
    assert info["corrections"] == 1 and sum(1 for r in train if r["text"] == "loves" and r["source"] == "correction") == 3
    assert all((tmp_path / "ocr" / r["file"]).exists() for r in train)


@pytest.mark.skipif(not (shutil.which("lstmtraining") and os.environ.get("DRAWID_TESSDATA_BEST")), reason="needs tesseract training tools and DRAWID_TESSDATA_BEST=<float eng.traineddata>")
def test_finetune_tesseract_end_to_end(tmp_path):
    from drawing_identifier.training.finetune_tesseract import finetune_tesseract
    from drawing_identifier.transcription.corrections import save_correction

    for i, word in enumerate(["loves", "rejects", "gives", "obeys", "hears", "calls", "fills"] * 2):
        img = np.full((48, 200, 3), 255, np.uint8)
        cv2.putText(img, word, (6, 34), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)
        save_correction(tmp_path / "ds", cv2.copyMakeBorder(img, i % 3, 0, 0, 0, cv2.BORDER_CONSTANT, value=(255, 255, 255)), word)
    model = finetune_tesseract(tmp_path / "ds", os.environ["DRAWID_TESSDATA_BEST"], tmp_path / "out", "eg", iterations=30, progress=False)
    assert model.exists() and model.name == "eg.traineddata"


# ------------------------------------------------------------------ index / search
def test_index_search_and_corrections_survive_reindex(tmp_path, simple_drawing):
    from drawing_identifier.index import index_graph, open_store, search, verify, correct
    from drawing_identifier.synthetic import generate_sample

    store, emb = open_store(str(tmp_path / "idx"))
    _, g = generate_sample(4)
    n = index_graph(store, emb, g, "page4")
    assert n == len(g.shapes)
    labelled = next(s for s in g.shapes if s.text_inside)
    word = g.text(labelled.text_inside[0]).text
    hits = search(store, emb, word, limit=3)
    assert hits and any(p["shape_id"] == labelled.id for _, _, p in hits)
    pid = hits[0][0]
    correct(store, emb, pid, "benefactress")
    assert verify(store, emb, pid)
    store2, _ = open_store(str(tmp_path / "idx"))  # persisted
    index_graph(store2, emb, g, "page4")  # re-analysis must not overwrite the human text
    p = store2.retrieve(pid)
    assert p["text"] == "benefactress" and p["human_corrected"] and p["original_text"]
    assert search(store2, emb, None, limit=50, min_depth=1) == [h for h in search(store2, emb, None, limit=50, min_depth=1)]
    assert all(p["depth"] >= 1 for _, _, p in search(store2, emb, None, limit=50, min_depth=1))


def test_cli_clean_index_search(tmp_path, capsys, simple_drawing):
    from drawing_identifier.cli import main

    img = tmp_path / "page.png"
    cv2.imwrite(str(img), _page_with_ovals())
    assert main(["clean", str(img), "--out", str(tmp_path / "clean"), "--debug"]) == 0
    assert (tmp_path / "clean" / "page_clean.png").exists() and (tmp_path / "clean" / "page_clean_debug.png").exists()
    assert main(["analyze", str(img), "--no-vlm", "--out", str(tmp_path / "out")]) == 0
    assert main(["index", str(tmp_path / "out"), "--store", str(tmp_path / "idx")]) == 0
    capsys.readouterr()
    assert main(["search", "--store", str(tmp_path / "idx"), "--type", "ellipse"]) == 0
    assert "ellipse" in capsys.readouterr().out


def test_generator_realism_options_are_consistent():
    from drawing_identifier.synthetic import SynthConfig, generate_sample

    cfg = SynthConfig(p_shaded=1.0, p_edge_clip=1.0, p_blobs=1.0)
    img, g = generate_sample(11, cfg)
    img2, g2 = generate_sample(11, cfg)
    assert np.array_equal(img, img2)
    h, w = img.shape[:2]
    assert any(s.bbox.x < 0 or s.bbox.y < 0 or s.bbox.x + s.bbox.w > w or s.bbox.y + s.bbox.h > h for s in g.shapes)
    assert any("shaded_area" in s.attributes for s in g.shapes)
    json.loads(g.to_json())
