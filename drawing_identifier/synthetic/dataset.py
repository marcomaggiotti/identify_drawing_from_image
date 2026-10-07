"""Build a synthetic training set and export it for the local models.

Layout of ``out_dir``::

    images/{train,val}/syn_000001.jpg     page images
    annotations/{train,val}/*.json        full ground truth (DiagramGraph JSON)
    labels/{train,val}/*.txt              YOLO segmentation labels  -> data.yaml
    data.yaml                             ultralytics dataset file
    coco_{train,val}.json                 COCO instance segmentation
    vlm_{train,val}.jsonl                 chat samples to fine-tune a VLM (same JSON the agents ask for)
    ocr/{train,val}/*.png + ocr/{split}.tsv   handwriting crops + transcription (TrOCR etc.)
"""

from __future__ import annotations

import json
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np

from ..prompts import SYSTEM_PROMPT, analysis_prompt, graph_to_vlm_target
from ..schema import DiagramGraph
from .generator import SynthConfig, generate_sample

DETECTOR_CLASSES = [
    "ellipse",
    "circle",
    "rectangle",
    "rounded_rectangle",
    "triangle",
    "diamond",
    "polygon",
    "scribble",
    "line",
    "arrow",
    "text",
]
CLASS_ID = {n: i for i, n in enumerate(DETECTOR_CLASSES)}
ALL_FORMATS = ("yolo", "coco", "vlm", "ocr")


def _poly_of_polyline(paths: list[list[tuple[float, float]]], width: float, shape: tuple[int, int]) -> list[tuple[float, float]]:
    """Outline polygon of a thick polyline (a connector as a segmentation mask)."""
    pts = [np.asarray(p, float) for p in paths if len(p) >= 2]
    if not pts:
        return []
    allp = np.vstack(pts)
    x0, y0 = np.floor(allp.min(axis=0) - width - 2).astype(int)
    x1, y1 = np.ceil(allp.max(axis=0) + width + 2).astype(int)
    m = np.zeros((max(1, y1 - y0), max(1, x1 - x0)), np.uint8)
    for p in pts:
        cv2.polylines(m, [np.round(p - [x0, y0]).astype(np.int32).reshape(-1, 1, 2)], False, 1, max(2, int(round(width))))
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return []
    c = max(cnts, key=cv2.contourArea)
    c = cv2.approxPolyDP(c, 1.0, True).reshape(-1, 2) + [x0, y0]
    return [(float(x), float(y)) for x, y in c]


def detector_objects(g: DiagramGraph) -> list[dict]:
    """Objects for detector training: shapes, connectors and text boxes with polygons."""
    objs = []
    for s in g.shapes:
        if s.type.value not in CLASS_ID or len(s.polygon) < 3:
            continue
        objs.append({"class": s.type.value, "polygon": list(s.polygon), "bbox": s.bbox.xyxy})
    for c in g.connections:
        paths = [c.path] + list(c.attributes.get("branches", []))
        width = max(4.0, 1.6 * float(c.attributes.get("width_px", 3.0)))
        poly = _poly_of_polyline(paths, width, (g.image.height, g.image.width))
        if len(poly) >= 3:
            xs = [p[0] for p in poly]
            ys = [p[1] for p in poly]
            objs.append({"class": c.type.value, "polygon": poly, "bbox": (min(xs), min(ys), max(xs), max(ys))})
    for t in g.texts:
        x0, y0, x1, y1 = t.bbox.xyxy
        objs.append({"class": "text", "polygon": [(x0, y0), (x1, y0), (x1, y1), (x0, y1)], "bbox": (x0, y0, x1, y1), "text": t.text, "crossed_out": t.crossed_out})
    return objs


def vlm_view(g: DiagramGraph) -> DiagramGraph:
    """What the analysis prompt asks the VLM to report: running prose that is neither inside
    nor next to a shape (nor a line label/end) is left out, as the prompt says to ignore it."""
    used = {ep.text_id for c in g.connections for ep in c.endpoints if ep.text_id}
    keep = [
        t
        for t in g.texts
        if "para" not in t.source or t.inside_shape_id or t.near_shape_ids or t.connection_id or t.id in used
    ]
    v = g.model_copy(deep=True)
    v.texts = [t.model_copy(deep=True) for t in keep]
    return v


def _yolo_lines(objs: list[dict], w: int, h: int) -> list[str]:
    lines = []
    for o in objs:
        pts = np.asarray(o["polygon"], float)
        pts[:, 0] = np.clip(pts[:, 0] / w, 0, 1)
        pts[:, 1] = np.clip(pts[:, 1] / h, 0, 1)
        lines.append(f"{CLASS_ID[o['class']]} " + " ".join(f"{v:.5f}" for v in pts.ravel()))
    return lines


def _area(poly) -> float:
    p = np.asarray(poly, float)
    x, y = p[:, 0], p[:, 1]
    return float(0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def _one(args) -> dict:
    idx, seed, split, out_dir, formats, cfg_dict = args
    cfg = SynthConfig(**cfg_dict)
    img, g = generate_sample(seed, cfg)
    out = Path(out_dir)
    name = f"syn_{idx:06d}"
    g.image.path = f"images/{split}/{name}.jpg"
    cv2.imwrite(str(out / "images" / split / f"{name}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 92])
    (out / "annotations" / split / f"{name}.json").write_text(g.to_json(include_trace=False), encoding="utf-8")
    h, w = img.shape[:2]
    objs = detector_objects(g)
    rec: dict = {"idx": idx, "name": name, "split": split, "width": w, "height": h, "seed": seed}
    if "yolo" in formats:
        (out / "labels" / split / f"{name}.txt").write_text("\n".join(_yolo_lines(objs, w, h)) + "\n", encoding="utf-8")
    if "coco" in formats:
        rec["coco"] = [
            {
                "category_id": CLASS_ID[o["class"]] + 1,
                "segmentation": [[round(float(v), 1) for v in np.asarray(o["polygon"]).ravel()]],
                "bbox": [round(o["bbox"][0], 1), round(o["bbox"][1], 1), round(o["bbox"][2] - o["bbox"][0], 1), round(o["bbox"][3] - o["bbox"][1], 1)],
                "area": round(_area(o["polygon"]), 1),
                "iscrowd": 0,
                "attributes": {k: o[k] for k in ("text", "crossed_out") if k in o},
            }
            for o in objs
        ]
    if "vlm" in formats:
        target = graph_to_vlm_target(vlm_view(g))
        rec["vlm"] = {
            "id": name,
            "images": [g.image.path],
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": "<image>\n" + analysis_prompt()},
                {"role": "assistant", "content": json.dumps(target, ensure_ascii=False, separators=(",", ":"))},
            ],
        }
    if "ocr" in formats:
        rows = []
        for t in g.texts:
            if not t.text:
                continue
            x0, y0, x1, y1 = (int(round(v)) for v in t.bbox.xyxy)
            pad = max(3, int(0.15 * (y1 - y0)))
            crop = img[max(0, y0 - pad) : min(h, y1 + pad), max(0, x0 - pad) : min(w, x1 + pad)]
            if crop.size == 0 or crop.shape[0] < 4 or crop.shape[1] < 4:
                continue
            # undo a 90-degree page rotation so the handwriting in the crop is upright
            if g.image.content_rotation == 90:
                crop = cv2.rotate(crop, cv2.ROTATE_90_COUNTERCLOCKWISE)
            elif g.image.content_rotation == 270:
                crop = cv2.rotate(crop, cv2.ROTATE_90_CLOCKWISE)
            fn = f"{name}_{t.id}.png"
            cv2.imwrite(str(out / "ocr" / split / fn), crop)
            rows.append(f"{split}/{fn}\t{t.text}\t{int(t.crossed_out)}")
        rec["ocr"] = rows
    return rec


def generate_dataset(
    out_dir: str | Path,
    n: int,
    seed: int = 0,
    val_fraction: float = 0.1,
    formats: Iterable[str] = ALL_FORMATS,
    cfg: SynthConfig | None = None,
    workers: int | None = None,
    progress: bool = True,
) -> dict:
    out = Path(out_dir)
    formats = tuple(formats)
    cfg = cfg or SynthConfig()
    if cfg.fonts_dir and not Path(cfg.fonts_dir).is_dir():
        cfg.fonts_dir = None
    for split in ("train", "val"):
        (out / "images" / split).mkdir(parents=True, exist_ok=True)
        (out / "annotations" / split).mkdir(parents=True, exist_ok=True)
        if "yolo" in formats:
            (out / "labels" / split).mkdir(parents=True, exist_ok=True)
        if "ocr" in formats:
            (out / "ocr" / split).mkdir(parents=True, exist_ok=True)
    n_val = max(1, int(round(n * val_fraction))) if (n > 1 and val_fraction > 0) else 0
    jobs = [(i + 1, seed * 1_000_003 + i, "val" if i < n_val else "train", str(out), formats, asdict(cfg)) for i in range(n)]
    workers = workers or max(1, min(8, (os.cpu_count() or 2)))
    recs: list[dict] = []
    if workers == 1:
        it = map(_one, jobs)
    else:
        ex = ProcessPoolExecutor(max_workers=workers)
        it = ex.map(_one, jobs, chunksize=4)
    for k, r in enumerate(it, 1):
        recs.append(r)
        if progress and (k % max(1, n // 20) == 0 or k == n):
            print(f"  generated {k}/{n}", flush=True)
    if workers != 1:
        ex.shutdown()
    recs.sort(key=lambda r: r["idx"])

    if "yolo" in formats:
        names = "\n".join(f"  {i}: {c}" for i, c in enumerate(DETECTOR_CLASSES))
        val_dir = "images/val" if n_val else "images/train"
        (out / "data.yaml").write_text(
            f"# synthetic hand-drawn diagrams (drawing_identifier)\npath: {out.resolve()}\ntrain: images/train\nval: {val_dir}\nnames:\n{names}\n",
            encoding="utf-8",
        )
    for split in ("train", "val"):
        part = [r for r in recs if r["split"] == split]
        if "coco" in formats:
            images, anns = [], []
            aid = 1
            for r in part:
                images.append({"id": r["idx"], "file_name": f"images/{split}/{r['name']}.jpg", "width": r["width"], "height": r["height"]})
                for a in r["coco"]:
                    anns.append({"id": aid, "image_id": r["idx"], **a})
                    aid += 1
            coco = {
                "info": {"description": "drawing_identifier synthetic hand drawings", "version": "1.0"},
                "images": images,
                "annotations": anns,
                "categories": [{"id": i + 1, "name": c} for i, c in enumerate(DETECTOR_CLASSES)],
            }
            (out / f"coco_{split}.json").write_text(json.dumps(coco), encoding="utf-8")
        if "vlm" in formats:
            with open(out / f"vlm_{split}.jsonl", "w", encoding="utf-8") as fh:
                for r in part:
                    fh.write(json.dumps(r["vlm"], ensure_ascii=False) + "\n")
        if "ocr" in formats:
            with open(out / "ocr" / f"{split}.tsv", "w", encoding="utf-8") as fh:
                fh.write("path\ttext\tcrossed_out\n")
                for r in part:
                    for row in r["ocr"]:
                        fh.write(row + "\n")
    summary = {
        "out_dir": str(out),
        "n_train": sum(1 for r in recs if r["split"] == "train"),
        "n_val": sum(1 for r in recs if r["split"] == "val"),
        "formats": list(formats),
        "classes": DETECTOR_CLASSES,
        "config": asdict(cfg),
    }
    (out / "dataset.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


