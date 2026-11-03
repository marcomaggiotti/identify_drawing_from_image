"""Human corrections -> OCR fine-tuning data.

The Benchmark tab saves a segment with its verified transcription ("Save as
correction"); synthetic pages give thousands more crops with exact text.
Both end up in one simple format, shared by every fine-tuning script::

    <dataset>/images/<name>.png
    <dataset>/manifest.jsonl        {"file": "images/<name>.png", "text": "...", "source": "...", ...}

(the author's export_finetune_dataset.py format).
"""

from __future__ import annotations

import hashlib
import json
import random
import shutil
import time
from pathlib import Path

import cv2
import numpy as np


def save_correction(dataset_dir: str | Path, crop_bgr: np.ndarray, text: str, meta: dict | None = None) -> dict:
    """Add (or update) one verified (segment image, transcription) pair; returns its record."""
    d = Path(dataset_dir)
    (d / "images").mkdir(parents=True, exist_ok=True)
    ok, buf = cv2.imencode(".png", crop_bgr)
    if not ok:
        raise ValueError("cannot encode the segment image")
    digest = hashlib.sha1(buf.tobytes()).hexdigest()[:16]
    rel = f"images/{digest}.png"
    (d / rel).write_bytes(buf.tobytes())
    rec = {"file": rel, "text": text.strip(), "source": "correction", "saved": time.strftime("%Y-%m-%dT%H:%M:%S"), **(meta or {})}
    rows = [r for r in load_manifest(d) if r["file"] != rel]  # a re-saved segment replaces its old text
    rows.append(rec)
    write_manifest(d, rows)
    return rec


def load_manifest(dataset_dir: str | Path) -> list[dict]:
    p = Path(dataset_dir) / "manifest.jsonl"
    if not p.exists():
        return []
    return [json.loads(ln) for ln in p.read_text(encoding="utf-8").splitlines() if ln.strip()]


def write_manifest(dataset_dir: str | Path, rows: list[dict]) -> None:
    p = Path(dataset_dir) / "manifest.jsonl"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")


def rows_from_synthetic(synth_dir: str | Path, split: str = "train") -> list[tuple[Path, str, dict]]:
    """(image path, text, meta) from the OCR crops of ``drawid synth``."""
    tsv = Path(synth_dir) / "ocr" / f"{split}.tsv"
    out = []
    if tsv.exists():
        for ln in tsv.read_text(encoding="utf-8").splitlines()[1:]:
            parts = ln.split("\t")
            if len(parts) >= 2 and parts[1].strip():
                out.append((Path(synth_dir) / "ocr" / parts[0], parts[1], {"source": "synthetic", "crossed_out": parts[2] == "1" if len(parts) > 2 else False}))
    return out


def build_ocr_dataset(out_dir: str | Path, synthetic: list[str] = (), corrections: list[str] = (), split: str = "train", limit: int | None = None, val_fraction: float = 0.1, seed: int = 0, repeat_corrections: int = 3) -> dict:
    """Merge synthetic crops and human corrections into one train/val dataset.

    Corrections are repeated ``repeat_corrections`` times in the training split: they are few,
    but they are the real hand, which is what the model must learn.
    """
    out = Path(out_dir)
    (out / "images").mkdir(parents=True, exist_ok=True)
    items: list[tuple[Path, str, dict]] = []
    for s in synthetic:
        items += rows_from_synthetic(s, split)
    rng = random.Random(seed)
    rng.shuffle(items)
    if limit:
        items = items[:limit]
    real: list[tuple[Path, str, dict]] = []
    for c in corrections:
        for r in load_manifest(c):
            real.append((Path(c) / r["file"], r["text"], {k: v for k, v in r.items() if k not in ("file", "text")}))
    rows_train, rows_val = [], []
    n_val_real = int(round(len(real) * val_fraction)) if len(real) >= 10 else 0
    n_val_syn = int(round(len(items) * val_fraction))
    for i, (src, text, meta) in enumerate(items + real):
        if not src.exists():
            continue
        name = f"{'s' if meta.get('source') == 'synthetic' else 'c'}{i:06d}{src.suffix or '.png'}"
        shutil.copyfile(src, out / "images" / name)
        rec = {"file": f"images/{name}", "text": text, **meta}
        is_real = meta.get("source") != "synthetic"
        is_val = (i < n_val_syn) if not is_real else (i - len(items) < n_val_real)
        if is_val:
            rows_val.append(rec)
        else:
            rows_train.extend([rec] * (repeat_corrections if is_real else 1))
    write_manifest(out, rows_train)
    (out / "manifest_val.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows_val), encoding="utf-8")
    return {"out_dir": str(out), "train": len(rows_train), "val": len(rows_val), "synthetic": len(items), "corrections": len(real)}
