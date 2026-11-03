"""Fine-tune Tesseract's LSTM on (segment image, transcription) pairs, so the cheapest
local tier of the consensus learns this particular hand.

Ported from the author's finetune_tesseract.py. Needs the Tesseract *training*
tools (``lstmtraining``, ``combine_tessdata``; on Debian/Ubuntu they come with
``apt install tesseract-ocr``) and a **float** ("best") base model, e.g.
https://github.com/tesseract-ocr/tessdata_best/raw/main/eng.traineddata
(the fast/int8 models shipped with most installers cannot be continued).

Mechanics: one WordStr box spanning each crop -> ``tesseract ... lstm.train``
-> .lstmf; ``combine_tessdata -e`` extracts the base LSTM; ``lstmtraining
--continue_from``; ``--stop_training`` writes ``<lang>.traineddata``.
Use the result as engine ``tesseract:psm7:<lang>@<out_dir>``.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

from PIL import Image


def _run(cmd: list[str]) -> str:
    env = {**os.environ, "OMP_THREAD_LIMIT": "1"}
    r = subprocess.run([str(c) for c in cmd], capture_output=True, text=True, env=env)
    if r.returncode != 0:
        raise RuntimeError(f"command failed: {' '.join(map(str, cmd))}\n{r.stdout[-1500:]}\n{r.stderr[-1500:]}")
    return r.stdout + r.stderr


def check_tools() -> list[str]:
    return [t for t in ("tesseract", "lstmtraining", "combine_tessdata") if shutil.which(t) is None]


def make_lstmf(image_path: Path, text: str, workdir: Path, name: str) -> Path:
    img = Image.open(image_path).convert("L")
    w, h = img.size
    tif = workdir / f"{name}.tif"
    img.save(tif)
    (workdir / f"{name}.box").write_text(f"WordStr 0 0 {w} {h} 0\t{' '.join(text.split())}\n", encoding="utf-8")
    _run(["tesseract", tif, workdir / name, "--psm", "7", "lstm.train"])
    out = workdir / f"{name}.lstmf"
    if not out.exists():
        raise RuntimeError(f"no .lstmf produced for {image_path}")
    return out


def finetune_tesseract(dataset_dir: str | Path, base_traineddata: str | Path, out_dir: str | Path = "models/tessdata", lang: str = "eg", iterations: int = 400, val_fraction: float = 0.1, limit: int | None = None, progress: bool = True) -> Path:
    missing = check_tools()
    if missing:
        raise RuntimeError(f"missing Tesseract training tools: {missing} (apt install tesseract-ocr, or build tesstrain)")
    base = Path(base_traineddata)
    if not base.is_file():
        raise FileNotFoundError(f"base traineddata not found: {base} (needs a *float* model from tessdata_best)")
    ds, out = Path(dataset_dir), Path(out_dir)
    items = [json.loads(ln) for ln in (ds / "manifest.jsonl").read_text(encoding="utf-8").splitlines() if ln.strip()]
    if limit:
        items = items[:limit]
    work = out / "_lstmf"
    work.mkdir(parents=True, exist_ok=True)
    lstmf: list[Path] = []
    seen: set[str] = set()
    for i, it in enumerate(items):
        if it["file"] in seen:
            continue  # repeated corrections: one example is enough for lstmtraining
        seen.add(it["file"])
        try:
            lstmf.append(make_lstmf(ds / it["file"], it["text"], work, f"ex{i:05d}"))
        except Exception as e:
            if progress:
                print(f"  skip {it['file']}: {str(e)[:120]}")
    if not lstmf:
        raise RuntimeError("no .lstmf examples were produced")
    n_val = max(1, int(len(lstmf) * val_fraction)) if len(lstmf) > 5 else 0
    train, val = lstmf[n_val:] or lstmf, lstmf[:n_val]
    (out / "train_list.txt").write_text("\n".join(map(str, train)) + "\n")
    if val:
        (out / "eval_list.txt").write_text("\n".join(map(str, val)) + "\n")
    if progress:
        print(f"{len(train)} train / {len(val)} eval examples")
    base_lstm = out / "base.lstm"
    _run(["combine_tessdata", "-e", base, base_lstm])
    ckpt = out / "checkpoint" / "finetune"
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["lstmtraining", "--continue_from", base_lstm, "--traineddata", base, "--train_listfile", out / "train_list.txt", "--model_output", ckpt, "--max_iterations", str(iterations)]
    if val:
        cmd += ["--eval_listfile", out / "eval_list.txt"]
    log = _run(cmd)
    if progress:
        print(log[-800:])
    final = out / f"{lang}.traineddata"
    _run(["lstmtraining", "--stop_training", "--continue_from", Path(str(ckpt) + "_checkpoint"), "--traineddata", base, "--model_output", final])
    return final
