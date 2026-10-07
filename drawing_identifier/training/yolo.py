"""Train the local shape/line/text detector on the synthetic dataset (ultralytics YOLO-seg)."""

from __future__ import annotations

import logging
import shutil
from pathlib import Path

log = logging.getLogger(__name__)


def train_detector(
    data_yaml: str | Path,
    model: str = "yolo11n-seg.pt",
    epochs: int = 50,
    imgsz: int = 1024,
    batch: int = 8,
    device: str | None = None,
    project: str | Path = "runs/detector",
    name: str = "train",
    workers: int = 4,
    export_to: str | Path | None = "models/detector.pt",
    **extra,
) -> Path:
    """Train and return the path of the best weights (also copied to ``export_to``).

    ``model`` may be a pretrained checkpoint (``yolo11n-seg.pt``, downloaded by
    ultralytics) or an architecture file (``yolo11n-seg.yaml``) to train from
    scratch when there is no internet access.
    """
    try:
        from ultralytics import YOLO
    except ImportError as e:  # pragma: no cover - optional dependency
        raise RuntimeError("training the detector needs ultralytics: pip install 'drawing-identifier[yolo]'") from e
    try:
        net = YOLO(model)
    except Exception as e:
        if str(model).endswith(".pt"):
            fallback = str(model)[:-3] + ".yaml"
            log.warning("could not load %s (%s); training %s from scratch", model, e, fallback)
            net = YOLO(fallback)
        else:
            raise
    args = dict(
        data=str(data_yaml),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        project=str(Path(project).resolve()),
        name=name,
        workers=workers,
        exist_ok=True,
        # documents: no colour jitter on hue, no vertical flips (text direction), mild geometry
        hsv_h=0.0,
        flipud=0.0,
        fliplr=0.0,
        degrees=3.0,
        mosaic=0.5,
        plots=False,
    )
    if device:
        args["device"] = device
    args.update(extra)
    net.train(**args)
    best = Path(net.trainer.best) if getattr(net, "trainer", None) else Path(project) / name / "weights" / "best.pt"
    if not best.exists():
        last = best.parent / "last.pt"
        best = last if last.exists() else best
    if export_to and best.exists():
        dst = Path(export_to)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(best, dst)
        log.info("copied %s -> %s", best, dst)
        return dst
    return best
