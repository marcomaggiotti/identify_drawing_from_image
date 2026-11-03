"""Transcription engines: everything that can turn an image segment into text.

Every VLM profile (or ad-hoc ``provider:model`` spec) and every local OCR
(Tesseract page modes, EasyOCR, TrOCR checkpoints) is wrapped in the same
:class:`Transcriber` interface, with metadata saying whether it runs locally,
how big it is and how expensive it is. The benchmark tab compares them; the
consensus cascade escalates through them from cheap/local to expensive/cloud.
"""

from __future__ import annotations

import logging
import re
import shutil
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Iterable
from urllib.parse import urlparse

import cv2
import numpy as np

from ..config import AppConfig, VLMProfile
from ..prompts import SYSTEM_PROMPT, TEXT_READ_PROMPT
from ..vision.preprocess import binarize, flatten_background, ink_gray, rotate

log = logging.getLogger(__name__)

TIER_NAMES = {
    0: "local OCR",
    1: "local VLM (small)",
    2: "local VLM (large)",
    3: "cloud (cheap)",
    4: "cloud (mid)",
    5: "cloud (top)",
}
TIER_WEIGHT = {0: 0.6, 1: 1.0, 2: 1.3, 3: 1.5, 4: 2.0, 5: 2.5}
VARIANTS = ("original", "enhanced", "binarized", "upscaled", "thickened")


@dataclass
class Reading:
    text: str = ""
    confidence: float | None = None
    crossed_out: bool = False
    seconds: float = 0.0
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None


# ----------------------------------------------------------------- image helpers
def crop_segment(image: np.ndarray, box: tuple[float, float, float, float] | None, pad: int = 4, rotation: int = 0) -> np.ndarray:
    """Crop ``box`` = (x0, y0, x1, y1) (any corner order) with padding, then rotate clockwise."""
    img = image
    if box is not None:
        h, w = img.shape[:2]
        x0, y0, x1, y1 = box
        x0, x1 = sorted((x0, x1))
        y0, y1 = sorted((y0, y1))
        x0, y0 = max(0, int(round(x0)) - pad), max(0, int(round(y0)) - pad)
        x1, y1 = min(w, int(round(x1)) + pad), min(h, int(round(y1)) + pad)
        if x1 - x0 >= 2 and y1 - y0 >= 2:
            img = img[y0:y1, x0:x1]
    img = np.ascontiguousarray(img)
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return rotate(img, int(rotation) % 360) if rotation else img


def make_variants(crop: np.ndarray, min_height: int = 64) -> dict[str, np.ndarray]:
    """The same segment prepared three ways; different engines prefer different ones."""
    img = crop
    if img.shape[0] < min_height:
        s = min_height / max(1, img.shape[0])
        img = cv2.resize(img, (max(1, int(img.shape[1] * s)), min_height), interpolation=cv2.INTER_CUBIC)
    gray = ink_gray(img)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4)).apply(gray)
    enhanced = cv2.cvtColor(cv2.normalize(clahe, None, 0, 255, cv2.NORM_MINMAX), cv2.COLOR_GRAY2BGR)
    flat = flatten_background(gray)
    ink = binarize(flat)
    bw = np.where(ink, 0, 255).astype(np.uint8)
    bw = cv2.copyMakeBorder(bw, 8, 8, 8, 8, cv2.BORDER_CONSTANT, value=255)
    # 2x upscale (Tesseract likes a ~30 px x-height) and a thickened copy that repairs thin,
    # broken pen strokes (renders from the author's TextAgent retry strategies)
    up = cv2.resize(bw, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    thick = cv2.erode(up, np.ones((2, 2), np.uint8))
    return {
        "original": img,
        "enhanced": enhanced,
        "binarized": cv2.cvtColor(bw, cv2.COLOR_GRAY2BGR),
        "upscaled": cv2.cvtColor(up, cv2.COLOR_GRAY2BGR),
        "thickened": cv2.cvtColor(thick, cv2.COLOR_GRAY2BGR),
    }


def sheet(crops: list[np.ndarray], ids: list[str]) -> np.ndarray:
    """Stack crops in one labelled image, for VLMs that accept a single image per call."""
    rows = []
    width = max(c.shape[1] for c in crops) + 110
    for c, tid in zip(crops, ids):
        row = np.full((c.shape[0] + 16, width, 3), 255, np.uint8)
        row[8 : 8 + c.shape[0], 100 : 100 + c.shape[1]] = c
        cv2.putText(row, tid, (6, row.shape[0] // 2 + 8), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        cv2.line(row, (0, row.shape[0] - 1), (width, row.shape[0] - 1), (200, 200, 200), 1)
        rows.append(row)
    return cv2.vconcat(rows)


# ----------------------------------------------------------------- model traits
_KNOWN_LOCAL_SIZES = {
    "llama3.2-vision": 11.0,
    "llava-phi": 3.8,
    "llava": 7.0,
    "moondream": 1.9,
    "minicpm-v": 8.0,
    "smolvlm": 2.2,
    "granite3.2-vision": 2.0,
    "gemma3": 4.0,
    "qwen2.5vl": 7.0,
    "qwen2.5-vl": 7.0,
    "florence": 0.8,
}
# relative price of hosted models, first matching substring wins (order matters)
_CLOUD_COST = [
    ("opus", 5.0),
    ("haiku", 1.0),
    ("sonnet", 3.0),
    ("flash-lite", 0.7),
    ("flash", 1.0),
    ("nano", 0.5),
    ("mini", 1.0),
    ("gemini", 3.0),
    ("gpt-5", 3.0),
    ("gpt-4.1", 3.0),
    ("gpt-4o", 3.0),
    ("o3", 4.0),
    ("o1", 4.0),
]
_LOCAL_HOSTS = ("localhost", "127.", "0.0.0.0", "::1", "192.168.", "10.", "host.docker.internal")


def _host_is_local(url: str | None) -> bool:
    if not url:
        return False
    host = (urlparse(url).hostname or "").lower()
    return host.endswith(".local") or host.startswith(_LOCAL_HOSTS) or any(host.startswith(f"172.{i}.") for i in range(16, 32))


def model_size_b(model: str) -> float | None:
    m = model.lower()
    hit = re.search(r"(?<![\d.])(\d+(?:\.\d+)?)\s*b(?![a-z])", m)
    if hit:
        return float(hit.group(1))
    for key, size in _KNOWN_LOCAL_SIZES.items():
        if key in m:
            return size
    return None


def vlm_traits(profile: VLMProfile) -> tuple[bool, float, float]:
    """(local, params_b, cost_rank) for a VLM profile; explicit profile fields win."""
    provider = profile.provider.lower()
    local = profile.local
    if local is None:
        local = provider in ("hf", "huggingface", "transformers", "ollama", "mock") or (
            provider in ("openai_compatible", "openai") and _host_is_local(profile.base_url)
        )
    size = profile.params_b if profile.params_b is not None else model_size_b(profile.model)
    if size is None:
        size = 7.0 if local else 100.0
    cost = profile.cost_rank
    if cost is None:
        if local:
            cost = 0.0
        else:
            m = profile.model.lower()
            cost = next((c for key, c in _CLOUD_COST if key in m), 3.0)
    return bool(local), float(size), float(cost)


def tier_of(kind: str, local: bool, params_b: float, cost_rank: float) -> int:
    if local:
        if kind == "ocr":
            return 0
        return 1 if params_b <= 4.0 else 2
    return 3 if cost_rank <= 1.5 else (4 if cost_rank <= 3.5 else 5)


# ----------------------------------------------------------------- engines
class Transcriber(ABC):
    """One way of reading text from an image segment."""

    kind: str = "ocr"
    preferred_variant: str = "original"
    serialize: bool = True  # local models: one call at a time
    needs_download: bool = False  # weights not on disk yet: only used when picked explicitly

    def __init__(self, id: str, label: str, local: bool = True, params_b: float = 0.05, cost_rank: float = 0.0, weight: float | None = None):
        self.id = id
        self.label = label
        self.local = local
        self.params_b = params_b
        self.cost_rank = cost_rank
        self.tier = tier_of(self.kind, local, params_b, cost_rank)
        self.weight = float(weight) if weight is not None else TIER_WEIGHT[self.tier]
        self._lock = threading.Lock()

    # -- to implement
    @abstractmethod
    def _read(self, image: np.ndarray) -> Reading: ...

    def available(self) -> tuple[bool, str]:
        return True, "ok"

    # -- public
    def read(self, image: np.ndarray) -> Reading:
        t0 = time.time()
        try:
            if self.serialize:
                with self._lock:
                    r = self._read(image)
            else:
                r = self._read(image)
        except Exception as e:  # an engine failure is a result, not a crash
            r = Reading(error=f"{e.__class__.__name__}: {e}"[:300])
        r.seconds = round(time.time() - t0, 3)
        r.text = (r.text or "").strip()
        return r

    def read_batch(self, images: dict[str, np.ndarray]) -> dict[str, Reading]:
        return {k: self.read(v) for k, v in images.items()}

    def info(self) -> dict[str, Any]:
        ok, why = self.available()
        return {
            "id": self.id,
            "label": self.label,
            "kind": self.kind,
            "local": self.local,
            "params_b": self.params_b,
            "cost_rank": self.cost_rank,
            "tier": self.tier,
            "tier_name": TIER_NAMES[self.tier],
            "weight": self.weight,
            "available": ok,
            "ready": ok and not self.needs_download,
            "status": why,
        }

    def __repr__(self) -> str:
        return f"<{type(self).__name__} {self.id} tier={self.tier}>"


def _conf(v: Any) -> float | None:
    try:
        c = float(v)
    except (TypeError, ValueError):
        return None
    if c > 1.0:
        c /= 100.0
    return max(0.0, min(1.0, c))


class VLMTranscriber(Transcriber):
    kind = "vlm"
    serialize = False

    def __init__(self, backend, name: str | None = None, batch_size: int = 6):
        local, size, cost = vlm_traits(backend.profile)
        name = name or backend.name
        super().__init__(f"vlm:{name}", f"{name} ({backend.profile.model})", local, size, cost, backend.profile.weight)
        self.backend = backend
        self.batch_size = batch_size
        self.serialize = backend.profile.provider.lower() in ("hf", "huggingface", "transformers")

    def available(self) -> tuple[bool, str]:
        return self.backend.is_available()

    @property
    def needs_download(self) -> bool:  # type: ignore[override]
        return bool(getattr(self.backend, "needs_download", False))

    def _read(self, image: np.ndarray) -> Reading:
        return self._batch({"S": image}).get("S") or Reading(error="no answer")

    def read_batch(self, images: dict[str, np.ndarray]) -> dict[str, Reading]:
        t0 = time.time()
        try:
            if self.serialize:
                with self._lock:
                    out = self._batch(images)
            else:
                out = self._batch(images)
        except Exception as e:
            err = f"{e.__class__.__name__}: {e}"[:300]
            out = {k: Reading(error=err) for k in images}
        dt = round((time.time() - t0) / max(1, len(images)), 3)
        for k in images:
            out.setdefault(k, Reading(error="missing from the answer"))
            out[k].seconds = dt
        return out

    def _batch(self, images: dict[str, np.ndarray]) -> dict[str, Reading]:
        from ..vlm.base import VLMError

        ids = list(images)
        cap = self.backend.max_images_per_call
        sheet_mode = cap == 1 and len(ids) > 1
        step = self.batch_size if sheet_mode else max(1, min(self.batch_size, cap))
        out: dict[str, Reading] = {}
        for i in range(0, len(ids), step):
            batch = ids[i : i + step]
            if sheet_mode:
                imgs = [sheet([images[b] for b in batch], batch)]
                prompt = TEXT_READ_PROMPT.format(n=len(batch), ids=", ".join(batch)).replace(
                    f"Each of the {len(batch)} images is a crop",
                    f"The image is a sheet of {len(batch)} crops, one per row, each labelled with its id on the left; each crop is",
                )
            else:
                imgs = [images[b] for b in batch]
                prompt = TEXT_READ_PROMPT.format(n=len(batch), ids=", ".join(batch))
            try:
                ans = self.backend.generate_json(prompt, imgs, system=SYSTEM_PROMPT, max_tokens=200 + 120 * len(batch))
            except VLMError as e:
                for b in batch:
                    out[b] = Reading(error=str(e)[:300])
                continue
            items = ans.get("items", []) if isinstance(ans, dict) else ans
            if isinstance(ans, dict) and "items" not in ans and len(batch) == 1 and "text" in ans:
                items = [{**ans, "id": batch[0]}]  # a single object instead of a list
            if not isinstance(items, list):
                items = []
            for k, it in enumerate(items):
                if not isinstance(it, dict):
                    continue
                tid = str(it.get("id") or (batch[k] if k < len(batch) else ""))
                if tid not in batch and k < len(batch):
                    tid = batch[k]
                if tid in batch:
                    txt = it.get("text")
                    co = it.get("crossed_out")
                    out[tid] = Reading(
                        text="" if txt is None else str(txt),
                        confidence=_conf(it.get("confidence")),
                        crossed_out=co if isinstance(co, bool) else str(co).strip().lower() in ("true", "yes", "1"),
                    )
        return out


def find_tesseract() -> str | None:
    """The tesseract binary: $TESSERACT_CMD, PATH, or the Windows installer's default folders
    (the UB-Mannheim installer does not add itself to PATH)."""
    import os

    env = os.environ.get("TESSERACT_CMD")
    if env and os.path.isfile(env):
        return env
    found = shutil.which("tesseract")
    if found:
        return found
    if os.name == "nt":
        for base in (os.environ.get("ProgramFiles", r"C:\Program Files"), os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)"), os.path.join(os.environ.get("LOCALAPPDATA", ""), "Programs")):
            cand = os.path.join(base, "Tesseract-OCR", "tesseract.exe")
            if os.path.isfile(cand):
                return cand
    return None


class TesseractTranscriber(Transcriber):
    preferred_variant = "binarized"
    PSM = {7: "single line", 6: "block", 8: "single word", 13: "raw line"}

    def __init__(self, psm: int = 7, lang: str = "eng", tessdata_dir: str | None = None):
        """``lang``/``tessdata_dir`` select e.g. a model fine-tuned with ``drawid finetune-tesseract``."""
        suffix = "" if lang == "eng" and not tessdata_dir else f":{lang}" + (f"@{tessdata_dir}" if tessdata_dir else "")
        super().__init__(f"tesseract:psm{psm}{suffix}", f"Tesseract ({self.PSM.get(psm, f'psm {psm}')}, {lang})", True, 0.02, 0.0)
        self.psm = psm
        self.lang = lang
        self.tessdata_dir = tessdata_dir

    def available(self) -> tuple[bool, str]:
        try:
            import pytesseract  # noqa: F401
        except ImportError:
            return False, "pip install pytesseract (and the tesseract binary)"
        if not find_tesseract():
            return False, "tesseract binary not found (Windows: winget install UB-Mannheim.TesseractOCR; apt install tesseract-ocr; brew install tesseract; or set TESSERACT_CMD)"
        return True, "ok"

    def _read(self, image: np.ndarray) -> Reading:
        import os

        import pytesseract

        pytesseract.pytesseract.tesseract_cmd = find_tesseract() or "tesseract"
        # tesseract's OpenMP threads thrash when several instances run at once (10x slower)
        os.environ.setdefault("OMP_THREAD_LIMIT", "1")
        config = f"--psm {self.psm}" + (f' --tessdata-dir "{self.tessdata_dir}"' if self.tessdata_dir else "")
        d = pytesseract.image_to_data(image[:, :, ::-1], lang=self.lang, config=config, output_type=pytesseract.Output.DICT)
        words, confs = [], []
        for txt, c in zip(d["text"], d["conf"]):
            c = float(c)
            if txt.strip() and c >= 0:
                words.append(txt.strip())
                confs.append(c)
        return Reading(text=" ".join(words), confidence=(sum(confs) / len(confs) / 100.0) if confs else None)


class EasyOCRTranscriber(Transcriber):
    preferred_variant = "enhanced"
    _readers: dict[str, Any] = {}
    _rlock = threading.Lock()

    def __init__(self, lang: str = "en"):
        super().__init__(f"easyocr:{lang}", f"EasyOCR ({lang})", True, 0.1, 0.0)
        self.lang = lang

    def available(self) -> tuple[bool, str]:
        try:
            import easyocr  # noqa: F401
        except ImportError:
            return False, "pip install easyocr"
        return True, "ok (models download on first use)"

    def _reader(self):
        with self._rlock:
            if self.lang not in self._readers:
                import easyocr

                try:
                    import torch

                    gpu = torch.cuda.is_available()
                except Exception:
                    gpu = False
                self._readers[self.lang] = easyocr.Reader([self.lang], gpu=gpu, verbose=False)
            return self._readers[self.lang]

    def _read(self, image: np.ndarray) -> Reading:
        res = self._reader().readtext(image[:, :, ::-1])
        if not res:
            return Reading(text="", confidence=None)
        # reading order: top-to-bottom lines, then left-to-right
        res.sort(key=lambda r: (round(min(p[1] for p in r[0]) / 20), min(p[0] for p in r[0])))
        return Reading(text=" ".join(r[1] for r in res), confidence=float(np.mean([r[2] for r in res])))


def _token_confidence(torch, scores, n: int) -> float | None:
    """Mean max-softmax probability of the generated tokens: a cheap self-assessed confidence."""
    if not scores:
        return None
    probs = [torch.softmax(sc[0], -1).max().item() for sc in scores[:n]]
    return float(np.mean(probs)) if probs else None


class _HFLocal(Transcriber):
    """Shared plumbing for local transformers OCR models (weights cached per model)."""

    _models: dict[str, Any] = {}

    def available(self) -> tuple[bool, str]:
        try:
            import torch  # noqa: F401
            import transformers  # noqa: F401
        except ImportError:
            return False, "pip install 'drawing-identifier[hf]'"
        from ..vlm.backends import hf_cached

        return True, ("ok" if hf_cached(self.model) else "downloads on first use")

    @property
    def needs_download(self) -> bool:  # type: ignore[override]
        from ..vlm.backends import hf_cached

        return not hf_cached(self.model)


class TrOCRTranscriber(_HFLocal):
    """Microsoft TrOCR (handwriting line recogniser); ``variant`` may also be a fine-tuned folder."""

    MODELS = {
        "small-handwritten": ("microsoft/trocr-small-handwritten", 0.06),
        "base-handwritten": ("microsoft/trocr-base-handwritten", 0.33),
        "large-handwritten": ("microsoft/trocr-large-handwritten", 0.56),
        "base-printed": ("microsoft/trocr-base-printed", 0.33),
    }

    def __init__(self, variant: str = "base-handwritten"):
        model, size = self.MODELS.get(variant, (variant, 0.33))
        super().__init__(f"trocr:{variant}", f"TrOCR {variant}", True, size, 0.0)
        self.model = model

    def _read(self, image: np.ndarray) -> Reading:
        import torch

        if self.model not in self._models:
            from transformers import TrOCRProcessor, VisionEncoderDecoderModel

            dev = "cuda" if torch.cuda.is_available() else "cpu"
            self._models[self.model] = (TrOCRProcessor.from_pretrained(self.model), VisionEncoderDecoderModel.from_pretrained(self.model).to(dev).eval(), dev)
        proc, model, dev = self._models[self.model]
        with torch.no_grad():
            pix = proc(images=image[:, :, ::-1], return_tensors="pt").pixel_values.to(dev)
            out = model.generate(pix, max_new_tokens=48, output_scores=True, return_dict_in_generate=True)
        text = proc.batch_decode(out.sequences, skip_special_tokens=True)[0]
        return Reading(text=text, confidence=_token_confidence(torch, out.scores, len(out.scores)))


class SmolVLMTranscriber(_HFLocal):
    """SmolVLM (HuggingFaceTB): a tiny open VLM that runs locally on CPU; the bottom VLM tier of
    the consensus. ``model`` may be a hub id or a folder fine-tuned with ``drawid finetune-ocr``.
    Ported from the author's SmolVlmOcr (prompt + token-probability confidence)."""

    kind = "vlm"
    PROMPT = "Transcribe the handwritten words in this image. Return ONLY the words, nothing else. If there are no words, return an empty string."
    MODELS = {"256m": "HuggingFaceTB/SmolVLM-256M-Instruct", "500m": "HuggingFaceTB/SmolVLM-500M-Instruct", "2.2b": "HuggingFaceTB/SmolVLM-Instruct"}

    def __init__(self, variant: str = "500m"):
        model = self.MODELS.get(variant.lower(), variant)

        size = {"256m": 0.256, "500m": 0.5, "2.2b": 2.2}.get(variant.lower()) or model_size_b(model) or 0.5
        super().__init__(f"smolvlm:{variant}", f"SmolVLM {variant}", True, size, 0.0)
        self.model = model

    def _read(self, image: np.ndarray) -> Reading:
        import re as _re

        import torch
        from PIL import Image

        if self.model not in self._models:
            import transformers
            from transformers import AutoProcessor

            dev = "cuda" if torch.cuda.is_available() else "cpu"
            cls = getattr(transformers, "AutoModelForImageTextToText", None) or getattr(transformers, "AutoModelForVision2Seq")
            dtype = torch.bfloat16 if dev == "cuda" else torch.float32
            self._models[self.model] = (AutoProcessor.from_pretrained(self.model), cls.from_pretrained(self.model, dtype=dtype).to(dev).eval(), dev)
        proc, model, dev = self._models[self.model]
        msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": self.PROMPT}]}]
        prompt = proc.apply_chat_template(msgs, add_generation_prompt=True)
        inputs = proc(text=prompt, images=[Image.fromarray(image[:, :, ::-1])], return_tensors="pt").to(dev)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=64, do_sample=False, output_scores=True, return_dict_in_generate=True)
        seq = out.sequences[0][inputs["input_ids"].shape[1] :]
        text = _re.sub(r"^\s*assistant[:\s]*", "", proc.decode(seq, skip_special_tokens=True).strip(), flags=_re.I)
        return Reading(text=text, confidence=_token_confidence(torch, out.scores, len(seq)))


class CallableTranscriber(Transcriber):
    """Wrap any ``fn(image) -> str | (str, confidence)``: custom engines, tests, demos."""

    def __init__(self, id: str, fn, kind: str = "vlm", local: bool = True, params_b: float = 1.0, cost_rank: float = 0.0, weight: float | None = None, label: str | None = None):
        self.kind = kind
        super().__init__(id, label or id, local, params_b, cost_rank, weight)
        self.fn = fn
        self.serialize = False

    def _read(self, image: np.ndarray) -> Reading:
        out = self.fn(image)
        if isinstance(out, Reading):
            return out
        if isinstance(out, tuple):
            return Reading(text=str(out[0]), confidence=_conf(out[1]))
        return Reading(text=str(out))


# ----------------------------------------------------------------- catalogue
def local_ocr_engines() -> list[Transcriber]:
    return [
        TesseractTranscriber(7),
        TesseractTranscriber(6),
        TesseractTranscriber(8),
        TesseractTranscriber(13),
        EasyOCRTranscriber("en"),
        *[TrOCRTranscriber(v) for v in TrOCRTranscriber.MODELS],
        SmolVLMTranscriber("256m"),
        SmolVLMTranscriber("500m"),
    ]


def engine_from_id(eid: str, config: AppConfig | None = None, registry=None) -> Transcriber | None:
    """Build an engine from its id, including custom ones:
    ``tesseract:psm7[:lang][@tessdata_dir]``, ``easyocr:<lang>``, ``trocr:<variant|folder>``,
    ``smolvlm:<256m|500m|2.2b|folder>``, ``vlm:<profile or provider:model spec>``."""
    kind, _, rest = eid.partition(":")
    if not rest:
        return None
    if kind == "tesseract":
        tessdata = None
        if "@" in rest:
            rest, tessdata = rest.split("@", 1)
        parts = rest.split(":")
        psm = int(parts[0].replace("psm", "") or 7)
        return TesseractTranscriber(psm, parts[1] if len(parts) > 1 else "eng", tessdata)
    if kind == "easyocr":
        return EasyOCRTranscriber(rest)
    if kind == "trocr":
        return TrOCRTranscriber(rest)
    if kind == "smolvlm":
        return SmolVLMTranscriber(rest)
    if kind == "vlm" and config is not None:
        from ..vlm.registry import VLMRegistry

        reg = registry or VLMRegistry(config)
        return VLMTranscriber(reg.get(rest), rest, config.text.batch_size)
    return None


def all_engines(config: AppConfig, registry=None, extra_specs: Iterable[str] = (), include_mock: bool | None = None) -> list[Transcriber]:
    """Every engine this configuration knows about (available or not)."""
    from ..vlm.registry import VLMRegistry

    registry = registry or VLMRegistry(config)
    include_mock = config.consensus.include_mock if include_mock is None else include_mock
    engines: list[Transcriber] = local_ocr_engines()
    for eid in config.consensus.extra_engines:  # e.g. fine-tuned trocr/smolvlm/tesseract models
        e = engine_from_id(eid, config, registry)
        if e is not None:
            engines.append(e)
    for name, prof in config.vlms.items():
        if prof.provider == "mock" and not include_mock:
            continue
        engines.append(VLMTranscriber(registry.get(name), name, config.text.batch_size))
    for spec in extra_specs:
        spec = spec.strip()
        if spec:
            engines.append(VLMTranscriber(registry.get(spec), spec, config.text.batch_size))
    return engines


@dataclass
class EngineSelection:
    engines: list[Transcriber] = field(default_factory=list)
    unavailable: dict[str, str] = field(default_factory=dict)  # id -> reason
    unknown: list[str] = field(default_factory=list)


def select_engines(config: AppConfig, selection: str | Iterable[str] = "auto", registry=None, extra_specs: Iterable[str] = ()) -> EngineSelection:
    """Resolve "auto" (all available), "local" (all available local ones) or a list of ids / VLM specs."""
    pool = all_engines(config, registry, extra_specs)
    by_id = {e.id: e for e in pool}
    out = EngineSelection()
    if isinstance(selection, str) and selection in ("auto", "all", "local"):
        wanted = [e.id for e in pool]
        explicit = False
    else:
        wanted = [selection] if isinstance(selection, str) else list(selection)
        explicit = True
    for wid in wanted:
        e = by_id.get(wid) or by_id.get(f"vlm:{wid}")
        if e is None and explicit and ":" in wid:
            try:
                e = engine_from_id(wid if wid.split(":", 1)[0] in ("tesseract", "easyocr", "trocr", "smolvlm", "vlm") else f"vlm:{wid}", config, registry)
            except Exception:
                e = None
        if e is None:
            out.unknown.append(wid)
            continue
        if selection == "local" and not e.local:
            continue
        ok, why = e.available()
        if ok and not explicit and e.needs_download:
            ok, why = False, why + " (pick it explicitly to download)"
        if ok:
            out.engines.append(e)
        else:
            out.unavailable[e.id] = why
    return out
