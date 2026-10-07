"""Common interface for every vision-language model backend."""

from __future__ import annotations

import base64
import io
import json
import logging
import os
import re
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np
from PIL import Image

from ..config import VLMProfile

log = logging.getLogger(__name__)

ImageLike = Image.Image | np.ndarray


class VLMError(RuntimeError):
    pass


@dataclass
class VLMResponse:
    text: str
    usage: dict[str, Any] = field(default_factory=dict)
    seconds: float = 0.0
    raw: Any = None


def to_pil(img: ImageLike) -> Image.Image:
    if isinstance(img, Image.Image):
        return img.convert("RGB")
    arr = np.asarray(img)
    if arr.ndim == 2:
        return Image.fromarray(arr.astype(np.uint8), "L").convert("RGB")
    # numpy images inside this project are BGR (OpenCV convention)
    return Image.fromarray(arr[:, :, ::-1].astype(np.uint8), "RGB")


def resize_max_side(img: Image.Image, max_side: int) -> Image.Image:
    w, h = img.size
    s = max(w, h)
    if max_side and s > max_side:
        r = max_side / s
        img = img.resize((max(1, round(w * r)), max(1, round(h * r))), Image.LANCZOS)
    return img


def encode_image(img: Image.Image, fmt: str = "PNG") -> tuple[str, str]:
    """Return (base64 string, media type)."""
    buf = io.BytesIO()
    if fmt.upper() == "JPEG":
        img.save(buf, format="JPEG", quality=90)
        media = "image/jpeg"
    else:
        img.save(buf, format="PNG", optimize=True)
        media = "image/png"
    return base64.b64encode(buf.getvalue()).decode("ascii"), media


_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.S)


def extract_json(text: str) -> Any:
    """Tolerant JSON extraction from a model answer (fences, prose around it...)."""
    if text is None:
        raise ValueError("empty response")
    candidates = [m.group(1) for m in _FENCE.finditer(text)] + [text]
    for cand in candidates:
        cand = cand.strip()
        try:
            return json.loads(cand)
        except Exception:
            pass
        # first balanced {...} or [...]
        for open_c, close_c in (("{", "}"), ("[", "]")):
            start = cand.find(open_c)
            while start != -1:
                depth = 0
                in_str = False
                esc = False
                for i in range(start, len(cand)):
                    ch = cand[i]
                    if in_str:
                        if esc:
                            esc = False
                        elif ch == "\\":
                            esc = True
                        elif ch == '"':
                            in_str = False
                        continue
                    if ch == '"':
                        in_str = True
                    elif ch == open_c:
                        depth += 1
                    elif ch == close_c:
                        depth -= 1
                        if depth == 0:
                            snippet = cand[start : i + 1]
                            try:
                                return json.loads(snippet)
                            except Exception:
                                # common model mistake: trailing commas
                                try:
                                    return json.loads(re.sub(r",\s*([}\]])", r"\1", snippet))
                                except Exception:
                                    break
                start = cand.find(open_c, start + 1)
    raise ValueError(f"no JSON object found in response: {text[:200]!r}")


class VLMBackend(ABC):
    """A vision-language model that answers a prompt about zero or more images."""

    provider: str = "base"

    def __init__(self, profile: VLMProfile, name: str | None = None):
        self.profile = profile
        self.name = name or f"{profile.provider}:{profile.model}"
        self.calls = 0
        self.total_seconds = 0.0

    # ----------------------------------------------------------- properties
    @property
    def model(self) -> str:
        return self.profile.model

    @property
    def max_images_per_call(self) -> int:
        return max(1, int(self.profile.max_images_per_call))

    def api_key(self, default_env: str | None = None) -> str | None:
        if self.profile.api_key:
            return self.profile.api_key
        env = self.profile.api_key_env or default_env
        return os.environ.get(env) if env else None

    def prepare_images(self, images: Sequence[ImageLike]) -> list[Image.Image]:
        return [resize_max_side(to_pil(im), self.profile.max_image_side) for im in images]

    # -------------------------------------------------------------- abstract
    @abstractmethod
    def _generate(
        self,
        prompt: str,
        images: list[Image.Image],
        system: str | None,
        max_tokens: int,
        json_mode: bool,
    ) -> VLMResponse: ...

    def is_available(self) -> tuple[bool, str]:
        return True, "ok"

    # ------------------------------------------------------------------ API
    def generate(
        self,
        prompt: str,
        images: Sequence[ImageLike] = (),
        system: str | None = None,
        max_tokens: int | None = None,
        json_mode: bool = False,
    ) -> VLMResponse:
        pil = self.prepare_images(images)
        if len(pil) > self.max_images_per_call:
            raise VLMError(f"{self.name} accepts at most {self.max_images_per_call} images per call")
        t0 = time.time()
        resp = self._generate(prompt, pil, system, max_tokens or self.profile.max_tokens, json_mode)
        resp.seconds = time.time() - t0
        self.calls += 1
        self.total_seconds += resp.seconds
        log.debug("%s answered in %.1fs: %s", self.name, resp.seconds, resp.text[:300])
        return resp

    def generate_json(
        self,
        prompt: str,
        images: Sequence[ImageLike] = (),
        system: str | None = None,
        max_tokens: int | None = None,
        retries: int = 1,
    ) -> Any:
        last_err: Exception | None = None
        text = ""
        for attempt in range(retries + 1):
            p = prompt
            if attempt > 0:
                p = (
                    prompt
                    + "\n\nYour previous answer was not valid JSON. Reply with ONLY the JSON value, no prose."
                )
            resp = self.generate(p, images, system=system, max_tokens=max_tokens, json_mode=True)
            text = resp.text
            try:
                return extract_json(text)
            except ValueError as e:
                last_err = e
        raise VLMError(f"{self.name} did not return valid JSON: {last_err}; got {text[:300]!r}")

    def __repr__(self) -> str:
        return f"<{type(self).__name__} {self.name}>"
