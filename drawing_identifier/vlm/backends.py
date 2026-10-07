"""Concrete VLM backends.

Cloud SDKs are optional extras and only imported when a backend is used:
``pip install drawing-identifier[anthropic]`` / ``[openai]`` / ``[gemini]`` / ``[hf]``.
Ollama is spoken to over plain HTTP, so it needs nothing extra.
"""

from __future__ import annotations

import io
import json
import logging
from typing import Any, Callable

import requests
from PIL import Image

from ..config import VLMProfile
from .base import VLMBackend, VLMError, VLMResponse, encode_image

log = logging.getLogger(__name__)


def _missing(pkg: str, extra: str) -> VLMError:
    return VLMError(f"Python package '{pkg}' is required for this VLM: pip install 'drawing-identifier[{extra}]'")


# --------------------------------------------------------------------------- Anthropic
class AnthropicBackend(VLMBackend):
    provider = "anthropic"

    def __init__(self, profile: VLMProfile, name: str | None = None):
        super().__init__(profile, name)
        self._client = None

    def is_available(self) -> tuple[bool, str]:
        if not self.api_key("ANTHROPIC_API_KEY"):
            return False, f"missing API key (${self.profile.api_key_env or 'ANTHROPIC_API_KEY'})"
        try:
            import anthropic  # noqa: F401
        except ImportError:
            return False, "pip install anthropic"
        return True, "ok"

    def _get_client(self):
        if self._client is None:
            try:
                import anthropic
            except ImportError as e:
                raise _missing("anthropic", "anthropic") from e
            kwargs: dict[str, Any] = {"api_key": self.api_key("ANTHROPIC_API_KEY"), "timeout": self.profile.timeout}
            if self.profile.base_url:
                kwargs["base_url"] = self.profile.base_url
            self._client = anthropic.Anthropic(**kwargs)
        return self._client

    def _generate(self, prompt, images, system, max_tokens, json_mode):
        content: list[dict[str, Any]] = []
        for im in images:
            data, media = encode_image(im)
            content.append({"type": "image", "source": {"type": "base64", "media_type": media, "data": data}})
        content.append({"type": "text", "text": prompt})
        kwargs: dict[str, Any] = {
            "model": self.profile.model,
            "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": content}],
        }
        if system:
            kwargs["system"] = system
        if self.profile.temperature is not None:
            kwargs["temperature"] = self.profile.temperature
        kwargs.update(self.profile.extra.get("request", {}))
        msg = self._get_client().messages.create(**kwargs)
        text = "".join(getattr(b, "text", "") for b in msg.content if getattr(b, "type", "") == "text")
        usage = {}
        if getattr(msg, "usage", None) is not None:
            usage = {"input_tokens": msg.usage.input_tokens, "output_tokens": msg.usage.output_tokens}
        return VLMResponse(text=text, usage=usage, raw=msg)


# --------------------------------------------------------------------------- OpenAI
class OpenAIBackend(VLMBackend):
    """OpenAI, and any OpenAI-compatible server (vLLM, LM Studio, llama.cpp, OpenRouter...)."""

    provider = "openai"

    def __init__(self, profile: VLMProfile, name: str | None = None, compatible: bool = False):
        super().__init__(profile, name)
        self.compatible = compatible
        self._client = None

    def is_available(self) -> tuple[bool, str]:
        try:
            import openai  # noqa: F401
        except ImportError:
            return False, "pip install openai"
        if self.compatible:
            if not self.profile.base_url:
                return False, "base_url required for an OpenAI-compatible server"
            try:
                r = requests.get(self.profile.base_url.rstrip("/") + "/models", timeout=2,
                                 headers={"Authorization": f"Bearer {self.api_key() or 'none'}"})
                if r.status_code >= 500:
                    return False, f"server error {r.status_code}"
            except Exception as e:  # pragma: no cover - network dependent
                return False, f"server not reachable: {e.__class__.__name__}"
            return True, "ok"
        if not self.api_key("OPENAI_API_KEY"):
            return False, f"missing API key (${self.profile.api_key_env or 'OPENAI_API_KEY'})"
        return True, "ok"

    def _get_client(self):
        if self._client is None:
            try:
                import openai
            except ImportError as e:
                raise _missing("openai", "openai") from e
            key = self.api_key(None if self.compatible else "OPENAI_API_KEY") or ("none" if self.compatible else None)
            kwargs: dict[str, Any] = {"api_key": key, "timeout": self.profile.timeout}
            if self.profile.base_url:
                kwargs["base_url"] = self.profile.base_url
            self._client = openai.OpenAI(**kwargs)
        return self._client

    def _generate(self, prompt, images, system, max_tokens, json_mode):
        content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
        for im in images:
            data, media = encode_image(im)
            content.append({"type": "image_url", "image_url": {"url": f"data:{media};base64,{data}"}})
        messages: list[dict[str, Any]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": content})
        kwargs: dict[str, Any] = {"model": self.profile.model, "messages": messages}
        # api.openai.com uses max_completion_tokens; most compatible servers still use max_tokens
        kwargs["max_tokens" if self.compatible else "max_completion_tokens"] = max_tokens
        if self.profile.temperature is not None:
            kwargs["temperature"] = self.profile.temperature
        if json_mode and self.profile.extra.get("json_mode", True):
            kwargs["response_format"] = {"type": "json_object"}
        kwargs.update(self.profile.extra.get("request", {}))
        resp = self._get_client().chat.completions.create(**kwargs)
        text = resp.choices[0].message.content or ""
        usage = {}
        if getattr(resp, "usage", None) is not None:
            usage = {"input_tokens": resp.usage.prompt_tokens, "output_tokens": resp.usage.completion_tokens}
        return VLMResponse(text=text, usage=usage, raw=resp)


# --------------------------------------------------------------------------- Gemini
class GeminiBackend(VLMBackend):
    provider = "gemini"

    def __init__(self, profile: VLMProfile, name: str | None = None):
        super().__init__(profile, name)
        self._client = None

    def _key(self) -> str | None:
        import os

        return self.api_key("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")

    def is_available(self) -> tuple[bool, str]:
        if not self._key():
            return False, "missing API key ($GEMINI_API_KEY or $GOOGLE_API_KEY)"
        try:
            from google import genai  # noqa: F401
        except ImportError:
            return False, "pip install google-genai"
        return True, "ok"

    def _generate(self, prompt, images, system, max_tokens, json_mode):
        try:
            from google import genai
            from google.genai import types
        except ImportError as e:
            raise _missing("google-genai", "gemini") from e
        if self._client is None:
            self._client = genai.Client(api_key=self._key())
        parts: list[Any] = []
        for im in images:
            buf = io.BytesIO()
            im.save(buf, format="PNG")
            parts.append(types.Part.from_bytes(data=buf.getvalue(), mime_type="image/png"))
        parts.append(prompt)
        cfg: dict[str, Any] = {"max_output_tokens": max_tokens}
        if system:
            cfg["system_instruction"] = system
        if self.profile.temperature is not None:
            cfg["temperature"] = self.profile.temperature
        if json_mode:
            cfg["response_mime_type"] = "application/json"
        resp = self._client.models.generate_content(
            model=self.profile.model, contents=parts, config=types.GenerateContentConfig(**cfg)
        )
        return VLMResponse(text=resp.text or "", raw=resp)


# --------------------------------------------------------------------------- Ollama
class OllamaBackend(VLMBackend):
    """Local models served by Ollama (qwen2.5vl, llama3.2-vision, llava, gemma3, minicpm-v...)."""

    provider = "ollama"

    @property
    def base_url(self) -> str:
        return (self.profile.base_url or "http://localhost:11434").rstrip("/")

    def is_available(self) -> tuple[bool, str]:
        try:
            r = requests.get(self.base_url + "/api/tags", timeout=2)
            r.raise_for_status()
        except Exception as e:
            return False, f"ollama not reachable at {self.base_url} ({e.__class__.__name__})"
        names = {m.get("name", "") for m in r.json().get("models", [])}
        model = self.profile.model
        if model not in names and f"{model}:latest" not in names:
            return False, f"model not pulled: ollama pull {model}"
        return True, "ok"

    def _generate(self, prompt, images, system, max_tokens, json_mode):
        msgs: list[dict[str, Any]] = []
        if system:
            msgs.append({"role": "system", "content": system})
        msgs.append({"role": "user", "content": prompt, "images": [encode_image(im)[0] for im in images]})
        options: dict[str, Any] = {"num_predict": max_tokens}
        options["temperature"] = 0.0 if self.profile.temperature is None else self.profile.temperature
        options.update(self.profile.extra.get("options", {}))
        payload: dict[str, Any] = {"model": self.profile.model, "messages": msgs, "stream": False, "options": options}
        if json_mode:
            payload["format"] = "json"
        r = requests.post(self.base_url + "/api/chat", json=payload, timeout=self.profile.timeout)
        if r.status_code != 200:
            raise VLMError(f"ollama error {r.status_code}: {r.text[:300]}")
        data = r.json()
        usage = {"input_tokens": data.get("prompt_eval_count"), "output_tokens": data.get("eval_count")}
        return VLMResponse(text=data.get("message", {}).get("content", ""), usage=usage, raw=data)


# --------------------------------------------------------------------------- HuggingFace
class HuggingFaceBackend(VLMBackend):
    """Local transformers model (Qwen2.5-VL, Qwen3-VL, Gemma 3, SmolVLM, Idefics3...).

    ``extra`` keys: ``device_map`` (default "auto"), ``dtype`` ("auto"|"bfloat16"|"float16"|"float32"),
    ``adapter_path`` (a PEFT LoRA produced by ``drawid finetune-vlm``), ``trust_remote_code``.
    """

    provider = "hf"

    def __init__(self, profile: VLMProfile, name: str | None = None):
        super().__init__(profile, name)
        self._model = None
        self._processor = None

    def is_available(self) -> tuple[bool, str]:
        try:
            import torch  # noqa: F401
            import transformers  # noqa: F401
        except ImportError:
            return False, "pip install 'drawing-identifier[hf]'"
        return True, "ok (weights load on first use)"

    def _load(self):
        if self._model is not None:
            return
        try:
            import torch
            import transformers
            from transformers import AutoProcessor
        except ImportError as e:
            raise _missing("transformers", "hf") from e
        ex = self.profile.extra
        dtype_name = ex.get("dtype", "auto")
        dtype = dtype_name if dtype_name == "auto" else getattr(torch, dtype_name)
        trc = bool(ex.get("trust_remote_code", False))
        model_cls = getattr(transformers, "AutoModelForImageTextToText", None) or getattr(
            transformers, "AutoModelForVision2Seq"
        )
        log.info("loading %s (this can take a while)", self.profile.model)
        self._processor = AutoProcessor.from_pretrained(self.profile.model, trust_remote_code=trc)
        self._model = model_cls.from_pretrained(
            self.profile.model, torch_dtype=dtype, device_map=ex.get("device_map", "auto"), trust_remote_code=trc
        )
        if ex.get("adapter_path"):
            from peft import PeftModel

            self._model = PeftModel.from_pretrained(self._model, ex["adapter_path"])
        self._model.eval()

    def _generate(self, prompt, images, system, max_tokens, json_mode):
        import torch

        self._load()
        content: list[dict[str, Any]] = [{"type": "image", "image": im} for im in images]
        content.append({"type": "text", "text": prompt})
        messages: list[dict[str, Any]] = []
        if system:
            messages.append({"role": "system", "content": [{"type": "text", "text": system}]})
        messages.append({"role": "user", "content": content})
        proc = self._processor
        text = proc.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        inputs = proc(text=[text], images=images or None, return_tensors="pt").to(self._model.device)
        with torch.no_grad():
            out = self._model.generate(**inputs, max_new_tokens=max_tokens, do_sample=False)
        gen = out[:, inputs["input_ids"].shape[1] :]
        answer = proc.batch_decode(gen, skip_special_tokens=True)[0]
        return VLMResponse(text=answer, usage={"output_tokens": int(gen.shape[1])})


# --------------------------------------------------------------------------- Mock
class MockBackend(VLMBackend):
    """Offline backend for tests and dry runs.

    Pass ``responder(prompt, images) -> str|dict`` to script answers; by default it answers ``{}``.
    """

    provider = "mock"

    def __init__(self, profile: VLMProfile, name: str | None = None, responder: Callable | None = None):
        super().__init__(profile, name)
        self.responder = responder
        self.history: list[dict[str, Any]] = []

    def _generate(self, prompt, images, system, max_tokens, json_mode):
        self.history.append({"prompt": prompt, "n_images": len(images), "system": system})
        if self.responder is None:
            ans: Any = "{}"
        else:
            ans = self.responder(prompt, images)
        if not isinstance(ans, str):
            ans = json.dumps(ans)
        return VLMResponse(text=ans)


def blank_image(w: int = 8, h: int = 8) -> Image.Image:
    return Image.new("RGB", (w, h), "white")
