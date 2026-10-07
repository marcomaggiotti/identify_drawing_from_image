"""Create VLM backends from profile names or ad-hoc ``provider:model[@base_url]`` specs."""

from __future__ import annotations

import logging
from typing import Callable

from ..config import AppConfig, VLMProfile
from .backends import (
    AnthropicBackend,
    GeminiBackend,
    HuggingFaceBackend,
    MockBackend,
    OllamaBackend,
    OpenAIBackend,
)
from .base import VLMBackend

log = logging.getLogger(__name__)

PROVIDER_ALIASES = {
    "anthropic": "anthropic",
    "claude": "anthropic",
    "openai": "openai",
    "gpt": "openai",
    "openai_compatible": "openai_compatible",
    "openai-compatible": "openai_compatible",
    "compat": "openai_compatible",
    "vllm": "openai_compatible",
    "lmstudio": "openai_compatible",
    "llamacpp": "openai_compatible",
    "openrouter": "openai_compatible",
    "gemini": "gemini",
    "google": "gemini",
    "ollama": "ollama",
    "hf": "hf",
    "huggingface": "hf",
    "transformers": "hf",
    "mock": "mock",
}

DEFAULT_BASE_URLS = {
    "ollama": "http://localhost:11434",
    "vllm": "http://localhost:8000/v1",
    "lmstudio": "http://localhost:1234/v1",
    "llamacpp": "http://localhost:8080/v1",
    "openrouter": "https://openrouter.ai/api/v1",
}

# Shown by `drawid vlms`: provider -> (example models, install hint)
PROVIDER_INFO = {
    "anthropic": ("claude-opus-5-5, claude-sonnet-5-5, claude-haiku-4-5-20251001", "pip install anthropic; export ANTHROPIC_API_KEY"),
    "openai": ("gpt-5, gpt-4.1, gpt-4o", "pip install openai; export OPENAI_API_KEY"),
    "openai_compatible": ("any model served by vLLM / LM Studio / llama.cpp / OpenRouter", "pip install openai; set base_url"),
    "gemini": ("gemini-2.5-pro, gemini-2.5-flash", "pip install google-genai; export GEMINI_API_KEY"),
    "ollama": ("qwen2.5vl:7b, llama3.2-vision, gemma3, minicpm-v, llava", "install ollama; ollama pull <model>"),
    "hf": ("Qwen/Qwen2.5-VL-3B-Instruct, Qwen/Qwen2.5-VL-7B-Instruct, HuggingFaceTB/SmolVLM-Instruct", "pip install 'drawing-identifier[hf]'"),
    "mock": ("mock", "built in, for tests"),
}

NO_VLM = {"none", "off", "no", "false", "", "cv"}


def parse_spec(spec: str) -> tuple[str, VLMProfile]:
    """``anthropic:claude-opus-5-5`` / ``ollama:qwen2.5vl:7b@http://gpu:11434`` -> profile."""
    base_url = None
    if "@" in spec:
        spec, base_url = spec.split("@", 1)
    if ":" not in spec:
        raise ValueError(f"VLM spec must look like provider:model, got {spec!r}")
    raw_provider, model = spec.split(":", 1)
    provider = PROVIDER_ALIASES.get(raw_provider.lower())
    if provider is None:
        raise ValueError(f"unknown VLM provider {raw_provider!r}; known: {sorted(set(PROVIDER_ALIASES))}")
    if base_url is None:
        base_url = DEFAULT_BASE_URLS.get(raw_provider.lower())
    extra = {}
    max_images = 8
    if provider == "ollama" and any(k in model for k in ("llama3.2-vision", "llava")):
        max_images = 1
    profile = VLMProfile(provider=provider, model=model, base_url=base_url, max_images_per_call=max_images, extra=extra)
    return f"{raw_provider}:{model}", profile


def build_backend(profile: VLMProfile, name: str | None = None, responder: Callable | None = None) -> VLMBackend:
    provider = PROVIDER_ALIASES.get(profile.provider.lower(), profile.provider.lower())
    if provider == "anthropic":
        return AnthropicBackend(profile, name)
    if provider == "openai":
        return OpenAIBackend(profile, name, compatible=bool(profile.base_url and "api.openai.com" not in profile.base_url))
    if provider == "openai_compatible":
        return OpenAIBackend(profile, name, compatible=True)
    if provider == "gemini":
        return GeminiBackend(profile, name)
    if provider == "ollama":
        return OllamaBackend(profile, name)
    if provider == "hf":
        return HuggingFaceBackend(profile, name)
    if provider == "mock":
        return MockBackend(profile, name, responder=responder)
    raise ValueError(f"unknown provider {profile.provider!r}")


class VLMRegistry:
    """Resolves which backend each agent should use and caches instances."""

    def __init__(self, config: AppConfig, overrides: dict[str, VLMBackend] | None = None):
        self.config = config
        self._cache: dict[str, VLMBackend | None] = {}
        self._overrides = dict(overrides or {})  # agent name or "*" -> backend instance
        self._auto: str | None | bool = False  # False = not resolved yet

    def resolve_name(self, selector: str | None) -> str | None:
        sel = (selector or "").strip()
        if sel.lower() in NO_VLM:
            return None
        if sel.lower() == "auto":
            return self._resolve_auto()
        return sel

    def _resolve_auto(self) -> str | None:
        if self._auto is not False:
            return self._auto  # type: ignore[return-value]
        chosen = None
        for name in self.config.auto_preference:
            prof = self.config.vlms.get(name)
            if prof is None:
                continue
            ok, why = build_backend(prof, name).is_available()
            if ok:
                chosen = name
                break
            log.debug("auto VLM: %s unavailable (%s)", name, why)
        if chosen is None:
            log.warning("No VLM available (no API keys / local servers found): running with classical CV only.")
        else:
            log.info("auto-selected VLM profile %r", chosen)
        self._auto = chosen
        return chosen

    def get(self, selector: str | None) -> VLMBackend | None:
        name = self.resolve_name(selector)
        if name is None:
            return None
        if name in self._cache:
            return self._cache[name]
        if name in self.config.vlms:
            backend = build_backend(self.config.vlms[name], name)
        else:
            label, prof = parse_spec(name)
            backend = build_backend(prof, label)
        self._cache[name] = backend
        return backend

    def for_agent(self, agent: str) -> VLMBackend | None:
        if agent in self._overrides:
            return self._overrides[agent]
        if "*" in self._overrides:
            return self._overrides["*"]
        return self.get(self.config.vlm_for(agent))

    def describe(self) -> list[dict[str, str]]:
        rows = []
        for name, prof in self.config.vlms.items():
            ok, why = build_backend(prof, name).is_available()
            rows.append({"name": name, "provider": prof.provider, "model": prof.model, "available": "yes" if ok else "no", "status": why})
        return rows
