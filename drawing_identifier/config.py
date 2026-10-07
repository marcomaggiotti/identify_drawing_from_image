"""Configuration: VLM profiles, per-agent VLM selection and agent parameters."""

from __future__ import annotations

import copy
from importlib import resources
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field


class VLMProfile(BaseModel):
    provider: str
    model: str
    base_url: str | None = None
    api_key_env: str | None = None
    api_key: str | None = None  # prefer api_key_env; never commit keys
    max_tokens: int = 4096
    temperature: float | None = None
    max_image_side: int = 1568
    max_images_per_call: int = 8
    bbox_format: str = "xyxy_1000"  # xyxy_1000 | yxyx_1000 | xyxy_pixels
    timeout: float = 180.0
    extra: dict[str, Any] = Field(default_factory=dict)


class OrchestratorConfig(BaseModel):
    mode: str = "pipeline"
    max_steps: int = 20
    stop_if_not_drawing: bool = False


class PreprocessConfig(BaseModel):
    max_side: int = 1800
    crop_page: bool = True
    orientation: str | int = "auto"
    orientation_use_vlm: bool = True
    remove_rules: bool = True


class ShapesConfig(BaseModel):
    detector: str = "cv"
    weights: str | None = None
    yolo_conf: float = 0.35
    min_shape_scale: float = 7.0
    gap_close_scale: float = 2.0
    fit_threshold: float = 0.86
    keep_subregions: bool = False
    max_merge: int = 3


class ConnectionsConfig(BaseModel):
    min_line_scale: float = 10.0
    heavy_ratio: float = 1.6
    attach_scale: float = 3.0


class TextConfig(BaseModel):
    near_scale: float = 2.5
    ocr: str = "vlm"
    batch_size: int = 6
    crop_pad: int = 6
    max_items: int = 150


class ToggleConfig(BaseModel):
    enabled: bool = True


class VerifierConfig(BaseModel):
    enabled: bool = True
    max_rounds: int = 2


class AppConfig(BaseModel):
    default_vlm: str = "auto"
    auto_preference: list[str] = Field(default_factory=list)
    agent_vlms: dict[str, str] = Field(default_factory=dict)
    vlms: dict[str, VLMProfile] = Field(default_factory=dict)
    orchestrator: OrchestratorConfig = Field(default_factory=OrchestratorConfig)
    preprocess: PreprocessConfig = Field(default_factory=PreprocessConfig)
    shapes: ShapesConfig = Field(default_factory=ShapesConfig)
    connections: ConnectionsConfig = Field(default_factory=ConnectionsConfig)
    text: TextConfig = Field(default_factory=TextConfig)
    vlm_analyst: ToggleConfig = Field(default_factory=ToggleConfig)
    verifier: VerifierConfig = Field(default_factory=VerifierConfig)

    def vlm_for(self, agent: str) -> str:
        return self.agent_vlms.get(agent, self.default_vlm)


def _deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict) and k != "agent_vlms":
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def default_config_dict() -> dict:
    text = resources.files("drawing_identifier").joinpath("resources/default.yaml").read_text()
    return yaml.safe_load(text)


def load_config(path: str | Path | None = None, overrides: dict | None = None) -> AppConfig:
    """Load defaults, then merge a user YAML file and programmatic overrides."""
    data = default_config_dict()
    if path:
        with open(path, "r", encoding="utf-8") as fh:
            data = _deep_merge(data, yaml.safe_load(fh) or {})
    if overrides:
        data = _deep_merge(data, overrides)
    return AppConfig.model_validate(data)


def set_dotted(data: dict, dotted: str, value: Any) -> None:
    """Set ``a.b.c=value`` inside a nested dict (used by ``--set`` on the CLI)."""
    keys = dotted.split(".")
    cur = data
    for k in keys[:-1]:
        cur = cur.setdefault(k, {})
    cur[keys[-1]] = value
