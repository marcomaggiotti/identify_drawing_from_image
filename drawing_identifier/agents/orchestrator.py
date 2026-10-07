"""Orchestrator: runs the agents either as a fixed plan with a verification loop
("pipeline") or lets a VLM planner choose the next agent at every step ("planner")."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from ..config import AppConfig, load_config
from ..export.formats import describe, to_dot, to_mermaid
from ..export.render import render_overlay
from ..prompts import PLANNER_PROMPT, SYSTEM_PROMPT
from ..schema import DiagramGraph
from ..vision.preprocess import load_image
from ..vlm.base import VLMBackend
from ..vlm.registry import VLMRegistry
from .base import Agent, AgentReport, AnalysisContext
from .perception import (
    ConnectionAgent,
    PreprocessAgent,
    ShapeDetectionAgent,
    TextLayoutAgent,
    TopologyAgent,
    TriageAgent,
)
from .reasoning import TextReaderAgent, VerifierAgent, VLMAnalystAgent

log = logging.getLogger(__name__)

AGENT_CLASSES: list[type[Agent]] = [
    PreprocessAgent,
    TriageAgent,
    ShapeDetectionAgent,
    ConnectionAgent,
    TextLayoutAgent,
    TopologyAgent,
    VLMAnalystAgent,
    TextReaderAgent,
    VerifierAgent,
]

PIPELINE = [
    "preprocess",
    "triage",
    "shapes",
    "connections",
    "text_layout",
    "topology",
    "vlm_analyst",
    "topology",
    "text_reader",
    "topology",
]

PLANNER_PARAMS = {
    "preprocess": {"rotation"},
    "shapes": {"relax", "mode", "min_shape_scale", "fit_threshold", "gap_close_scale"},
    "text_reader": {"force", "ids"},
}


@dataclass
class AnalysisResult:
    graph: DiagramGraph
    context: AnalysisContext

    @property
    def image(self) -> np.ndarray:
        return self.context.image

    def overlay(self) -> np.ndarray:
        return render_overlay(self.context.image, self.graph)

    def description(self) -> str:
        return describe(self.graph)

    def save(self, out_dir: str | Path, stem: str | None = None, include_trace: bool = True) -> dict[str, str]:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        stem = stem or (Path(self.context.source).stem if self.context.source else "drawing")
        paths = {
            "json": out / f"{stem}.json",
            "overlay": out / f"{stem}_overlay.png",
            "working": out / f"{stem}_working.png",
            "dot": out / f"{stem}.dot",
            "mermaid": out / f"{stem}.mmd",
            "description": out / f"{stem}.md",
        }
        paths["json"].write_text(self.graph.to_json(include_trace=include_trace), encoding="utf-8")
        cv2.imwrite(str(paths["overlay"]), self.overlay())
        cv2.imwrite(str(paths["working"]), self.context.image)
        paths["dot"].write_text(to_dot(self.graph), encoding="utf-8")
        paths["mermaid"].write_text(to_mermaid(self.graph), encoding="utf-8")
        md = [f"# {stem}", ""]
        if self.graph.summary:
            md += [f"> {self.graph.summary}", ""]
        md += [self.description(), "", "## Agent trace", ""]
        md += [f"{e.step}. **{e.agent}** ({e.seconds:.1f}s): {e.message}" for e in self.graph.trace]
        md += ["", "```mermaid", to_mermaid(self.graph), "```"]
        paths["description"].write_text("\n".join(md), encoding="utf-8")
        return {k: str(v) for k, v in paths.items()}


def normalize_array(image: np.ndarray) -> np.ndarray:
    """Any numpy image (gray, (H,W,1), BGRA, float 0..1 / 0..255, uint16) -> uint8 BGR."""
    img = np.asarray(image)
    if img.ndim == 3 and img.shape[2] == 1:
        img = img[:, :, 0]
    if img.dtype != np.uint8:
        f = img.astype(np.float64)
        if np.issubdtype(img.dtype, np.floating) and f.size and np.nanmax(f) <= 1.0:
            f = f * 255.0
        elif img.dtype == np.uint16:
            f = f / 257.0
        img = np.clip(np.nan_to_num(f), 0, 255).astype(np.uint8)
    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if img.shape[2] == 4:
        a = img[:, :, 3:4].astype(np.float32) / 255.0
        return (img[:, :, :3].astype(np.float32) * a + 255.0 * (1.0 - a)).astype(np.uint8)
    return np.ascontiguousarray(img[:, :, :3])


class Orchestrator:
    def __init__(
        self,
        config: AppConfig | None = None,
        registry: VLMRegistry | None = None,
        vlm: VLMBackend | None = None,
        agent_vlms: dict[str, VLMBackend] | None = None,
    ):
        """``vlm`` forces one backend for every agent; ``agent_vlms`` overrides per agent."""
        self.config = config or load_config()
        overrides: dict[str, VLMBackend] = dict(agent_vlms or {})
        if vlm is not None:
            overrides.setdefault("*", vlm)
        self.registry = registry or VLMRegistry(self.config, overrides)
        if registry is not None and overrides:
            self.registry._overrides.update(overrides)
        self.agents: dict[str, Agent] = {}
        for cls in AGENT_CLASSES:
            backend = self.registry.for_agent(cls.name) if cls.uses_vlm else None
            self.agents[cls.name] = cls(self.config, backend)
        self.planner_vlm = self.registry.for_agent("planner")

    # ------------------------------------------------------------------ public
    def describe_agents(self) -> list[str]:
        return [a.describe() for a in self.agents.values()]

    def analyze(self, image: str | Path | np.ndarray, source: str | None = None, mode: str | None = None) -> AnalysisResult:
        t0 = time.time()
        if isinstance(image, (str, Path)):
            source = source or str(image)
            img = load_image(str(image))
        else:
            img = normalize_array(image)
        ctx = AnalysisContext(config=self.config, original=img, source=source)
        mode = (mode or self.config.orchestrator.mode).lower()
        if mode == "planner" and self.planner_vlm is not None:
            self._run_planner(ctx)
        else:
            if mode == "planner":
                ctx.graph.notes.append("planner mode requested but no VLM is available: used the fixed pipeline")
            self._run_pipeline(ctx)
        self._finalize(ctx)
        ctx.log("orchestrator", f"done in {time.time() - t0:.1f}s", time.time() - t0)
        return AnalysisResult(ctx.graph, ctx)

    # ---------------------------------------------------------------- helpers
    def _run(self, ctx: AnalysisContext, name: str, **params: Any) -> AgentReport:
        ctx.step += 1
        agent = self.agents[name]
        log.info("[%d] %s %s", ctx.step, name, params or "")
        rep = agent(ctx, **params)
        log.info("     -> %s", rep.summary)
        return rep

    def _run_pipeline(self, ctx: AnalysisContext) -> None:
        for name in PIPELINE:
            if "stop" in ctx.facts:
                break
            agent = self.agents[name]
            if not agent.ready(ctx):
                continue
            if not agent.can_run():
                ctx.log(name, "skipped: no VLM configured" if agent.needs_vlm or agent.uses_vlm else "skipped")
                continue
            self._run(ctx, name)
        self._verification_loop(ctx)

    def _verification_loop(self, ctx: AnalysisContext) -> None:
        verifier = self.agents["verifier"]
        if "stop" in ctx.facts or not verifier.can_run() or not self.config.verifier.enabled:
            return
        for _ in range(max(0, self.config.verifier.max_rounds)):
            if not verifier.ready(ctx):
                return
            self._run(ctx, "verifier")
            self._run(ctx, "topology")
            if "verified" in ctx.facts:
                return
            requests, ctx.requests = ctx.requests, []
            for r in requests:
                self._handle_request(ctx, r)

    def _handle_request(self, ctx: AnalysisContext, r: str) -> None:
        if r == "shapes":
            self._run(ctx, "shapes", mode="merge", relax=True)
        elif r == "connections":
            self._run(ctx, "connections")
            self._run(ctx, "text_layout")  # connectors and text share the leftover ink
        elif r == "text_reader" and self.agents["text_reader"].can_run():
            self._run(ctx, "text_reader", force=True)
        self._run(ctx, "topology")

    def _default_next(self, ctx: AnalysisContext) -> str | None:
        gave_up = {r.agent for r in ctx.reports if r.skipped}  # skipped/failed once: do not insist
        for name in PIPELINE:
            a = self.agents[name]
            if name in gave_up:
                continue
            if a.ready(ctx) and a.can_run() and not all(p in ctx.facts for p in a.provides):
                return name
        if "topology" not in ctx.facts and self.agents["topology"].ready(ctx):
            return "topology"
        v = self.agents["verifier"]
        runs = sum(1 for r in ctx.reports if r.agent == "verifier")
        if "verifier" in gave_up:
            return None
        if v.ready(ctx) and v.can_run() and self.config.verifier.enabled and "verified" not in ctx.facts and runs < self.config.verifier.max_rounds:
            return "verifier"
        return None

    def _run_planner(self, ctx: AnalysisContext) -> None:
        history: list[str] = []
        last: list[str] = []
        for _ in range(self.config.orchestrator.max_steps):
            if "stop" in ctx.facts:
                break
            lines = []
            for a in self.agents.values():
                status = "ready" if (a.ready(ctx) and a.can_run()) else ("unavailable" if not a.can_run() else f"needs {list(a.requires)}")
                done = " (done)" if a.provides and all(p in ctx.facts for p in a.provides) else ""
                lines.append(f"- {a.name} [{status}]{done}: {a.description}")
            prompt = PLANNER_PROMPT.format(
                agents="\n".join(lines),
                state=ctx.state_summary(),
                history="\n".join(history[-8:]) or "(nothing yet)",
            )
            images = []
            if ctx.prep is not None:
                img = ctx.prep.image
                s = 768.0 / max(img.shape[:2])
                images = [cv2.resize(img, (int(img.shape[1] * s), int(img.shape[0] * s)), interpolation=cv2.INTER_AREA)] if s < 1 else [img]
            action, params, reason = None, {}, ""
            try:
                ans = self.planner_vlm.generate_json(prompt, images, system=SYSTEM_PROMPT, max_tokens=400)
                action = str(ans.get("action", "")).strip()
                params = ans.get("params") or {}
                reason = str(ans.get("reason", ""))
            except Exception as e:
                log.warning("planner call failed: %s", e)
            if action == "finish":
                if "topology" in ctx.facts:
                    ctx.log("planner", f"finish: {reason}")
                    break
                action = None
            agent = self.agents.get(action or "")
            gave_up = {r.agent for r in ctx.reports if r.skipped}
            if agent is None or not agent.ready(ctx) or not agent.can_run() or action in gave_up or last[-2:] == [action, action]:
                fallback = self._default_next(ctx)
                if action:
                    ctx.log("planner", f"rejected action {action!r}; falling back to {fallback!r}")
                action, params = fallback, {}
                if action is None:
                    break
            allowed = PLANNER_PARAMS.get(action, set())
            params = {k: v for k, v in (params or {}).items() if k in allowed} if isinstance(params, dict) else {}
            if reason:
                ctx.log("planner", f"-> {action} {params or ''}: {reason}")
            rep = self._run(ctx, action, **params)
            history.append(f"{action}: {rep.summary}")
            last.append(action)
            if action in ctx.requests:
                ctx.requests.remove(action)

    def _finalize(self, ctx: AnalysisContext) -> None:
        g = ctx.graph
        if "stop" not in ctx.facts and "shapes" in ctx.facts:
            if self.agents["topology"].ready(ctx):
                self._run(ctx, "topology")
        g.renumber()
        if not g.summary and (g.shapes or g.texts):
            g.summary = describe(g).splitlines()[0]
        vlms = sorted({a.vlm.name for a in self.agents.values() if a.vlm is not None})
        if self.planner_vlm is not None and self.config.orchestrator.mode == "planner":
            vlms = sorted(set(vlms) | {self.planner_vlm.name})
        g.notes.append(f"VLMs used: {', '.join(vlms) if vlms else 'none (classical CV only)'}")


def analyze(image: str | Path | np.ndarray, config: AppConfig | None = None, vlm: str | VLMBackend | None = None, **kw: Any) -> AnalysisResult:
    """One-call convenience API: ``analyze("page.jpg", vlm="ollama:qwen2.5vl:7b")``."""
    config = config or load_config()
    if isinstance(vlm, str):
        config = config.model_copy(update={"default_vlm": vlm})
        return Orchestrator(config, **kw).analyze(image)
    return Orchestrator(config, vlm=vlm, **kw).analyze(image)
