"""Agent abstraction and the shared blackboard (AnalysisContext)."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, ClassVar

import numpy as np

from ..config import AppConfig
from ..schema import DiagramGraph, TraceEvent
from ..vision.preprocess import Preprocessed
from ..vlm.base import VLMBackend


@dataclass
class AgentReport:
    agent: str
    summary: str
    data: dict[str, Any] = field(default_factory=dict)
    skipped: bool = False
    seconds: float = 0.0


@dataclass
class AnalysisContext:
    """Blackboard shared by all agents for one image."""

    config: AppConfig
    original: np.ndarray
    source: str | None = None
    prep: Preprocessed | None = None
    graph: DiagramGraph = field(default_factory=DiagramGraph)
    facts: set[str] = field(default_factory=set)
    text_height: float | None = None
    artifacts: dict[str, Any] = field(default_factory=dict)
    feedback: list[dict] = field(default_factory=list)
    reports: list[AgentReport] = field(default_factory=list)
    requests: list[str] = field(default_factory=list)  # agents another agent asked to re-run
    step: int = 0

    @property
    def image(self) -> np.ndarray:
        return self.prep.image if self.prep is not None else self.original

    @property
    def sw(self) -> float:
        return self.prep.stroke_width if self.prep is not None else 2.0

    @property
    def th(self) -> float:
        return self.text_height or max(10.0, 6 * self.sw)

    def log(self, agent: str, message: str, seconds: float = 0.0, **data: Any) -> None:
        self.graph.trace.append(TraceEvent(step=self.step, agent=agent, message=message, data=data, seconds=round(seconds, 3)))

    def state_summary(self) -> str:
        g = self.graph
        st = g.stats()
        lines = [
            f"facts: {sorted(self.facts)}",
            f"image: {g.image.width}x{g.image.height}, rotation {g.image.rotation}, is_drawing={g.is_drawing}",
            f"shapes: {st['shapes']} {st['shape_types']}, max depth {st['max_depth']}",
            f"connections: {st['connections']}, texts: {st['texts']} ({st['texts_read']} transcribed), relations: {st['relations']}",
        ]
        low = [s.id for s in g.shapes if s.confidence < 0.6]
        if low:
            lines.append(f"low-confidence shapes: {low[:15]}")
        if self.feedback:
            last = self.feedback[-1]
            lines.append(f"last verifier: complete={last.get('is_complete')} issues={last.get('issues', [])[:5]}")
        if self.requests:
            lines.append(f"re-run requested: {self.requests}")
        return "\n".join(lines)


class Agent:
    """One specialised step. Agents read and write the blackboard; they never call each other."""

    name: ClassVar[str] = "agent"
    description: ClassVar[str] = ""
    requires: ClassVar[tuple[str, ...]] = ()
    provides: ClassVar[tuple[str, ...]] = ()
    uses_vlm: ClassVar[bool] = False
    needs_vlm: ClassVar[bool] = False  # cannot do anything useful without one

    def __init__(self, config: AppConfig, vlm: VLMBackend | None = None):
        self.config = config
        self.vlm = vlm

    def ready(self, ctx: AnalysisContext) -> bool:
        return all(r in ctx.facts for r in self.requires) and "stop" not in ctx.facts

    def can_run(self) -> bool:
        return not (self.needs_vlm and self.vlm is None)

    def __call__(self, ctx: AnalysisContext, **params: Any) -> AgentReport:
        t0 = time.time()
        if not self.can_run():
            rep = AgentReport(self.name, "skipped: no VLM configured", skipped=True)
        else:
            rep = self.run(ctx, **params)
        rep.seconds = time.time() - t0
        ctx.reports.append(rep)
        ctx.log(self.name, rep.summary, rep.seconds, **{k: v for k, v in rep.data.items() if _jsonable(v)})
        for p in self.provides:
            if not rep.skipped:
                ctx.facts.add(p)
        return rep

    def run(self, ctx: AnalysisContext, **params: Any) -> AgentReport:  # pragma: no cover - abstract
        raise NotImplementedError

    def describe(self) -> str:
        vlm = f" [VLM: {self.vlm.name}]" if self.vlm else (" [no VLM]" if self.uses_vlm else "")
        return f"{self.name}: {self.description}{vlm}"


def _jsonable(v: Any) -> bool:
    return isinstance(v, (str, int, float, bool, type(None), list, dict, tuple))
