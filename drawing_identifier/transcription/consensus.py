"""Escalating consensus between transcription engines.

Engines are ordered **local first**, then by size and price, and grouped in
tiers (0 local OCR, 1 small local VLMs, 2 large local VLMs, 3 cheap cloud,
4 mid cloud, 5 top cloud). The cascade runs the cheapest tier, votes, and only
escalates to the next (bigger / more expensive) tier when the answers so far
do not agree. On a batch of segments, only the still-disputed segments are
sent up, so expensive models see just the hard cases.

Voting: readings are clustered by normalised edit-distance similarity. A cluster
wins when it is backed by at least ``min_agree`` independent engine families
and holds at least ``agreement`` of the weighted vote (weights grow with the
tier: a large model's reading counts more than Tesseract's). Several modes of
one engine (e.g. four Tesseract page modes) collapse into one family vote, so an
engine cannot agree with itself.
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..config import ConsensusConfig
from .engines import TIER_NAMES, Reading, Transcriber, make_variants
from .metrics import medoid, normalize, similarity


def family_of(engine_id: str) -> str:
    """Engines sharing a family are not independent witnesses (tesseract:psm7 / tesseract:psm6...)."""
    if engine_id.startswith("vlm:"):
        return engine_id
    return engine_id.split(":", 1)[0]


def rank_engines(engines: list[Transcriber]) -> list[Transcriber]:
    """Local before remote, then smaller tier, cheaper, smaller model."""
    return sorted(engines, key=lambda e: (not e.local, e.tier, e.cost_rank, e.params_b))  # stable: catalogue order on ties


@dataclass
class Vote:
    engine_id: str
    label: str
    tier: int
    local: bool
    weight: float
    text: str = ""
    confidence: float | None = None
    crossed_out: bool = False
    seconds: float = 0.0
    error: str | None = None
    variant: str = "original"
    agrees: bool = False  # set after the decision: supports the final reading

    @property
    def family(self) -> str:
        return family_of(self.engine_id)

    def score(self) -> float:
        c = 0.7 if self.confidence is None else self.confidence
        return self.weight * (0.5 + 0.5 * c)


@dataclass
class ConsensusResult:
    text: str = ""
    status: str = "no_answer"  # consensus | best_effort | no_answer
    agreement: float = 0.0  # weighted share of the votes behind ``text``
    supporters: list[str] = field(default_factory=list)  # engine ids that agree
    votes: list[Vote] = field(default_factory=list)
    tiers_run: list[int] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)  # engines never called (consensus came first)
    crossed_out: bool = False
    seconds: float = 0.0

    @property
    def calls(self) -> int:
        return len(self.votes)

    @property
    def independent(self) -> int:
        """Distinct engine families behind the answer (Tesseract modes count once)."""
        return len({family_of(s) for s in self.supporters})

    @property
    def remote_calls(self) -> int:
        return sum(1 for v in self.votes if not v.local)

    def summary(self) -> str:
        tiers = ", ".join(TIER_NAMES[t] for t in self.tiers_run) or "none"
        return (
            f"{self.status}: {self.text!r} (agreement {self.agreement:.0%}, {self.independent} independent engine(s) agree; "
            f"{self.calls} call(s), {self.remote_calls} remote; tiers run: {tiers}; {len(self.skipped)} engine(s) not needed)"
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "status": self.status,
            "agreement": round(self.agreement, 4),
            "supporters": self.supporters,
            "independent_engines": self.independent,
            "tiers_run": self.tiers_run,
            "skipped": self.skipped,
            "calls": self.calls,
            "remote_calls": self.remote_calls,
            "seconds": round(self.seconds, 3),
            "votes": [
                {
                    "engine": v.engine_id,
                    "tier": v.tier,
                    "local": v.local,
                    "weight": v.weight,
                    "variant": v.variant,
                    "text": v.text,
                    "confidence": v.confidence,
                    "agrees": v.agrees,
                    "seconds": v.seconds,
                    "error": v.error,
                }
                for v in self.votes
            ],
        }


def _family_votes(votes: list[Vote], cs: bool) -> list[Vote]:
    """One vote per engine family: its most central reading, at the family's best weight."""
    fams: dict[str, list[Vote]] = {}
    for v in votes:
        fams.setdefault(v.family, []).append(v)
    out = []
    for members in fams.values():
        if len(members) == 1:
            out.append(members[0])
            continue
        i = medoid([m.text for m in members], [m.weight for m in members], cs)
        rep = members[i]
        confs = [m.confidence for m in members if m.confidence is not None]
        out.append(
            Vote(
                engine_id=rep.engine_id,
                label=rep.label,
                tier=rep.tier,
                local=rep.local,
                weight=max(m.weight for m in members),
                text=rep.text,
                confidence=float(np.mean(confs)) if confs else None,
                variant=rep.variant,
            )
        )
    return out


def fuse_words(readings: list[tuple[str, float]]) -> str:
    """Word-level weighted vote (ROVER-style) of readings aligned on the strongest one."""
    import difflib

    readings = sorted(readings, key=lambda r: -r[1])
    pivot = readings[0][0].split()
    votes: list[dict[str, float]] = [{w: readings[0][1]} for w in pivot]
    for text, weight in readings[1:]:
        other = text.split()
        for op, i0, i1, j0, j1 in difflib.SequenceMatcher(None, pivot, other, autojunk=False).get_opcodes():
            if op == "equal" or (op == "replace" and i1 - i0 == j1 - j0):
                for k in range(i1 - i0):
                    votes[i0 + k][other[j0 + k]] = votes[i0 + k].get(other[j0 + k], 0.0) + weight
            elif op == "replace":  # n words read as m words ("heis" vs "he is"): vote for the span as a whole
                span = " ".join(other[j0:j1])
                votes[i0][span] = votes[i0].get(span, 0.0) + weight
                for k in range(i0 + 1, i1):
                    votes[k][""] = votes[k].get("", 0.0) + weight
            elif op == "delete":
                for k in range(i0, i1):
                    votes[k][""] = votes[k].get("", 0.0) + weight
    words = [max(v.items(), key=lambda kv: kv[1])[0] for v in votes]
    return " ".join(w for w in words if w)


def decide(votes: list[Vote], params: ConsensusConfig) -> tuple[str, str, float, list[str], bool]:
    """Weighted clustering vote -> (text, status, agreement share, supporting engine ids, crossed_out)."""
    cs = params.case_sensitive
    valid = [v for v in votes if v.error is None and normalize(v.text, cs)]
    if not valid:
        return "", "no_answer", 0.0, [], False
    fam = _family_votes(valid, cs)
    fam.sort(key=lambda v: -v.score())
    clusters: list[list[Vote]] = []
    for v in fam:
        for c in clusters:
            if similarity(c[0].text, v.text, cs) >= params.similarity:
                c.append(v)
                break
        else:
            clusters.append([v])
    total = sum(v.score() for v in fam)
    best = max(clusters, key=lambda c: (sum(v.score() for v in c), len(c)))
    share = sum(v.score() for v in best) / total if total else 0.0
    # final spelling: the strongest engine of the winning cluster, or a word-level vote
    # when three or more independent engines back it (they correct each other's slips)
    winner = max(best, key=lambda v: (v.weight, v.confidence or 0.0))
    text = winner.text
    if len(best) >= 3:
        text = fuse_words([(v.text, v.score()) for v in best]) or text
    best_fams = {v.family for v in best}
    supporters: list[str] = []
    for v in valid:
        if (v.family in best_fams or similarity(v.text, text, cs) >= params.similarity) and v.engine_id not in supporters:
            supporters.append(v.engine_id)
    crossed = sum(v.weight for v in valid if v.crossed_out and v.engine_id in supporters) > 0.5 * sum(
        v.weight for v in valid if v.engine_id in supporters
    )
    status = "consensus" if (len(best_fams) >= params.min_agree and share >= params.agreement) else "best_effort"
    return text, status, share, supporters, crossed


def plan(engines: list[Transcriber], params: ConsensusConfig) -> list[list[Transcriber]]:
    """Escalation steps: cheapest tier first; remote tiers only if allowed."""
    ranked = [e for e in rank_engines(engines) if e.tier <= params.max_tier and (params.allow_remote or e.local)]
    if params.per_engine:
        return [[e] for e in ranked]
    steps: list[list[Transcriber]] = []
    for e in ranked:
        if steps and steps[-1][0].tier == e.tier and steps[-1][0].local == e.local:
            steps[-1].append(e)
        else:
            steps.append([e])
    return steps


class ConsensusCascade:
    def __init__(self, engines: list[Transcriber], params: ConsensusConfig | None = None, workers: int = 4):
        self.engines = engines
        self.params = params or ConsensusConfig()
        self.workers = max(1, workers)
        self.steps = plan(engines, self.params)

    def describe(self) -> list[str]:
        return [f"step {i + 1} [{TIER_NAMES[s[0].tier]}]: " + ", ".join(e.id for e in s) for i, s in enumerate(self.steps)]

    def run(self, image: np.ndarray) -> ConsensusResult:
        return self.run_batch({"segment": image})["segment"]

    def run_batch(self, images: dict[str, np.ndarray]) -> dict[str, ConsensusResult]:
        t0 = time.time()
        variants = {k: make_variants(v) for k, v in images.items()}
        results = {k: ConsensusResult() for k in images}
        pending = set(images)
        for step in self.steps:
            if not pending:
                for k in images:
                    results[k].skipped.extend(e.id for e in step)
                continue
            ids = sorted(pending)

            def run_engine(e: Transcriber) -> tuple[Transcriber, dict[str, Reading]]:
                return e, e.read_batch({k: variants[k][e.preferred_variant] for k in ids})

            with ThreadPoolExecutor(max_workers=min(self.workers, len(step))) as ex:
                outs = list(ex.map(run_engine, step))
            for e, readings in outs:
                for k in ids:
                    r = readings.get(k) or Reading(error="no answer")
                    results[k].votes.append(
                        Vote(e.id, e.label, e.tier, e.local, e.weight, r.text, r.confidence, r.crossed_out, r.seconds, r.error, e.preferred_variant)
                    )
            for k in images:
                if k not in pending:
                    results[k].skipped.extend(e.id for e in step)
            for k in ids:
                res = results[k]
                if step[0].tier not in res.tiers_run:
                    res.tiers_run.append(step[0].tier)
                text, status, share, sup, crossed = decide(res.votes, self.params)
                if status == "consensus":
                    pending.discard(k)
        for k, res in results.items():
            res.text, res.status, res.agreement, res.supporters, res.crossed_out = decide(res.votes, self.params)
            for v in res.votes:
                v.agrees = v.engine_id in res.supporters
            res.seconds = time.time() - t0
        return results
