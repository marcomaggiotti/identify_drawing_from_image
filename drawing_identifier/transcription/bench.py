"""Run many transcription engines on one segment and compare them side by side."""

from __future__ import annotations

import csv
import io
import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass

import numpy as np

from ..config import ConsensusConfig
from .consensus import ConsensusResult, Vote, decide, family_of, rank_engines
from .engines import TIER_NAMES, Transcriber, make_variants
from .metrics import cer, normalize, similarity, wer

COLUMNS = ["engine", "tier", "variant", "reading", "CER", "agree", "conf", "sec", "status"]


@dataclass
class BenchRow:
    engine_id: str
    label: str
    kind: str
    local: bool
    tier: int
    weight: float
    variant: str
    text: str = ""
    confidence: float | None = None
    crossed_out: bool = False
    seconds: float = 0.0
    error: str | None = None
    cer: float | None = None
    wer: float | None = None
    exact: bool | None = None
    agreement: float | None = None  # weighted mean similarity to the other engines' readings
    in_consensus: bool = False

    def as_row(self) -> list:
        """One line of the comparison table (WER / exact match are in the CSV/JSON export)."""

        def f(v, nd=2):
            return None if v is None else round(float(v), nd)

        status = ("error: " + self.error) if self.error else ("✔ consensus" if self.in_consensus else "")
        return [
            self.engine_id,
            TIER_NAMES[self.tier],
            self.variant,
            self.text if self.error is None else "",
            f(self.cer),
            f(self.agreement),
            f(self.confidence),
            f(self.seconds, 1),
            status,
        ]


def run_segment_benchmark(
    crop: np.ndarray,
    engines: list[Transcriber],
    variants: list[str] | tuple[str, ...] = ("original",),
    reference: str | None = None,
    workers: int = 6,
    case_sensitive: bool = True,
) -> list[BenchRow]:
    """Every engine x every image variant on the same segment (engines run concurrently)."""
    imgs = make_variants(crop)
    variants = [v for v in variants if v in imgs] or ["original"]
    jobs = [(e, v) for e in rank_engines(engines) for v in variants]

    def job(ev):
        e, v = ev
        r = e.read(imgs[v])
        return BenchRow(e.id, e.label, e.kind, e.local, e.tier, e.weight, v, r.text, r.confidence, r.crossed_out, r.seconds, r.error)

    with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
        rows = list(ex.map(job, jobs))
    return score_rows(rows, reference, case_sensitive=case_sensitive)


def score_rows(rows: list[BenchRow], reference: str | None = None, case_sensitive: bool = True, params: ConsensusConfig | None = None) -> list[BenchRow]:
    """(Re)compute metrics: against ``reference`` when given, and agreement between engines."""
    ref = normalize(reference, case_sensitive) if reference else ""
    ok = [r for r in rows if r.error is None]
    for r in rows:
        if r.error is not None:
            r.cer = r.wer = r.exact = r.agreement = None
            continue
        hyp = normalize(r.text, case_sensitive)
        if ref:
            r.cer = cer(ref, hyp)
            r.wer = wer(ref, hyp)
            r.exact = hyp == ref
        else:
            r.cer = r.wer = r.exact = None
        others = [o for o in ok if family_of(o.engine_id) != family_of(r.engine_id)]
        if others:
            wsum = sum(o.weight for o in others)
            r.agreement = sum(o.weight * similarity(r.text, o.text, case_sensitive) for o in others) / wsum
        else:
            r.agreement = None
    cons = consensus_of_rows(rows, params or ConsensusConfig(case_sensitive=case_sensitive))
    for r in rows:
        r.in_consensus = r.error is None and cons.status != "no_answer" and r.engine_id in cons.supporters and similarity(r.text, cons.text, case_sensitive) >= (params or ConsensusConfig()).similarity
    return rows


def consensus_of_rows(rows: list[BenchRow], params: ConsensusConfig | None = None) -> ConsensusResult:
    """Consensus over everything that ran (no escalation): one vote per engine and variant."""
    params = params or ConsensusConfig()
    votes = [Vote(r.engine_id, r.label, r.tier, r.local, r.weight, r.text, r.confidence, r.crossed_out, r.seconds, r.error, r.variant) for r in rows]
    res = ConsensusResult(votes=votes, tiers_run=sorted({r.tier for r in rows}))
    res.text, res.status, res.agreement, res.supporters, res.crossed_out = decide(votes, params)
    for v in votes:
        v.agrees = v.engine_id in res.supporters
    return res


def sort_rows(rows: list[BenchRow], by: str = "auto") -> list[BenchRow]:
    if by == "auto":
        by = "cer" if any(r.cer is not None for r in rows) else "agreement"
    if by == "cer":
        return sorted(rows, key=lambda r: (r.error is not None, r.cer if r.cer is not None else 9, -(r.agreement or 0)))
    if by == "agreement":
        return sorted(rows, key=lambda r: (r.error is not None, -(r.agreement or 0), r.tier))
    return sorted(rows, key=lambda r: (not r.local, r.tier, r.engine_id, r.variant))


def rows_to_csv(rows: list[BenchRow]) -> str:
    buf = io.StringIO()
    w = csv.writer(buf)
    fields = list(asdict(rows[0]).keys()) if rows else list(BenchRow.__dataclass_fields__)
    w.writerow(fields)
    for r in rows:
        w.writerow([asdict(r)[f] for f in fields])
    return buf.getvalue()


def rows_to_json(rows: list[BenchRow], consensus: ConsensusResult | None = None, cascade: ConsensusResult | None = None, reference: str | None = None, box=None) -> str:
    return json.dumps(
        {
            "segment": box,
            "reference": reference,
            "results": [asdict(r) for r in rows],
            "consensus_all": consensus.to_dict() if consensus else None,
            "cascade": cascade.to_dict() if cascade else None,
        },
        indent=2,
        ensure_ascii=False,
    )
