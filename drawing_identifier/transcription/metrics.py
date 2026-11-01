"""String metrics for comparing transcriptions: normalisation, CER/WER, similarity, diff."""

from __future__ import annotations

import difflib
import html
import re
import unicodedata

_QUOTES = str.maketrans({"’": "'", "‘": "'", "′": "'", "″": "''", "“": '"', "”": '"', "`": "'", "–": "-", "—": "-"})


def normalize(s: str | None, case_sensitive: bool = True) -> str:
    """Unicode-normalise, unify quotes/primes/dashes, collapse whitespace."""
    if not s:
        return ""
    t = unicodedata.normalize("NFKC", str(s)).translate(_QUOTES)
    t = re.sub(r"\s+", " ", t).strip().strip('"')
    return t if case_sensitive else t.lower()


def levenshtein(a, b) -> int:
    """Edit distance between two sequences (strings or token lists)."""
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def cer(reference: str, hypothesis: str) -> float:
    """Character error rate of ``hypothesis`` against ``reference`` (can exceed 1)."""
    a, b = (reference or "").strip(), (hypothesis or "").strip()
    if not a:
        return 0.0 if not b else 1.0
    return levenshtein(a, b) / len(a)


def wer(reference: str, hypothesis: str) -> float:
    a, b = (reference or "").split(), (hypothesis or "").split()
    if not a:
        return 0.0 if not b else 1.0
    return levenshtein(a, b) / len(a)


def similarity(a: str | None, b: str | None, case_sensitive: bool = True) -> float:
    """1 - normalised edit distance, in [0, 1]; two empty strings are identical."""
    x, y = normalize(a, case_sensitive), normalize(b, case_sensitive)
    if not x and not y:
        return 1.0
    return 1.0 - levenshtein(x, y) / max(len(x), len(y))


def medoid(texts: list[str], weights: list[float] | None = None, case_sensitive: bool = True) -> int:
    """Index of the text closest (weighted) to all the others."""
    if not texts:
        return -1
    w = weights or [1.0] * len(texts)
    best, best_i = -1.0, 0
    for i, t in enumerate(texts):
        s = sum(w[j] * similarity(t, u, case_sensitive) for j, u in enumerate(texts) if j != i)
        if s > best:
            best, best_i = s, i
    return best_i


def diff_html(reference: str, hypothesis: str) -> str:
    """Character diff: deletions (missing in the hypothesis) struck in red, insertions in green."""
    a, b = reference or "", hypothesis or ""
    out = []
    for op, i0, i1, j0, j1 in difflib.SequenceMatcher(None, a, b, autojunk=False).get_opcodes():
        if op == "equal":
            out.append(html.escape(b[j0:j1]))
        if op in ("delete", "replace"):
            out.append(f'<del style="background:#fdd;color:#a00">{html.escape(a[i0:i1])}</del>')
        if op in ("insert", "replace"):
            out.append(f'<ins style="background:#dfd;color:#060;text-decoration:none">{html.escape(b[j0:j1])}</ins>')
    return "".join(out) or '<span style="color:#888">(empty)</span>'
