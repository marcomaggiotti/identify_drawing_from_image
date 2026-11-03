"""Searchable store of analysed drawings: one document per shape (text inside / near it,
nesting, connections, page) so you can find e.g. every cut that says "loves" or every
graph nested three deep.

Ported from the author's IndexingAgent / VerifierAgent / query_eg: a deterministic
character-3-gram hashing embedder (no model, no network; swap in a semantic embedder
behind :class:`Embedder`), stable point ids per page/shape, and **human corrections
survive re-indexing**. Backends: a Qdrant server (``url``), embedded Qdrant
(``qdrant:<folder>``) or, with no extra dependency, a local JSONL + numpy store.
"""

from __future__ import annotations

import hashlib
import json
import re
import uuid
from pathlib import Path
from typing import Any, Callable

import numpy as np

from .schema import DiagramGraph


class Embedder:
    name = "none"
    dim = 0

    def embed(self, text: str) -> np.ndarray:  # pragma: no cover - interface
        raise NotImplementedError


class HashNgramEmbedder(Embedder):
    """Lexical (exact / fuzzy) similarity from hashed character 3-grams."""

    name = "hash-3gram"
    dim = 384

    def embed(self, text: str) -> np.ndarray:
        v = np.zeros(self.dim, np.float32)
        t = re.sub(r"\s+", " ", (text or "").lower()).strip()
        if t:
            s = f"^{t}$"
            for i in range(len(s) - 2):
                h = int(hashlib.md5(s[i : i + 3].encode()).hexdigest(), 16)
                v[h % self.dim] += 1.0
                v[(h // self.dim) % self.dim] += 0.5
        n = np.linalg.norm(v)
        return v / n if n > 0 else v


# ---------------------------------------------------------------- stores
class LocalStore:
    """Dependency-free store: payloads in points.jsonl, vectors in vectors.npy."""

    def __init__(self, folder: str | Path, dim: int):
        self.dir = Path(folder)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.dim = dim
        self.points: dict[str, dict] = {}
        self.vectors: dict[str, np.ndarray] = {}
        pj, vn = self.dir / "points.jsonl", self.dir / "vectors.npy"
        if pj.exists():
            rows = [json.loads(ln) for ln in pj.read_text(encoding="utf-8").splitlines() if ln.strip()]
            vecs = np.load(vn) if vn.exists() else np.zeros((len(rows), dim), np.float32)
            for r, v in zip(rows, vecs):
                self.points[r["id"]] = r["payload"]
                self.vectors[r["id"]] = v

    def retrieve(self, pid: str) -> dict | None:
        return self.points.get(pid)

    def upsert(self, pid: str, vec: np.ndarray, payload: dict) -> None:
        self.points[pid] = payload
        self.vectors[pid] = np.asarray(vec, np.float32)

    def flush(self) -> None:
        ids = list(self.points)
        (self.dir / "points.jsonl").write_text("".join(json.dumps({"id": i, "payload": self.points[i]}, ensure_ascii=False) + "\n" for i in ids), encoding="utf-8")
        np.save(self.dir / "vectors.npy", np.stack([self.vectors[i] for i in ids]) if ids else np.zeros((0, self.dim), np.float32))

    def query(self, vec: np.ndarray, limit: int, where: Callable[[dict], bool] | None = None) -> list[tuple[str, float, dict]]:
        ids = [i for i in self.points if where is None or where(self.points[i])]
        if not ids:
            return []
        m = np.stack([self.vectors[i] for i in ids])
        scores = m @ np.asarray(vec, np.float32)
        order = np.argsort(-scores)[:limit]
        return [(ids[k], float(scores[k]), self.points[ids[k]]) for k in order]

    def scroll(self, where: Callable[[dict], bool] | None = None) -> list[tuple[str, dict]]:
        return [(i, p) for i, p in self.points.items() if where is None or where(p)]


class QdrantStore:
    COLLECTION = "drawings"

    def __init__(self, dim: int, url: str | None = None, path: str | None = None):
        from qdrant_client import QdrantClient, models

        self.models = models
        self.client = QdrantClient(url=url) if url else QdrantClient(path=path)
        if self.COLLECTION not in [c.name for c in self.client.get_collections().collections]:
            self.client.create_collection(self.COLLECTION, vectors_config=models.VectorParams(size=dim, distance=models.Distance.COSINE))

    def retrieve(self, pid: str) -> dict | None:
        r = self.client.retrieve(self.COLLECTION, ids=[pid], with_payload=True)
        return r[0].payload if r else None

    def upsert(self, pid: str, vec: np.ndarray, payload: dict) -> None:
        self.client.upsert(self.COLLECTION, points=[self.models.PointStruct(id=pid, vector=np.asarray(vec).tolist(), payload=payload)])

    def flush(self) -> None:
        pass

    def query(self, vec: np.ndarray, limit: int, where: Callable[[dict], bool] | None = None) -> list[tuple[str, float, dict]]:
        # structural filters are applied client-side so both backends accept the same predicates
        hits = self.client.query_points(self.COLLECTION, query=np.asarray(vec).tolist(), limit=limit * 10 if where else limit, with_payload=True).points
        out = [(str(h.id), float(h.score), h.payload) for h in hits if where is None or where(h.payload)]
        return out[:limit]

    def scroll(self, where: Callable[[dict], bool] | None = None) -> list[tuple[str, dict]]:
        pts, out, nxt = None, [], None
        while True:
            pts, nxt = self.client.scroll(self.COLLECTION, limit=256, offset=nxt, with_payload=True)
            out += [(str(p.id), p.payload) for p in pts if where is None or where(p.payload)]
            if nxt is None:
                return out


def open_store(location: str = "data/index", url: str | None = None, embedder: Embedder | None = None):
    """``url`` -> Qdrant server; ``qdrant:<folder>`` -> embedded Qdrant; otherwise a local folder store."""
    emb = embedder or HashNgramEmbedder()
    if url:
        return QdrantStore(emb.dim, url=url), emb
    if location.startswith("qdrant:"):
        return QdrantStore(emb.dim, path=location[len("qdrant:") :]), emb
    return LocalStore(location, emb.dim), emb


# ---------------------------------------------------------------- documents
def point_id(page: str, shape_id: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"{page}/{shape_id}"))


def _embed_text(p: dict) -> str:
    """Only the words are embedded; structure (type, depth, page...) is filtered on, not matched."""
    return f"{p.get('text', '')} {p.get('text_near', '')}".strip()


def shape_documents(g: DiagramGraph, page: str, overlay_file: str | None = None) -> list[tuple[str, str, dict]]:
    """(point id, text to embed, payload) for every shape of an analysed page."""
    docs = []
    for s in g.shapes:
        inside = [g.text(t).text for t in s.text_inside if g.text(t) and g.text(t).text]
        near = [g.text(t).text for t in s.text_near if g.text(t) and g.text(t).text]
        children = g.children(s.id)
        conns = [c for c in g.connections if s.id in [ep.shape_id for ep in c.endpoints] or any(t in c.attached_ids() for t in s.text_inside)]
        text = " ".join(inside)
        payload = {
            "page": page,
            "shape_id": s.id,
            "type": s.type.value,
            "depth": s.depth,
            "parent": s.parent_id,
            "n_children": len(children),
            "child_types": [c.type.value for c in children],
            "text": text,
            "text_near": " ".join(near),
            "n_connections": len(conns),
            "heavy_lines": sum(1 for c in conns if c.heavy),
            "bbox": list(s.bbox.xyxy),
            "crossed_out": s.crossed_out,
            "confidence": s.confidence,
            "overlay_file": overlay_file,
            "human_corrected": False,
            "original_text": "",
        }
        docs.append((point_id(page, s.id), _embed_text(payload), payload))
    return docs


def index_graph(store, emb: Embedder, g: DiagramGraph, page: str, overlay_file: str | None = None) -> int:
    n = 0
    for pid, summary, payload in shape_documents(g, page, overlay_file):
        old = store.retrieve(pid)
        if old and old.get("human_corrected"):
            # a fresh analysis never overwrites a human transcription; geometry still updates
            payload.update(text=old["text"], human_corrected=True, original_text=old.get("original_text", ""))
            summary = _embed_text(payload)
        store.upsert(pid, emb.embed(summary), payload)
        n += 1
    store.flush()
    return n


def correct(store, emb: Embedder, pid: str, text: str) -> dict:
    """Record a human transcription for one indexed shape."""
    p = store.retrieve(pid)
    if p is None:
        raise KeyError(pid)
    if not p.get("human_corrected"):
        p["original_text"] = p.get("text", "")
    p.update(text=text, human_corrected=True)
    store.upsert(pid, emb.embed(_embed_text(p)), p)
    store.flush()
    return p


def make_filter(shape_type: str | None = None, min_depth: int | None = None, min_children: int | None = None, page: str | None = None, corrected: bool | None = None) -> Callable[[dict], bool] | None:
    conds: list[Callable[[dict], bool]] = []
    if shape_type:
        conds.append(lambda p: p.get("type") == shape_type)
    if min_depth is not None:
        conds.append(lambda p: p.get("depth", 0) >= min_depth)
    if min_children is not None:
        conds.append(lambda p: p.get("n_children", 0) >= min_children)
    if page:
        conds.append(lambda p: page in str(p.get("page", "")))
    if corrected is not None:
        conds.append(lambda p: bool(p.get("human_corrected")) == corrected)
    if not conds:
        return None
    return lambda p: all(c(p) for c in conds)


def search(store, emb: Embedder, text: str | None = None, limit: int = 10, **filters: Any) -> list[tuple[str, float, dict]]:
    where = make_filter(**filters)
    if not text:
        return [(i, 1.0, p) for i, p in store.scroll(where)][:limit]
    # vector recall, then an exact-token bonus: trigram vectors are weak for one-letter labels ("A", "p")
    q = text.split()
    hits = []
    for pid, score, p in store.query(emb.embed(text), max(limit * 20, 50), where):
        words = f"{p.get('text', '')} {p.get('text_near', '')}".split()
        bonus = sum(1 for t in q if t in words) / len(q) if q else 0.0
        hits.append((pid, round(score + 0.5 * bonus, 4), p))
    hits = [h for h in hits if h[1] > 0.05]  # nothing in common with the query
    hits.sort(key=lambda h: -h[1])
    return hits[:limit]


def verify(store, emb: Embedder, pid: str) -> bool:
    """Round trip: querying with a document's own text must bring that document back first."""
    p = store.retrieve(pid)
    if not p or not p.get("text", "").strip():
        return True
    hits = store.query(emb.embed(_embed_text(p)), 5)
    # identical texts tie; the document must be among the best-scoring ones
    return bool(hits) and any(h[0] == pid and h[1] >= hits[0][1] - 1e-6 for h in hits)
