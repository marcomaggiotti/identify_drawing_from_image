"""Connector (line / arrow / line-of-identity) extraction.

Shape outlines are erased from the ink; what is left is lines and text.
Components are separated with skeleton statistics: a connector's skeleton is
about as long as its extent, a written word's skeleton is several times longer.
Heavy strokes (Peirce's lines of identity) are extracted first by stroke width.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field

import cv2
import numpy as np
from skimage.morphology import skeletonize

from .geometry import disk, neighbour_count, simplify

N8 = [(-1, -1), (0, -1), (1, -1), (-1, 0), (1, 0), (-1, 1), (0, 1), (1, 1)]


@dataclass
class LineCandidate:
    path: list[tuple[float, float]]  # main polyline (longest skeleton path)
    endpoints: list[tuple[float, float]]
    heads: list[tuple[float, float]] = field(default_factory=list)  # arrow tips
    heavy: bool = False
    length: float = 0.0
    width: float = 0.0
    mask_bbox: tuple[int, int, int, int] = (0, 0, 0, 0)
    pieces: int = 1


@dataclass
class LineParams:
    stroke_width: float = 2.0
    heavy_ratio: float = 1.6
    min_line_scale: float = 10.0
    max_ratio: float = 1.7  # skeleton length / extent above which a component is text
    min_length_px: float = 0.0  # e.g. 2.5 x text height, set by the agent
    max_endpoints: int = 5  # a branching line of identity has 3-4 ends; handwriting has many


def shape_ring_mask(shape_hw: tuple[int, int], contours: list[np.ndarray], sw: float) -> np.ndarray:
    ring = np.zeros(shape_hw, np.uint8)
    t = max(2, int(round(1.8 * sw + 2)))
    for c in contours:
        if len(c) >= 3:
            cv2.polylines(ring, [np.round(c).astype(np.int32).reshape(-1, 1, 2)], True, 1, t)
    return ring.astype(bool)


def _bfs_far(skel_pts: set, start):
    dist = {start: 0}
    prev = {start: None}
    q = deque([start])
    last = start
    while q:
        p = q.popleft()
        last = p
        for dx, dy in N8:
            n = (p[0] + dx, p[1] + dy)
            if n in skel_pts and n not in dist:
                dist[n] = dist[p] + 1
                prev[n] = p
                q.append(n)
    far = max(dist, key=dist.get)
    return far, dist, prev


def _path_to(prev, end):
    out = []
    p = end
    while p is not None:
        out.append(p)
        p = prev[p]
    return out[::-1]


def analyse_skeleton(skel: np.ndarray, offset=(0, 0), sw: float = 2.0) -> dict:
    """Endpoints, junctions, arrow heads and the longest path of one skeleton component."""
    skel = skel.copy()
    ox, oy = offset
    spur = max(3, int(round(1.5 * sw)) + 1)
    ys_, xs_ = np.nonzero(skel)
    extent = max(int(xs_.max() - xs_.min()), int(ys_.max() - ys_.min())) if len(xs_) else 0
    barb_max = max(8, int(round(6 * sw)), int(0.15 * extent))
    heads: list[tuple[int, int]] = []

    def walk_from(ep, deg):
        """Walk from an endpoint until a junction; returns (pixels, junction or None)."""
        pix = [ep]
        prev = None
        cur = ep
        while True:
            nbrs = [
                (cur[0] + dx, cur[1] + dy)
                for dx, dy in N8
                if 0 <= cur[0] + dx < skel.shape[1] and 0 <= cur[1] + dy < skel.shape[0] and skel[cur[1] + dy, cur[0] + dx]
            ]
            nbrs = [n for n in nbrs if n != prev and n not in pix]
            if not nbrs:
                return pix, None
            if any(deg[n[1], n[0]] >= 3 for n in nbrs):
                j = next(n for n in nbrs if deg[n[1], n[0]] >= 3)
                return pix, j
            prev, cur = cur, nbrs[0]
            pix.append(cur)
            if len(pix) > 10 * barb_max:
                return pix, None

    for _ in range(2):  # prune twice: noise spurs, then arrow barbs
        deg = neighbour_count(skel)
        eys, exs = np.nonzero(deg == 1)
        branches = []
        for x, y in zip(exs, eys):
            pix, j = walk_from((int(x), int(y)), deg)
            if j is not None:
                branches.append((pix, j))
        # arrow heads: >=2 short barbs meeting at the same junction cluster
        jl = cv2.connectedComponents(cv2.dilate((deg >= 3).astype(np.uint8), np.ones((3, 3), np.uint8)), connectivity=8)[1]
        by_j: dict[int, list] = {}
        for pix, j in branches:
            by_j.setdefault(int(jl[j[1], j[0]]), []).append((pix, j))
        removed = False
        total = int(skel.sum())
        for key, lst in by_j.items():
            barbs = [b for b in lst if spur < len(b[0]) <= barb_max]
            if len(barbs) >= 2 and total > 3 * sum(len(b[0]) for b in barbs):
                for pix, _ in barbs:
                    for x, y in pix:
                        skel[y, x] = False
                heads.append(barbs[0][1])
                removed = True
        for pix, j in branches:
            if len(pix) <= spur:
                for x, y in pix:
                    skel[y, x] = False
                removed = True
        if not removed:
            break

    pts = set(zip(*np.nonzero(skel)[::-1]))
    pts = {(int(x), int(y)) for x, y in pts}
    if not pts:
        return {"path": [], "endpoints": [], "heads": [], "length": 0}
    deg = neighbour_count(skel)
    eys, exs = np.nonzero(deg == 1)
    endpoints = [(int(x), int(y)) for x, y in zip(exs, eys)]
    start = endpoints[0] if endpoints else next(iter(pts))
    a, _, _ = _bfs_far(pts, start)
    b, dist, prev = _bfs_far(pts, a)
    path = _path_to(prev, b)
    # the ends of the longest path are line ends even when pruning left a tiny junction knot there
    for q in (a, b):
        if all((q[0] - e[0]) ** 2 + (q[1] - e[1]) ** 2 > (2 * sw) ** 2 for e in endpoints):
            endpoints.append(q)
    # arrow heads collapse into the endpoint nearest to them
    head_pts = []
    for h in heads:
        e = min(endpoints, key=lambda e: (e[0] - h[0]) ** 2 + (e[1] - h[1]) ** 2)
        if (e[0] - h[0]) ** 2 + (e[1] - h[1]) ** 2 <= (barb_max * 1.5) ** 2:
            head_pts.append(e)
    # merge endpoints that are within a few pixels (staircase artefacts)
    merged: list[tuple[int, int]] = []
    for e in endpoints:
        if all((e[0] - m[0]) ** 2 + (e[1] - m[1]) ** 2 > (2 * sw) ** 2 for m in merged):
            merged.append(e)
    sh = lambda p: (float(p[0] + ox), float(p[1] + oy))  # noqa: E731
    return {
        "path": [sh(p) for p in path],
        "endpoints": [sh(e) for e in merged],
        "heads": [sh(h) for h in head_pts],
        "length": float(len(pts)),
    }


def extract_lines(ink: np.ndarray, ring: np.ndarray, params: LineParams) -> tuple[list[LineCandidate], np.ndarray, np.ndarray]:
    """Return (connectors, text_mask, connector_mask)."""
    sw = max(1.0, params.stroke_width)
    residual = ink & ~ring
    min_len = max(params.min_line_scale * sw, params.min_length_px)
    dist = cv2.distanceTransform(residual.astype(np.uint8), cv2.DIST_L2, 5)
    # distance values are quantised; the offset keeps ordinary 2-3px pen strokes out
    heavy_r = 0.5 * params.heavy_ratio * sw + 0.75
    heavy_core = dist >= heavy_r
    heavy_mask = (cv2.dilate(heavy_core.astype(np.uint8), disk(int(math.ceil(heavy_r)) + 1)) > 0) & residual

    lines: list[LineCandidate] = []
    conn_mask = np.zeros_like(ink)
    for heavy_pass, mask in ((True, heavy_mask), (False, residual & ~heavy_mask)):
        n, lab, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
        for i in range(1, n):
            x, y, w, h, area = stats[i]
            extent = math.hypot(w, h)
            if max(w, h) < min_len:
                continue
            comp = lab[y : y + h, x : x + w] == i
            sk = skeletonize(comp)
            sk_len = float(sk.sum())
            if sk_len == 0:
                continue
            ratio = sk_len / max(extent, 1.0)
            mean_w = float(area) / sk_len
            if ratio > params.max_ratio:
                continue  # wiggly: handwriting
            if not heavy_pass and mean_w > 2.5 * sw:
                continue  # blob / scribble
            if heavy_pass and mean_w < 1.3 * sw:
                heavy_pass_c = False
            else:
                heavy_pass_c = heavy_pass
            info = analyse_skeleton(sk, (x, y), sw)
            if len(info["path"]) < 2 or len(info["endpoints"]) > params.max_endpoints:
                continue  # many loose ends: a word or a scribble, not a connector
            path = simplify(info["path"], 0.6 * sw, closed=False)
            lines.append(
                LineCandidate(
                    path=path,
                    endpoints=info["endpoints"],
                    heads=info["heads"],
                    heavy=heavy_pass_c,
                    length=sk_len,
                    width=mean_w,
                    mask_bbox=(int(x), int(y), int(x + w), int(y + h)),
                )
            )
            conn_mask[y : y + h, x : x + w] |= comp

    lines = join_across_rings(lines, ring, sw)
    grow = cv2.dilate(conn_mask.astype(np.uint8), disk(1)).astype(bool)
    text_mask = residual & ~grow
    return lines, text_mask, conn_mask


def join_across_rings(lines: list[LineCandidate], ring: np.ndarray, sw: float) -> list[LineCandidate]:
    """Erasing outlines cuts lines that cross them; glue the pieces back together."""
    if len(lines) < 2:
        return lines
    gap = 2.0 * (1.8 * sw + 2) + 2 * sw
    H, W = ring.shape

    def on_ring(p):
        x, y = int(round(p[0])), int(round(p[1]))
        r = int(math.ceil(sw)) + 2
        return ring[max(0, y - r) : min(H, y + r + 1), max(0, x - r) : min(W, x + r + 1)].any()

    parent = list(range(len(lines)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    links = []
    for i, a in enumerate(lines):
        for j in range(i + 1, len(lines)):
            b = lines[j]
            if a.heavy != b.heavy:
                continue
            for ea in a.endpoints:
                for eb in b.endpoints:
                    d = math.dist(ea, eb)
                    if d <= gap and on_ring(ea) and on_ring(eb):
                        links.append((d, i, j, ea, eb))
    links.sort()
    used_ends: set = set()
    for d, i, j, ea, eb in links:
        if (i, ea) in used_ends or (j, eb) in used_ends:
            continue
        ri, rj = find(i), find(j)
        if ri == rj:
            continue
        parent[rj] = ri
        used_ends.update({(i, ea), (j, eb)})

    groups: dict[int, list[int]] = {}
    for i in range(len(lines)):
        groups.setdefault(find(i), []).append(i)
    out = []
    for idx in groups.values():
        if len(idx) == 1:
            out.append(lines[idx[0]])
            continue
        parts = [lines[i] for i in idx]
        joined_ends = {e for (k, e) in used_ends if k in idx}
        endpoints = [e for p in parts for e in p.endpoints if e not in joined_ends]
        # stitch paths greedily, nearest end to nearest start
        remaining = parts[:]
        cur = remaining.pop(0)
        path = list(cur.path)
        while remaining:
            best = None
            for k, p in enumerate(remaining):
                for rev in (False, True):
                    pp = p.path[::-1] if rev else p.path
                    for at_end in (True, False):
                        anchor = path[-1] if at_end else path[0]
                        d = math.dist(anchor, pp[0] if at_end else pp[-1])
                        if best is None or d < best[0]:
                            best = (d, k, pp, at_end)
            _, k, pp, at_end = best
            path = path + list(pp) if at_end else list(pp) + path
            remaining.pop(k)
        out.append(
            LineCandidate(
                path=path,
                endpoints=endpoints or [path[0], path[-1]],
                heads=[h for p in parts for h in p.heads],
                heavy=parts[0].heavy,
                length=sum(p.length for p in parts),
                width=float(np.mean([p.width for p in parts])),
                mask_bbox=(
                    min(p.mask_bbox[0] for p in parts),
                    min(p.mask_bbox[1] for p in parts),
                    max(p.mask_bbox[2] for p in parts),
                    max(p.mask_bbox[3] for p in parts),
                ),
                pieces=len(parts),
            )
        )
    return out
