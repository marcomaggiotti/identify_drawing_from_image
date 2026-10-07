"""VLM-backed agents: text reading, holistic analysis + fusion, verification."""

from __future__ import annotations

import json
import logging
from typing import Any

import cv2
import numpy as np

from ..export.render import render_overlay
from ..prompts import (
    SYSTEM_PROMPT,
    TEXT_READ_PROMPT,
    VERIFY_PROMPT,
    analysis_prompt,
    bbox_polygon,
    from_model_box,
    graph_to_vlm_target,
    parse_vlm_graph,
    _as_bool,
    _ref,
    _shape_type,
)
from ..schema import (
    BBox,
    Connection,
    ConnectionType,
    DiagramGraph,
    Endpoint,
    Shape,
    ShapeType,
    TextItem,
    TextPlacement,
)
from .base import Agent, AgentReport, AnalysisContext

log = logging.getLogger(__name__)


# --------------------------------------------------------------------------- text reader
class TextReaderAgent(Agent):
    name = "text_reader"
    description = "Read (transcribe) the handwriting in every text region: VLM on image crops, or a local OCR (TrOCR / Tesseract)."
    requires = ("text_regions",)
    provides = ("text_read",)
    uses_vlm = True

    def __init__(self, config, vlm=None):
        super().__init__(config, vlm)
        self._trocr = None

    def can_run(self) -> bool:
        mode = self.config.text.ocr
        if mode == "none":
            return False
        if mode == "vlm":
            return self.vlm is not None
        return True

    def _crop(self, img: np.ndarray, t: TextItem, th: float) -> np.ndarray:
        pad = int(self.config.text.crop_pad + 0.25 * th)
        x0, y0, x1, y1 = (int(round(v)) for v in t.bbox.xyxy)
        h, w = img.shape[:2]
        crop = img[max(0, y0 - pad) : min(h, y1 + pad), max(0, x0 - pad) : min(w, x1 + pad)]
        if crop.size == 0:
            crop = np.full((32, 32, 3), 255, np.uint8)
        if crop.shape[0] < 56:
            s = 56.0 / crop.shape[0]
            crop = cv2.resize(crop, (max(1, int(crop.shape[1] * s)), 56), interpolation=cv2.INTER_CUBIC)
        return crop

    def run(self, ctx: AnalysisContext, force: bool = False, ids: list[str] | None = None, **_: Any) -> AgentReport:
        g = ctx.graph
        # a forced re-read never overwrites transcriptions the verifier corrected
        todo = [t for t in g.texts if (force and "verifier" not in t.source) or not t.text or t.confidence < 0.5]
        if ids:
            todo = [t for t in g.texts if t.id in set(ids)]
        prio = {TextPlacement.INSIDE: 0, TextPlacement.NEAR: 1, TextPlacement.ON_LINE: 1, TextPlacement.FREE: 2}
        todo.sort(key=lambda t: (prio[t.placement], t.bbox.y, t.bbox.x))
        todo = todo[: self.config.text.max_items]
        if not todo:
            return AgentReport(self.name, "nothing to read")
        mode = self.config.text.ocr
        crops = {t.id: self._crop(ctx.image, t, ctx.th) for t in todo}
        results: dict[str, dict] = {}
        if mode == "vlm":
            results = self._read_vlm(crops)
        elif mode == "trocr":
            results = self._read_trocr(crops)
        elif mode == "tesseract":
            results = self._read_tesseract(crops)
        n = 0
        for t in todo:
            r = results.get(t.id)
            if not isinstance(r, dict):
                continue
            raw_txt = r.get("text")
            txt = "" if raw_txt is None else str(raw_txt).strip()
            if txt:
                t.text = txt
                n += 1
            t.crossed_out = bool(_as_bool(r.get("crossed_out"), t.crossed_out))
            try:
                conf = float(r.get("confidence") or 0.7)
            except (TypeError, ValueError):
                conf = 0.7
            t.confidence = round(min(1.0, max(0.0, conf)), 3)
            if mode not in t.source:
                t.source.append(mode if mode != "vlm" else "vlm_ocr")
        return AgentReport(self.name, f"transcribed {n}/{len(todo)} text region(s) with {mode}", {"read": n, "requested": len(todo)})

    # ---- VLM
    def _read_vlm(self, crops: dict[str, np.ndarray]) -> dict[str, dict]:
        ids = list(crops)
        per_call = max(1, min(self.config.text.batch_size, self.vlm.max_images_per_call))
        sheet_mode = self.vlm.max_images_per_call == 1 and self.config.text.batch_size > 1
        step = self.config.text.batch_size if sheet_mode else per_call
        out: dict[str, dict] = {}
        for i in range(0, len(ids), step):
            batch = ids[i : i + step]
            if sheet_mode:
                images = [_sheet([crops[b] for b in batch], batch)]
                prompt = TEXT_READ_PROMPT.format(n=len(batch), ids=", ".join(batch)).replace(
                    f"Each of the {len(batch)} images is a crop",
                    f"The image is a sheet of {len(batch)} crops, one per row, each labelled with its id on the left; each crop is",
                )
            else:
                images = [crops[b] for b in batch]
                prompt = TEXT_READ_PROMPT.format(n=len(batch), ids=", ".join(batch))
            try:
                ans = self.vlm.generate_json(prompt, images, system=SYSTEM_PROMPT, max_tokens=200 + 120 * len(batch))
            except Exception as e:
                log.warning("text_reader VLM batch failed: %s", e)
                continue
            items = ans.get("items", []) if isinstance(ans, dict) else ans
            if isinstance(items, list):
                for k, it in enumerate(items):
                    if not isinstance(it, dict):
                        continue
                    tid = str(it.get("id") or (batch[k] if k < len(batch) else ""))
                    if tid in batch:
                        out[tid] = it
        return out

    # ---- local OCR
    def _read_trocr(self, crops):
        if self._trocr is None:
            from transformers import pipeline

            self._trocr = pipeline("image-to-text", model="microsoft/trocr-base-handwritten")
        from PIL import Image

        out = {}
        for tid, c in crops.items():
            r = self._trocr(Image.fromarray(c[:, :, ::-1]))
            out[tid] = {"text": r[0]["generated_text"] if r else "", "confidence": 0.6}
        return out

    def _read_tesseract(self, crops):
        import pytesseract

        out = {}
        for tid, c in crops.items():
            txt = pytesseract.image_to_string(c, config="--psm 7").strip()
            out[tid] = {"text": txt, "confidence": 0.4}
        return out


def _sheet(crops: list[np.ndarray], ids: list[str]) -> np.ndarray:
    rows = []
    width = max(c.shape[1] for c in crops) + 110
    for c, tid in zip(crops, ids):
        row = np.full((c.shape[0] + 16, width, 3), 255, np.uint8)
        row[8 : 8 + c.shape[0], 100 : 100 + c.shape[1]] = c
        cv2.putText(row, tid, (6, row.shape[0] // 2 + 8), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        cv2.line(row, (0, row.shape[0] - 1), (width, row.shape[0] - 1), (200, 200, 200), 1)
        rows.append(row)
    return cv2.vconcat(rows)


# --------------------------------------------------------------------------- VLM analyst
def _match(boxes_a: list[BBox], b: BBox, min_iou: float) -> int | None:
    best, best_i = None, None
    for i, a in enumerate(boxes_a):
        iou = a.iou(b)
        if iou >= min_iou and (best is None or iou > best):
            best, best_i = iou, i
    if best_i is None:  # loose match: centre inside and similar size
        cx, cy = b.center
        for i, a in enumerate(boxes_a):
            if a.contains_point((cx, cy)) and 0.5 < (a.area / max(b.area, 1)) < 2.0:
                return i
    return best_i


class VLMAnalystAgent(Agent):
    name = "vlm_analyst"
    description = "Ask the VLM for a holistic reading of the whole drawing (shapes, texts, connections) and fuse it with the geometric detections: confirms types, adds missed elements, provides transcriptions."
    requires = ("text_regions",)
    provides = ("vlm_analysis",)
    uses_vlm = True
    needs_vlm = True

    def run(self, ctx: AnalysisContext, **_: Any) -> AgentReport:
        if not self.config.vlm_analyst.enabled:
            return AgentReport(self.name, "disabled in config", skipped=True)
        g = ctx.graph
        w, h = g.image.width, g.image.height
        fmt = self.vlm.profile.bbox_format
        try:
            data = self.vlm.generate_json(analysis_prompt(fmt), [ctx.image], system=SYSTEM_PROMPT, max_tokens=self.vlm.profile.max_tokens)
            vg = parse_vlm_graph(data, w, h, fmt, source="vlm", sent_size=_sent_size(self.vlm, ctx.image))
            stats = fuse(g, vg)
        except Exception as e:
            log.warning("vlm_analyst failed: %s", e)
            return AgentReport(self.name, f"VLM answer unusable: {e.__class__.__name__}", skipped=True)
        if vg.summary:
            g.summary = vg.summary
        if g.is_drawing is None and vg.is_drawing is not None:
            g.is_drawing = vg.is_drawing
        return AgentReport(
            self.name,
            "VLM saw {vs} shapes / {vt} texts / {vc} connections; fused: {matched_shapes} matched, {added_shapes} shapes added, {retyped} retyped, {texts_filled} texts transcribed, {added_texts} texts added, {added_conns} connections added".format(
                vs=len(vg.shapes), vt=len(vg.texts), vc=len(vg.connections), **stats
            ),
            stats,
        )


def fuse(g: DiagramGraph, vg: DiagramGraph) -> dict[str, int]:
    """Merge a VLM reading ``vg`` into the geometric graph ``g`` (in place)."""
    stats = dict(matched_shapes=0, added_shapes=0, retyped=0, texts_filled=0, added_texts=0, added_conns=0)
    idmap: dict[str, str] = {}
    for vs in vg.shapes:
        i = _match([s.bbox for s in g.shapes], vs.bbox, 0.45)
        if i is not None:
            s = g.shapes[i]
            idmap[vs.id] = s.id
            stats["matched_shapes"] += 1
            if "vlm" not in s.source:
                s.source.append("vlm")
            s.confidence = round(min(1.0, s.confidence + 0.15), 3)
            if vs.type not in (ShapeType.UNKNOWN, s.type) and (
                s.type in (ShapeType.REGION, ShapeType.POLYGON, ShapeType.UNKNOWN) or s.confidence < 0.9
            ):
                s.attributes["cv_type"] = s.type.value
                s.type = vs.type
                stats["retyped"] += 1
            if vs.crossed_out:
                s.crossed_out = True
        else:
            if any(s.bbox.iou(vs.bbox) > 0.3 for s in g.shapes):
                continue
            nid = g.next_id("S")
            idmap[vs.id] = nid
            g.shapes.append(vs.model_copy(update={"id": nid, "parent_id": None, "confidence": 0.5}))
            stats["added_shapes"] += 1
    for vt in vg.texts:
        i = _match([t.bbox for t in g.texts], vt.bbox, 0.3)
        if i is not None:
            t = g.texts[i]
            idmap[vt.id] = t.id
            if vt.text and not t.text:
                t.text = vt.text
                t.confidence = 0.6
                stats["texts_filled"] += 1
            t.crossed_out = t.crossed_out or vt.crossed_out
            if "vlm" not in t.source:
                t.source.append("vlm")
        else:
            nid = g.next_id("T")
            idmap[vt.id] = nid
            g.texts.append(vt.model_copy(update={"id": nid, "inside_shape_id": None, "near_shape_ids": [], "confidence": 0.5}))
            stats["added_texts"] += 1
    existing_pairs = {frozenset(c.attached_ids()) for c in g.connections}
    for vc in vg.connections:
        ends = [idmap.get(ep.text_id or ep.shape_id or "", None) for ep in vc.endpoints]
        ends = [e for e in ends if e]
        if len(ends) < 2 or frozenset(ends) in existing_pairs:
            continue
        eps = []
        for e in ends:
            item = g.text(e) or g.shape(e)
            pt = item.bbox.center if item else (0.0, 0.0)
            is_text = g.text(e) is not None
            eps.append(Endpoint(point=pt, text_id=e if is_text else None, shape_id=None if is_text else e, attachment="text" if is_text else "boundary"))
        path = vc.path if len(vc.path) >= 2 else [ep.point for ep in eps]
        g.connections.append(
            Connection(
                id=g.next_id("C"),
                type=vc.type,
                path=path,
                endpoints=eps,
                heavy=vc.heavy,
                confidence=0.5,
                source=["vlm"],
                attributes={"locked_ends": True},
            )
        )
        existing_pairs.add(frozenset(ends))
        stats["added_conns"] += 1
    return stats


# --------------------------------------------------------------------------- verifier
class VerifierAgent(Agent):
    name = "verifier"
    description = "Show the VLM the drawing next to an overlay of the current detections (numbered marks) and apply its corrections; can ask other agents to re-run."
    requires = ("topology",)
    provides = ()  # sets "verified" itself, only when the VLM says the analysis is complete
    uses_vlm = True
    needs_vlm = True

    def run(self, ctx: AnalysisContext, **_: Any) -> AgentReport:
        if not self.config.verifier.enabled:
            return AgentReport(self.name, "disabled in config", skipped=True)
        g = ctx.graph
        overlay = render_overlay(ctx.image, g)
        target = graph_to_vlm_target(g)
        target.pop("summary", None)
        analysis = json.dumps(target, separators=(",", ":"))
        if len(analysis) > 24000:
            analysis = analysis[:24000] + "...(truncated)"
        try:
            ans = self.vlm.generate_json(VERIFY_PROMPT.format(analysis=analysis), [ctx.image, overlay], system=SYSTEM_PROMPT)
        except Exception as e:
            log.warning("verifier failed: %s", e)
            ctx.facts.add("verified")  # do not loop forever on a broken VLM
            return AgentReport(self.name, f"VLM call failed: {e.__class__.__name__}", skipped=True)
        if not isinstance(ans, dict):
            ans = {}
        changes = apply_corrections(g, ans, self.vlm.profile.bbox_format, _sent_size(self.vlm, ctx.image))
        ctx.feedback.append(ans)
        if ans.get("summary"):
            g.summary = str(ans["summary"])
        issues = ans.get("issues") or []
        for issue in issues if isinstance(issues, list) else [issues]:
            g.notes.append(f"verifier: {issue}")
        rr = ans.get("rerun") or []
        reruns = [r for r in (rr if isinstance(rr, list) else [rr]) if r in ("shapes", "text_reader", "connections")]
        complete = bool(_as_bool(ans.get("is_complete"), not any(changes.values())))
        ctx.facts.add("verifier_ran")
        if complete:
            ctx.facts.add("verified")
        else:
            ctx.facts.discard("verified")
            ctx.requests.extend(r for r in reruns if r not in ctx.requests)
        changed = ", ".join(f"{k}={v}" for k, v in changes.items() if v)
        return AgentReport(
            self.name,
            f"{'complete' if complete else 'needs work'}; applied {changed or 'no changes'}"
            + (f"; re-run requested: {reruns}" if reruns else ""),
            {"is_complete": complete, **changes, "rerun": reruns, "invalidates": ["topology"] if any(changes.values()) else []},
        )


def _as_list(v: Any) -> list:
    return v if isinstance(v, list) else ([] if v in (None, "", {}) else [v])


def _as_dict(v: Any) -> dict:
    return v if isinstance(v, dict) else {}


def _drop_dangling(g: DiagramGraph) -> None:
    sids = {s.id for s in g.shapes}
    tids = {t.id for t in g.texts}
    for s in g.shapes:
        if s.parent_id not in sids:
            s.parent_id = None
    for t in g.texts:
        if t.inside_shape_id not in sids:
            t.inside_shape_id = None
        t.near_shape_ids = [x for x in t.near_shape_ids if x in sids]
    for c in g.connections:
        for ep in c.endpoints:
            if ep.shape_id and ep.shape_id not in sids:
                ep.shape_id = None
            if ep.text_id and ep.text_id not in tids:
                ep.text_id = None


def apply_corrections(g: DiagramGraph, ans: dict, bbox_format: str = "xyxy_1000", sent_size: tuple[int, int] | None = None) -> dict[str, int]:
    """Apply a verifier answer to the graph. Malformed fields are ignored, never fatal."""
    w, h = g.image.width, g.image.height
    ch = dict(removed_shapes=0, retyped=0, added_shapes=0, text_fixed=0, removed_texts=0, removed_conns=0, added_conns=0)
    rm = {r for r in (_ref(x) for x in _as_list(ans.get("remove_shapes"))) if r}
    if rm:
        before = len(g.shapes)
        g.shapes = [s for s in g.shapes if s.id not in rm]
        ch["removed_shapes"] = before - len(g.shapes)
    rt = {r for r in (_ref(x) for x in _as_list(ans.get("remove_texts"))) if r}
    if rt:
        before = len(g.texts)
        g.texts = [t for t in g.texts if t.id not in rt]
        ch["removed_texts"] = before - len(g.texts)
    rc = {r for r in (_ref(x) for x in _as_list(ans.get("remove_connections"))) if r}
    if rc:
        before = len(g.connections)
        g.connections = [c for c in g.connections if c.id not in rc]
        ch["removed_conns"] = before - len(g.connections)
    _drop_dangling(g)  # before anything new is added, so no reference can latch onto a new element
    for sid, typ in _as_dict(ans.get("retype_shapes")).items():
        s = g.shape(str(sid))
        st = _shape_type(typ)
        if s is not None and st != ShapeType.UNKNOWN and st != s.type:
            s.attributes["previous_type"] = s.type.value
            s.type = st
            if "verifier" not in s.source:
                s.source.append("verifier")
            ch["retyped"] += 1
    for raw in _as_list(ans.get("add_shapes")):
        if not isinstance(raw, dict):
            continue
        b = from_model_box(raw.get("bbox"), w, h, bbox_format, sent_size)
        if b is None or any(s.bbox.iou(b) > 0.5 for s in g.shapes):
            continue
        st = _shape_type(raw.get("type"))
        g.shapes.append(Shape(id=g.next_id("S"), type=st, bbox=b.round(), polygon=bbox_polygon(b, st), confidence=0.5, source=["verifier"]))
        ch["added_shapes"] += 1
    for tid, txt in _as_dict(ans.get("text_corrections")).items():
        t = g.text(str(tid))
        if t is not None and isinstance(txt, (str, int, float)) and str(txt).strip() and str(txt).strip() != t.text:
            t.text = str(txt).strip()
            t.confidence = 0.8
            if "verifier" not in t.source:
                t.source.append("verifier")
            ch["text_fixed"] += 1
    for raw in _as_list(ans.get("add_connections")):
        if not isinstance(raw, dict):
            continue
        ends = [r for r in (_ref(e) for e in _as_list(raw.get("ends"))) if r]
        items = [(e, g.text(e) or g.shape(e)) for e in ends]
        items = [(e, it) for e, it in items if it is not None]
        if len(items) < 2:
            continue
        eps = [
            Endpoint(
                point=it.bbox.center,
                text_id=e if isinstance(it, TextItem) else None,
                shape_id=e if isinstance(it, Shape) else None,
                attachment="text" if isinstance(it, TextItem) else "boundary",
                is_head=(str(raw.get("type", "")) == "arrow" and k == len(items) - 1),
            )
            for k, (e, it) in enumerate(items)
        ]
        ctype = ConnectionType.ARROW if str(raw.get("type", "line")) == "arrow" else ConnectionType.LINE
        g.connections.append(
            Connection(id=g.next_id("C"), type=ctype, path=[ep.point for ep in eps], endpoints=eps, confidence=0.5, source=["verifier"], attributes={"locked_ends": True})
        )
        ch["added_conns"] += 1
    return ch


def _sent_size(vlm, image: np.ndarray) -> tuple[int, int]:
    """Size of the copy of ``image`` the backend actually sends (needed for pixel bbox formats)."""
    h, w = image.shape[:2]
    m = vlm.profile.max_image_side
    if m and max(w, h) > m:
        r = m / max(w, h)
        return max(1, round(w * r)), max(1, round(h * r))
    return w, h
