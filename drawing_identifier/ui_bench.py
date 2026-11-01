"""Benchmark tab: pick a segment of a page, transcribe it with every available engine
(VLMs, local OCRs, image variants), check the readings against a reference and
compare them all in one view; or run the local-first consensus cascade.

The callbacks are plain functions (RGB numpy in, data out) so they can be tested
without a browser; :func:`build_benchmark_tab` wires them into Gradio.
"""

from __future__ import annotations

import html
import tempfile
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .config import AppConfig, ConsensusConfig, load_config
from .transcription import TIER_NAMES, VARIANTS, all_engines, crop_segment, make_variants, select_engines
from .transcription.bench import (
    COLUMNS,
    BenchRow,
    consensus_of_rows,
    rows_to_csv,
    rows_to_json,
    run_segment_benchmark,
    score_rows,
    sort_rows,
)
from .transcription.consensus import ConsensusCascade, ConsensusResult, rank_engines
from .transcription.metrics import cer, diff_html

Box = tuple[float, float, float, float]
EXAMPLES_DIR = Path(__file__).resolve().parent.parent / "examples" / "images"


# ----------------------------------------------------------------- page / segment
def full_box(rgb: np.ndarray | None) -> Box:
    if rgb is None:
        return (0, 0, 0, 0)
    h, w = rgb.shape[:2]
    return (0, 0, w, h)


def clean_box(rgb: np.ndarray | None, box) -> Box | None:
    if rgb is None or box is None:
        return None
    try:
        x0, y0, x1, y1 = (float(v or 0) for v in box)
    except (TypeError, ValueError):
        return None
    h, w = rgb.shape[:2]
    x0, x1 = sorted((min(max(0, x0), w), min(max(0, x1), w)))
    y0, y1 = sorted((min(max(0, y0), h), min(max(0, y1), h)))
    if x1 - x0 < 2 or y1 - y0 < 2:
        return None
    return (x0, y0, x1, y1)


def render_page(rgb: np.ndarray | None, box: Box | None, regions: dict | None = None, pending: tuple | None = None) -> np.ndarray | None:
    """The page with detected regions (thin green), the selected segment (thick red) and a pending first click."""
    if rgb is None:
        return None
    out = np.ascontiguousarray(rgb.copy())
    t = max(2, int(round(max(out.shape[:2]) / 500)))
    for r in (regions or {}).values():
        x0, y0, x1, y1 = (int(round(v)) for v in r["box"])
        cv2.rectangle(out, (x0, y0), (x1, y1), (40, 160, 40), max(1, t - 1))
    if box is not None:
        x0, y0, x1, y1 = (int(round(v)) for v in box)
        cv2.rectangle(out, (x0, y0), (x1, y1), (230, 30, 30), t + 1)
    if pending is not None:
        cv2.drawMarker(out, (int(pending[0]), int(pending[1])), (230, 30, 30), cv2.MARKER_CROSS, 8 * t, t + 1)
    return out


def segment(rgb: np.ndarray | None, box: Box | None, rotation: int = 0, pad: int = 4) -> np.ndarray | None:
    """Cropped (and rotated) segment, RGB."""
    if rgb is None:
        return None
    bgr = np.ascontiguousarray(rgb[:, :, ::-1])
    return crop_segment(bgr, box, int(pad or 0), int(rotation or 0))[:, :, ::-1]


def variant_gallery(seg_rgb: np.ndarray | None) -> list[tuple[np.ndarray, str]]:
    if seg_rgb is None:
        return []
    v = make_variants(np.ascontiguousarray(seg_rgb[:, :, ::-1]))
    return [(img[:, :, ::-1], name) for name, img in v.items()]


def click(rgb: np.ndarray | None, points: list, xy: tuple[float, float]) -> tuple[list, Box | None]:
    """Two clicks define a box; a third click starts a new one."""
    pts = list(points or [])
    if len(pts) >= 2:
        pts = []
    pts.append((float(xy[0]), float(xy[1])))
    if len(pts) == 2:
        (ax, ay), (bx, by) = pts
        return pts, clean_box(rgb, (ax, ay, bx, by))
    return pts, None


def detect_regions(rgb: np.ndarray | None, config_path: str | None = None) -> tuple[dict[str, dict], int]:
    """Text regions found by the classical pipeline, in the uploaded image's coordinates."""
    if rgb is None:
        return {}, 0
    from .agents import Orchestrator
    from .evaluation import map_to_original

    cfg = load_config(config_path, {"default_vlm": "none", "verifier": {"enabled": False}, "vlm_analyst": {"enabled": False}, "text": {"ocr": "none"}})
    res = Orchestrator(cfg).analyze(np.ascontiguousarray(rgb[:, :, ::-1]), source="benchmark")
    g = map_to_original(res.graph)
    rot = int(res.graph.image.rotation or 0)
    out: dict[str, dict] = {}
    for t in g.texts:
        x0, y0, x1, y1 = t.bbox.xyxy
        where = f"inside {t.inside_shape_id}" if t.inside_shape_id else (f"near {t.near_shape_ids[0]}" if t.near_shape_ids else t.placement.value)
        label = f"{t.id} · {where} · {int(x1 - x0)}×{int(y1 - y0)} px"
        out[label] = {"box": (x0, y0, x1, y1), "rotation": rot, "text_id": t.id}
    return out, rot


# ----------------------------------------------------------------- engines
def engine_catalogue(config: AppConfig, custom_specs: str = "") -> list[dict[str, Any]]:
    specs = [s.strip() for s in (custom_specs or "").split(",") if s.strip()]
    return [e.info() for e in rank_engines(all_engines(config, extra_specs=specs))]


def engine_choices(infos: list[dict[str, Any]]) -> list[tuple[str, str]]:
    """Checkbox entries for the engines that can run (ready now, or after downloading weights)."""
    out = []
    for i in infos:
        if not i["available"]:
            continue
        mark = "✓" if i["ready"] else "↓"
        note = "" if i["ready"] else " (downloads weights)"
        out.append((f"{mark} {i['id']} · {TIER_NAMES[i['tier']]}{note}", i["id"]))
    return out


def unavailable_markdown(infos: list[dict[str, Any]]) -> str:
    rows = [i for i in infos if not i["available"]]
    if not rows:
        return "Every configured engine is available."
    lines = ["| engine | tier | how to enable |", "|---|---|---|"]
    lines += [f"| `{i['id']}` | {TIER_NAMES[i['tier']]} | {html.escape(i['status'])} |" for i in rows]
    return "\n".join(lines)


def preset(infos: list[dict[str, Any]], which: str) -> list[str]:
    if which == "ready":
        return [i["id"] for i in infos if i["ready"]]
    if which == "local":
        return [i["id"] for i in infos if i["ready"] and i["local"]]
    if which == "all":
        return [i["id"] for i in infos if i["available"]]
    return []


def resolve(config: AppConfig, engine_ids: list[str], custom_specs: str = ""):
    specs = [s.strip() for s in (custom_specs or "").split(",") if s.strip()]
    wanted = list(engine_ids or []) + [f"vlm:{s}" for s in specs]
    return select_engines(config, wanted, extra_specs=specs)


def consensus_params(min_agree=2, agreement=0.6, similarity=0.85, case_sensitive=True, allow_remote=True, per_engine=False) -> ConsensusConfig:
    return ConsensusConfig(
        min_agree=int(min_agree),
        agreement=float(agreement),
        similarity=float(similarity),
        case_sensitive=bool(case_sensitive),
        allow_remote=bool(allow_remote),
        per_engine=bool(per_engine),
    )


# ----------------------------------------------------------------- runs
def compare(config: AppConfig, seg_rgb: np.ndarray, engine_ids: list[str], custom_specs: str, variants: list[str], reference: str, params: ConsensusConfig):
    sel = resolve(config, engine_ids, custom_specs)
    if not sel.engines:
        return [], None, sel
    bgr = np.ascontiguousarray(seg_rgb[:, :, ::-1])
    rows = run_segment_benchmark(bgr, sel.engines, variants or ["original"], reference or None, case_sensitive=params.case_sensitive)
    rows = score_rows(rows, reference or None, params.case_sensitive, params)
    return rows, consensus_of_rows(rows, params), sel


def cascade(config: AppConfig, seg_rgb: np.ndarray, engine_ids: list[str], custom_specs: str, params: ConsensusConfig):
    sel = resolve(config, engine_ids, custom_specs)
    if not sel.engines:
        return None, sel, []
    casc = ConsensusCascade(sel.engines, params)
    return casc.run(np.ascontiguousarray(seg_rgb[:, :, ::-1])), sel, casc.describe()


def rows_from_cascade(res: ConsensusResult, reference: str | None, params: ConsensusConfig) -> list[BenchRow]:
    rows = [
        BenchRow(v.engine_id, v.label, "vlm" if v.engine_id.startswith("vlm:") else "ocr", v.local, v.tier, v.weight, v.variant, v.text, v.confidence, v.crossed_out, v.seconds, v.error)
        for v in res.votes
    ]
    return score_rows(rows, reference or None, params.case_sensitive, params)


# ----------------------------------------------------------------- presentation
def table(rows: list[BenchRow], sort_by: str = "auto") -> list[list]:
    return [r.as_row() for r in sort_rows(rows, sort_by)]


def summary_markdown(rows: list[BenchRow], cons: ConsensusResult | None, reference: str | None, sel=None) -> str:
    if not rows:
        msg = "No engine could run."
        if sel is not None and (sel.unavailable or sel.unknown):
            msg += " " + "; ".join(f"`{k}`: {v}" for k, v in sel.unavailable.items())
            if sel.unknown:
                msg += f"; unknown: {sel.unknown}"
        return msg
    ok = [r for r in rows if r.error is None]
    lines = [f"**{len(rows)} reading(s)** from {len({r.engine_id for r in rows})} engine(s), {len(rows) - len(ok)} failed."]
    if reference:
        best = min((r for r in ok if r.cer is not None), key=lambda r: r.cer, default=None)
        exact = [r for r in ok if r.exact]
        if best is not None:
            lines.append(f"Best against the reference: **{best.engine_id}** ({best.variant}) with CER {best.cer:.3f}" + (f"; {len(exact)} exact match(es)." if exact else "."))
    if cons is not None and cons.status != "no_answer":
        tag = "✅ consensus" if cons.status == "consensus" else "⚠️ best effort (engines disagree)"
        lines.append(f"{tag}: **{html.escape(cons.text)}** — agreement {cons.agreement:.0%}, {cons.independent} independent engine(s).")
        if reference:
            lines.append(f"Consensus vs reference: CER {cer(reference, cons.text):.3f}")
    if sel is not None and sel.unavailable:
        lines.append("Skipped (not available): " + ", ".join(f"`{k}` ({v})" for k, v in sel.unavailable.items()))
    return "\n\n".join(lines)


def diff_panel(rows: list[BenchRow], target: str, label: str, sort_by: str = "auto") -> str:
    """One table with every reading: diff-highlighted against the target, scores, status."""
    if not rows:
        return ""
    th = "padding:4px 8px;text-align:left;border-bottom:2px solid #ccc;white-space:nowrap"
    td = "padding:4px 8px;border-bottom:1px solid #eee;vertical-align:top"
    head = (
        f"<p style='margin:6px 0'>Readings compared with the <b>{html.escape(label)}</b> "
        f"<code style='font-size:15px'>{html.escape(target) or '(empty)'}</code> — "
        "<del style='background:#fdd;color:#a00'>missing</del> <ins style='background:#dfd;color:#060;text-decoration:none'>extra</ins>; "
        "green rows support the consensus.</p>"
    )
    cols = ["engine", "tier", "variant", "reading", "CER", "agree", "conf", "sec"]
    trs = []
    for r in sort_rows(rows, sort_by):
        if r.error:
            body = f"<span style='color:#a00'>error: {html.escape(r.error)}</span>"
        else:
            body = diff_html(target, r.text) if target else (html.escape(r.text) or "<span style='color:#888'>(empty)</span>")
        bg = "#eef8ee" if r.in_consensus else "transparent"

        def num(v, nd=2):
            return "" if v is None else f"{v:.{nd}f}"

        where = "🖥" if r.local else "☁"
        trs.append(
            f"<tr style='background:{bg}'>"
            f"<td style='{td};white-space:nowrap'><code>{html.escape(r.engine_id)}</code></td>"
            f"<td style='{td};white-space:nowrap'>{where} {TIER_NAMES[r.tier]}</td>"
            f"<td style='{td}'>{html.escape(r.variant)}</td>"
            f"<td style='{td};font-family:monospace;font-size:15px'>{body}</td>"
            f"<td style='{td}'>{num(r.cer, 3)}</td><td style='{td}'>{num(r.agreement)}</td>"
            f"<td style='{td}'>{num(r.confidence)}</td><td style='{td}'>{num(r.seconds, 1)}</td></tr>"
        )
    header = "".join(f"<th style='{th}'>{c}</th>" for c in cols)
    return head + f"<div style='overflow-x:auto'><table style='border-collapse:collapse;width:100%'><tr>{header}</tr>{''.join(trs)}</table></div>"


def cascade_markdown(res: ConsensusResult | None, plan_lines: list[str], sel=None) -> str:
    if res is None:
        extra = ""
        if sel is not None and sel.unavailable:
            extra = " " + "; ".join(f"`{k}`: {v}" for k, v in sel.unavailable.items())
        return "No engine could run." + extra
    tag = {"consensus": "✅ consensus", "best_effort": "⚠️ best effort", "no_answer": "❌ no answer"}[res.status]
    out = [
        f"### Consensus cascade: {tag}",
        f"**{html.escape(res.text) or '(empty)'}** — agreement {res.agreement:.0%}, {res.independent} independent engine(s) agree; "
        f"{res.calls} call(s) ({res.remote_calls} remote), {len(res.skipped)} engine(s) not needed, {res.seconds:.1f} s.",
        "",
        "Escalation plan (local first, then by size and price):",
        "",
        *[f"{'✔' if any(e in ln for e in [v.engine_id for v in res.votes]) else '·'} {ln}" for ln in plan_lines],
        "",
        "| step | engine | tier | reading | conf | agrees | s |",
        "|---|---|---|---|---|---|---|",
    ]
    for i, v in enumerate(res.votes, 1):
        reading = f"error: {v.error}" if v.error else (v.text or "(empty)")
        conf = "" if v.confidence is None else f"{v.confidence:.2f}"
        out.append(f"| {i} | `{v.engine_id}` | {TIER_NAMES[v.tier]} | {html.escape(reading)[:80]} | {conf} | {'✔' if v.agrees else ''} | {v.seconds:.2f} |")
    if res.skipped:
        out += ["", "Not needed: " + ", ".join(f"`{s}`" for s in res.skipped)]
    return "\n".join(out)


def export(rows: list[BenchRow], cons: ConsensusResult | None, casc: ConsensusResult | None, reference: str | None, box) -> list[str]:
    d = Path(tempfile.mkdtemp(prefix="drawid_bench_"))
    (d / "transcriptions.json").write_text(rows_to_json(rows, cons, casc, reference, list(box) if box else None), encoding="utf-8")
    (d / "transcriptions.csv").write_text(rows_to_csv(rows), encoding="utf-8")
    return [str(d / "transcriptions.json"), str(d / "transcriptions.csv")]


def load_synthetic_sample(dataset_dir: str, split: str = "val", index: int = 0) -> tuple[np.ndarray | None, str, str]:
    """(RGB crop, reference text, message) from the OCR crops of ``drawid synth``."""
    tsv = Path(dataset_dir) / "ocr" / f"{split}.tsv"
    if not tsv.exists():
        return None, "", f"not found: {tsv} (generate one with `drawid synth --out {dataset_dir}`)"
    rows = [ln.rstrip("\n").split("\t") for ln in tsv.read_text(encoding="utf-8").splitlines()[1:] if ln.strip()]
    if not rows:
        return None, "", f"{tsv} is empty"
    i = int(index or 0) % len(rows)
    img = cv2.imread(str(Path(dataset_dir) / "ocr" / rows[i][0]))
    if img is None:
        return None, "", f"cannot read {rows[i][0]}"
    return img[:, :, ::-1].copy(), rows[i][1], f"sample {i + 1}/{len(rows)}: `{rows[i][0]}`"


def list_examples() -> list[str]:
    if not EXAMPLES_DIR.is_dir():
        return []
    return sorted(p.name for p in EXAMPLES_DIR.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"})


def load_example(name: str) -> np.ndarray | None:
    if not name:
        return None
    from .vision.preprocess import load_image

    return load_image(str(EXAMPLES_DIR / name))[:, :, ::-1].copy()


# ----------------------------------------------------------------- gradio wiring
def build_benchmark_tab(gr, config_path: str | None = None) -> None:
    """Create the Benchmark tab inside the current ``gr.Blocks`` context."""
    base_cfg = load_config(config_path)
    infos0 = engine_catalogue(base_cfg)

    with gr.Tab("Benchmark"):
        gr.Markdown(
            "### Transcription benchmark\n"
            "Select a segment of a page, transcribe it with every engine you tick (VLMs, local OCR, image variants), "
            "check the readings against a reference and compare them all here. "
            "**Consensus cascade** runs the engines from local/small to cloud/expensive and stops as soon as they agree."
        )
        orig = gr.State(None)
        points = gr.State([])
        regions = gr.State({})
        rows_state = gr.State([])
        infos = gr.State(infos0)
        last = gr.State({})
        with gr.Row():
            with gr.Column(scale=1, min_width=360):
                src = gr.Image(type="numpy", label="1. Page or segment image", sources=["upload", "clipboard"], height=220)
                example = gr.Dropdown(list_examples(), value=None, label="…or an example page")
                with gr.Accordion("…or a synthetic sample with known text", open=False):
                    ds_dir = gr.Textbox(value="data/synth", label="dataset folder (from `drawid synth`)")
                    with gr.Row():
                        ds_split = gr.Radio(["val", "train"], value="val", label="split")
                        ds_index = gr.Number(value=0, precision=0, label="sample #", minimum=0)
                    ds_btn = gr.Button("Load sample")
                    ds_msg = gr.Markdown()
                gr.Markdown("**2. Segment** — click two corners on the page (right), pick a detected region, or type coordinates.")
                with gr.Row():
                    x0 = gr.Number(label="x0", precision=0, min_width=60)
                    y0 = gr.Number(label="y0", precision=0, min_width=60)
                    x1 = gr.Number(label="x1", precision=0, min_width=60)
                    y1 = gr.Number(label="y1", precision=0, min_width=60)
                with gr.Row():
                    rot = gr.Radio([0, 90, 180, 270], value=0, label="rotate segment (clockwise)")
                    pad = gr.Slider(0, 40, value=4, step=1, label="padding (px)")
                with gr.Row():
                    whole_btn = gr.Button("Whole image", size="sm")
                    detect_btn = gr.Button("Detect text regions", size="sm")
                region_dd = gr.Dropdown([], label="detected text regions", allow_custom_value=False)
                gr.Markdown("**3. Engines** — ordered local → cloud, small → large. ✓ ready · ↓ downloads weights on first use")
                engines_cb = gr.CheckboxGroup(engine_choices(infos0), value=preset(infos0, "ready"), label="engines")
                with gr.Accordion("Engines not available here (how to enable them)", open=False):
                    unavail_md = gr.Markdown(unavailable_markdown(infos0))
                with gr.Row():
                    b_ready = gr.Button("ready", size="sm")
                    b_local = gr.Button("local only", size="sm")
                    b_all = gr.Button("everything available", size="sm")
                    b_none = gr.Button("none", size="sm")
                    b_refresh = gr.Button("↻ refresh", size="sm")
                custom = gr.Textbox(label="extra VLMs (specs, comma separated)", placeholder="ollama:minicpm-v, openai_compatible:Qwen/Qwen2.5-VL-7B-Instruct@http://localhost:8000/v1")
                variants_cb = gr.CheckboxGroup(list(VARIANTS), value=["original"], label="image variants (every engine runs on each)")
                reference = gr.Textbox(label="4. Reference transcription (optional ground truth)", placeholder="what is really written")
                with gr.Accordion("Consensus settings", open=False):
                    min_agree = gr.Slider(1, 4, value=base_cfg.consensus.min_agree, step=1, label="independent engines that must agree")
                    agreement = gr.Slider(0.3, 1.0, value=base_cfg.consensus.agreement, step=0.05, label="weighted agreement needed")
                    simil = gr.Slider(0.5, 1.0, value=base_cfg.consensus.similarity, step=0.01, label="similarity for two readings to agree")
                    case_cb = gr.Checkbox(value=base_cfg.consensus.case_sensitive, label="case sensitive")
                    remote_cb = gr.Checkbox(value=base_cfg.consensus.allow_remote, label="allow escalation to cloud models")
                    per_engine_cb = gr.Checkbox(value=base_cfg.consensus.per_engine, label="escalate one engine at a time")
                with gr.Row():
                    run_all = gr.Button("Compare all selected engines", variant="primary")
                    run_casc = gr.Button("Consensus cascade (local → cloud)", variant="secondary")
                with gr.Row():
                    sort_dd = gr.Dropdown(["auto", "cer", "agreement", "engine"], value="auto", label="sort by", scale=1)
                    rescore_btn = gr.Button("Re-score with reference", scale=1)
            with gr.Column(scale=2):
                page = gr.Image(label="Page — click two corners to select the segment", interactive=False, height=460)
                with gr.Row():
                    seg_view = gr.Image(label="Selected segment", interactive=False, height=170)
                    gal = gr.Gallery(label="Image variants", columns=3, height=170, object_fit="contain")
                summary = gr.Markdown()
                diff_view = gr.HTML()
                casc_md = gr.Markdown()
                with gr.Accordion("Raw table (copy / sort)", open=False):
                    result_table = gr.Dataframe(headers=COLUMNS, value=[], interactive=False, wrap=True, label="All readings")
                files = gr.File(label="Download results", file_count="multiple")

        # ---- callbacks (thin wrappers)
        def _view(img, bx, by, ex, ey, r, p, regs, pts=None):
            box = clean_box(img, (bx, by, ex, ey))
            pending = pts[0] if pts and len(pts) == 1 else None
            seg = segment(img, box, r, p) if img is not None else None
            return render_page(img, box, regs, pending), seg, variant_gallery(seg)

        def on_image(img):
            b = full_box(img)
            page_img, seg, g = _view(img, *b, 0, 4, {})
            return img, [], {}, gr.update(choices=[], value=None), *b, 0, page_img, seg, g

        def on_example(name):
            return load_example(name) if name else gr.update()

        def on_sample(d, split, idx):
            img, ref, msg = load_synthetic_sample(d, split, idx)
            return (img if img is not None else gr.update()), ref, msg

        def on_click(img, pts, bx, by, ex, ey, r, p, regs, evt: gr.SelectData):
            if img is None or evt is None or evt.index is None:
                return pts, bx, by, ex, ey, gr.update(), gr.update(), gr.update()
            pts, box = click(img, pts, (evt.index[0], evt.index[1]))
            if box is not None:
                bx, by, ex, ey = box
            page_img, seg, g = _view(img, bx, by, ex, ey, r, p, regs, pts)
            return pts, bx, by, ex, ey, page_img, seg, g

        on_click.__annotations__["evt"] = gr.SelectData  # postponed annotations hide the type from gradio

        def on_box(img, bx, by, ex, ey, r, p, regs):
            return _view(img, bx, by, ex, ey, r, p, regs)

        def on_whole(img, r, p, regs):
            b = full_box(img)
            return (*b, *_view(img, *b, r, p, regs))

        def on_detect(img):
            if img is None:
                return {}, gr.update(choices=[], value=None), gr.update(), "Upload an image first."
            regs, rotation = detect_regions(img, config_path)
            msg = f"{len(regs)} text region(s) found" + (f"; the page is rotated {rotation}° (segments are turned upright)" if rotation else "")
            return regs, gr.update(choices=list(regs), value=None), render_page(img, None, regs), msg

        def on_region(label, img, regs, p):
            if not label or label not in (regs or {}):
                return (gr.update(),) * 8
            r = regs[label]
            bx, by, ex, ey = r["box"]
            page_img, seg, g = _view(img, bx, by, ex, ey, r["rotation"], p, regs)
            return bx, by, ex, ey, r["rotation"], page_img, seg, g

        def on_preset(inf, which):
            return preset(inf, which)

        def on_refresh(spec):
            cfg = load_config(config_path)
            inf = engine_catalogue(cfg, spec)
            extra = [i["id"] for i in inf if i["id"].startswith("vlm:") and i["available"] and i["id"][4:] in [s.strip() for s in (spec or "").split(",")]]
            return inf, gr.update(choices=engine_choices(inf), value=preset(inf, "ready") + extra), unavailable_markdown(inf)

        def on_compare(img, bx, by, ex, ey, r, p, eng, spec, vars_, ref, ma, ag, si, cs, rem, pe, sort_by):
            if img is None:
                return [], "Upload an image first.", "", [], None, {}
            box = clean_box(img, (bx, by, ex, ey)) or full_box(img)
            seg = segment(img, box, r, p)
            params = consensus_params(ma, ag, si, cs, rem, pe)
            cfg = load_config(config_path)
            rows, cons, sel = compare(cfg, seg, eng, spec, vars_, ref, params)
            target, label = (ref, "reference") if ref else ((cons.text, "consensus") if cons and cons.text else ("", "consensus"))
            st = {"box": box, "consensus": cons.to_dict() if cons else None}
            return (
                table(rows, sort_by),
                summary_markdown(rows, cons, ref, sel),
                diff_panel(rows, target, label, sort_by),
                [r_.__dict__ for r_ in rows],
                export(rows, cons, None, ref, box) if rows else None,
                st,
            )

        def on_cascade(img, bx, by, ex, ey, r, p, eng, spec, ref, ma, ag, si, cs, rem, pe, sort_by):
            if img is None:
                return "Upload an image first.", [], "", "", [], None
            box = clean_box(img, (bx, by, ex, ey)) or full_box(img)
            seg = segment(img, box, r, p)
            params = consensus_params(ma, ag, si, cs, rem, pe)
            cfg = load_config(config_path)
            res, sel, plan_lines = cascade(cfg, seg, eng, spec, params)
            if res is None:
                return cascade_markdown(None, [], sel), [], "", "", [], None
            rows = rows_from_cascade(res, ref, params)
            target, label = (ref, "reference") if ref else (res.text, "cascade answer")
            return (
                cascade_markdown(res, plan_lines, sel),
                table(rows, sort_by),
                summary_markdown(rows, res, ref, sel),
                diff_panel(rows, target, label, sort_by),
                [r_.__dict__ for r_ in rows],
                export(rows, None, res, ref, box),
            )

        def on_rescore(rows_d, ref, ma, ag, si, cs, rem, pe, sort_by, st):
            rows = [BenchRow(**d) for d in (rows_d or [])]
            if not rows:
                return [], "Run a comparison first.", "", rows_d, None
            params = consensus_params(ma, ag, si, cs, rem, pe)
            rows = score_rows(rows, ref or None, params.case_sensitive, params)
            cons = consensus_of_rows(rows, params)
            target, label = (ref, "reference") if ref else (cons.text, "consensus")
            box = (st or {}).get("box")
            return table(rows, sort_by), summary_markdown(rows, cons, ref), diff_panel(rows, target, label, sort_by), [r_.__dict__ for r_ in rows], export(rows, cons, None, ref, box)

        coords = [x0, y0, x1, y1]
        views = [page, seg_view, gal]
        src.change(on_image, [src], [orig, points, regions, region_dd, *coords, rot, page, seg_view, gal])
        example.change(on_example, [example], [src])
        ds_btn.click(on_sample, [ds_dir, ds_split, ds_index], [src, reference, ds_msg])
        page.select(on_click, [orig, points, *coords, rot, pad, regions], [points, *coords, *views])
        for comp in coords:
            comp.input(on_box, [orig, *coords, rot, pad, regions], views)
        rot.input(on_box, [orig, *coords, rot, pad, regions], views)
        pad.release(on_box, [orig, *coords, rot, pad, regions], views)
        whole_btn.click(on_whole, [orig, rot, pad, regions], [*coords, *views])
        detect_btn.click(on_detect, [orig], [regions, region_dd, page, summary])
        region_dd.input(on_region, [region_dd, orig, regions, pad], [*coords, rot, *views])
        b_ready.click(lambda inf: on_preset(inf, "ready"), [infos], [engines_cb])
        b_local.click(lambda inf: on_preset(inf, "local"), [infos], [engines_cb])
        b_all.click(lambda inf: on_preset(inf, "all"), [infos], [engines_cb])
        b_none.click(lambda inf: on_preset(inf, "none"), [infos], [engines_cb])
        b_refresh.click(on_refresh, [custom], [infos, engines_cb, unavail_md])
        settings = [min_agree, agreement, simil, case_cb, remote_cb, per_engine_cb]
        run_all.click(
            on_compare,
            [orig, *coords, rot, pad, engines_cb, custom, variants_cb, reference, *settings, sort_dd],
            [result_table, summary, diff_view, rows_state, files, last],
        )
        run_casc.click(
            on_cascade,
            [orig, *coords, rot, pad, engines_cb, custom, reference, *settings, sort_dd],
            [casc_md, result_table, summary, diff_view, rows_state, files],
        )
        rescore_btn.click(on_rescore, [rows_state, reference, *settings, sort_dd, last], [result_table, summary, diff_view, rows_state, files])
        sort_dd.input(on_rescore, [rows_state, reference, *settings, sort_dd, last], [result_table, summary, diff_view, rows_state, files])

