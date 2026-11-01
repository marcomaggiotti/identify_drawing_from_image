"""Command line interface: ``drawid <command> ...``"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import yaml

from .config import default_config_dict, load_config, set_dotted


def _overrides(args) -> dict:
    o: dict = {}
    if getattr(args, "vlm", None) and isinstance(args.vlm, str):
        o["default_vlm"] = args.vlm
    if getattr(args, "no_vlm", False):
        o["default_vlm"] = "none"
    for item in getattr(args, "agent_vlm", None) or []:
        agent, _, spec = item.partition("=")
        if not spec:
            raise SystemExit(f"--agent-vlm expects agent=profile_or_spec, got {item!r}")
        o.setdefault("agent_vlms", {})[agent.strip()] = spec.strip()
    if getattr(args, "mode", None):
        set_dotted(o, "orchestrator.mode", args.mode)
    if getattr(args, "orientation", None) is not None:
        set_dotted(o, "preprocess.orientation", args.orientation)
    if getattr(args, "detector", None):
        set_dotted(o, "shapes.detector", args.detector)
    if getattr(args, "weights", None):
        set_dotted(o, "shapes.weights", args.weights)
    if getattr(args, "ocr", None):
        set_dotted(o, "text.ocr", args.ocr)
    if getattr(args, "no_verify", False):
        set_dotted(o, "verifier.enabled", False)
    for item in getattr(args, "set", None) or []:
        key, _, val = item.partition("=")
        set_dotted(o, key.strip(), yaml.safe_load(val))
    return o


def _config(args):
    return load_config(getattr(args, "config", None), _overrides(args))


def _add_common(p: argparse.ArgumentParser, vlm_multi: bool = False) -> None:
    p.add_argument("--config", help="YAML config file (see configs/example.yaml)")
    if not vlm_multi:
        p.add_argument("--vlm", help="VLM for all agents: profile name, provider:model[@base_url], 'auto' or 'none'")
    p.add_argument("--no-vlm", action="store_true", help="classical CV (+ local detector) only")
    p.add_argument("--agent-vlm", action="append", metavar="AGENT=VLM", help="per-agent VLM, e.g. text_reader=ollama-qwen (repeatable)")
    p.add_argument("--mode", choices=["pipeline", "planner"], help="fixed pipeline or VLM planner")
    p.add_argument("--orientation", help="auto | none | 0 | 90 | 180 | 270")
    p.add_argument("--detector", choices=["cv", "yolo", "cv+yolo"], help="shape detector")
    p.add_argument("--weights", help="local detector weights (drawid train-detector)")
    p.add_argument("--ocr", help="how text is read: vlm | consensus | tesseract | easyocr | trocr | <engine id> | none")
    p.add_argument("--no-verify", action="store_true", help="skip the VLM verification loop")
    p.add_argument("--set", action="append", metavar="KEY=VALUE", help="override any config value, e.g. shapes.fit_threshold=0.84")


# ------------------------------------------------------------------ commands
def cmd_analyze(args) -> int:
    from .agents import Orchestrator

    cfg = _config(args)
    orc = Orchestrator(cfg)
    images: list[Path] = []
    for p in args.images:
        pp = Path(p)
        if pp.is_dir():
            images += sorted(x for x in pp.iterdir() if x.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff", ".bmp"})
        else:
            images.append(pp)
    for img in images:
        res = orc.analyze(str(img))
        paths = res.save(args.out)
        st = res.graph.stats()
        print(f"{img.name}: {st['shapes']} shapes {st['shape_types']}, {st['connections']} connections, {st['texts']} texts ({st['texts_read']} read)")
        if args.print:
            print(res.description())
        print("  -> " + ", ".join(f"{k}: {v}" for k, v in paths.items()))
    return 0


def cmd_vlms(args) -> int:
    from .vlm import PROVIDER_INFO, VLMRegistry

    cfg = _config(args)
    reg = VLMRegistry(cfg)
    print("Configured VLM profiles (select with --vlm NAME, or per agent with --agent-vlm agent=NAME):\n")
    rows = reg.describe()
    w = max(len(r["name"]) for r in rows) if rows else 4
    for r in rows:
        print(f"  {r['name']:<{w}}  {r['provider']:<17} {r['model']:<36} available={r['available']:<3}  {r['status']}")
    print(f"\ndefault_vlm: {cfg.default_vlm}  (auto order: {', '.join(cfg.auto_preference)})")
    if cfg.agent_vlms:
        print("agent_vlms: " + ", ".join(f"{k}={v}" for k, v in cfg.agent_vlms.items()))
    print("\nAd-hoc specs: provider:model[@base_url]")
    for prov, (models, how) in PROVIDER_INFO.items():
        print(f"  {prov:<17} e.g. {models}   ({how})")
    return 0


def cmd_agents(args) -> int:
    from .agents import Orchestrator

    orc = Orchestrator(_config(args))
    print(f"mode: {orc.config.orchestrator.mode}; planner VLM: {orc.planner_vlm.name if orc.planner_vlm else 'none'}\n")
    for line in orc.describe_agents():
        print(" - " + line)
    return 0


def cmd_synth(args) -> int:
    from .synthetic import SynthConfig, generate_dataset, generate_sample

    cfg = SynthConfig(fonts_dir=args.fonts_dir)
    if args.config:
        with open(args.config, encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
        for k, v in (data.get("synthetic") or {}).items():
            if hasattr(cfg, k):
                setattr(cfg, k, tuple(v) if isinstance(getattr(cfg, k), tuple) else v)
    if args.preview:
        import cv2

        from .export.render import render_overlay

        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        for i in range(args.preview):
            img, g = generate_sample(args.seed * 1000 + i, cfg)
            cv2.imwrite(str(out / f"preview_{i}.jpg"), img)
            cv2.imwrite(str(out / f"preview_{i}_gt.jpg"), render_overlay(img, g))
        print(f"wrote {args.preview} preview(s) to {out}")
        return 0
    formats = [f.strip() for f in args.formats.split(",") if f.strip()]
    s = generate_dataset(args.out, args.n, args.seed, args.val_fraction, formats, cfg, args.workers)
    print(json.dumps({k: v for k, v in s.items() if k != "config"}, indent=2))
    return 0


def cmd_train_detector(args) -> int:
    from .training.yolo import train_detector

    p = train_detector(args.data, args.model, args.epochs, args.imgsz, args.batch, args.device, args.project, args.name, args.workers, args.out)
    print(f"weights: {p}\nuse them with: drawid analyze IMAGE --detector cv+yolo --weights {p}")
    return 0


def cmd_finetune_vlm(args) -> int:
    from .training.finetune_vlm import finetune_vlm

    p = finetune_vlm(
        args.data,
        args.base,
        args.out,
        args.epochs,
        args.lr,
        args.batch_size,
        args.grad_accum,
        args.lora_r,
        args.max_image_side,
        args.max_steps,
        args.limit,
    )
    print(f"adapter: {p}\nselect it with: drawid analyze IMAGE --vlm local-finetuned  (or set vlms.local-finetuned.extra.adapter_path)")
    return 0


def cmd_eval(args) -> int:
    from .agents import Orchestrator
    from .evaluation import evaluate_dataset, report_markdown

    cfg = _config(args)
    r = evaluate_dataset(args.data, Orchestrator(cfg), args.split, args.limit, args.save_dir)
    print(json.dumps(r, indent=2))
    print()
    print(report_markdown([(args.name or cfg.default_vlm, r)]))
    return 0


def cmd_benchmark(args) -> int:
    from .agents import Orchestrator
    from .evaluation import evaluate_dataset, report_markdown, save_report

    rows = []
    for spec in args.vlm or ["none"]:
        args_vlm = argparse.Namespace(**{**vars(args), "vlm": spec})
        cfg = _config(args_vlm)
        print(f"== {spec}")
        r = evaluate_dataset(args.data, Orchestrator(cfg), args.split, args.limit, None, progress=False)
        rows.append((spec, r))
    md = report_markdown(rows)
    print(md)
    if args.report:
        Path(args.report).write_text(md + "\n", encoding="utf-8")
        save_report(Path(args.report).with_suffix(".json"), rows)
    return 0


def cmd_engines(args) -> int:
    from .transcription.engines import TIER_NAMES
    from .ui_bench import engine_catalogue

    cfg = _config(args)
    infos = engine_catalogue(cfg, ",".join(args.vlm_spec or []))
    print("Transcription engines, in consensus order (local first, then by size and price):\n")
    for i in infos:
        mark = "ready" if i["ready"] else ("download" if i["available"] else "no")
        size = f"{i['params_b']:g}B" if i["params_b"] < 99 else "cloud"
        print(f"  {i['id']:<28} tier {i['tier']} {TIER_NAMES[i['tier']]:<18} {size:>6}  weight {i['weight']:<4} {mark:<8} {i['status']}")
    print("\nready = usable now; download = weights fetched on first explicit use; select with --engines id1,id2 or auto/local")
    return 0


def _parse_box(text: str | None):
    if not text:
        return None
    vals = [float(v) for v in text.replace(" ", "").split(",")]
    if len(vals) != 4:
        raise SystemExit("--box expects x0,y0,x1,y1")
    return tuple(vals)


def cmd_transcribe(args) -> int:
    from .transcription import crop_segment, select_engines
    from .transcription.bench import consensus_of_rows, rows_to_json, run_segment_benchmark, sort_rows
    from .transcription.consensus import ConsensusCascade
    from .vision.preprocess import load_image

    cfg = _config(args)
    cc = cfg.consensus.model_copy(
        update={
            k: v
            for k, v in {
                "min_agree": args.min_agree,
                "agreement": args.agreement,
                "similarity": args.similarity,
                "allow_remote": False if args.no_remote else None,
                "per_engine": True if args.per_engine else None,
            }.items()
            if v is not None
        }
    )
    selection = args.engines if args.engines in ("auto", "local") else [e.strip() for e in args.engines.split(",") if e.strip()]
    specs = args.vlm_spec or []
    if not isinstance(selection, str):
        selection = selection + [f"vlm:{s}" for s in specs]
    sel = select_engines(cfg, selection, extra_specs=specs)
    if sel.unavailable and isinstance(selection, str):
        print(f"  {len(sel.unavailable)} engine(s) not available, see `drawid engines`")
    else:
        for k, why in sel.unavailable.items():
            print(f"  skipped {k}: {why}")
    for k in sel.unknown:
        print(f"  unknown engine {k!r} (see `drawid engines`)")
    if not sel.engines:
        print("no engine available")
        return 1
    seg = crop_segment(load_image(args.image), _parse_box(args.box), args.pad, args.rotate)
    if args.consensus:
        casc = ConsensusCascade(sel.engines, cc)
        print("\n".join(casc.describe()))
        res = casc.run(seg)
        for i, v in enumerate(res.votes, 1):
            reading = f"error: {v.error}" if v.error else v.text
            print(f"  {i:>2}. {v.engine_id:<26} {reading!r:<40} {'agrees' if v.agrees else ''}")
        print(res.summary())
        if args.json:
            Path(args.json).write_text(rows_to_json([], None, res, args.reference, args.box), encoding="utf-8")
        return 0
    variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    rows = run_segment_benchmark(seg, sel.engines, variants, args.reference, case_sensitive=cc.case_sensitive)
    cons = consensus_of_rows(rows, cc)
    w = max(len(r.engine_id) for r in rows)
    for r in sort_rows(rows):
        score = f"CER {r.cer:.3f}" if r.cer is not None else (f"agree {r.agreement:.2f}" if r.agreement is not None else "")
        reading = f"error: {r.error}" if r.error else r.text
        print(f"  {r.engine_id:<{w}} {r.variant:<9} {score:<11} {r.seconds:>5.2f}s  {reading!r}")
    print(cons.summary())
    if args.json:
        Path(args.json).write_text(rows_to_json(rows, cons, None, args.reference, args.box), encoding="utf-8")
        print(f"saved {args.json}")
    return 0


def cmd_ui(args) -> int:
    from .ui import launch

    launch(args.config, args.host, args.port, args.share)
    return 0


def cmd_config(args) -> int:
    print(yaml.safe_dump(default_config_dict(), sort_keys=False))
    return 0


def cmd_fonts(args) -> int:
    from .synthetic.fonts import download_fonts

    return download_fonts(args.out)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="drawid", description="Agentic identification of hand drawings: shapes, nesting, connections and text.")
    ap.add_argument("-v", "--verbose", action="count", default=0)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("analyze", help="analyse one or more images (or folders)")
    p.add_argument("images", nargs="+")
    p.add_argument("--out", default="outputs")
    p.add_argument("--print", action="store_true", help="print the structure description")
    _add_common(p)
    p.set_defaults(func=cmd_analyze)

    p = sub.add_parser("vlms", help="list VLM profiles/providers and whether they are usable here")
    _add_common(p)
    p.set_defaults(func=cmd_vlms)

    p = sub.add_parser("agents", help="list the agents and the VLM each one would use")
    _add_common(p)
    p.set_defaults(func=cmd_agents)

    p = sub.add_parser("synth", help="generate a synthetic training set (or --preview a few pages)")
    p.add_argument("--out", default="data/synth")
    p.add_argument("-n", type=int, default=1000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--val-fraction", type=float, default=0.1)
    p.add_argument("--formats", default="yolo,coco,vlm,ocr")
    p.add_argument("--fonts-dir", default="assets/fonts", help="handwriting TTF fonts (drawid fonts)")
    p.add_argument("--workers", type=int, default=None)
    p.add_argument("--preview", type=int, default=0, help="only write N preview pages with ground-truth overlays")
    p.add_argument("--config", help="YAML with a `synthetic:` section overriding SynthConfig fields")
    p.set_defaults(func=cmd_synth)

    p = sub.add_parser("train-detector", help="train the local YOLO-seg detector on a synthetic set")
    p.add_argument("--data", required=True, help="data.yaml written by drawid synth")
    p.add_argument("--model", default="yolo11n-seg.pt", help="checkpoint or architecture (.yaml = from scratch)")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--imgsz", type=int, default=1024)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--device", default=None, help="cpu, 0, 0,1, mps ...")
    p.add_argument("--project", default="runs/detector")
    p.add_argument("--name", default="train")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--out", default="models/detector.pt")
    p.set_defaults(func=cmd_train_detector)

    p = sub.add_parser("finetune-vlm", help="LoRA fine-tune a local VLM on the synthetic vlm_*.jsonl")
    p.add_argument("--data", required=True)
    p.add_argument("--base", default="Qwen/Qwen2.5-VL-3B-Instruct")
    p.add_argument("--out", default="runs/vlm-lora")
    p.add_argument("--epochs", type=float, default=1.0)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--grad-accum", type=int, default=8)
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--max-image-side", type=int, default=1024)
    p.add_argument("--max-steps", type=int, default=-1)
    p.add_argument("--limit", type=int, default=None)
    p.set_defaults(func=cmd_finetune_vlm)

    p = sub.add_parser("eval", help="score the pipeline against a synthetic split")
    p.add_argument("--data", required=True)
    p.add_argument("--split", default="val")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--save-dir", default=None)
    p.add_argument("--name", default=None)
    _add_common(p)
    p.set_defaults(func=cmd_eval)

    p = sub.add_parser("benchmark", help="compare several VLMs on the same synthetic split")
    p.add_argument("--data", required=True)
    p.add_argument("--vlm", action="append", help="profile or spec; repeat to compare (use 'none' for CV only)")
    p.add_argument("--split", default="val")
    p.add_argument("--limit", type=int, default=20)
    p.add_argument("--report", default=None, help="write a markdown (+ .json) report")
    _add_common(p, vlm_multi=True)
    p.set_defaults(func=cmd_benchmark)

    p = sub.add_parser("engines", help="list transcription engines (VLMs + local OCR) in consensus order")
    p.add_argument("--vlm-spec", action="append", help="extra VLM spec to include, e.g. ollama:minicpm-v")
    _add_common(p)
    p.set_defaults(func=cmd_engines)

    p = sub.add_parser("transcribe", help="read one segment with many engines and compare them, or run the consensus cascade")
    p.add_argument("image")
    p.add_argument("--box", help="segment x0,y0,x1,y1 in image pixels (default: whole image)")
    p.add_argument("--rotate", type=int, default=0, choices=[0, 90, 180, 270], help="rotate the segment clockwise")
    p.add_argument("--pad", type=int, default=4)
    p.add_argument("--engines", default="auto", help="auto | local | comma-separated ids (see `drawid engines`)")
    p.add_argument("--vlm-spec", action="append", help="extra VLM to include, e.g. ollama:minicpm-v (repeatable)")
    p.add_argument("--variants", default="original", help="original,enhanced,binarized")
    p.add_argument("--reference", help="ground-truth text to score against")
    p.add_argument("--consensus", action="store_true", help="run the local-first escalating cascade instead of every engine")
    p.add_argument("--min-agree", type=int)
    p.add_argument("--agreement", type=float)
    p.add_argument("--similarity", type=float)
    p.add_argument("--no-remote", action="store_true", help="never escalate to cloud models")
    p.add_argument("--per-engine", action="store_true", help="escalate one engine at a time")
    p.add_argument("--json", help="save all readings to this JSON file")
    _add_common(p)
    p.set_defaults(func=cmd_transcribe)

    p = sub.add_parser("ui", help="web UI with VLM selection (needs gradio)")
    p.add_argument("--config")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=7860)
    p.add_argument("--share", action="store_true")
    p.set_defaults(func=cmd_ui)

    p = sub.add_parser("config", help="print the default configuration")
    p.set_defaults(func=cmd_config)

    p = sub.add_parser("fonts", help="download free handwriting fonts for the synthetic generator")
    p.add_argument("--out", default="assets/fonts")
    p.set_defaults(func=cmd_fonts)
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    level = logging.WARNING - 10 * min(args.verbose, 2)
    logging.basicConfig(level=level, format="%(levelname)s %(name)s: %(message)s")
    return int(args.func(args) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
