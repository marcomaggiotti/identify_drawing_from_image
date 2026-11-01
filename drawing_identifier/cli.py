"""Command line interface: ``drawid <command> ...``"""

from __future__ import annotations

import argparse
import json
import logging
import sys
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
    p.add_argument("--ocr", choices=["vlm", "trocr", "tesseract", "none"], help="how text is read")
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
