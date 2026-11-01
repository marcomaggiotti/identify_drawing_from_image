"""Small Gradio web UI: upload a drawing, pick the VLM (globally or per agent), inspect the result."""

from __future__ import annotations

import numpy as np

from .config import load_config

AGENT_SLOTS = ["preprocess", "triage", "vlm_analyst", "text_reader", "verifier", "planner"]
DEFAULT = "(default)"


def build_app(config_path: str | None = None):
    try:
        import gradio as gr
    except ImportError as e:  # pragma: no cover - optional dependency
        raise RuntimeError("the UI needs gradio: pip install 'drawing-identifier[ui]'") from e
    from .agents import Orchestrator
    from .export.formats import to_mermaid
    from .vlm import VLMRegistry

    base_cfg = load_config(config_path)
    profiles = list(base_cfg.vlms)
    choices = ["auto", "none"] + profiles + ["custom"]

    def availability():
        rows = VLMRegistry(base_cfg).describe()
        return [[r["name"], r["provider"], r["model"], r["available"], r["status"]] for r in rows]

    def run(image, vlm, custom, mode, detector, weights, ocr, verify, *agent_choices):
        if image is None:
            return None, "Upload an image first.", "{}", ""
        sel = custom.strip() if vlm == "custom" else vlm
        overrides: dict = {
            "default_vlm": sel or "auto",
            "orchestrator": {"mode": mode},
            "shapes": {"detector": detector, "weights": weights or None},
            "text": {"ocr": ocr},
            "verifier": {"enabled": bool(verify)},
            "agent_vlms": {a: c for a, c in zip(AGENT_SLOTS, agent_choices) if c and c != DEFAULT},
        }
        cfg = load_config(config_path, overrides)
        bgr = np.asarray(image)[:, :, ::-1].copy()
        res = Orchestrator(cfg).analyze(bgr, source="upload")
        overlay = res.overlay()[:, :, ::-1]
        md = (f"> {res.graph.summary}\n\n" if res.graph.summary else "") + res.description()
        md += "\n\n**Agent trace**\n\n" + "\n".join(f"{e.step}. `{e.agent}` ({e.seconds:.1f}s): {e.message}" for e in res.graph.trace)
        return overlay, md, res.graph.to_json(include_trace=False), to_mermaid(res.graph)

    with gr.Blocks(title="Hand drawing identifier") as app:
        gr.Markdown("## Hand drawing identifier\nShapes, how they are nested and connected, and the text inside/near them.")
        with gr.Row():
            with gr.Column(scale=1):
                image = gr.Image(type="numpy", label="Drawing")
                vlm = gr.Dropdown(choices, value=base_cfg.default_vlm if base_cfg.default_vlm in choices else "auto", label="VLM for all agents")
                custom = gr.Textbox(label="custom VLM spec", placeholder="ollama:qwen2.5vl:7b  |  openai_compatible:Qwen/Qwen2.5-VL-7B-Instruct@http://localhost:8000/v1")
                with gr.Accordion("Per-agent VLM", open=False):
                    agent_dd = [gr.Dropdown([DEFAULT, "none"] + profiles, value=DEFAULT, label=a) for a in AGENT_SLOTS]
                mode = gr.Radio(["pipeline", "planner"], value=base_cfg.orchestrator.mode, label="Orchestration")
                detector = gr.Radio(["cv", "cv+yolo", "yolo"], value=base_cfg.shapes.detector, label="Shape detector")
                weights = gr.Textbox(value=base_cfg.shapes.weights or "", label="Local detector weights")
                ocr = gr.Radio(["vlm", "trocr", "tesseract", "none"], value=base_cfg.text.ocr, label="Text reading")
                verify = gr.Checkbox(value=base_cfg.verifier.enabled, label="VLM verification loop")
                go = gr.Button("Analyse", variant="primary")
                with gr.Accordion("VLM availability", open=False):
                    gr.Dataframe(availability(), headers=["profile", "provider", "model", "available", "status"], interactive=False)
            with gr.Column(scale=2):
                overlay = gr.Image(label="Detections")
                desc = gr.Markdown()
                with gr.Tab("JSON"):
                    js = gr.Code(language="json")
                with gr.Tab("Mermaid"):
                    mm = gr.Code()
        go.click(run, [image, vlm, custom, mode, detector, weights, ocr, verify, *agent_dd], [overlay, desc, js, mm])
    return app


def launch(config_path: str | None = None, host: str = "127.0.0.1", port: int = 7860, share: bool = False) -> None:
    build_app(config_path).launch(server_name=host, server_port=port, share=share)


