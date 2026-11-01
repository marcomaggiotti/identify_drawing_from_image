"""LoRA fine-tuning of a local VLM (default Qwen2.5-VL-3B) on the synthetic ``vlm_*.jsonl`` files.

The training target is exactly the JSON the ``vlm_analyst`` agent asks for, so the
resulting adapter can be selected like any other VLM::

    drawid finetune-vlm --data data/synth --base Qwen/Qwen2.5-VL-3B-Instruct --out runs/vlm-lora
    drawid analyze page.jpg --vlm local-finetuned        # profile in the default config

Needs a GPU for realistic sizes: ``pip install 'drawing-identifier[hf]'``.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image

log = logging.getLogger(__name__)


def load_jsonl(path: str | Path, limit: int | None = None) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                rows.append(json.loads(line))
            if limit and len(rows) >= limit:
                break
    return rows


def to_chat(record: dict) -> list[dict]:
    """Our JSONL (plain-string contents with an ``<image>`` marker) -> multimodal chat messages."""
    msgs = []
    for m in record["messages"]:
        content = m["content"]
        if m["role"] == "user":
            parts: list[dict[str, Any]] = []
            text = content
            if "<image>" in text:
                text = text.replace("<image>\n", "").replace("<image>", "")
                parts.append({"type": "image"})
            parts.append({"type": "text", "text": text})
            msgs.append({"role": "user", "content": parts})
        else:
            msgs.append({"role": m["role"], "content": [{"type": "text", "text": content}]})
    return msgs


def _load_image(root: Path, rel: str, max_side: int) -> Image.Image:
    im = Image.open(root / rel).convert("RGB")
    w, h = im.size
    s = max_side / max(w, h)
    if s < 1:
        im = im.resize((max(1, int(w * s)), max(1, int(h * s))), Image.LANCZOS)
    return im


@dataclass
class VLMCollator:
    """Tokenises full conversations and masks everything but the assistant answer."""

    processor: Any
    root: Path
    max_image_side: int = 1024

    def __call__(self, batch: list[dict]) -> dict:
        fulls, prompts, images = [], [], []
        for rec in batch:
            msgs = to_chat(rec)
            fulls.append(self.processor.apply_chat_template(msgs, tokenize=False))
            prompts.append(self.processor.apply_chat_template(msgs[:-1], tokenize=False, add_generation_prompt=True))
            images.append([_load_image(self.root, p, self.max_image_side) for p in rec.get("images", [])])
        enc = self.processor(text=fulls, images=images, return_tensors="pt", padding=True)
        labels = enc["input_ids"].clone()
        pad_id = getattr(getattr(self.processor, "tokenizer", self.processor), "pad_token_id", None)
        if pad_id is not None:
            labels[labels == pad_id] = -100
        for i, (p, ims) in enumerate(zip(prompts, images)):
            n_prompt = self.processor(text=[p], images=[ims], return_tensors="pt")["input_ids"].shape[1]
            labels[i, :n_prompt] = -100
        enc["labels"] = labels
        return dict(enc)


def finetune_vlm(
    data_dir: str | Path,
    base_model: str = "Qwen/Qwen2.5-VL-3B-Instruct",
    out_dir: str | Path = "runs/vlm-lora",
    epochs: float = 1.0,
    learning_rate: float = 1e-4,
    batch_size: int = 1,
    grad_accum: int = 8,
    lora_r: int = 16,
    max_image_side: int = 1024,
    max_steps: int = -1,
    limit: int | None = None,
    bf16: bool = True,
    gradient_checkpointing: bool = True,
) -> Path:
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForImageTextToText, AutoProcessor, Trainer, TrainingArguments

    data_dir = Path(data_dir)
    train = load_jsonl(data_dir / "vlm_train.jsonl", limit)
    val_path = data_dir / "vlm_val.jsonl"
    val = load_jsonl(val_path, max(8, (limit or 10**9) // 10)) if val_path.exists() else None
    log.info("training on %d samples (%s validation)", len(train), len(val) if val else 0)

    processor = AutoProcessor.from_pretrained(base_model)
    tok = getattr(processor, "tokenizer", None)
    if tok is not None:
        tok.padding_side = "right"
    dtype = torch.bfloat16 if (bf16 and torch.cuda.is_available()) else torch.float32
    model = AutoModelForImageTextToText.from_pretrained(base_model, dtype=dtype, device_map="auto" if torch.cuda.is_available() else None)
    if gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
    lora = LoraConfig(
        r=lora_r,
        lora_alpha=2 * lora_r,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    args = TrainingArguments(
        output_dir=str(out_dir),
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        num_train_epochs=epochs,
        max_steps=max_steps,
        learning_rate=learning_rate,
        warmup_steps=20,
        lr_scheduler_type="cosine",
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=2,
        eval_strategy="epoch" if val else "no",
        bf16=dtype == torch.bfloat16,
        remove_unused_columns=False,
        report_to=[],
        dataloader_num_workers=2,
    )
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train,
        eval_dataset=val,
        data_collator=VLMCollator(processor, data_dir, max_image_side),
    )
    trainer.train()
    out = Path(out_dir)
    model.save_pretrained(out)
    processor.save_pretrained(out)
    (out / "drawid_adapter.json").write_text(json.dumps({"base_model": base_model, "max_image_side": max_image_side}, indent=2))
    log.info("adapter saved to %s", out)
    return out
