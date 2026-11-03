"""Fine-tune local handwriting readers on an OCR dataset (``drawid ocr-dataset``):

* **TrOCR** (encoder-decoder line recogniser): full fine-tune, standard recipe;
* **SmolVLM** (tiny local VLM): LoRA through :func:`finetune_vlm`, merged into a
  standalone model folder.

Use the results as consensus engines ``trocr:<out_dir>`` / ``smolvlm:<out_dir>``
(``consensus.extra_engines`` in the config), so the bottom tiers of the cascade
read this particular hand and the expensive models are needed less often.
Ported from the author's finetune_smolvlm_lora.py; needs ``pip install
'drawing-identifier[hf]'`` and, realistically, a GPU.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

log = logging.getLogger(__name__)

SMOLVLM_PROMPT = "Transcribe the handwritten words in this image. Return ONLY the words, nothing else. If there are no words, return an empty string."


def _manifest(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(ln) for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]


def write_vlm_ocr_jsonl(dataset_dir: str | Path, prompt: str = SMOLVLM_PROMPT) -> tuple[str, str]:
    """manifest.jsonl -> chat samples for :func:`finetune_vlm` (image + prompt -> transcription)."""
    d = Path(dataset_dir)
    names = []
    for src, dst in (("manifest.jsonl", "vlm_ocr_train.jsonl"), ("manifest_val.jsonl", "vlm_ocr_val.jsonl")):
        with open(d / dst, "w", encoding="utf-8") as fh:
            for r in _manifest(d / src):
                fh.write(json.dumps({"images": [r["file"]], "messages": [{"role": "user", "content": "<image>\n" + prompt}, {"role": "assistant", "content": r["text"]}]}, ensure_ascii=False) + "\n")
        names.append(dst)
    return names[0], names[1]


def finetune_smolvlm(dataset_dir: str | Path, base: str = "HuggingFaceTB/SmolVLM-500M-Instruct", out_dir: str | Path = "models/smolvlm-ocr", epochs: float = 8, lr: float = 1e-4, batch_size: int = 2, lora_r: int = 16, max_steps: int = -1) -> Path:
    from .finetune_vlm import finetune_vlm

    train, val = write_vlm_ocr_jsonl(dataset_dir)
    return finetune_vlm(dataset_dir, base, out_dir, epochs, lr, batch_size, 1, lora_r, 512, max_steps, None, True, False, train, val, merge=True)


def finetune_trocr(dataset_dir: str | Path, base: str = "microsoft/trocr-base-handwritten", out_dir: str | Path = "models/trocr-ocr", epochs: int = 5, lr: float = 3e-5, batch_size: int = 8, max_steps: int = -1) -> Path:
    import torch
    from PIL import Image
    from transformers import TrOCRProcessor, VisionEncoderDecoderModel

    d = Path(dataset_dir)
    train = _manifest(d / "manifest.jsonl")
    if not train:
        raise ValueError(f"no training rows in {d / 'manifest.jsonl'}")
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    proc = TrOCRProcessor.from_pretrained(base)
    model = VisionEncoderDecoderModel.from_pretrained(base).to(dev)
    tok = proc.tokenizer
    model.config.decoder_start_token_id = model.config.decoder_start_token_id or tok.cls_token_id or tok.bos_token_id
    model.config.pad_token_id = tok.pad_token_id
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    model.train()
    step = 0
    for epoch in range(epochs):
        order = torch.randperm(len(train)).tolist()
        total = 0.0
        for i in range(0, len(order), batch_size):
            rows = [train[j] for j in order[i : i + batch_size]]
            pix = proc(images=[Image.open(d / r["file"]).convert("RGB") for r in rows], return_tensors="pt").pixel_values.to(dev)
            labels = tok([r["text"] for r in rows], padding=True, return_tensors="pt").input_ids
            labels[labels == tok.pad_token_id] = -100
            loss = model(pixel_values=pix, labels=labels.to(dev)).loss
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss)
            step += 1
            if 0 < max_steps <= step:
                break
        log.info("epoch %d loss %.4f", epoch + 1, total / max(1, (len(order) + batch_size - 1) // batch_size))
        if 0 < max_steps <= step:
            break
    out = Path(out_dir)
    model.save_pretrained(out)
    proc.save_pretrained(out)
    return out
