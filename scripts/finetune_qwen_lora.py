#!/usr/bin/env python
"""QLoRA-finetune Qwen2.5-7B (or 14B) on AgentRag SFT samples with Unsloth.

Input format (one JSON per line in `data/finetune/llm_sft.jsonl`):
    {"system": str, "user": str, "output": str}

After training, merges LoRA adapter back into the base, exports a 16-bit safetensors
checkpoint to `models/qwen-agentrag-7b/`. Convert to GGUF with
`scripts/convert_to_ollama.sh`.

Hardware:
    7B QLoRA r=16 → ~11 GB VRAM (fits 16 GB)
    14B QLoRA r=8 → ~15 GB VRAM (tight; reduce max_seq_length if OOM)

Usage:
    uv run python scripts/finetune_qwen_lora.py \\
        --base unsloth/Qwen2.5-7B-Instruct \\
        --train data/finetune/llm_sft.jsonl \\
        --out models/qwen-agentrag-7b \\
        --epochs 1 --max-seq 4096 --r 16
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main(args: argparse.Namespace) -> None:
    # Heavy imports happen here so the file is at least syntax-importable
    # without unsloth installed (CI / lint-only environments).
    try:
        from unsloth import FastLanguageModel
    except ImportError as exc:
        raise SystemExit(
            "unsloth is required. Install with:\n"
            "    uv pip install \"unsloth[cu121-torch240]\""
        ) from exc

    from trl import SFTTrainer
    from transformers import TrainingArguments
    from datasets import Dataset

    logger.info("loading base: %s", args.base)
    model, tok = FastLanguageModel.from_pretrained(
        args.base,
        max_seq_length=args.max_seq,
        dtype=None,
        load_in_4bit=True,
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=args.r,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_alpha=args.r * 2,
        lora_dropout=0.0,
        bias="none",
        use_gradient_checkpointing="unsloth",
    )

    rows = []
    with open(args.train, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            rows.append({"text": tok.apply_chat_template(
                [
                    {"role": "system", "content": r.get("system", "")},
                    {"role": "user", "content": r.get("user", "")},
                    {"role": "assistant", "content": r.get("output", "")},
                ],
                tokenize=False,
            )})
    if not rows:
        raise SystemExit("empty training file")
    logger.info("training rows: %d", len(rows))

    ds = Dataset.from_list(rows)

    out_path = Path(args.out)
    out_path.mkdir(parents=True, exist_ok=True)

    trainer = SFTTrainer(
        model=model,
        tokenizer=tok,
        train_dataset=ds,
        dataset_text_field="text",
        max_seq_length=args.max_seq,
        args=TrainingArguments(
            output_dir=str(out_path / "checkpoints"),
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=args.grad_accum,
            warmup_steps=10,
            num_train_epochs=args.epochs,
            learning_rate=2e-4,
            fp16=False,
            bf16=True,
            logging_steps=10,
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="linear",
            seed=42,
            save_strategy="no",
        ),
    )
    trainer.train()

    logger.info("merging LoRA into base and saving fp16 to %s", out_path)
    model.save_pretrained_merged(str(out_path), tok, save_method="merged_16bit")
    logger.info("done. Next: bash scripts/convert_to_ollama.sh %s qwen-agentrag", out_path)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base", default="unsloth/Qwen2.5-7B-Instruct")
    p.add_argument("--train", default="data/finetune/llm_sft.jsonl")
    p.add_argument("--out", default="models/qwen-agentrag-7b")
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--max-seq", type=int, default=4096)
    p.add_argument("--r", type=int, default=16)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--grad-accum", type=int, default=4)
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
