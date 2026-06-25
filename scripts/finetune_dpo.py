#!/usr/bin/env python
"""KTO/ORPO LoRA-finetune a small instruct model on AgentRag preference data.

Reference-free (KTO/ORPO) so it fits a 16 GB card with margin — unlike 7B DPO,
which holds chosen+rejected forwards AND a reference model (~16 GB, OOM-prone).

Input (from scripts/mine_preference.py):
    --method kto   → one JSON/line: {"prompt": str, "completion": str, "label": bool}
    --method orpo  → one JSON/line: {"prompt": str, "chosen": str, "rejected": str}

After training, merges the LoRA adapter and exports 16-bit safetensors to --out;
convert to GGUF with scripts/convert_to_ollama.sh.

Hardware (RTX 5060 Ti 16 GB):
    3B  KTO/ORPO QLoRA r=16, seq 2048 → ~8-11 GB   (safe)
    7B                                → ~14-16 GB   (tight; lower --max-seq/--r or use 3B)

Usage (on the GPU box):
    uv pip install "unsloth[cu121-torch240]" trl
    uv run python scripts/finetune_dpo.py --method kto \\
        --train data/finetune/preference.jsonl --out models/qwen-agentrag-pref-3b
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def load_preference_records(path: str, method: str) -> list[dict]:
    """Read the miner's JSONL and keep only well-formed rows for the chosen method.
    Malformed rows are skipped (count logged). Raises SystemExit if none remain."""
    kept: list[dict] = []
    skipped = 0
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            skipped += 1
            continue
        if method == "kto":
            if (isinstance(r.get("prompt"), str) and r["prompt"]
                    and isinstance(r.get("completion"), str) and r["completion"]
                    and isinstance(r.get("label"), bool)):
                kept.append({"prompt": r["prompt"], "completion": r["completion"], "label": r["label"]})
                continue
        else:  # orpo
            chosen, rejected = r.get("chosen"), r.get("rejected")
            if (isinstance(r.get("prompt"), str) and r["prompt"]
                    and isinstance(chosen, str) and chosen
                    and isinstance(rejected, str) and rejected
                    and chosen != rejected):
                kept.append({"prompt": r["prompt"], "chosen": chosen, "rejected": rejected})
                continue
        skipped += 1
    logger.info("loaded %d %s records (%d skipped)", len(kept), method, skipped)
    if not kept:
        raise SystemExit(f"no valid {method} records in {path}")
    return kept


def main(args: argparse.Namespace) -> None:
    records = load_preference_records(args.train, args.method)
    try:
        from datasets import Dataset
        from unsloth import FastLanguageModel
        if args.method == "kto":
            from trl import KTOConfig, KTOTrainer
        else:
            from trl import ORPOConfig, ORPOTrainer
    except ImportError as exc:
        raise SystemExit(
            "Training deps missing. On the GPU box:\n"
            '    uv pip install "unsloth[cu121-torch240]" trl'
        ) from exc

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.base, max_seq_length=args.max_seq, load_in_4bit=True, dtype=None,
    )
    model = FastLanguageModel.get_peft_model(
        model, r=args.r, lora_alpha=2 * args.r, use_gradient_checkpointing="unsloth",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
    )
    dataset = Dataset.from_list(records)
    common = dict(
        per_device_train_batch_size=1,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        learning_rate=5e-6,
        logging_steps=10,
        beta=args.beta,
        max_length=args.max_seq,
        output_dir=str(Path(args.out) / "trainer"),
    )
    if args.method == "kto":
        trainer = KTOTrainer(model=model, args=KTOConfig(**common),
                             train_dataset=dataset, tokenizer=tokenizer)
    else:
        trainer = ORPOTrainer(model=model, args=ORPOConfig(**common),
                              train_dataset=dataset, tokenizer=tokenizer)
    trainer.train()
    model.save_pretrained_merged(args.out, tokenizer, save_method="merged_16bit")
    logger.info("saved merged model → %s", args.out)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--method", choices=["kto", "orpo"], default="kto")
    p.add_argument("--base", default="unsloth/Qwen2.5-3B-Instruct")
    p.add_argument("--train", default="data/finetune/preference.jsonl")
    p.add_argument("--out", default="models/qwen-agentrag-pref-3b")
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--max-seq", type=int, default=2048)
    p.add_argument("--r", type=int, default=16)
    p.add_argument("--beta", type=float, default=0.1)
    p.add_argument("--grad-accum", type=int, default=8)
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
