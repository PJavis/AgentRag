# Preference Trainer (KTO/ORPO) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Author `scripts/finetune_dpo.py` — a reference-free KTO/ORPO LoRA trainer (3B default) consuming the preference miner's JSONL, VRAM-safe for a 16 GB card.

**Architecture:** A pure, unit-tested `load_preference_records` loader + a `main()` that (mirroring `finetune_qwen_lora.py`) imports Unsloth/TRL inside the function with an install-hint `SystemExit`, builds an HF Dataset, runs `KTOTrainer`/`ORPOTrainer`, and exports a merged 16-bit model. Training is author-only (runs on the GPU box).

**Tech Stack:** Unsloth + TRL (`KTOTrainer`/`ORPOTrainer`) + datasets — heavy deps imported lazily; not in `pyproject` (script `SystemExit`s with a hint).

## Global Constraints

- Heavy imports (`unsloth`, `trl`, `datasets`) live INSIDE `main()`; the module must be importable (and `load_preference_records` testable) without them.
- Methods: `kto` (reads `{prompt,completion,label}`) and `orpo` (reads `{prompt,chosen,rejected}`) — both reference-free. DPO-with-reference is intentionally NOT offered (OOM-risky at 16 GB).
- CLI defaults are frugal: base `unsloth/Qwen2.5-3B-Instruct`, `--max-seq 2048`, `--r 16`, `--beta 0.1`, `--grad-accum 8`, batch size 1.
- Malformed rows skipped (count logged); zero valid rows → `SystemExit`.

---

## File Structure

| Path | Responsibility |
|---|---|
| `scripts/finetune_dpo.py` (create) | loader + author-only KTO/ORPO trainer + CLI |
| `tests/eval/test_finetune_dpo.py` (create) | offline unit tests for the loader |

---

### Task 1: finetune_dpo.py loader (TDD) + trainer scaffold

**Files:**
- Create: `scripts/finetune_dpo.py`
- Create: `tests/eval/test_finetune_dpo.py`

**Interfaces:**
- Produces: `load_preference_records(path: str, method: str) -> list[dict]` — `method` in `{"kto","orpo"}`; returns kept rows, raises `SystemExit` if none.

- [ ] **Step 1: Write the failing loader tests.**

```python
# tests/eval/test_finetune_dpo.py
import json

import pytest

from scripts.finetune_dpo import load_preference_records

_LONG = "x" * 40


def _write(tmp_path, rows):
    p = tmp_path / "pref.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    return str(p)


def test_kto_keeps_valid_and_skips_bad(tmp_path):
    path = _write(tmp_path, [
        {"prompt": "Q1", "completion": "good " + _LONG, "label": True},
        {"prompt": "Q2", "completion": "bad " + _LONG, "label": False},
        {"prompt": "Q3", "completion": "no label " + _LONG},        # skip: no label
        {"prompt": "", "completion": "empty prompt " + _LONG, "label": True},  # skip
        {"prompt": "Q4", "completion": "x", "label": "yes"},         # skip: label not bool
    ])
    rows = load_preference_records(path, "kto")
    assert len(rows) == 2
    assert rows[0] == {"prompt": "Q1", "completion": "good " + _LONG, "label": True}


def test_orpo_keeps_valid_and_skips_identical(tmp_path):
    path = _write(tmp_path, [
        {"prompt": "Q1", "chosen": "c " + _LONG, "rejected": "r " + _LONG},
        {"prompt": "Q2", "chosen": "same", "rejected": "same"},      # skip: identical
        {"prompt": "Q3", "chosen": "only chosen " + _LONG},          # skip: no rejected
    ])
    rows = load_preference_records(path, "orpo")
    assert len(rows) == 1
    assert rows[0]["chosen"] == "c " + _LONG and rows[0]["rejected"] == "r " + _LONG


def test_raises_when_no_valid_rows(tmp_path):
    path = _write(tmp_path, [{"prompt": "", "completion": "", "label": True}])
    with pytest.raises(SystemExit):
        load_preference_records(path, "kto")
```

- [ ] **Step 2: Run them — expect FAIL** (`ModuleNotFoundError: No module named 'scripts.finetune_dpo'`).

Run: `uv run pytest tests/eval/test_finetune_dpo.py -q`

- [ ] **Step 3: Write `scripts/finetune_dpo.py`.**

```python
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
```

- [ ] **Step 4: Run the loader tests — expect PASS** (3 passed).

Run: `uv run pytest tests/eval/test_finetune_dpo.py -q`

- [ ] **Step 5: Import smoke (no trl/unsloth needed).**

Run: `uv run python -c "import scripts.finetune_dpo as m; print(m.parse_args.__name__, m.load_preference_records.__name__)"`
Expected: `parse_args load_preference_records` — the module imports without the GPU deps (they're inside `main`).

- [ ] **Step 6: No-regression.**

Run: `uv run pytest -q --ignore=tests/ontology --ignore=tests/ingestion`
Expected: PASS (prior green + 3 new).

- [ ] **Step 7: Commit.**

```bash
git add scripts/finetune_dpo.py tests/eval/test_finetune_dpo.py
git commit -m "feat(finetune): KTO/ORPO preference trainer (reference-free, 3B, VRAM-safe)

Consumes mine_preference.py JSONL → reference-free KTO ({prompt,completion,label}) or
ORPO ({prompt,chosen,rejected}) LoRA on a 3B base (default Qwen2.5-3B), frugal settings
→ ~8-11 GB on a 16 GB card (avoids the 7B-DPO OOM ceiling). Heavy deps imported inside
main with an install hint; load_preference_records unit-tested offline. Author-only —
training runs on the GPU box."
```

---

## Self-Review

**Spec coverage:** `load_preference_records` → Step 3 + tests Step 1; KTO/ORPO `main()` (lazy imports, Unsloth+TRL, merge/export) → Step 3; CLI frugal defaults → Step 3 `parse_args`; hardware table + reference-free rationale → docstring; malformed-skip + empty-raise → Step 3 + tests. All spec sections mapped.

**Placeholder scan:** none — full module + tests inline, exact commands + expected output. Training is author-only by design (spec "Out of scope: running the training"), not a missing step.

**Type consistency:** `load_preference_records(path, method) -> list[dict]` identical across Step 3 def, `main()` call, and the tests' import; KTO keys `prompt/completion/label`, ORPO keys `prompt/chosen/rejected` consistent between the loader, the docstring, and the miner's output schema.
