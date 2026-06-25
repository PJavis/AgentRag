# Preference trainer (KTO / ORPO, VRAM-safe) — design

**Date:** 2026-06-25 · **Objective:** author `scripts/finetune_dpo.py` that turns the
preference miner's output into a LoRA-tuned model, **reference-free** (KTO/ORPO) on a
**3B base** so it fits a 16 GB card with margin — not 7B DPO (which hugs 16 GB).

## Context

`scripts/mine_preference.py` emits KTO `{prompt, completion, label}` (default) or DPO
`{prompt, chosen, rejected}` (`--format dpo`). The existing `scripts/finetune_qwen_lora.py`
(SFT) sets the pattern: Unsloth + TRL, heavy imports **inside `main()`** with a clear
`SystemExit` install hint (so the file is syntax-importable without the libs), merges
LoRA → exports safetensors → `convert_to_ollama.sh`.

`trl`/`unsloth` are NOT in `pyproject` (same as the qwen script) — the script
`SystemExit`s with an install hint. **Hardware:** the user's training box is a 16 GB
RTX 5060 Ti. 7B DPO holds chosen+rejected forwards AND a reference model → ~16 GB (OOM
risk). **KTO and ORPO are reference-free** → fit comfortably; a **3B base** adds margin.

## Scope

In: `scripts/finetune_dpo.py` (methods `kto` + `orpo`, 3B default, frugal) + a unit-tested
pure loader. Out: running the training (author-only — runs on the home box); adding
`trl`/`unsloth` to `pyproject`; 7B/DPO-with-reference (deliberately excluded as OOM-risky).

## Design

### Testable core (pure, offline)
`load_preference_records(path: str, method: str) -> list[dict]` — read JSONL; per method
keep only well-formed rows:
- `kto`: requires non-empty `prompt`, `completion`, and a boolean `label`.
- `orpo`: requires non-empty `prompt`, `chosen`, `rejected`, with `chosen != rejected`.
Malformed rows are skipped (logged count); returns the kept rows. Raises `SystemExit`
if zero valid rows.

### main() (mirrors finetune_qwen_lora.py)
- Heavy imports inside, guarded: `unsloth.FastLanguageModel`, `trl.{KTOTrainer,KTOConfig,
  ORPOTrainer,ORPOConfig}`, `datasets.Dataset` — `ImportError` → `SystemExit` with
  `uv pip install "unsloth[cu121-torch240]" trl` hint.
- Load base (4-bit) + LoRA (`r`, `lora_alpha=2*r`, gradient checkpointing) via Unsloth.
- Build `Dataset.from_list(load_preference_records(...))`.
- `kto` → `KTOTrainer` with `KTOConfig(beta, per_device_train_batch_size=1,
  gradient_accumulation_steps, max_length=max_seq, ...)`.
- `orpo` → `ORPOTrainer` with `ORPOConfig(beta, ... same frugal args)`.
- Train → `model.save_pretrained_merged(out, tokenizer, save_method="merged_16bit")`.

### CLI (frugal VRAM defaults)
`--method {kto,orpo}` (default kto), `--base` (default `unsloth/Qwen2.5-3B-Instruct`),
`--train` (default `data/finetune/preference.jsonl`), `--out` (default
`models/qwen-agentrag-pref-3b`), `--epochs 1`, `--max-seq 2048`, `--r 16`, `--beta 0.1`,
`--grad-accum 8`.

### Docstring hardware table
- 3B KTO/ORPO QLoRA r=16, seq 2048 → ~8-11 GB (safe on 16 GB).
- 7B → ~14-16 GB (tight) — reduce `--max-seq`/`--r` or stay on 3B.
- DPO-with-reference is intentionally not offered (OOM-prone at 16 GB).

## Data flow
`adapter_chat_feedback` → `mine_preference.py` (kto|dpo jsonl) → `finetune_dpo.py`
(`load_preference_records` → Unsloth+TRL KTO/ORPO LoRA) → merged 16-bit safetensors →
`convert_to_ollama.sh` → Ollama model used by `LLM_TASK_MODEL_MAP`.

## Error handling
Heavy-import failure → actionable `SystemExit`. Zero valid records → `SystemExit`.
Malformed rows skipped, not fatal.

## Testing
`tests/eval/test_finetune_dpo.py` (offline, no GPU/trl): `load_preference_records` keeps
valid KTO rows / valid ORPO rows; skips rows missing keys, with wrong label type, or
`chosen == rejected`; raises on an all-malformed file. (Training itself is author-only.)
