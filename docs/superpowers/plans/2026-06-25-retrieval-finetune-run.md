# Retrieval Fine-tune — Run Plan (embedding + reranker)

P3 step 1-2 of `FINETUNE_STRATEGY.md`. **All scripts already exist** — this is a *run*
plan for the **16 GB home box** (RTX 5060 Ti), not new code. Highest retrieval ROI
(+10-20% recall@10 / +5-10% MRR), low OOM risk. The 6 GB session box can't train these
safely — run on the 16 GB card.

> **VRAM budget (16 GB):** embedding LoRA (e5-base / bge-m3) ~5-6 GB; reranker FT ~5 GB.
> Both leave wide margin. (Contrast: 7B LLM DPO ~16 GB — avoid; see the KTO/ORPO trainer.)

## Prerequisites
- A corpus ingested (ES `agentrag_segments`) so synthetic Q-gen + hard-negs have content.
- An LLM for synthetic question generation: Gemini (key set) or local Ollama (qwen).
- `unsloth`/`sentence-transformers` on the GPU box (`sentence-transformers` is already a dep).

## Pipeline (run in order)

```bash
# 1. Mine (query, positive, negative) triplets — thumbs feedback + synthetic Q-gen + hard negs
make mine-pairs
#   = uv run python scripts/mine_finetune_pairs.py --out data/finetune/embed_triplets.jsonl
#   tune: --synth-chunks 800 --questions-per-chunk 3 --hard-negs-per-positive 4
head -3 data/finetune/embed_triplets.jsonl | jq .   # spot-check

# 2. Split 90/10 train/test
make split-pairs
#   → data/finetune/embed_train.jsonl + embed_test.jsonl

# 3a. Fine-tune the EMBEDDING model (LoRA) — ~5-6 GB
uv run python scripts/finetune_embedding.py \
    --base BAAI/bge-m3 \
    --train data/finetune/embed_train.jsonl \
    --out models/agentrag-embed-v1

# 3b. (optional, step 2) Fine-tune the RERANKER cross-encoder — ~5 GB
uv run python scripts/finetune_reranker.py \
    --base BAAI/bge-reranker-v2-m3 \
    --train data/finetune/embed_train.jsonl \
    --out models/agentrag-rerank-v1

# 4. GATE — Recall@K + MRR, baseline vs candidate. Promote ONLY on a real gain.
uv run python scripts/eval_retrieval.py \
    --baseline BAAI/bge-m3 \
    --candidate models/agentrag-embed-v1 \
    --test data/finetune/embed_test.jsonl
```

## Promote (only if eval_retrieval shows a gain)

- **Embedding:** serve the fine-tuned model via TEI and point `.env` at it:
  ```bash
  make serve-embed        # TEI (openai-compatible) on :8080 serving models/agentrag-embed-v1
  # .env:
  #   EMBEDDING_PROVIDER=openai
  #   EMBEDDING_MODEL=models/agentrag-embed-v1   (or the served name)
  #   EMBEDDING_BASE_URL=http://127.0.0.1:8080/v1/
  ```
  Re-ingest so the index uses the new embeddings.
- **Reranker:** point `RETRIEVAL_RERANK_MODEL=models/agentrag-rerank-v1` (keep
  `RETRIEVAL_RERANK_BACKEND=local_cross_encoder`). The config guard requires a real
  cross-encoder path here — a local model dir qualifies.

## Decision rule
The strategy says **stop after step 2 (reranker) unless a specific LLM failing mode**.
Embedding+reranker is the structural ceiling-raiser; the LLM KTO/ORPO trainer
(`scripts/finetune_dpo.py`) is step 4 — only worth it after ≥500 thumbs ratings and a
named style/format failure.

## Why not here
The 6 GB session GPU is too small to train these safely (e5/bge-m3 LoRA ~6 GB at the
edge). Run on the 16 GB box. Synthetic Q-gen (`mine-pairs`) is CPU/LLM-bound and *can*
run anywhere with a key, but pairing it with the GPU train + eval in one place is simpler.
