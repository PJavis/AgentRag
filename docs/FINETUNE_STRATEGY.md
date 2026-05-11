# AgentRag — Local Finetune Strategy (16 GB VRAM)

Replace cloud Gemini with **self-hosted Ollama + finetuned models** on a single
16 GB GPU box, without losing answer quality. Every step below is wired to
existing `.env` knobs — no code changes required to swap providers, only
config edits.

---

## How `.env` controls everything

`src/agentrag/config.py` parses `.env` on every process start. The four
provider/model pairs that matter:

| Pair | What it does | Local finetune target |
|---|---|---|
| `EMBEDDING_PROVIDER` + `EMBEDDING_MODEL` + `EMBEDDING_BASE_URL` | Vectorising queries + chunks | TEI container (OpenAI-compat) serving your finetuned bge/e5 |
| `RETRIEVAL_RERANK_BACKEND` + `RETRIEVAL_RERANK_MODEL` | Reranks top-K | `local_cross_encoder` loads your finetuned bge-reranker in-process |
| `EXTRACTION_PROVIDER` + `EXTRACTION_MODEL` | StructMem extract + SQL compile + JSON tasks | Ollama serving your LoRA-finetuned Qwen GGUF |
| `AGENT_PROVIDER` + `AGENT_MODEL` | Answer synthesis | Ollama serving Qwen 14B (or your LoRA-merged variant) |

Plus `LLM_TASK_MODEL_MAP` for per-task routing (`classify`, `decide`,
`answer`, etc.) and `RETRIEVAL_RERANK_PROVIDER` if you want rerank via
Ollama instead of cross-encoder.

**End-state .env (after finetune complete):**

```env
# All local. No cloud keys needed.
OLLAMA_BASE_URL=http://127.0.0.1:11434/v1/
OLLAMA_API_KEY=ollama

# Embedding via TEI (openai-compatible) — exposes finetuned model on :8080
EMBEDDING_PROVIDER=openai
EMBEDDING_MODEL=agentrag-embed-v1
EMBEDDING_BASE_URL=http://127.0.0.1:8080/v1/
OPENAI_API_KEY=tei-dummy

# LLM via Ollama — point at finetuned merged GGUF
EXTRACTION_PROVIDER=ollama
EXTRACTION_MODEL=qwen-agentrag                # was qwen2.5:7b-instruct
AGENT_PROVIDER=ollama
AGENT_MODEL=qwen2.5:14b-instruct              # or qwen-agentrag-14b if you LoRA-ed it

# Reranker via local cross-encoder (finetuned weights on disk)
RETRIEVAL_RERANK_ENABLED=true
RETRIEVAL_RERANK_BACKEND=local_cross_encoder
RETRIEVAL_RERANK_MODEL=./models/agentrag-rerank-v1

# Vision still useful for scanned PDFs
VISION_PROVIDER=ollama
VISION_MODEL=llava:13b
VISION_BASE_URL=http://127.0.0.1:11434/v1/

# Routing: cheap classify/decide on 3B, heavy synthesize on 14B
LLM_ROUTING_ENABLED=true
LLM_TASK_MODEL_MAP={"classify":"llama3.2:3b","decide":"llama3.2:3b","schema_discovery":"qwen-agentrag","sql_compile":"qwen-agentrag","synthesize":"qwen2.5:14b-instruct","answer":"qwen2.5:14b-instruct"}

LLM_COST_TRACKING_ENABLED=true                # local cost = 0 USD, latency still tracked
```

That's the destination. Below is how to get there.

---

## VRAM Budget Plan (16 GB)

Peak coexistence during normal operation:

| Component | VRAM | Notes |
|---|---|---|
| `qwen2.5:14b-instruct` Q4_K_M | ~9 GB | Loaded on first request, stays |
| `agentrag-embed-v1` via TEI | ~1.5 GB | Resident |
| `bge-reranker-v2-m3` cross-encoder | ~0.5 GB | In API process |
| `llama3.2:3b` (classify/decide) | ~2.5 GB | Ollama swaps in/out |
| Reserve for context + KV cache | ~2.5 GB | |
| **Peak total** | **~16 GB** | Tight but works |

Vision (llava:13b ~4 GB) **only loaded during ingest**, so it can co-exist
with 14B during ingest by swapping out the 3B classifier. Ollama handles
the swap automatically.

Training peaks (not concurrent with serving — schedule overnight):

| Job | VRAM | Time on 16 GB |
|---|---|---|
| Embedding LoRA / full FT | ~6 GB | 20–40 min / 5k pairs |
| Reranker FT | ~5 GB | 30–60 min / 10k pairs |
| Qwen 7B QLoRA (unsloth) | ~12 GB | 1–2 hr / 1k samples |
| Qwen 14B QLoRA (unsloth, r=8) | ~15 GB | 3–5 hr / 1k samples (tight) |

---

## ROI Order

| Layer | Effort | Win | Status after each step |
|---|---|---|---|
| **1. Embedding** | 1 day | +10–20% recall@10 | Often "good enough" — stop here |
| **2. Reranker** | 1 day | +5–10% MRR | Solid precision boost |
| **3. LLM 7B QLoRA** | 2–3 days | Style + JSON consistency | Optional |
| **4. DPO** | 2–3 days | +3–5% preference | Needs ≥500 thumbs ratings |
| **5. Vision** | 1 week | Marginal | Almost never worth it |

> Stop at step 2 unless you have a specific failing mode (e.g. agent
> answers in wrong format, ignores citations).

---

## Data Pipeline

### Sources (free)

1. **Thumbs feedback** (`adapter_chat_feedback`) — `rating=+1` rows → positives, `-1` → negatives or DPO rejected.
2. **Tool traces** in `chat_messages.tool_trace` — every assistant message captures the chunks the agent actually used. Join with feedback for ground truth.
3. **Synthetic Q-gen** — feed each chunk to a local LLM ("Write 3 questions answerable by this passage"). Use Qwen-32B-Q4_K_M on the 16 GB box (~14 GB peak — works alone), or your existing Gemini quota.
4. **Hard negatives** — for every `(q, pos)`, run the current embedding, take top-20 hits, drop the positive, keep 4 plausible-but-wrong.

### Target sizes

| Layer | Min | Sweet spot |
|---|---|---|
| Embedding triplets | 500 | 5k |
| Reranker pairs | 1k | 10k |
| LLM SFT | 200 | 1k |
| DPO preferences | 200 | 1k |

---

## Helper Scripts (all live in `scripts/`)

| Script | Purpose |
|---|---|
| `mine_finetune_pairs.py` | DB → triplet JSONL (feedback + synthetic + hard negs) |
| `finetune_embedding.py` | sentence-transformers training, saves to `models/agentrag-embed-v1/` |
| `finetune_reranker.py` | Cross-encoder training, saves to `models/agentrag-rerank-v1/` |
| `finetune_qwen_lora.py` | Unsloth QLoRA on Qwen2.5-7B, merges to fp16 |
| `convert_to_ollama.sh` | GGUF quantize + `ollama create` |
| `eval_retrieval.py` | Recall@K + MRR before/after, gate the promotion |

Makefile targets glue them together. Below.

---

## Roadmap — week by week (16 GB box)

### Week 1 — Data

```bash
# 1. Mine real-user pairs + synth + hard negs
make mine-pairs

# 2. Spot-check first 50 rows
head -50 data/finetune/embed_triplets.jsonl | jq .

# 3. Split 90/10 train/test
make split-pairs
```

### Week 2 — Embedding + Reranker

```bash
# Train + eval embedding
make train-embed
make eval-embed                    # gate ≥5pt recall@10 lift
make serve-embed                   # spins up TEI on :8080

# Train + eval reranker
make train-rerank
make eval-rerank                   # gate ≥3pt MRR lift

# Update .env to point at new models, restart API
```

### Week 3 (optional) — LLM LoRA

```bash
# 1. Mine 1k SFT samples from Gemini-era chat logs
make mine-sft

# 2. QLoRA on Qwen2.5-7B
make train-llm-lora

# 3. Convert + register with Ollama
make convert-llm

# 4. A/B against baseline qwen2.5:7b-instruct
make eval-llm
```

### Week 4 — Production loop

```bash
# Cron @ 02:00 nightly
0 2 * * *  cd /home/agentrag && make retrain-embedding-nightly
```

---

## Promotion Gates

Never blind-swap. Each finetune passes `scripts/eval_retrieval.py` before
the `.env` flip:

```
$ make eval-embed
                  baseline    candidate     delta
recall@5            0.62         0.74      +0.12
recall@10           0.74         0.85      +0.11
mrr@10              0.51         0.61      +0.10
PROMOTE: yes (passes all gates)
```

Gate thresholds in `scripts/eval_retrieval.py`:
- `recall@10` ≥ +5 pts
- `mrr@10` ≥ +3 pts
- No regression on any metric > 2 pts

---

## Common Traps

- **Don't full-SFT.** LoRA gives 90% of the win at 5% of the cost.
- **Don't train on auto-generated data without spot-check.** Synthetic Q-gen produces ~10–20% noise — sample 50 by hand before training.
- **Don't deploy without A/B.** Finetune can regress.
- **Don't run two heavy LLMs (14B + 13B) simultaneously.** Stagger via Ollama `keep_alive=0` after vision job.
- **Don't put models in git.** Use `.gitignore`'d `models/` volume.
- **Don't skip the eval script.** Quantitative gate is what makes this loop honest.
