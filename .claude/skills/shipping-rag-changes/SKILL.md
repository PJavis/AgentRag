---
name: shipping-rag-changes
description: Use when changing AgentRag ingestion, parsing, OCR/vision, chunking, retrieval, reranking, relevance thresholds, agent routing, config settings, or eval scripts — and when reporting any result from an ingest, re-ingest, eval, or A/B. Triggers on "re-ingest", "change the parser", "new setting", "tune the floor", "run the eval", "compare arms", "it says success", "ingested 0", "why did pages disappear". A checklist of silent-success traps that each cost real time in this repo.
---

# Shipping RAG changes

## Overview

The expensive bugs in this repo did not crash. Each one **reported success while
doing nothing or the wrong thing**, and was caught only by diffing against a backup
or by a human reading answers. The module changes every time; the classes don't.

**Core principle:** a green status line is a claim, not evidence. Before trusting
one, name the artefact that would differ if it were lying, and look at it.

## The six classes

| Class | How it lied | Rule |
|---|---|---|
| **200-with-error** | `/chat` returns `{"error": ...}` with HTTP 200; a baseline run logged 19 rejected calls as "19 answered, 0 failed" | Validate the payload (`error` key, non-empty `answer`, `citations`), never just the status. `scripts/ops/stack_monitor.py --chat` does this. |
| **Wrong model, no error** | `VISION_BASE_URL` pointed at Ollama → scanned pages described by a TEXT model → ~40 chars → filtered → pages vanished. API model name as cross-encoder → load fails → swallowed → rerank inert. Compose `tei` serves bge-m3 where the index is 768-dim e5-FT. | Put the guard in `config_validation.py` (fail at startup), with a test, and mutation-test it (disable the guard, confirm the test fails). |
| **Stale cache = "skipped"** | Re-ingest reported `skipped` for every document because the cache key was file bytes; a code-only parser fix still doesn't invalidate it. | Any new parse-affecting setting must enter the cache key (PR #18 pattern). After a parser CODE change, delete the documents (`DELETE /documents/{id}`) and re-ingest. |
| **Lying counters** | Repair runs reported `"ingested": 0` while doing hours of work; segment counts move with chunking config. | Diff the corpus against a backup **by characters**, per document — not by run report, not by segment count. |
| **Scale drift** | rerank_score moved from sigmoid(prob) (0.5–0.731) to a probability; an old floor value would keep its number with a new meaning. A floor of 0.3 was inert for months. | When a score changes meaning, rename the setting and make the old name fail with the converted value (`_validate_relevance_scale_settings`). Check a threshold is inside the score's real range before A/B-ing it. |
| **Irreproducible eval** | `AGENT_TEMPERATURE=0.3` gave 15/19 vs 13/19 for identical code; chunker page labels silently deleted gold chunks and produced a significant-but-spurious result. | Pin temperature 0 and sample. Never filter gold by the chunker's page label. State the power needed before quoting a number (62 table-citing answers per arm for 0.30→0.10). |

## Before you start

- [ ] `.venv/bin/python scripts/ops/stack_monitor.py` is OK (see the `agentrag-stack-monitor` skill for triage). Rerank on CPU makes every timing meaningless.
- [ ] Re-ingest planned? Check the DeepSeek balance in that output: one pass is ~4.6k contextualize calls. Run ingest **in-process on the host** (has tesseract; gunicorn kills HTTP ingest at 120 s).
- [ ] Back up what you are about to rewrite (corpus text per document) so you can diff afterwards.

## While changing code

- [ ] New setting that changes parsing → in the ingest cache key.
- [ ] New setting that can silently pick the wrong model/endpoint → rule in `config_validation.py` + test + mutation check.
- [ ] Editing `src/` with the stack running → `docker compose up -d api worker` (bind-mounted `./src` gives partial reloads; plain `restart` does not re-read `env_file`).
- [ ] New tests go outside `tests/ingestion` and `tests/ontology` — CI ignores those two.

## Before you report a result

- [ ] Every count names its stage ("35 → 19 at classification", not "19") and every moved item is attributed, by ablation, not by reasoning.
- [ ] "Pre-existing failure" is proven on the base commit in a worktree (`git worktree add --detach <tmp> master`, run with `.venv/bin/python` + `PYTHONPATH=<tmp>`), not inferred.
- [ ] Ingest result checked by character diff, not by the run report.
- [ ] Eval result: temperature pinned, sample size vs the stated power, and arms compared on the same corpus snapshot.
