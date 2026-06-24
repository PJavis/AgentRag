"""Ablation benchmark runner — subprocess-per-config matrix over run_benchmark.py.

This sweeps the RAG workstream flags (Contextual Retrieval, RAPTOR, CRAG,
Adaptive Routing, Semantic Cache, Agent Multi-hop) by running the full
9-metric benchmark (scripts/eval/run_benchmark.py) once per configuration, then
emits a markdown comparison table across the judged metrics + latency/cost/failure.

Why subprocess-per-config:
  The config is a pydantic BaseSettings *singleton* read once at import time
  (env overrides .env). Toggling a flag therefore only takes effect in a fresh
  process that sets the flag in its environment BEFORE importing settings. Each
  ablation is run as a separate `subprocess.run` with a tailored env.

Why re-ingest each config:
  We do NOT pass --skip-ingest. Contextual Retrieval / RAPTOR change *how the
  corpus is indexed*, so the index must be rebuilt under each config's ingest
  flags for the comparison to be honest. STRUCTMEM_INGEST_MODE is forced to
  "sync" so StructMem/RAPTOR/CR extraction fully completes before scoring
  (matching the KG benchmark methodology).

This is a LONG, LIVE-STACK, ONE-TIME run — each config re-ingests and runs the
agent over N×(suites) questions plus an LLM judge, so the full matrix takes
hours. It needs the live stack: Postgres + Elasticsearch + a DeepSeek API key
(judge + LLM) + Ollama (embeddings). It is NOT run in CI.

Usage (run by a human, on the live stack):
  uv run python scripts/eval/run_ablation.py --suite both --n 20
  uv run python scripts/eval/run_ablation.py --only baseline,full --n 10
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]

# Config-name → env-override dict. Each override turns workstream flags "true"
# in the child process (all default off). "baseline" runs the plain system.
ABLATIONS: dict[str, dict[str, str]] = {
    "baseline": {},
    "cr": {"CONTEXTUAL_RETRIEVAL_ENABLED": "true"},
    # Isolated single-WS rows (P1 Task 6) — clean attribution per feature.
    "crag_only": {"CRAG_ENABLED": "true"},
    "fastpath_only": {"ADAPTIVE_ROUTING_ENABLED": "true"},
    "semcache_only": {"SEMANTIC_CACHE_ENABLED": "true"},
    "multihop_only": {"AGENT_MULTIHOP_ENABLED": "true"},
    "cr_raptor": {"CONTEXTUAL_RETRIEVAL_ENABLED": "true", "RAPTOR_ENABLED": "true"},
    "cr_raptor_crag": {
        "CONTEXTUAL_RETRIEVAL_ENABLED": "true",
        "RAPTOR_ENABLED": "true",
        "CRAG_ENABLED": "true",
    },
    "full": {
        "CONTEXTUAL_RETRIEVAL_ENABLED": "true",
        "RAPTOR_ENABLED": "true",
        "CRAG_ENABLED": "true",
        "SEMANTIC_CACHE_ENABLED": "true",
        "AGENT_MULTIHOP_ENABLED": "true",
    },
}

# Judged metric keys (exact, from src/agentrag/eval/deepeval_metrics.py
# METRIC_TARGETS) plus the computed report metrics we tabulate.
JUDGED_METRICS = (
    "contextual_recall",
    "contextual_precision",
    "faithfulness",
    "answer_correctness",
    "citation_accuracy",
)


# Every workstream flag the ablation toggles. Forced OFF for every config
# before applying the row's overrides so "baseline" is a TRUE all-off baseline
# and each row isolates exactly its flag(s) — independent of whatever the
# operator's .env has enabled. Without this, a .env with CR/RAPTOR=true (the
# common case) leaks into every child via pydantic's .env read and silently
# turns "baseline" into CR+RAPTOR, destroying per-WS attribution.
WS_FLAGS: tuple[str, ...] = (
    "CONTEXTUAL_RETRIEVAL_ENABLED",
    "RAPTOR_ENABLED",
    "CRAG_ENABLED",
    "AGENT_MULTIHOP_ENABLED",
    "ADAPTIVE_ROUTING_ENABLED",
    "SEMANTIC_CACHE_ENABLED",
)


def build_env(base: dict, overrides: dict) -> dict:
    """New env = copy(base) + all WS flags forced OFF + overrides + sync ingest.

    Never mutates ``base``. All WS_FLAGS are explicitly set "false" first so the
    child process can't inherit a flag from the operator's .env (env vars take
    precedence over .env in pydantic settings) — baseline is genuinely all-off
    and each row turns on only the flag(s) in ``overrides``.

    STRUCTMEM_INGEST_MODE is forced to "sync" so the index is fully built
    (StructMem/RAPTOR/CR extraction done) before scoring. UPLOAD_DEDUPE_BY_HASH
    is forced off so each config's re-ingest actually rebuilds the index with
    that config's CR/RAPTOR fields — otherwise an already-present gold corpus
    dedupes the re-ingest to a no-op and CR/RAPTOR silently never apply (the
    index stays flat == baseline).
    """
    env = dict(base)
    for flag in WS_FLAGS:
        env[flag] = "false"
    env.update(overrides)
    env["STRUCTMEM_INGEST_MODE"] = "sync"
    env["UPLOAD_DEDUPE_BY_HASH"] = "false"
    return env


def wipe_corpus_indices() -> None:
    """Delete the corpus ES indices so each config rebuilds clean (no leftover
    chunks from a previous config / prior run mixing into retrieval). Best
    effort — a missing index is fine."""
    import urllib.error
    import urllib.request

    es = os.environ.get("ELASTICSEARCH_URL", "http://localhost:9200").rstrip("/")
    for idx in (
        os.environ.get("ELASTICSEARCH_INDEX_NAME", "agentrag_segments"),
        os.environ.get("STRUCTMEM_INDEX", "agentrag_memory_doc"),
        os.environ.get("ELASTICSEARCH_ENTITY_INDEX_NAME", "agentrag_entities"),
        os.environ.get("ELASTICSEARCH_RELATIONSHIP_INDEX_NAME", "agentrag_relationships"),
    ):
        try:
            req = urllib.request.Request(f"{es}/{idx}", method="DELETE")
            urllib.request.urlopen(req, timeout=30).read()
            print(f"[ABLATION] wiped index {idx}")
        except urllib.error.HTTPError as e:
            if e.code != 404:
                print(f"[ABLATION] wipe {idx} HTTP {e.code}")
        except Exception as exc:  # noqa: BLE001
            print(f"[ABLATION] wipe {idx} failed: {type(exc).__name__}: {exc}")


def wipe_corpus_db() -> None:
    """Delete Postgres documents + segments so re-ingest is truly fresh.

    CRITICAL: wiping ES alone is not enough. `save_document_and_segments`
    dedupes against Postgres (source_id, content_hash); a prior run's docs
    (graph_status='done') would make the re-ingest return 'skipped' and
    `ingest_folder` then skips `index_segments` — leaving ES empty (recall 0).
    Clearing the PG corpus forces a real re-index under each config's flags.
    """
    import asyncio

    async def _wipe() -> None:
        import sys
        if str(ROOT) not in sys.path:
            sys.path.insert(0, str(ROOT))  # parent script lacks repo root on path
        from sqlalchemy import delete
        from src.agentrag.database import AsyncSessionLocal
        from src.agentrag.database.models import Document, Segment

        async with AsyncSessionLocal() as session:
            await session.execute(delete(Segment))
            await session.execute(delete(Document))
            await session.commit()

    try:
        asyncio.run(_wipe())
        print("[ABLATION] wiped Postgres documents + segments")
    except Exception as exc:  # noqa: BLE001
        print(f"[ABLATION] wipe PG failed: {type(exc).__name__}: {exc}")


def build_cmd(
    suite: str, n: int, judge_provider: str, out_path: str, skip_ingest: bool = False
) -> list[str]:
    """argv to run the benchmark as a child process.

    skip_ingest=True reuses the index already built by the first config of the
    same index-shape group (only CR/RAPTOR change the index; query-time flags
    like CRAG/MULTIHOP/ADAPTIVE_ROUTING/SEMANTIC_CACHE don't), so siblings run
    against it without a costly re-ingest + graph-extraction.
    """
    cmd = [
        sys.executable,
        str(HERE / "run_benchmark.py"),
        "--suite",
        suite,
        "--n",
        str(n),
        "--judge-provider",
        judge_provider,
        "--out",
        out_path,
    ]
    if skip_ingest:
        cmd.append("--skip-ingest")
    return cmd


def index_shape(env: dict) -> tuple[str, str]:
    """The (CONTEXTUAL_RETRIEVAL, RAPTOR) pair that determines the ingested
    index. Two configs sharing this pair have an identical index and can share
    a single ingest — the remaining WS flags only affect query time."""
    return (
        str(env.get("CONTEXTUAL_RETRIEVAL_ENABLED", "false")).lower(),
        str(env.get("RAPTOR_ENABLED", "false")).lower(),
    )


def _fmt(value) -> str:
    """Format a metric cell; None / missing → 'n/a'."""
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def _row_from_report(name: str, report: dict) -> list[str]:
    """Build one markdown table row from a benchmark report dict."""
    metrics = report.get("metrics", {})
    cells = [name]
    for key in JUDGED_METRICS:
        m = metrics.get(key) or {}
        cells.append(_fmt(m.get("mean")))
    latency = metrics.get("latency_ms") or {}
    cells.append(_fmt(latency.get("p50")))
    cells.append(_fmt(metrics.get("cost_per_query_usd")))
    fr = metrics.get("failure_rate") or {}
    cells.append(_fmt(fr.get("value")))
    return cells


def _build_markdown(date: str, suite: str, n: int, judge_provider: str, rows: list[list[str]]) -> str:
    header = ["config", *JUDGED_METRICS, "latency_p50_ms", "cost_per_query_usd", "failure_rate"]
    sep = ["---"] * len(header)
    lines = [
        f"# RAG ablation benchmark — {date}",
        "",
        f"- suite: `{suite}`  ·  n per dataset: `{n}`  ·  judge: `{judge_provider}`",
        f"- generated: {datetime.datetime.now().isoformat(timespec='seconds')}",
        "",
        "Each config ran in a **separate process** with all workstream flags forced to a",
        "known baseline (off) then its own flag(s) set in the child env before `settings`",
        "import. Configs are **grouped by index-shape** `(CR, RAPTOR)`: the first member of",
        "each shape re-ingests (`STRUCTMEM_INGEST_MODE=sync`); query-time-only siblings",
        "(CRAG/MULTIHOP/ADAPTIVE_ROUTING/SEMANTIC_CACHE) reuse that index via `--skip-ingest`.",
        "",
        "## Acceptance gates",
        "",
        "- `answer_correctness` baseline > 0.792 → target ≥ 0.85",
        "- `contextual_precision` baseline > 0.819 → target ≥ 0.88",
        "- `faithfulness` ≥ 0.80",
        "- `latency_p50` < 10000 ms",
        "",
        "Values below are metric **means** (judged 0–1); latency in ms, cost in USD.",
        "",
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(sep) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    p = argparse.ArgumentParser(description="Ablation benchmark matrix (subprocess per config)")
    p.add_argument("--suite", default="both", choices=["vn", "en", "both"])
    p.add_argument("--n", type=int, default=20, help="examples per dataset")
    p.add_argument("--judge-provider", default="deepseek", choices=["gemini", "deepseek", "openai"])
    p.add_argument("--date", default=None, help="report date stamp (default: today)")
    p.add_argument("--only", default=None, help="comma list of ABLATIONS configs to run (subset)")
    args = p.parse_args()

    date = args.date or datetime.date.today().isoformat()

    if args.only:
        selected = [c.strip() for c in args.only.split(",") if c.strip()]
        unknown = [c for c in selected if c not in ABLATIONS]
        if unknown:
            raise SystemExit(f"unknown ablation config(s): {unknown}. known: {list(ABLATIONS)}")
    else:
        selected = list(ABLATIONS)

    eval_dir = ROOT / "data" / "eval"
    eval_dir.mkdir(parents=True, exist_ok=True)

    # Group configs by index-shape so each distinct (CR, RAPTOR) index is
    # ingested ONCE; siblings reuse it via --skip-ingest. The first member of a
    # group wipes + re-ingests; the rest skip ingest. This cuts the dominant
    # cost (per-config re-ingest + StructMem graph-extraction) from one-per-row
    # to one-per-index-shape (e.g. 8 rows → 3 ingests). Output order still
    # follows `selected`.
    groups: dict[tuple[str, str], list[str]] = {}
    group_order: list[tuple[str, str]] = []
    for name in selected:
        shape = index_shape(build_env(os.environ, ABLATIONS[name]))
        if shape not in groups:
            groups[shape] = []
            group_order.append(shape)
        groups[shape].append(name)

    rows_by_name: dict[str, list[str]] = {}
    for shape in group_order:
        members = groups[shape]
        print("#" * 70)
        print(f"[ABLATION] index-shape CR={shape[0]} RAPTOR={shape[1]}  members={members}")
        print(f"[ABLATION]   → ingest once for '{members[0]}', --skip-ingest for {members[1:] or '[]'}")
        print("#" * 70)
        for i, name in enumerate(members):
            overrides = ABLATIONS[name]
            out_path = eval_dir / f"_ablation_{name}.json"
            env = build_env(os.environ, overrides)
            first = i == 0
            cmd = build_cmd(
                args.suite, args.n, args.judge_provider, str(out_path), skip_ingest=not first
            )

            print("=" * 70)
            mode = "INGEST" if first else "skip-ingest (reuse index)"
            print(f"[ABLATION] {name}  overrides={overrides or '{}'}  [{mode}]")
            print(f"[ABLATION] {' '.join(cmd)}")
            print("=" * 70)

            # Only the first member of an index-shape group rebuilds the corpus.
            # Wiping BOTH ES indices AND the Postgres corpus is required: the PG
            # (source_id, content_hash) dedupe in save_document_and_segments
            # would otherwise return 'skipped' and the ES re-index never runs.
            # Siblings must NOT wipe — they reuse the index this ingest built.
            if first:
                wipe_corpus_indices()
                wipe_corpus_db()

            try:
                subprocess.run(cmd, env=env, cwd=str(ROOT), check=False)
            except Exception as exc:  # subprocess couldn't even launch
                print(f"[ABLATION] {name} subprocess error: {type(exc).__name__}: {exc}")

            # Be defensive: a run may have failed (no/partial report). Record the
            # row as "(run failed)" and keep going so one bad config doesn't kill
            # the whole matrix.
            if not out_path.exists():
                print(f"[ABLATION] {name}: report missing → recording '(run failed)'")
                rows_by_name[name] = [name] + ["(run failed)"] * (len(JUDGED_METRICS) + 3)
                continue
            try:
                report = json.loads(out_path.read_text(encoding="utf-8"))
            except Exception as exc:
                print(f"[ABLATION] {name}: report unreadable ({type(exc).__name__}) → '(run failed)'")
                rows_by_name[name] = [name] + ["(run failed)"] * (len(JUDGED_METRICS) + 3)
                continue
            rows_by_name[name] = _row_from_report(name, report)

    rows = [rows_by_name[name] for name in selected if name in rows_by_name]
    md = _build_markdown(date, args.suite, args.n, args.judge_provider, rows)
    docs_dir = ROOT / "docs" / "eval"
    docs_dir.mkdir(parents=True, exist_ok=True)
    md_path = docs_dir / f"benchmark_ablation_{date}.md"
    md_path.write_text(md, encoding="utf-8")
    print(f"\n[ABLATION] comparison table → {md_path}")


if __name__ == "__main__":
    main()
