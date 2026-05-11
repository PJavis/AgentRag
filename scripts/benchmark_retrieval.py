from __future__ import annotations

import argparse
import asyncio
import json
import socket
import statistics
import sys
import time
from pathlib import Path
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.pam.config import settings
from src.pam.config_validation import validate_settings
from src.pam.retrieval.elasticsearch_retriever import ElasticsearchRetriever


def ensure_elasticsearch_reachable(url: str) -> None:
    parsed = urlparse(url)
    host = parsed.hostname or "localhost"
    port = parsed.port or 9200
    try:
        with socket.create_connection((host, port), timeout=2):
            return
    except OSError as exc:
        raise SystemExit(
            f"Elasticsearch is not reachable at {host}:{port}. "
            f"Start Elasticsearch or fix ELASTICSEARCH_URL before running retrieval benchmark. "
            f"Original error: {exc}"
        ) from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark sparse, dense, and hybrid retrieval."
    )
    parser.add_argument(
        "benchmark_file",
        help="Path to benchmark JSON file",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Top K results to inspect",
    )
    parser.add_argument(
        "--modes",
        nargs="+",
        default=["sparse", "dense", "hybrid", "hybrid_kg"],
        help="Modes to benchmark",
    )
    return parser.parse_args()


def load_cases(path: Path) -> list[dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise SystemExit("Benchmark file must contain a JSON array")
    return payload


def is_relevant(result: dict, case: dict) -> bool:
    expected_title = case.get("expected_document_title")
    expected_section = case.get("expected_section_path")
    expected_hash = case.get("expected_content_hash")
    content_contains = case.get("content_contains")

    if expected_title and result.get("document_title") != expected_title:
        return False
    if expected_section and result.get("section_path") != expected_section:
        return False
    if expected_hash and result.get("content_hash") != expected_hash:
        return False
    if content_contains and content_contains not in (result.get("content") or ""):
        return False
    return True


def first_relevant_rank(results: list[dict], case: dict) -> int | None:
    for item in results:
        if is_relevant(item, case):
            return item["rank"]
    return None


async def benchmark_mode(
    retriever: ElasticsearchRetriever,
    cases: list[dict],
    mode: str,
    top_k: int,
) -> dict:
    latencies: list[float] = []
    ranks: list[int | None] = []
    query_reports: list[dict] = []

    for case in cases:
        query = case["query"]
        document_title = case.get("document_title")

        started = time.perf_counter()
        response = await retriever.search(
            query=query,
            mode=mode,
            top_k=top_k,
            document_title=document_title,
        )
        latency_ms = (time.perf_counter() - started) * 1000
        latencies.append(latency_ms)

        results = response["results"]
        relevant_rank = first_relevant_rank(results, case)
        ranks.append(relevant_rank)

        query_reports.append(
            {
                "query": query,
                "latency_ms": round(latency_ms, 2),
                "first_relevant_rank": relevant_rank,
                "hit": relevant_rank is not None,
                "top_result": results[0] if results else None,
            }
        )

    hit_count = sum(1 for rank in ranks if rank is not None)
    reciprocal_ranks = [1 / rank for rank in ranks if rank is not None]
    summary = {
        "mode": mode,
        "queries": len(cases),
        "hit_rate": round(hit_count / len(cases), 4) if cases else 0.0,
        "mrr": round(sum(reciprocal_ranks) / len(cases), 4) if cases else 0.0,
        "avg_latency_ms": round(sum(latencies) / len(latencies), 2) if latencies else 0.0,
        "p95_latency_ms": (
            round(statistics.quantiles(latencies, n=20)[18], 2)
            if len(latencies) >= 2
            else (round(latencies[0], 2) if latencies else 0.0)
        ),
        "query_reports": query_reports,
    }
    return summary


async def run(benchmark_file: Path, top_k: int, modes: list[str]) -> dict:
    cases = load_cases(benchmark_file)
    retriever = ElasticsearchRetriever()

    invalid_modes = sorted(set(modes) - {"sparse", "dense", "hybrid", "hybrid_kg"})
    if invalid_modes:
        raise SystemExit(f"Invalid modes: {', '.join(invalid_modes)}")

    summaries = []
    for mode in modes:
        summaries.append(await benchmark_mode(retriever, cases, mode, top_k))

    return {
        "benchmark_file": str(benchmark_file),
        "top_k": top_k,
        "modes": modes,
        "embedding_provider": settings.EMBEDDING_PROVIDER,
        "embedding_model": settings.EMBEDDING_MODEL,
        "summaries": summaries,
    }


def main() -> None:
    args = parse_args()
    benchmark_file = Path(args.benchmark_file)
    if not benchmark_file.exists():
        raise SystemExit(f"Benchmark file not found: {benchmark_file}")

    validate_settings(settings)
    ensure_elasticsearch_reachable(settings.ELASTICSEARCH_URL)
    report = asyncio.run(run(benchmark_file, args.top_k, args.modes))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
