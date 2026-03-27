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

from src.pam.agent.service import AgentService
from src.pam.config import settings
from src.pam.config_validation import validate_settings


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
            f"Start Elasticsearch or fix ELASTICSEARCH_URL before running agent benchmark. "
            f"Original error: {exc}"
        ) from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark local agent latency and quality hints.")
    parser.add_argument("benchmark_file", help="Path to benchmark JSON file")
    parser.add_argument("--repeat", type=int, default=1, help="Repeat each case N times")
    return parser.parse_args()


def load_cases(path: Path) -> list[dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise SystemExit("Benchmark file must contain a JSON array")
    return payload


def _match_item(item: dict, case: dict) -> bool:
    expected_title = case.get("expected_document_title")
    expected_section = case.get("expected_section_path")
    expected_hash = case.get("expected_content_hash")

    if expected_title and item.get("document_title") != expected_title:
        return False
    if expected_section and item.get("section_path") != expected_section:
        return False
    if expected_hash and item.get("content_hash") != expected_hash:
        return False
    return True


def evaluate_response(case: dict, response: dict) -> dict:
    context = response.get("context") or []
    citations = response.get("citations") or []
    answer = (response.get("answer") or "").strip()
    answer_lower = answer.lower()

    expected_answer_contains = case.get("expected_answer_contains")
    if isinstance(expected_answer_contains, str):
        expected_answer_contains = [expected_answer_contains]
    if not isinstance(expected_answer_contains, list):
        expected_answer_contains = []

    answer_hit = True
    if expected_answer_contains:
        answer_hit = all(str(token).lower() in answer_lower for token in expected_answer_contains)

    context_hit = any(_match_item(item, case) for item in context)
    citation_hit = any(_match_item(item, case) for item in citations)
    insufficient = ("insufficient" in answer_lower) or ("không đủ" in answer_lower)
    return {
        "answer_hit": answer_hit,
        "context_hit": context_hit,
        "citation_hit": citation_hit,
        "insufficient": insufficient,
    }


async def run_case(agent: AgentService, case: dict) -> dict:
    question = case["question"]
    document_title = case.get("document_title")

    started = time.perf_counter()
    response = await agent.chat(question=question, document_title=document_title)
    latency_ms = (time.perf_counter() - started) * 1000

    tool_trace = response.get("tool_trace") or []
    eval_report = evaluate_response(case, response)
    return {
        "question": question,
        "document_title": document_title,
        "latency_ms": round(latency_ms, 2),
        "tool_steps": len(tool_trace),
        "tools": [step.get("tool_name") for step in tool_trace],
        "context_count": len(response.get("context") or []),
        "citation_count": len(response.get("citations") or []),
        "answer_chars": len(response.get("answer") or ""),
        **eval_report,
    }


async def run(benchmark_file: Path, repeat: int) -> dict:
    cases = load_cases(benchmark_file)
    agent = AgentService()
    reports: list[dict] = []

    for _ in range(repeat):
        for case in cases:
            reports.append(await run_case(agent, case))

    latencies = [item["latency_ms"] for item in reports]
    p95 = (
        round(statistics.quantiles(latencies, n=20)[18], 2)
        if len(latencies) >= 2
        else (round(latencies[0], 2) if latencies else 0.0)
    )

    def ratio(key: str) -> float:
        if not reports:
            return 0.0
        return round(sum(1 for item in reports if item.get(key)) / len(reports), 4)

    return {
        "benchmark_file": str(benchmark_file),
        "repeat": repeat,
        "cases": len(cases),
        "runs": len(reports),
        "agent_provider": settings.AGENT_PROVIDER or settings.EXTRACTION_PROVIDER,
        "agent_model": settings.AGENT_MODEL or settings.EXTRACTION_MODEL,
        "avg_latency_ms": round(sum(latencies) / len(latencies), 2) if latencies else 0.0,
        "p95_latency_ms": p95,
        "avg_tool_steps": round(sum(item["tool_steps"] for item in reports) / len(reports), 2)
        if reports
        else 0.0,
        "avg_context_count": round(sum(item["context_count"] for item in reports) / len(reports), 2)
        if reports
        else 0.0,
        "avg_citation_count": round(sum(item["citation_count"] for item in reports) / len(reports), 2)
        if reports
        else 0.0,
        "answer_hit_rate": ratio("answer_hit"),
        "context_hit_rate": ratio("context_hit"),
        "citation_hit_rate": ratio("citation_hit"),
        "insufficient_rate": ratio("insufficient"),
        "query_reports": reports,
    }


def main() -> None:
    args = parse_args()
    benchmark_file = Path(args.benchmark_file)
    if not benchmark_file.exists():
        raise SystemExit(f"Benchmark file not found: {benchmark_file}")
    if args.repeat <= 0:
        raise SystemExit("--repeat must be > 0")

    validate_settings(settings)
    ensure_elasticsearch_reachable(settings.ELASTICSEARCH_URL)
    report = asyncio.run(run(benchmark_file, args.repeat))
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
