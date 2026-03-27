from __future__ import annotations

import argparse
import asyncio
import json
import logging
import socket
import statistics
import sys
import time
from collections import Counter
from pathlib import Path
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.pam.config import settings
from src.pam.config_validation import validate_settings
from src.pam.graph.graphiti_service import GraphitiService
from src.pam.ingestion.chunkers.hybrid_chunker import HybridChunker


class WarningCounterHandler(logging.Handler):
    def __init__(self) -> None:
        super().__init__(level=logging.WARNING)
        self.counter: Counter[str] = Counter()

    def emit(self, record: logging.LogRecord) -> None:
        message = record.getMessage()
        if "Source entity not found in nodes for edge relation" in message:
            self.counter["missing_source_entity"] += 1
        elif "Target entity not found in nodes for edge relation" in message:
            self.counter["missing_target_entity"] += 1
        elif "invalid duplicate_facts idx values" in message:
            self.counter["invalid_duplicate_facts"] += 1
        elif "invalid contradicted_facts idx values" in message:
            self.counter["invalid_contradicted_facts"] += 1
        elif "Could not find source or target node for extracted edge" in message:
            self.counter["unresolved_edge_endpoint"] += 1
        else:
            self.counter["other_warnings"] += 1


def build_graph_chunker() -> HybridChunker:
    return HybridChunker(
        max_tokens=settings.GRAPH_CHUNK_MAX_TOKENS,
        overlap_tokens=settings.GRAPH_CHUNK_OVERLAP_TOKENS,
        tokenizer_model=settings.CHUNK_TOKENIZER_MODEL,
        split_on_headings=True,
        split_on_paragraphs=False,
    )


async def run(path: Path, max_chunks: int | None) -> dict:
    content = path.read_text(encoding="utf-8")
    chunker = build_graph_chunker()
    chunks = chunker.chunk(content, metadata={"document_title": path.name})
    if max_chunks is not None:
        chunks = chunks[:max_chunks]

    progress_events: list[dict] = []
    warning_handler = WarningCounterHandler()
    graphiti_logger = logging.getLogger("graphiti_core")
    root_logger = logging.getLogger()
    graphiti_logger.addHandler(warning_handler)
    root_logger.addHandler(warning_handler)

    def on_progress(payload: dict) -> None:
        progress_events.append(payload)
        print(
            json.dumps(
                {
                    "progress": f"{payload['completed']}/{payload['total']}",
                    "chunk_index": payload["chunk_index"],
                    "cached": payload["cached"],
                    "duration_ms": payload["duration_ms"],
                }
            ),
            flush=True,
        )

    service = GraphitiService()
    await service.build_indices()
    group_id = service.normalize_group_id(path.stem)

    try:
        started = time.perf_counter()
        results = await service.sync_chunks(
            chunks=chunks,
            group_id=group_id,
            document_ref=str(path),
            progress_callback=on_progress,
        )
        total_ms = (time.perf_counter() - started) * 1000
    finally:
        graphiti_logger.removeHandler(warning_handler)
        root_logger.removeHandler(warning_handler)

    durations = [result.get("duration_ms", 0.0) for result in results]
    cached_count = sum(1 for result in results if result.get("cached"))
    entity_counts = [len(result.get("entities", [])) for result in results]
    edge_counts = [len(result.get("relationships", [])) for result in results]
    chunks_without_edges = sum(1 for count in edge_counts if count == 0)
    chunks_without_entities = sum(1 for count in entity_counts if count == 0)
    p95_chunk_ms = (
        round(statistics.quantiles(durations, n=20)[18], 2)
        if len(durations) >= 2
        else (round(durations[0], 2) if durations else 0.0)
    )

    report = {
        "file": str(path),
        "chunks_requested": len(chunks),
        "chunks_processed": len(results),
        "cached_chunks": cached_count,
        "uncached_chunks": len(results) - cached_count,
        "cache_hit_rate": round(cached_count / len(results), 4) if results else 0.0,
        "total_ms": round(total_ms, 2),
        "avg_chunk_ms": round(sum(durations) / len(durations), 2) if durations else 0.0,
        "p95_chunk_ms": p95_chunk_ms,
        "max_chunk_ms": round(max(durations), 2) if durations else 0.0,
        "total_entities": sum(entity_counts),
        "total_edges": sum(edge_counts),
        "avg_entities_per_chunk": round(sum(entity_counts) / len(entity_counts), 2)
        if entity_counts
        else 0.0,
        "avg_edges_per_chunk": round(sum(edge_counts) / len(edge_counts), 2)
        if edge_counts
        else 0.0,
        "chunks_without_entities": chunks_without_entities,
        "chunks_without_edges": chunks_without_edges,
        "graph_max_concurrency": settings.GRAPH_MAX_CONCURRENCY,
        "provider": settings.EXTRACTION_PROVIDER,
        "model": settings.EXTRACTION_MODEL,
        "group_id": group_id,
        "progress_events": len(progress_events),
        "warning_counts": dict(warning_handler.counter),
    }
    return report


def ensure_neo4j_reachable(uri: str) -> None:
    parsed = urlparse(uri)
    host = parsed.hostname or "localhost"
    port = parsed.port or 7687
    try:
        with socket.create_connection((host, port), timeout=2):
            return
    except OSError as exc:
        raise SystemExit(
            f"Neo4j is not reachable at {host}:{port}. "
            f"Start Neo4j or fix NEO4J_URI before running graph benchmark. "
            f"Original error: {exc}"
        ) from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark full graph extraction for a document."
    )
    parser.add_argument("path", help="Path to a markdown document")
    parser.add_argument(
        "--max-chunks",
        type=int,
        default=None,
        help="Only run the first N graph chunks",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    path = Path(args.path)
    if not path.exists():
        raise SystemExit(f"File not found: {path}")
    validate_settings(settings)
    ensure_neo4j_reachable(settings.NEO4J_URI)
    report = asyncio.run(run(path=path, max_chunks=args.max_chunks))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
