from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.pam.config import settings
from src.pam.config_validation import validate_settings
from src.pam.ingestion.chunkers.hybrid_chunker import HybridChunker
from src.pam.ingestion.embedders.factory import build_embedding_provider


def build_search_chunker() -> HybridChunker:
    return HybridChunker(
        max_tokens=settings.SEARCH_CHUNK_MAX_TOKENS,
        overlap_tokens=settings.SEARCH_CHUNK_OVERLAP_TOKENS,
        tokenizer_model=settings.CHUNK_TOKENIZER_MODEL,
        split_on_headings=True,
        split_on_paragraphs=settings.SEARCH_CHUNK_BY_PARAGRAPH,
    )


def build_graph_chunker() -> HybridChunker:
    return HybridChunker(
        max_tokens=settings.GRAPH_CHUNK_MAX_TOKENS,
        overlap_tokens=settings.GRAPH_CHUNK_OVERLAP_TOKENS,
        tokenizer_model=settings.CHUNK_TOKENIZER_MODEL,
        split_on_headings=True,
        split_on_paragraphs=False,
    )


async def run(path: Path, embed: bool) -> dict:
    content = path.read_text(encoding="utf-8")

    search_chunker = build_search_chunker()
    graph_chunker = build_graph_chunker()

    started = time.perf_counter()
    search_chunks = search_chunker.chunk(content, metadata={"document_title": path.name})
    chunk_search_ms = (time.perf_counter() - started) * 1000

    started = time.perf_counter()
    graph_chunks = graph_chunker.chunk(content, metadata={"document_title": path.name})
    chunk_graph_ms = (time.perf_counter() - started) * 1000

    report = {
        "file": str(path),
        "bytes": len(content.encode("utf-8")),
        "search_chunks": len(search_chunks),
        "graph_chunks": len(graph_chunks),
        "chunk_search_ms": round(chunk_search_ms, 2),
        "chunk_graph_ms": round(chunk_graph_ms, 2),
        "embedding_provider": settings.EMBEDDING_PROVIDER,
        "embedding_model": settings.EMBEDDING_MODEL,
        "extraction_provider": settings.EXTRACTION_PROVIDER,
        "extraction_model": settings.EXTRACTION_MODEL,
    }

    if embed:
        embedder = build_embedding_provider(settings)
        texts = [chunk["content"] for chunk in search_chunks]
        started = time.perf_counter()
        embeddings = await embedder.embed(texts)
        embed_ms = (time.perf_counter() - started) * 1000
        report["embedding_count"] = len(embeddings)
        report["embed_ms"] = round(embed_ms, 2)

    if search_chunks:
        report["first_search_section"] = search_chunks[0]["section_path"]
        report["first_search_chars"] = len(search_chunks[0]["content"])
    if graph_chunks:
        report["first_graph_section"] = graph_chunks[0]["section_path"]
        report["first_graph_chars"] = len(graph_chunks[0]["content"])

    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark chunking and optional embeddings for a document."
    )
    parser.add_argument("path", help="Path to a markdown document")
    parser.add_argument(
        "--embed",
        action="store_true",
        help="Also run the configured embedding provider",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    path = Path(args.path)
    if not path.exists():
        raise SystemExit(f"File not found: {path}")
    validate_settings(settings)
    report = asyncio.run(run(path=path, embed=args.embed))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
