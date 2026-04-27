"""
Chunking quality evaluation — phân tích chất lượng chunks trong ES index.

Metrics:
  short_chunk_rate    % chunks < min_chars (orphan headers, empty sections)
  avg_content_length  trung bình độ dài nội dung (chars)
  p50/p95_length      percentile distribution
  section_coverage    số section_path unique / tổng chunks
  dedup_rate          % chunks có content_hash trùng (duplicates)
"""
from __future__ import annotations

import statistics
from dataclasses import dataclass


@dataclass
class ChunkingReport:
    document_title: str | None
    total_chunks: int
    short_chunk_count: int       # < min_chars
    short_chunk_rate: float      # 0-1
    avg_length: float
    median_length: float
    p95_length: float
    min_length: int
    max_length: int
    unique_sections: int
    duplicate_chunks: int
    dedup_rate: float            # 0-1
    min_chars_threshold: int

    def as_dict(self) -> dict:
        return {
            "document_title": self.document_title,
            "total_chunks": self.total_chunks,
            "short_chunk_rate": round(self.short_chunk_rate, 4),
            "short_chunk_count": self.short_chunk_count,
            "avg_length_chars": round(self.avg_length, 1),
            "median_length_chars": round(self.median_length, 1),
            "p95_length_chars": round(self.p95_length, 1),
            "min_length_chars": self.min_length,
            "max_length_chars": self.max_length,
            "unique_sections": self.unique_sections,
            "duplicate_chunks": self.duplicate_chunks,
            "dedup_rate": round(self.dedup_rate, 4),
            "min_chars_threshold": self.min_chars_threshold,
        }

    def summary_line(self) -> str:
        return (
            f"chunks={self.total_chunks} | "
            f"short={self.short_chunk_count}({self.short_chunk_rate:.1%}) | "
            f"avg={self.avg_length:.0f}c | "
            f"p95={self.p95_length:.0f}c | "
            f"dupes={self.duplicate_chunks}({self.dedup_rate:.1%})"
        )


async def evaluate_chunking(
    es_client,
    index_name: str,
    document_title: str | None = None,
    min_chars: int = 80,
    max_chunks: int = 5000,
) -> ChunkingReport:
    """Kéo chunks từ ES và tính metrics."""
    query: dict = {"query": {"match_all": {}}, "size": max_chunks, "_source": ["content", "section_path", "content_hash", "document_title"]}
    if document_title:
        query["query"] = {"term": {"document_title.keyword": document_title}}

    resp = await es_client.search(index=index_name, body=query)
    hits = resp["hits"]["hits"]

    if not hits:
        return ChunkingReport(
            document_title=document_title, total_chunks=0,
            short_chunk_count=0, short_chunk_rate=0.0,
            avg_length=0.0, median_length=0.0, p95_length=0.0,
            min_length=0, max_length=0, unique_sections=0,
            duplicate_chunks=0, dedup_rate=0.0,
            min_chars_threshold=min_chars,
        )

    lengths: list[int] = []
    sections: set[str] = set()
    hashes: list[str] = []

    for hit in hits:
        src = hit["_source"]
        content = src.get("content") or ""
        lengths.append(len(content))
        sections.add(src.get("section_path") or "")
        h = src.get("content_hash")
        if h:
            hashes.append(h)

    short_count = sum(1 for l in lengths if l < min_chars)
    dup_count = len(hashes) - len(set(hashes))

    sorted_lengths = sorted(lengths)
    n = len(lengths)
    p95_idx = max(0, int(n * 0.95) - 1)

    return ChunkingReport(
        document_title=document_title,
        total_chunks=n,
        short_chunk_count=short_count,
        short_chunk_rate=short_count / n if n else 0.0,
        avg_length=statistics.mean(lengths) if lengths else 0.0,
        median_length=statistics.median(lengths) if lengths else 0.0,
        p95_length=float(sorted_lengths[p95_idx]) if sorted_lengths else 0.0,
        min_length=min(lengths) if lengths else 0,
        max_length=max(lengths) if lengths else 0,
        unique_sections=len(sections),
        duplicate_chunks=dup_count,
        dedup_rate=dup_count / n if n else 0.0,
        min_chars_threshold=min_chars,
    )
