"""Freshness check (eval metric #9).

Path-independent, run once: ingest a document, then RE-ingest an updated version
of the SAME document, and assert the current content out-ranks the stale prior
version in retrieval. Guards against the index serving outdated chunks after a
re-ingest.
"""
from __future__ import annotations

import shutil
import tempfile
from pathlib import Path
from typing import Any

_TITLE = "freshness_probe"
_QUERY = "Liều khởi đầu của thuốc Freshzine là bao nhiêu?"
_STALE_DOSE = "5 mg"
_FRESH_DOSE = "10 mg"


def _write_doc(dirpath: Path, dose: str) -> None:
    (dirpath / f"{_TITLE}.md").write_text(
        f"# Freshzine\n\nFreshzine là thuốc thử nghiệm. "
        f"Liều khởi đầu của Freshzine là {dose} mỗi ngày.\n",
        encoding="utf-8",
    )


def _rank_of(hits: list[dict[str, Any]], needle: str) -> int | None:
    for h in hits:
        if needle in (h.get("content") or ""):
            return int(h.get("rank") or 0)
    return None


async def run_freshness_check() -> dict[str, Any]:
    """Ingest v1 (stale) → v2 (fresh), then query. Pass if fresh out-ranks stale.

    Returns {"pass": bool, "fresh_rank", "stale_rank", "detail"}.
    """
    from src.agentrag.ingestion.pipeline import ingest_folder
    from src.agentrag.retrieval.elasticsearch_retriever import ElasticsearchRetriever

    tmp = Path(tempfile.mkdtemp(prefix="freshness_"))
    try:
        # v1 — stale.
        _write_doc(tmp, _STALE_DOSE)
        await ingest_folder(str(tmp))
        # v2 — fresh (same title, updated dose).
        _write_doc(tmp, _FRESH_DOSE)
        await ingest_folder(str(tmp))

        retriever = ElasticsearchRetriever()
        try:
            res = await retriever.search(query=_QUERY, mode="hybrid", top_k=10, document_title=_TITLE)
        finally:
            await retriever.store.client.close()

        hits = res.get("results", [])
        fresh_rank = _rank_of(hits, _FRESH_DOSE)
        stale_rank = _rank_of(hits, _STALE_DOSE)
        # Pass: fresh present AND (stale absent OR fresh ranks higher = smaller rank).
        passed = fresh_rank is not None and (stale_rank is None or fresh_rank <= stale_rank)
        return {
            "pass": passed,
            "fresh_rank": fresh_rank,
            "stale_rank": stale_rank,
            "detail": f"fresh({_FRESH_DOSE})@{fresh_rank} vs stale({_STALE_DOSE})@{stale_rank}",
        }
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
