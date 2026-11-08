"""Startup check: the configured embedder must produce vectors of the size the
segment index was built with.

A wrong embedding server (e.g. bge-m3 1024-dim where the index holds 768-dim
e5-FT vectors) gives no error at ingest-config time; kNN just breaks or quietly
searches a different vector space. Fail at startup instead.

Only a definite mismatch raises. If ES or the embedder cannot be reached yet
(TEI still loading), the check is skipped with a warning so a slow dependency
does not keep the api down.
"""
from __future__ import annotations

import asyncio
import logging

logger = logging.getLogger(__name__)

_PROBE_TEXT = "embedding dimension check"


class EmbeddingDimMismatch(RuntimeError):
    pass


async def verify_embedding_dims(*, es, embedder, index: str, timeout_s: float = 10.0) -> str:
    from elasticsearch import NotFoundError

    try:
        mapping = await asyncio.wait_for(es.indices.get_mapping(index=index), timeout_s)
    except NotFoundError:
        return f"no index {index!r} yet — nothing to compare"
    except Exception as exc:
        return f"skipped: cannot read mapping of {index!r} ({type(exc).__name__}: {exc})"

    index_dims = None
    for body in (mapping or {}).values():
        field = (body.get("mappings", {}).get("properties", {}) or {}).get("embedding") or {}
        index_dims = field.get("dims")
        if index_dims:
            break
    if not index_dims:
        return f"skipped: {index!r} has no embedding.dims in its mapping"

    try:
        vectors = await asyncio.wait_for(embedder.embed([_PROBE_TEXT]), timeout_s)
    except Exception as exc:
        return f"skipped: embedder unreachable ({type(exc).__name__}: {exc})"
    embed_dims = len(vectors[0]) if vectors else 0

    if embed_dims != index_dims:
        raise EmbeddingDimMismatch(
            f"Embedder returns {embed_dims}-dim vectors but index {index!r} holds "
            f"{index_dims}-dim vectors. The embedding server is serving a different "
            "model than the one the corpus was indexed with (check which TEI is on "
            "EMBEDDING_BASE_URL — `make serve-embed` serves the e5-FT model)."
        )
    return f"embedding dims match: {index_dims}"


async def check_embedding_dims_at_startup(settings) -> None:
    from src.agentrag.ingestion.embedders.factory import build_embedding_provider
    from src.agentrag.ingestion.stores.elasticsearch_store import _get_shared_es_client

    result = await verify_embedding_dims(
        es=_get_shared_es_client(),
        embedder=build_embedding_provider(settings),
        index=settings.ELASTICSEARCH_INDEX_NAME,
    )
    (logger.warning if result.startswith("skipped") else logger.info)("startup: %s", result)
