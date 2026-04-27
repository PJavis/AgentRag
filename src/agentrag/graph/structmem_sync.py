"""StructMem sync — thay thế entity_sync.py.

Build và index factual/relational entries vào pam_entries index.
"""
from __future__ import annotations

from datetime import datetime, timezone
from hashlib import sha256
from typing import Any

from src.agentrag.ingestion.embedders.base import BaseEmbeddingProvider
from src.agentrag.ingestion.stores.elasticsearch_store import ElasticsearchStore


def _make_entry_id(group_id: str, chunk_position: int, entry_type: str, content: str) -> str:
    key = f"{group_id}|{chunk_position}|{entry_type}|{content[:80]}"
    return sha256(key.encode("utf-8")).hexdigest()


def build_entry_docs(
    structmem_results: list[dict[str, Any]],
    document_title: str,
    group_id: str,
) -> list[dict[str, Any]]:
    """Convert raw StructMem extraction results thành ES documents."""
    docs: list[dict[str, Any]] = []
    now = datetime.now(timezone.utc).isoformat()

    for result in structmem_results:
        chunk_position = result.get("chunk_position", result.get("chunk_index", 0))
        content_hash = result.get("content_hash", "")

        for entry in result.get("factual_entries", []) or []:
            content = (entry.get("content") or "").strip()
            if not content:
                continue
            docs.append(
                {
                    "_id": _make_entry_id(group_id, chunk_position, "factual", content),
                    "content": content,
                    "entry_type": "factual",
                    "fact_type": entry.get("fact_type") or None,
                    "subject": entry.get("subject") or None,
                    "confidence": entry.get("confidence") or None,
                    "relation_type": None,
                    "source_entity": None,
                    "target_entity": None,
                    "document_title": document_title,
                    "group_id": group_id,
                    "chunk_position": chunk_position,
                    "content_hash": content_hash,
                    "consolidated": False,
                    "created_at": now,
                }
            )

        for entry in result.get("relational_entries", []) or []:
            content = (entry.get("content") or "").strip()
            if not content:
                continue
            docs.append(
                {
                    "_id": _make_entry_id(group_id, chunk_position, "relational", content),
                    "content": content,
                    "entry_type": "relational",
                    "fact_type": None,
                    "subject": None,
                    "confidence": entry.get("confidence") or None,
                    "relation_type": entry.get("relation_type") or None,
                    "source_entity": (entry.get("source_entity") or "").strip() or None,
                    "target_entity": (entry.get("target_entity") or "").strip() or None,
                    "document_title": document_title,
                    "group_id": group_id,
                    "chunk_position": chunk_position,
                    "content_hash": content_hash,
                    "consolidated": False,
                    "created_at": now,
                }
            )

    return docs


async def index_structmem_views(
    *,
    es_store: ElasticsearchStore,
    embedder: BaseEmbeddingProvider,
    structmem_results: list[dict[str, Any]],
    document_title: str,
    group_id: str,
) -> dict[str, int]:
    """Embed entries và bulk-index vào pam_entries."""
    docs = build_entry_docs(
        structmem_results=structmem_results,
        document_title=document_title,
        group_id=group_id,
    )
    if not docs:
        return {"entries_indexed": 0}

    texts = [doc["content"] for doc in docs]
    embeddings = await embedder.embed(texts)
    for doc, embedding in zip(docs, embeddings):
        doc["embedding"] = embedding

    await es_store.index_entries(docs)
    return {"entries_indexed": len(docs)}
