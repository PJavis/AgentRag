from __future__ import annotations

from typing import Any

from src.pam.ingestion.embedders.base import BaseEmbeddingProvider
from src.pam.ingestion.stores.elasticsearch_store import ElasticsearchStore


def _normalize_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _normalize_keywords(raw: Any) -> list[str]:
    if raw is None:
        return []
    if isinstance(raw, str):
        return [raw.strip()] if raw.strip() else []
    if isinstance(raw, list):
        return [str(item).strip() for item in raw if str(item).strip()]
    return []


def _dedupe_entities(entities: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[tuple[str, str, str, str]] = set()
    deduped: list[dict[str, Any]] = []
    for entity in entities:
        key = (
            _normalize_text(entity.get("document_title")).lower(),
            _normalize_text(entity.get("group_id")).lower(),
            _normalize_text(entity.get("name")).lower(),
            _normalize_text(entity.get("type")).lower(),
        )
        if not key[2]:
            continue
        if key in seen:
            continue
        seen.add(key)
        deduped.append(entity)
    return deduped


def _dedupe_relationships(relationships: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[tuple[str, str, str, str, str, str]] = set()
    deduped: list[dict[str, Any]] = []
    for rel in relationships:
        key = (
            _normalize_text(rel.get("document_title")).lower(),
            _normalize_text(rel.get("group_id")).lower(),
            _normalize_text(rel.get("src_entity")).lower(),
            _normalize_text(rel.get("tgt_entity")).lower(),
            _normalize_text(rel.get("rel_type")).lower(),
            _normalize_text(rel.get("edge_uuid")).lower(),
        )
        if not key[2] and not key[3]:
            continue
        if key in seen:
            continue
        seen.add(key)
        deduped.append(rel)
    return deduped


def build_entity_relationship_docs(
    graph_results: list[dict[str, Any]],
    document_title: str,
    group_id: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    entities: list[dict[str, Any]] = []
    relationships: list[dict[str, Any]] = []

    for result in graph_results:
        for entity in result.get("entity_records", []) or []:
            name = _normalize_text(entity.get("name"))
            if not name:
                continue
            entities.append(
                {
                    "name": name,
                    "type": _normalize_text(entity.get("type")),
                    "description": _normalize_text(entity.get("description")),
                    "document_title": document_title,
                    "group_id": group_id,
                    "source_node_uuid": _normalize_text(entity.get("node_uuid")),
                    "metadata": {"content_hash": result.get("content_hash")},
                }
            )
        for rel in result.get("relationship_records", []) or []:
            src_entity = _normalize_text(rel.get("src_entity"))
            tgt_entity = _normalize_text(rel.get("tgt_entity"))
            rel_type = _normalize_text(rel.get("rel_type"))
            if not (src_entity or tgt_entity or rel_type):
                continue
            relationships.append(
                {
                    "src_entity": src_entity,
                    "tgt_entity": tgt_entity,
                    "rel_type": rel_type,
                    "keywords": _normalize_keywords(rel.get("keywords")),
                    "description": _normalize_text(rel.get("description")),
                    "document_title": document_title,
                    "group_id": group_id,
                    "source_node_uuid": _normalize_text(rel.get("source_node_uuid")),
                    "target_node_uuid": _normalize_text(rel.get("target_node_uuid")),
                    "edge_uuid": _normalize_text(rel.get("edge_uuid")),
                    "metadata": {"content_hash": result.get("content_hash")},
                }
            )

        # Backward-compatible fallback for old cached schema.
        for entity_name in result.get("entities", []) or []:
            name = _normalize_text(entity_name)
            if not name:
                continue
            entities.append(
                {
                    "name": name,
                    "type": "",
                    "description": "",
                    "document_title": document_title,
                    "group_id": group_id,
                    "source_node_uuid": "",
                    "metadata": {"content_hash": result.get("content_hash")},
                }
            )

    return _dedupe_entities(entities), _dedupe_relationships(relationships)


async def index_graph_entity_views(
    *,
    es_store: ElasticsearchStore,
    embedder: BaseEmbeddingProvider,
    graph_results: list[dict[str, Any]],
    document_title: str,
    group_id: str,
) -> dict[str, int]:
    entities, relationships = build_entity_relationship_docs(
        graph_results=graph_results,
        document_title=document_title,
        group_id=group_id,
    )

    if entities:
        entity_texts = [
            " ".join(
                part for part in [
                    entity.get("name", ""),
                    entity.get("type", ""),
                    entity.get("description", ""),
                ] if part
            )
            for entity in entities
        ]
        entity_embeddings = await embedder.embed(entity_texts)
        for entity, embedding in zip(entities, entity_embeddings):
            entity["embedding"] = embedding
        await es_store.index_entities(entities)

    if relationships:
        rel_texts = [
            " ".join(
                part
                for part in [
                    rel.get("src_entity", ""),
                    rel.get("rel_type", ""),
                    rel.get("tgt_entity", ""),
                    rel.get("description", ""),
                    " ".join(rel.get("keywords", [])),
                ]
                if part
            )
            for rel in relationships
        ]
        rel_embeddings = await embedder.embed(rel_texts)
        for rel, embedding in zip(relationships, rel_embeddings):
            rel["embedding"] = embedding
        await es_store.index_relationships(relationships)

    return {
        "entities_indexed": len(entities),
        "relationships_indexed": len(relationships),
    }
