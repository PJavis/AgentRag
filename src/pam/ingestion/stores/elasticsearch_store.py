from __future__ import annotations

import uuid
from hashlib import sha256
from typing import Any

from elasticsearch import AsyncElasticsearch

from src.pam.config import settings


class ElasticsearchStore:
    def __init__(self):
        self.client = AsyncElasticsearch([settings.ELASTICSEARCH_URL])
        self.index_name = settings.ELASTICSEARCH_INDEX_NAME
        self.entity_index_name = settings.ELASTICSEARCH_ENTITY_INDEX_NAME
        self.relationship_index_name = settings.ELASTICSEARCH_RELATIONSHIP_INDEX_NAME

    async def _get_index_embedding_dims(self, index_name: str) -> int | None:
        """Trả về số dims hiện tại của field embedding trong index, hoặc None nếu không có."""
        try:
            mapping = await self.client.indices.get_mapping(index=index_name)
            props = mapping[index_name]["mappings"].get("properties", {})
            return props.get("embedding", {}).get("dims")
        except Exception:
            return None

    async def _recreate_index_if_dims_changed(
        self, index_name: str, embedding_dims: int
    ) -> bool:
        """Xóa và trả về True nếu index tồn tại với dims khác. False nếu không cần xóa."""
        exists = await self.client.indices.exists(index=index_name)
        if not exists:
            return False
        current_dims = await self._get_index_embedding_dims(index_name)
        if current_dims is not None and current_dims != embedding_dims:
            await self.client.indices.delete(index=index_name)
            return True
        return False

    async def ensure_index(self, embedding_dims: int) -> None:
        await self._recreate_index_if_dims_changed(self.index_name, embedding_dims)
        exists = await self.client.indices.exists(index=self.index_name)
        if exists:
            return

        mapping = {
            "settings": {
                "analysis": {
                    "analyzer": {
                        "default": {
                            "type": "standard",
                        }
                    }
                }
            },
            "mappings": {
                "properties": {
                    "content": {"type": "text"},
                    "document_title": {
                        "type": "text",
                        "fields": {"keyword": {"type": "keyword"}},
                    },
                    "section_path": {
                        "type": "text",
                        "fields": {"keyword": {"type": "keyword"}},
                    },
                    "position": {"type": "integer"},
                    "content_hash": {"type": "keyword"},
                    "metadata": {"type": "object", "enabled": True},
                    "embedding": {
                        "type": "dense_vector",
                        "dims": embedding_dims,
                        "index": True,
                        "similarity": "cosine",
                    },
                }
            },
        }
        await self.client.indices.create(index=self.index_name, body=mapping)

    async def ensure_entity_index(self, embedding_dims: int) -> None:
        await self._recreate_index_if_dims_changed(self.entity_index_name, embedding_dims)
        exists = await self.client.indices.exists(index=self.entity_index_name)
        if exists:
            return
        mapping = {
            "settings": {
                "analysis": {
                    "analyzer": {
                        "default": {
                            "type": "standard",
                        }
                    }
                }
            },
            "mappings": {
                "properties": {
                    "name": {
                        "type": "text",
                        "fields": {"keyword": {"type": "keyword"}},
                    },
                    "type": {
                        "type": "text",
                        "fields": {"keyword": {"type": "keyword"}},
                    },
                    "description": {"type": "text"},
                    "document_title": {
                        "type": "text",
                        "fields": {"keyword": {"type": "keyword"}},
                    },
                    "group_id": {"type": "keyword"},
                    "source_node_uuid": {"type": "keyword"},
                    "embedding": {
                        "type": "dense_vector",
                        "dims": embedding_dims,
                        "index": True,
                        "similarity": "cosine",
                    },
                    "metadata": {"type": "object", "enabled": True},
                }
            },
        }
        await self.client.indices.create(index=self.entity_index_name, body=mapping)

    async def ensure_relationship_index(self, embedding_dims: int) -> None:
        await self._recreate_index_if_dims_changed(self.relationship_index_name, embedding_dims)
        exists = await self.client.indices.exists(index=self.relationship_index_name)
        if exists:
            return
        mapping = {
            "settings": {
                "analysis": {
                    "analyzer": {
                        "default": {
                            "type": "standard",
                        }
                    }
                }
            },
            "mappings": {
                "properties": {
                    "src_entity": {
                        "type": "text",
                        "fields": {"keyword": {"type": "keyword"}},
                    },
                    "tgt_entity": {
                        "type": "text",
                        "fields": {"keyword": {"type": "keyword"}},
                    },
                    "rel_type": {
                        "type": "text",
                        "fields": {"keyword": {"type": "keyword"}},
                    },
                    "keywords": {"type": "keyword"},
                    "description": {"type": "text"},
                    "document_title": {
                        "type": "text",
                        "fields": {"keyword": {"type": "keyword"}},
                    },
                    "group_id": {"type": "keyword"},
                    "source_node_uuid": {"type": "keyword"},
                    "target_node_uuid": {"type": "keyword"},
                    "edge_uuid": {"type": "keyword"},
                    "embedding": {
                        "type": "dense_vector",
                        "dims": embedding_dims,
                        "index": True,
                        "similarity": "cosine",
                    },
                    "metadata": {"type": "object", "enabled": True},
                }
            },
        }
        await self.client.indices.create(
            index=self.relationship_index_name,
            body=mapping,
        )

    async def index_segments(self, chunks: list[dict[str, Any]], document_title: str):
        if not chunks:
            return

        first_embedding = chunks[0].get("embedding")
        if not isinstance(first_embedding, list) or not first_embedding:
            raise ValueError("Chunks must contain non-empty embeddings before indexing")

        await self.ensure_index(len(first_embedding))

        actions = []
        for chunk in chunks:
            actions.append(
                {
                    "index": {
                        "_index": self.index_name,
                        "_id": str(uuid.uuid4()),
                    }
                }
            )
            actions.append(
                {
                    "content": chunk["content"],
                    "embedding": chunk["embedding"],
                    "document_title": document_title,
                    "section_path": chunk.get("section_path"),
                    "position": chunk.get("position"),
                    "content_hash": chunk.get("content_hash"),
                    "metadata": chunk.get("metadata", {}),
                }
            )

        response = await self.client.bulk(body=actions, refresh=True)
        if response.get("errors"):
            raise RuntimeError(f"Elasticsearch bulk indexing had errors: {response}")

    async def index_entities(self, entities: list[dict[str, Any]]) -> None:
        if not entities:
            return
        first_embedding = entities[0].get("embedding")
        if not isinstance(first_embedding, list) or not first_embedding:
            raise ValueError("Entities must contain non-empty embeddings before indexing")
        await self.ensure_entity_index(len(first_embedding))

        actions: list[dict[str, Any]] = []
        for entity in entities:
            entity_id = self._stable_entity_id(entity)
            actions.append(
                {
                    "index": {
                        "_index": self.entity_index_name,
                        "_id": entity_id,
                    }
                }
            )
            actions.append(entity)

        response = await self.client.bulk(body=actions, refresh=True)
        if response.get("errors"):
            raise RuntimeError(f"Elasticsearch entity indexing had errors: {response}")

    async def index_relationships(self, relationships: list[dict[str, Any]]) -> None:
        if not relationships:
            return
        first_embedding = relationships[0].get("embedding")
        if not isinstance(first_embedding, list) or not first_embedding:
            raise ValueError(
                "Relationships must contain non-empty embeddings before indexing"
            )
        await self.ensure_relationship_index(len(first_embedding))

        actions: list[dict[str, Any]] = []
        for rel in relationships:
            rel_id = self._stable_relationship_id(rel)
            actions.append(
                {
                    "index": {
                        "_index": self.relationship_index_name,
                        "_id": rel_id,
                    }
                }
            )
            actions.append(rel)

        response = await self.client.bulk(body=actions, refresh=True)
        if response.get("errors"):
            raise RuntimeError(
                f"Elasticsearch relationship indexing had errors: {response}"
            )

    async def sparse_search(
        self,
        query: str,
        top_k: int | None = None,
        document_title: str | None = None,
    ) -> list[dict[str, Any]]:
        size = top_k or settings.RETRIEVAL_TOP_K
        query_body: dict[str, Any] = {
            "multi_match": {
                "query": query,
                "fields": ["content^2", "document_title^1.5", "section_path"],
                "type": "best_fields",
            }
        }
        if document_title:
            query_body = {
                "bool": {
                    "must": [query_body],
                    "filter": [
                        {"term": {"document_title.keyword": document_title}},
                    ],
                }
            }
        response = await self.client.search(
            index=self.index_name,
            size=size,
            query=query_body,
        )
        return self._normalize_hits(response.get("hits", {}).get("hits", []), "sparse")

    async def dense_search(
        self,
        query_embedding: list[float],
        top_k: int | None = None,
        num_candidates: int | None = None,
        document_title: str | None = None,
    ) -> list[dict[str, Any]]:
        size = top_k or settings.RETRIEVAL_TOP_K
        candidates = num_candidates or settings.RETRIEVAL_NUM_CANDIDATES
        search_body: dict[str, Any] = {
            "knn": {
                "field": "embedding",
                "query_vector": query_embedding,
                "k": size,
                "num_candidates": max(candidates, size),
            },
            "size": size,
        }
        if document_title:
            search_body["knn"]["filter"] = {
                "term": {"document_title.keyword": document_title}
            }
        response = await self.client.search(index=self.index_name, **search_body)
        return self._normalize_hits(response.get("hits", {}).get("hits", []), "dense")

    async def hybrid_search(
        self,
        query: str,
        query_embedding: list[float],
        top_k: int | None = None,
        num_candidates: int | None = None,
        rrf_k: int | None = None,
        document_title: str | None = None,
    ) -> list[dict[str, Any]]:
        size = top_k or settings.RETRIEVAL_TOP_K
        sparse_hits = await self.sparse_search(
            query=query,
            top_k=size,
            document_title=document_title,
        )
        dense_hits = await self.dense_search(
            query_embedding=query_embedding,
            top_k=size,
            num_candidates=num_candidates,
            document_title=document_title,
        )
        return self._rrf_fuse(
            sparse_hits=sparse_hits,
            dense_hits=dense_hits,
            top_k=size,
            rrf_k=rrf_k or settings.RETRIEVAL_RRF_K,
        )

    async def get_chunks_by_hashes(self, content_hashes: list[str]) -> list[dict[str, Any]]:
        if not content_hashes:
            return []
        response = await self.client.search(
            index=self.index_name,
            size=len(content_hashes),
            query={
                "terms": {
                    "content_hash": content_hashes,
                }
            },
        )
        return self._normalize_hits(response.get("hits", {}).get("hits", []), "lookup")

    def _normalize_hits(
        self, hits: list[dict[str, Any]], source: str
    ) -> list[dict[str, Any]]:
        normalized = []
        for rank, hit in enumerate(hits, start=1):
            payload = hit.get("_source", {})
            normalized.append(
                {
                    "id": hit.get("_id"),
                    "score": hit.get("_score", 0.0),
                    "rank": rank,
                    "source": source,
                    "content": payload.get("content"),
                    "document_title": payload.get("document_title"),
                    "section_path": payload.get("section_path"),
                    "position": payload.get("position"),
                    "content_hash": payload.get("content_hash"),
                    "metadata": payload.get("metadata", {}),
                }
            )
        return normalized

    def _rrf_fuse(
        self,
        sparse_hits: list[dict[str, Any]],
        dense_hits: list[dict[str, Any]],
        top_k: int,
        rrf_k: int,
    ) -> list[dict[str, Any]]:
        fused: dict[str, dict[str, Any]] = {}

        def apply_rrf(hits: list[dict[str, Any]], label: str) -> None:
            for hit in hits:
                doc_id = hit["id"]
                if doc_id not in fused:
                    fused[doc_id] = {
                        **hit,
                        "rrf_score": 0.0,
                        "sources": [],
                        "sparse_score": None,
                        "dense_score": None,
                    }
                fused[doc_id]["rrf_score"] += 1.0 / (rrf_k + hit["rank"])
                fused[doc_id]["sources"].append(label)
                if label == "sparse":
                    fused[doc_id]["sparse_score"] = hit["score"]
                else:
                    fused[doc_id]["dense_score"] = hit["score"]

        apply_rrf(sparse_hits, "sparse")
        apply_rrf(dense_hits, "dense")

        ranked = sorted(
            fused.values(),
            key=lambda item: item["rrf_score"],
            reverse=True,
        )
        for rank, item in enumerate(ranked, start=1):
            item["rank"] = rank
        return ranked[:top_k]

    def _stable_entity_id(self, entity: dict[str, Any]) -> str:
        material = "|".join(
            [
                str(entity.get("document_title", "")),
                str(entity.get("group_id", "")),
                str(entity.get("name", "")).strip().lower(),
                str(entity.get("type", "")).strip().lower(),
            ]
        )
        return sha256(material.encode("utf-8")).hexdigest()

    def _stable_relationship_id(self, relationship: dict[str, Any]) -> str:
        material = "|".join(
            [
                str(relationship.get("document_title", "")),
                str(relationship.get("group_id", "")),
                str(relationship.get("src_entity", "")).strip().lower(),
                str(relationship.get("tgt_entity", "")).strip().lower(),
                str(relationship.get("rel_type", "")).strip().lower(),
                str(relationship.get("edge_uuid", "")),
            ]
        )
        return sha256(material.encode("utf-8")).hexdigest()
