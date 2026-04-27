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
        self.entries_index_name = settings.STRUCTMEM_ENTRIES_INDEX_NAME
        self.synthesis_index_name = settings.STRUCTMEM_SYNTHESIS_INDEX_NAME

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

    # ------------------------------------------------------------------
    # StructMem — pam_entries index
    # ------------------------------------------------------------------

    async def ensure_entries_index(self, embedding_dims: int) -> None:
        await self._recreate_index_if_dims_changed(self.entries_index_name, embedding_dims)
        exists = await self.client.indices.exists(index=self.entries_index_name)
        if exists:
            return
        mapping = {
            "settings": {"analysis": {"analyzer": {"default": {"type": "standard"}}}},
            "mappings": {
                "properties": {
                    "content": {"type": "text"},
                    "entry_type": {"type": "keyword"},
                    "fact_type": {"type": "keyword"},
                    "subject": {
                        "type": "text",
                        "fields": {"keyword": {"type": "keyword"}},
                    },
                    "relation_type": {"type": "keyword"},
                    "source_entity": {
                        "type": "text",
                        "fields": {"keyword": {"type": "keyword"}},
                    },
                    "target_entity": {
                        "type": "text",
                        "fields": {"keyword": {"type": "keyword"}},
                    },
                    "confidence": {"type": "keyword"},
                    "document_title": {
                        "type": "text",
                        "fields": {"keyword": {"type": "keyword"}},
                    },
                    "group_id": {"type": "keyword"},
                    "chunk_position": {"type": "integer"},
                    "content_hash": {"type": "keyword"},
                    "consolidated": {"type": "boolean"},
                    "embedding": {
                        "type": "dense_vector",
                        "dims": embedding_dims,
                        "index": True,
                        "similarity": "cosine",
                    },
                    "created_at": {"type": "date"},
                }
            },
        }
        await self.client.indices.create(index=self.entries_index_name, body=mapping)

    async def index_entries(self, entries: list[dict[str, Any]]) -> None:
        if not entries:
            return
        first_embedding = entries[0].get("embedding")
        if not isinstance(first_embedding, list) or not first_embedding:
            raise ValueError("Entries must contain non-empty embeddings before indexing")
        await self.ensure_entries_index(len(first_embedding))

        actions: list[dict[str, Any]] = []
        for entry in entries:
            doc_id = entry.pop("_id", None) or sha256(
                entry.get("content", "").encode("utf-8")
            ).hexdigest()
            actions.append({"index": {"_index": self.entries_index_name, "_id": doc_id}})
            actions.append(entry)

        response = await self.client.bulk(body=actions, refresh=True)
        if response.get("errors"):
            raise RuntimeError(f"Elasticsearch entries indexing had errors: {response}")

    async def search_entries(
        self,
        query_embedding: list[float],
        query_text: str,
        group_ids: list[str] | None = None,
        top_k: int | None = None,
    ) -> list[dict[str, Any]]:
        """Hybrid kNN + BM25 search trên pam_entries."""
        size = top_k or settings.RETRIEVAL_TOP_K
        knn_filter: dict[str, Any] | None = None
        bm25_filter: dict[str, Any] | None = None
        if group_ids:
            knn_filter = {"terms": {"group_id": group_ids}}
            bm25_filter = {"terms": {"group_id": group_ids}}

        sparse_query: dict[str, Any] = {"match": {"content": {"query": query_text}}}
        if bm25_filter:
            sparse_query = {"bool": {"must": [sparse_query], "filter": [bm25_filter]}}

        sparse_resp = await self.client.search(
            index=self.entries_index_name,
            size=size,
            query=sparse_query,
        )
        sparse_hits = self._normalize_entry_hits(
            sparse_resp.get("hits", {}).get("hits", []), "structmem"
        )

        knn_body: dict[str, Any] = {
            "knn": {
                "field": "embedding",
                "query_vector": query_embedding,
                "k": size,
                "num_candidates": max(settings.RETRIEVAL_NUM_CANDIDATES, size),
            },
            "size": size,
        }
        if knn_filter:
            knn_body["knn"]["filter"] = knn_filter
        dense_resp = await self.client.search(index=self.entries_index_name, **knn_body)
        dense_hits = self._normalize_entry_hits(
            dense_resp.get("hits", {}).get("hits", []), "structmem"
        )

        return self._rrf_fuse_entries(sparse_hits, dense_hits, size, settings.RETRIEVAL_RRF_K)

    async def get_entries_by_chunk_position(
        self,
        group_id: str,
        chunk_positions: list[int],
    ) -> list[dict[str, Any]]:
        """Lấy tất cả entries tại các chunk positions cho trước (dùng trong consolidation)."""
        if not chunk_positions:
            return []
        response = await self.client.search(
            index=self.entries_index_name,
            size=1000,
            query={
                "bool": {
                    "filter": [
                        {"term": {"group_id": group_id}},
                        {"terms": {"chunk_position": chunk_positions}},
                    ]
                }
            },
        )
        hits = response.get("hits", {}).get("hits", [])
        return [hit.get("_source", {}) for hit in hits]

    async def get_unconsolidated_entries(
        self,
        group_id: str,
        max_size: int = 500,
    ) -> list[dict[str, Any]]:
        """Trả về entries chưa consolidate của một group."""
        response = await self.client.search(
            index=self.entries_index_name,
            size=max_size,
            sort=[{"chunk_position": "asc"}],
            query={
                "bool": {
                    "filter": [
                        {"term": {"group_id": group_id}},
                        {"term": {"consolidated": False}},
                    ]
                }
            },
        )
        hits = response.get("hits", {}).get("hits", [])
        return [{"_id": h["_id"], **h.get("_source", {})} for h in hits]

    async def mark_entries_consolidated(self, entry_ids: list[str]) -> None:
        """Bulk update consolidated=True cho danh sách entry ids."""
        if not entry_ids:
            return
        actions: list[dict[str, Any]] = []
        for eid in entry_ids:
            actions.append({"update": {"_index": self.entries_index_name, "_id": eid}})
            actions.append({"doc": {"consolidated": True}})
        await self.client.bulk(body=actions, refresh=False)

    def _normalize_entry_hits(
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
                    "section_path": "structmem_entries",
                    "position": payload.get("chunk_position"),
                    "content_hash": payload.get("content_hash"),
                    "metadata": {
                        "entry_type": payload.get("entry_type"),
                        "group_id": payload.get("group_id"),
                        "chunk_position": payload.get("chunk_position"),
                        "relation_type": payload.get("relation_type"),
                        "source_entity": payload.get("source_entity"),
                        "target_entity": payload.get("target_entity"),
                    },
                }
            )
        return normalized

    def _rrf_fuse_entries(
        self,
        sparse_hits: list[dict[str, Any]],
        dense_hits: list[dict[str, Any]],
        top_k: int,
        rrf_k: int,
    ) -> list[dict[str, Any]]:
        fused: dict[str, dict[str, Any]] = {}
        for hits in (sparse_hits, dense_hits):
            for hit in hits:
                doc_id = hit["id"]
                if doc_id not in fused:
                    fused[doc_id] = {**hit, "rrf_score": 0.0}
                fused[doc_id]["rrf_score"] += 1.0 / (rrf_k + hit["rank"])
        ranked = sorted(fused.values(), key=lambda x: x["rrf_score"], reverse=True)
        for rank, item in enumerate(ranked, start=1):
            item["rank"] = rank
        return ranked[:top_k]

    # ------------------------------------------------------------------
    # StructMem — pam_synthesis index
    # ------------------------------------------------------------------

    async def ensure_synthesis_index(self, embedding_dims: int) -> None:
        await self._recreate_index_if_dims_changed(self.synthesis_index_name, embedding_dims)
        exists = await self.client.indices.exists(index=self.synthesis_index_name)
        if exists:
            return
        mapping = {
            "settings": {"analysis": {"analyzer": {"default": {"type": "standard"}}}},
            "mappings": {
                "properties": {
                    "content": {"type": "text"},
                    "hypothesis_type": {"type": "keyword"},
                    "supporting_entry_ids": {"type": "keyword"},
                    "entities_involved": {
                        "type": "text",
                        "fields": {"keyword": {"type": "keyword"}},
                    },
                    "confidence": {"type": "keyword"},
                    "reasoning": {"type": "text"},
                    "document_title": {
                        "type": "text",
                        "fields": {"keyword": {"type": "keyword"}},
                    },
                    "group_id": {"type": "keyword"},
                    "consolidation_run_id": {"type": "keyword"},
                    "source_entry_count": {"type": "integer"},
                    "embedding": {
                        "type": "dense_vector",
                        "dims": embedding_dims,
                        "index": True,
                        "similarity": "cosine",
                    },
                    "created_at": {"type": "date"},
                }
            },
        }
        await self.client.indices.create(index=self.synthesis_index_name, body=mapping)

    async def index_synthesis(self, synthesis_docs: list[dict[str, Any]]) -> None:
        if not synthesis_docs:
            return
        first_embedding = synthesis_docs[0].get("embedding")
        if not isinstance(first_embedding, list) or not first_embedding:
            raise ValueError("Synthesis docs must contain non-empty embeddings before indexing")
        await self.ensure_synthesis_index(len(first_embedding))

        actions: list[dict[str, Any]] = []
        for doc in synthesis_docs:
            doc_id = doc.pop("_id", None) or sha256(
                doc.get("content", "").encode("utf-8")
            ).hexdigest()
            actions.append({"index": {"_index": self.synthesis_index_name, "_id": doc_id}})
            actions.append(doc)

        response = await self.client.bulk(body=actions, refresh=True)
        if response.get("errors"):
            raise RuntimeError(f"Elasticsearch synthesis indexing had errors: {response}")

    async def search_synthesis(
        self,
        query_embedding: list[float],
        query_text: str,
        group_ids: list[str] | None = None,
        top_k: int | None = None,
    ) -> list[dict[str, Any]]:
        """Hybrid search trên pam_synthesis."""
        size = top_k or max(settings.RETRIEVAL_TOP_K // 2, 3)
        knn_filter: dict[str, Any] | None = None
        if group_ids:
            knn_filter = {"terms": {"group_id": group_ids}}

        sparse_query: dict[str, Any] = {"match": {"content": {"query": query_text}}}
        if group_ids:
            sparse_query = {
                "bool": {
                    "must": [sparse_query],
                    "filter": [{"terms": {"group_id": group_ids}}],
                }
            }

        sparse_resp = await self.client.search(
            index=self.synthesis_index_name,
            size=size,
            query=sparse_query,
        )
        sparse_hits = self._normalize_synthesis_hits(
            sparse_resp.get("hits", {}).get("hits", [])
        )

        knn_body: dict[str, Any] = {
            "knn": {
                "field": "embedding",
                "query_vector": query_embedding,
                "k": size,
                "num_candidates": max(settings.RETRIEVAL_NUM_CANDIDATES, size),
            },
            "size": size,
        }
        if knn_filter:
            knn_body["knn"]["filter"] = knn_filter
        dense_resp = await self.client.search(index=self.synthesis_index_name, **knn_body)
        dense_hits = self._normalize_synthesis_hits(dense_resp.get("hits", {}).get("hits", []))

        return self._rrf_fuse_entries(sparse_hits, dense_hits, size, settings.RETRIEVAL_RRF_K)

    def _normalize_synthesis_hits(self, hits: list[dict[str, Any]]) -> list[dict[str, Any]]:
        normalized = []
        for rank, hit in enumerate(hits, start=1):
            payload = hit.get("_source", {})
            normalized.append(
                {
                    "id": hit.get("_id"),
                    "score": hit.get("_score", 0.0),
                    "rank": rank,
                    "source": "synthesis",
                    "content": payload.get("content"),
                    "document_title": payload.get("document_title"),
                    "section_path": "structmem_synthesis",
                    "position": None,
                    "content_hash": hit.get("_id"),
                    "metadata": {
                        "hypothesis_type": payload.get("hypothesis_type"),
                        "group_id": payload.get("group_id"),
                        "entities_involved": payload.get("entities_involved"),
                        "confidence": payload.get("confidence"),
                    },
                }
            )
        return normalized

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
