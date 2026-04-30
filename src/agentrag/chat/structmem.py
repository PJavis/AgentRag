from __future__ import annotations

"""
Chat StructMem — áp dụng dual-perspective memory extraction cho lịch sử hội thoại.

Sau mỗi turn (user + assistant), 2 LLM calls song song trích xuất:
  - factual entries:   facts được nêu / xác nhận trong lượt hội thoại
  - relational entries: cách các chủ đề kết nối, intent người dùng

Entries được lưu vào Elasticsearch với embedding để semantic search.
Cross-turn consolidation tổng hợp insights khi tích luỹ đủ ngưỡng.

Retrieval trả về list[dict] được inject vào prompt của AgentService thay thế
sliding-window flat history khi CHAT_STRUCTMEM_ENABLED=true.
"""

import asyncio
import json
import logging
import uuid
from datetime import datetime, timezone
from typing import Any

from elasticsearch import AsyncElasticsearch
from openai import AsyncOpenAI

from src.agentrag.config import settings
from src.agentrag.ingestion.embedders.factory import build_embedding_provider

logger = logging.getLogger(__name__)

# ── Extraction prompts (conversation-specific, khác doc StructMem) ──────────

_FACTUAL_SYSTEM = """\
You are a conversation analyst. Given a conversation turn (user question + assistant response), \
extract factual statements capturing what was discussed, confirmed, or established.

Each entry must be:
- Standalone and interpretable without surrounding context
- Grounded in what was actually said (do not invent)
- Useful for answering future questions in this same conversation

Output ONLY valid JSON:
{"factual_entries": [{"content": "<standalone statement>", "subject": "<main topic>", "confidence": "<high|medium|low>"}]}

Rules:
- Extract 2–6 entries depending on information density
- Include specific names, numbers, and named concepts
- Do not output anything outside the JSON"""

_RELATIONAL_SYSTEM = """\
You are a conversation analyst. Given a conversation turn, extract relational dynamics \
showing how topics connect and how user intent evolves.

Capture:
- Topic connections: how the user's question relates to the answer given
- Follow-up signals: what this exchange reveals about underlying user goal
- Causal links: what prior context or need drove this question
- Comparison or contrast drawn between concepts

Output ONLY valid JSON:
{"relational_entries": [{"content": "<relationship description>", "source_entity": "<entity A>", "target_entity": "<entity B>", "relation_type": "<topic_flow|clarification|consequence|comparison|follow_up>", "confidence": "<high|medium|low>"}]}

Rules:
- Extract 1–4 entries
- Both entities must be clearly present in the exchange
- Do not output anything outside the JSON"""

_SYNTHESIS_SYSTEM = """\
You are a conversation analyst. Given memory entries from multiple conversation turns, \
identify cross-turn patterns and higher-order insights about user needs and intent.

Output ONLY valid JSON:
{"synthesis_entries": [{"content": "<synthesis statement>", "hypothesis_type": "<pattern|preference|goal|constraint|topic_evolution>", "supporting_entry_ids": ["<id1>", "<id2>"], "confidence": "<high|medium|low>", "reasoning": "<one sentence>"}]}

Rules:
- Produce 2–4 synthesis entries
- Each must reference at least 2 distinct entries by their ID
- Focus on insights that help predict or answer future questions
- Do not output anything outside the JSON"""


class ChatMemoryService:
    """
    Dual-perspective memory extraction + semantic retrieval cho chat history.
    Sử dụng cùng ES cluster với document StructMem nhưng index riêng.
    """

    def __init__(self) -> None:
        self._es = AsyncElasticsearch([settings.ELASTICSEARCH_URL])
        self._entries_idx = settings.CHAT_MEMORY_INDEX
        self._synthesis_idx = settings.CHAT_MEMORY_SYNTHESIS_INDEX
        self._embedder = build_embedding_provider(settings)
        self._llm = self._build_llm_client()
        self._model = settings.EXTRACTION_MODEL
        self._indices_ready = False

    # ── LLM backend (reuse EXTRACTION_* settings) ────────────────────────────

    def _build_llm_client(self) -> AsyncOpenAI:
        provider = settings.EXTRACTION_PROVIDER
        if provider == "openai":
            return AsyncOpenAI(
                api_key=settings.OPENAI_API_KEY or "x",
                base_url=settings.EXTRACTION_BASE_URL,
            )
        if provider == "ollama":
            return AsyncOpenAI(
                api_key=settings.OLLAMA_API_KEY,
                base_url=settings.EXTRACTION_BASE_URL or settings.OLLAMA_BASE_URL,
            )
        if provider == "gemini":
            return AsyncOpenAI(
                api_key=settings.GEMINI_API_KEY or "x",
                base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
            )
        # hf_inference
        return AsyncOpenAI(
            api_key=settings.HF_TOKEN or "x",
            base_url=settings.EXTRACTION_BASE_URL or settings.HF_OPENAI_BASE_URL,
        )

    # ── ES index management ──────────────────────────────────────────────────

    async def _ensure_indices(self, dims: int) -> None:
        if self._indices_ready:
            return
        entry_mapping = {
            "mappings": {
                "properties": {
                    "conversation_id": {"type": "keyword"},
                    "turn_id": {"type": "keyword"},
                    "turn_timestamp": {"type": "date"},
                    "entry_type": {"type": "keyword"},
                    "content": {"type": "text"},
                    "subject": {"type": "keyword"},
                    "source_entity": {"type": "keyword"},
                    "target_entity": {"type": "keyword"},
                    "relation_type": {"type": "keyword"},
                    "confidence": {"type": "keyword"},
                    "consolidated": {"type": "boolean"},
                    "embedding": {
                        "type": "dense_vector",
                        "dims": dims,
                        "index": True,
                        "similarity": "cosine",
                    },
                }
            }
        }
        synth_mapping = {
            "mappings": {
                "properties": {
                    "conversation_id": {"type": "keyword"},
                    "content": {"type": "text"},
                    "hypothesis_type": {"type": "keyword"},
                    "supporting_entry_ids": {"type": "keyword"},
                    "confidence": {"type": "keyword"},
                    "reasoning": {"type": "text"},
                    "created_at": {"type": "date"},
                    "embedding": {
                        "type": "dense_vector",
                        "dims": dims,
                        "index": True,
                        "similarity": "cosine",
                    },
                }
            }
        }
        for idx, body in [(self._entries_idx, entry_mapping), (self._synthesis_idx, synth_mapping)]:
            if not await self._es.indices.exists(index=idx):
                await self._es.indices.create(index=idx, body=body)
        self._indices_ready = True

    # ── LLM helper ───────────────────────────────────────────────────────────

    async def _llm_json(self, system: str, user: str) -> dict[str, Any]:
        try:
            resp = await self._llm.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                temperature=0.0,
                response_format={"type": "json_object"},
            )
            return json.loads(resp.choices[0].message.content or "{}")
        except Exception as exc:
            logger.debug("ChatMemory LLM call failed: %s", exc)
            return {}

    # ── Public API ───────────────────────────────────────────────────────────

    async def process_turn(
        self,
        conversation_id: str,
        user_message: str,
        assistant_message: str,
        turn_id: str,
        turn_timestamp: str,
    ) -> None:
        """Trích xuất dual-perspective entries từ một turn, embed và index vào ES."""
        turn_text = f"User: {user_message}\nAssistant: {assistant_message}"
        prompt = f"Conversation turn:\n{turn_text}"

        factual_raw, relational_raw = await asyncio.gather(
            self._llm_json(_FACTUAL_SYSTEM, prompt),
            self._llm_json(_RELATIONAL_SYSTEM, prompt),
        )
        factual_entries: list[dict] = factual_raw.get("factual_entries") or []
        relational_entries: list[dict] = relational_raw.get("relational_entries") or []

        all_entries = factual_entries + relational_entries
        contents = [e.get("content", "") for e in all_entries if e.get("content")]
        if not contents:
            logger.debug("ChatMemory: no entries extracted for turn %s", turn_id)
            return

        embeddings = await self._embedder.embed(contents)
        await self._ensure_indices(len(embeddings[0]))

        docs: list[dict[str, Any]] = []
        for entry, emb in zip(factual_entries, embeddings[: len(factual_entries)]):
            if not entry.get("content"):
                continue
            docs.append({
                "conversation_id": conversation_id,
                "turn_id": turn_id,
                "turn_timestamp": turn_timestamp,
                "entry_type": "factual",
                "content": entry["content"],
                "subject": entry.get("subject", ""),
                "confidence": entry.get("confidence", "medium"),
                "consolidated": False,
                "embedding": emb,
            })

        for entry, emb in zip(relational_entries, embeddings[len(factual_entries) :]):
            if not entry.get("content"):
                continue
            docs.append({
                "conversation_id": conversation_id,
                "turn_id": turn_id,
                "turn_timestamp": turn_timestamp,
                "entry_type": "relational",
                "content": entry["content"],
                "source_entity": entry.get("source_entity", ""),
                "target_entity": entry.get("target_entity", ""),
                "relation_type": entry.get("relation_type", ""),
                "confidence": entry.get("confidence", "medium"),
                "consolidated": False,
                "embedding": emb,
            })

        for doc in docs:
            await self._es.index(index=self._entries_idx, body=doc)

        logger.debug("ChatMemory: indexed %d entries for turn %s", len(docs), turn_id)

    async def retrieve(
        self,
        conversation_id: str,
        query: str,
        top_k: int | None = None,
    ) -> list[dict[str, Any]]:
        """
        Semantic search trên entries + synthesis của conversation_id.
        Trả về list[dict] để inject vào prompt của AgentService.
        """
        k = top_k or settings.CHAT_MEMORY_TOP_K
        try:
            if not await self._es.indices.exists(index=self._entries_idx):
                return []
            query_emb = (await self._embedder.embed([query]))[0]

            entry_resp = await self._es.search(
                index=self._entries_idx,
                body={
                    "knn": {
                        "field": "embedding",
                        "query_vector": query_emb,
                        "k": k,
                        "num_candidates": k * 5,
                        "filter": {"term": {"conversation_id": conversation_id}},
                    },
                    "size": k,
                },
            )
            results: list[dict[str, Any]] = []
            for hit in entry_resp["hits"]["hits"]:
                src = hit["_source"]
                results.append({
                    "content": src.get("content", ""),
                    "entry_type": src.get("entry_type", ""),
                    "turn_timestamp": src.get("turn_timestamp", ""),
                    "confidence": src.get("confidence", ""),
                    "score": hit.get("_score", 0.0),
                })

            # Synthesis entries (top 3)
            if await self._es.indices.exists(index=self._synthesis_idx):
                synth_resp = await self._es.search(
                    index=self._synthesis_idx,
                    body={
                        "knn": {
                            "field": "embedding",
                            "query_vector": query_emb,
                            "k": 3,
                            "num_candidates": 20,
                            "filter": {"term": {"conversation_id": conversation_id}},
                        },
                        "size": 3,
                    },
                )
                for hit in synth_resp["hits"]["hits"]:
                    src = hit["_source"]
                    results.append({
                        "content": src.get("content", ""),
                        "entry_type": "synthesis",
                        "turn_timestamp": src.get("created_at", ""),
                        "confidence": src.get("confidence", ""),
                        "score": hit.get("_score", 0.0),
                    })

            results.sort(key=lambda x: x["score"], reverse=True)
            return results[:k]

        except Exception as exc:
            logger.warning("ChatMemory.retrieve failed: %s", exc)
            return []

    async def count_unconsolidated(self, conversation_id: str) -> int:
        try:
            if not await self._es.indices.exists(index=self._entries_idx):
                return 0
            resp = await self._es.count(
                index=self._entries_idx,
                body={
                    "query": {
                        "bool": {
                            "must": [
                                {"term": {"conversation_id": conversation_id}},
                                {"term": {"consolidated": False}},
                            ]
                        }
                    }
                },
            )
            return int(resp["count"])
        except Exception:
            return 0

    async def consolidate(self, conversation_id: str) -> None:
        """Cross-turn consolidation: tổng hợp entries thành higher-level hypotheses."""
        try:
            if not await self._es.indices.exists(index=self._entries_idx):
                return

            # 1. Lấy buffer (unconsolidated entries)
            buf_resp = await self._es.search(
                index=self._entries_idx,
                body={
                    "query": {
                        "bool": {
                            "must": [
                                {"term": {"conversation_id": conversation_id}},
                                {"term": {"consolidated": False}},
                            ]
                        }
                    },
                    "size": 100,
                    "sort": [{"turn_timestamp": "asc"}],
                },
            )
            buffer = [{"_id": h["_id"], **h["_source"]} for h in buf_resp["hits"]["hits"]]
            if not buffer:
                return

            # 2. Aggregate embedding → find historical seeds
            buf_texts = [e.get("content", "") for e in buffer]
            embs = await self._embedder.embed(buf_texts)
            dims = len(embs[0])
            await self._ensure_indices(dims)

            agg = [sum(v[i] for v in embs) / len(embs) for i in range(dims)]
            seed_resp = await self._es.search(
                index=self._entries_idx,
                body={
                    "knn": {
                        "field": "embedding",
                        "query_vector": agg,
                        "k": settings.STRUCTMEM_CONSOLIDATION_HISTORY_TOP_K,
                        "num_candidates": 50,
                        "filter": {"term": {"conversation_id": conversation_id}},
                    },
                    "size": settings.STRUCTMEM_CONSOLIDATION_HISTORY_TOP_K,
                },
            )
            buf_ids = {e["_id"] for e in buffer}
            historical = [
                {"_id": h["_id"], **h["_source"]}
                for h in seed_resp["hits"]["hits"]
                if h["_id"] not in buf_ids
            ]

            # 3. LLM synthesis
            def _fmt(entries: list[dict], label: str) -> str:
                lines = [
                    f"[{e['_id'][:8]}] ({e.get('entry_type', '?')}) {e.get('content', '')}"
                    for e in entries
                ]
                return f"--- {label} ---\n" + "\n".join(lines)

            result = await self._llm_json(
                _SYNTHESIS_SYSTEM,
                f"{_fmt(buffer, 'Current buffer')}\n\n{_fmt(historical, 'Historical context')}",
            )
            synth_entries: list[dict] = result.get("synthesis_entries") or []

            if synth_entries:
                synth_texts = [e.get("content", "") for e in synth_entries]
                synth_embs = await self._embedder.embed(synth_texts)
                now = datetime.now(timezone.utc).isoformat()
                for entry, emb in zip(synth_entries, synth_embs):
                    await self._es.index(
                        index=self._synthesis_idx,
                        body={
                            "conversation_id": conversation_id,
                            "content": entry.get("content", ""),
                            "hypothesis_type": entry.get("hypothesis_type", ""),
                            "supporting_entry_ids": entry.get("supporting_entry_ids", []),
                            "confidence": entry.get("confidence", "medium"),
                            "reasoning": entry.get("reasoning", ""),
                            "created_at": now,
                            "embedding": emb,
                        },
                    )
                logger.info(
                    "ChatMemory: synthesized %d entries for conv %s",
                    len(synth_entries),
                    conversation_id,
                )

            # 4. Mark buffer consolidated
            for entry in buffer:
                await self._es.update(
                    index=self._entries_idx,
                    id=entry["_id"],
                    body={"doc": {"consolidated": True}},
                )

        except Exception as exc:
            logger.warning("ChatMemory.consolidate failed for %s: %s", conversation_id, exc)
