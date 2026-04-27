"""Cross-chunk consolidation worker — Phase 3 của StructMem.

Khi số chunks của một group vượt ngưỡng STRUCTMEM_CONSOLIDATION_THRESHOLD,
một ConsolidationJob được enqueue. Worker này:
1. Lấy các entries chưa consolidated từ pam_entries
2. Embed buffer → tìm top-K historical seeds
3. Reconstruct context tại cùng chunk_position với seeds
4. LLM synthesis call → cross-chunk relational hypotheses
5. Index vào pam_synthesis
6. Mark buffer entries là consolidated=True
"""
from __future__ import annotations

import asyncio
import json
import logging
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
from typing import Any

from openai import AsyncOpenAI

from src.agentrag.config import settings
from src.agentrag.ingestion.embedders.factory import build_embedding_provider
from src.agentrag.ingestion.stores.elasticsearch_store import ElasticsearchStore

logger = logging.getLogger(__name__)

consolidation_queue: asyncio.Queue[ConsolidationJob] = asyncio.Queue()

_SYNTHESIS_SYSTEM_PROMPT = """\
You are a knowledge synthesis specialist. You will receive memory entries extracted from different passages of a document.
Your task is to identify cross-passage patterns, generalizations, and higher-order relational hypotheses that are not directly stated in any single passage but are supported by reading multiple passages together.

Synthesize only what the evidence supports. Do not speculate beyond what the entries imply.

Output ONLY valid JSON with this exact schema:
{"synthesis_entries": [{"content": "<full natural-language synthesis statement>", "hypothesis_type": "<pattern|generalization|contradiction|causal_chain|gap>", "supporting_entry_ids": ["<entry_id_1>", "<entry_id_2>"], "entities_involved": ["<entity_1>", "<entity_2>"], "confidence": "<high|medium|low>", "reasoning": "<one sentence explaining why this synthesis is supported>"}]}

Rules:
- Produce 2 to 6 synthesis entries
- Each synthesis must reference at least 2 distinct source entries (by the entry id field)
- Do not restate individual facts — only cross-entry insights
- If passages are too fragmented for meaningful synthesis, return {"synthesis_entries": []}
- Do not output anything outside the JSON object\
"""


@dataclass(frozen=True)
class ConsolidationJob:
    group_id: str
    document_id: uuid.UUID
    trigger_chunk_count: int


async def process_consolidation_job(job: ConsolidationJob) -> None:
    embedder = build_embedding_provider(settings)
    es_store = ElasticsearchStore()

    # 1. Lấy entries chưa consolidated
    buffer_entries = await es_store.get_unconsolidated_entries(
        group_id=job.group_id,
        max_size=settings.STRUCTMEM_CONSOLIDATION_HISTORY_TOP_K * 5,
    )
    if not buffer_entries:
        logger.info("No unconsolidated entries for group %s — skipping", job.group_id)
        return

    logger.info(
        "Consolidating %s entries for group %s (doc %s)",
        len(buffer_entries),
        job.group_id,
        job.document_id,
    )

    # 2. Embed buffer aggregation → cosine search lấy historical seeds
    buffer_texts = [e.get("content", "") for e in buffer_entries]
    buffer_embedding_list = await embedder.embed(buffer_texts)
    # aggregate: mean embedding
    dims = len(buffer_embedding_list[0]) if buffer_embedding_list else 0
    if dims == 0:
        logger.warning("Empty embeddings for consolidation, aborting")
        return

    agg_embedding = [
        sum(vec[i] for vec in buffer_embedding_list) / len(buffer_embedding_list)
        for i in range(dims)
    ]

    top_k = settings.STRUCTMEM_CONSOLIDATION_HISTORY_TOP_K
    seed_hits = await es_store.search_entries(
        query_embedding=agg_embedding,
        query_text=" ".join(buffer_texts[:5]),  # top-5 texts để BM25
        group_ids=[job.group_id],
        top_k=top_k,
    )

    # 3. Reconstruct context: tất cả entries tại các chunk_position của seeds
    seed_positions = list(
        {
            h.get("metadata", {}).get("chunk_position")
            for h in seed_hits
            if h.get("metadata", {}).get("chunk_position") is not None
        }
    )
    historical_entries = await es_store.get_entries_by_chunk_position(
        group_id=job.group_id,
        chunk_positions=seed_positions,
    )

    # Loại bỏ entries đã có trong buffer (tránh duplicate context)
    buffer_ids = {e.get("_id") for e in buffer_entries}
    historical_entries = [e for e in historical_entries if e.get("_id") not in buffer_ids]

    # 4. Format context và gọi LLM
    synthesis_docs = await _run_synthesis(
        group_id=job.group_id,
        document_id=str(job.document_id),
        buffer_entries=buffer_entries,
        historical_entries=historical_entries,
    )

    # 5. Embed và index vào pam_synthesis
    if synthesis_docs:
        synth_texts = [d["content"] for d in synthesis_docs]
        synth_embeddings = await embedder.embed(synth_texts)
        for doc, emb in zip(synthesis_docs, synth_embeddings):
            doc["embedding"] = emb
        await es_store.index_synthesis(synthesis_docs)
        logger.info("Indexed %s synthesis entries for group %s", len(synthesis_docs), job.group_id)

    # 6. Mark buffer entries as consolidated
    entry_ids = [e.get("_id") for e in buffer_entries if e.get("_id")]
    await es_store.mark_entries_consolidated(entry_ids)
    logger.info("Marked %s entries as consolidated for group %s", len(entry_ids), job.group_id)


def _format_entries_for_prompt(entries: list[dict[str, Any]], label: str) -> str:
    if not entries:
        return f"[No {label} entries]"
    lines = []
    for entry in entries:
        eid = entry.get("_id", "unknown")
        entry_type = entry.get("entry_type", "?")
        content = entry.get("content", "")
        lines.append(f"[{eid[:12]}] ({entry_type}) {content}")
    return "\n".join(lines)


async def _run_synthesis(
    group_id: str,
    document_id: str,
    buffer_entries: list[dict[str, Any]],
    historical_entries: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    from src.agentrag.graph.structmem_service import StructMemService

    svc = StructMemService()
    client: AsyncOpenAI = svc._client
    model: str = svc._model

    factual_buffer = [e for e in buffer_entries if e.get("entry_type") == "factual"]
    relational_buffer = [e for e in buffer_entries if e.get("entry_type") == "relational"]
    all_buffer = factual_buffer + relational_buffer

    user_msg = (
        f"Document group: {group_id}\n\n"
        f"FACTUAL ENTRIES (current buffer):\n"
        f"{_format_entries_for_prompt(factual_buffer, 'factual')}\n\n"
        f"RELATIONAL ENTRIES (current buffer):\n"
        f"{_format_entries_for_prompt(relational_buffer, 'relational')}\n\n"
        f"HISTORICAL CONTEXT (semantically similar entries from earlier passages):\n"
        f"{_format_entries_for_prompt(historical_entries, 'historical')}\n\n"
        "Synthesize cross-passage relational hypotheses from these entries."
    )

    try:
        response = await client.chat.completions.create(
            model=model,
            temperature=0.0,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": _SYNTHESIS_SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
        )
        raw = response.choices[0].message.content or "{}"
        data = json.loads(raw)
        raw_entries = data.get("synthesis_entries", [])
        if not isinstance(raw_entries, list):
            return []
    except Exception as exc:
        logger.warning("Synthesis LLM call failed for group %s: %s", group_id, exc)
        return []

    run_id = sha256(f"{group_id}|{document_id}|{datetime.now(timezone.utc).isoformat()}".encode()).hexdigest()[:16]
    now = datetime.now(timezone.utc).isoformat()
    docs: list[dict[str, Any]] = []
    for entry in raw_entries:
        if not isinstance(entry, dict):
            continue
        content = (entry.get("content") or "").strip()
        if not content:
            continue
        doc_id = sha256(f"{group_id}|{run_id}|{content[:80]}".encode()).hexdigest()
        docs.append(
            {
                "_id": doc_id,
                "content": content,
                "hypothesis_type": entry.get("hypothesis_type") or "pattern",
                "supporting_entry_ids": entry.get("supporting_entry_ids") or [],
                "entities_involved": entry.get("entities_involved") or [],
                "confidence": entry.get("confidence") or "medium",
                "reasoning": (entry.get("reasoning") or "").strip(),
                "document_title": next(
                    (e.get("document_title", "") for e in all_buffer if e.get("document_title")),
                    "",
                ),
                "group_id": group_id,
                "consolidation_run_id": run_id,
                "source_entry_count": len(all_buffer) + len(historical_entries),
                "created_at": now,
            }
        )
    return docs


async def run_consolidation_worker(stop_event: asyncio.Event) -> None:
    logger.info("Consolidation worker started")
    while True:
        if stop_event.is_set() and consolidation_queue.empty():
            break
        try:
            job = await asyncio.wait_for(consolidation_queue.get(), timeout=0.5)
        except asyncio.TimeoutError:
            continue
        try:
            await process_consolidation_job(job)
        except Exception:
            logger.exception(
                "Unhandled error in consolidation worker for group %s — worker continues",
                job.group_id,
            )
        finally:
            consolidation_queue.task_done()
    logger.info("Consolidation worker stopped")
