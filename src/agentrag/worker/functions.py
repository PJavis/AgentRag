"""ARQ task functions — one per background job type.

Each function signature: async def name(ctx: dict, *, ...kwargs) -> None
ctx['redis'] is the ArqRedis pool, available for chaining jobs.

Run the worker:
    arq src.agentrag.worker.settings.WorkerSettings
"""
from __future__ import annotations

import logging
import uuid

logger = logging.getLogger(__name__)


async def graph_ingest(
    ctx: dict,
    *,
    document_id: str,
    folder_path: str,
    source_id: str,
    title: str,
    parsed_cache_path: str | None = None,
) -> None:
    """Parse, chunk, extract StructMem entries, and index a document."""
    from src.agentrag.graph.graph_jobs import GraphIngestJob, process_graph_job

    job = GraphIngestJob(
        document_id=uuid.UUID(document_id),
        folder_path=folder_path,
        source_id=source_id,
        title=title,
        parsed_cache_path=parsed_cache_path,
    )
    await process_graph_job(job, arq_pool=ctx["redis"])


async def consolidate(
    ctx: dict,
    *,
    group_id: str,
    document_id: str,
    trigger_chunk_count: int,
) -> None:
    """Cross-chunk consolidation — synthesise higher-level hypotheses from buffered entries."""
    from src.agentrag.graph.consolidation_jobs import ConsolidationJob, process_consolidation_job

    job = ConsolidationJob(
        group_id=group_id,
        document_id=uuid.UUID(document_id),
        trigger_chunk_count=trigger_chunk_count,
    )
    await process_consolidation_job(job)


async def chat_memory(
    ctx: dict,
    *,
    conversation_id: str,
    user_message: str,
    assistant_message: str,
    turn_id: str,
    turn_timestamp: str,
) -> None:
    """Extract dual-perspective memory entries from a chat turn and trigger consolidation if needed."""
    from src.agentrag.chat.structmem import ChatMemoryService
    from src.agentrag.config import settings

    svc = ChatMemoryService()
    await svc.process_turn(
        conversation_id=conversation_id,
        user_message=user_message,
        assistant_message=assistant_message,
        turn_id=turn_id,
        turn_timestamp=turn_timestamp,
    )
    count = await svc.count_unconsolidated(conversation_id)
    if count >= settings.CHAT_MEMORY_CONSOLIDATION_THRESHOLD:
        logger.info(
            "chat_memory: triggering consolidation for conv %s (%d unconsolidated)",
            conversation_id,
            count,
        )
        await svc.consolidate(conversation_id)
