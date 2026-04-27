# main.py
import asyncio
import os
import shutil
import tempfile
from contextlib import asynccontextmanager

from fastapi import FastAPI, Body, HTTPException, UploadFile, File
from fastapi.responses import StreamingResponse
from openai import RateLimitError
from sqlalchemy import select, delete

from src.agentrag.chat.history import ConversationStore
from src.agentrag.config import settings
from src.agentrag.config_validation import validate_settings
from src.agentrag.database import AsyncSessionLocal
from src.agentrag.database.models import Document, Segment
from src.agentrag.health.providers import collect_provider_health
from src.agentrag.ingestion.pipeline import ingest_folder
from src.agentrag.graph.graph_jobs import run_graph_worker
from src.agentrag.graph.consolidation_jobs import run_consolidation_worker
from src.agentrag.retrieval.elasticsearch_retriever import ElasticsearchRetriever
from src.agentrag.agent.service import AgentService
from src.agentrag.services.llm_gateway import LLMGateway


@asynccontextmanager
async def lifespan(app: FastAPI):
    validate_settings(settings)
    stop = asyncio.Event()
    graph_worker = asyncio.create_task(run_graph_worker(stop))
    consolidation_worker = asyncio.create_task(run_consolidation_worker(stop))
    yield
    stop.set()
    await graph_worker
    await consolidation_worker


app = FastAPI(lifespan=lifespan)


@app.get("/config/validate")
async def config_validate():
    try:
        validate_settings(settings)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {
        "ok": True,
        "providers": {
            "embedding": settings.EMBEDDING_PROVIDER,
            "extraction": settings.EXTRACTION_PROVIDER,
            "agent": settings.AGENT_PROVIDER or settings.EXTRACTION_PROVIDER,
        },
    }


@app.get("/health/providers")
async def provider_health():
    report = collect_provider_health(settings)
    if not report["ok"]:
        raise HTTPException(status_code=400, detail=report)
    return report


@app.post("/ingest/folder")
async def ingest(payload: dict = Body(...)):
    folder_path = payload.get("folder_path")
    if not folder_path:
        return {"error": "folder_path is required"}
    mode = payload.get("graph_ingest_mode")
    if mode is not None and mode not in ("sync", "async"):
        return {"error": "graph_ingest_mode must be 'sync' or 'async'"}
    result = await ingest_folder(folder_path, graph_ingest_mode=mode)
    return result


@app.post("/ingest/upload")
async def ingest_upload(file: UploadFile = File(...)):
    """Upload a single file (PDF, DOCX, PPTX, HTML, MD, XLSX, CSV, TXT) and ingest it."""
    tmp_dir = tempfile.mkdtemp(prefix="agentrag_upload_")
    try:
        dest = os.path.join(tmp_dir, file.filename or "upload")
        with open(dest, "wb") as f:
            shutil.copyfileobj(file.file, f)
        result = await ingest_folder(tmp_dir)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
    return result


@app.get("/documents")
async def list_documents(limit: int = 50):
    async with AsyncSessionLocal() as session:
        result = await session.execute(
            select(Document).order_by(Document.created_at.desc()).limit(limit)
        )
        docs = result.scalars().all()
    return {
        "documents": [
            {
                "document_id": str(d.id),
                "title": d.title,
                "source_type": d.source_type,
                "graph_status": d.graph_status,
                "graph_synced": d.graph_synced,
                "created_at": d.created_at.isoformat() if d.created_at else None,
            }
            for d in docs
        ]
    }


@app.delete("/documents/{document_id}")
async def delete_document(document_id: str):
    async with AsyncSessionLocal() as session:
        result = await session.execute(
            select(Document).where(Document.id == document_id)
        )
        doc = result.scalar_one_or_none()
        if doc is None:
            raise HTTPException(status_code=404, detail="document not found")
        await session.execute(delete(Segment).where(Segment.document_id == doc.id))
        await session.delete(doc)
        await session.commit()
    return {"ok": True, "document_id": document_id}


@app.post("/search")
async def search(payload: dict = Body(...)):
    query = payload.get("query")
    if not query:
        return {"error": "query is required"}

    mode = payload.get("mode", "hybrid_kg")
    top_k = payload.get("top_k")
    document_title = payload.get("document_title")
    rerank = payload.get("rerank")

    try:
        retriever = ElasticsearchRetriever()
        result = await retriever.search(
            query=query,
            mode=mode,
            top_k=top_k,
            document_title=document_title,
            rerank=rerank,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return result


@app.get("/documents/{document_id}/graph-status")
async def graph_status(document_id: str):
    async with AsyncSessionLocal() as session:
        result = await session.execute(
            select(Document).where(Document.id == document_id)
        )
        document = result.scalar_one_or_none()

    if document is None:
        raise HTTPException(status_code=404, detail="document not found")

    total = document.graph_total_chunks or 0
    processed = document.graph_processed_chunks or 0
    failed = document.graph_failed_chunks or 0
    progress = (processed / total) if total else 0.0

    return {
        "document_id": str(document.id),
        "title": document.title,
        "graph_status": document.graph_status,
        "graph_synced": document.graph_synced,
        "graph_total_chunks": total,
        "graph_processed_chunks": processed,
        "graph_failed_chunks": failed,
        "graph_progress": progress,
        "graph_last_error": document.graph_last_error,
    }


@app.post("/chat")
async def chat(payload: dict = Body(...)):
    question = payload.get("question")
    if not question:
        return {"error": "question is required"}
    document_title = payload.get("document_title")
    conversation_id = payload.get("conversation_id")
    conversation_title = payload.get("conversation_title")
    store = ConversationStore()
    conversation = await store.get_or_create_conversation(
        conversation_id=conversation_id,
        title=conversation_title,
    )
    history = await store.list_messages(
        conversation_id=conversation["conversation_id"],
        limit=settings.CHAT_HISTORY_WINDOW,
    )
    agent = AgentService()
    try:
        await store.append_message(
            conversation_id=conversation["conversation_id"],
            role="user",
            content=question,
            extra_metadata={"document_title": document_title},
        )
        result = await agent.chat(
            question=question,
            document_title=document_title,
            chat_history=history,
        )
        await store.append_message(
            conversation_id=conversation["conversation_id"],
            role="assistant",
            content=result.get("answer", ""),
            citations=result.get("citations", []),
            tool_trace=result.get("tool_trace", []),
            timings_ms=result.get("timings_ms", {}),
            extra_metadata={"document_title": document_title},
        )
        result["conversation_id"] = conversation["conversation_id"]
        return result
    except RateLimitError as exc:
        raise HTTPException(status_code=429, detail=str(exc)) from exc


@app.get("/metrics")
async def metrics():
    """LLM cost + token usage summary (chỉ có data khi LLM_COST_TRACKING_ENABLED=true)."""
    gateway = LLMGateway()
    return gateway.cost_summary()


@app.post("/conversations")
async def create_conversation(payload: dict = Body(default={})):
    store = ConversationStore()
    conversation = await store.create_conversation(
        title=payload.get("title"),
        extra_metadata=payload.get("metadata"),
    )
    return conversation


@app.get("/conversations")
async def list_conversations(limit: int = 20):
    store = ConversationStore()
    return {"conversations": await store.list_conversations(limit=limit)}


@app.get("/conversations/{conversation_id}/messages")
async def list_conversation_messages(conversation_id: str, limit: int = 20):
    store = ConversationStore()
    conversation = await store.get_conversation(conversation_id)
    if conversation is None:
        raise HTTPException(status_code=404, detail="conversation not found")
    messages = await store.list_messages(conversation_id=conversation_id, limit=limit)
    return {
        "conversation_id": conversation_id,
        "messages": messages,
    }


@app.delete("/conversations/{conversation_id}")
async def delete_conversation(conversation_id: str):
    store = ConversationStore()
    deleted = await store.delete_conversation(conversation_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="conversation not found")
    return {"ok": True, "conversation_id": conversation_id}


@app.post("/chat/stream")
async def chat_stream(payload: dict = Body(...)):
    """SSE streaming chat. Returns Server-Sent Events with event types: status, token, done, error."""
    question = payload.get("question")
    if not question:
        return {"error": "question is required"}
    document_title = payload.get("document_title")
    conversation_id = payload.get("conversation_id")
    conversation_title = payload.get("conversation_title")

    store = ConversationStore()
    conversation = await store.get_or_create_conversation(
        conversation_id=conversation_id,
        title=conversation_title,
    )
    history = await store.list_messages(
        conversation_id=conversation["conversation_id"],
        limit=settings.CHAT_HISTORY_WINDOW,
    )
    await store.append_message(
        conversation_id=conversation["conversation_id"],
        role="user",
        content=question,
        extra_metadata={"document_title": document_title},
    )

    agent = AgentService()

    async def event_generator():
        collected_tokens: list[str] = []
        async for chunk in agent.chat_stream(
            question=question,
            document_title=document_title,
            chat_history=history,
        ):
            if chunk.startswith("event: token"):
                import json as _json
                data_line = chunk.split("data: ", 1)[-1].strip()
                try:
                    token = _json.loads(data_line).get("text", "")
                    collected_tokens.append(token)
                except Exception:
                    pass
            yield chunk

        full_answer = "".join(collected_tokens)
        if full_answer:
            await store.append_message(
                conversation_id=conversation["conversation_id"],
                role="assistant",
                content=full_answer,
                extra_metadata={"document_title": document_title},
            )

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "X-Conversation-Id": conversation["conversation_id"],
        },
    )
