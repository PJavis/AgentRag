"""Search endpoints mapping to AgentRag retrieval."""
from __future__ import annotations

import json
from datetime import datetime, timezone

from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse

from src.agentrag.adapter.models import AskRequest, SearchRequest, SearchResponse, SearchResult
from src.agentrag.agent.service import AgentService
from src.agentrag.retrieval.elasticsearch_retriever import ElasticsearchRetriever

router = APIRouter(prefix="/search")


def _hit_to_result(hit: dict) -> dict:
    now = datetime.now(timezone.utc).isoformat()
    return SearchResult(
        id=str(hit.get("id", "")),
        title=hit.get("document_title"),
        parent_id=None,
        final_score=float(hit.get("score") or 0.0),
        type="source",
        source_type=hit.get("segment_type", "text"),
        created=now,
        updated=now,
    ).model_dump()


@router.post("")
async def search(body: SearchRequest):
    retriever = ElasticsearchRetriever()
    mode = "hybrid" if body.type == "vector" else "sparse"
    result = await retriever.search(query=body.query, mode=mode, top_k=body.limit)
    hits = result.get("results", [])
    filtered = [h for h in hits if float(h.get("score") or 0.0) >= body.minimum_score]
    return SearchResponse(
        results=[_hit_to_result(h) for h in filtered],
        total_count=len(filtered),
        search_type=body.type,
    ).model_dump()


@router.post("/ask")
async def ask_sse(body: AskRequest, request: Request):
    """SSE streaming answer — open-notebook format."""
    agent = AgentService()

    async def event_stream():
        yield json.dumps({"type": "strategy", "reasoning": "Searching knowledge base...", "search_terms": [body.question]}) + "\n"

        full_tokens: list[str] = []
        last_done: dict = {}

        async for chunk in agent.chat_stream(
            question=body.question,
            document_title=None,
            chat_history=[],
            conversation_id=None,
        ):
            if chunk.startswith("event: token"):
                data_line = chunk.split("data: ", 1)[-1].strip()
                try:
                    token = json.loads(data_line).get("text", "")
                    full_tokens.append(token)
                    yield json.dumps({"type": "answer", "content": token}) + "\n"
                except Exception:
                    pass
            elif chunk.startswith("event: done"):
                data_line = chunk.split("data: ", 1)[-1].strip()
                try:
                    last_done = json.loads(data_line)
                except Exception:
                    pass

        final = "".join(full_tokens) or last_done.get("answer", "")
        yield json.dumps({"type": "final_answer", "content": final}) + "\n"
        yield json.dumps({"type": "complete"}) + "\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/plain",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.post("/ask/simple")
async def ask_simple(body: AskRequest):
    agent = AgentService()
    result = await agent.chat(
        question=body.question,
        document_title=None,
        chat_history=[],
        conversation_id=None,
    )
    return {
        "answer": result.get("answer", ""),
        "citations": result.get("citations", []),
    }
