import asyncio
import json
from src.agentrag.agent import graph_service as gs


class _FakeGraph:
    async def ainvoke(self, initial, config):  # noqa: ANN001
        return {
            "answer": "Nhồi máu cơ tim là tắc động mạch vành.",
            "citations": [{"source": 1, "document_title": "Tim mạch"}],
            "highlights": [],
            "reasoning_path": "fast",
            "sql_query": None,
            "tool_trace": [{"tool_name": "search_hybrid_kg", "tool_input": {"q": "x"},
                            "tool_output": {"semantic_cache_hit": True, "mode": "hybrid_kg"}}],
        }


def _collect(agen):
    async def run():
        out = []
        async for f in agen:
            out.append(f)
        return out
    return asyncio.run(run())


def _parse(frames):
    evs = []
    for f in frames:
        lines = f.strip().split("\n")
        ev = lines[0].split("event: ", 1)[1]
        data = json.loads(lines[1].split("data: ", 1)[1])
        evs.append((ev, data))
    return evs


def test_chat_stream_runs_graph_and_streams_answer(monkeypatch):
    monkeypatch.setattr(gs, "_GRAPH", _FakeGraph())
    svc = gs.GraphAgentService()
    frames = _collect(svc.chat_stream(question="Triệu chứng nhồi máu cơ tim?", conversation_id="c1"))
    evs = _parse(frames)
    kinds = [e for e, _ in evs]
    assert "status" in kinds
    # token frames concatenate to the graph's answer
    answer = "".join(d["text"] for e, d in evs if e == "token")
    assert answer == "Nhồi máu cơ tim là tắc động mạch vành."
    # done frame carries the graph's citations + reasoning_path
    done = [d for e, d in evs if e == "done"][0]
    assert done["reasoning_path"] == "fast"
    assert done["citations"] == [{"source": 1, "document_title": "Tim mạch"}]
    assert done.get("semantic_cache_hit") is True  # from _message_signals


def test_chat_stream_emits_error_frame_on_failure(monkeypatch):
    class _Boom:
        async def ainvoke(self, initial, config):  # noqa: ANN001
            raise RuntimeError("graph blew up")
    monkeypatch.setattr(gs, "_GRAPH", _Boom())
    svc = gs.GraphAgentService()
    frames = _collect(svc.chat_stream(question="x", conversation_id="c2"))
    evs = _parse(frames)
    assert any(e == "error" and "graph blew up" in d.get("message", "") for e, d in evs)
