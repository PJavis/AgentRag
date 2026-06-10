import asyncio
from src.agentrag.ingestion.contextualizer import Contextualizer


class _FakeGateway:
    def __init__(self):
        self.calls = 0

    async def text_response(self, system_prompt, user_prompt, task="general"):  # noqa: ANN001
        self.calls += 1
        return f"CTX for: {user_prompt[:20]}"


def test_contextualize_sets_context_text_on_each_chunk(tmp_path):
    gw = _FakeGateway()
    ctx = Contextualizer(gw, cache_dir=str(tmp_path))
    chunks = [
        {"content": "Nhồi máu cơ tim là ...", "content_hash": "h1"},
        {"content": "Điều trị bằng aspirin ...", "content_hash": "h2"},
    ]
    out = asyncio.run(ctx.contextualize_chunks("Whole doc text", chunks, "Tim mạch"))
    assert all(c.get("context_text") for c in out)
    assert gw.calls == 2


def test_contextualize_uses_cache_on_second_run(tmp_path):
    gw = _FakeGateway()
    ctx = Contextualizer(gw, cache_dir=str(tmp_path))
    chunks = [{"content": "x", "content_hash": "h1"}]
    asyncio.run(ctx.contextualize_chunks("doc", chunks, "T"))
    first_calls = gw.calls
    # Fresh chunk dict, same hash → served from disk cache, no new LLM call.
    chunks2 = [{"content": "x", "content_hash": "h1"}]
    asyncio.run(ctx.contextualize_chunks("doc", chunks2, "T"))
    assert gw.calls == first_calls
    assert chunks2[0]["context_text"]
