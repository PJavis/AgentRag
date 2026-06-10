import asyncio
from src.agentrag.ingestion.raptor import RaptorBuilder


class _FakeGateway:
    async def text_response(self, system_prompt, user_prompt, task="general"):  # noqa: ANN001
        return "SUMMARY"


class _FakeEmbedder:
    async def embed(self, texts):  # noqa: ANN001
        return [[float(len(t) % 7), 1.0, 0.0] for t in texts]


def _leaves(n):
    return [
        {"content": f"leaf {i} about topic {i % 3}", "content_hash": f"h{i}",
         "embedding": [float(i % 3), 1.0, 0.0], "system_tag": "tim_mach",
         "specialty_tag": ["noi"], "node_level": 0}
        for i in range(n)
    ]


def test_skips_when_too_few_leaves(monkeypatch):
    from src.agentrag.ingestion import raptor as R
    monkeypatch.setattr(R.settings, "RAPTOR_MIN_LEAVES", 8)
    builder = RaptorBuilder(_FakeGateway(), _FakeEmbedder())
    out = asyncio.run(builder.build(_leaves(5), "Doc"))
    assert out == []  # no summary nodes for tiny docs


def test_builds_summary_nodes_with_level_and_children(monkeypatch):
    from src.agentrag.ingestion import raptor as R
    monkeypatch.setattr(R.settings, "RAPTOR_MIN_LEAVES", 4)
    monkeypatch.setattr(R.settings, "RAPTOR_MAX_LEVELS", 2)
    monkeypatch.setattr(R.settings, "RAPTOR_CLUSTER_SIZE", 3)
    builder = RaptorBuilder(_FakeGateway(), _FakeEmbedder())
    out = asyncio.run(builder.build(_leaves(12), "Doc"))
    assert out, "expected at least one summary node"
    for node in out:
        assert node["node_level"] >= 1
        assert node["segment_type"] == "raptor_summary"
        assert node["child_ids"]                 # links to children
        assert node["embedding"]                  # embedded
        assert node["content"] == "SUMMARY"
        assert node["system_tag"] == "tim_mach"   # propagated from children
