"""S3 — EmbeddingService TTL cache behaviour."""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest


@pytest.fixture
def svc():
    """EmbeddingService backed by a counting mock provider."""
    fake = MagicMock()
    fake.embed = AsyncMock(side_effect=lambda texts: [[float(i)] for i, _ in enumerate(texts)])
    with patch("src.agentrag.services.embedding_service.build_embedding_provider", return_value=fake):
        from src.agentrag.services.embedding_service import EmbeddingService
        s = EmbeddingService(cache_size=64, cache_ttl_s=60, cache_max_batch=4)
    return s, fake


@pytest.mark.asyncio
async def test_first_call_misses_then_caches(svc):
    s, fake = svc
    out1 = await s.embed(["alpha"])
    out2 = await s.embed(["alpha"])
    assert out1 == out2
    fake.embed.assert_awaited_once_with(["alpha"])
    stats = s.cache_stats
    assert stats["hits"] == 1
    assert stats["misses"] == 1


@pytest.mark.asyncio
async def test_mixed_batch_only_misses_go_to_provider(svc):
    s, fake = svc
    await s.embed(["a", "b"])      # both miss
    fake.embed.reset_mock()
    out = await s.embed(["a", "c", "b"])  # only 'c' misses
    fake.embed.assert_awaited_once_with(["c"])
    assert len(out) == 3
    stats = s.cache_stats
    assert stats["hits"] == 2
    assert stats["misses"] == 3


@pytest.mark.asyncio
async def test_large_batch_bypasses_cache(svc):
    s, fake = svc  # cache_max_batch=4
    texts = ["t%d" % i for i in range(10)]
    await s.embed(texts)
    fake.embed.assert_awaited_once_with(texts)
    assert s.cache_stats["skips"] == 10
    assert s.cache_stats["hits"] == 0


@pytest.mark.asyncio
async def test_reset_cache_clears_state(svc):
    s, fake = svc
    await s.embed(["x"])
    s.reset_cache()
    fake.embed.reset_mock()
    await s.embed(["x"])
    fake.embed.assert_awaited_once_with(["x"])
