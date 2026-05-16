"""S4 plane-split acceptance tests.

Verifies:
  - Concrete services satisfy their Protocol contracts.
  - ServiceContainer returns stable instances (singleton-per-process).
  - Container.override() injects mocks.
  - FederatedRetriever no longer auto-constructs DomainRouter.
  - RetrievalService merges filters → overrides.
  - Reasoning helpers (services/reasoning_knowledge.py) are pure.
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.agentrag.services import protocols as P
from src.agentrag.services.container import (
    ServiceContainer,
    get_container,
    reset_container,
)
from src.agentrag.services.retrieval_service import RetrievalService
from src.agentrag.services import reasoning_knowledge as RK


# -- Protocol satisfaction (structural, runtime_checkable) -------------------

def test_embedding_service_satisfies_protocol():
    with patch("src.agentrag.services.embedding_service.build_embedding_provider") as bf:
        bf.return_value = MagicMock(embed=AsyncMock(return_value=[]))
        from src.agentrag.services.embedding_service import EmbeddingService
        svc = EmbeddingService()
    assert isinstance(svc, P.EmbeddingProtocol)


def test_storage_service_satisfies_protocol():
    from src.agentrag.services.storage_service import StorageService
    svc = StorageService(es_store=MagicMock())
    assert isinstance(svc, P.StorageProtocol)


def test_retrieval_service_satisfies_protocol():
    svc = RetrievalService(base=MagicMock())
    assert isinstance(svc, P.RetrievalProtocol)


# -- ServiceContainer --------------------------------------------------------

def test_container_singleton_per_process():
    reset_container()
    a = get_container()
    b = get_container()
    assert a is b


def test_container_lazy_instantiation():
    """Properties must not trigger construction at __init__ time."""
    c = ServiceContainer()
    # Internal slots all None until accessed
    assert c._llm is None
    assert c._embedding is None
    assert c._retrieval is None


def test_container_override_injects_mock():
    c = ServiceContainer()
    fake = MagicMock(name="fake-retrieval")
    c.override(retrieval=fake)
    assert c.retrieval is fake


def test_container_override_rejects_unknown_service():
    c = ServiceContainer()
    with pytest.raises(KeyError):
        c.override(nonexistent_service=MagicMock())


# -- FederatedRetriever no longer auto-builds router -------------------------

def test_federated_default_no_router():
    """S4 contract: default FederatedRetriever() ships without a router."""
    from src.agentrag.retrieval.federated import FederatedRetriever
    fr = FederatedRetriever(base=MagicMock())
    assert fr._router is None


@pytest.mark.asyncio
async def test_federated_no_router_no_routing():
    """Without injected router, no override → no domain filtering happens."""
    from src.agentrag.retrieval.federated import FederatedRetriever
    base = MagicMock()
    base.search = AsyncMock(return_value={"hits": []})
    fr = FederatedRetriever(base=base, router=None)
    out = await fr.search(query="đau ngực", mode="hybrid")
    assert "domain_route" not in out
    assert base.search.await_args.kwargs.get("filters") is None


# -- RetrievalService filter merging -----------------------------------------

@pytest.mark.asyncio
async def test_retrieval_service_filters_to_overrides():
    """Generic filters dict → system_override + specialty_override."""
    base = MagicMock()
    base.search = AsyncMock(return_value={"hits": []})
    svc = RetrievalService(base=base)
    await svc.search(
        query="x",
        mode="hybrid",
        filters={"systems": ["tim_mach"], "specialties": ["noi"]},
    )
    # base.search receives `filters` (built by FederatedRetriever)
    call = base.search.await_args
    assert call.kwargs["filters"] == {
        "systems": ["tim_mach"],
        "specialties": ["noi"],
    }


@pytest.mark.asyncio
async def test_retrieval_service_explicit_override_wins():
    """Direct override beats filters dict if both supplied."""
    base = MagicMock()
    base.search = AsyncMock(return_value={"hits": []})
    svc = RetrievalService(base=base)
    await svc.search(
        query="x",
        mode="hybrid",
        filters={"systems": ["ho_hap"]},
        system_override="tim_mach",
    )
    assert base.search.await_args.kwargs["filters"] == {"systems": ["tim_mach"]}


# -- Reasoning helpers are pure (no IO) --------------------------------------

def test_reasoning_knowledge_expand_query_pure():
    class _Intent:
        intent = "structured"
        query_type = "aggregation"
    out = RK.expand_query("how many", _Intent())
    assert "count" in out
    assert "how many" in out


def test_reasoning_knowledge_semantic_no_expansion():
    class _Intent:
        intent = "semantic"
        query_type = None
    assert RK.expand_query("what is X", _Intent()) == "what is X"


def test_reasoning_knowledge_mode_to_tool():
    assert RK.mode_to_tool("hybrid_kg") == "search_hybrid_kg"
    assert RK.mode_to_tool("dense") == "search_dense"
    assert RK.mode_to_tool("unknown") == "search_hybrid_kg"


def test_reasoning_knowledge_normalize_tool_call_fallback():
    name, inp = RK.normalize_tool_call(
        tool_name="nonexistent",
        tool_input={"foo": "bar"},
        question="what is X?",
        document_title="doc",
        valid_tools={"search_hybrid_kg", "search_dense"},
    )
    assert name == "search_hybrid_kg"
    assert inp["query"] == "what is X?"
    assert inp["document_title"] == "doc"


def test_reasoning_knowledge_normalize_tool_call_passthrough():
    name, inp = RK.normalize_tool_call(
        tool_name="search_dense",
        tool_input={"query": "x"},
        question="x",
        document_title="doc",
        valid_tools={"search_dense"},
    )
    assert name == "search_dense"
    assert inp["document_title"] == "doc"
