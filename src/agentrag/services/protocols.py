"""Execution Plane service Protocols (S4).

These Protocol types define the contract Reasoning Plane code uses to
talk to the Execution Plane. Concrete classes in `ingestion/` and
`retrieval/` implicitly satisfy these — no inheritance required.

Reasoning code MUST type its dependencies against these protocols, never
against concrete implementations.
"""
from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class EmbeddingProtocol(Protocol):
    """Embed text(s) into dense vectors."""

    async def embed(self, texts: list[str]) -> list[list[float]]: ...


@runtime_checkable
class VisionProtocol(Protocol):
    """Vision LLM description for image bytes or files."""

    async def describe(
        self,
        image_bytes: bytes,
        mime: str = "image/jpeg",
        context: str = "",
    ) -> str: ...


@runtime_checkable
class RetrievalProtocol(Protocol):
    """Hybrid retrieval entry point (filters-only; routing is Reasoning's job)."""

    async def search(
        self,
        query: str,
        *,
        document_title: str | None = None,
        top_k: int | None = None,
        mode: str = "hybrid_kg",
        rerank: bool | None = None,
        dense_query: str | None = None,
        filters: dict[str, Any] | None = None,
    ) -> dict[str, Any]: ...


@runtime_checkable
class StorageProtocol(Protocol):
    """CRUD facade over Postgres + Elasticsearch document stores."""

    async def get_chunks_by_hashes(self, content_hashes: list[str]) -> list[dict[str, Any]]: ...
    async def list_documents(self) -> list[dict[str, Any]]: ...


@runtime_checkable
class RerankerProtocol(Protocol):
    """Rerank a list of hits given the query."""

    async def rerank(
        self,
        query: str,
        hits: list[dict[str, Any]],
        top_k: int | None = None,
    ) -> list[dict[str, Any]]: ...


@runtime_checkable
class LLMProtocol(Protocol):
    """Minimal LLM gateway contract used by Reasoning Plane."""

    async def chat_completion(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        task: str = "default",
        temperature: float | None = None,
    ) -> tuple[str, float]: ...

    async def json_response(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        task: str = "default",
        temperature: float | None = None,
    ) -> tuple[dict[str, Any], float]: ...
