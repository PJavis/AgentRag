from __future__ import annotations

import logging
from hashlib import sha256
from typing import TYPE_CHECKING, Any

from src.agentrag.config import settings

if TYPE_CHECKING:
    from src.agentrag.services.llm_gateway import LLMGateway
    from src.agentrag.ingestion.embedders.base import BaseEmbeddingProvider

logger = logging.getLogger(__name__)

_SUMMARY_SYSTEM = """\
You write a faithful, self-contained summary of several related passages from a
medical document. Capture the shared topic and the key facts/relationships.
Same language as the passages. Output ONLY the summary text, no preamble."""


def _cluster_indices(vectors: list[list[float]], n_clusters: int) -> list[list[int]]:
    """Return groups of row indices via UMAP->GaussianMixture hard assignment.
    Falls back to contiguous chunking if reduction/fit fails or n is tiny."""
    n = len(vectors)
    if n_clusters <= 1 or n <= n_clusters:
        return [list(range(n))]
    try:
        import numpy as np
        import umap
        from sklearn.mixture import GaussianMixture

        arr = np.asarray(vectors, dtype="float32")
        n_components = min(10, max(2, n - 2))
        reduced = umap.UMAP(
            n_neighbors=min(15, n - 1), n_components=n_components, metric="cosine",
        ).fit_transform(arr)
        gm = GaussianMixture(n_components=n_clusters, random_state=42)
        labels = gm.fit_predict(reduced)
        groups: dict[int, list[int]] = {}
        for idx, lab in enumerate(labels):
            groups.setdefault(int(lab), []).append(idx)
        return [g for g in groups.values() if g]
    except Exception as exc:  # numerical / convergence issues -> contiguous split
        logger.warning("RAPTOR clustering fell back to contiguous: %s", exc)
        size = max(1, n // n_clusters)
        return [list(range(i, min(i + size, n))) for i in range(0, n, size)]


class RaptorBuilder:
    """WS2 — build a collapsed RAPTOR tree: recursively cluster node embeddings,
    summarize each cluster, embed the summary, and emit summary nodes carrying
    `node_level`, `child_ids`, and domain tags propagated (union) from children.
    Returned nodes are appended to the same `agentrag_segments` index."""

    def __init__(self, gateway: "LLMGateway", embedder: "BaseEmbeddingProvider") -> None:
        self._gateway = gateway
        self._embedder = embedder

    async def build(
        self, leaf_chunks: list[dict[str, Any]], document_title: str
    ) -> list[dict[str, Any]]:
        if len(leaf_chunks) < settings.RAPTOR_MIN_LEAVES:
            return []
        summary_nodes: list[dict[str, Any]] = []
        current = leaf_chunks
        for level in range(1, settings.RAPTOR_MAX_LEVELS + 1):
            vectors = [c.get("embedding") for c in current]
            if any(v is None for v in vectors):
                break
            n_clusters = max(2, len(current) // max(settings.RAPTOR_CLUSTER_SIZE, 2))
            groups = _cluster_indices(vectors, n_clusters)
            if len(groups) >= len(current):  # no compression -> stop
                break
            level_nodes: list[dict[str, Any]] = []
            for group in groups:
                members = [current[i] for i in group]
                node = await self._summarize_group(members, document_title, level)
                if node is not None:
                    level_nodes.append(node)
            if not level_nodes:
                break
            # Embed this level's summaries so the next level can cluster them.
            embeddings = await self._embedder.embed([n["content"] for n in level_nodes])
            for node, emb in zip(level_nodes, embeddings):
                node["embedding"] = emb
            summary_nodes.extend(level_nodes)
            current = level_nodes
            if len(current) <= 1:  # reached root
                break
        return summary_nodes

    async def _summarize_group(
        self, members: list[dict[str, Any]], document_title: str, level: int
    ) -> dict[str, Any] | None:
        joined = "\n\n---\n\n".join(m["content"] for m in members)
        try:
            summary = await self._gateway.text_response(
                system_prompt=_SUMMARY_SYSTEM,
                user_prompt=f"Document: {document_title}\n\nPassages:\n{joined}",
                task=settings.RAPTOR_SUMMARY_TASK,
            )
        except Exception as exc:
            logger.warning("RAPTOR summary failed (%s): %s", document_title, exc)
            return None
        summary = (summary or "").strip()
        if not summary:
            return None
        systems = {m.get("system_tag") for m in members if m.get("system_tag")}
        specialties: set[str] = set()
        for m in members:
            specialties.update(m.get("specialty_tag") or [])
        child_ids = [m["content_hash"] for m in members if m.get("content_hash")]
        return {
            "content": summary,
            "content_hash": sha256(summary.encode("utf-8")).hexdigest(),
            "segment_type": "raptor_summary",
            "node_level": level,
            "child_ids": child_ids,
            "section_path": f"{document_title} / summary L{level}",
            "position": None,
            "system_tag": next(iter(systems), None),
            "specialty_tag": sorted(specialties),
            "canonical_terms": [],
            "metadata": {"document_title": document_title, "raptor_level": level},
        }
