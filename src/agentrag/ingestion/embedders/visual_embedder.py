"""Visual embedder — CLIP-based cross-modal text↔image embedding.

Used to:
  1. Embed image bytes at ingest → store in ES segment.image_embedding.
  2. Embed text query at search → kNN against image_embedding.

Both vectors live in the same CLIP semantic space, so a Vietnamese text
query like "X-quang phổi viêm" can directly match an image vector.

Lazy-loaded singleton: HuggingFace model downloads on first use (~600MB
for clip-ViT-B-32-multilingual-v1). After that, runs locally on CPU or
GPU per `VISUAL_EMBEDDING_DEVICE`.
"""
from __future__ import annotations

import io
import logging
import threading
from typing import Any

from src.agentrag.config import settings

logger = logging.getLogger(__name__)


class VisualEmbedder:
    """Singleton wrapper around sentence-transformers CLIP model."""

    _instance: "VisualEmbedder | None" = None
    _lock = threading.Lock()

    @classmethod
    def get(cls) -> "VisualEmbedder":
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls()
        return cls._instance

    def __init__(self) -> None:
        if not settings.VISUAL_EMBEDDING_ENABLED:
            raise RuntimeError("VISUAL_EMBEDDING_ENABLED=false")
        self._model = None
        self._device = settings.VISUAL_EMBEDDING_DEVICE
        self._model_name = settings.VISUAL_EMBEDDING_MODEL

    def _load(self):
        if self._model is not None:
            return self._model
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "sentence-transformers not installed (needed for visual embedding)"
            ) from exc
        device = self._device
        if device == "auto":
            try:
                import torch  # type: ignore
                device = "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                device = "cpu"
        logger.info(
            "VisualEmbedder: loading %s on %s", self._model_name, device,
        )
        self._model = SentenceTransformer(self._model_name, device=device)
        return self._model

    def embed_images(self, images: list[bytes]) -> list[list[float]]:
        """Embed PIL-decodable image bytes. Returns list of `dims` floats per image."""
        if not images:
            return []
        try:
            from PIL import Image  # type: ignore
        except ImportError as exc:
            raise ImportError("Pillow required for visual embedding") from exc
        model = self._load()
        pil_images = []
        for b in images:
            try:
                img = Image.open(io.BytesIO(b)).convert("RGB")
            except Exception as exc:
                logger.warning("VisualEmbedder: bad image bytes: %s", exc)
                # Fallback to zero vector so caller alignment stays.
                pil_images.append(None)
                continue
            pil_images.append(img)
        valid_idx = [i for i, p in enumerate(pil_images) if p is not None]
        valid_imgs = [pil_images[i] for i in valid_idx]
        if not valid_imgs:
            return [[0.0] * settings.VISUAL_EMBEDDING_DIMS for _ in images]
        vecs = model.encode(valid_imgs, convert_to_numpy=True, show_progress_bar=False)
        out: list[list[float]] = [[] for _ in images]
        zeros = [0.0] * settings.VISUAL_EMBEDDING_DIMS
        for i, idx in enumerate(valid_idx):
            out[idx] = vecs[i].tolist()
        for j, _ in enumerate(images):
            if not out[j]:
                out[j] = zeros
        return out

    def embed_text(self, text: str) -> list[float]:
        """Embed a query string into the CLIP text-encoder space (same space
        as the image vectors above)."""
        if not text:
            return [0.0] * settings.VISUAL_EMBEDDING_DIMS
        model = self._load()
        vec = model.encode([text], convert_to_numpy=True, show_progress_bar=False)
        return vec[0].tolist()
