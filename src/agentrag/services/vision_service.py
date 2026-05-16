"""VisionService — Execution Plane facade for vision LLM (S4).

Thin wrapper around `ImageParser`. Reasoning code never constructs the
parser directly; it asks the container for a `VisionService`.

If vision is disabled (`VISION_PROVIDER` unset), `enabled` is False and
`describe` raises — Reasoning code is expected to check `enabled` first.
"""
from __future__ import annotations

from src.agentrag.config import Settings, settings as global_settings
from src.agentrag.ingestion.parsers.image_parser import ImageParser
from src.agentrag.services.llm_gateway import LLMGateway


class VisionService:
    def __init__(
        self,
        llm_gateway: LLMGateway,
        settings: Settings | None = None,
    ) -> None:
        cfg = settings or global_settings
        self._enabled = bool(cfg.VISION_PROVIDER and cfg.VISION_MODEL)
        self._parser: ImageParser | None = (
            ImageParser(llm_gateway) if self._enabled else None
        )

    @property
    def enabled(self) -> bool:
        return self._enabled

    async def describe(
        self,
        image_bytes: bytes,
        mime: str = "image/jpeg",
        context: str = "",
    ) -> str:
        if not self._parser:
            raise RuntimeError("VisionService disabled (VISION_PROVIDER unset)")
        return await self._parser.describe(image_bytes, mime, context)

    async def parse_file(
        self,
        file_path: str,
        document_title: str | None = None,
    ) -> dict:
        if not self._parser:
            raise RuntimeError("VisionService disabled (VISION_PROVIDER unset)")
        return await self._parser.parse(file_path, document_title)
