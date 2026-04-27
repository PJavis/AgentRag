from __future__ import annotations

from typing import Any

from src.agentrag.agent.context import ContextAssembler


class ContextAssemblyService:
    """
    Context Assembly Engine (service facade).
    """

    def __init__(self):
        self._assembler = ContextAssembler()

    def assemble(
        self,
        question: str,
        tool_outputs: list[dict[str, Any]],
    ) -> dict[str, Any]:
        return self._assembler.assemble(question, tool_outputs)
