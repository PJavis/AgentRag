"""Query rewriting utilities: HyDE, decomposition, and reflection-driven reformulation.

HyDE (Hypothetical Document Embeddings):
    Generate a short hypothetical answer → embed it for kNN search → better semantic match
    when the user's query vocabulary differs from the document vocabulary (e.g., Vietnamese
    query against an English Excel table).

Query Decomposition:
    Break a complex multi-hop question into simpler sub-questions, each answered by one
    retrieval step. The agent loop handles execution sequentially.
"""
from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from src.agentrag.services.llm_gateway import LLMGateway

logger = logging.getLogger(__name__)

_HYDE_SYSTEM = """\
You are a search assistant. Given a question, write a SHORT hypothetical document passage \
(2-4 sentences, ≤80 words) that would directly answer it.

Rules:
- Write as if it is an excerpt from the actual source document.
- Use specific terminology, numbers, and entity names the document would use.
- Do NOT frame it as "The answer is...". Write it as document text.
- If the question is in Vietnamese but the data is likely in English (e.g., game data tables), \
  write the hypothetical passage in English so it matches the document vocabulary.

Return JSON: {"hypothetical": "<passage text>"}
"""

_DECOMPOSE_SYSTEM = """\
You are a query planner. Decompose a complex question into 2-4 independent sub-questions \
that can each be answered with a single retrieval step.

Rules:
- Each sub-question must be simpler and more specific than the original.
- Order them so later sub-questions can use answers from earlier ones.
- If the question is simple enough for one retrieval, return just the original question.

Return JSON: {"sub_questions": ["q1", "q2", ...]}
"""


class QueryRewriter:
    """LLM-powered query rewriting for improved retrieval quality."""

    def __init__(self, llm_gateway: LLMGateway) -> None:
        self._llm = llm_gateway

    async def make_hyde_text(self, question: str) -> str | None:
        """Generate a hypothetical document passage for kNN embedding (HyDE).

        Returns the hypothetical text, or None if generation fails.
        """
        try:
            raw, _ = await self._llm.json_response(
                system_prompt=_HYDE_SYSTEM,
                user_prompt=json.dumps({"question": question}, ensure_ascii=False),
                task="decide",  # reuse decide task slot (lightweight LLM call)
                temperature=0.0,  # deterministic: same question must not flip retrieval run-to-run
            )
            text = (raw.get("hypothetical") or "").strip()
            if text:
                logger.debug("HyDE generated: %s", text[:120])
            return text or None
        except Exception as exc:
            logger.debug("HyDE generation failed (non-fatal): %s", exc)
            return None

    async def decompose(self, question: str) -> list[str]:
        """Decompose a complex question into ordered sub-questions.

        Returns a list with at least the original question as fallback.
        """
        try:
            raw, _ = await self._llm.json_response(
                system_prompt=_DECOMPOSE_SYSTEM,
                user_prompt=json.dumps({"question": question}, ensure_ascii=False),
                task="decide",
                temperature=0.0,  # deterministic: same question must not flip retrieval run-to-run
            )
            subs = raw.get("sub_questions")
            if isinstance(subs, list) and subs:
                # Filter empty strings, cap at 4 sub-questions
                cleaned = [str(q).strip() for q in subs if str(q).strip()][:4]
                if cleaned:
                    logger.debug("Decomposed into %d sub-questions", len(cleaned))
                    return cleaned
        except Exception as exc:
            logger.debug("Query decomposition failed (non-fatal): %s", exc)
        return [question]

    def augment_with_hyde(self, original_query: str, hyde_text: str) -> str:
        """Append hypothetical text to the query string.

        This is "soft HyDE": both BM25 and kNN benefit from the expanded vocabulary.
        True HyDE (kNN-only) would require a separate embedding path in the retriever.
        """
        return f"{original_query}\n{hyde_text}"
