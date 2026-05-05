from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from src.agentrag.services.llm_gateway import LLMGateway

from src.agentrag.structured.sql_engine import ProvenanceRecord, SQLEngineOutput

_VI_RE = re.compile(
    r"[àáảãạăắặằẳẵâấầẩẫậèéẻẽẹêếềệểễìíỉĩịòóỏõọôốồổỗộơớờởỡợùúủũụưứừửữựỳýỷỹỵđ]",
    re.IGNORECASE,
)

_SYNTH_SYSTEM_TEMPLATE = """\
{lang_instruction}
You are an answer synthesis assistant. Given a question and structured query results with their sources, \
generate a clear, concise natural language answer.

Rules:
- Base your answer ONLY on the provided query results.
- Be direct and specific — include actual numbers, names, and comparisons from the results.
- Return JSON: {{"answer": str, "citations": list}}
- Each citation must include: {{"document_title": str, "section_path": str, "content_hash": str}}
- Only cite sources that actually appear in the provenance list.
"""


@dataclass
class SynthesizerOutput:
    answer: str
    citations: list[dict[str, Any]]
    sql_result_summary: str


class AnswerSynthesizer:
    """
    Tổng hợp câu trả lời tự nhiên từ SQL result + provenance chain.
    Mỗi claim đều traceable về source document.
    """

    def __init__(self, llm_gateway: LLMGateway) -> None:
        self._llm = llm_gateway

    async def synthesize(
        self,
        question: str,
        sql_result: SQLEngineOutput,
        candidate_chunks: list[dict[str, Any]],
        query_type: str,
        chat_history: list[dict[str, Any]] | None = None,
    ) -> SynthesizerOutput:
        lang_instruction = (
            "Ngôn ngữ phản hồi: Tiếng Việt. Toàn bộ câu trả lời PHẢI bằng tiếng Việt."
            if _VI_RE.search(question)
            else "Response language: English."
        )
        system_prompt = _SYNTH_SYSTEM_TEMPLATE.format(lang_instruction=lang_instruction)

        formatted = self._format_result(sql_result.result_rows, query_type)
        provenance_context = self._build_provenance_context(
            sql_result.provenance, candidate_chunks
        )

        user_prompt = json.dumps(
            {
                "question": question,
                "query_type": query_type,
                "sql_results": formatted,
                "provenance": provenance_context,
            },
            ensure_ascii=False,
        )

        try:
            raw, _latency = await self._llm.json_response(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                task="synthesize",
            )
        except Exception as exc:
            # Fallback: trả kết quả SQL thô
            return SynthesizerOutput(
                answer=f"Kết quả truy vấn:\n{formatted}",
                citations=[],
                sql_result_summary=formatted,
            )

        answer = str(raw.get("answer", "")).strip() or formatted
        raw_citations = raw.get("citations") or []

        # Ground citations: chỉ giữ citations có trong provenance
        allowed_hashes = {p.source_chunk_hash for p in sql_result.provenance if p.source_chunk_hash}
        citations: list[dict[str, Any]] = []
        for c in raw_citations:
            if not isinstance(c, dict):
                continue
            ch = c.get("content_hash", "")
            if ch in allowed_hashes:
                citations.append({
                    "document_title": c.get("document_title", ""),
                    "section_path": c.get("section_path", ""),
                    "content_hash": ch,
                })

        return SynthesizerOutput(
            answer=answer,
            citations=citations,
            sql_result_summary=formatted,
        )

    # ── Formatting ────────────────────────────────────────────────────────────

    def _format_result(self, rows: list[dict[str, Any]], query_type: str) -> str:
        if not rows:
            return "Không tìm thấy kết quả."

        if query_type in ("comparison", "ranking"):
            return self._to_markdown_table(rows)

        if query_type == "aggregation":
            # Single value hoặc key-value pairs
            if len(rows) == 1:
                row = rows[0]
                if len(row) == 1:
                    val = next(iter(row.values()))
                    return str(val)
                return ", ".join(f"{k}={v}" for k, v in row.items())
            return self._to_markdown_table(rows)

        # multi_filter, multi_hop, default → numbered list
        parts = []
        for i, row in enumerate(rows, 1):
            parts.append(f"{i}. " + " | ".join(f"{k}: {v}" for k, v in row.items() if not k.startswith("_")))
        return "\n".join(parts)

    @staticmethod
    def _to_markdown_table(rows: list[dict[str, Any]]) -> str:
        if not rows:
            return ""
        # Lọc bỏ _source_* columns
        cols = [k for k in rows[0] if not k.startswith("_")]
        header = "| " + " | ".join(cols) + " |"
        sep = "| " + " | ".join("---" for _ in cols) + " |"
        data_rows = []
        for row in rows:
            data_rows.append("| " + " | ".join(str(row.get(c, "")) for c in cols) + " |")
        return "\n".join([header, sep] + data_rows)

    def _build_provenance_context(
        self,
        provenance: list[ProvenanceRecord],
        candidate_chunks: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """
        Join ProvenanceRecord với candidate_chunks trên content_hash.
        Trả list dùng làm context cho LLM synthesis.
        """
        chunk_map = {
            c.get("content_hash", ""): c
            for c in candidate_chunks
            if c.get("content_hash")
        }
        result: list[dict[str, Any]] = []
        seen_hashes: set[str] = set()

        for prov in provenance:
            h = prov.source_chunk_hash
            if not h or h in seen_hashes:
                continue
            seen_hashes.add(h)
            chunk = chunk_map.get(h)
            if chunk:
                result.append({
                    "content_hash": h,
                    "document_title": chunk.get("document_title", prov.source_doc),
                    "section_path": chunk.get("section_path", prov.source_section),
                    "excerpt": (chunk.get("content") or "")[:400],
                })
            elif prov.source_doc:
                result.append({
                    "content_hash": h,
                    "document_title": prov.source_doc,
                    "section_path": prov.source_section,
                    "excerpt": "",
                })
        return result


if __name__ == "__main__":
    synth = AnswerSynthesizer.__new__(AnswerSynthesizer)

    # Test _format_result comparison
    rows = [
        {"name": "A", "rating": "4.5"},
        {"name": "B", "rating": "3.8"},
    ]
    table = synth._format_result(rows, "comparison")
    assert "| name | rating |" in table
    assert "A" in table and "B" in table
    print("[OK] _format_result comparison → markdown table")

    # Test _format_result aggregation single value
    agg = synth._format_result([{"count": 42}], "aggregation")
    assert agg == "42"
    print("[OK] _format_result aggregation → single value")

    # Test _to_markdown_table
    result = synth._to_markdown_table([{"col1": "v1", "_source_doc": "hidden"}])
    assert "_source_doc" not in result
    assert "col1" in result
    print("[OK] _to_markdown_table strips _source_* columns")

    # Test _build_provenance_context
    from src.agentrag.structured.sql_engine import ProvenanceRecord
    prov = [ProvenanceRecord(0, "hash1", "Doc1", "sec1")]
    chunks = [{"content_hash": "hash1", "document_title": "Doc1", "section_path": "sec1", "content": "some text"}]
    ctx = synth._build_provenance_context(prov, chunks)
    assert len(ctx) == 1
    assert ctx[0]["content_hash"] == "hash1"
    assert "some text" in ctx[0]["excerpt"]
    print("[OK] _build_provenance_context")

    print("[ALL PASS] AnswerSynthesizer unit tests")
