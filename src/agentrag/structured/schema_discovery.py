from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from src.agentrag.services.llm_gateway import LLMGateway

from src.agentrag.config import settings


@dataclass
class TableDef:
    name: str
    columns: list[str]
    primary_key: str | None
    description: str = ""


@dataclass
class JoinKey:
    from_table: str
    from_column: str
    to_table: str
    to_column: str


@dataclass
class RelationalSchema:
    tables: list[TableDef]
    join_keys: list[JoinKey]
    query_focus: list[str]       # columns quan trọng nhất với câu hỏi này
    source_chunks_used: list[str]  # content_hash của chunks dùng để suy schema
    is_empty: bool = False       # True nếu không suy ra được schema hợp lệ


_SYSTEM_PROMPT = """\
You are a schema inference expert. Given a natural language question and text samples from documents, \
infer the minimal relational schema needed to answer the question.

Rules:
- Infer only what is needed to answer the question — minimal schema.
- Each table must have a primary_key (the field that uniquely identifies a row).
- query_focus lists column names most relevant to answering the question.
- join_keys links foreign keys between tables when multiple tables are needed.
- If one table suffices, set join_keys to [].
- Column names should be simple snake_case strings.

Return JSON in this exact structure:
{
  "tables": [
    {
      "name": "table_name",
      "columns": ["col1", "col2", "col3"],
      "primary_key": "col1",
      "description": "what this table represents"
    }
  ],
  "join_keys": [
    {"from_table": "t1", "from_column": "fk", "to_table": "t2", "to_column": "pk"}
  ],
  "query_focus": ["col2", "col3"]
}

Examples:

Question: "So sánh rating của sản phẩm A, B, C"
Query type: comparison
Schema:
{
  "tables": [{"name": "product", "columns": ["name", "rating", "price"], "primary_key": "name", "description": "Product entities"}],
  "join_keys": [],
  "query_focus": ["name", "rating"]
}

Question: "Có bao nhiêu nhân viên ở mỗi phòng ban?"
Query type: aggregation
Schema:
{
  "tables": [{"name": "employee", "columns": ["id", "name", "department"], "primary_key": "id", "description": "Employee records"}],
  "join_keys": [],
  "query_focus": ["department"]
}
"""


class SchemaDiscoveryModule:
    """
    Suy ra minimal relational schema đặc thù cho câu hỏi.
    Một LLM call với top-N chunks làm context.
    """

    def __init__(self, llm_gateway: LLMGateway) -> None:
        self._llm = llm_gateway

    async def discover(
        self,
        question: str,
        query_type: str,
        candidate_chunks: list[dict[str, Any]],
        document_title: str | None = None,
        max_chunks: int | None = None,
    ) -> RelationalSchema:
        limit = max_chunks or settings.STRUCTURED_MAX_CHUNKS_FOR_SCHEMA
        top_chunks = candidate_chunks[:limit]

        if not top_chunks:
            return RelationalSchema(
                tables=[], join_keys=[], query_focus=[],
                source_chunks_used=[], is_empty=True,
            )

        chunk_texts = self._format_chunks(top_chunks)
        source_hashes = [c.get("content_hash", "") for c in top_chunks if c.get("content_hash")]

        user_prompt = json.dumps(
            {
                "question": question,
                "query_type": query_type,
                "document_title": document_title,
                "text_samples": chunk_texts,
            },
            ensure_ascii=False,
        )

        try:
            raw, _latency = await self._llm.json_response(
                system_prompt=_SYSTEM_PROMPT,
                user_prompt=user_prompt,
                task="schema_discovery",
            )
        except Exception:
            return RelationalSchema(
                tables=[], join_keys=[], query_focus=[],
                source_chunks_used=source_hashes, is_empty=True,
            )

        schema = self._parse_response(raw, source_hashes)
        return schema

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _format_chunks(chunks: list[dict[str, Any]]) -> list[dict[str, str]]:
        """Truncate mỗi chunk xuống 300 chars để tiết kiệm tokens."""
        result = []
        for chunk in chunks:
            content = (chunk.get("content") or "")[:300]
            result.append({
                "section": chunk.get("section_path", ""),
                "content": content,
            })
        return result

    @staticmethod
    def _parse_response(
        raw: dict[str, Any],
        source_hashes: list[str],
    ) -> RelationalSchema:
        try:
            tables_raw = raw.get("tables") or []
            tables: list[TableDef] = []
            for t in tables_raw:
                name = str(t.get("name", "")).strip()
                columns = [str(c) for c in (t.get("columns") or []) if c]
                pk = t.get("primary_key") or (columns[0] if columns else None)
                if name and columns:
                    tables.append(TableDef(
                        name=name,
                        columns=columns,
                        primary_key=pk,
                        description=str(t.get("description", "")),
                    ))

            join_keys: list[JoinKey] = []
            for jk in (raw.get("join_keys") or []):
                join_keys.append(JoinKey(
                    from_table=str(jk.get("from_table", "")),
                    from_column=str(jk.get("from_column", "")),
                    to_table=str(jk.get("to_table", "")),
                    to_column=str(jk.get("to_column", "")),
                ))

            query_focus = [str(c) for c in (raw.get("query_focus") or []) if c]

            is_empty = len(tables) == 0
            return RelationalSchema(
                tables=tables,
                join_keys=join_keys,
                query_focus=query_focus,
                source_chunks_used=source_hashes,
                is_empty=is_empty,
            )
        except Exception:
            return RelationalSchema(
                tables=[], join_keys=[], query_focus=[],
                source_chunks_used=source_hashes, is_empty=True,
            )


if __name__ == "__main__":
    raw_response = {
        "tables": [
            {
                "name": "product",
                "columns": ["name", "rating", "price"],
                "primary_key": "name",
                "description": "Product entities",
            }
        ],
        "join_keys": [],
        "query_focus": ["name", "rating"],
    }
    schema = SchemaDiscoveryModule._parse_response(raw_response, ["hash1", "hash2"])
    assert not schema.is_empty
    assert len(schema.tables) == 1
    assert schema.tables[0].name == "product"
    assert schema.tables[0].primary_key == "name"
    assert schema.query_focus == ["name", "rating"]
    assert schema.source_chunks_used == ["hash1", "hash2"]

    empty = SchemaDiscoveryModule._parse_response({"tables": []}, [])
    assert empty.is_empty

    print("[OK] SchemaDiscoveryModule._parse_response smoke test passed")
