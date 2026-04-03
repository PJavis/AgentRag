from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from src.pam.services.llm_gateway import LLMGateway

from src.pam.config import settings
from src.pam.structured.schema_discovery import TableDef


@dataclass
class ExtractionStats:
    table_name: str
    total_rows: int = 0
    valid_rows: int = 0
    null_pk_dropped: int = 0
    type_coerced: int = 0
    conflict_resolved: int = 0


@dataclass
class ExtractionOutput:
    database: dict[str, list[dict[str, Any]]]
    # table_name → list of rows. Mỗi row có _source_chunk_hash và _source_doc.
    stats: list[ExtractionStats]
    is_empty: bool


_EXTRACT_SYSTEM = """\
You are a data extraction assistant. Given a schema and a text passage, extract rows matching the schema.

Rules:
- Return a JSON array of row objects following the schema columns.
- Only extract information clearly present in the text.
- If no relevant data exists in this passage, return an empty array [].
- Use null for missing values.
- Do NOT invent or hallucinate values.
- Primary key must not be null or empty string.

Return format: [{"col1": val, "col2": val, ...}, ...]
"""


class StructuredExtractor:
    """
    Trích xuất rows từ text chunks theo relational schema.
    Phase 1: sequential (một chunk tại một thời điểm).
    Phase 2 (M6): async batch với asyncio.gather.
    """

    def __init__(self, llm_gateway: LLMGateway) -> None:
        self._llm = llm_gateway

    async def extract(
        self,
        chunks: list[dict[str, Any]],
        schema: "RelationalSchema",  # noqa: F821
        question: str,
        max_chunks: int | None = None,
    ) -> ExtractionOutput:
        from src.pam.structured.schema_discovery import RelationalSchema  # local import

        limit = max_chunks or settings.STRUCTURED_MAX_CHUNKS_FOR_EXTRACT
        working_chunks = chunks[:limit]

        # database: table_name → list of rows (pre-Level B)
        raw_database: dict[str, list[dict[str, Any]]] = {t.name: [] for t in schema.tables}
        stats_map: dict[str, ExtractionStats] = {t.name: ExtractionStats(t.name) for t in schema.tables}

        # ── Async batch extraction (M6) ───────────────────────────────────────
        tasks = [
            self._extract_chunk(chunk, table, question)
            for chunk in working_chunks
            for table in schema.tables
        ]
        all_results = await asyncio.gather(*tasks, return_exceptions=True)

        # Map results về đúng chunk + table
        idx = 0
        for chunk in working_chunks:
            for table in schema.tables:
                result = all_results[idx]
                idx += 1
                if isinstance(result, Exception):
                    continue
                rows: list[dict[str, Any]] = result  # type: ignore[assignment]
                for row in rows:
                    stats_map[table.name].total_rows += 1
                    validated = self._validate_row_level_a(row, table)
                    if validated is None:
                        stats_map[table.name].null_pk_dropped += 1
                        continue
                    validated["_source_chunk_hash"] = chunk.get("content_hash", "")
                    validated["_source_doc"] = chunk.get("document_title", "")
                    validated["_source_section"] = chunk.get("section_path", "")
                    validated["_source_position"] = chunk.get("position", 0)
                    raw_database[table.name].append(validated)
                    stats_map[table.name].valid_rows += 1

        # ── CLEAR Level B: cross-row consistency ─────────────────────────────
        database: dict[str, list[dict[str, Any]]] = {}
        for table in schema.tables:
            cleaned, conflicts = self._validate_cross_row_level_b(
                raw_database[table.name], table
            )
            database[table.name] = cleaned
            stats_map[table.name].conflict_resolved = conflicts

        total_rows = sum(len(rows) for rows in database.values())
        return ExtractionOutput(
            database=database,
            stats=list(stats_map.values()),
            is_empty=(total_rows == 0),
        )

    async def _extract_chunk(
        self,
        chunk: dict[str, Any],
        table: TableDef,
        question: str,
    ) -> list[dict[str, Any]]:
        content = (chunk.get("content") or "")
        if not content.strip():
            return []

        schema_desc = json.dumps(
            {
                "table": table.name,
                "columns": table.columns,
                "primary_key": table.primary_key,
                "description": table.description,
            },
            ensure_ascii=False,
        )
        user_prompt = json.dumps(
            {
                "question": question,
                "schema": json.loads(schema_desc),
                "text": content,
            },
            ensure_ascii=False,
        )

        try:
            raw, _latency = await self._llm.json_response(
                system_prompt=_EXTRACT_SYSTEM,
                user_prompt=user_prompt,
                task="extract",
            )
        except Exception:
            return []

        # LLM nên trả list, nhưng đôi khi bọc trong dict
        if isinstance(raw, list):
            return raw
        if isinstance(raw, dict):
            # Thử các key phổ biến
            for key in ("rows", "data", "results", table.name):
                if isinstance(raw.get(key), list):
                    return raw[key]
        return []

    def _validate_row_level_a(
        self,
        row: dict[str, Any],
        table: TableDef,
    ) -> dict[str, Any] | None:
        """
        CLEAR Level A — single-row validation:
        1. Drop nếu primary_key null hoặc empty.
        2. Coerce numeric-looking strings thành float.
        Trả None để drop row, trả dict đã validate để giữ.
        """
        if not isinstance(row, dict):
            return None

        # Check primary key
        pk = table.primary_key
        if pk:
            pk_val = row.get(pk)
            if pk_val is None or str(pk_val).strip() == "":
                return None

        # Coerce numeric strings
        coerced = {}
        for k, v in row.items():
            if k.startswith("_"):
                coerced[k] = v
                continue
            if isinstance(v, str):
                stripped = v.strip()
                try:
                    coerced[k] = float(stripped) if "." in stripped else int(stripped)
                except (ValueError, TypeError):
                    coerced[k] = v
            else:
                coerced[k] = v

        return coerced

    def _validate_cross_row_level_b(
        self,
        rows: list[dict[str, Any]],
        table: TableDef,
    ) -> tuple[list[dict[str, Any]], int]:
        """
        CLEAR Level B — cross-row consistency:
        1. Dedup: rows giống nhau (trừ _source_* columns) → giữ row đầu tiên.
        2. Conflict: cùng PK nhưng khác giá trị non-source column
           → giữ row đến từ chunk có position thấp hơn (specific hơn).
        Trả (cleaned_rows, conflict_count).
        """
        if not rows or not table.primary_key:
            return rows, 0

        pk = table.primary_key
        conflicts = 0
        # pk_value → row đã chọn
        pk_map: dict[str, dict[str, Any]] = {}

        for row in rows:
            pk_val = str(row.get(pk, ""))
            if not pk_val:
                continue

            if pk_val not in pk_map:
                pk_map[pk_val] = row
                continue

            existing = pk_map[pk_val]
            # Kiểm tra xem có conflict không (khác giá trị ở non-source column)
            data_cols = [c for c in table.columns if c != pk]
            has_conflict = any(
                str(row.get(c)) != str(existing.get(c))
                for c in data_cols
                if row.get(c) is not None and existing.get(c) is not None
            )

            if has_conflict:
                conflicts += 1
                # Giữ row từ chunk có _source_position nhỏ hơn (chunk đến sớm hơn = specific hơn)
                row_pos = int(row.get("_source_position", 9999))
                existing_pos = int(existing.get("_source_position", 9999))
                if row_pos < existing_pos:
                    pk_map[pk_val] = row
            # Nếu không conflict (duplicate): giữ nguyên existing

        return list(pk_map.values()), conflicts


if __name__ == "__main__":
    from src.pam.structured.schema_discovery import TableDef

    extractor = StructuredExtractor.__new__(StructuredExtractor)  # no llm needed for unit test

    table = TableDef(name="product", columns=["name", "rating", "price"], primary_key="name")

    # Valid row
    row1 = {"name": "ProductA", "rating": "4.5", "price": "100"}
    result1 = extractor._validate_row_level_a(row1, table)
    assert result1 is not None
    assert result1["rating"] == 4.5
    assert result1["price"] == 100
    print("[OK] numeric coercion")

    # Null PK → drop
    row2 = {"name": None, "rating": "4.5"}
    result2 = extractor._validate_row_level_a(row2, table)
    assert result2 is None
    print("[OK] null pk dropped")

    # Empty PK → drop
    row3 = {"name": "", "rating": "4.5"}
    result3 = extractor._validate_row_level_a(row3, table)
    assert result3 is None
    print("[OK] empty pk dropped")

    # Non-dict → drop
    result4 = extractor._validate_row_level_a("bad", table)
    assert result4 is None
    print("[OK] non-dict dropped")

    # CLEAR Level B — dedup
    rows_dup = [
        {"name": "A", "rating": 4.5, "_source_position": 1},
        {"name": "A", "rating": 4.5, "_source_position": 2},  # duplicate
        {"name": "B", "rating": 3.8, "_source_position": 1},
    ]
    cleaned, conflicts = extractor._validate_cross_row_level_b(rows_dup, table)
    assert len(cleaned) == 2  # A (dedup) + B
    assert conflicts == 0
    print("[OK] CLEAR Level B: dedup")

    # CLEAR Level B — conflict resolution (keep lower position)
    rows_conflict = [
        {"name": "A", "rating": 4.5, "_source_position": 3},
        {"name": "A", "rating": 4.9, "_source_position": 1},  # conflict, lower pos → wins
    ]
    cleaned2, conflicts2 = extractor._validate_cross_row_level_b(rows_conflict, table)
    assert len(cleaned2) == 1
    assert conflicts2 == 1
    assert cleaned2[0]["rating"] == 4.9  # position 1 wins
    print("[OK] CLEAR Level B: conflict resolved by position")

    print("[ALL PASS] StructuredExtractor unit tests")
