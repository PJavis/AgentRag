from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from src.agentrag.services.llm_gateway import LLMGateway

from src.agentrag.config import settings
from src.agentrag.structured.schema_discovery import RelationalSchema


@dataclass
class ProvenanceRecord:
    row_index: int
    source_chunk_hash: str
    source_doc: str
    source_section: str | None


@dataclass
class SQLEngineOutput:
    sql_query: str
    result_rows: list[dict[str, Any]]
    provenance: list[ProvenanceRecord]
    execution_ok: bool
    fallback_reason: str | None
    retry_count: int


_SQL_SYSTEM = """\
You are a SQLite SQL query generator. Given a schema and a natural language question, write a single SQLite SELECT query.

Rules:
- Return ONLY the SQL query string — no explanation, no markdown code blocks.
- Use CAST(col AS REAL) for numeric comparisons/aggregations when needed.
- All column values are stored as TEXT in SQLite; use CAST for arithmetic.
- Only use SELECT statements. Never use INSERT, UPDATE, DELETE, DROP, CREATE.
- Use table and column names exactly as provided in the schema.
- Do not add semicolons at the end.
"""

_SQL_RETRY_SYSTEM = """\
You are a SQLite SQL query fixer. The previous SQL query failed with an error.
Fix the query based on the error message and schema.

Rules:
- Return ONLY the corrected SQL query string.
- Only use SELECT statements.
- Use table and column names exactly as provided in the schema.
"""


class SQLReasoningEngine:
    """
    Biên dịch câu hỏi → SQL, thực thi trên in-memory SQLite.
    Thay thế LLM reasoning bằng SQL engine deterministic.
    """

    def __init__(
        self,
        llm_gateway: LLMGateway,
        max_retries: int | None = None,
    ) -> None:
        self._llm = llm_gateway
        self._max_retries = max_retries if max_retries is not None else settings.STRUCTURED_SQL_MAX_RETRIES

    async def execute(
        self,
        question: str,
        schema: RelationalSchema,
        database: dict[str, list[dict[str, Any]]],
        query_type: str,
    ) -> SQLEngineOutput:
        if schema.is_empty or not database:
            return SQLEngineOutput(
                sql_query="",
                result_rows=[],
                provenance=[],
                execution_ok=False,
                fallback_reason="empty_schema_or_database",
                retry_count=0,
            )

        # Compile SQL
        sql_query = await self._compile_sql(question, schema, query_type)

        # Execute với retry
        retry_count = 0
        last_error: str | None = None
        result_rows: list[dict[str, Any]] = []

        for attempt in range(self._max_retries + 1):
            retry_count = attempt
            try:
                result_rows = self._run_sql(sql_query, schema, database)
                last_error = None
                break
            except sqlite3.Error as exc:
                last_error = str(exc)
                if attempt < self._max_retries:
                    sql_query = await self._compile_sql(
                        question, schema, query_type,
                        previous_sql=sql_query,
                        error_message=last_error,
                    )

        if last_error is not None:
            return SQLEngineOutput(
                sql_query=sql_query,
                result_rows=[],
                provenance=[],
                execution_ok=False,
                fallback_reason=f"sql_error:{last_error}",
                retry_count=retry_count,
            )

        provenance = self._map_provenance(result_rows, database, schema)

        return SQLEngineOutput(
            sql_query=sql_query,
            result_rows=result_rows,
            provenance=provenance,
            execution_ok=True,
            fallback_reason=None,
            retry_count=retry_count,
        )

    # ── SQL Compilation ───────────────────────────────────────────────────────

    async def _compile_sql(
        self,
        question: str,
        schema: RelationalSchema,
        query_type: str,
        previous_sql: str | None = None,
        error_message: str | None = None,
    ) -> str:
        schema_desc = self._schema_to_text(schema)

        if previous_sql and error_message:
            system_prompt = _SQL_RETRY_SYSTEM
            user_prompt = json.dumps(
                {
                    "question": question,
                    "schema": schema_desc,
                    "previous_sql": previous_sql,
                    "error": error_message,
                },
                ensure_ascii=False,
            )
        else:
            system_prompt = _SQL_SYSTEM
            user_prompt = json.dumps(
                {
                    "question": question,
                    "query_type": query_type,
                    "schema": schema_desc,
                },
                ensure_ascii=False,
            )

        try:
            raw, _latency = await self._llm.json_response(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                task="sql_compile",
            )
        except Exception as exc:
            raise sqlite3.Error(f"LLM compile failed: {exc}") from exc

        # LLM trả JSON — tìm SQL string
        sql = self._extract_sql_from_response(raw)
        return sql

    @staticmethod
    def _extract_sql_from_response(raw: dict[str, Any] | str) -> str:
        """Trích xuất SQL string từ LLM response (có thể là dict hoặc string)."""
        if isinstance(raw, str):
            sql = raw.strip()
        elif isinstance(raw, dict):
            # Thử các key phổ biến
            for key in ("sql", "query", "sql_query", "result", "answer"):
                if isinstance(raw.get(key), str):
                    sql = raw[key].strip()
                    break
            else:
                # Fallback: join tất cả string values
                sql = " ".join(str(v) for v in raw.values() if isinstance(v, str)).strip()
        else:
            sql = str(raw).strip()

        # Strip markdown code block nếu có
        if sql.startswith("```"):
            lines = sql.split("\n")
            lines = [l for l in lines if not l.startswith("```")]
            sql = "\n".join(lines).strip()

        return sql

    # ── SQLite Execution ──────────────────────────────────────────────────────

    def _run_sql(
        self,
        sql: str,
        schema: RelationalSchema,
        database: dict[str, list[dict[str, Any]]],
    ) -> list[dict[str, Any]]:
        """Tạo in-memory SQLite, hydrate data, execute SQL, trả rows."""
        # Safety check: chỉ cho phép SELECT
        sql_upper = sql.strip().upper()
        if not sql_upper.startswith("SELECT"):
            raise sqlite3.Error(f"Only SELECT statements are allowed. Got: {sql[:50]}")

        conn = sqlite3.connect(":memory:")
        conn.row_factory = sqlite3.Row
        try:
            self._hydrate_db(conn, schema, database)
            cursor = conn.execute(sql)
            rows = [dict(row) for row in cursor.fetchall()]
            return rows
        finally:
            conn.close()

    def _hydrate_db(
        self,
        conn: sqlite3.Connection,
        schema: RelationalSchema,
        database: dict[str, list[dict[str, Any]]],
    ) -> None:
        """CREATE TABLE và INSERT rows. Strip _source_* metadata trước khi INSERT."""
        for table in schema.tables:
            cols_def = ", ".join(f'"{col}" TEXT' for col in table.columns)
            conn.execute(f'CREATE TABLE IF NOT EXISTS "{table.name}" ({cols_def})')

            rows = database.get(table.name, [])
            for row in rows:
                # Strip _source_* metadata columns
                data = {k: v for k, v in row.items() if not k.startswith("_")}
                # Chỉ insert columns có trong schema
                schema_cols = [c for c in table.columns if c in data]
                if not schema_cols:
                    continue
                placeholders = ", ".join("?" for _ in schema_cols)
                col_names = ", ".join(f'"{c}"' for c in schema_cols)
                values = [str(data[c]) if data[c] is not None else None for c in schema_cols]
                conn.execute(
                    f'INSERT INTO "{table.name}" ({col_names}) VALUES ({placeholders})',
                    values,
                )
        conn.commit()

    def _map_provenance(
        self,
        result_rows: list[dict[str, Any]],
        database: dict[str, list[dict[str, Any]]],
        schema: RelationalSchema,
    ) -> list[ProvenanceRecord]:
        """
        Map result_rows về source chunks thông qua primary key.
        Với mỗi result row, tìm row gốc trong database có cùng PK value.
        """
        provenance: list[ProvenanceRecord] = []

        for idx, result_row in enumerate(result_rows):
            found_hash = ""
            found_doc = ""
            found_section: str | None = None

            for table in schema.tables:
                pk = table.primary_key
                if not pk or pk not in result_row:
                    continue
                pk_val = str(result_row[pk])
                for orig_row in database.get(table.name, []):
                    if str(orig_row.get(pk, "")) == pk_val:
                        found_hash = orig_row.get("_source_chunk_hash", "")
                        found_doc = orig_row.get("_source_doc", "")
                        found_section = orig_row.get("_source_section")
                        break
                if found_hash:
                    break

            provenance.append(ProvenanceRecord(
                row_index=idx,
                source_chunk_hash=found_hash,
                source_doc=found_doc,
                source_section=found_section,
            ))

        return provenance

    @staticmethod
    def _schema_to_text(schema: RelationalSchema) -> dict[str, Any]:
        return {
            "tables": [
                {
                    "name": t.name,
                    "columns": t.columns,
                    "primary_key": t.primary_key,
                }
                for t in schema.tables
            ],
            "join_keys": [
                {
                    "from_table": jk.from_table,
                    "from_column": jk.from_column,
                    "to_table": jk.to_table,
                    "to_column": jk.to_column,
                }
                for jk in schema.join_keys
            ],
        }


if __name__ == "__main__":
    from src.agentrag.structured.schema_discovery import RelationalSchema, TableDef

    engine = SQLReasoningEngine.__new__(SQLReasoningEngine)

    schema = RelationalSchema(
        tables=[TableDef(name="product", columns=["name", "rating", "price"], primary_key="name")],
        join_keys=[],
        query_focus=["name", "rating"],
        source_chunks_used=[],
    )

    database: dict[str, list[dict]] = {
        "product": [
            {"name": "A", "rating": "4.5", "price": "100", "_source_chunk_hash": "h1", "_source_doc": "Doc1", "_source_section": "sec1"},
            {"name": "B", "rating": "3.8", "price": "80", "_source_chunk_hash": "h2", "_source_doc": "Doc1", "_source_section": "sec2"},
            {"name": "C", "rating": "4.1", "price": "90", "_source_chunk_hash": "h3", "_source_doc": "Doc2", "_source_section": "sec1"},
        ]
    }

    # Test _hydrate_db
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    engine._hydrate_db(conn, schema, database)
    rows = [dict(r) for r in conn.execute('SELECT * FROM "product"').fetchall()]
    assert len(rows) == 3, f"Expected 3, got {len(rows)}"
    conn.close()
    print("[OK] _hydrate_db: 3 rows inserted")

    # Test _run_sql
    rows = engine._run_sql(
        'SELECT name, rating FROM "product" ORDER BY CAST(rating AS REAL) DESC',
        schema, database,
    )
    assert rows[0]["name"] == "A"
    assert rows[1]["name"] == "C"
    print("[OK] _run_sql: ORDER BY rating DESC works")

    # Test SELECT only safety
    try:
        engine._run_sql('DROP TABLE "product"', schema, database)
        assert False, "Should have raised"
    except sqlite3.Error:
        print("[OK] safety: DROP TABLE rejected")

    # Test provenance
    result_rows = [{"name": "A", "rating": "4.5"}]
    prov = engine._map_provenance(result_rows, database, schema)
    assert prov[0].source_chunk_hash == "h1"
    assert prov[0].source_doc == "Doc1"
    print("[OK] provenance mapping")

    # Test _extract_sql_from_response
    assert engine._extract_sql_from_response({"sql": "SELECT 1"}) == "SELECT 1"
    assert engine._extract_sql_from_response("SELECT 1") == "SELECT 1"
    assert "SELECT" in engine._extract_sql_from_response({"query": "SELECT name FROM t"})
    print("[OK] _extract_sql_from_response")

    print("[ALL PASS] SQLReasoningEngine unit tests")
