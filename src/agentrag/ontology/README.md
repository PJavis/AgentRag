# ontology — shared medical taxonomy + term resolver (canonical terms / synonyms / ICD-10)

## Mục đích / Purpose
Layer 1 của domain-partitioned medical KB (S5). Lưu một bảng các **canonical
Vietnamese medical terms** kèm synonyms, hierarchy (parent/child), `system_tag`
(15 hệ cơ quan), `specialty_tags` (chuyên khoa) và `icd10_code`. Cung cấp
`TermResolver` để map một chuỗi free-form → canonical + tags (exact → synonym →
trigram fuzzy). Mục đích: tag chunk khi ingest (`SectionTagger`) và phục vụ
taxonomy tĩnh cho UI/router. Đây là một **vocabulary store thuần** — không gọi
LLM, không phụ thuộc vào `ServiceContainer`.

## Plane
**Execution Plane / Infrastructure.** Stateless IO worker: `TermResolver` mở
session Postgres riêng cho mỗi call và đọc bảng `ontology_terms`. Không chứa
prompt hay branching logic (Reasoning Plane). Quyết định "dùng tag nào để route"
nằm ở `orchestration/domain_router.py` và `retrieval/federated.py`, không ở đây.

## Key files
| File | Responsibility |
|------|----------------|
| `models.py` | `OntologyTerm` — SQLAlchemy model cho bảng `ontology_terms` (PK UUID, `canonical`, `canonical_norm`, `synonyms` JSONB, `system_tag`, `specialty_tags` JSONB, `parent_id` self-FK, `icd10_code`, `source`, `notes`, timestamps). |
| `schema.py` | `ResolvedTerm` — Pydantic wire-format trả về từ resolver (`canonical`, `synonyms`, `system_tag`, `specialty_tags`, `icd10_code`, `confidence`, `source`). |
| `resolver.py` | `TermResolver` (exact / synonym / fuzzy + `find_in_text` + `expand_query`); helpers `_norm()` và `_to_resolved()`. |
| `__init__.py` | Docstring-only; không export gì. |

## Public interface
Import trực tiếp (không qua `ServiceContainer`, không có Protocol trong
`services/protocols.py`):

```python
from src.agentrag.ontology.resolver import TermResolver
from src.agentrag.ontology.schema import ResolvedTerm
from src.agentrag.ontology.models import OntologyTerm

resolver = TermResolver()                                   # không cần session
hit  = await resolver.resolve("nhoi mau co tim")            # ResolvedTerm | None
hit  = await resolver.resolve("Chương 3", strict=True)      # bỏ qua fuzzy
hits = await resolver.find_in_text(content, max_terms=10)   # list[ResolvedTerm]
expanded = await resolver.expand_query("hở van hai lá")     # str
```

`TermResolver()` **không nhận session** — mỗi method tự mở
`AsyncSessionLocal()` (xem Gotchas). Method `resolve(term, *, strict=False)`:
strict=True tắt fuzzy. `find_in_text` quét toàn bảng bằng word-boundary regex.

Consumers thực tế trong repo:
- `ingestion/section_tagger.py` — `SectionTagger(resolver=...)` gọi
  `resolve(strict=True)` cho từng segment của `section_path`, rồi
  `find_in_text(content)` làm fallback.
- `scripts/seed_ontology.py` — insert/upsert `OntologyTerm` rows.

## Data flow
**Inputs:** một string (term, section title, hoặc chunk content) + bảng
`ontology_terms` đã được seed.

**resolve():** normalize qua `_norm()` (NFD → bỏ dấu → `đ→d` → lowercase →
collapse spaces) rồi:
1. **Exact** match trên `canonical_norm`.
2. **Synonym** — `ILIKE '%"<lower_term>"%'` trên `synonyms` cast sang String
   (substring trong JSONB text, có dấu nháy kép để khớp phần tử array).
3. **Fuzzy** — `pg_trgm` `similarity(canonical_norm, norm) > 0.45`, top-1 theo
   sim desc; bỏ qua khi `strict=True`. `confidence` = similarity (`< 1.0`).

**find_in_text():** load **toàn bộ** rows một lần, với mỗi row thử
`\b<needle>\b` (canonical + synonyms, lowercase, regex word-boundary) trên text;
dừng ở `max_terms`. Word-boundary tránh false positive acronym ngắn (vd "MI"
không khớp "programMIng").

**expand_query():** gọi `find_in_text(query, max_terms=5)`, ghép thêm canonical +
tối đa 3 synonyms (dedup, bỏ token đã có trong query) vào cuối query string.

**Outputs:** `ResolvedTerm | None` (resolve) hoặc `list[ResolvedTerm]`
(find_in_text) hoặc `str` (expand_query).

**Upstream callers:** `SectionTagger` (ingestion pipeline). **Downstream:** chỉ
Postgres (`AsyncSessionLocal`, bảng `ontology_terms`).

## Config
| Flag (`src/agentrag/config.py`) | Mặc định | Ảnh hưởng tới module này |
|---|---|---|
| `TAGGING_ENABLED` | `True` | Bật/tắt `SectionTagger` (caller của resolver) khi ingest. Module này không tự đọc flag — flag được kiểm ở ingestion. |

`_FUZZY_THRESHOLD = 0.45` là **hằng số hard-coded** trong `resolver.py`, không
phải env flag. `DOMAIN_ROUTER_CONFIDENCE_THRESHOLD` / `DOMAIN_ROUTER_TOP_K` thuộc
về router/federated retrieval, không đọc trong package `ontology/`.

## Gotchas
- **Resolver tự mở session, không nhận session injection.** Chữ ký
  `TermResolver()` không có tham số session; mỗi `resolve` / `find_in_text` mở
  một `AsyncSessionLocal()` mới. Gọi nhiều lần = nhiều session.
- **`find_in_text` load full-table mỗi lần gọi** (`select(OntologyTerm)` rồi
  scan trong Python). Ổn với vài chục–vài trăm terms; không phù hợp nếu bảng
  phình to. Không dùng index.
- **Hai hàm `_norm` riêng biệt phải giữ đồng bộ:** `resolver._norm` và
  `section_tagger._norm` (bản trong section_tagger còn strip ký tự không
  alphanumeric). Lệch normalize → tag miss. Seeder cũng có bản `_norm` riêng.
- **Synonym match là substring ILIKE trên text cast của JSONB**, không phải JSONB
  containment — index GIN `ix_ontology_synonyms_gin` (plain JSONB GIN từ
  migration `2026051501`) **không** tăng tốc query này. Chỉ
  `ix_ontology_canonical_trgm` (GIN `gin_trgm_ops` trên `canonical_norm`, từ
  migration `2026051502`) phục vụ bước fuzzy.
- **`expand_query` hiện chưa có caller nội bộ.** Trùng tên với
  `KnowledgeService.expand_query(query, intent)` ở `services/knowledge_service.py`
  và `services/reasoning_knowledge.py:expand_query(query, intent)` — đó là hàm
  KHÁC (chữ ký khác, intent-based), không liên quan tới ontology. Đừng nhầm.
- **Fuzzy cần extension `pg_trgm`.** Thiếu extension → `func.similarity` lỗi ở
  runtime. Bật qua migration `2026051502_enable_pg_trgm.py`.
- **`source` không có giá trị mặc định ở DB** (`nullable=False`, không default);
  seeder set `'custom'` hoặc `'icd10_vn'`. Upsert key idempotent là
  `(canonical_norm, source)`.

## Adjacent (ngoài package này)
- `ingestion/section_tagger.py` — `SectionTagger` (caller chính của resolver).
- `retrieval/federated.py` — `FederatedRetriever` + `system_override` /
  `specialty_override`; route bằng `orchestration/domain_router.py:DomainRouter`.
- `retrieval/context.py` — ContextVars `set_domain_filter` / `get_domain_filter`
  (và `set_document_scope` / `get_document_scope`) propagate override per-turn.
- `adapter/routers/ontology.py` — `GET /ontology/systems`, `/ontology/specialties`
  (taxonomy tĩnh cho UI dropdown).
- `scripts/seed_ontology.py` — seed từ `data/ontology/custom_terms.yaml` (`--yaml`)
  và CSV ICD-10 (`--icd10`, default path `data/ontology/icd10_vn.csv` — file CSV
  này hiện chưa có trong repo; chỉ YAML được seed).
- `scripts/backfill_tags.py` — tag lại các segment đã ingest trước S5.
- Migrations: `2026051501_add_ontology_terms.py`, `2026051502_enable_pg_trgm.py`.

> Lưu ý: package `ontology/` **không** bị đụng tới bởi đợt RAG-enhancement
> 2026-06 (Contextual Retrieval, RAPTOR, CRAG, adaptive routing, semantic cache).
> Không có flag mới hay code path mới ở đây.
