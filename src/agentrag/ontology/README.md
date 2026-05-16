# Ontology Module

Shared medical taxonomy + term resolver for AgentRag's domain-partitioned KB
(S5). Hỗ trợ tagging chunk, lọc retrieval, và mở rộng query.

## Schema

`OntologyTerm` (table `ontology_terms`):

| Column           | Type           | Notes |
|------------------|----------------|-------|
| `id`             | UUID           | PK |
| `canonical`      | str            | Tên chuẩn (VD: "nhồi máu cơ tim") |
| `canonical_norm` | str (indexed)  | Slugified — strip dấu + `đ→d` + lowercase |
| `synonyms`       | JSONB list     | Đồng nghĩa, normalize same as canonical |
| `system_tag`     | str?           | `tim_mach` / `ho_hap` / … (15 hệ) |
| `specialty_tags` | JSONB list     | `noi` / `ngoai` / … (14 chuyên khoa) |
| `parent_id`      | UUID?          | Self-ref cho phân cấp (cha/con) |
| `icd10_code`     | str (indexed)? | VD `I21` cho NMCT |
| `source`         | str            | `custom` \| `icd10_vn` |
| `notes`          | text?          | Mô tả tự do |

Indexes: GIN trigram (`canonical_norm`, `synonyms`) cho fuzzy match — bật
qua extension `pg_trgm` (migration `2026051502_enable_pg_trgm.py`).

## TermResolver

```python
from src.agentrag.ontology.resolver import TermResolver

resolver = TermResolver(session)
hit = await resolver.resolve("nhoi mau co tim")        # ResolvedTerm | None
hit = await resolver.resolve("Chương 3", strict=True)  # bỏ qua fuzzy
expanded = await resolver.expand_query("hở van hai lá")  # query + canonical + synonyms
hits = await resolver.find_in_text(chunk_content, max_terms=10)
```

Strategy:

1. **Exact** trên `canonical_norm` (sau khi normalize).
2. **Synonym** JSONB substring lookup.
3. **Fuzzy** pg_trgm `similarity ≥ 0.45` — bỏ qua khi `strict=True`.

`find_in_text` dùng `\b{needle}\b` word-boundary để tránh false positive
acronym ngắn (VD "MI" không match "programMIng").

## Ingestion pipeline

`SectionTagger` chạy mỗi chunk:

1. Đi từng segment trong `section_path`, gọi `resolve(strict=True)`.
2. Bỏ qua heading generic (`tong quan`, `phan N`, `chuong N`, …).
3. Nếu vẫn chưa có `system_tag` → fallback `find_in_text(content)`.
4. Ghi `system_tag`, `specialty_tag`, `canonical_terms` vào ES mapping.

Bật/tắt qua `TAGGING_ENABLED`.

## Retrieval

`FederatedRetriever` wrap base `ElasticsearchRetriever` + `DomainRouter`:

- User chỉ định `system_override` / `specialty_override` → bỏ router, lọc thẳng.
- Else: gọi router → top-1 nếu `confidence ≥ DOMAIN_ROUTER_CONFIDENCE_THRESHOLD`,
  ngược lại top-K (`DOMAIN_ROUTER_TOP_K`).
- ContextVar `_domain_filter` được set bởi `AgentService.chat()` và đọc bởi
  `AgentTools._current_filters()` — không phải threadlocal kwarg qua mọi call site.

## Seeder

```bash
python scripts/seed_ontology.py \
  --yaml data/ontology/custom_terms.yaml \
  --icd10 data/ontology/icd10_vn.csv
```

Upsert idempotent theo compound key `(canonical_norm, source)`. YAML 59
canonical terms (triệu chứng, bệnh, giải phẫu, thủ thuật, thuốc, cấp cứu,
chẩn đoán hình ảnh, sản khoa).

## Backfill

```bash
python scripts/backfill_tags.py --dry-run     # preview
python scripts/backfill_tags.py               # apply
```

ES scroll + bulk update — tag lại segments đã ingest trước S5.

## Adapter endpoints

`GET /on/api/ontology/systems` và `/specialties` — taxonomy tĩnh cho UI
dropdown, sync với prompt của `DomainRouter` + closed-set của
`SectionTagger`. Public route (không cần Bearer).

## File map

```
ontology/
├── __init__.py
├── models.py     # OntologyTerm SQLAlchemy
├── schema.py     # ResolvedTerm pydantic
├── resolver.py   # TermResolver (exact/synonym/fuzzy + find_in_text)
└── README.md     # this file
```

Adjacent:

- `src/agentrag/ingestion/section_tagger.py` — chunk tagger
- `src/agentrag/retrieval/federated.py` — router + filter wrapper
- `src/agentrag/retrieval/context.py` — ContextVar plumbing
- `src/agentrag/orchestration/domain_router.py` — SLM classifier
- `src/agentrag/adapter/routers/ontology.py` — taxonomy endpoints
- `scripts/seed_ontology.py`, `scripts/backfill_tags.py`
- `data/ontology/custom_terms.yaml`
- `migrations/versions/2026051501_add_ontology_terms.py`
- `migrations/versions/2026051502_enable_pg_trgm.py`
