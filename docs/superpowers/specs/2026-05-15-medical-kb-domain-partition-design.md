# Medical KB Domain Partitioning — Design Spec

**Date**: 2026-05-15
**Status**: Approved (verbal, brainstorming session)
**Author**: dungnq + Claude
**Sub-project**: S5 (first of 5 refactor sub-projects)

## Context

AgentRag targets Vietnamese medical students. Documents span multiple
medical subspecialties (anatomy, internal med, surgery, OB/GYN, pediatrics,
emergency, etc.). Current architecture treats all sources uniformly — no
domain awareness — which causes:

1. Cross-document leak: chat about "Thần kinh chi dưới" retrieves chunks
   from "MCQ GÃY XƯƠNG" because both mention "đau".
2. Vague queries ("đau ngực") return unfocused results because retriever
   doesn't know to prioritize cardio + pulmo + GI.
3. Vietnamese medical synonyms aren't normalized (ho ra máu / khái huyết,
   chân / chi dưới / lower limb).

## Decisions (brainstorming output)

| ID | Decision | Rationale |
|---|---|---|
| D1 | Tag schema: 2-dim (hệ cơ quan × chuyên khoa lâm sàng) | User pick A+B. Captures anatomy + clinical view |
| D2 | Granularity: section-primary + chunk override | User pick D. Textbooks organize by section already; LLM override only for generic headings |
| D3 | Federation: confidence-based + UI override | User pick C. Smart routing + user escape hatch |
| D4 | Ontology: hybrid custom YAML + ICD-10 VN seed | User pick D. Free, no cold-start, extensible |

## Architecture

### 3 knowledge layers

```
┌────────────────────────────────────────────────────────┐
│ Layer 1: Global Shared Ontology                        │
│ - Postgres table: ontology_terms                       │
│ - canonical_term → synonyms + tags + ICD-10 code       │
│ - Used by: TermResolver (query expansion, tagging)     │
└────────────────────────────────────────────────────────┘
                        ↑ resolves canonical
┌────────────────────────────────────────────────────────┐
│ Layer 2: Domain-Tagged Knowledge                       │
│ - ES agentrag_segments + new fields:                   │
│     system_tag (keyword), specialty_tag (keyword[])    │
│ - Section-level inheritance, chunk-level override      │
└────────────────────────────────────────────────────────┘
                        ↑ filters/routes
┌────────────────────────────────────────────────────────┐
│ Layer 3: Cross-Domain Federation                       │
│ - FederatedRetriever wraps ElasticsearchRetriever      │
│ - DomainRouter (SLM) picks 1-3 domains by confidence   │
│ - UI filter dropdown overrides routing                 │
└────────────────────────────────────────────────────────┘
```

### Tag taxonomy

**Hệ cơ quan (system_tag, single-value)** — closed set:
- `tim_mach` — Tim mạch
- `ho_hap` — Hô hấp
- `tieu_hoa` — Tiêu hóa
- `than_kinh` — Thần kinh
- `noi_tiet` — Nội tiết
- `co_xuong_khop` — Cơ - xương - khớp
- `huyet_hoc` — Huyết học
- `tiet_nieu` — Tiết niệu
- `sinh_duc` — Sinh dục (nam/nữ)
- `da_lieu` — Da liễu
- `mat_tmh` — Mắt - Tai mũi họng
- `tam_than` — Tâm thần
- `mien_dich` — Miễn dịch / Dị ứng
- `nhi_khoa` — Nhi (cross-cut với system)
- `da_he` — Đa hệ thống / không thuộc 1 hệ riêng

**Chuyên khoa lâm sàng (specialty_tag, multi-value)** — closed set:
- `noi` — Nội
- `ngoai` — Ngoại
- `san` — Sản phụ khoa
- `nhi` — Nhi
- `cap_cuu` — Cấp cứu
- `hoi_suc` — Hồi sức tích cực
- `truyen_nhiem` — Truyền nhiễm
- `ung_buou` — Ung bướu
- `chan_doan_hinh_anh` — Chẩn đoán hình ảnh
- `xet_nghiem` — Xét nghiệm / Cận lâm sàng
- `duoc_ly` — Dược lý
- `giai_phau` — Giải phẫu (basic science)
- `sinh_ly_benh` — Sinh lý bệnh
- `general` — Chung / không thuộc chuyên khoa cụ thể

### Components

#### C1. Ontology storage — `ontology_terms` table

```sql
CREATE TABLE ontology_terms (
  id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  canonical       VARCHAR(255) NOT NULL,            -- "Đau ngực"
  canonical_norm  VARCHAR(255) NOT NULL,            -- "dau nguc" (no accent, lower) for FTS
  synonyms        JSONB DEFAULT '[]'::jsonb,        -- ["đau tức ngực", "thống tâm", "chest pain"]
  system_tag      VARCHAR(32),                      -- tim_mach | ... | NULL if symptom spans
  specialty_tags  JSONB DEFAULT '[]'::jsonb,        -- ["noi","cap_cuu"]
  parent_id       UUID REFERENCES ontology_terms,   -- hierarchy (Tim mạch → Van tim → Van hai lá)
  icd10_code      VARCHAR(16),                      -- "I20.0" nullable
  source          VARCHAR(16) NOT NULL,             -- 'custom' | 'icd10_vn' | 'mesh'
  notes           TEXT,
  created_at      TIMESTAMPTZ DEFAULT NOW(),
  updated_at      TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX ix_ontology_canonical_norm ON ontology_terms (canonical_norm);
CREATE INDEX ix_ontology_icd10 ON ontology_terms (icd10_code);
CREATE INDEX ix_ontology_synonyms_gin ON ontology_terms USING GIN (synonyms);
```

#### C2. ES schema migration

Add fields to `agentrag_segments` mapping (non-breaking, nullable):

```json
{
  "system_tag":      {"type": "keyword"},
  "specialty_tag":   {"type": "keyword"},
  "canonical_terms": {"type": "keyword"}
}
```

`canonical_terms` is denormalized list of canonical terms detected in chunk
(for highlight + filter by entity).

#### C3. `OntologySeederScript` — `scripts/seed_ontology.py`

One-shot. Inputs:
- `data/ontology/icd10_vn.csv` (downloaded from Bộ Y tế)
- `data/ontology/custom_terms.yaml` (curated)

YAML format:
```yaml
- canonical: Đau ngực
  synonyms: [đau tức ngực, thống tâm, chest pain]
  system_tag: null   # spans multiple
  specialty_tags: [noi, cap_cuu]
  icd10: R07.4
- canonical: Van hai lá
  synonyms: [mitral valve, van hai lá tim]
  system_tag: tim_mach
  specialty_tags: [tim_mach, giai_phau]
  parent: Tim
```

Idempotent: upsert by `(canonical_norm, source)`.

#### C4. `TermResolver` — `src/agentrag/ontology/resolver.py`

```python
class TermResolver:
    def resolve(self, term: str) -> ResolvedTerm:
        """
        term → canonical + synonyms + system_tag + specialty_tags

        1. Normalize (lower, strip accent for fuzzy)
        2. Exact match on canonical_norm
        3. Synonym lookup (JSONB contains)
        4. Trigram fuzzy match if no exact
        5. If nothing → optional SLM fallback (configurable)
        """

    def expand_query(self, query: str) -> str:
        """Inject canonical + synonyms into query for retrieval."""

    def find_in_text(self, text: str, max_terms: int = 10) -> list[ResolvedTerm]:
        """Detect ontology terms in chunk content. Used by SectionTagger."""
```

#### C5. `SectionTagger` — `src/agentrag/ingestion/section_tagger.py`

```python
class SectionTagger:
    """Assign system_tag + specialty_tag to a chunk based on section_path.

    Strategy:
      1. Parse section_path (e.g., "Chương 3 / Hệ tim mạch / Tim")
      2. Resolve each segment via TermResolver
      3. Aggregate: most specific child wins for system_tag
      4. Fallback to chunk-content scan if section_path generic
         ("Tổng quan", "Mở đầu", "Phần 1") via SLM call
    """

    async def tag_chunk(self, chunk: dict) -> dict:
        return {**chunk, "system_tag": ..., "specialty_tag": [...], "canonical_terms": [...]}
```

Integrate into `ingestion/pipeline.py` right before ES indexing.

#### C6. `DomainRouter` — `src/agentrag/orchestration/domain_router.py`

SLM-driven (llama3.2:3b OK for this). Prompt:

```
System: You are a medical domain router. Given a Vietnamese query,
identify which body system(s) and clinical specialty(s) are most
relevant. Return JSON: {systems: [tim_mach,...], specialties: [...],
confidence: 0.0-1.0}.

User: {query}
```

Output:
```python
@dataclass
class DomainRoute:
    systems: list[str]          # filtered by confidence
    specialties: list[str]
    confidence: float
    all_systems: list[str]      # raw model output before filter
```

Threshold: if `confidence < 0.7` → use top-3 systems. Else top-1.

#### C7. `FederatedRetriever` — `src/agentrag/retrieval/federated.py`

```python
class FederatedRetriever:
    def __init__(self, base: ElasticsearchRetriever, router: DomainRouter):
        self._base = base
        self._router = router

    async def search(
        self,
        query: str,
        document_title: str | None = None,
        system_override: str | None = None,    # from UI dropdown
        specialty_override: list[str] | None = None,
        **kwargs,
    ) -> dict:
        if system_override:
            tags = {"systems": [system_override]}
        else:
            route = await self._router.classify(query)
            tags = {"systems": route.systems, "specialties": route.specialties}

        # Pass tags as ES filter clauses (terms query)
        return await self._base.search(query=query, document_title=document_title, filters=tags, **kwargs)
```

#### C8. Frontend filter

In `frontend/src/components/notebook/ChatPanel.tsx` add:

```tsx
<DomainFilter
  value={domainOverride}
  onChange={setDomainOverride}
  options={SYSTEM_TAGS}  // fetched from /on/api/ontology/systems
/>
```

`/on/api/ontology/systems` returns list of known system_tag values + Vietnamese labels.

When set, pass to chat payload:
```json
{
  "session_id": "...",
  "message": "...",
  "domain_filter": {"system": "tim_mach"}
}
```

`/on/api/chat/execute` passes through to retriever.

### Migration / rollout

| Step | Action | Reversible? |
|---|---|---|
| 1 | Alembic migration: create `ontology_terms` table | Yes (drop_table) |
| 2 | ES mapping update: add fields nullable | Yes (no breaking) |
| 3 | Seed ICD-10 VN (download CSV + run seeder) | Yes (truncate table) |
| 4 | Curate initial custom YAML (~50 terms covering current corpus) | Manual |
| 5 | Backfill existing segments: batch script tags via TermResolver | Yes (NULL out tags) |
| 6 | Update ingest pipeline: SectionTagger step | Yes (feature-flag `TAGGING_ENABLED`) |
| 7 | Wire DomainRouter + FederatedRetriever into agent + adapter chat | Yes (env flag `DOMAIN_FILTER_ENABLED`) |
| 8 | UI dropdown component + endpoint | Yes (omit from render) |

Each step independently rollback-able via flag toggles.

### Acceptance criteria

1. **Tagging coverage**: ≥ 95% of new chunks have `system_tag` set after pipeline run on current corpus.
2. **Resolver accuracy**: ontology lookup hit rate ≥ 80% on benchmark set of 50 medical terms (manual eval).
3. **Domain routing**: 10/10 obvious queries route correctly ("đau ngực" → tim_mach + ho_hap, "gãy xương" → co_xuong_khop, "ngạt thở trẻ em" → ho_hap + nhi).
4. **UI filter**: selecting "Tim mạch" in dropdown limits retrieval to `system_tag=tim_mach`.
5. **No regression**: existing chat E2E tests pass with `DOMAIN_FILTER_ENABLED=false`.
6. **Performance**: domain routing adds ≤ 500ms p95 latency (SLM call cached for session).

### Config additions

```env
# Ontology + domain routing
ONTOLOGY_ENABLED=true
TAGGING_ENABLED=true                     # SectionTagger in ingest
DOMAIN_FILTER_ENABLED=true               # FederatedRetriever active
DOMAIN_ROUTER_CONFIDENCE_THRESHOLD=0.7   # below → top-3 expansion
DOMAIN_ROUTER_TOP_K=3                    # max domains when confidence low
ICD10_CSV_PATH=data/ontology/icd10_vn.csv
CUSTOM_ONTOLOGY_PATH=data/ontology/custom_terms.yaml
```

### Out of scope (deferred to later sub-projects)

- **S4**: Cross-domain conflict resolution (when two domains disagree)
- **S4**: KG traversal over domain boundaries
- **S4**: Reasoning Plane vs Execution Plane split — this spec keeps existing agent loop
- **S1**: Dashboard for domain-routing telemetry
- **S2**: Visual graph of which domains were searched per turn
- **S3**: Caching domain classification per session
- English ontology / SNOMED CT / UMLS integration
- Multi-tenant per-domain permission

### Files to create

```
src/agentrag/ontology/
  __init__.py
  resolver.py          (TermResolver)
  models.py            (SQLAlchemy OntologyTerm)
  schema.py            (Pydantic ResolvedTerm)

src/agentrag/orchestration/
  domain_router.py     (DomainRouter SLM)

src/agentrag/retrieval/
  federated.py         (FederatedRetriever)

src/agentrag/ingestion/
  section_tagger.py    (SectionTagger)

scripts/
  seed_ontology.py     (one-shot seeder)
  backfill_tags.py     (existing segments)

data/ontology/
  custom_terms.yaml    (curated)
  icd10_vn.csv         (download manually from Bộ Y tế)

migrations/versions/
  YYYYMMDD_add_ontology_terms.py
  YYYYMMDD_add_segment_domain_tags.py  (alembic doesn't manage ES; helper)

frontend/src/components/notebook/
  DomainFilter.tsx     (new)

frontend/src/lib/api/
  ontology.ts          (new — GET /on/api/ontology/systems)

docs/superpowers/specs/
  2026-05-15-medical-kb-domain-partition-design.md   (this)
```

### Files to modify

```
src/agentrag/ingestion/pipeline.py       (call SectionTagger before ES index)
src/agentrag/ingestion/stores/elasticsearch_store.py (mapping update)
src/agentrag/retrieval/elasticsearch_retriever.py  (accept filters param)
src/agentrag/agent/service.py            (pass DomainRoute or override to retriever)
src/agentrag/adapter/routers/chat.py     (accept domain_filter in body)
src/agentrag/adapter/models.py           (ExecuteChatRequest +domain_filter field)
src/agentrag/config.py                   (new env vars)
.env / .env.example                      (new defaults)
README.md                                (new §5.10 Ontology, §16 KB layers)
```

## Roadmap after S5

User pre-approved I pick order. My plan:

| Order | Sub-project | Rationale |
|---|---|---|
| 1 | **S5** (this spec) | Foundation; later code aligns to domain layout |
| 2 | **S4** Reasoning/Execution layering | Architecture rework; touches everything. Best done after S5 establishes data layout |
| 3 | **S1** Token dashboard | Quick win; telemetry foundation for S3 |
| 4 | **S2** LangGraph trace UI | UX win; leverages existing admin panel; ~1 week |
| 5 | **S3** Speed/latency optimization | Last — needs S1 telemetry to identify bottlenecks |

Each gets its own spec → plan → impl cycle.

## Open questions

None. All decisions locked during brainstorming session.
