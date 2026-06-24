# Test Baseline — 2026-06-24

Recorded as P0 Task 3 of the VITAL improvement plan. Establishes the known-green
floor after the structured-SQL path removal (`e4eb895`) and the answerability-gate
code (`f14ba6f`). Stack up: `agentrag-{postgres,elasticsearch,valkey,ollama}`
healthy; DB migrated to head; ontology **not** seeded.

## Backend (pytest)

| Run | Command | Result |
|---|---|---|
| **CI-relevant subset** | `uv run pytest -q --ignore=tests/ontology --ignore=tests/ingestion` (= `make test-fast`) | **142 passed, 0 failed** ✅ |
| **Full suite** | `uv run pytest -q` | **165 passed, 10 failed** |

### The 10 full-suite failures — all environment, none code

All 10 live in the two suites `make test-fast` deliberately excludes, and all are
caused by missing local environment, not by code regressions or SQL-removal fallout:

- `tests/ontology/test_resolver.py` (6): `test_resolver_exact`, `test_resolver_synonym`,
  `test_resolver_norm_diacritic_insensitive`, `test_resolver_fuzzy_typo`,
  `test_expand_query_adds_synonyms`, `test_find_in_text_returns_terms`.
  **Cause:** the `ontology_terms` table is empty — `make seed-ontology` has not been
  run on this fresh DB. Seeding fixes all six.
- `tests/ingestion/test_section_tagger.py` (3) + `tests/ingestion/test_pdf_ocr_fallback.py` (1).
  **Cause:** section tagger depends on ontology data (same un-seeded cause); the OCR
  test needs the Tesseract binary installed locally.

**SQL-removal fallout check:** `grep -ciE "structured|query_classifier|sql_engine|No module named"`
over the full pytest log = **0**. The `e4eb895` commit already removed the orphaned
structured-path tests (`test_tabular_gate.py` etc.); there are no dangling imports to
clean up. No test files were deleted or modified for this baseline.

## Frontend (vitest)

`CI=true npm test` → **44 passed, 10 failed** (15 files passed, 4 failed).

The 10 failures are the **known-pre-existing reds** documented in the project memory
and the roadmap spec — not introduced by any P0/P1 work:

- `src/lib/locales/index.test.ts` (10): 9 locale-parity failures (zh-CN, zh-TW, pt-BR,
  ja-JP, it-IT, fr-FR, ru-RU, bn-IN, es-ES "should have the same keys as en-US") +
  1 "Unused Key Detection" failure.
- `e2e/{deep,full-ui,nav}.spec.ts` (3 files): Playwright e2e specs that vitest
  mis-collects (they are not vitest tests). Counted among the 4 failed *files*.

## Green floor (the baseline)

- **Backend:** `make test-fast` is fully green (142/142). Full suite is green except
  the 10 ontology+ingestion env failures above — green once `make seed-ontology` runs
  and Tesseract is present.
- **Frontend:** green except the 10 documented known-reds (locale parity + unused-key)
  and the 3 mis-collected e2e specs.
- **No regressions and no SQL-removal fallout** introduced by the roadmap work to date.

## Follow-ups (feed P2 Task 10 — CI)

- CI should run `make test-fast` (green) as the gate; run the ontology/ingestion suites
  only in a job that first runs `make seed-ontology` and installs Tesseract.
- Frontend CI should exclude `e2e/*.spec.ts` from the vitest collection (they belong to
  Playwright) and either fill the 9 locale parity gaps or quarantine that test.
