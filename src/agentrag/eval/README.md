# eval — offline RAG quality evaluation harness (retrieval, answer, chunking, benchmarks)

## Mục đích / Purpose
Thư viện đo chất lượng RAG **offline**, chạy bởi scripts ngoài (`scripts/eval/*`), không phải đường đi request runtime. Nó cung cấp: (1) golden-dataset schema + generator, (2) retrieval metrics (recall@K / MRR / NDCG), (3) answer-quality LLM-as-judge, (4) chunking-quality analysis trên ES index, (5) DeepEval 5-metric harness, (6) RAGAS row mappers, (7) HuggingFace benchmark-dataset loaders, và (8) freshness probe (re-ingest → stale chunk phải bị fresh out-rank). Mục tiêu là cổng chất lượng (quality gates) trước khi merge/release, và ablation so sánh các flag RAG.

These are pure/async helper functions + dataclasses. They are imported by CLI runners in `scripts/eval/`; nothing in the serving path imports this package.

## Plane
**Infrastructure / offline tooling.** Không thuộc Reasoning hay Execution Plane: không phục vụ request, không đăng ký vào `ServiceContainer`, không đọc `settings.*`. Nó *gọi vào* các plane khác (retriever, llm_gateway, ingestion pipeline, ES client) thông qua các đối tượng được truyền vào từ caller.

## Key files
| File | Responsibility |
|------|----------------|
| `dataset.py` | `GoldenQuestion` / `GoldenDataset` dataclasses (load/save JSON ở `data/eval/<title>.json`) + `generate_golden_dataset(...)` — dùng LLM sinh QA pairs từ nội dung tài liệu. |
| `retrieval_eval.py` | `evaluate_retrieval_mode(retriever, questions, mode, top_k, ...)` → `RetrievalModeReport`. Tính recall@{1,3,5,10}, MRR, NDCG, latency p95 per mode (sparse/dense/hybrid/hybrid_kg). Relevance = section match HOẶC keyword match. |
| `answer_eval.py` | `evaluate_answer(...)` LLM-as-judge cho 4 chiều (faithfulness, answer_relevance, context_precision, correctness) → `AnswerScore`; `aggregate_answer_scores(...)` → `AnswerEvalReport` (gồm hallucination_rate). |
| `chunking_eval.py` | `evaluate_chunking(es_client, index_name, ...)` → `ChunkingReport`. Kéo chunks từ ES, tính short-chunk rate, length percentiles, section coverage, dedup rate (theo `content_hash`). |
| `deepeval_metrics.py` | DeepEval harness cho 5 metric LLM-judged (`build_judge`, `build_metrics`, `score_cases`) + `METRIC_TARGETS` thresholds. Judge mặc định Gemini 2.5 Flash; chọn được deepseek/openai. |
| `ragas_eval.py` | Pure mappers `build_ragas_row(...)` / `extract_context_texts(...)` → RAGAS sample dict. **Không** import ragas/langchain (xem Gotchas). |
| `benchmark_datasets.py` | HF dataset loaders (`load_eval_examples`, `load_suite`) → `EvalExample`. Registry `DATASETS` (vn_bkai/vn_legal/en_covidqa/en_pubmedqa) + `SUITES` (vn/en/both). |
| `freshness.py` | `run_freshness_check()` — ingest v1(stale)→v2(fresh) cùng title, query, pass nếu fresh out-ranks stale. Có side-effect lên ES (eval metric #9). |
| `__init__.py` | Rỗng — không re-export; import trực tiếp theo submodule. |

## Public interface
Tất cả truy cập là **direct import theo submodule** (không qua `ServiceContainer`, không qua Protocol). Dependency bên ngoài được **dependency-injected qua tham số** (caller dựng và truyền vào):

- `dataset.generate_golden_dataset(document_title, document_content, llm_gateway, n_questions=15, content_preview_chars=8000) -> GoldenDataset` — `llm_gateway` phải có `async json_response(system_prompt, user_prompt, task) -> (dict|list, latency)`.
- `dataset.GoldenDataset.load(path) / .save(path)`, `GoldenQuestion` dataclass.
- `retrieval_eval.evaluate_retrieval_mode(retriever, questions, mode, top_k=10, document_title=None) -> RetrievalModeReport` — `retriever` phải có `async search(query, mode, top_k, document_title) -> {"results": [...]}`; mỗi result có `rank`, `section_path`, `content`.
- `answer_eval.evaluate_answer(question_id, question, answer, packed_context, expected_answer, llm_gateway) -> AnswerScore` (task `"eval_judge"`); `aggregate_answer_scores(list[AnswerScore]) -> AnswerEvalReport`.
- `chunking_eval.evaluate_chunking(es_client, index_name, document_title=None, min_chars=80, max_chunks=5000) -> ChunkingReport` — `es_client` là async Elasticsearch client.
- `deepeval_metrics.build_judge(provider="gemini", model=None, api_key=None)`, `build_metrics(judge)`, `score_cases(cases: list[CaseInput], judge) -> {"summary", "per_case"}`, `CaseInput`, `METRIC_TARGETS`.
- `ragas_eval.build_ragas_row(*, question, answer, context_items, ground_truth="") -> dict`, `extract_context_texts(context_items) -> list[str]`.
- `benchmark_datasets.load_suite(suite, n=30)`, `load_eval_examples(name, n=30)`, `normalize_row(...)`, `EvalExample`, `DATASETS`, `SUITES`.
- `freshness.run_freshness_check() -> {"pass", "fresh_rank", "stale_rank", "detail"}` — tự import `ingestion.pipeline.ingest_folder` và `retrieval.elasticsearch_retriever.ElasticsearchRetriever`.

Mọi report dataclass có `.as_dict()` để serialize ra JSON; `RetrievalModeReport`/`ChunkingReport` còn có summary helpers.

## Data flow
**Callers (scripts/eval/):**
- `run_eval.py` → `evaluate_chunking`, `evaluate_retrieval_mode`, `evaluate_answer`, `aggregate_answer_scores`, `GoldenDataset` (cổng chất lượng chính, per-document).
- `generate_dataset.py` → `generate_golden_dataset`.
- `run_benchmark.py` → `load_suite` (benchmark_datasets) + `build_judge`/`score_cases`/`CaseInput`/`METRIC_TARGETS` (deepeval) + `extract_context_texts` (ragas) + `run_freshness_check` (freshness) = 9-metric benchmark.
- `run_ablation.py` → subprocess sweep gọi `run_benchmark.py` một lần / cấu hình flag, đọc `METRIC_TARGETS` keys để tabulate.
- `run_ragas.py` → `GoldenDataset` + `build_ragas_row`, dump rows JSON cho `score_ragas.py` (venv riêng).

**Flow điển hình (run_benchmark):** load golden/HF examples → ingest `gold_contexts` vào ES → chạy agent `chat()` → đóng gói thành `CaseInput` (question, actual_output, expected_output, retrieval_context) → `score_cases` chạy 5 DeepEval metrics qua judge LLM → trả summary (mean / pass_rate / meets_target). Retrieval và answer metrics đọc shape result/packed_context mà retrieval + agent layers phát ra.

**Downstream deps:** retriever Protocol shape (`search`), llm gateway (`json_response`), ES client, `ingestion.pipeline`, `retrieval.elasticsearch_retriever`, các thư viện ngoài `deepeval` / `datasets` (HF).

## Config
**Module này KHÔNG đọc `settings.*` nào.** Đã verify: không có import `config`/`settings`/`ServiceContainer` trong package. Tham số (top_k, min_chars, judge provider/model/api_key) được truyền trực tiếp hoặc qua CLI args của runner. Các RAG flag (`CONTEXTUAL_RETRIEVAL_ENABLED`, `RAPTOR_ENABLED`, `CRAG_ENABLED`, `ADAPTIVE_ROUTING_ENABLED`, `SEMANTIC_CACHE_ENABLED`, `AGENT_MULTIHOP_ENABLED`) được `scripts/eval/run_ablation.py` set qua **environment của subprocess con** — chúng tác động tới ingestion/agent đang được đo, không phải tới code eval. RAGAS judge/embedding config nằm ở `scripts/eval/score_ragas.py` (venv cô lập).

## Recent additions (2026-06)
Toàn bộ workstream RAG mới (Contextual Retrieval, RAPTOR, CRAG, adaptive routing, semantic cache, agent multi-hop) được **đo từ bên ngoài** qua eval module này — eval code không tham chiếu trực tiếp các flag/đường đi đó, nhưng các file mới của module phục vụ chính việc đo chúng:
- `benchmark_datasets.py` — loaders 2 ngôn ngữ (VN sailor2 + EN ragbench), suites vn/en/both.
- `deepeval_metrics.py` — 5-metric judged harness (recall, precision, faithfulness, answer_correctness, citation_accuracy) với `METRIC_TARGETS`.
- `ragas_eval.py` — pure RAGAS row mappers, decoupled khỏi langchain.
- `freshness.py` — re-ingest freshness probe (metric #9).
Ablation matrix (`run_ablation.py`) re-ingest + re-score mỗi cấu hình flag để so sánh trung thực; `STRUCTMEM_INGEST_MODE=sync` được ép trong child để extraction hoàn tất trước khi chấm. (Tham chiếu commit gần đây: báo cáo benchmark VN+EN 06/06, so sánh StructMem/KG có-vs-không.)

## Gotchas
- **RAGAS venv split:** `ragas_eval.py` chỉ chứa pure mappers — KHÔNG import `ragas`/`langchain`. Lý do: RAGAS hard-require `langchain-core <0.3` xung đột với `langchain-core 1.x` của `langgraph`. Eval RAGAS là 2 bước: `run_ragas.py` (venv app) dump rows JSON → `score_ragas.py` (venv cô lập) chấm. Đừng thêm import ragas vào file này.
- **Hai import path:** runner cũ `run_eval.py` và `generate_dataset.py` import qua alias `src.pam.eval.*`; runner mới (`run_benchmark.py`, `run_ablation.py`, `run_ragas.py`) dùng `src.agentrag.eval.*`. Cùng package, alias legacy — đừng nhầm là hai module.
- **freshness.py có side-effect & cần live stack:** nó thực sự ingest 2 lần vào ES rồi query; phải đóng `retriever.store.client`. Chỉ chạy trên live stack, không phải CI.
- **`score_cases` nuốt lỗi judge per-case:** một metric lỗi (JSON judge fail, v.v.) ghi `"ERR:<Exception>"` cho case đó và bị loại khỏi mean — toàn run không crash, nhưng `n` của metric có thể nhỏ hơn số case. Kiểm tra `n` trong summary.
- **Relevance trong retrieval_eval là binary, OR-logic & substring:** `_is_relevant` match `section_path` (normalize space/underscore) HOẶC keyword substring trong `content`. NDCG/recall do đó nhạy với cách viết `relevant_sections`/`relevant_keywords` của golden question — sai section name → metric tụt giả.
- **p95 dùng index `int(n*0.95)-1`:** xấp xỉ thô (không nội suy); với n nhỏ chỉ mang tính chỉ báo.
- **`load_eval_examples` skip rows thiếu question hoặc thiếu gold_contexts** — số example trả về có thể < `n` yêu cầu. Stream từ HF (cần mạng + `datasets`).
- **answer_eval cap context ở 6 chunks / 1500 chars mỗi chunk** (`_format_context`); judge không thấy context đầy đủ nếu packed_context dài hơn.
- **`evaluate_chunking` query ES theo `document_title.keyword`** và cap `max_chunks=5000` — index lớn hơn sẽ bị cắt, metric chỉ phản ánh phần đầu.
