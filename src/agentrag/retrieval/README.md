# retrieval — Hybrid search engine (BM25 + dense kNN + StructMem/RAPTOR) with rerank, RRF fusion & caching

## Mục đích / Purpose
Đây là lõi truy hồi (retrieval) của AgentRag: nhận một query, chạy tìm kiếm trên
Elasticsearch theo 4 chế độ (sparse / dense / hybrid / hybrid_kg), hợp nhất nhiều
nguồn bằng Reciprocal Rank Fusion (RRF), rerank ứng viên, rồi cân bằng/ giới hạn
các loại segment trước khi trả về. Module này thuần IO + heuristic ranking — không
chứa prompt hay quyết định agent. Nó cũng cung cấp tiện ích viết lại query (HyDE,
decompose) và ContextVars truyền filter/scope theo từng lượt chat.

## Plane
**Execution Plane.** Stateless IO worker: nó gọi Elasticsearch + embedder + reranker
và trả dữ liệu, không ra quyết định reasoning. Routing theo domain do Reasoning Plane
(`DomainRouter`) làm rồi truyền `system_override`/`specialty_override` xuống đây.
Ngoại lệ nhỏ: `context.py` là cơ chế truyền tham số per-turn (infrastructure-style
ContextVars), không phải logic quyết định.

## Key files
| File | Responsibility |
|---|---|
| `elasticsearch_retriever.py` | `ElasticsearchRetriever` — orchestrator chính: 4 search mode, RRF fusion đa nguồn, rerank, dedupe, image-ratio cap, RAPTOR summary cap, 60s `_RESULT_CACHE`, semantic-cache wrapper, domain-filter fallback. |
| `reranker.py` | `LLMReranker` — rerank ứng viên qua local cross-encoder (mặc định), LLM chat (OpenAI-compat), hoặc Ollama native `/api/rerank`. Trả `(hits, reranked, reason)`. |
| `federated.py` | `FederatedRetriever` — wrapper filter-only quanh `ElasticsearchRetriever`; dịch override thành filter clauses; router opt-in (mặc định `None`). |
| `query_rewriter.py` | `QueryRewriter` — HyDE (`make_hyde_text`), decomposition multi-hop (`decompose`), soft-HyDE augment (`augment_with_hyde`). |
| `context.py` | ContextVars per-turn: `set/get_domain_filter`, `set/get_document_scope`. Async-safe, không cần thread kwargs khắp cây gọi. |

## Public interface
Reasoning code KHÔNG instantiate `ElasticsearchRetriever` trực tiếp. Đường vào chuẩn:

- **`ServiceContainer.retrieval`** → `services/retrieval_service.py::RetrievalService`
  (facade), satisfies `services/protocols.py::RetrievalProtocol`. Chuỗi wrap:
  `RetrievalService` → `FederatedRetriever` → `ElasticsearchRetriever`.
- **`RetrievalService.search(query, *, document_title, top_k, mode="hybrid_kg",
  rerank, dense_query, filters, system_override, specialty_override) -> dict`** —
  gộp `filters={"systems":[...],"specialties":[...]}` HOẶC override S5 UI rồi gọi
  `FederatedRetriever`.
- **`ElasticsearchRetriever.search(query, mode="hybrid_kg", top_k, document_title,
  rerank, dense_query, filters) -> dict`** — entrypoint thực; `mode` ∈
  `{sparse, dense, hybrid, hybrid_kg}` (giá trị khác → `ValueError`).
- **`LLMReranker.maybe_rerank(query, candidates, top_k, force) -> (hits, reranked, reason)`**
  và `candidate_size(requested_top_k, force)` — được retriever gọi nội bộ.
- **`QueryRewriter(llm_gateway)`** — khởi tạo với một `LLMGateway`; gọi từ
  `services/knowledge_service.py` (HyDE/decompose), không qua container.
- **`context.set_document_scope / set_domain_filter`** — set ở
  `adapter/routers/chat.py` và `agent/graph_service.py`; đọc ở `agent/tools.py`
  khi gọi retriever cho lượt chat hiện tại.

Kết quả `search()` là dict: `results` (list hit), `mode`, `top_k`, `document_title`,
`reranked`, `rerank_requested`, `rerank_reason`, và (mode hybrid_kg) `graph_reason`.
Khi fallback có thể thêm `domain_filter_fallback` / `domain_filter_attempted`;
semantic-cache hit thêm `semantic_cache_hit: True`.

## Data flow
**Upstream callers:** `KnowledgeService` (tool dispatch, bootstrap search, HyDE/
decompose), agent graph nodes, và endpoint `/search` — tất cả qua `RetrievalService`.

**Trong `_search_uncached` (pipeline):**
1. Check `_RESULT_CACHE` (TTL 60s, key = sha256 của query/mode/top_k/scope/rerank/filters/dense_query).
2. `_rewrite_query` (chỉ bơm từ khoá "features/tính năng" cho query đặc thù) → BM25 text.
3. Embed `dense_query` (HyDE-augmented) hoặc `query` cho kNN.
4. `candidate_size()` nới pool lên `RETRIEVAL_RERANK_TOP_N` nếu rerank bật.
5. Gọi store: `sparse_search` / `dense_search` / `hybrid_search`
   (`ingestion/stores/elasticsearch_store.py`).
6. **hybrid_kg:** nếu `_should_use_graph(query)` → `_entries_search` chạy song song
   `search_entries` + `search_synthesis` (cả hai đều trên `agentrag_memory_doc` — unified STRUCTMEM_INDEX, R4 collapse),
   rồi `_rrf_fuse_multi_source` chunk + structmem (k=`RETRIEVAL_RRF_K`).
7. **Multi-vector (P2.9):** nếu `VISUAL_EMBEDDING_ENABLED` và `_is_image_intent(query)`
   → `store.visual_search` (CLIP kNN) rồi RRF-fuse vào pool.
8. `_dedupe_hits` (theo `content_hash` / fingerprint 40 token đầu) →
   `_rerank_hits` → `_apply_query_intent_ranking` (chỉ features-query) →
   `_balance_segment_types_for_query` (cap ảnh) → `_cap_summary_nodes` (cap RAPTOR) →
   `_finalize_ranks` (gán `rank` 1..n, giữ `retrieval_rank` cũ).

**Downstream deps:** `ingestion/stores/elasticsearch_store.ElasticsearchStore` (index),
`ingestion/embedders/factory.build_embedding_provider` (embed query),
`graph/structmem_service.StructMemService` (normalize group_id theo `document_title`),
`services/semantic_cache.SemanticCache`, `common/langfuse_client.make_async_openai`
(reranker client).

## Config
| Key | Default | Tác dụng |
|---|---|---|
| `RETRIEVAL_TOP_K` | `10` | Số kết quả trả về (khi `top_k=None`). |
| `RETRIEVAL_NUM_CANDIDATES` | `50` | Candidate pool ES (dùng trong store). |
| `RETRIEVAL_RRF_K` | `60` | Hằng số k cho RRF fusion đa nguồn. |
| `RETRIEVAL_RERANK_ENABLED` | `false` | Bật rerank mặc định (param `rerank` override per-call). |
| `RETRIEVAL_RERANK_TOP_N` | `20` | Số ứng viên đưa vào reranker / cận pool. |
| `RETRIEVAL_RERANK_BACKEND` | `local_cross_encoder` | `local_cross_encoder` (bge-reranker-v2-m3, free) hoặc `llm_chat`. |
| `RETRIEVAL_RERANK_PROVIDER` | `None` | Provider cho `llm_chat` (`openai`/`gemini`/`hf_inference`/`ollama`); fallback `AGENT_PROVIDER` / `EXTRACTION_PROVIDER`. |
| `RETRIEVAL_RERANK_MODEL` | `None` | Model reranker; fallback `AGENT_MODEL`/`EXTRACTION_MODEL`. |
| `RETRIEVAL_RERANK_BASE_URL` | `None` | Override base URL reranker. |
| `RETRIEVAL_RERANK_TEMPERATURE` | `0.0` | Nhiệt độ cho `llm_chat` rerank. |
| `RETRIEVAL_MAX_IMAGE_RATIO` | `0.3` | Tỉ lệ tối đa segment ảnh trong top_k (nới 0.7 khi query đòi hình). |
| `VISUAL_EMBEDDING_ENABLED` | `true` | Bật nhánh multi-vector CLIP visual kNN. |
| `RAPTOR_SUMMARY_MAX_RATIO` | `0.4` | Tỉ lệ tối đa node summary (`node_level>=1`) trong kết quả. |
| `DOMAIN_FILTER_ENABLED` | `true` | Nếu `false`, `FederatedRetriever` bỏ mọi domain filter. |
| `SEMANTIC_CACHE_ENABLED` | `false` | Bật semantic cache (chỉ cho query mặc-định-scope, không filter/document). |
| `SEMANTIC_CACHE_THRESHOLD` | `0.97` | Ngưỡng cosine để coi là cache hit. |
| `SEMANTIC_CACHE_TTL_SECONDS` | `120` | TTL semantic cache. |
| `SEMANTIC_CACHE_MAX_ITEMS` | `256` | Số entry tối đa. |

> Module này KHÔNG đọc `CONTEXTUAL_RETRIEVAL_ENABLED` — Contextual Retrieval xảy ra
> ở ingestion (`ingestion/contextualizer.py`); retrieval chỉ search text đã giàu context.

## Recent additions (2026-06)
Tất cả mặc định-OFF trừ khi ghi chú khác:
- **Semantic retrieval cache** (`SEMANTIC_CACHE_ENABLED`): `search_cached` embed query
  rồi tra `SemanticCache`; chỉ áp cho query không filter & không `document_title`
  (tránh rò scope chéo). Hit gắn `semantic_cache_hit: True` (UI shim đọc cờ này).
- **RAPTOR summary cap** (`RAPTOR_SUMMARY_MAX_RATIO`, default 0.4 — luôn áp): `_cap_summary_nodes`
  giữ hit có `node_level>=1` ≤ ratio để một query không trả toàn summary; phần thừa
  bị đẩy xuống cuối làm backfill. (Citation mang `node_level`/`context_text` từ store.)
- **Local cross-encoder rerank** giờ là backend mặc định (`bge-reranker-v2-m3` qua
  `sentence-transformers`, lazy-load, chạy trong `asyncio.to_thread`); thay cho default
  `llm_chat` cũ. Thêm nhánh **Ollama native `/api/rerank`** khi provider là `ollama`.
- **Multi-vector / visual retrieval** (`VISUAL_EMBEDDING_ENABLED`, default ON): RRF-fuse
  CLIP visual kNN khi `_is_image_intent(query)`.

## Gotchas
- **Thứ tự post-processing là load-bearing:** dedupe → rerank → intent-rank → image
  cap → summary cap → finalize. `_finalize_ranks` gán lại `rank` cuối cùng; đừng giả
  định `rank` từ store còn đúng sau pipeline (giá trị gốc lưu ở `retrieval_rank`).
- **Domain-filter fallback giữ scope cứng:** khi filter có kết quả rỗng, `search()`
  retry chỉ relax soft filter (`systems`/`specialties`) NHƯNG không bao giờ bỏ
  `document_titles` — bỏ sẽ rò document của notebook/source khác vào chat đã scope.
  Nếu filter chỉ gồm `document_titles` (scope rỗng thật) thì KHÔNG fallback.
- **Semantic cache bị bypass** khi có `filters` hoặc `document_title`, và khi embed
  query lỗi (rơi về `_search_uncached`). Đừng dựa vào cache cho truy vấn có scope.
- **`_RESULT_CACHE` là module-level**, dùng chung mọi instance, TTL 60s. Cập nhật nội
  dung mới có thể bị che tối đa 60s với cùng query.
- **Rerank không bao giờ raise:** mọi lỗi (client, schema lạ, exception) → trả ranking
  gốc kèm `rerank_reason` mô tả (vd `reranker_exception:*`, `disabled_by_config`,
  `not_enough_candidates`). Kiểm `reranked: bool` để biết có rerank thật hay không.
- **`RETRIEVAL_RERANK_MODEL` chỉ bắt buộc cho backend `llm_chat`** — `_resolve_backend`
  raise `ValueError` nếu thiếu model/API key. Backend `local_cross_encoder` tự dùng
  `dengcao/bge-reranker-v2-m3` nếu không set model, không cần API key.
- **`dense_query` vs `query`:** `query` giữ "sạch" cho BM25 + rerank + intent ranking;
  `dense_query` (HyDE-augmented) chỉ dùng để embed cho kNN. Truyền nhầm sẽ làm hỏng
  keyword match.
- **`FederatedRetriever.router` mặc định `None`** (auto-routing đã tắt). Nhánh
  `_router.classify` chỉ chạy khi test/legacy chủ động inject router; production để
  Reasoning Plane (`DomainRouter`) routing rồi truyền override.
