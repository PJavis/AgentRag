# Module: `retrieval` — Hybrid Search Engine

**Vị trí:** `src/agentrag/retrieval/`

Hybrid search engine kết hợp BM25 (sparse), vector KNN (dense), và StructMem knowledge entries. Hỗ trợ reranking bằng LLM hoặc local cross-encoder.

---

## Files

| File | Class | Mô tả |
|---|---|---|
| `elasticsearch_retriever.py` | `ElasticsearchRetriever` | Orchestrator chính — 4 search modes, RRF fusion |
| `reranker.py` | `LLMReranker` | Reranking với LLM chat hoặc `sentence-transformers` |

---

## Search Modes

| Mode | Mô tả |
|---|---|
| `sparse` | BM25 full-text search trên `agentrag_segments` |
| `dense` | KNN vector search trên embedding field |
| `hybrid` | RRF fusion sparse + dense |
| `hybrid_kg` | RRF fusion sparse + dense + StructMem entries + synthesis |

Agent mặc định dùng `hybrid_kg`.

---

## `ElasticsearchRetriever`

### `search(query, mode, top_k, document_title, rerank) → dict`

```json
{
  "results": [
    {
      "content": "...",
      "document_title": "...",
      "section_path": "...",
      "position": 0,
      "content_hash": "...",
      "score": 0.95,
      "source": "hybrid | sparse | dense | structmem | synthesis"
    }
  ]
}
```

### Internal methods

| Method | Mô tả |
|---|---|
| `_sparse_search()` | BM25 trên `agentrag_segments` |
| `_dense_search()` | KNN trên embedding field |
| `_entries_search()` | Parallel search `agentrag_entries` + `agentrag_synthesis` |
| `_rrf_fuse()` | Reciprocal Rank Fusion, k=`RETRIEVAL_RRF_K` |

---

## `LLMReranker`

Rerank top candidates, giảm từ `RETRIEVAL_NUM_CANDIDATES` xuống `RETRIEVAL_RERANK_TOP_N`.

| Backend | Cơ chế |
|---|---|
| `llm_chat` | LLM chấm điểm relevance từng chunk (JSON response) |
| `local_cross_encoder` | `sentence-transformers` CrossEncoder chạy local |

---

## Tương tác

| Module | Vai trò |
|---|---|
| `ingestion.stores.ElasticsearchStore` | Index mà retriever search vào |
| `ingestion.embedders` | Embed query trước khi KNN search |
| `services.KnowledgeService` | Gọi `retriever.search()` qua tool dispatch |
| `main.py` | Expose trực tiếp qua `/search` endpoint |

---

## Config liên quan

| Key | Default | Mô tả |
|---|---|---|
| `RETRIEVAL_TOP_K` | `10` | Số kết quả trả về |
| `RETRIEVAL_NUM_CANDIDATES` | `50` | Candidate pool trước rerank |
| `RETRIEVAL_RRF_K` | `60` | RRF constant k |
| `RETRIEVAL_RERANK_ENABLED` | `false` | Bật reranking |
| `RETRIEVAL_RERANK_TOP_N` | `20` | Số results sau rerank |
| `RETRIEVAL_RERANK_BACKEND` | `llm_chat` | `llm_chat` hoặc `local_cross_encoder` |
| `RETRIEVAL_RERANK_MODEL` | (fallback AGENT_MODEL) | Model dùng cho reranker |
| `ELASTICSEARCH_URL` | `http://localhost:9200` | ES endpoint |
| `ELASTICSEARCH_INDEX_NAME` | `agentrag_segments` | Index cho chunks |
| `STRUCTMEM_ENTRIES_INDEX_NAME` | `agentrag_entries` | Index cho StructMem entries |
| `STRUCTMEM_SYNTHESIS_INDEX_NAME` | `agentrag_synthesis` | Index cho synthesis |
