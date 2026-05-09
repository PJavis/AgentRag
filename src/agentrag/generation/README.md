# Module: `generation` — Mindmap & Structured Summary

**Vị trí:** `src/agentrag/generation/`

Sinh artifacts học tập từ document đã ingest: mindmap (Mermaid) và structured summary theo template y khoa. Hai service độc lập, đều retrieve chunks từ Elasticsearch rồi gọi LLM một lần (không qua agent loop).

---

## Files

| File | Class | Mô tả |
|---|---|---|
| `mindmap_service.py` | `MindmapService` | Sinh Mermaid mindmap + concept hierarchy, cache TTL 24h |
| `summary_service.py` | `SummaryService` | Sinh tóm tắt cấu trúc theo style: `study_note` / `clinical` / `quick_review` |

---

## `MindmapService`

```python
service = MindmapService()
result = await service.generate(
    document_title="giai_phau_lam_sang",
    focus_topic="hệ tim mạch",   # optional
    max_depth=3,
)
# {
#   "mermaid": "mindmap\n  root((Giải phẫu...))\n    Hệ tuần hoàn\n      Tim\n      Mạch máu",
#   "concepts": [{"name": "Tim", "parent": "Hệ tuần hoàn", "level": 2}, ...],
#   "cached": false,
# }
```

### Pipeline

```
generate(document_title, focus_topic, max_depth)
  │
  ├── cache lookup (in-process dict, key = "title|focus|depth", TTL 24h)
  │
  ├── ElasticsearchStore.sparse_search(top_k=30, document_title=...)
  ├── _build_context(chunks, max_chunks=30)   ← truncate content 600 chars
  │
  ├── LLMGateway.json_response(task="mindmap")
  │
  └── return {mermaid, concepts, cached}
```

**Empty document**: nếu không có chunks, trả mindmap stub `mindmap\n  root((Title))\n    Không tìm thấy nội dung`.

---

## `SummaryService`

Hai mode template, language-aware:

### Style: `study_note` / `clinical`

Iterate qua **medical template** 9 sections (Vietnamese):

```
Định nghĩa & Phân loại  →  Dịch tễ học  →  Nguyên nhân & Yếu tố nguy cơ
  →  Sinh lý bệnh  →  Triệu chứng lâm sàng  →  Cận lâm sàng & Chẩn đoán
  →  Điều trị  →  Biến chứng  →  Tiên lượng & Theo dõi
```

Mỗi section:
1. ES sparse_search với `query = document_title + " " + heading`, `top_k=8`
2. LLM call → `{summary, key_points, important_terms}`
3. Section trống (không liên quan) → bỏ qua

`asyncio.gather()` cho overview + 9 sections song song.

### Style: `quick_review`

Single LLM call với 30 chunks → tạo overview + sections gọn nhẹ trong 1 lần (không phân theo template).

### Response

```json
{
  "title": "...",
  "style": "study_note",
  "overview": "Bệnh hở van hai lá là...",
  "sections": [
    {
      "heading": "Sinh lý bệnh",
      "summary": "Van hai lá bị hở khiến...",
      "key_points": ["Trào ngược máu...", "..."],
      "important_terms": [{"term": "regurgitant fraction", "definition": "..."}]
    }
  ]
}
```

---

## API endpoints (`main.py`)

### `POST /generate/mindmap`

```bash
curl -X POST http://localhost:8000/generate/mindmap \
  -H "Content-Type: application/json" \
  -d '{"document_title": "giai_phau", "focus_topic": null, "max_depth": 3}'
```

### `POST /generate/summary`

```bash
curl -X POST http://localhost:8000/generate/summary \
  -d '{"document_title": "giai_phau", "style": "clinical"}'
# style: study_note | clinical | quick_review
```

---

## Tương tác

| Module | Vai trò |
|---|---|
| `ingestion.stores.ElasticsearchStore` | sparse_search lấy chunks theo document |
| `services.LLMGateway` | json_response (task="mindmap" hoặc "summary") |
| `main.py` | Expose `/generate/mindmap`, `/generate/summary` |

---

## Config liên quan

Không có config riêng — cả hai service dùng `ELASTICSEARCH_URL` + LLM provider mặc định. Nếu bật `LLM_ROUTING_ENABLED`, có thể map riêng task `mindmap` / `summary` trong `LLM_TASK_MODEL_MAP`.

```env
LLM_TASK_MODEL_MAP={"mindmap": "qwen2.5:7b-instruct", "summary": "gpt-4o"}
```

---

## Notes

- **Cache invalidation**: gọi `MindmapService().invalidate(document_title)` sau khi re-ingest document. Hiện tại cache là in-process dict, không persist qua restart.
- **Highlights** (3-5 điểm quan trọng nhất trong câu trả lời chat) được sinh trực tiếp trong agent answer prompt và adapter `_direct_rag` — không qua module này.
