# Module: `agents` — Multi-Agent Workers

**Vị trí:** `src/pam/agents/`

Framework multi-agent cho các câu hỏi phức tạp đòi hỏi thu thập dữ liệu, phân tích insight và tổng hợp báo cáo từ nhiều nguồn. Mỗi agent có trách nhiệm độc lập, chạy song song qua `asyncio.gather`.

---

## Files

| File | Class | Mô tả |
|---|---|---|
| `data_agent.py` | `DataAgent` | Thu thập dữ liệu — gọi semantic hoặc structured path tùy loại câu hỏi |
| `insight_agent.py` | `InsightAgent` | Trích xuất business insights từ dữ liệu đã thu thập |
| `report_agent.py` | `ReportAgent` | Tổng hợp nhiều insights thành báo cáo có cấu trúc |

---

## Luồng xử lý

```
complex question
  │
  ├──▶ decompose thành N sub-questions
  │
  ├──▶ asyncio.gather([DataAgent.run(q) for q in sub_questions])
  │         │ mỗi DataAgent:
  │         │   ├── intent = structured → StructuredReasoningPipeline
  │         │   └── intent = semantic   → AgentService.chat()
  │
  ├──▶ asyncio.gather([InsightAgent.extract(data) for data in results])
  │
  └──▶ ReportAgent.synthesize(all_insights) → structured report
```

---

## Tương tác

| Module | Vai trò |
|---|---|
| `agent.AgentService` | DataAgent gọi semantic path |
| `structured.StructuredReasoningPipeline` | DataAgent gọi SQL path |
| `services.LLMGateway` | InsightAgent + ReportAgent gọi LLM |

---

## Ghi chú

Module này phục vụ các use case phân tích chuyên sâu (so sánh nhiều tài liệu, báo cáo tổng hợp). Với câu hỏi thông thường, dùng `agent.AgentService` trực tiếp là đủ.
