# Chat Answer Polish + Speed — Design

> 4 phases A → B → C → D, each shippable independently.
> Bundles S7 (speed) + S8 (token cap) + S9 (Ask&Search wiring) +
> S10 (inline images) + S12 (NotebookLM citations) + S13 (follow-ups).
> S11 (image generation) explicitly dropped.

## Phase A — Output token cap (S8)

**Problem.** Ollama defaults `num_predict=128`; cloud OpenAI defaults `max_tokens=1024`. "List 10 differences" type answers get truncated mid-sentence.

**Change.**

```python
# src/agentrag/config.py
AGENT_MAX_OUTPUT_TOKENS: int = 131072   # 128K — model self-caps in practice
```

Pass `max_tokens=settings.AGENT_MAX_OUTPUT_TOKENS` to every `chat.completions.create()` in:

- `src/agentrag/agent/llm.py` — `json_response`, `text_response`, `stream_text`
- `src/agentrag/services/llm_gateway.py` — `vision_response`

Add to `.env` + `.env.example`:

```env
# Per-call output ceiling. 128K = let model self-cap. Lower (e.g. 8192)
# for fast local Ollama on small models.
AGENT_MAX_OUTPUT_TOKENS=131072
```

**Acceptance.** "Liệt kê 30 thuốc kháng viêm" → assistant response ≥ 4096 tokens, no mid-sentence truncation.

---

## Phase B — NotebookLM-style answer polish (S10 + S12 + S13)

Three intertwined features on the same render path; bundle.

### B.1 Citation surface (S12)

`agent/service.py::_ground_citations()` currently returns
`{document_title, section_path, page, excerpt, content_hash}`. Extend:

```python
{
  "document_title": str,
  "section_path":   str | None,
  "page":           int | None,
  "page_label":     str | None,     # "p.47"
  "excerpt":        str,            # 300 chars, **bold** preserved
  "content_hash":   str,
  "mime":           str | None,     # text/markdown | application/pdf | …
  "segment_type":   str,            # text | image | table
  "image_url":      str | None,     # /api/images/foo.jpg when segment_type=image
}
```

`excerpt` widened to 300 chars and bold-wrapped around the matched
query terms (cheap regex against the question words). Frontend can
render Markdown safely; KaTeX inline-math survives because the bold
wrapping skips `$...$` segments.

### B.2 Follow-up suggestions (S13)

New module `src/agentrag/agent/followups.py`:

```python
async def generate_followups(
    question: str,
    answer: str,
    citations: list[dict],
    llm_gateway: LLMGateway,
) -> list[str]:
    """One LLM call → up to 3 short follow-up questions in Vietnamese.
    Cached 5 min by (question, answer hash). Returns [] on failure."""
```

LLM task name `"followup"` so users can route to a cheap model via
`LLM_TASK_MODEL_MAP` (e.g. `{"followup":"llama3.2:3b"}`). Prompt:

```
You are a helpful study tutor. Given QUESTION and ANSWER, propose 3
short follow-up questions the student might ask next. Vietnamese.
Return strict JSON: {"follow_ups": ["...", "...", "..."]}. Max 80 chars
per question.
```

Cache: `cachetools.TTLCache(maxsize=512, ttl=300)` keyed by
`hashlib.sha256(question + answer).hexdigest()`. Reuses the same
pattern as `_DIRECT_RAG_CACHE` in `adapter/routers/chat.py`.

### B.3 Adapter integration

`adapter/routers/chat.execute_chat` after the assistant `append_message`:

```python
follow_ups = await generate_followups(...)
# persist on extra_metadata.follow_ups so list_messages surfaces it
```

`ChatMessage` adapter model gains `follow_ups: list[str] | None = None`.
`_msg_to_chat` pulls from `extra_metadata.follow_ups`.

`Citation` Pydantic also extended with the new fields above.

### B.4 Frontend

**Types** (`frontend/src/lib/types/api.ts`):

```ts
export interface Citation {
  document_title: string
  section_path?: string | null
  page?: number | null
  page_label?: string | null
  excerpt: string
  content_hash: string
  mime?: string | null
  segment_type?: 'text' | 'image' | 'table'
  image_url?: string | null
}

export interface NotebookChatMessage {
  // …existing
  follow_ups?: string[]
}
```

**Components** (`frontend/src/components/source/`):

- `CitationHoverCard.tsx` — Radix HoverCard (add `@radix-ui/react-hover-card`
  to `package.json`). Layout:
  ```
  ┌──────────────────────────────────────────┐
  │ Lec 11.S2.4.MD …               [p.47]    │
  │ Chương 3 / Hệ tim mạch                   │
  ├──────────────────────────────────────────┤
  │ Van **hai lá** nằm giữa tâm nhĩ trái     │
  │ và tâm thất trái. Nó gồm hai lá van …    │
  └──────────────────────────────────────────┘
  ```
  When `image_url` present, replace the excerpt body with a thumbnail
  (max-h-32, rounded). For `mime: text/markdown` render the excerpt
  via the existing `ReactMarkdown + remark-math + rehype-katex` chain
  so KaTeX formulas survive.

- `FollowupChips.tsx` — three pill buttons under the AI bubble. Click →
  `onSendMessage(chipText)`. Disabled while streaming.

- `InlineImageCitation.tsx` — when an answer cites an image segment,
  render the image inline near the citation marker as a small floating
  card with caption (`<aside class="float-right …"><img …/></aside>`).
  Image segments don't appear in the popover — they appear in the body.

**Citation rendering pipeline** (`utils/source-references.ts` exists):

- Keep `[1]`-style markers in the markdown answer.
- Map each `[N]` to a `<CitationHoverCard citation={…}/>` via the
  existing `convertReferencesToCompactMarkdown` util — swap the
  `LinkComponent` factory in `ChatPanel.AIMessageContent` for
  `HoverCardComponent`.
- Click does nothing; pure hover.

**ChatPanel changes**:

```tsx
{message.type === 'ai' && (
  <>
    <AIMessageContent
      content={message.content}
      citations={message.citations}
      onCitationHover={...}      // new
    />
    {message.follow_ups?.length ? (
      <FollowupChips
        suggestions={message.follow_ups}
        onSelect={(q) => onSendMessage(q, modelOverride)}
        disabled={isStreaming}
      />
    ) : null}
  </>
)}
```

### Acceptance B

1. Ask any chat question → hovering each `[N]` in the answer shows
   tooltip with section + page + excerpt.
2. If retrieval surfaces an image segment, an inline thumbnail renders
   in the answer body with caption.
3. Three follow-up chips appear under the AI bubble; clicking sends.
4. Citation containing a KaTeX formula renders the formula (no
   `$x^2$` literal text).
5. Re-asking the same question → follow-ups served from cache (no
   extra LLM call).

---

## Phase C — Latency (S7)

Target: ≤10s p50, ≤20s p95 on local Ollama qwen 7B.

**Three config-only levers (Phase C-minimal):**

1. **Smaller decide model.** Default `LLM_TASK_MODEL_MAP` to:
   ```json
   {
     "classify": "llama3.2:3b",
     "decide": "llama3.2:3b",
     "domain_router": "llama3.2:3b",
     "followup": "llama3.2:3b",
     "answer": "qwen-agentrag"
   }
   ```
   Decide loop drops from 4×8s = 32s to 4×1.5s = 6s. Biggest single win.

2. **`AGENT_MAX_STEPS=2`** (default was 4). With reasonable first-shot
   retrieval, 2 steps is enough. Doc the trade-off in README.

3. **`AGENT_PLAN_TRIGGER_MIN_CHARS=120`** (default 60). Skip planner for
   most queries.

**Phase C-streaming (optional, follow-up PR):**

`/chat/execute` is sync POST. Move to SSE `/chat/execute/stream` that
yields token-by-token + final citations event. Frontend `useNotebookChat`
needs SSE consumer (mirror `useAsk`). Defer unless config levers alone
miss the target.

### Acceptance C

Median of 10 chat turns on local qwen-agentrag + llama3.2:3b decide
≤10s end-to-end. Measured via `/cost` dashboard p50 latency on
`task=answer`.

---

## Phase D — Ask & Search UI wiring (S9)

`/search` page exists; `useAsk` SSE consumer exists. Verify + polish:

1. Backend `/api/search/ask` returns the same `citations` + `follow_ups`
   shape (reuse Phase B output).
2. Frontend `StreamingResponse` component renders citations via the
   new `CitationHoverCard` (shared component).
3. Add follow-up chips to Ask result, same `FollowupChips` component.
4. Add a "Use as notebook source" button (already in
   `SaveToNotebooksDialog`) — just verify wired.

### Acceptance D

User runs Ask on the Knowledge Base, gets an answer with hover
citations + follow-up chips identical to the chat experience.

---

## Out of scope

- Image generation (S11 dropped per user request).
- True streaming chat unless Phase C-minimal fails (Phase C-streaming
  is a follow-up).
- Citation page-render via pdf.js (HoverCard excerpt is enough; the
  "Open original" button on SourceDetail already covers full PDF
  view).

## File map

| Path | Action | Phase |
|---|---|---|
| `src/agentrag/config.py` | + `AGENT_MAX_OUTPUT_TOKENS` | A |
| `src/agentrag/agent/llm.py` | pass `max_tokens` | A |
| `src/agentrag/services/llm_gateway.py` | pass `max_tokens` in vision | A |
| `.env`, `.env.example` | document new vars | A |
| `src/agentrag/agent/service.py::_ground_citations` | enrich citation shape | B |
| `src/agentrag/agent/followups.py` | NEW — generate_followups + TTL cache | B |
| `src/agentrag/adapter/models.py` | extend Citation + ChatMessage models | B |
| `src/agentrag/adapter/routers/chat.py` | invoke + persist follow_ups | B |
| `src/agentrag/chat/history.py` | round-trip follow_ups via extra_metadata | B |
| `frontend/package.json` | `@radix-ui/react-hover-card` | B |
| `frontend/src/lib/types/api.ts` | extend Citation + NotebookChatMessage | B |
| `frontend/src/components/source/CitationHoverCard.tsx` | NEW | B |
| `frontend/src/components/source/FollowupChips.tsx` | NEW | B |
| `frontend/src/components/source/InlineImageCitation.tsx` | NEW | B |
| `frontend/src/components/source/ChatPanel.tsx` | wire 3 new comps | B |
| `frontend/src/lib/utils/source-references.ts` | swap link → hover factory | B |
| `.env`, `.env.example` | `LLM_TASK_MODEL_MAP` example, `AGENT_MAX_STEPS=2`, `AGENT_PLAN_TRIGGER_MIN_CHARS=120` | C |
| `README.md` §5.6 | document the new task routing defaults | C |
| `frontend/src/app/(dashboard)/search/page.tsx` | render new citation + chips | D |
| `frontend/src/components/search/StreamingResponse.tsx` | use shared comps | D |

## Tests

- `tests/agent/test_followups.py` — JSON parse, cache hit, []-on-failure.
- `tests/adapter/test_citation_shape.py` — extended Citation Pydantic.
- `tests/integration/test_followup_persistence.py` — follow_ups round-trip via DB.
- Manual: load chat, hover `[1]`, confirm popover. Click chip → new turn.
- Manual: image citation → inline thumbnail rendered.

## Roll-back plan

- Phase A: revert `AGENT_MAX_OUTPUT_TOKENS` to 1024 default.
- Phase B: feature flag `CHAT_FOLLOWUPS_ENABLED=false` skips the LLM
  call; UI hides chips when array empty (no flag needed). Hover card
  is purely additive — disabling renders the existing reference markers.
- Phase C: revert config to old defaults.
- Phase D: separate PR per route.
