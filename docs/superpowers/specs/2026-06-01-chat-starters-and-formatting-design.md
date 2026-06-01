# Chat Starters + Consistent Answer Formatting — Design

**Date:** 2026-06-01
**Branch:** feat/ragas-langfuse-reranker
**Status:** Approved pending user spec review

## Problem

When a chat session is empty (no messages yet), the user wants NotebookLM/ChatGPT-style
**starter suggestions** to pick from (e.g. "Summary", "Tell me about X in the document").
Today there are only 3 hardcoded chips in a small row above the input box.

Separately, long answers should **reliably** come out structured like ChatGPT/Claude/NotebookLM:
section headings, numbered lists, and a mix of bullet points + prose passages — not a wall
of text. The render pipeline already supports this; the model output is inconsistent.

## Goals

1. Replace the 3 hardcoded input-row chips with a NotebookLM-style **hero** starter area in the
   empty state, on **both** chat surfaces (single-source chat and multi-source notebook chat).
2. Starters are **hybrid**: 2 fixed universal chips (client-side, instant) + up to 3
   **document-aware dynamic** ones generated from existing summary/insights.
3. Strengthen the answer-formatting prompt so long answers consistently use headings +
   numbered lists + mixed bullets/passages.

## Non-Goals

- No change to the Markdown render pipeline — `ChatPanel.tsx:589-634` already renders
  h1–h6, `ol`/`ul`, tables, KaTeX, etc.
- No precompute-on-ingest path (no new DB column / ingest change). Starters are fetched on
  chat open and TTL-cached in-process.
- No change to the existing post-answer `follow_ups` feature (`followups.py`) — starters are a
  separate, parallel mechanism for the *empty* state.

## Decisions (from brainstorming)

| Question | Decision |
|----------|----------|
| Chip changes | Hero placement + richer set + document-aware dynamic |
| Generation | Hybrid: 2 fixed (FE) + 2–3 dynamic (LLM, cached) |
| Surfaces | Both source chat and notebook chat |
| Dynamic content source | Existing summary/insights (+ titles); fall back to title-only |
| Fetch timing | On chat open; backend TTL-cached per doc |
| Formatting issue | Inconsistent → strengthen backend prompt |

---

## Part A — Starter suggestions

### A1. Backend generator — `src/agentrag/agent/starters.py`

Mirror the structure of `src/agentrag/agent/followups.py`:

```
async def generate_starters(
    kind: str,                 # "source" | "notebook"
    titles: list[str],         # document title(s) / notebook name
    summary_text: str,         # existing summary or insights text, may be ""
    llm_gateway: Any,
) -> list[str]:
```

- **Cheap LLM call**, task name `"starter"` so it routes via `LLM_TASK_MODEL_MAP` to a small model.
- **Cache:** module-level `TTLCache(maxsize=512, ttl=...)`, key = `sha256(kind + titles + summary_text[:1500])`.
- **System prompt:** "Given the document/notebook title(s) and summary, propose up to 3 short
  starter questions a student might ask to begin exploring it. Each ≤ 80 characters.
  Vietnamese only. Return strict JSON `{\"starters\": [\"...\", ...]}`."
- **Failure / empty input → `[]`.** Never raises into the request path.
- If `summary_text` is empty, the call still runs on titles alone (lower specificity).

### A2. Endpoints

Both return only the **dynamic** list; fixed chips live on the frontend.

Response model (new, in `adapter/models.py`): `ChatStartersResponse { starters: list[str] }`.

- **Source:** `GET /api/sources/{source_id}/chat/starters` on `source_router`
  (`chat.py`, alongside the existing `/sources/{source_id}/chat/...` routes).
  - Load source title + its summary/insights (reuse whatever the source detail / summary
    service already exposes; read-only, no regeneration).
  - Call `generate_starters(kind="source", titles=[title], summary_text=summary)`.
- **Notebook:** `GET /chat/starters` on `notebook_router` (notebook id resolved the same way the
  other `notebook_router` routes resolve it).
  - Gather included source titles + notebook name; summary_text = concatenated source
    summaries/insights (truncated).
  - Call `generate_starters(kind="notebook", titles=[...], summary_text=...)`.

Both endpoints are best-effort: any failure returns `{ "starters": [] }` with 200, so the UI
silently falls back to fixed chips only.

### A3. Frontend API + hook

- `lib/api/source-chat.ts`: `getStarters(sourceId) -> Promise<{ starters: string[] }>`.
- `lib/api/chat.ts`: `getNotebookStarters(notebookId) -> Promise<{ starters: string[] }>`.
- New hook `lib/hooks/useChatStarters.ts`:
  ```
  useChatStarters({ kind: 'source' | 'notebook', id, enabled })
  ```
  - TanStack Query, `queryKey: ['chatStarters', kind, id]`.
  - `enabled` passed by caller — true only while the chat is empty (`messages.length === 0`).
  - Returns `{ starters, isLoading }`. Errors swallowed → `starters: []`.

### A4. Frontend UI — `ChatPanel.tsx`

ChatPanel stays **presentational**: parents own the query and pass results down.

New props:
```
dynamicStarters?: string[]
startersLoading?: boolean
```

**Empty-state hero (replaces L226–235):**
- Keep the bot icon + title + subtitle.
- Below it, render a NotebookLM-style grid of **starter cards**:
  - 2 **fixed** i18n chips with full underlying prompts:
    - 📋 Summary → "Tóm tắt chi tiết tài liệu này"
    - 🔍 Key points → "Liệt kê các điểm chính trong tài liệu"
  - Up to 3 **dynamic** cards from `dynamicStarters` (the starter string *is* the prompt).
  - While `startersLoading`, show 2–3 skeleton placeholders in the dynamic slots.
- Each card `onClick` → `onSendMessage(prompt, modelOverride, domainFilter?, verbosity)`
  (same call shape used today by the existing chips).

**Remove** the old small chip row at L403–428 (the bottom-of-input chips) to avoid duplication.

**Wiring:**
- `ChatColumn.tsx` (notebook): call `useChatStarters({ kind: 'notebook', id: notebookId, enabled: chat.messages.length === 0 })`, pass `dynamicStarters` + `startersLoading` to ChatPanel.
- `sources/[id]/page.tsx`: call `useChatStarters({ kind: 'source', id: sourceId, enabled: chat.messages.length === 0 })`, pass through.

---

## Part B — Consistent answer formatting

Strengthen the existing shared rules and detailed-mode block — **prompt only, no FE render change.**

- `src/agentrag/agent/service.py:86` `MARKDOWN_FORMAT_RULES` — tighten so the model MUST, for any
  multi-part / long answer:
  - put each section under a `##` (or `###`) heading on its **own line**;
  - use **numbered lists** for sequences, steps, ranked items, and enumerations;
  - use `- ` bullets for parallel/unordered facts;
  - **mix prose passages with bullets** — never answer a long question as pure bullets, and
    never as an unbroken wall of text.
- `service.py:633-665` (detailed verbosity block) — reinforce the same, phrased for the
  detailed/"tóm tắt"/"explain" path that already triggers structured multi-paragraph output.

Keep the rules concise and additive; do not contradict the existing instructions.

---

## Testing

- **`starters.py`:** unit test the cache key + JSON parse + empty/failure → `[]` (mock gateway),
  mirroring any existing `followups` test.
- **Endpoints:** request test asserting `{ starters: [...] }` shape and graceful `[]` on
  generator failure.
- **Frontend:** `ChatColumn.test.tsx` already mocks ChatPanel; add a render test that the
  empty-state shows the 2 fixed cards and skeletons while loading, and that clicking a card
  calls `onSendMessage` with the right prompt.

## Risks / Mitigations

- **Latency on first open** — mitigated by TTL cache + fixed chips rendering instantly while
  dynamic fill in; dynamic failure is invisible (falls back to fixed only).
- **Vietnamese-only starters** — consistent with existing `followups.py` behaviour; revisit if
  i18n of generated questions is needed later.
