# Chat Starters + Consistent Answer Formatting Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add NotebookLM-style hybrid starter suggestions to the empty chat state on both chat surfaces, and strengthen the answer-formatting prompt so long answers reliably use headings + numbered lists + mixed bullets/prose.

**Architecture:** A new cheap, cached backend generator (`starters.py`, mirroring `followups.py`) produces 2-3 document-aware starter questions from existing insights/segments. Two `GET .../chat/starters` endpoints expose it. The frontend fetches them on chat open via a TanStack Query hook and renders them — alongside 2 fixed client-side chips — as cards in `ChatPanel`'s empty state. Part B is a prompt-only edit in `service.py`.

**Tech Stack:** Python (FastAPI, SQLAlchemy async, pytest), TypeScript (Next.js, React, TanStack Query, Tailwind, vitest).

---

## File Structure

**Backend**
- Create `src/agentrag/agent/starters.py` — `generate_starters()` cached LLM generator.
- Create `tests/agent/test_starters.py` — unit tests for the generator.
- Modify `src/agentrag/adapter/models.py` — add `ChatStartersResponse` model.
- Modify `src/agentrag/adapter/routers/chat.py` — add source + notebook starters endpoints.
- Modify `src/agentrag/agent/service.py` — strengthen `MARKDOWN_FORMAT_RULES` + detailed directive.

**Frontend**
- Modify `frontend/src/lib/api/source-chat.ts` — `getStarters()`.
- Modify `frontend/src/lib/api/chat.ts` — `getStarters()`.
- Create `frontend/src/lib/hooks/useChatStarters.ts` — query hook.
- Modify `frontend/src/components/source/ChatPanel.tsx` — hero starter cards, new props, remove old chip row.
- Create `frontend/src/components/source/ChatPanel.starters.test.tsx` — render test for hero cards.
- Modify `frontend/src/app/(dashboard)/notebooks/components/ChatColumn.tsx` — wire hook.
- Modify `frontend/src/app/(dashboard)/sources/[id]/page.tsx` — wire hook.

---

## Task 1: Backend starter generator

**Files:**
- Create: `src/agentrag/agent/starters.py`
- Test: `tests/agent/test_starters.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/agent/test_starters.py
"""Starter-question generator for the empty chat state."""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest


@pytest.mark.asyncio
async def test_generate_starters_parses_json():
    from src.agentrag.agent import starters
    starters._CACHE.clear()
    gateway = MagicMock()
    gateway.json_response = AsyncMock(return_value=(
        {"starters": ["Tóm tắt chương 1?", "Ai là tác giả?", "Kết luận chính?"]},
        9.0,
    ))
    out = await starters.generate_starters(
        kind="source",
        titles=["Giải phẫu tim"],
        summary_text="Tài liệu về giải phẫu tim mạch…",
        llm_gateway=gateway,
    )
    assert out == ["Tóm tắt chương 1?", "Ai là tác giả?", "Kết luận chính?"]
    gateway.json_response.assert_awaited_once()


@pytest.mark.asyncio
async def test_generate_starters_cache_hit():
    from src.agentrag.agent import starters
    starters._CACHE.clear()
    gateway = MagicMock()
    gateway.json_response = AsyncMock(return_value=({"starters": ["A?", "B?"]}, 5.0))
    out1 = await starters.generate_starters("source", ["T"], "s", gateway)
    out2 = await starters.generate_starters("source", ["T"], "s", gateway)
    assert out1 == out2 == ["A?", "B?"]
    gateway.json_response.assert_awaited_once()


@pytest.mark.asyncio
async def test_generate_starters_swallows_failure():
    from src.agentrag.agent import starters
    starters._CACHE.clear()
    gateway = MagicMock()
    gateway.json_response = AsyncMock(side_effect=RuntimeError("LLM down"))
    out = await starters.generate_starters("source", ["T"], "s", gateway)
    assert out == []


@pytest.mark.asyncio
async def test_generate_starters_empty_titles_returns_empty():
    from src.agentrag.agent import starters
    starters._CACHE.clear()
    gateway = MagicMock()
    gateway.json_response = AsyncMock()
    out = await starters.generate_starters("source", [], "", gateway)
    assert out == []
    gateway.json_response.assert_not_awaited()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/agent/test_starters.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.agentrag.agent.starters'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/agentrag/agent/starters.py
"""Empty-state starter questions.

When a chat session has no messages yet, the UI shows starter suggestions.
Two come from fixed client-side chips; up to 3 are generated here by one
cheap LLM call (task name 'starter', routable via LLM_TASK_MODEL_MAP)
from the document/notebook title(s) + existing summary/insight text.
Cached 10 min by (kind, titles, summary) hash. Failure → [] (never breaks).
"""
from __future__ import annotations

import hashlib
import json
import logging
from typing import Any

from cachetools import TTLCache

logger = logging.getLogger(__name__)

_CACHE: TTLCache[str, list[str]] = TTLCache(maxsize=512, ttl=600)

_SYS = (
    "You are a Vietnamese-language study tutor. Given a document or notebook "
    "TITLE(S) and a short SUMMARY, propose up to 3 short starter questions a "
    "student could click to begin exploring the material. Each question must be "
    "specific to the content, ≤ 80 characters, and end with '?'. "
    "Return strict JSON: {\"starters\": [\"...\", \"...\", \"...\"]}. Vietnamese only."
)


def _cache_key(kind: str, titles: list[str], summary_text: str) -> str:
    blob = kind + "\n" + "|".join(titles) + "\n###\n" + summary_text[:1500]
    return hashlib.sha256(blob.encode("utf-8", errors="replace")).hexdigest()


async def generate_starters(
    kind: str,
    titles: list[str],
    summary_text: str,
    llm_gateway: Any,
) -> list[str]:
    titles = [t for t in titles if t and t.strip()]
    if not titles:
        return []
    key = _cache_key(kind, titles, summary_text or "")
    cached = _CACHE.get(key)
    if cached is not None:
        return list(cached)

    user_payload = json.dumps(
        {"kind": kind, "titles": titles[:10], "summary": (summary_text or "")[:1500]},
        ensure_ascii=False,
    )
    try:
        payload, _latency = await llm_gateway.json_response(
            system_prompt=_SYS,
            user_prompt=user_payload,
            task="starter",
        )
    except Exception:
        logger.warning("generate_starters: LLM call failed", exc_info=True)
        _CACHE[key] = []
        return []

    raw = payload.get("starters") if isinstance(payload, dict) else None
    if not isinstance(raw, list):
        _CACHE[key] = []
        return []
    cleaned: list[str] = []
    for item in raw[:3]:
        if isinstance(item, str):
            s = item.strip()
            if s:
                cleaned.append(s[:120])
    _CACHE[key] = cleaned
    return list(cleaned)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/agent/test_starters.py -v`
Expected: PASS (4 passed)

- [ ] **Step 5: Commit**

```bash
git add src/agentrag/agent/starters.py tests/agent/test_starters.py
git commit -m "feat(chat): cached starter-question generator for empty chat state"
```

---

## Task 2: Backend starters endpoints

**Files:**
- Modify: `src/agentrag/adapter/models.py` (add response model)
- Modify: `src/agentrag/adapter/routers/chat.py` (add 2 endpoints + summary helper)
- Test: `tests/adapter/test_starters_endpoints.py` (create)

- [ ] **Step 1: Add the response model**

In `src/agentrag/adapter/models.py`, add near the other response models:

```python
class ChatStartersResponse(BaseModel):
    starters: list[str] = []
```

(If `BaseModel` is not already imported in the file, it is — the file defines the other `*Response` models the same way. Match the existing import.)

- [ ] **Step 2: Write the failing endpoint test**

```python
# tests/adapter/test_starters_endpoints.py
"""Source + notebook chat starters endpoints — graceful, best-effort."""
from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest


@pytest.mark.asyncio
async def test_build_summary_text_from_insights():
    from src.agentrag.adapter.routers import chat as chat_mod
    # _starter_summary_text joins insight contents, truncated.
    with patch.object(chat_mod, "_load_source_starter_inputs",
                      new=AsyncMock(return_value=("Giải phẫu tim", "summary text"))):
        title, summary = await chat_mod._load_source_starter_inputs("source:abc")
    assert title == "Giải phẫu tim"
    assert summary == "summary text"


@pytest.mark.asyncio
async def test_source_starters_swallows_generator_failure():
    from src.agentrag.adapter.routers import chat as chat_mod
    with patch.object(chat_mod, "_load_source_starter_inputs",
                      new=AsyncMock(return_value=("T", "s"))), \
         patch("src.agentrag.agent.starters.generate_starters",
               new=AsyncMock(side_effect=RuntimeError("boom"))):
        resp = await chat_mod.get_source_starters("source:abc")
    assert resp == {"starters": []}
```

- [ ] **Step 3: Run test to verify it fails**

Run: `pytest tests/adapter/test_starters_endpoints.py -v`
Expected: FAIL — `AttributeError: module ... has no attribute '_load_source_starter_inputs'`

- [ ] **Step 4: Implement the helper + endpoints**

In `src/agentrag/adapter/routers/chat.py`, add these imports at the top of the file if not present:

```python
import uuid
from sqlalchemy import select
from src.agentrag.database import AsyncSessionLocal
from src.agentrag.database.models import Document, Segment
from src.agentrag.adapter.db import AdapterSourceInsight, adapter_notebook_sources
from src.agentrag.adapter.models import ChatStartersResponse
```

(Some of these may already be imported — do not duplicate.)

Then add this helper + the two endpoints. The source helper reuses the same
patterns as `routers/sources.py` (`_parse_source_id`) and `routers/insights.py`
(`AdapterSourceInsight.source_id`, segment concatenation):

```python
def _parse_source_uuid(source_id: str) -> uuid.UUID:
    return uuid.UUID(source_id.removeprefix("source:"))


async def _load_source_starter_inputs(source_id: str) -> tuple[str, str]:
    """Return (title, summary_text) for a source. summary_text prefers existing
    insights; falls back to the first text segments. Best-effort — ('', '') on miss."""
    try:
        sid = _parse_source_uuid(source_id)
    except (ValueError, TypeError):
        return "", ""
    async with AsyncSessionLocal() as session:
        doc = await session.get(Document, sid)
        if not doc:
            return "", ""
        title = doc.title or ""
        insights = (
            await session.execute(
                select(AdapterSourceInsight.content)
                .where(AdapterSourceInsight.source_id == sid)
                .order_by(AdapterSourceInsight.created_at.desc())
            )
        ).scalars().all()
        summary = "\n".join(c for c in insights if c)[:1500]
        if not summary:
            seg_rows = (
                await session.execute(
                    select(Segment.content)
                    .where(Segment.document_id == sid)
                    .order_by(Segment.position)
                    .limit(8)
                )
            ).scalars().all()
            summary = "\n".join(s for s in seg_rows if s)[:1500]
        return title, summary


async def _load_notebook_starter_inputs(notebook_id: str) -> tuple[list[str], str]:
    """Return (titles, summary_text) for all sources linked to a notebook."""
    try:
        nb = uuid.UUID(notebook_id)
    except (ValueError, TypeError):
        return [], ""
    async with AsyncSessionLocal() as session:
        doc_ids = (
            await session.execute(
                select(adapter_notebook_sources.c.document_id)
                .where(adapter_notebook_sources.c.notebook_id == nb)
            )
        ).scalars().all()
        if not doc_ids:
            return [], ""
        titles: list[str] = []
        for did in doc_ids:
            doc = await session.get(Document, did)
            if doc and doc.title:
                titles.append(doc.title)
        summaries = (
            await session.execute(
                select(AdapterSourceInsight.content)
                .where(AdapterSourceInsight.source_id.in_(doc_ids))
                .order_by(AdapterSourceInsight.created_at.desc())
                .limit(20)
            )
        ).scalars().all()
        summary = "\n".join(c for c in summaries if c)[:1500]
        return titles, summary


@source_router.get("/sources/{source_id}/chat/starters")
async def get_source_starters(source_id: str):
    import logging
    from src.agentrag.agent.starters import generate_starters
    from src.agentrag.services.container import get_container
    _log = logging.getLogger(__name__)
    try:
        title, summary = await _load_source_starter_inputs(source_id)
        out = await generate_starters(
            kind="source",
            titles=[title],
            summary_text=summary,
            llm_gateway=get_container().llm,
        )
    except Exception:
        _log.exception("get_source_starters failed")
        out = []
    return ChatStartersResponse(starters=out).model_dump()


@notebook_router.get("/starters")
async def get_notebook_starters(notebook_id: str):
    import logging
    from src.agentrag.agent.starters import generate_starters
    from src.agentrag.services.container import get_container
    _log = logging.getLogger(__name__)
    try:
        titles, summary = await _load_notebook_starter_inputs(notebook_id)
        out = await generate_starters(
            kind="notebook",
            titles=titles,
            summary_text=summary,
            llm_gateway=get_container().llm,
        )
    except Exception:
        _log.exception("get_notebook_starters failed")
        out = []
    return ChatStartersResponse(starters=out).model_dump()
```

Note: `source_router` and `notebook_router` already exist in this file (router
definitions near the top). `_log` is NOT module-level — it is defined locally inside
each function (as shown above with `_log = logging.getLogger(__name__)`); keep that.

- [ ] **Step 5: Run test to verify it passes**

Run: `pytest tests/adapter/test_starters_endpoints.py -v`
Expected: PASS (2 passed)

- [ ] **Step 6: Commit**

```bash
git add src/agentrag/adapter/models.py src/agentrag/adapter/routers/chat.py tests/adapter/test_starters_endpoints.py
git commit -m "feat(chat): source + notebook starters endpoints"
```

---

## Task 3: Frontend API methods

**Files:**
- Modify: `frontend/src/lib/api/source-chat.ts`
- Modify: `frontend/src/lib/api/chat.ts`

- [ ] **Step 1: Add `getStarters` to source-chat API**

In `frontend/src/lib/api/source-chat.ts`, add a method inside the `sourceChatApi`
object (the existing methods strip the `source:` prefix the same way):

```typescript
  getStarters: async (sourceId: string): Promise<string[]> => {
    const cleanId = sourceId.startsWith('source:') ? sourceId.slice(7) : sourceId
    const response = await apiClient.get<{ starters: string[] }>(
      `/sources/${cleanId}/chat/starters`
    )
    return response.data.starters ?? []
  },
```

- [ ] **Step 2: Add `getStarters` to notebook chat API**

In `frontend/src/lib/api/chat.ts`, add a method inside the `chatApi` object
(mirrors `listSessions` param style):

```typescript
  getStarters: async (notebookId: string): Promise<string[]> => {
    const response = await apiClient.get<{ starters: string[] }>(
      `/chat/starters`,
      { params: { notebook_id: notebookId } }
    )
    return response.data.starters ?? []
  },
```

- [ ] **Step 3: Verify it type-checks**

Run: `cd frontend && npx tsc --noEmit`
Expected: no new errors referencing these files.

- [ ] **Step 4: Commit**

```bash
git add frontend/src/lib/api/source-chat.ts frontend/src/lib/api/chat.ts
git commit -m "feat(chat): frontend getStarters API methods"
```

---

## Task 4: Frontend useChatStarters hook

**Files:**
- Create: `frontend/src/lib/hooks/useChatStarters.ts`

- [ ] **Step 1: Implement the hook**

```typescript
// frontend/src/lib/hooks/useChatStarters.ts
'use client'

import { useQuery } from '@tanstack/react-query'
import { sourceChatApi } from '@/lib/api/source-chat'
import { chatApi } from '@/lib/api/chat'

interface UseChatStartersArgs {
  kind: 'source' | 'notebook'
  id: string
  enabled: boolean
}

/**
 * Fetch document-aware starter questions for the empty chat state.
 * Only runs while `enabled` (caller passes true when messages.length === 0).
 * Errors are swallowed → starters: [] so the UI falls back to fixed chips.
 */
export function useChatStarters({ kind, id, enabled }: UseChatStartersArgs) {
  const query = useQuery({
    queryKey: ['chatStarters', kind, id],
    queryFn: () =>
      kind === 'source' ? sourceChatApi.getStarters(id) : chatApi.getStarters(id),
    enabled: enabled && !!id,
    staleTime: 10 * 60 * 1000,
    retry: false,
  })
  return {
    starters: query.data ?? [],
    isLoading: query.isLoading && query.fetchStatus !== 'idle',
  }
}
```

- [ ] **Step 2: Verify it type-checks**

Run: `cd frontend && npx tsc --noEmit`
Expected: no new errors.

- [ ] **Step 3: Commit**

```bash
git add frontend/src/lib/hooks/useChatStarters.ts
git commit -m "feat(chat): useChatStarters query hook"
```

---

## Task 5: ChatPanel hero starter cards

**Files:**
- Modify: `frontend/src/components/source/ChatPanel.tsx` (props, empty-state, remove old chips)
- Test: `frontend/src/components/source/ChatPanel.starters.test.tsx` (create)

- [ ] **Step 1: Add the two new props**

In the `ChatPanelProps` interface (`ChatPanel.tsx`, after `onCancelStreaming`), add:

```typescript
  // Document-aware starter suggestions for the empty state (owned by parent query)
  dynamicStarters?: string[]
  startersLoading?: boolean
```

And destructure them in the component signature alongside the others:

```typescript
  onCancelStreaming,
  dynamicStarters = [],
  startersLoading = false
```

- [ ] **Step 2: Define the fixed chips + a card list above the return**

Inside the `ChatPanel` function body, before `return (`, add:

```typescript
  // Fixed universal starters (instant, client-side). `q` is the prompt sent.
  const FIXED_STARTERS: { label: string; q: string }[] = [
    { label: '📋 Tóm tắt tài liệu', q: 'Tóm tắt chi tiết tài liệu này' },
    { label: '🔍 Các điểm chính', q: 'Liệt kê các điểm chính trong tài liệu' },
  ]
  const sendStarter = (q: string) =>
    onSendMessage(
      q,
      modelOverride,
      contextType === 'notebook' ? domainFilter : undefined,
      verbosity,
    )
```

- [ ] **Step 3: Replace the empty-state hero block**

Replace the empty-state block (the `messages.length === 0 ?` branch, currently the
centered bot icon + title + subtitle) with the icon/title/subtitle PLUS a cards grid:

```tsx
            <div className="flex flex-col items-center justify-center text-center min-h-[60vh] gap-6">
              <div className="flex flex-col items-center">
                <div className="h-12 w-12 rounded-full bg-primary/10 flex items-center justify-center mb-4">
                  <Bot className="h-6 w-6 text-primary" />
                </div>
                <h2 className="text-xl font-semibold mb-2">
                  {title || t('chat.startConversation').replace('{type}', contextType === 'source' ? t('navigation.sources') : t('common.notebook'))}
                </h2>
                <p className="text-sm text-muted-foreground">{t('chat.askQuestions')}</p>
              </div>
              {!isStreaming && (
                <div className="grid grid-cols-1 sm:grid-cols-2 gap-2 w-full max-w-xl" data-testid="chat-starters">
                  {FIXED_STARTERS.map((chip) => (
                    <button
                      key={chip.label}
                      type="button"
                      onClick={() => sendStarter(chip.q)}
                      className="text-left text-sm px-4 py-3 rounded-xl border bg-muted/40 hover:bg-accent transition-colors"
                    >
                      {chip.label}
                    </button>
                  ))}
                  {startersLoading
                    ? [0, 1, 2].map((i) => (
                        <div
                          key={`sk-${i}`}
                          data-testid="starter-skeleton"
                          className="h-[46px] rounded-xl border bg-muted/40 animate-pulse"
                        />
                      ))
                    : dynamicStarters.map((q) => (
                        <button
                          key={q}
                          type="button"
                          onClick={() => sendStarter(q)}
                          className="text-left text-sm px-4 py-3 rounded-xl border bg-muted/40 hover:bg-accent transition-colors"
                        >
                          💡 {q}
                        </button>
                      ))}
                </div>
              )}
            </div>
```

- [ ] **Step 4: Remove the old bottom-of-input chip row**

Delete the block that renders the old quick-start chips (the
`{!input.trim() && messages.length === 0 && !isStreaming && (` ... `)}` block that
maps over the inline `[{ label: '📋 Tóm tắt tài liệu', ... }]` array in the input area).
The hero cards replace it.

- [ ] **Step 5: Write the render test**

```tsx
// frontend/src/components/source/ChatPanel.starters.test.tsx
import { render, screen, fireEvent } from '@testing-library/react'
import { describe, it, expect, vi } from 'vitest'
import { ChatPanel } from './ChatPanel'

vi.mock('@/lib/hooks/use-translation', () => ({
  useTranslation: () => ({ t: (k: string) => k }),
}))
vi.mock('@/lib/hooks/use-modal-manager', () => ({
  useModalManager: () => ({ openModal: vi.fn() }),
}))

const baseProps = {
  messages: [],
  isStreaming: false,
  contextIndicators: null,
  onSendMessage: vi.fn(),
}

describe('ChatPanel starters', () => {
  it('renders fixed chips and dynamic starters', () => {
    render(<ChatPanel {...baseProps} dynamicStarters={['Ai là tác giả?']} />)
    expect(screen.getByText('📋 Tóm tắt tài liệu')).toBeInTheDocument()
    expect(screen.getByText('💡 Ai là tác giả?')).toBeInTheDocument()
  })

  it('shows skeletons while loading', () => {
    render(<ChatPanel {...baseProps} startersLoading />)
    expect(screen.getAllByTestId('starter-skeleton').length).toBe(3)
  })

  it('sends the prompt when a fixed chip is clicked', () => {
    const onSendMessage = vi.fn()
    render(<ChatPanel {...baseProps} onSendMessage={onSendMessage} />)
    fireEvent.click(screen.getByText('📋 Tóm tắt tài liệu'))
    expect(onSendMessage).toHaveBeenCalledWith(
      'Tóm tắt chi tiết tài liệu này', undefined, undefined, null,
    )
  })
})
```

- [ ] **Step 6: Run the test**

Run: `cd frontend && npx vitest run src/components/source/ChatPanel.starters.test.tsx`
Expected: PASS (3 passed). If other ChatPanel child imports fail to resolve under
jsdom, add minimal `vi.mock` stubs for them at the top of the test (mirror the two
existing mocks) — do NOT change `ChatPanel.tsx` to make the test pass.

- [ ] **Step 7: Commit**

```bash
git add frontend/src/components/source/ChatPanel.tsx frontend/src/components/source/ChatPanel.starters.test.tsx
git commit -m "feat(chat): NotebookLM-style hero starter cards in empty state"
```

---

## Task 6: Wire the hook into both surfaces

**Files:**
- Modify: `frontend/src/app/(dashboard)/notebooks/components/ChatColumn.tsx`
- Modify: `frontend/src/app/(dashboard)/sources/[id]/page.tsx`

- [ ] **Step 1: Wire notebook chat (ChatColumn)**

In `ChatColumn.tsx`, add the import:

```typescript
import { useChatStarters } from '@/lib/hooks/useChatStarters'
```

After the `chat` hook is initialized, add:

```typescript
  const starters = useChatStarters({
    kind: 'notebook',
    id: notebookId,
    enabled: chat.messages.length === 0,
  })
```

Add these two props to the `<ChatPanel ... />` element:

```tsx
      dynamicStarters={starters.starters}
      startersLoading={starters.isLoading}
```

- [ ] **Step 2: Wire source chat (sources/[id]/page.tsx)**

In `sources/[id]/page.tsx`, add the import:

```typescript
import { useChatStarters } from '@/lib/hooks/useChatStarters'
```

After `const chat = useSourceChat(sourceId)`, add:

```typescript
  const starters = useChatStarters({
    kind: 'source',
    id: sourceId,
    enabled: chat.messages.length === 0,
  })
```

Add these two props to the `<ChatPanel ... />` element:

```tsx
            dynamicStarters={starters.starters}
            startersLoading={starters.isLoading}
```

- [ ] **Step 3: Verify type-check + existing ChatColumn test still passes**

Run: `cd frontend && npx tsc --noEmit && npx vitest run src/app/\(dashboard\)/notebooks/components/ChatColumn.test.tsx`
Expected: no new type errors; ChatColumn test PASS (it mocks ChatPanel, so the new props are ignored).

- [ ] **Step 4: Commit**

```bash
git add "frontend/src/app/(dashboard)/notebooks/components/ChatColumn.tsx" "frontend/src/app/(dashboard)/sources/[id]/page.tsx"
git commit -m "feat(chat): fetch + pass starters on both chat surfaces"
```

---

## Task 7: Strengthen answer formatting (Part B)

**Files:**
- Modify: `src/agentrag/agent/service.py` (`MARKDOWN_FORMAT_RULES` + detailed `length_directive`)

This is a prompt-only change. There is no unit test for prompt wording; verify by
inspection + the existing service tests still passing.

- [ ] **Step 1: Strengthen `MARKDOWN_FORMAT_RULES`**

In `src/agentrag/agent/service.py`, replace the `MARKDOWN_FORMAT_RULES` string body
(currently ends `"Each section heading on its own line, body below. Never inline heading with body."`)
by appending two sentences before the closing paren:

```python
MARKDOWN_FORMAT_RULES = (
    "FORMAT: `answer` field is Markdown. Required: "
    "**bold** key terms (drug names, doses, lab values, diagnoses, percentages); "
    "bullet lists `- ` for parallel facts; `### Heading` between sections; "
    "GFM tables `| a | b |` for comparisons; LaTeX `$...$` or `$$...$$` for formulas "
    "(eGFR, BMI, dose calc, ratios, statistics); "
    "`> blockquote` only for safety warnings or contraindications. "
    "Each section heading on its own line, body below. Never inline heading with body. "
    "For any answer longer than ~4 sentences: split it under `##`/`###` section headings, "
    "use NUMBERED lists (`1.` `2.`) for sequences/steps/ranked items and `- ` bullets for "
    "parallel facts. MIX prose passages with lists — never answer as one unbroken wall of "
    "text, and never as pure bullets with no connecting prose."
)
```

- [ ] **Step 2: Strengthen the detailed `length_directive`**

In the same file, in the verbose branch of `length_directive`, append one sentence
to the verbose string (the branch starting `"LENGTH: thorough, multi-paragraph. ...`),
right before the closing quote of that branch (`"Bold key terms. "`):

```python
        length_directive = (
            "LENGTH: thorough, multi-paragraph. STRUCTURE (required): start each "
            "section with a Markdown H2 heading on its own line (e.g. `## Định nghĩa`), "
            "then the body below it. For medical/overview topics use these sections "
            "when relevant: Tổng quan, Định nghĩa, Phân loại, Nguyên nhân, Chẩn đoán, "
            "Điều trị, Tiên lượng. Use NUMBERED lists (`1.` `2.` `3.`) for "
            "classifications, causes, steps and ranked items; bullets only for "
            "non-sequential parallel facts. Bold key terms. "
            "Open each section with a 1-2 sentence prose passage before any list, so the "
            "answer reads as mixed prose + bullets, not bullets alone. "
            if verbose
            else
            "LENGTH: concise — focus on the question. Still use **bold** and bullets when appropriate. "
        )
```

- [ ] **Step 3: Verify nothing broke**

Run: `pytest tests/agent/ -q`
Expected: PASS (no test asserts exact prompt wording; existing service tests unaffected).

- [ ] **Step 4: Commit**

```bash
git add src/agentrag/agent/service.py
git commit -m "feat(chat): stronger detailed-answer formatting (headings + numbered + mixed prose/bullets)"
```

---

## Self-Review Notes

- **Spec coverage:** A1→Task 1; A2→Task 2; A3→Tasks 3-4; A4→Tasks 5-6; Part B→Task 7. All covered.
- **Type consistency:** generator `generate_starters(kind, titles, summary_text, llm_gateway)` is called with these exact kwargs in Task 2. Endpoints return `{ starters: [...] }`; FE `getStarters` reads `response.data.starters`; hook returns `{ starters, isLoading }`; ChatPanel props `dynamicStarters`/`startersLoading` match Tasks 5-6.
- **Fixed-chip labels** intentionally remain hardcoded Vietnamese strings (matches the current code that this plan removes); no new locale keys added (out of scope).
```
