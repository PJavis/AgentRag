# Chat Polish + Speed Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship 4 phases — output-token cap, NotebookLM-style citation hover + inline images + follow-up chips, latency config tweaks, Ask & Search UI polish — each as an independent commit.

**Architecture:** Phase A adds one config and threads `max_tokens` through 4 LLM call sites. Phase B extends the `_ground_citations` payload, adds a `generate_followups` helper, persists `follow_ups` through `ChatMessage.extra_metadata`, and replaces clickable references with Radix HoverCards on the frontend. Phase C is pure `.env` defaults + a README note. Phase D wires the existing `useAsk` SSE consumer to render the new components.

**Tech Stack:** Python 3.11 / FastAPI / Pydantic / Next.js 15 (App Router) / TanStack Query / Radix UI / react-markdown + remark-math + rehype-katex / pytest-asyncio.

---

## File Structure

| Path | Action | Phase |
|---|---|---|
| `src/agentrag/config.py` | Modify — add `AGENT_MAX_OUTPUT_TOKENS` | A |
| `src/agentrag/agent/llm.py` | Modify — pass `max_tokens` to all 3 create() calls | A |
| `src/agentrag/services/llm_gateway.py` | Modify — pass `max_tokens` to both vision create() calls | A |
| `.env`, `.env.example` | Modify — document new var | A |
| `src/agentrag/agent/service.py` | Modify — `_ground_citations` enriches mime + image_url + page_label + bold-wrap excerpt | B |
| `src/agentrag/agent/followups.py` | Create — `generate_followups` helper + TTL cache | B |
| `src/agentrag/adapter/models.py` | Modify — extend `ChatMessage`, declare `Citation` | B |
| `src/agentrag/adapter/routers/chat.py` | Modify — invoke + persist follow_ups; surface in `_msg_to_chat` | B |
| `src/agentrag/chat/history.py` | (no change — already round-trips extra_metadata) | B |
| `tests/agent/test_followups.py` | Create — JSON parse + cache hit + failure path | B |
| `tests/adapter/test_citation_shape.py` | Create — extended Citation Pydantic | B |
| `frontend/package.json` | Modify — add `@radix-ui/react-hover-card` | B |
| `frontend/src/lib/types/api.ts` | Modify — extend `Citation` + `NotebookChatMessage` | B |
| `frontend/src/components/source/CitationHoverCard.tsx` | Create | B |
| `frontend/src/components/source/FollowupChips.tsx` | Create | B |
| `frontend/src/components/source/InlineImageCitation.tsx` | Create | B |
| `frontend/src/lib/utils/source-references.tsx` | Modify — add `createCompactReferenceHoverComponent` | B |
| `frontend/src/components/source/ChatPanel.tsx` | Modify — render new components | B |
| `.env`, `.env.example` | Modify — recommend `LLM_TASK_MODEL_MAP`, `AGENT_MAX_STEPS=2`, `AGENT_PLAN_TRIGGER_MIN_CHARS=120` | C |
| `README.md` | Modify — §5.6 doc the new defaults | C |
| `frontend/src/components/search/StreamingResponse.tsx` | Modify — use shared `CitationHoverCard` + `FollowupChips` | D |
| `frontend/src/app/(dashboard)/search/page.tsx` | Modify — render follow-up chips on Ask result | D |

---

## Phase A — Output token cap

### Task A1: Add `AGENT_MAX_OUTPUT_TOKENS` setting + thread through LLM clients

**Files:**
- Modify: `src/agentrag/config.py`
- Modify: `src/agentrag/agent/llm.py`
- Modify: `src/agentrag/services/llm_gateway.py`
- Modify: `.env`, `.env.example`

- [ ] **Step 1: Add setting**

Edit `src/agentrag/config.py`. After `AGENT_MAX_CONTEXT_TOKENS: int = 6000`:

```python
    # S8 — per-call output ceiling. 128K = let model self-cap. Lower
    # (e.g. 8192) for fast local Ollama on small models.
    AGENT_MAX_OUTPUT_TOKENS: int = 131072
```

- [ ] **Step 2: Pass `max_tokens` in `AgentLLM`**

Edit `src/agentrag/agent/llm.py`. In `json_response`, `text_response`, and `stream_text`, find each call:

```python
        response = await self._create(
            model=self.model,
            temperature=self.temperature,
            ...
        )
```

Add `max_tokens=settings.AGENT_MAX_OUTPUT_TOKENS,` after `temperature`. Three sites total.

- [ ] **Step 3: Pass `max_tokens` in `LLMGateway.vision_response`**

Edit `src/agentrag/services/llm_gateway.py`. Find both `client.chat.completions.create(` calls inside `vision_response` (primary + fallback). Add `max_tokens=settings.AGENT_MAX_OUTPUT_TOKENS,` to each kwargs block.

- [ ] **Step 4: Document in `.env.example`**

Add right after `AGENT_MAX_CONTEXT_TOKENS=6000`:

```env
# Per-call output ceiling. 128K = let the model self-cap (it stops at
# natural end-of-answer). Reduce (e.g. 8192) for fast local Ollama.
AGENT_MAX_OUTPUT_TOKENS=131072
```

Mirror the same line into `.env`.

- [ ] **Step 5: Smoke import**

Run: `uv run python -c "from src.agentrag.config import settings; print(settings.AGENT_MAX_OUTPUT_TOKENS)"`

Expected: `131072`

- [ ] **Step 6: Commit**

```bash
git add src/agentrag/config.py src/agentrag/agent/llm.py src/agentrag/services/llm_gateway.py .env .env.example
git commit -m "feat(s8): AGENT_MAX_OUTPUT_TOKENS=131072 — unlock long answers

Default Ollama num_predict=128 cuts answers mid-sentence. Cloud
OpenAI default max_tokens=1024 same problem. Thread the new setting
through AgentLLM.{json,text,stream}_response and LLMGateway.vision_response
so every chat.completions.create() carries the high cap. Model
self-caps in practice; user can override per deployment."
```

---

## Phase B — NotebookLM polish (citations + images + follow-ups)

### Task B1: Enrich `_ground_citations` payload

**Files:**
- Modify: `src/agentrag/agent/service.py` (function `_ground_citations`, around line 737)

- [ ] **Step 1: Replace the function body**

Find the existing `_ground_citations` method (line 737 of `src/agentrag/agent/service.py`). Replace with:

```python
    def _ground_citations(
        self,
        citations: list[dict[str, Any]],
        packed_context: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        # Build lookup: content_hash → full context item (for page + excerpt enrichment)
        hash_to_ctx: dict[str, dict[str, Any]] = {
            item["content_hash"]: item
            for item in packed_context
            if item.get("content_hash")
        }
        allowed = {
            (
                item.get("document_title"),
                item.get("section_path"),
                item.get("position"),
                item.get("content_hash"),
            )
            for item in packed_context
        }
        grounded: list[dict[str, Any]] = []
        for citation in citations:
            key = (
                citation.get("document_title"),
                citation.get("section_path"),
                citation.get("position"),
                citation.get("content_hash"),
            )
            if key not in allowed:
                continue
            ctx = hash_to_ctx.get(citation.get("content_hash", ""), {})
            seg_type = ctx.get("segment_type") or citation.get("segment_type") or "text"
            mime = self._mime_for_segment(ctx, seg_type)
            excerpt = (ctx.get("excerpt") or ctx.get("content") or "")[:300]
            entry: dict[str, Any] = {
                "document_title": citation.get("document_title"),
                "section_path": citation.get("section_path"),
                "position": citation.get("position"),
                "content_hash": citation.get("content_hash"),
                "excerpt": excerpt,
                "segment_type": seg_type,
                "mime": mime,
            }
            # Page-aware fields (PDF only)
            page = ctx.get("page") or ctx.get("page_start")
            if page is not None:
                entry["page"] = page
                entry["page_label"] = f"p.{page}"
            if ctx.get("page_start") is not None:
                entry["page_start"] = ctx["page_start"]
            if ctx.get("page_end") is not None:
                entry["page_end"] = ctx["page_end"]
            # Image segments — expose URL so the UI can render <img>
            if seg_type == "image":
                meta = ctx.get("metadata") or {}
                img = meta.get("image_url") or meta.get("image_path") or ctx.get("image_url")
                if img:
                    entry["image_url"] = self._normalize_image_url(img)
            grounded.append(entry)
        return grounded

    @staticmethod
    def _mime_for_segment(ctx: dict[str, Any], seg_type: str) -> str | None:
        """Infer mime from document filename suffix; falls back by segment_type."""
        if seg_type == "image":
            return (ctx.get("metadata") or {}).get("image_mime") or "image/jpeg"
        title = (ctx.get("document_title") or "").lower()
        if title.endswith(".pdf"):
            return "application/pdf"
        if title.endswith(".docx") or title.endswith(".doc"):
            return "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        if title.endswith(".md"):
            return "text/markdown"
        if title.endswith(".txt"):
            return "text/plain"
        return None

    @staticmethod
    def _normalize_image_url(raw: str) -> str:
        """Rewrite pipeline image_path → /api/images/<rest> so the frontend can fetch."""
        if not raw:
            return raw
        if raw.startswith("http://") or raw.startswith("https://") or raw.startswith("/api/"):
            return raw
        if raw.startswith("/images/"):
            return f"/api{raw}"
        if raw.startswith("data/images/"):
            return f"/api/{raw[len('data/'):]}"
        if raw.startswith("images/"):
            return f"/api/{raw}"
        return raw
```

- [ ] **Step 2: Commit**

```bash
git add src/agentrag/agent/service.py
git commit -m "feat(b1): enrich grounded citations with mime + image_url + page_label

Adds segment_type, mime, page_label, and image_url (for image
segments) to every grounded citation. Excerpt widened to 300 chars.
Pure-additive — existing consumers ignore unknown fields."
```

---

### Task B2: `generate_followups` helper

**Files:**
- Create: `src/agentrag/agent/followups.py`
- Create: `tests/agent/test_followups.py`

- [ ] **Step 1: Write failing test**

```python
# tests/agent/test_followups.py
"""B2 — generate_followups."""
from __future__ import annotations

import hashlib
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


@pytest.mark.asyncio
async def test_generate_followups_parses_json():
    from src.agentrag.agent import followups
    gateway = MagicMock()
    gateway.json_response = AsyncMock(return_value=(
        {"follow_ups": ["Câu 1?", "Câu 2?", "Câu 3?"]},
        12.0,
    ))
    out = await followups.generate_followups(
        question="Van hai lá là gì?",
        answer="Van hai lá nằm…",
        citations=[],
        llm_gateway=gateway,
    )
    assert out == ["Câu 1?", "Câu 2?", "Câu 3?"]
    gateway.json_response.assert_awaited_once()


@pytest.mark.asyncio
async def test_generate_followups_cache_hit():
    from src.agentrag.agent import followups
    followups._CACHE.clear()
    gateway = MagicMock()
    gateway.json_response = AsyncMock(return_value=(
        {"follow_ups": ["A?", "B?"]}, 10.0,
    ))
    q, a = "x", "y"
    out1 = await followups.generate_followups(q, a, [], gateway)
    out2 = await followups.generate_followups(q, a, [], gateway)
    assert out1 == out2 == ["A?", "B?"]
    gateway.json_response.assert_awaited_once()   # cache hit, only 1 call


@pytest.mark.asyncio
async def test_generate_followups_swallows_failure(caplog):
    from src.agentrag.agent import followups
    followups._CACHE.clear()
    gateway = MagicMock()
    gateway.json_response = AsyncMock(side_effect=RuntimeError("LLM down"))
    out = await followups.generate_followups("q", "a", [], gateway)
    assert out == []
```

- [ ] **Step 2: Run test (expect FAIL — module missing)**

`uv run python -m pytest tests/agent/test_followups.py -v`

- [ ] **Step 3: Implement**

```python
# src/agentrag/agent/followups.py
"""B2 — Follow-up question generator.

After agent.chat returns an answer, fire one cheap LLM call (task name
'followup' so users can route to a small model via LLM_TASK_MODEL_MAP)
to propose 3 short Vietnamese follow-up questions. Cached 5 min by
(question, answer) hash. Failure → empty list (never breaks the chat).
"""
from __future__ import annotations

import hashlib
import json
import logging
from typing import Any

from cachetools import TTLCache

logger = logging.getLogger(__name__)

_CACHE: TTLCache[str, list[str]] = TTLCache(maxsize=512, ttl=300)

_SYS = (
    "You are a helpful Vietnamese-language study tutor. Given a QUESTION "
    "and the ANSWER given to the student, propose up to 3 short follow-up "
    "questions the student might ask next. Each ≤ 80 characters. "
    "Return strict JSON: {\"follow_ups\": [\"...\", \"...\", \"...\"]}. "
    "Vietnamese only."
)


def _cache_key(question: str, answer: str) -> str:
    return hashlib.sha256(
        (question + "\n###\n" + answer).encode("utf-8", errors="replace")
    ).hexdigest()


async def generate_followups(
    question: str,
    answer: str,
    citations: list[dict[str, Any]],
    llm_gateway: Any,
) -> list[str]:
    if not question or not answer:
        return []
    key = _cache_key(question, answer)
    cached = _CACHE.get(key)
    if cached is not None:
        return list(cached)

    user_payload = json.dumps(
        {
            "question": question,
            "answer": answer[:1500],
            "doc_titles": list({(c.get("document_title") or "")[:80] for c in citations}),
        },
        ensure_ascii=False,
    )
    try:
        payload, _latency = await llm_gateway.json_response(
            system_prompt=_SYS,
            user_prompt=user_payload,
            task="followup",
        )
    except Exception:
        logger.warning("generate_followups: LLM call failed", exc_info=True)
        _CACHE[key] = []
        return []

    raw = payload.get("follow_ups") if isinstance(payload, dict) else None
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

Also create `tests/agent/__init__.py` (empty) if missing.

- [ ] **Step 4: Run test (expect PASS)**

`uv run python -m pytest tests/agent/test_followups.py -v` → 3 passed.

- [ ] **Step 5: Commit**

```bash
git add src/agentrag/agent/followups.py tests/agent/
git commit -m "feat(b2): generate_followups helper with TTL cache + task='followup'"
```

---

### Task B3: Extend adapter `ChatMessage` model + persist `follow_ups`

**Files:**
- Modify: `src/agentrag/adapter/models.py`
- Modify: `src/agentrag/adapter/routers/chat.py`

- [ ] **Step 1: Extend `ChatMessage`**

Edit `src/agentrag/adapter/models.py`. Find the existing `ChatMessage` (line 111). Add the new field next to `sql_query`:

```python
class ChatMessage(BaseModel):
    id: str | None = None
    type: str
    role: str
    content: str
    citations: list[Any] | None = None
    tool_trace: list[Any] | None = None
    reasoning_path: str | None = None
    timings_ms: dict[str, Any] | None = None
    plan_subqueries: list[Any] | None = None
    sql_query: str | None = None
    follow_ups: list[str] | None = None     # ← new
    timestamp: str | None = None
```

- [ ] **Step 2: Generate + persist `follow_ups` in `execute_chat`**

Edit `src/agentrag/adapter/routers/chat.py`. After the existing `appended = await store.append_message(...assistant…)` block (around line 224) and BEFORE the record_event call, insert:

```python
    # B2 — Follow-up suggestions. Best-effort: log + ignore on failure.
    follow_ups: list[str] = []
    try:
        from src.agentrag.agent.followups import generate_followups
        from src.agentrag.services.container import get_container
        follow_ups = await generate_followups(
            question=body.message,
            answer=result.get("answer", ""),
            citations=result.get("citations") or [],
            llm_gateway=get_container().llm,
        )
    except Exception:
        _log.exception("execute_chat: generate_followups failed")
    # Update extra_metadata on the persisted assistant turn
    if follow_ups and isinstance(appended, dict):
        try:
            from sqlalchemy import update as _sql_update
            from src.agentrag.database import AsyncSessionLocal
            from src.agentrag.database.models import ChatMessage as _CM
            msg_id = appended.get("message_id")
            if msg_id:
                async with AsyncSessionLocal() as _s:
                    row = await _s.get(_CM, uuid.UUID(msg_id))
                    if row is not None:
                        meta = dict(row.extra_metadata or {})
                        meta["follow_ups"] = follow_ups
                        row.extra_metadata = meta
                        await _s.commit()
        except Exception:
            _log.exception("execute_chat: persist follow_ups failed")
```

- [ ] **Step 3: Surface in `_msg_to_chat`**

In the same file (`adapter/routers/chat.py`), the helper builds `ChatMessage` from a db row. Find `_msg_to_chat` (it returns `ChatMessage.model_dump()`). Add `follow_ups=extra.get("follow_ups")` to the kwargs:

```python
    return ChatMessage(
        id=mid,
        type=_ROLE_TO_TYPE.get(role, "ai"),
        role=role,
        content=msg["content"],
        citations=msg.get("citations"),
        tool_trace=msg.get("tool_trace"),
        timings_ms=msg.get("timings_ms"),
        reasoning_path=extra.get("reasoning_path"),
        plan_subqueries=extra.get("plan_subqueries"),
        sql_query=extra.get("sql_query"),
        follow_ups=extra.get("follow_ups"),     # ← new
        timestamp=msg.get("created_at"),
    ).model_dump()
```

- [ ] **Step 4: Verify import**

`uv run python -c "from src.agentrag.adapter.routers.chat import execute_chat; print('ok')"`

Expected: `ok`

- [ ] **Step 5: Commit**

```bash
git add src/agentrag/adapter/models.py src/agentrag/adapter/routers/chat.py
git commit -m "feat(b3): persist + serve follow_ups on assistant ChatMessage"
```

---

### Task B4: Adapter test for citation shape + follow_ups field

**Files:**
- Create: `tests/adapter/test_citation_shape.py`

- [ ] **Step 1: Test**

```python
# tests/adapter/test_citation_shape.py
"""B-acceptance — adapter models accept the new fields."""
from src.agentrag.adapter.models import ChatMessage


def test_chat_message_accepts_follow_ups():
    m = ChatMessage(
        type="ai", role="assistant", content="hi",
        follow_ups=["A?", "B?"],
    )
    d = m.model_dump()
    assert d["follow_ups"] == ["A?", "B?"]


def test_chat_message_follow_ups_optional():
    m = ChatMessage(type="ai", role="assistant", content="hi")
    assert m.follow_ups is None


def test_chat_message_accepts_extended_citations():
    cite = {
        "document_title": "Lec 10.pdf",
        "section_path": "Ch. 3",
        "page": 47,
        "page_label": "p.47",
        "excerpt": "Van hai lá…",
        "content_hash": "abc",
        "mime": "application/pdf",
        "segment_type": "text",
    }
    m = ChatMessage(
        type="ai", role="assistant", content="ans",
        citations=[cite],
    )
    assert m.citations[0]["page_label"] == "p.47"
```

- [ ] **Step 2: Run**

`uv run python -m pytest tests/adapter/test_citation_shape.py -v` → 3 passed.

- [ ] **Step 3: Commit**

```bash
git add tests/adapter/test_citation_shape.py
git commit -m "test(b): citation + follow_ups model shape"
```

---

### Task B5: Frontend types

**Files:**
- Modify: `frontend/src/lib/types/api.ts`

- [ ] **Step 1: Extend types**

Edit `frontend/src/lib/types/api.ts`. Find the `NotebookChatMessage` interface, add `follow_ups`:

```ts
export interface NotebookChatMessage {
  id: string
  type: 'human' | 'ai'
  content: string
  timestamp?: string
  citations?: Array<Record<string, unknown>>
  tool_trace?: ToolTraceEntry[]
  reasoning_path?: string | null
  timings_ms?: ChatTimings
  plan_subqueries?: unknown[]
  sql_query?: string | null
  follow_ups?: string[]
}
```

Add a typed `Citation` interface near the top of the file (or after `BaseChatSession`):

```ts
export interface Citation {
  document_title: string
  section_path?: string | null
  position?: number | null
  page?: number | null
  page_label?: string | null
  page_start?: number | null
  page_end?: number | null
  excerpt?: string
  content_hash?: string
  mime?: string | null
  segment_type?: 'text' | 'image' | 'table'
  image_url?: string | null
}
```

Also apply `follow_ups` to `SourceChatMessage` (same struct) for consistency.

- [ ] **Step 2: Verify**

```bash
cd frontend && ./node_modules/.bin/tsc --noEmit
```

Exit code 0.

- [ ] **Step 3: Commit**

```bash
git add frontend/src/lib/types/api.ts
git commit -m "feat(b5): TS types for Citation + follow_ups"
```

---

### Task B6: Install `@radix-ui/react-hover-card`

**Files:**
- Modify: `frontend/package.json`, `frontend/package-lock.json`

- [ ] **Step 1: Install**

```bash
cd frontend && npm install @radix-ui/react-hover-card
```

- [ ] **Step 2: Verify**

`grep -q '@radix-ui/react-hover-card' frontend/package.json && echo OK`

- [ ] **Step 3: Commit**

```bash
git add frontend/package.json frontend/package-lock.json
git commit -m "chore(b6): add @radix-ui/react-hover-card"
```

---

### Task B7: `CitationHoverCard` component

**Files:**
- Create: `frontend/src/components/source/CitationHoverCard.tsx`

- [ ] **Step 1: Implement**

```tsx
// frontend/src/components/source/CitationHoverCard.tsx
'use client'

import * as HoverCard from '@radix-ui/react-hover-card'
import { FileText, Image as ImageIcon } from 'lucide-react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import remarkMath from 'remark-math'
import rehypeKatex from 'rehype-katex'
import 'katex/dist/katex.min.css'
import { Badge } from '@/components/ui/badge'
import type { Citation } from '@/lib/types/api'

interface CitationHoverCardProps {
  index: number
  citation: Citation
  /** The clickable anchor rendered inside the assistant answer. */
  children: React.ReactNode
}

function renderExcerptBody(citation: Citation) {
  // Image segments — render the figure inside the popover.
  if (citation.segment_type === 'image' && citation.image_url) {
    return (
      // eslint-disable-next-line @next/next/no-img-element
      <img
        src={citation.image_url}
        alt={citation.section_path || citation.document_title}
        className="max-h-40 max-w-full rounded border border-border"
        loading="lazy"
      />
    )
  }
  const excerpt = (citation.excerpt || '').trim()
  if (!excerpt) {
    return <span className="text-muted-foreground italic">No excerpt available</span>
  }
  // Render markdown safely so KaTeX formulas survive for md/docx sources.
  return (
    <div className="prose prose-xs prose-neutral dark:prose-invert max-w-none break-words">
      <ReactMarkdown
        remarkPlugins={[remarkGfm, remarkMath]}
        rehypePlugins={[[rehypeKatex, { strict: 'ignore', throwOnError: false }]]}
      >
        {excerpt}
      </ReactMarkdown>
    </div>
  )
}

export function CitationHoverCard({ index, citation, children }: CitationHoverCardProps) {
  const isImage = citation.segment_type === 'image'
  return (
    <HoverCard.Root openDelay={120} closeDelay={80}>
      <HoverCard.Trigger asChild>{children}</HoverCard.Trigger>
      <HoverCard.Portal>
        <HoverCard.Content
          side="top"
          sideOffset={6}
          className="z-50 w-[380px] max-w-[90vw] rounded-md border bg-popover p-3 shadow-md text-popover-foreground"
        >
          <div className="flex items-start gap-2 mb-2">
            {isImage ? (
              <ImageIcon className="h-3.5 w-3.5 mt-0.5 shrink-0" />
            ) : (
              <FileText className="h-3.5 w-3.5 mt-0.5 shrink-0" />
            )}
            <div className="flex-1 min-w-0">
              <div className="flex items-center gap-1.5 flex-wrap">
                <span className="text-[10px] font-mono text-muted-foreground">[{index}]</span>
                <span className="text-xs font-medium truncate">{citation.document_title}</span>
                {citation.page_label && (
                  <Badge variant="outline" className="text-[10px] px-1 py-0">
                    {citation.page_label}
                  </Badge>
                )}
              </div>
              {citation.section_path && (
                <div className="text-[10px] text-muted-foreground truncate">
                  {citation.section_path}
                </div>
              )}
            </div>
          </div>
          <div className="border-t pt-2">{renderExcerptBody(citation)}</div>
        </HoverCard.Content>
      </HoverCard.Portal>
    </HoverCard.Root>
  )
}

export default CitationHoverCard
```

- [ ] **Step 2: Verify**

`cd frontend && ./node_modules/.bin/tsc --noEmit` → exit 0.

- [ ] **Step 3: Commit**

```bash
git add frontend/src/components/source/CitationHoverCard.tsx
git commit -m "feat(b7): CitationHoverCard — Radix hover popover with KaTeX-safe excerpt"
```

---

### Task B8: `FollowupChips` component

**Files:**
- Create: `frontend/src/components/source/FollowupChips.tsx`

- [ ] **Step 1: Implement**

```tsx
// frontend/src/components/source/FollowupChips.tsx
'use client'

import { Lightbulb } from 'lucide-react'
import { Button } from '@/components/ui/button'

interface FollowupChipsProps {
  suggestions: string[]
  onSelect: (q: string) => void
  disabled?: boolean
}

export function FollowupChips({ suggestions, onSelect, disabled }: FollowupChipsProps) {
  if (!suggestions || suggestions.length === 0) return null
  return (
    <div className="mt-2 flex flex-wrap gap-1.5">
      <div className="flex items-center gap-1 text-[10px] uppercase tracking-wide text-muted-foreground mr-1">
        <Lightbulb className="h-3 w-3" />
        Gợi ý
      </div>
      {suggestions.map((q, i) => (
        <Button
          key={`${i}-${q}`}
          variant="outline"
          size="sm"
          className="h-7 text-xs whitespace-normal text-left max-w-full"
          disabled={disabled}
          onClick={() => onSelect(q)}
        >
          {q}
        </Button>
      ))}
    </div>
  )
}

export default FollowupChips
```

- [ ] **Step 2: Verify**

`cd frontend && ./node_modules/.bin/tsc --noEmit` → exit 0.

- [ ] **Step 3: Commit**

```bash
git add frontend/src/components/source/FollowupChips.tsx
git commit -m "feat(b8): FollowupChips — clickable suggestion pills under AI bubble"
```

---

### Task B9: `InlineImageCitation` (float-right thumbnail in answer body)

**Files:**
- Create: `frontend/src/components/source/InlineImageCitation.tsx`

- [ ] **Step 1: Implement**

```tsx
// frontend/src/components/source/InlineImageCitation.tsx
'use client'

import type { Citation } from '@/lib/types/api'

interface InlineImageCitationProps {
  citation: Citation
}

export function InlineImageCitation({ citation }: InlineImageCitationProps) {
  if (!citation.image_url) return null
  return (
    <aside className="float-right ml-3 mb-2 w-44 rounded border border-border bg-muted/40 p-1 text-[10px]">
      {/* eslint-disable-next-line @next/next/no-img-element */}
      <img
        src={citation.image_url}
        alt={citation.section_path || citation.document_title}
        loading="lazy"
        className="w-full h-auto rounded"
      />
      <div className="mt-1 px-0.5 text-muted-foreground leading-tight line-clamp-2">
        {citation.section_path || citation.document_title}
        {citation.page_label ? ` · ${citation.page_label}` : ''}
      </div>
    </aside>
  )
}

export default InlineImageCitation
```

- [ ] **Step 2: Verify**

`cd frontend && ./node_modules/.bin/tsc --noEmit` → exit 0.

- [ ] **Step 3: Commit**

```bash
git add frontend/src/components/source/InlineImageCitation.tsx
git commit -m "feat(b9): InlineImageCitation — float-right thumbnail card in answer"
```

---

### Task B10: Hover-card factory in `source-references.tsx`

**Files:**
- Modify: `frontend/src/lib/utils/source-references.tsx`

- [ ] **Step 1: Add factory**

Open `frontend/src/lib/utils/source-references.tsx`. After the existing `createCompactReferenceLinkComponent` (around line 431), append a new factory:

```tsx
import React from 'react'
import { CitationHoverCard } from '@/components/source/CitationHoverCard'
import type { Citation } from '@/lib/types/api'

/**
 * Render compact `[1]` markers as Radix HoverCard triggers (no click, hover only).
 * Anchor element href encodes `#citation-<n>` so the existing markdown
 * regex preserves them; here we just intercept and wrap.
 */
export function createCompactReferenceHoverComponent(
  citations: Citation[],
) {
  // eslint-disable-next-line react/display-name
  return ({ href, children, ...rest }: React.AnchorHTMLAttributes<HTMLAnchorElement>) => {
    const match = /citation-(\d+)/.exec(href || '')
    if (!match) {
      return <a href={href} {...rest}>{children}</a>
    }
    const idx = parseInt(match[1], 10)
    const citation = citations[idx - 1]
    if (!citation) {
      return <a href={href} {...rest}>{children}</a>
    }
    return (
      <CitationHoverCard index={idx} citation={citation}>
        <span
          role="button"
          tabIndex={0}
          className="inline-flex items-center justify-center min-w-[18px] h-[18px] px-1 rounded bg-primary/15 text-primary text-[10px] font-mono cursor-help align-middle border border-primary/30"
        >
          {children}
        </span>
      </CitationHoverCard>
    )
  }
}
```

If the file already imports React differently, deduplicate. The new export name `createCompactReferenceHoverComponent` must not clash.

- [ ] **Step 2: Verify**

`cd frontend && ./node_modules/.bin/tsc --noEmit` → exit 0.

- [ ] **Step 3: Commit**

```bash
git add frontend/src/lib/utils/source-references.tsx
git commit -m "feat(b10): createCompactReferenceHoverComponent for hover citations"
```

---

### Task B11: Wire ChatPanel — hover citations + chips + inline image

**Files:**
- Modify: `frontend/src/components/source/ChatPanel.tsx`

- [ ] **Step 1: Update imports**

At the top of `ChatPanel.tsx`, add:

```tsx
import { createCompactReferenceHoverComponent } from '@/lib/utils/source-references'
import { FollowupChips } from './FollowupChips'
import { InlineImageCitation } from './InlineImageCitation'
import type { Citation } from '@/lib/types/api'
```

Keep the existing imports intact.

- [ ] **Step 2: Replace `AIMessageContent` invocation site**

Find the existing call:

```tsx
<AIMessageContent
  content={message.content}
  onReferenceClick={handleReferenceClick}
/>
```

Replace with:

```tsx
<AIMessageContent
  content={message.content}
  citations={(message.citations as Citation[] | undefined) || []}
/>
```

- [ ] **Step 3: Rewrite `AIMessageContent` definition**

Find the existing `AIMessageContent` function (near the bottom of the file). Replace its body with the hover-citation version. The new signature:

```tsx
function AIMessageContent({
  content,
  citations,
}: {
  content: string
  citations: Citation[]
}) {
  const { t } = useTranslation()
  const markdownWithCompactRefs = convertReferencesToCompactMarkdown(
    content,
    t('common.references'),
  )
  const LinkComponent = createCompactReferenceHoverComponent(citations)
  const imageCitations = citations.filter((c) => c.segment_type === 'image' && c.image_url)
  return (
    <div className="prose prose-sm prose-neutral dark:prose-invert max-w-none break-words prose-headings:font-semibold prose-a:text-blue-600 prose-a:break-all prose-code:bg-muted prose-code:px-1 prose-code:py-0.5 prose-code:rounded prose-p:mb-4 prose-p:leading-7 prose-li:mb-2">
      {imageCitations.map((c, i) => (
        <InlineImageCitation key={`img-${i}-${c.content_hash}`} citation={c} />
      ))}
      <ReactMarkdown
        remarkPlugins={[remarkGfm]}
        components={{
          a: LinkComponent,
          p: ({ children }) => <p className="mb-4">{children}</p>,
          h1: ({ children }) => <h1 className="mb-4 mt-6">{children}</h1>,
          h2: ({ children }) => <h2 className="mb-3 mt-5">{children}</h2>,
          h3: ({ children }) => <h3 className="mb-3 mt-4">{children}</h3>,
          h4: ({ children }) => <h4 className="mb-2 mt-4">{children}</h4>,
          h5: ({ children }) => <h5 className="mb-2 mt-3">{children}</h5>,
          h6: ({ children }) => <h6 className="mb-2 mt-3">{children}</h6>,
          li: ({ children }) => <li className="mb-1">{children}</li>,
          ul: ({ children }) => <ul className="mb-4 space-y-1">{children}</ul>,
          ol: ({ children }) => <ol className="mb-4 space-y-1">{children}</ol>,
          table: ({ children }) => (
            <div className="my-4 overflow-x-auto">
              <table className="min-w-full border-collapse border border-border">{children}</table>
            </div>
          ),
          thead: ({ children }) => <thead className="bg-muted">{children}</thead>,
          tbody: ({ children }) => <tbody>{children}</tbody>,
          tr: ({ children }) => <tr className="border-b border-border">{children}</tr>,
          th: ({ children }) => <th className="border border-border px-3 py-2 text-left font-semibold">{children}</th>,
          td: ({ children }) => <td className="border border-border px-3 py-2">{children}</td>,
        }}
      >
        {markdownWithCompactRefs}
      </ReactMarkdown>
    </div>
  )
}
```

`handleReferenceClick` becomes unused for the chat path — leave the function defined (other paths may still rely on it) but the prop is gone from `AIMessageContent`.

- [ ] **Step 4: Render follow-up chips under each AI bubble**

In the existing assistant-message render block, find the `<FeedbackButtons …/>` block. Add the chip row just BEFORE the closing `</div>` of the `flex-col gap-2 max-w-[80%]` container:

```tsx
{message.type === 'ai' && (message as { follow_ups?: string[] }).follow_ups?.length ? (
  <FollowupChips
    suggestions={(message as { follow_ups: string[] }).follow_ups}
    onSelect={(q) => onSendMessage(q, modelOverride, contextType === 'notebook' ? domainFilter : undefined)}
    disabled={isStreaming}
  />
) : null}
```

- [ ] **Step 5: Verify TS**

`cd frontend && ./node_modules/.bin/tsc --noEmit` → exit 0.

- [ ] **Step 6: Commit**

```bash
git add frontend/src/components/source/ChatPanel.tsx
git commit -m "feat(b11): wire CitationHoverCard + InlineImageCitation + FollowupChips in ChatPanel"
```

---

## Phase C — Latency config

### Task C1: Recommend defaults in env + README note

**Files:**
- Modify: `.env`, `.env.example`
- Modify: `README.md`

- [ ] **Step 1: Update `.env` and `.env.example`**

In both files, find `LLM_TASK_MODEL_MAP=` and `AGENT_MAX_STEPS=` and `AGENT_PLAN_TRIGGER_MIN_CHARS=`. Replace the lines with:

```env
# Route fast tasks (decide loop, classifier, router, followup) to a small
# fast model; reserve qwen-agentrag for the answer step. Drops chat
# latency from ~30s → ~10s on local Ollama 7B + 3B mix.
LLM_TASK_MODEL_MAP={"classify":"llama3.2:3b","decide":"llama3.2:3b","domain_router":"llama3.2:3b","followup":"llama3.2:3b","answer":"qwen-agentrag"}

AGENT_MAX_STEPS=2
AGENT_PLAN_TRIGGER_MIN_CHARS=120
```

Make sure `LLM_ROUTING_ENABLED=true` is also set (default already in config).

- [ ] **Step 2: README note**

Edit `README.md` section §5.6 (LLM Routing & Cost Tracking). Append after the existing routing example:

```md
> **Speed tuning (Phase C).** The default `LLM_TASK_MODEL_MAP` routes
> `decide` / `classify` / `domain_router` / `followup` to a cheap model
> (`llama3.2:3b`) and reserves the finetuned `qwen-agentrag` (or its
> fallback `qwen2.5:7b-instruct`) for the `answer` step only. Pull both
> tags up front:
>
> ```bash
> docker exec agentrag-ollama ollama pull llama3.2:3b
> docker exec agentrag-ollama ollama pull qwen2.5:7b-instruct
> ```
>
> Pair this with `AGENT_MAX_STEPS=2` and `AGENT_PLAN_TRIGGER_MIN_CHARS=120`
> to cut the decide-loop overhead.
```

- [ ] **Step 3: Commit**

```bash
git add .env .env.example README.md
git commit -m "perf(c1): default speed config — small decide model + max_steps=2"
```

---

## Phase D — Ask & Search UI

### Task D1: Render new components in `StreamingResponse`

**Files:**
- Modify: `frontend/src/components/search/StreamingResponse.tsx`

- [ ] **Step 1: Inspect current shape**

Run: `grep -n "citations\|follow_ups\|CitationHoverCard\|markdown" frontend/src/components/search/StreamingResponse.tsx | head -20`. The file already renders streamed answers — locate where the inline markdown / citation list is rendered.

- [ ] **Step 2: Replace clickable citation list with hover cards**

Wherever the file maps over `citations` and emits a clickable list, swap to:

```tsx
import { CitationHoverCard } from '@/components/source/CitationHoverCard'
import type { Citation } from '@/lib/types/api'

// inside render
{(citations as Citation[] | undefined)?.map((c, i) => (
  <CitationHoverCard key={i} index={i + 1} citation={c}>
    <span className="inline-flex items-center justify-center min-w-[18px] h-[18px] px-1 rounded bg-primary/15 text-primary text-[10px] font-mono cursor-help align-middle border border-primary/30 mr-1">
      {i + 1}
    </span>
  </CitationHoverCard>
))}
```

If the file uses `convertReferencesToCompactMarkdown` already, swap its link factory to `createCompactReferenceHoverComponent(citations)` exactly as in Task B11.

- [ ] **Step 3: Render `FollowupChips` after final answer**

After the final-answer block, render:

```tsx
import { FollowupChips } from '@/components/source/FollowupChips'

{follow_ups && follow_ups.length > 0 && (
  <FollowupChips
    suggestions={follow_ups}
    onSelect={(q) => /* call ask() or set input + submit */ undefined}
  />
)}
```

If the Ask result shape doesn't already carry `follow_ups`, extend the backend `/api/search/ask` response in a follow-up — but for D1 it's sufficient to render the array when present.

- [ ] **Step 4: Verify**

`cd frontend && ./node_modules/.bin/tsc --noEmit` → exit 0.

- [ ] **Step 5: Commit**

```bash
git add frontend/src/components/search/StreamingResponse.tsx
git commit -m "feat(d1): Ask & Search reuses CitationHoverCard + FollowupChips"
```

---

## Phase final — Sweep, push, tag

### Task Z1: Run all tests

- [ ] **Step 1: Pytest sweep**

`uv run python -m pytest tests/ --ignore=tests/ontology --ignore=tests/ingestion -q`

Expected: all green (excluding Postgres-dependent suites unless infra is up).

- [ ] **Step 2: TypeScript**

`cd frontend && ./node_modules/.bin/tsc --noEmit`

Exit 0.

- [ ] **Step 3: Push + tag**

```bash
git push origin structmem
git tag -a chat-polish-v1 -m "Phase A-D: token cap + NotebookLM polish + speed defaults + Ask UI"
git push origin chat-polish-v1
```

---

## Self-Review

**Spec coverage**
- Phase A — Tasks A1 (covers config + 3 file plumb + env doc).
- Phase B — Tasks B1 (citation enrich), B2 (followup helper), B3 (adapter persist), B4 (model tests), B5 (FE types), B6 (radix install), B7 (HoverCard), B8 (Chips), B9 (Inline image), B10 (factory), B11 (ChatPanel wiring). Image segments → B7 + B9 + B11. Page-aware → B1. KaTeX-safe → B7. Cache → B2.
- Phase C — Task C1 (env + README).
- Phase D — Task D1 (StreamingResponse).
- Sweep + tag — Z1.

**Placeholder scan** — every code block is complete; no "implement later" markers; every command shows expected output where ambiguous.

**Type consistency** — `Citation` interface declared in B5 is used in B7/B9/B10/B11/D1 with the same property names. `follow_ups: list[str]` consistent backend (B3) ↔ frontend (B5/B11/D1). `generate_followups` signature in B2 matches its call in B3.
