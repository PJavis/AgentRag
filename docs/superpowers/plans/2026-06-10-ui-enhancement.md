# UI Enhancement Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Surface the 5 new RAG signals (fast-path, semantic cache, CRAG critique/corrective, multi-hop, RAPTOR, contextual retrieval) in the chat UI, and add cost-dashboard charts + skeleton loaders + light animation.

**Architecture:** Frontend changes in `frontend/src` (types → small focused components → wiring) plus a thin backend shim that propagates already-computed signals into the chat response. New deps: `recharts`, `tailwindcss-animate`. RAG chips guard on signal presence → zero visual change when backend flags are OFF.

**Tech Stack:** Next.js 16 / React 19, Radix + Tailwind (OKLCH tokens), react-query, react-markdown. Tests: **Vitest** (`vitest run`) + `@testing-library/react` + jsdom. Backend: Python + pytest.

**Conventions (read before starting):**
- Frontend tests: files `*.test.ts(x)` next to source; import `{ describe, it, expect, vi }` from `'vitest'`, `render`/`screen` from `'@testing-library/react'`. Mirror `frontend/src/components/common/ConfirmDialog.test.tsx`.
- Run ONE frontend test: from `frontend/`, `npx vitest run src/path/file.test.tsx`. Run all: `npm test`. Typecheck: `npx tsc --noEmit`.
- Backend tests from repo root: `uv run pytest <path> -v`.
- i18n: add new strings to `frontend/src/lib/locales/en-US/index.ts`; other locales fall back to en-US automatically (i18next `fallbackLng: 'en-US'`). There is NO `vi-VN` locale. Components read strings via `const { t } = useTranslation()` then `t('namespace.key')`.
- UI primitives live in `frontend/src/components/ui/`; styling via CVA + Tailwind + OKLCH CSS vars (`--chart-1..5` exist in `globals.css`).
- Commit after each task. Work on branch `feat/ragas-langfuse-reranker`.

---

## File Structure

**New files:**
- `frontend/src/components/source/MessageSignals.tsx` (+ `.test.tsx`) — chip row under each answer.
- `frontend/src/components/ui/skeleton.tsx` — skeleton primitive.
- `frontend/src/lib/utils/cost-charts.ts` (+ `.test.ts`) — pure chart data transforms.
- `frontend/src/components/cost/CostCharts.tsx` — recharts panel.

**Modified:**
- `src/agentrag/agent/context.py` — `_stage_citation_pack` preserves `node_level`/`context_text`.
- `src/agentrag/agent/service.py` — `_build_packed_citations` copies them; chat-response message signals.
- `src/agentrag/agent/graph_service.py` — `GraphAgentService.chat` return adds message-level signals.
- `frontend/src/lib/types/api.ts` — Citation + message types.
- `frontend/src/components/source/ChatPanel.tsx` — swap reasoning-path badge → `MessageSignals`.
- `frontend/src/components/source/TraceDialog.tsx` — critique status, multi-hop, corrective, diagnostics, fast-path.
- `frontend/src/components/source/CitationHoverCard.tsx` — RAPTOR badge + context line.
- `frontend/src/app/(dashboard)/cost/page.tsx` — Charts tab.
- `frontend/tailwind.config.ts`, `frontend/package.json` — deps.
- `frontend/src/lib/locales/en-US/index.ts` — new strings.

---

## PHASE 1 — Backend signal propagation

### Task 1.1: Preserve + expose `node_level` and `context_text` on citations

**Files:**
- Modify: `src/agentrag/agent/context.py:202-219` (`_stage_citation_pack`)
- Modify: `src/agentrag/agent/service.py:729-738` (`_build_packed_citations`)
- Test: `tests/agent/test_citation_signal_fields.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/agent/test_citation_signal_fields.py
from src.agentrag.agent.service import AgentService


def test_build_packed_citations_carries_node_level_and_context_text():
    svc = AgentService.__new__(AgentService)
    packed = [
        {"document_title": "Tim mạch", "content": "Nhồi máu cơ tim ...",
         "segment_type": "raptor_summary", "node_level": 1,
         "context_text": "From the cardiology chapter on MI."},
        {"document_title": "Tim mạch", "content": "Aspirin ...",
         "segment_type": "text", "node_level": 0},
    ]
    out = svc._build_packed_citations(packed)
    assert out[0]["node_level"] == 1
    assert out[0]["segment_type"] == "raptor_summary"
    assert out[0]["context_text"] == "From the cardiology chapter on MI."
    # leaf with no context_text → field absent or None, never crashes
    assert out[1]["node_level"] == 0
    assert out[1].get("context_text") in (None, "")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/agent/test_citation_signal_fields.py -v`
Expected: FAIL (`KeyError: 'node_level'`)

- [ ] **Step 3: Add the fields in both places**

In `src/agentrag/agent/context.py`, inside `_stage_citation_pack`'s `entry` dict (after the `"segment_type": item.get("segment_type", "text"),` line ~212), add:

```python
                "node_level": item.get("node_level", 0),
                "context_text": item.get("context_text"),
```

In `src/agentrag/agent/service.py`, inside `_build_packed_citations`'s `entry` dict (after the `"segment_type": seg_type,` line ~736), add:

```python
                "node_level": item.get("node_level", 0),
                "context_text": item.get("context_text"),
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/agent/test_citation_signal_fields.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/agentrag/agent/context.py src/agentrag/agent/service.py tests/agent/test_citation_signal_fields.py
git commit -m "feat(agent): propagate node_level + context_text onto citations (UI signals)"
```

---

### Task 1.2: Expose message-level signals (semantic_cache_hit, retrieval_mode)

**Files:**
- Modify: `src/agentrag/agent/graph_service.py` (`GraphAgentService.chat` return ~`:474-487`; add a module helper `_message_signals`)
- Test: `tests/agent/test_message_signals.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/agent/test_message_signals.py
from src.agentrag.agent.graph_service import _message_signals


def test_message_signals_reads_cache_hit_and_mode_from_first_trace():
    tool_trace = [
        {"tool_name": "search_hybrid_kg",
         "tool_output": {"semantic_cache_hit": True, "mode": "hybrid_kg",
                         "domain_route": "tim_mach"}},
        {"tool_name": "search_hybrid_kg", "tool_output": {"mode": "hybrid"}},
    ]
    sig = _message_signals(tool_trace)
    assert sig["semantic_cache_hit"] is True
    assert sig["retrieval_mode"] == "hybrid_kg"
    assert sig["domain_route"] == "tim_mach"


def test_message_signals_defaults_when_absent():
    sig = _message_signals([])
    assert sig["semantic_cache_hit"] is False
    assert sig["retrieval_mode"] is None
    assert sig["domain_route"] is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/agent/test_message_signals.py -v`
Expected: FAIL (`ImportError: cannot import name '_message_signals'`)

- [ ] **Step 3: Add helper + include in chat return**

In `src/agentrag/agent/graph_service.py`, add a module-level helper (near the other module helpers, after `_chain_query`):

```python
def _message_signals(tool_trace: list[dict[str, Any]] | None) -> dict[str, Any]:
    """Derive UI message-level signals from the tool trace: whether the first
    retrieval was a semantic-cache hit, the retrieval mode, and the domain
    route. All default-absent so the UI chips stay hidden when unavailable."""
    cache_hit = False
    mode: str | None = None
    domain: str | None = None
    for entry in tool_trace or []:
        out = entry.get("tool_output") or {}
        if not isinstance(out, dict):
            continue
        if out.get("semantic_cache_hit"):
            cache_hit = True
        if mode is None and out.get("mode"):
            mode = out.get("mode")
        if domain is None and out.get("domain_route"):
            domain = out.get("domain_route")
    return {"semantic_cache_hit": cache_hit, "retrieval_mode": mode, "domain_route": domain}
```

In `GraphAgentService.chat`, in the returned dict (the `return { ... }` ~line 474), add (after the `"context": state.get("packed_context", []),` line):

```python
            **_message_signals(state.get("tool_trace")),
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/agent/test_message_signals.py -v`
Expected: PASS

- [ ] **Step 5: Run agent suite for regressions**

Run: `uv run pytest tests/agent/ -q`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add src/agentrag/agent/graph_service.py tests/agent/test_message_signals.py
git commit -m "feat(agent): expose semantic_cache_hit/retrieval_mode/domain_route on chat response"
```

---

## PHASE 2 — Frontend types

### Task 2.1: Extend Citation + message types

**Files:**
- Modify: `frontend/src/lib/types/api.ts:153-168` (Citation), `:170-182` (SourceChatMessage), `:245-257` (NotebookChatMessage)
- Test: `frontend/src/lib/types/signal-types.test.ts`

- [ ] **Step 1: Write the failing test** (a compile-level guard that the fields exist)

```typescript
// frontend/src/lib/types/signal-types.test.ts
import { describe, it, expect } from 'vitest'
import type { Citation, NotebookChatMessage } from './api'

describe('RAG signal type fields', () => {
  it('Citation accepts node_level, context_text, raptor_summary segment_type', () => {
    const c: Citation = {
      document_title: 'x', node_level: 1, context_text: 'ctx',
      segment_type: 'raptor_summary',
    }
    expect(c.node_level).toBe(1)
    expect(c.segment_type).toBe('raptor_summary')
  })
  it('NotebookChatMessage accepts semantic_cache_hit + domain_route + retrieval_mode', () => {
    const m: NotebookChatMessage = {
      id: '1', type: 'ai', content: 'a',
      semantic_cache_hit: true, domain_route: 'tim_mach', retrieval_mode: 'hybrid_kg',
    }
    expect(m.semantic_cache_hit).toBe(true)
  })
})
```

- [ ] **Step 2: Run test to verify it fails**

Run (from `frontend/`): `npx vitest run src/lib/types/signal-types.test.ts`
Expected: FAIL (TS error — `node_level` not in Citation).

- [ ] **Step 3: Extend the types**

In `frontend/src/lib/types/api.ts`, change `Citation.segment_type` (line 166) and add two fields:

```typescript
  segment_type?: 'text' | 'image' | 'table' | 'raptor_summary'
  node_level?: number | null
  context_text?: string | null
```

Add to BOTH `SourceChatMessage` (after `follow_ups?: string[]` ~line 181) and `NotebookChatMessage` (after `follow_ups?: string[]` ~line 256):

```typescript
  semantic_cache_hit?: boolean | null
  domain_route?: string | null
  retrieval_mode?: string | null
```

- [ ] **Step 4: Run test to verify it passes**

Run: `npx vitest run src/lib/types/signal-types.test.ts`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add frontend/src/lib/types/api.ts frontend/src/lib/types/signal-types.test.ts
git commit -m "feat(types): add RAG signal fields to Citation + chat message"
```

---

## PHASE 3 — MessageSignals chips

### Task 3.1: `MessageSignals` component + helper

**Files:**
- Create: `frontend/src/components/source/MessageSignals.tsx`
- Test: `frontend/src/components/source/MessageSignals.test.tsx`

- [ ] **Step 1: Write the failing test**

```tsx
// frontend/src/components/source/MessageSignals.test.tsx
import { describe, it, expect } from 'vitest'
import { render, screen } from '@testing-library/react'
import { MessageSignals, deriveCritiqueState } from './MessageSignals'
import type { NotebookChatMessage } from '@/lib/types/api'

function msg(over: Partial<NotebookChatMessage>): NotebookChatMessage {
  return { id: '1', type: 'ai', content: 'a', ...over }
}

describe('deriveCritiqueState', () => {
  it('verified when critique ran and no corrective', () => {
    expect(deriveCritiqueState(msg({ timings_ms: { critique: 12 } }))).toBe('verified')
  })
  it('corrected when a corrective trace entry exists', () => {
    expect(deriveCritiqueState(msg({
      timings_ms: { critique: 12 },
      tool_trace: [{ tool_name: 'search_hybrid_kg', corrective: true }],
    }))).toBe('corrected')
  })
  it('null when critique never ran', () => {
    expect(deriveCritiqueState(msg({}))).toBeNull()
  })
})

describe('MessageSignals', () => {
  it('shows fast-path + cache chips', () => {
    render(<MessageSignals message={msg({ reasoning_path: 'fast', semantic_cache_hit: true })} />)
    expect(screen.getByText(/fast path/i)).toBeInTheDocument()
    expect(screen.getByText(/instant/i)).toBeInTheDocument()
  })
  it('renders nothing when no signals present', () => {
    const { container } = render(<MessageSignals message={msg({})} />)
    expect(container.textContent).toBe('')
  })
})
```

- [ ] **Step 2: Run test to verify it fails**

Run: `npx vitest run src/components/source/MessageSignals.test.tsx`
Expected: FAIL (module not found).

- [ ] **Step 3: Implement `MessageSignals.tsx`**

```tsx
// frontend/src/components/source/MessageSignals.tsx
'use client'

import { Zap, Brain, Database, RotateCcw, Hospital, Sparkles } from 'lucide-react'
import { Badge } from '@/components/ui/badge'
import type { NotebookChatMessage, SourceChatMessage } from '@/lib/types/api'

type ChatMsg = NotebookChatMessage | SourceChatMessage

export function deriveCritiqueState(message: ChatMsg): 'verified' | 'corrected' | null {
  const ranCritique = typeof message.timings_ms?.critique === 'number'
  if (!ranCritique) return null
  const corrected = (message.tool_trace ?? []).some(
    (e) => (e as Record<string, unknown>).corrective === true,
  )
  return corrected ? 'corrected' : 'verified'
}

const PATH_META: Record<string, { label: string; icon: typeof Zap }> = {
  fast: { label: 'Fast path', icon: Zap },
  semantic: { label: 'Semantic', icon: Brain },
  structured: { label: 'Structured', icon: Database },
}

export function MessageSignals({ message }: { message: ChatMsg }) {
  const path = message.reasoning_path || ''
  const pathMeta = PATH_META[path]
  const critique = deriveCritiqueState(message)
  const chips: React.ReactNode[] = []

  if (pathMeta) {
    const Icon = pathMeta.icon
    chips.push(
      <Badge key="path" variant="outline" className="text-[10px] h-5 gap-1 animate-in fade-in"
        title="Reasoning path used for this answer">
        <Icon className="h-3 w-3" />{pathMeta.label}
      </Badge>,
    )
  } else if (path) {
    chips.push(
      <Badge key="path" variant="outline" className="text-[10px] h-5 font-mono animate-in fade-in">
        {path}
      </Badge>,
    )
  }

  if (message.semantic_cache_hit) {
    chips.push(
      <Badge key="cache" variant="secondary"
        className="text-[10px] h-5 gap-1 animate-in fade-in text-amber-700 dark:text-amber-300"
        title="Served from semantic cache — near-instant">
        <Sparkles className="h-3 w-3" />Instant · cached
      </Badge>,
    )
  }

  if (critique === 'verified') {
    chips.push(
      <Badge key="crit" variant="outline" className="text-[10px] h-5 gap-1 animate-in fade-in
        text-emerald-700 dark:text-emerald-300 border-emerald-300/50"
        title="Answer passed the CRAG grounding check">
        <Brain className="h-3 w-3" />Verified
      </Badge>,
    )
  } else if (critique === 'corrected') {
    chips.push(
      <Badge key="crit" variant="outline" className="text-[10px] h-5 gap-1 animate-in fade-in
        text-blue-700 dark:text-blue-300 border-blue-300/50"
        title="Answer was re-retrieved + revised by CRAG correction">
        <RotateCcw className="h-3 w-3" />Self-corrected
      </Badge>,
    )
  }

  if (message.domain_route) {
    chips.push(
      <Badge key="domain" variant="outline" className="text-[10px] h-5 gap-1 animate-in fade-in"
        title="Domain the query was routed to">
        <Hospital className="h-3 w-3" />{message.domain_route}
      </Badge>,
    )
  }

  if (chips.length === 0) return null
  return <div className="flex items-center gap-1 flex-wrap">{chips}</div>
}

export default MessageSignals
```

- [ ] **Step 4: Run test to verify it passes**

Run: `npx vitest run src/components/source/MessageSignals.test.tsx`
Expected: PASS (5 assertions across 5 tests).

- [ ] **Step 5: Commit**

```bash
git add frontend/src/components/source/MessageSignals.tsx frontend/src/components/source/MessageSignals.test.tsx
git commit -m "feat(chat): MessageSignals chips (fast-path, cache, verified, domain)"
```

---

### Task 3.2: Wire `MessageSignals` into ChatPanel

**Files:**
- Modify: `frontend/src/components/source/ChatPanel.tsx:348-356` (replace raw reasoning_path badge), import at top.
- Test: covered by `MessageSignals.test.tsx` + manual typecheck.

- [ ] **Step 1: Add the import**

Near the other component imports at the top of `frontend/src/components/source/ChatPanel.tsx`, add:

```tsx
import { MessageSignals } from './MessageSignals'
```

- [ ] **Step 2: Replace the reasoning-path badge block**

Replace lines ~348-356 (the `{(message as { reasoning_path?: string }).reasoning_path && (<Badge ...>...</Badge>)}` block) with:

```tsx
                          <MessageSignals message={message} />
```

- [ ] **Step 3: Typecheck**

Run (from `frontend/`): `npx tsc --noEmit`
Expected: no new errors. (If `message` type is a union missing fields, the MessageSignals prop accepts both `NotebookChatMessage | SourceChatMessage`, so it compiles.)

- [ ] **Step 4: Run the chat panel tests**

Run: `npx vitest run src/components/source/ChatPanel.starters.test.tsx src/components/source/MessageSignals.test.tsx`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add frontend/src/components/source/ChatPanel.tsx
git commit -m "feat(chat): use MessageSignals chips in place of raw reasoning-path badge"
```

---

## PHASE 4 — TraceDialog upgrade

### Task 4.1: Multi-hop grouping + corrective rows + fast-path explainer

**Files:**
- Modify: `frontend/src/components/source/TraceDialog.tsx` (add helper `groupMultiHop`, render corrective/multihop in `ToolTraceList`, add fast-path info block, mark critique stage)
- Test: `frontend/src/components/source/TraceDialog.test.tsx`

- [ ] **Step 1: Write the failing test**

```tsx
// frontend/src/components/source/TraceDialog.test.tsx
import { describe, it, expect } from 'vitest'
import { render, screen } from '@testing-library/react'
import { TraceDialog, groupMultiHop } from './TraceDialog'
import type { NotebookChatMessage } from '@/lib/types/api'

describe('groupMultiHop', () => {
  it('flags multihop and corrective entries', () => {
    const grouped = groupMultiHop([
      { tool_name: 'a', multihop: true },
      { tool_name: 'b', corrective: true },
      { tool_name: 'c' },
    ])
    expect(grouped[0].isMultiHop).toBe(true)
    expect(grouped[1].isCorrective).toBe(true)
    expect(grouped[2].isMultiHop).toBe(false)
  })
})

describe('TraceDialog', () => {
  const base: NotebookChatMessage = {
    id: '1', type: 'ai', content: 'a',
    timings_ms: { total: 100, answer: 50 },
  }
  it('shows fast-path explainer when reasoning_path=fast', () => {
    render(<TraceDialog open onOpenChange={() => {}}
      message={{ ...base, reasoning_path: 'fast' }} />)
    expect(screen.getByText(/fast path/i)).toBeInTheDocument()
  })
  it('labels a corrective tool entry', () => {
    render(<TraceDialog open onOpenChange={() => {}}
      message={{ ...base, tool_trace: [{ tool_name: 'search_hybrid_kg', corrective: true }] }} />)
    expect(screen.getByText(/corrective/i)).toBeInTheDocument()
  })
})
```

- [ ] **Step 2: Run test to verify it fails**

Run: `npx vitest run src/components/source/TraceDialog.test.tsx`
Expected: FAIL (`groupMultiHop` not exported).

- [ ] **Step 3: Add the helper + rendering**

In `frontend/src/components/source/TraceDialog.tsx`, add the exported helper near the top (after the imports):

```tsx
export interface TaggedTrace extends ToolTraceEntry {
  isMultiHop: boolean
  isCorrective: boolean
}

export function groupMultiHop(entries: ToolTraceEntry[]): TaggedTrace[] {
  return (entries ?? []).map((e) => ({
    ...e,
    isMultiHop: (e as Record<string, unknown>).multihop === true,
    isCorrective: (e as Record<string, unknown>).corrective === true,
  }))
}
```

Change `ToolTraceList` to tag entries and badge multihop/corrective. Replace its body's `entries.map((e, i) => {` header region: compute `const tagged = groupMultiHop(entries)` at the top of the component, iterate `tagged`, and inside each row header (next to the `{name}` span, ~line 105) add:

```tsx
                {e.isCorrective && (
                  <Badge variant="outline" className="text-[10px] gap-0.5 text-blue-600 dark:text-blue-300">
                    <RotateCcw className="h-2.5 w-2.5" />corrective
                  </Badge>
                )}
                {e.isMultiHop && (
                  <Badge variant="outline" className="text-[10px] text-violet-600 dark:text-violet-300">
                    hop
                  </Badge>
                )}
```

Add `RotateCcw` to the lucide import (line 13): `import { ChevronRight, Activity, Clock, ListTree, RotateCcw } from 'lucide-react'`.

Add a fast-path explainer block in the `ScrollArea` content, right after the `chitchat` block (~line 199):

```tsx
            {path === 'fast' && (
              <div className="border rounded-lg p-3 bg-emerald-50/40 dark:bg-emerald-950/20 border-emerald-200/60 dark:border-emerald-900/40 text-sm">
                <div className="font-medium text-emerald-900 dark:text-emerald-200 mb-1">
                  ⚡ Fast path
                </div>
                <div className="text-xs text-muted-foreground">
                  Simple, single-domain question — answered with one retrieve + one LLM call,
                  skipping the plan→decide→tool loop for lower latency.
                </div>
              </div>
            )}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `npx vitest run src/components/source/TraceDialog.test.tsx`
Expected: PASS

- [ ] **Step 5: Typecheck + commit**

Run: `npx tsc --noEmit` (expect no new errors), then:

```bash
git add frontend/src/components/source/TraceDialog.tsx frontend/src/components/source/TraceDialog.test.tsx
git commit -m "feat(trace): multi-hop/corrective tags + fast-path explainer"
```

---

### Task 4.2: Per-tool retrieval diagnostics + critique stage marker

**Files:**
- Modify: `frontend/src/components/source/TraceDialog.tsx` (diagnostics row in expanded tool detail; critique stage ✓/↻ marker)
- Test: extend `frontend/src/components/source/TraceDialog.test.tsx`

- [ ] **Step 1: Add the failing test (append to TraceDialog.test.tsx)**

```tsx
  it('shows cache-hit + mode diagnostics for a tool', () => {
    render(<TraceDialog open onOpenChange={() => {}}
      message={{ id: '1', type: 'ai', content: 'a',
        tool_trace: [{ tool_name: 'search_hybrid_kg',
          tool_output: { semantic_cache_hit: true, mode: 'hybrid_kg' } }] }} />)
    // diagnostics chip text appears once the row is rendered
    expect(screen.getByText(/cached/i)).toBeInTheDocument()
  })
```

- [ ] **Step 2: Run test to verify it fails**

Run: `npx vitest run src/components/source/TraceDialog.test.tsx`
Expected: FAIL (no "cached" text).

- [ ] **Step 3: Render diagnostics**

In `ToolTraceList`, in each row header (the right-side `<div className="flex gap-1 shrink-0">` block ~line 112), prepend a cache badge derived from `tool_output`:

```tsx
                {((e.tool_output as Record<string, unknown> | undefined)?.semantic_cache_hit) && (
                  <Badge variant="secondary" className="text-[10px] gap-0.5 text-amber-700 dark:text-amber-300">
                    cached
                  </Badge>
                )}
                {(e.tool_output as Record<string, unknown> | undefined)?.mode != null && (
                  <Badge variant="outline" className="text-[10px] font-mono">
                    {String((e.tool_output as Record<string, unknown>).mode)}
                  </Badge>
                )}
```

For the critique stage marker: in `StageNode`, when `label === 'Critique'` (or pass a `marker` prop), it already renders from `stageEntries`. Keep minimal — the critique stage already appears in the pipeline when `timings_ms.critique > 0` (STAGE_ORDER already includes it). No extra code needed beyond confirming it renders; this step's test only asserts the diagnostics badge.

- [ ] **Step 4: Run test to verify it passes**

Run: `npx vitest run src/components/source/TraceDialog.test.tsx`
Expected: PASS (all TraceDialog tests).

- [ ] **Step 5: Commit**

```bash
git add frontend/src/components/source/TraceDialog.tsx frontend/src/components/source/TraceDialog.test.tsx
git commit -m "feat(trace): per-tool cache/mode diagnostics"
```

---

## PHASE 5 — Citation hover (RAPTOR + contextual)

### Task 5.1: RAPTOR summary badge + context_text line

**Files:**
- Modify: `frontend/src/components/source/CitationHoverCard.tsx`
- Test: `frontend/src/components/source/CitationHoverCard.test.tsx`

- [ ] **Step 1: Write the failing test**

```tsx
// frontend/src/components/source/CitationHoverCard.test.tsx
import { describe, it, expect } from 'vitest'
import { render, screen } from '@testing-library/react'
import { CitationHoverCard } from './CitationHoverCard'
import type { Citation } from '@/lib/types/api'

function open(citation: Citation) {
  // Radix HoverCard content renders into a portal only on open; assert via the
  // exported pure header renderer instead. We render the trigger + force-open by
  // passing the citation to the content through a test wrapper is complex, so we
  // test the pure helpers (see below) plus a smoke render.
  return render(
    <CitationHoverCard index={1} citation={citation}>
      <button>ref</button>
    </CitationHoverCard>,
  )
}

describe('CitationHoverCard', () => {
  it('renders trigger without crashing for a raptor summary citation', () => {
    open({ document_title: 'Tim mạch', node_level: 1, segment_type: 'raptor_summary',
           context_text: 'From the cardiology chapter', excerpt: 'summary text' })
    expect(screen.getByText('ref')).toBeInTheDocument()
  })
})
```

> Note: Radix HoverCard.Content is portalled + only mounts on hover, so a DOM-level assertion on the badge is brittle in jsdom. This task therefore extracts a pure, testable header renderer and unit-tests THAT (next test), keeping the component test a smoke render.

Add a second test file for the pure renderer:

```tsx
// frontend/src/components/source/citation-hover-helpers.test.tsx
import { describe, it, expect } from 'vitest'
import { render, screen } from '@testing-library/react'
import { CitationContextLine, RaptorBadge } from './CitationHoverCard'

describe('citation hover helpers', () => {
  it('RaptorBadge shows level for node_level>=1', () => {
    render(<RaptorBadge nodeLevel={2} />)
    expect(screen.getByText(/L2/)).toBeInTheDocument()
  })
  it('RaptorBadge renders nothing for leaf nodes', () => {
    const { container } = render(<RaptorBadge nodeLevel={0} />)
    expect(container.textContent).toBe('')
  })
  it('CitationContextLine shows context text', () => {
    render(<CitationContextLine text="From cardiology" />)
    expect(screen.getByText(/From cardiology/)).toBeInTheDocument()
  })
  it('CitationContextLine renders nothing when empty', () => {
    const { container } = render(<CitationContextLine text={null} />)
    expect(container.textContent).toBe('')
  })
})
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `npx vitest run src/components/source/citation-hover-helpers.test.tsx`
Expected: FAIL (`RaptorBadge` not exported).

- [ ] **Step 3: Add the helpers + use them in the card**

In `frontend/src/components/source/CitationHoverCard.tsx`, add `Layers` to the lucide import and export two small components:

```tsx
import { FileText, Image as ImageIcon, Layers } from 'lucide-react'
```

```tsx
export function RaptorBadge({ nodeLevel }: { nodeLevel?: number | null }) {
  if (!nodeLevel || nodeLevel < 1) return null
  return (
    <Badge variant="outline" className="text-[10px] px-1 py-0 gap-0.5 text-violet-600 dark:text-violet-300 border-violet-300/50">
      <Layers className="h-2.5 w-2.5" />Summary · L{nodeLevel}
    </Badge>
  )
}

export function CitationContextLine({ text }: { text?: string | null }) {
  const t = (text || '').trim()
  if (!t) return null
  return <div className="text-[10px] text-muted-foreground italic mb-1 line-clamp-2">{t}</div>
}
```

In the card header (the row with `[{index}]` + `document_title` + `page_label` badge, ~line 65-72), add `<RaptorBadge nodeLevel={citation.node_level} />` next to the `page_label` badge. Just above the excerpt body (the `<div className="border-t pt-2">` ~line 81), insert `<CitationContextLine text={citation.context_text} />`:

```tsx
          <div className="border-t pt-2">
            <CitationContextLine text={citation.context_text} />
            {renderExcerptBody(citation)}
          </div>
```

Also make `raptor_summary` use the FileText icon path (it's not an image): the existing `isImage = citation.segment_type === 'image'` already handles that correctly (raptor_summary → FileText). No change needed.

- [ ] **Step 4: Run tests to verify they pass**

Run: `npx vitest run src/components/source/citation-hover-helpers.test.tsx src/components/source/CitationHoverCard.test.tsx`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add frontend/src/components/source/CitationHoverCard.tsx frontend/src/components/source/CitationHoverCard.test.tsx frontend/src/components/source/citation-hover-helpers.test.tsx
git commit -m "feat(citation): RAPTOR summary badge + contextual-retrieval context line"
```

---

## PHASE 6 — Cost charts + skeletons + animation

### Task 6.1: Add deps (recharts, tailwindcss-animate)

**Files:**
- Modify: `frontend/package.json`, `frontend/tailwind.config.ts`

- [ ] **Step 1: Install deps**

Run (from `frontend/`):
```bash
npm install recharts && npm install -D tailwindcss-animate
```

- [ ] **Step 2: Register the tailwind plugin**

In `frontend/tailwind.config.ts`, add `tailwindcss-animate` to the `plugins` array (import at top: `import animate from 'tailwindcss-animate'`, then `plugins: [..., animate]`). Read the current file first to match the existing plugins entry (it already uses `@tailwindcss/typography`).

- [ ] **Step 3: Verify build still compiles**

Run: `npx tsc --noEmit` (expect no errors) and `npx vitest run src/lib/config.test.ts` (sanity — expect PASS).

- [ ] **Step 4: Commit**

```bash
git add frontend/package.json frontend/package-lock.json frontend/tailwind.config.ts
git commit -m "build(frontend): add recharts + tailwindcss-animate"
```

---

### Task 6.2: Cost chart data transforms (pure, unit-tested)

**Files:**
- Create: `frontend/src/lib/utils/cost-charts.ts`
- Test: `frontend/src/lib/utils/cost-charts.test.ts`

- [ ] **Step 1: Write the failing test**

```typescript
// frontend/src/lib/utils/cost-charts.test.ts
import { describe, it, expect } from 'vitest'
import { perModelSpend, costOverTime } from './cost-charts'
import type { CostBucket, CostEntry } from '@/lib/api/metrics'

const bucket = (usd: number, calls = 1): CostBucket => ({
  calls, in_tokens: 0, out_tokens: 0, total_latency_ms: 0, avg_latency_ms: 0, usd,
})

describe('perModelSpend', () => {
  it('returns sorted [{name, usd, calls}] descending by usd', () => {
    const out = perModelSpend({ 'gpt-4': bucket(2, 3), 'mini': bucket(5, 9) })
    expect(out[0]).toEqual({ name: 'mini', usd: 5, calls: 9 })
    expect(out[1].name).toBe('gpt-4')
  })
  it('empty input → empty array', () => {
    expect(perModelSpend({})).toEqual([])
  })
})

describe('costOverTime', () => {
  it('buckets entries by minute and accumulates usd', () => {
    const entries: CostEntry[] = [
      { id: 1, timestamp: 60, task: 'chat', model: 'm', latency_ms: 0, in_tokens: 0, out_tokens: 0, usd: 0.1, usage_source: 'estimate' },
      { id: 2, timestamp: 90, task: 'chat', model: 'm', latency_ms: 0, in_tokens: 0, out_tokens: 0, usd: 0.2, usage_source: 'estimate' },
      { id: 3, timestamp: 130, task: 'chat', model: 'm', latency_ms: 0, in_tokens: 0, out_tokens: 0, usd: 0.4, usage_source: 'estimate' },
    ]
    const out = costOverTime(entries)
    // minute 1 (60-119s) → 0.3 ; minute 2 (120-179s) → 0.4
    expect(out).toHaveLength(2)
    expect(out[0].usd).toBeCloseTo(0.3)
    expect(out[1].usd).toBeCloseTo(0.4)
  })
  it('empty → empty', () => {
    expect(costOverTime([])).toEqual([])
  })
})
```

- [ ] **Step 2: Run test to verify it fails**

Run: `npx vitest run src/lib/utils/cost-charts.test.ts`
Expected: FAIL (module not found).

- [ ] **Step 3: Implement `cost-charts.ts`**

```typescript
// frontend/src/lib/utils/cost-charts.ts
import type { CostBucket, CostEntry } from '@/lib/api/metrics'

export interface ModelSpend { name: string; usd: number; calls: number }
export interface TimePoint { t: number; label: string; usd: number; tokens: number }

export function perModelSpend(perModel: Record<string, CostBucket>): ModelSpend[] {
  return Object.entries(perModel)
    .map(([name, b]) => ({ name, usd: b.usd, calls: b.calls }))
    .sort((a, b) => b.usd - a.usd)
}

export function costOverTime(entries: CostEntry[], bucketSec = 60): TimePoint[] {
  if (!entries.length) return []
  const buckets = new Map<number, TimePoint>()
  for (const e of entries) {
    const key = Math.floor(e.timestamp / bucketSec) * bucketSec
    const prev = buckets.get(key)
    const label = new Date(key * 1000).toLocaleTimeString(undefined, { hour12: false })
    if (prev) {
      prev.usd += e.usd
      prev.tokens += e.in_tokens + e.out_tokens
    } else {
      buckets.set(key, { t: key, label, usd: e.usd, tokens: e.in_tokens + e.out_tokens })
    }
  }
  return Array.from(buckets.values()).sort((a, b) => a.t - b.t)
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `npx vitest run src/lib/utils/cost-charts.test.ts`
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
git add frontend/src/lib/utils/cost-charts.ts frontend/src/lib/utils/cost-charts.test.ts
git commit -m "feat(cost): pure chart data transforms (perModelSpend, costOverTime)"
```

---

### Task 6.3: `CostCharts` component + Charts tab

**Files:**
- Create: `frontend/src/components/cost/CostCharts.tsx`
- Modify: `frontend/src/app/(dashboard)/cost/page.tsx` (add a "Charts" tab)
- Test: smoke render in `frontend/src/components/cost/CostCharts.test.tsx`

- [ ] **Step 1: Write the failing smoke test**

```tsx
// frontend/src/components/cost/CostCharts.test.tsx
import { describe, it, expect, vi } from 'vitest'
import { render } from '@testing-library/react'
import { CostCharts } from './CostCharts'

// recharts needs a sized container in jsdom; mock ResponsiveContainer to a fixed box.
vi.mock('recharts', async (orig) => {
  const actual = await orig<typeof import('recharts')>()
  return { ...actual, ResponsiveContainer: ({ children }: { children: React.ReactNode }) =>
    <div style={{ width: 400, height: 200 }}>{children}</div> }
})

describe('CostCharts', () => {
  it('renders without crashing with empty data', () => {
    const { container } = render(<CostCharts perModel={{}} recent={[]} />)
    expect(container).toBeTruthy()
  })
})
```

- [ ] **Step 2: Run test to verify it fails**

Run: `npx vitest run src/components/cost/CostCharts.test.tsx`
Expected: FAIL (module not found).

- [ ] **Step 3: Implement `CostCharts.tsx`**

```tsx
// frontend/src/components/cost/CostCharts.tsx
'use client'

import {
  ResponsiveContainer, AreaChart, Area, BarChart, Bar, XAxis, YAxis,
  CartesianGrid, Tooltip as RTooltip,
} from 'recharts'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import type { CostBucket, CostEntry } from '@/lib/api/metrics'
import { perModelSpend, costOverTime } from '@/lib/utils/cost-charts'

const AXIS = { fontSize: 10 }

export function CostCharts({
  perModel, recent,
}: { perModel: Record<string, CostBucket>; recent: CostEntry[] }) {
  const spend = perModelSpend(perModel)
  const series = costOverTime(recent)

  return (
    <div className="grid gap-4 md:grid-cols-2">
      <Card>
        <CardHeader><CardTitle className="text-sm">Cost over time</CardTitle></CardHeader>
        <CardContent className="h-56">
          <ResponsiveContainer width="100%" height="100%">
            <AreaChart data={series}>
              <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
              <XAxis dataKey="label" tick={AXIS} />
              <YAxis tick={AXIS} />
              <RTooltip />
              <Area type="monotone" dataKey="usd" stroke="var(--chart-1)" fill="var(--chart-1)" fillOpacity={0.2} />
            </AreaChart>
          </ResponsiveContainer>
        </CardContent>
      </Card>
      <Card>
        <CardHeader><CardTitle className="text-sm">Spend by model</CardTitle></CardHeader>
        <CardContent className="h-56">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={spend}>
              <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
              <XAxis dataKey="name" tick={AXIS} />
              <YAxis tick={AXIS} />
              <RTooltip />
              <Bar dataKey="usd" fill="var(--chart-2)" radius={[4, 4, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </CardContent>
      </Card>
    </div>
  )
}

export default CostCharts
```

- [ ] **Step 4: Run test to verify it passes**

Run: `npx vitest run src/components/cost/CostCharts.test.tsx`
Expected: PASS

- [ ] **Step 5: Add the Charts tab to the cost page**

In `frontend/src/app/(dashboard)/cost/page.tsx`: import `CostCharts` and lazily render it in a new tab. Read the file's `<Tabs>`/`<TabsList>`/`<TabsContent>` usage first. Add a `<TabsTrigger value="charts">Charts</TabsTrigger>` and a matching:

```tsx
        <TabsContent value="charts">
          <CostCharts perModel={summary?.per_model ?? {}} recent={recent ?? []} />
        </TabsContent>
```

(Use the existing `useCostSummary()` / `useRecentCostCalls()` data already in the page — match the variable names the page uses for the summary + recent entries.)

- [ ] **Step 6: Typecheck + commit**

Run: `npx tsc --noEmit` (expect no new errors).

```bash
git add frontend/src/components/cost/CostCharts.tsx frontend/src/components/cost/CostCharts.test.tsx "frontend/src/app/(dashboard)/cost/page.tsx"
git commit -m "feat(cost): recharts Charts tab (cost-over-time + spend-by-model)"
```

---

### Task 6.4: Skeleton primitive + apply to cost cards

**Files:**
- Create: `frontend/src/components/ui/skeleton.tsx`
- Modify: `frontend/src/app/(dashboard)/cost/page.tsx` (loading state uses Skeleton)
- Test: `frontend/src/components/ui/skeleton.test.tsx`

- [ ] **Step 1: Write the failing test**

```tsx
// frontend/src/components/ui/skeleton.test.tsx
import { describe, it, expect } from 'vitest'
import { render } from '@testing-library/react'
import { Skeleton } from './skeleton'

describe('Skeleton', () => {
  it('renders an animated placeholder with merged className', () => {
    const { container } = render(<Skeleton className="h-8 w-20" />)
    const el = container.firstChild as HTMLElement
    expect(el.className).toContain('animate-pulse')
    expect(el.className).toContain('h-8')
  })
})
```

- [ ] **Step 2: Run test to verify it fails**

Run: `npx vitest run src/components/ui/skeleton.test.tsx`
Expected: FAIL (module not found).

- [ ] **Step 3: Implement `skeleton.tsx`**

```tsx
// frontend/src/components/ui/skeleton.tsx
import { cn } from '@/lib/utils'

export function Skeleton({ className, ...props }: React.HTMLAttributes<HTMLDivElement>) {
  return <div className={cn('animate-pulse rounded-md bg-muted', className)} {...props} />
}

export default Skeleton
```

- [ ] **Step 4: Run test to verify it passes**

Run: `npx vitest run src/components/ui/skeleton.test.tsx`
Expected: PASS

- [ ] **Step 5: Use Skeleton on the cost page loading state**

In `frontend/src/app/(dashboard)/cost/page.tsx`, where the summary cards render while `isLoading`, show four `<Skeleton className="h-24 w-full" />` placeholders instead of a bare spinner. (Read the page's loading branch; match its grid layout for the four cards.)

- [ ] **Step 6: Typecheck + commit**

Run: `npx tsc --noEmit`.

```bash
git add frontend/src/components/ui/skeleton.tsx frontend/src/components/ui/skeleton.test.tsx "frontend/src/app/(dashboard)/cost/page.tsx"
git commit -m "feat(ui): Skeleton primitive + cost dashboard skeletons"
```

---

## PHASE 7 — i18n + final verification

### Task 7.1: Add i18n strings for new UI

**Files:**
- Modify: `frontend/src/lib/locales/en-US/index.ts`
- (Components already render English labels inline in MessageSignals/TraceDialog for v1; this task moves the user-facing labels to i18n keys and updates the components to use `t(...)`.)

- [ ] **Step 1: Add a `ragSignals` namespace to en-US**

In `frontend/src/lib/locales/en-US/index.ts`, add a top-level key (alongside `chat`, `common`, etc.):

```typescript
  ragSignals: {
    fastPath: "Fast path",
    semantic: "Semantic",
    structured: "Structured",
    instantCached: "Instant · cached",
    verified: "Verified",
    selfCorrected: "Self-corrected",
    corrective: "corrective",
    summaryLevel: "Summary · L{level}",
    fastPathTitle: "Reasoning path used for this answer",
    cacheTitle: "Served from semantic cache — near-instant",
    verifiedTitle: "Answer passed the CRAG grounding check",
    correctedTitle: "Answer was re-retrieved + revised by CRAG correction",
  },
```

- [ ] **Step 2: Use `t()` in MessageSignals + RaptorBadge**

In `MessageSignals.tsx` and `CitationHoverCard.tsx` (RaptorBadge), import `useTranslation` (`import { useTranslation } from '@/lib/hooks/use-translation'`) and replace the literal English labels/titles with `t('ragSignals.*')`. For `summaryLevel`, do `t('ragSignals.summaryLevel').replace('{level}', String(nodeLevel))`.

- [ ] **Step 3: Run the affected tests**

The existing component tests assert on the rendered English text (e.g. `/fast path/i`, `/instant/i`, `/L2/`). Since en-US is the default locale and tests render without a language override (falling back to en-US), the visible text is unchanged → tests still pass. Run:

`npx vitest run src/components/source/MessageSignals.test.tsx src/components/source/citation-hover-helpers.test.tsx`
Expected: PASS. If any test fails because i18n isn't initialized in jsdom, the component's `t()` returns the key — in that case update those two tests to assert on the key text, OR ensure the test imports `@/lib/i18n` for init (check how `ChatPanel.starters.test.tsx` handles i18n and mirror it).

- [ ] **Step 4: Commit**

```bash
git add frontend/src/lib/locales/en-US/index.ts frontend/src/components/source/MessageSignals.tsx frontend/src/components/source/CitationHoverCard.tsx
git commit -m "i18n(chat): RAG-signal labels via i18next (en-US source)"
```

---

### Task 7.2: Full verification — tests, types, build

**Files:** none (verification only)

- [ ] **Step 1: Run the full frontend test suite**

Run (from `frontend/`): `npm test`
Expected: PASS (all existing + new tests). Note any pre-existing failures unrelated to this work and report them separately.

- [ ] **Step 2: Typecheck**

Run: `npx tsc --noEmit`
Expected: no errors.

- [ ] **Step 3: Production build**

Run: `npm run build`
Expected: build succeeds (Next.js compiles; recharts + new components included). If the build flags an unused import or a hook-rule issue, fix it minimally and re-run.

- [ ] **Step 4: Run the backend tests touched by Phase 1**

Run (repo root): `uv run pytest tests/agent/ -q`
Expected: PASS.

- [ ] **Step 5: Commit any fixups**

```bash
git add -A
git commit -m "chore(ui): fixups from full typecheck + build" || echo "nothing to commit"
```

---

## Self-Review

**Spec coverage:** WS-Phase1 (backend signals) → Tasks 1.1–1.2. Frontend types → 2.1. Chat chips → 3.1–3.2. Trace upgrade (critique/multi-hop/corrective/diagnostics/fast-path) → 4.1–4.2. Citation hover (RAPTOR + context) → 5.1. Cost charts → 6.1–6.3. Skeletons → 6.4. Animation → `animate-in fade-in` on chips (3.1) + `animate-pulse` skeleton (6.4) + tailwindcss-animate plugin (6.1). i18n → 7.1. Testing/build → 7.2. All spec sections mapped. (Spec's "VN-facing locale" corrected: no `vi-VN` exists; en-US is the source, others fall back.)

**Placeholder scan:** every code step shows full code; commands have expected output. The two `cost/page.tsx` wiring steps (6.3 Step 5, 6.4 Step 5) say "match the page's existing variable names" rather than hardcoding them because the exact local names depend on the page's current `useCostSummary()` destructuring — the implementer reads the file (a few lines) and wires accordingly; the surrounding code (TabsContent, Skeleton grid) is fully specified.

**Type consistency:** `MessageSignals` / `deriveCritiqueState` (3.1) consumed in 3.2. `groupMultiHop` / `TaggedTrace` (4.1) used in 4.2. `RaptorBadge` / `CitationContextLine` (5.1). `perModelSpend` / `costOverTime` / `ModelSpend` / `TimePoint` (6.2) consumed in 6.3. `Skeleton` (6.4). Citation/message fields (2.1) consumed throughout. Backend `_message_signals` (1.2) and citation fields (1.1) match the frontend types (2.1).
