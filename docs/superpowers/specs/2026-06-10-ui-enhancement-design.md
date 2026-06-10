# UI Enhancement — Surface RAG Signals + Dashboard Polish

**Date**: 2026-06-10
**Status**: Approved (verbal, brainstorming session)
**Author**: dungnq + Claude
**Branch**: `feat/ragas-langfuse-reranker`
**Scope**: One comprehensive spec, 4 phases. Frontend (`frontend/src`) + a thin backend signal-propagation shim.

## Context

The backend just shipped 5 RAG enhancements (Contextual Retrieval, RAPTOR, CRAG critique, adaptive fast-path, semantic cache — see `2026-06-10-rag-enhancement-design.md`), all behind default-OFF flags. The Next.js frontend (`frontend/`, Next 16 / React 19 / Radix + Tailwind OKLCH / react-query / react-markdown) has a solid chat experience but **none of the new RAG signals are surfaced**, and several aren't even propagated into the chat response:

| Backend signal | Current UI state |
|---|---|
| `reasoning_path="fast"` (adaptive) | raw string badge, no styling/meaning (`ChatPanel.tsx:348-356`) |
| `semantic_cache_hit` | not propagated to message, not typed, not shown |
| CRAG critique / `corrective:true` trace | `timings_ms.critique` rendered but unexplained; corrective step not flagged (`TraceDialog.tsx`) |
| multi-hop `tool_trace[i].multihop` | silently ignored (open-dict, never rendered) |
| RAPTOR `node_level>=1` / `segment_type="raptor_summary"` | Citation type too narrow (`api.ts:166`); summary vs leaf indistinguishable |
| Contextual Retrieval `context_text` | not typed on Citation; hover shows only `excerpt` (`CitationHoverCard.tsx`) |

General polish gaps: cost dashboard is tables-only (no chart lib), no skeleton loaders, minimal animation.

## Goals & Non-goals

**Goals:**
- Make every new RAG signal visible + legible in chat (chips + upgraded trace + richer citation hovers).
- Propagate the missing signals from backend → chat response.
- Lift dashboard polish: cost charts (recharts), skeleton loaders, light CSS animation.
- Graceful degradation: zero visual change when backend flags are OFF (chips guard on signal presence).

**Non-goals:**
- No separate "Why this answer" slide-over (chips + trace cover it).
- No data-grid library, no framer-motion, no mobile redesign beyond chart responsiveness.
- No change to backend RAG logic — only propagate signals already computed.

## Decisions (brainstorming output)

| ID | Decision | Rationale |
|---|---|---|
| D1 | Scope = both (RAG signals + polish), phased | User pick |
| D2 | Charts = Recharts | User pick — declarative, themeable to OKLCH, React 19 ok |
| D3 | Animation = Tailwind + `tailwindcss-animate`, no framer-motion | User pick — light, consistent with current code |
| D4 | RAG chips guard on signal presence | Graceful when flags OFF |
| D5 | Thin backend shim to propagate signals | Some signals not yet in chat response |
| D6 | i18n all new strings | Match existing i18next pattern |

## Architecture

New deps: `recharts`, `tailwindcss-animate`. New components are small + single-purpose. Recharts themed via existing `--chart-1..5` OKLCH CSS vars (`globals.css`). Tests: jest + jest-dom (`src/test/setup.ts`).

### Data flow (new signals)
```
retrieval _normalize_hits (already carries context_text, node_level — Task 0.2)
  → packed_context
  → _build_packed_citations  [SHIM: copy node_level, context_text, segment_type]
  → chat response.citations
graph_service return  [SHIM: add semantic_cache_hit, retrieval_mode, domain_route]
  → frontend message
  → MessageSignals chips + CitationHoverCard + TraceDialog
```

---

## PHASE 1 — Chat RAG signal chips (+ backend plumbing)

### 1a. Backend signal propagation
- `agent/service.py` `_build_packed_citations` (~`:682-719`): copy `node_level`, `context_text`, `segment_type` from each packed_context item onto the citation dict it builds (fields already present on retrieval hits via `_normalize_hits`).
- Chat response assembly (`agent/graph_service.py` `GraphAgentService.chat` return ~`:474-487`, and the streaming `done` event): add `semantic_cache_hit` (read from the bootstrap tool_output's retrieval payload, which carries `semantic_cache_hit` when WS5 hit), `retrieval_mode`, and `domain_route` (from `domain_filter`/router when present). Default-absent when not available.
- Backend test: `tests/adapter/test_citation_shape.py` (extend) — assert a citation built from a packed item with `node_level`/`context_text` carries those fields.

### 1b. Frontend types (`frontend/src/lib/types/api.ts`)
- `Citation`: add `node_level?: number | null`, `context_text?: string | null`; widen `segment_type` to `'text' | 'image' | 'table' | 'raptor_summary'`.
- `NotebookChatMessage` + `SourceChatMessage`: add `semantic_cache_hit?: boolean | null`, `domain_route?: string | null`, `retrieval_mode?: string | null`.

### 1c. `MessageSignals.tsx` (new, `frontend/src/components/source/`)
A chip row rendered under each AI answer, replacing the raw `reasoning_path` badge block (`ChatPanel.tsx:348-356`). Each chip guards on presence:
- **Reasoning path**: icon+color per value — `fast` → `⚡ Fast path`, `semantic` → `Semantic`, `structured` → `Structured`, `summary`/`chitchat` as today. Uses Badge variants + lucide icons (Zap, Brain, Database).
- **Cache**: `semantic_cache_hit` → `⚡ Instant · cached` (emphasizes the speed win).
- **Verification**: derive from `timings_ms.critique` presence + a `corrective` flag — `🧠 Verified` (critique ran, grounded) or `↻ Self-corrected` (corrective fired). Helper `deriveCritiqueState(message)`.
- **Domain**: `domain_route` → `🏥 {domain}`.
- Tooltip on each chip (existing `ui/tooltip`) explaining the signal. i18n strings.

### 1d. Tests
`frontend/src/components/source/MessageSignals.test.tsx` — renders the right chips for: fast-path, cache-hit, critique-verified, corrected, none (graceful empty).

---

## PHASE 2 — Trace dialog upgrade (`frontend/src/components/source/TraceDialog.tsx`)

- **Critique stage**: extend `STAGE_ORDER` (`:34`) to include `critique`; render its node with a ✓ (passed) or ↻ (corrected) marker driven by trace.
- **Multi-hop chains**: detect `tool_trace[i].multihop === true`; render those entries as an ordered chain ("Hop 1 → Hop 2") with the carried-forward context snippet (from `tool_input.query`). Helper `groupMultiHop(toolTrace)`.
- **Corrective entries**: `corrective === true` → distinct row style + `↻ Corrective re-retrieve` label (RotateCcw icon).
- **Per-tool retrieval diagnostics**: in `ToolTraceList`, surface from `tool_output` (when present): `semantic_cache_hit`, `mode`, `domain_route`, number of hits.
- **Fast-path explanation**: when `reasoning_path==="fast"`, show a compact info block (mirror the existing summary/chitchat special-case `:178-199`): "Answered via fast path — single retrieve + answer, no agent loop."
- Tests: `TraceDialog.test.tsx` — multi-hop grouping renders chain; corrective row labeled; critique stage shows; fast-path info block.

---

## PHASE 3 — Citation hover (`frontend/src/components/source/CitationHoverCard.tsx`)

- **RAPTOR**: when `citation.node_level >= 1`, show a `Σ Summary · L{node_level}` badge + tree/layers icon in the header; subtle accent so summary citations read differently from leaf chunks.
- **Contextual Retrieval**: when `citation.context_text` present, render it as a muted one-line "context" above the `excerpt` (e.g. *"From the cardiology chapter on MI"*).
- Tests: `CitationHoverCard.test.tsx` — RAPTOR badge appears for node_level≥1; context_text line renders when present; neither shows when absent.

---

## PHASE 4 — Dashboard charts + skeletons + animation

### 4a. Cost dashboard (`frontend/src/app/(dashboard)/cost/page.tsx`)
Add a "Charts" tab (alongside By Task / By Model / Recent) using recharts:
- **Cost over time** — AreaChart from `useRecentCostCalls` entries (bucket by minute/hour).
- **Per-model spend** — BarChart from `CostSummary.per_model`.
- **Latency p50/p95** — ComposedChart (bars) from `per_model`/`per_task` buckets.
- **Token usage** — stacked AreaChart (in/out tokens over recent calls).
- Pure transform helpers in `frontend/src/lib/utils/cost-charts.ts` (bucketing, series shaping) — **unit-tested** independent of rendering.
- Charts themed with `--chart-1..5` CSS vars; responsive via recharts `ResponsiveContainer`.

### 4b. Skeleton loaders
- New `frontend/src/components/ui/skeleton.tsx` (shadcn-style: `animate-pulse rounded-md bg-muted`).
- Apply on initial loads: cost cards, activity feed, notebook columns, chat session list — replace bare `LoadingSpinner` where a content-shaped skeleton reads better.

### 4c. Animation (`tailwindcss-animate`)
- Add the plugin to `tailwind.config.ts`.
- Chip mount: `animate-in fade-in` on `MessageSignals` chips.
- Skeleton shimmer (pulse) on `Skeleton`.
- CSS number ticker on cost summary cards (count-up via CSS/transition or a tiny `useCountUp` hook — no dep).
- Smoother trace-node selection transition.

### 4d. Tests
`frontend/src/lib/utils/cost-charts.test.ts` — bucketing + series transforms produce correct shapes for empty / single / many entries.

---

## Cross-cutting

- **i18n**: every new user-facing string added to `frontend/src/lib/locales/en-US/index.ts` (+ the VN-facing locale used in chat) under a new `ragSignals` / `trace` / `cost` namespace; components use `t(...)`.
- **Theming/dark mode**: all new UI uses existing tokens; recharts reads CSS vars; verify dark mode for charts + chips.
- **Graceful degradation**: every RAG chip/section guards on signal presence; with backend flags OFF the chat looks exactly as today.
- **Testing**: run `npm test` (jest) in `frontend/`; new component + util tests must pass. Backend shim covered by one pytest.

## File structure

**New files:**
- `frontend/src/components/source/MessageSignals.tsx` (+ `.test.tsx`)
- `frontend/src/components/ui/skeleton.tsx`
- `frontend/src/lib/utils/cost-charts.ts` (+ `.test.ts`)
- chart subcomponents under `frontend/src/components/cost/` (e.g. `CostCharts.tsx`)

**Modified:**
- `frontend/src/lib/types/api.ts` — Citation + message types.
- `frontend/src/components/source/ChatPanel.tsx` — swap reasoning-path badge → `MessageSignals`.
- `frontend/src/components/source/TraceDialog.tsx` — critique stage, multi-hop, corrective, diagnostics, fast-path.
- `frontend/src/components/source/CitationHoverCard.tsx` — RAPTOR badge + context line.
- `frontend/src/app/(dashboard)/cost/page.tsx` — Charts tab.
- `frontend/tailwind.config.ts` — `tailwindcss-animate`.
- `frontend/package.json` — recharts, tailwindcss-animate.
- `frontend/src/lib/locales/*` — new strings.
- `src/agentrag/agent/service.py`, `src/agentrag/agent/graph_service.py` — signal propagation shim.

## Risks & mitigations

| Risk | Mitigation |
|---|---|
| Backend signals not actually present in response | Phase 1a shim + a pytest verifying citation/message fields |
| Recharts bundle size | Single chart lib, lazy-load the Charts tab; tables remain default |
| Chips clutter the message | Compact, presence-guarded, tooltip-explained; max ~4 chips |
| Dark-mode chart colors | Use `--chart-*` CSS vars (already dark-aware) |
| i18n drift (hardcoded strings) | All strings via `t()`; review in spec self-check |

## Open questions (resolve during planning)

- `semantic_cache_hit` source: it's on the retrieval payload inside the bootstrap tool_output — confirm exact key path when wiring 1a (lean: read `tool_output.semantic_cache_hit` from the first hybrid_kg trace entry).
- Whether to show the verification chip only when CRAG ran vs always — lean: only when `timings_ms.critique` present (i.e. CRAG enabled), else hide.
- Cost "over time" bucket granularity — lean: auto (minute if <2h span, else hour).
