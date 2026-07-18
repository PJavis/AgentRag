# Vision Slice A — answer-time multimodal routing (2026-07-19)

## Why

Multimodal/vision (roadmap P7) is decomposed into three slices; this is **Slice A**, the
smallest and cheapest, with no re-ingest. Today the answer-time multimodal path exists but is
dead: when packed context has image segments, `service._answer` calls
`llm_gateway.json_response_multimodal(task="answer")`, which routes to the **text** `answer`
model (`deepseek-v4-flash`, no vision) — the image call fails and silently falls back to
text-only (`service.py:975`). So even once images are captioned+retrieved (Slice B/C), the
answer node can never actually *look at* the pixels. Slice A fixes the routing so image-bearing
turns reach a vision-capable model (Gemini), while text turns keep using DeepSeek.

## Goal / success criteria

An image-bearing answer turn is answered by a **vision** model reading the actual image, not the
text-fallback. Done when:

1. `json_response_multimodal(...)` routes to a configured vision model (`VISION_ANSWER_MODEL`,
   default `gemini-2.5-flash`), NOT the text `answer` task model.
2. A turn whose `packed_context` contains an `image` segment with a reachable `image_url`
   produces an answer from the vision model (verified: not the text-only fallback path).
3. Text-only turns are unchanged — still answered by the `answer` task model (`deepseek-v4-flash`).
4. When `VISION_ANSWER_MODEL` is empty, the multimodal path is skipped and the turn answers
   text-only (no failing call, no regression) — so this is safe with vision disabled.

## Non-goals (later slices)

- **Slice B:** enabling ingest-time image captioning (`VISION_PROVIDER=ollama`) + re-ingesting a
  small image-heavy subset.
- **Slice C:** full-corpus re-ingest + e2e eval on the image-dependent misses.
- Answer-time image PHI: only the single retrieved image per query reaches Gemini; bulk
  captioning stays local (Slice B). No image reaches Gemini here unless a turn already has an
  image segment in context (none do until Slice B/C ingests images) — so Slice A is inert on the
  current text-only corpus and provably safe to merge ahead of the re-ingest.

## What exists (no change needed)

- `service._answer` (`src/agentrag/agent/service.py:944-980`): collects `image_urls` from
  `packed_context` segments where `segment_type == "image"` (cap 4), and if any, calls
  `json_response_multimodal(system_prompt, user_prompt, image_urls, task="answer")` with a
  try/except text fallback.
- `LLMGateway.json_response_multimodal` (`services/llm_gateway.py:82-98`) → `_resolve_client(task)`
  → task→model via `LLM_TASK_MODEL_MAP`; provider derived from model-name prefix in
  `agent/llm.py` (`gemini-*` → Gemini + `GEMINI_API_KEY`).
- `_resolve_client` caches per-model `AgentLLM` clients in `self._routed_clients`.

## Design

### Config (new)
`src/agentrag/config.py`:
```python
#: Vision-capable model for ANSWER-TIME multimodal grounding (reads the retrieved image's
#: pixels). Independent of the text `answer` model. gemini-* → Gemini via GEMINI_API_KEY.
#: Empty string disables answer-time vision (turns answer text-only). Only image-bearing
#: turns use it, so it is inert until images are ingested (Slice B/C).
VISION_ANSWER_MODEL: str = "gemini-2.5-flash"
```

### Gateway routing change
`LLMGateway.json_response_multimodal` resolves its client from `VISION_ANSWER_MODEL` directly,
NOT from the `task` argument's text model:
```python
async def json_response_multimodal(self, system_prompt, user_text, image_urls, task="general"):
    model = settings.VISION_ANSWER_MODEL
    if not model:
        raise VisionDisabledError("VISION_ANSWER_MODEL empty")   # caller falls back to text
    client = self._client_for_model(model)   # cached AgentLLM(model_override=model)
    ...
```
Add a small helper `_client_for_model(model)` (extract the caching stanza already inlined in
`_resolve_client` steps 1/2 so both use it — DRY) and a `VisionDisabledError`. `task` stays only
for cost attribution.

### service._answer
The existing `try: json_response_multimodal(...) except Exception: <text fallback>` already
handles the empty-model case (raises → caught → text fallback). Change: pass `task="answer_vision"`
(cost attribution only) instead of `task="answer"`. No control-flow change.

### Data flow
```
image-bearing turn → service._answer collects image_urls
  → json_response_multimodal → VISION_ANSWER_MODEL (gemini-2.5-flash) → reads pixels → answer
text-only turn → json_response (task="answer") → deepseek-v4-flash    (unchanged)
VISION_ANSWER_MODEL="" → multimodal raises VisionDisabledError → text fallback (unchanged)
```

## Testing

- **Unit (`tests/agent/` or `tests/services/`):**
  - `json_response_multimodal` resolves a client whose model is `VISION_ANSWER_MODEL`
    (monkeypatch to `gemini-2.5-flash`, assert the routed client's model / that a Gemini-prefixed
    `AgentLLM` is built) — NOT the `answer` task model.
  - `VISION_ANSWER_MODEL=""` → `json_response_multimodal` raises `VisionDisabledError` (caller's
    fallback covered by the existing text path).
  - text-only path (`json_response`, task="answer") still resolves the `answer` model — unchanged.
- **Live verify (controller):** construct a `packed_context` with one `image` segment pointing at
  a real reachable image URL (a hand-placed test image served via the `/images` mount or a public
  URL), drive `_answer`, and confirm the response reflects the image content via Gemini (log shows
  the multimodal branch taken, not the `except → text fallback`). Confirm a text-only turn still
  routes to deepseek.

## Risks
- **Wrong-model silent fallback:** if routing regresses to the text model, images fail → text
  fallback, masking the bug. Mitigation: the live verify asserts the multimodal branch is taken
  (not the except path), e.g. via a log/among the returned metadata.
- **Cost/PHI:** none on the current corpus — no turn has image segments until Slice B/C ingests
  them, so `VISION_ANSWER_MODEL` is never invoked yet. Safe to merge ahead of re-ingest.
