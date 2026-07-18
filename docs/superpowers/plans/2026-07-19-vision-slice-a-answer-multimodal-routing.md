# Vision Slice A — answer-time multimodal routing — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Route the answer-time multimodal call to a dedicated vision model (`VISION_ANSWER_MODEL`, default `gemini-2.5-flash`) instead of the text `answer` model (deepseek), so image-bearing turns are answered from the image's pixels while text turns are unchanged.

**Architecture:** Add one config setting and change `LLMGateway.json_response_multimodal` to resolve its client from `VISION_ANSWER_MODEL` (not the `task`→text-model map). Empty model raises `VisionDisabledError`, which the existing `service._answer` try/except turns into the text fallback. No re-ingest; inert on today's text-only corpus (no turn has image segments until Slices B/C).

**Tech Stack:** Python, pydantic settings, `AgentLLM` (model-name-prefix → provider), pytest.

## Global Constraints

- `VISION_ANSWER_MODEL: str = "gemini-2.5-flash"` — vision model for answer-time grounding; `gemini-*` → Gemini via `GEMINI_API_KEY`. Empty string disables answer-time vision (text fallback).
- The text `answer` model (`deepseek-v4-flash`) MUST remain unchanged for text-only turns.
- Multimodal model selection comes from `VISION_ANSWER_MODEL`, NOT from `LLM_TASK_MODEL_MAP` (which defaults to `"{}"` and is .env-driven — a map key would silently break the feature).

---

### Task 1: Route answer-time multimodal to `VISION_ANSWER_MODEL`

**Files:**
- Modify: `src/agentrag/config.py` (add `VISION_ANSWER_MODEL`, near the `VISION_*` block ~line 209)
- Modify: `src/agentrag/services/llm_gateway.py` (`VisionDisabledError`, `_client_for_model`, `json_response_multimodal`)
- Modify: `src/agentrag/agent/service.py:973` (`task="answer"` → `task="answer_vision"` in the multimodal call — cost attribution only)
- Test: `tests/services/test_llm_gateway_multimodal.py` (new)

**Interfaces:**
- Consumes: `settings.VISION_ANSWER_MODEL`; `AgentLLM(model_override=...)` (sets `.model`); `LLMGateway._routed_clients` (dict model→AgentLLM), `LLMGateway._default_client`.
- Produces: `LLMGateway._client_for_model(model: str) -> AgentLLM` (cached); `VisionDisabledError(Exception)`; `json_response_multimodal` now routes by `VISION_ANSWER_MODEL`.

- [ ] **Step 1: Write the failing tests**

Create `tests/services/test_llm_gateway_multimodal.py`:
```python
import pytest
from src.agentrag.services.llm_gateway import LLMGateway, VisionDisabledError
from src.agentrag.config import settings


def test_multimodal_routes_to_vision_answer_model(monkeypatch):
    """Answer-time multimodal must resolve the VISION_ANSWER_MODEL client, not the
    text `answer` model."""
    monkeypatch.setattr(settings, "VISION_ANSWER_MODEL", "gemini-2.5-flash")
    gw = LLMGateway()
    client = gw._client_for_model(settings.VISION_ANSWER_MODEL)
    assert client.model == "gemini-2.5-flash"  # gemini-prefixed → Gemini provider
    # and it is cached / reused
    assert gw._client_for_model("gemini-2.5-flash") is client


def test_multimodal_disabled_when_model_empty(monkeypatch):
    """Empty VISION_ANSWER_MODEL → json_response_multimodal raises VisionDisabledError
    (caller falls back to text-only)."""
    monkeypatch.setattr(settings, "VISION_ANSWER_MODEL", "")
    gw = LLMGateway()
    with pytest.raises(VisionDisabledError):
        import asyncio
        asyncio.run(gw.json_response_multimodal("sys", "user", ["http://x/img.png"], task="answer_vision"))


def test_text_answer_model_unchanged(monkeypatch):
    """Text path still resolves the `answer` task model (not the vision model)."""
    monkeypatch.setattr(settings, "LLM_ROUTING_ENABLED", True)
    monkeypatch.setattr(settings, "LLM_TASK_MODEL_MAP", '{"answer":"deepseek-v4-flash"}')
    gw = LLMGateway()
    client = gw._resolve_client("answer", content="hi")
    assert client.model == "deepseek-v4-flash"
```

- [ ] **Step 2: Run — expect FAIL**

Run: `uv run pytest tests/services/test_llm_gateway_multimodal.py -v`
Expected: FAIL — `ImportError: cannot import name 'VisionDisabledError'` (and `_client_for_model` missing).

- [ ] **Step 3: Add the config setting**

In `src/agentrag/config.py`, after `VISION_INGEST_MODE` (~line 209):
```python
    #: Vision-capable model for ANSWER-TIME multimodal grounding (reads the retrieved
    #: image's pixels). Independent of the text `answer` model. gemini-* → Gemini via
    #: GEMINI_API_KEY. Empty string disables answer-time vision (turns answer text-only).
    #: Only image-bearing turns use it — inert until images are ingested (vision slices B/C).
    VISION_ANSWER_MODEL: str = "gemini-2.5-flash"
```

- [ ] **Step 4: Implement gateway routing**

In `src/agentrag/services/llm_gateway.py`:

Add the exception near the top (after imports, before the class or module-level):
```python
class VisionDisabledError(RuntimeError):
    """Raised when answer-time multimodal is requested but VISION_ANSWER_MODEL is empty."""
```

Add the helper method on `LLMGateway` (extract the caching stanza so `_resolve_client` can reuse it too — DRY):
```python
    def _client_for_model(self, model: str) -> AgentLLM:
        """Cached AgentLLM for an explicit model name (provider derived from prefix)."""
        if model not in self._routed_clients:
            self._routed_clients[model] = AgentLLM(model_override=model)
        return self._routed_clients[model]
```
Refactor `_resolve_client` steps 1 & 2 to call `self._client_for_model(large_model)` /
`self._client_for_model(override_model)` instead of the inlined cache blocks (behavior identical).

Replace `json_response_multimodal` body's client resolution:
```python
    async def json_response_multimodal(self, system_prompt, user_text, image_urls, task="general"):
        model = settings.VISION_ANSWER_MODEL
        if not model:
            raise VisionDisabledError("VISION_ANSWER_MODEL is empty; answer-time vision disabled")
        client = self._client_for_model(model)
        started = time.perf_counter()
        payload = await client.json_response_multimodal(
            system_prompt, user_text, image_urls, task=task
        )
        latency_ms = (time.perf_counter() - started) * 1000
        return payload, latency_ms
```

- [ ] **Step 5: Update the call site (cost-attribution label)**

In `src/agentrag/agent/service.py`, the multimodal call (~line 973): change `task="answer"` to
`task="answer_vision"`. Leave the `except Exception:` text fallback exactly as-is (it now also
catches `VisionDisabledError` → text-only).

- [ ] **Step 6: Run — expect PASS**

Run: `uv run pytest tests/services/test_llm_gateway_multimodal.py -v`
Expected: 3 PASS.

- [ ] **Step 7: Guard against regressions in related suites**

Run: `uv run pytest tests/services/ tests/agent/ -q`
Expected: all PASS (no existing test asserted multimodal→answer-model routing).

- [ ] **Step 8: Commit**

```bash
git add src/agentrag/config.py src/agentrag/services/llm_gateway.py src/agentrag/agent/service.py tests/services/test_llm_gateway_multimodal.py
git commit -m "feat(vision): route answer-time multimodal to VISION_ANSWER_MODEL (gemini)"
```

---

### Task 2: Live verification — image turn answered by Gemini, text turn unchanged

Controller-owned runtime check (the routing must actually take the multimodal branch, not
silently fall back). No commit unless it surfaces a fix.

**Files:** none (verification only; may write `/tmp` scratch).

**Interfaces:** Consumes `GraphAgentService` / `service._answer`; `settings.VISION_ANSWER_MODEL`; a reachable test image URL.

- [ ] **Step 1: Confirm Gemini vision reachable**

Run:
```bash
source .env
curl -s -m20 "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions" \
  -H "Authorization: Bearer $GEMINI_API_KEY" -H 'Content-Type: application/json' \
  -d '{"model":"gemini-2.5-flash","messages":[{"role":"user","content":[{"type":"text","text":"say OK"}]}],"max_tokens":5}' -o /dev/null -w "http=%{http_code}\n"
```
Expected: `http=200`.

- [ ] **Step 2: Drive `_answer` with a hand-placed image segment**

Write `/tmp/vision_verify.py` — build a `packed_context` with one `image` segment pointing at a
public test image (a labelled diagram), call the answer path with `VISION_ANSWER_MODEL=gemini-2.5-flash`,
and assert the multimodal branch was taken (answer reflects the image; no `multimodal call failed`
in logs):
```python
import asyncio, logging, sys
sys.path.insert(0, ".")
logging.basicConfig(level=logging.INFO)
# Call the gateway directly to prove routing end-to-end to Gemini vision:
from src.agentrag.services.llm_gateway import LLMGateway
async def main():
    gw = LLMGateway()
    payload, ms = await gw.json_response_multimodal(
        system_prompt="Describe the image in one word as JSON {\"answer\": <word>}.",
        user_text="What is shown?",
        image_urls=["https://upload.wikimedia.org/wikipedia/commons/thumb/8/85/Smiley.svg/120px-Smiley.svg.png"],
        task="answer_vision",
    )
    print("VISION PAYLOAD:", payload, "ms=", round(ms))
asyncio.run(main())
```
Run: `VISION_ANSWER_MODEL=gemini-2.5-flash PYTHONPATH=. uv run python /tmp/vision_verify.py`
Expected: a JSON payload whose answer describes the image (e.g. "smiley"/"face") — proving the
call reached Gemini vision and read the pixels, not a text-model failure.

- [ ] **Step 3: Confirm text-only turn still uses deepseek**

Run:
```bash
PYTHONPATH=. uv run python -c "
from src.agentrag.services.llm_gateway import LLMGateway
gw = LLMGateway()
print('text answer model:', gw._resolve_client('answer', content='hi').model)
"
```
Expected: `deepseek-v4-flash` (text path unchanged).

- [ ] **Step 4: Record the result**

Note the vision payload + the text-model confirmation in the ledger / PR description as the
Slice A verification evidence. No code change.
