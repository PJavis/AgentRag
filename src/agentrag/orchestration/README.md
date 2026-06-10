# orchestration — SLM domain router (query → medical body-systems / specialties)

## Mục đích / Purpose
This module holds the **routers/planners that decide WHERE to send a query** before
retrieval runs. Today it contains a single concern: `DomainRouter`, an SLM-driven
classifier that maps a free-text Vietnamese medical query to a set of body
**systems** (e.g. `tim_mach`, `ho_hap`) and clinical **specialties** (e.g. `noi`,
`cap_cuu`). The Reasoning Plane calls `DomainRouter.classify(query)` and forwards the
picks as `system_override` / `specialty_override` filters into retrieval, so search
can be federated/narrowed to the right medical domain (S5 federated retrieval).

## Plane
**Reasoning Plane.** It owns a prompt (`SYSTEM_PROMPT`) and makes an LLM decision
about routing — it never does the IO of filtering/searching itself. It calls the
Execution Plane (`LLMGateway`) to run the classification, then hands the decision
back to the caller, who translates it into retrieval filters.

## Key files
| File | Responsibility |
|---|---|
| `domain_router.py` | `DomainRouter.classify(query) -> DomainRoute`. SLM prompt + JSON parse + confidence-thresholded top-1 vs top-K selection over the fixed Vietnamese medical taxonomy. |
| `__init__.py` | Package docstring only ("routers + planners that decide where to send a query"). No exports. |

## Public interface
```python
@dataclass
class DomainRoute:
    systems: list[str]          # taxonomy values, e.g. ["tim_mach"]
    specialties: list[str]      # e.g. ["noi", "cap_cuu"]
    confidence: float
    raw: dict                   # the raw LLM JSON payload (default {})

class DomainRouter:
    def __init__(self) -> None: ...          # constructs its own LLMGateway()
    async def classify(self, query: str) -> DomainRoute: ...
```

**How it's accessed.** Reasoning/API code does NOT import `DomainRouter` directly.
It goes through the `ServiceContainer` so there's a single lazily-built instance:

```python
from src.agentrag.services.container import get_container
route = await get_container().domain_router.classify(query)
```

(`container.py::domain_router` is a lazy property — note the property itself
comments that this is a Reasoning-Plane service that merely *lives* in the
container for single-instance reuse.)

The legacy/back-compat path is `FederatedRetriever(router=DomainRouter())` in
`retrieval/federated.py`, which auto-routes internally — but only when an explicit
router is injected (kept for tests + legacy entry points; cold in normal S4 flow).

## Data flow
**Inputs:** a raw user query string.

**What it does:**
1. Calls `LLMGateway.json_response(system_prompt=SYSTEM_PROMPT, user_prompt={"query": query}, task="domain_router")`.
   The `task="domain_router"` tag lets `LLM_TASK_MODEL_MAP` route this to a cheap
   model when `LLM_ROUTING_ENABLED`.
2. Defensively parses the payload — any exception, non-dict payload, or bad
   `confidence` degrades to empty/`0.0` (never raises).
3. Keeps only `str` entries from `systems` / `specialties`.
4. **Selection rule:** if `confidence >= DOMAIN_ROUTER_CONFIDENCE_THRESHOLD` and at
   least one system was returned → take **top-1** system (narrow). Otherwise → take
   up to `DOMAIN_ROUTER_TOP_K` systems (broaden federation when ambiguous).
   Specialties are always truncated to `top_k`.

**Outputs:** a `DomainRoute`.

**Upstream callers (who calls `.classify`):**
- `adapter/routers/chat.py` — `execute_chat`, `regenerate`, `execute_chat_stream`.
  Only when the user did NOT pick a domain manually and `DOMAIN_FILTER_ENABLED` is
  true. The UI `domain_filter` override always wins over the router. The picks are
  translated into `{"system": route.systems[0], "specialties": route.specialties}`.

**Downstream:** the picks become `system_override` / `specialty_override` kwargs to
`RetrievalService` / `FederatedRetriever`, which apply them as ES filter clauses.

## Config
Read from `src/agentrag/config.py` (`settings.*`):

| Setting | Default | Effect |
|---|---|---|
| `DOMAIN_ROUTER_CONFIDENCE_THRESHOLD` | `0.7` | At/above this confidence, collapse to the single top-1 system. |
| `DOMAIN_ROUTER_TOP_K` | `3` | Max systems (when below threshold) and max specialties returned. |
| `DOMAIN_FILTER_ENABLED` | `True` | Gates whether callers (`chat.py`, `FederatedRetriever`) invoke routing at all. Read by callers, **not** by `DomainRouter` itself. |

Routing model selection is governed indirectly by `LLM_ROUTING_ENABLED` +
`LLM_TASK_MODEL_MAP` via the `task="domain_router"` tag (see `services/llm_gateway.py`).

## Gotchas
- **`classify` never raises.** On LLM error / malformed JSON it returns an empty
  `DomainRoute(systems=[], specialties=[], confidence=0.0)`. Callers must treat
  "empty route" as "no domain filter", not as a failure — `chat.py` does exactly
  this (and additionally swallows exceptions around the call).
- **Closed taxonomy.** The prompt allows ONLY the fixed system/specialty values
  listed in `SYSTEM_PROMPT`. These must stay in sync with the ontology — see
  `adapter/routers/ontology.py` and `ontology/schema.py`, both of which note they
  mirror the `DomainRouter` prompt sets. Changing one without the others drifts the
  filter values away from what's actually indexed.
- **Confidence is model-reported**, not calibrated. The top-1-vs-top-K split rides
  entirely on whatever number the SLM emits; tune via the threshold setting rather
  than expecting reliable probabilities.
- **Top-1 vs top-K asymmetry:** the threshold collapses *systems* to a single pick,
  but specialties are always returned up to `top_k` regardless of confidence.
- **`DomainRoute.raw` carries the full payload** for debugging, but `out["domain_route"]`
  surfaced for the UI signal (built in `FederatedRetriever.search` and read in
  `agent/graph_service.py` to populate `domain_route`) only includes
  `{systems, specialties, confidence}` — not `raw`.
