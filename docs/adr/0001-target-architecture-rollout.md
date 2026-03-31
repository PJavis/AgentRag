# ADR 0001: Target Architecture Rollout

## Status

Accepted (Phase A started).

## Context

PAM currently works end-to-end but mixes orchestration and service concerns in the agent layer.  
We need a production-ready rollout path toward:

- Supervisor-style orchestration
- Service boundaries for knowledge, security, context assembly, and LLM gateway
- Stable contracts for future multi-agent expansion

## Decision

Adopt incremental rollout (not big-bang), with immediate service facades:

- `KnowledgeService`: tool/retrieval facade
- `ContextAssemblyService`: dedicated context assembly entrypoint
- `SecurityService`: query-time policy gate (v1 validation + result filter)
- `LLMGateway`: centralized LLM call facade (v1 latency hook, future routing/cost)

`AgentService` remains the Supervisor Agent and now orchestrates through these services.

## Consequences

Positive:

- Clearer boundaries for future `Data/Insight/Report` worker agents.
- Lower risk migration path with unchanged `/chat` API response shape.
- Easier observability and policy insertion points.

Tradeoffs:

- Temporary duplicate abstraction during transition.
- Some services are v1 pass-through until policy/routing logic is expanded.

## Rollout Plan

### Phase A (in progress)

1. Create service facades and wire `AgentService`.
2. Keep existing behavior/backward compatibility.
3. Add architecture docs and implementation backlog.

### Phase B

1. Expand `SecurityService` to metadata/document-level authz policies.
2. Move query expansion + intent-aware retrieval to `KnowledgeService`.
3. Add structured traces/metrics per service stage.

### Phase C

1. Add model routing/cost accounting in `LLMGateway`.
2. Split specialized worker agents on top of stable contracts.
3. Add MCP server integration on top of the same service layer.
