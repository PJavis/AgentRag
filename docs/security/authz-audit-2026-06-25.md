# AuthZ depth audit — 2026-06-25 (P4)

Scope: per-resource authorization on the adapter HTTP endpoints. Question: beyond
*authentication* (are you logged in?), do endpoints enforce *authorization* (is this
**your** resource?). Method: enumerate every router endpoint and check whether it scopes
the operation to the caller's `user_id`.

## DECISION (2026-06-25): single-tenant / on-prem — Finding 1 ACCEPTED

**Deployment model: single-tenant.** VITAL runs as one on-prem / per-clinic instance —
**not** multi-account across VMs. Under this model `AUTH_ENABLED` is an **access gate**
(are you allowed in?), **not** a tenant-isolation boundary, and there is effectively one
tenant. Therefore **Finding 1 (IDOR/BOLA) is accepted by design and is NOT a launch
blocker** for this deployment. The ~39-endpoint shared-ownership-dependency refactor in
Recommendation #2 is **deliberately not done** — it would add cost and risk breaking the
legacy single-user mode for zero benefit at one tenant.

**This acceptance is conditional. Re-open Finding 1 as a launch blocker BEFORE any of:**
- enabling real per-user accounts / `AUTH_ALLOW_SIGNUP` for untrusted users,
- deploying a shared instance serving more than one clinic/tenant,
- exposing the resource endpoints to a second VM / multi-tenant front end.

At that point implement Recommendation #2 (one shared ownership dependency + list-scoping)
first. Findings 2–4 (inconsistent scoping, list-endpoint leakage, `stubs.py`) are likewise
N/A under single-tenant and re-open with it.

## Posture by router (`get_identity` + user-scope filter present?)

| Router | Endpoints | User-scoped? |
|---|---|---|
| `chat.py` | 13 | ✅ yes (`get_identity`, `user_id` filters; `DELETE /chat/account` 403-guards) |
| `activity.py` | 4 | ✅ yes |
| `search.py` | 3 | ✅ yes |
| `notebooks.py` | 9 | ❌ **no ownership check** |
| `sources.py` | 12 | ❌ **no** (2 incidental `user_id` refs, not a guard) |
| `notes.py` | 5 | ❌ **no** |
| `insights.py` | 5 | ❌ **no** |
| `transformations.py` | 8 | ❌ **no** |
| `models.py` / `config.py` / `ontology.py` | 34 | ⚪ config/catalog (not per-user data) — lower risk |
| `stubs.py` | 29 | ⚪ open-notebook stubs (review if wired to real data) |

There is **no global ownership dependency/middleware** — each endpoint is on its own.

## Finding 1 — IDOR / BOLA on resource endpoints · **High (if multi-tenant)**

`notebooks`, `sources`, `notes`, `insights`, `transformations` operate on a resource by
its id with no check that it belongs to the caller. Example (`sources.py:526`):

```python
async def delete_source(source_id: str):
    doc = await session.get(Document, _parse_source_id(source_id))   # no user_id check
    ...delete...
```

`Document` HAS a `user_id`; it is simply not checked. Same shape in `delete_notebook`,
`delete_note`, `delete_insight`, `delete_transformation`, and the corresponding GET/PUT
handlers. **Impact:** any authenticated user can read, modify, or delete another user's
notebooks/sources/notes by guessing or enumerating UUIDs — for a clinical tool, that is
unauthorized access to another patient's/clinician's documents.

**Severity is deployment-dependent:**
- **Multi-tenant** (real per-user accounts): **High** — direct cross-tenant access.
- **Single-user / legacy mode** (one shared account; `AUTH_ENABLED` gates *access*, not
  isolation): **N/A** — there is effectively one tenant. The code today reflects this
  single-tenant origin (open-notebook).

## Finding 2 — partial, inconsistent scoping · Medium

`chat`/`activity`/`search` were retrofitted with `user_id` scoping; the document/notebook
surface was not. This inconsistency means the *chat* history is private but the *sources*
behind it are not — a confusing and unsafe split for the same data.

## Recommendations

1. **Decide the tenancy model explicitly.** If the product is single-user/on-prem
   per-clinic, document that (`AUTH_ENABLED` = access gate, not isolation) and the
   IDOR is accepted. If multi-tenant, treat Finding 1 as a launch blocker.
2. **If multi-tenant — add one shared ownership dependency**, not 39 ad-hoc checks:
   a FastAPI dependency that, given a resource id + `get_identity(request)`, loads the
   resource, resolves its owner (`Document.user_id`; notes/insights/transformations via
   their owning document/notebook), and raises `403` on mismatch. Apply to every by-id
   GET/PUT/DELETE on `notebooks`/`sources`/`notes`/`insights`/`transformations`.
   - Preserve legacy/anonymous: when `identity.is_legacy` or `user_id is None`, fall back
     to today's behavior (no isolation) so single-user deployments don't break.
3. **List-endpoint scoping:** GET-collection handlers should filter `WHERE user_id =
   identity.user_id` (else they leak the existence of other users' resources).
4. **Audit `stubs.py` (29)** — confirm none expose real per-user data unscoped.

## Not changed in this audit
Fixing Finding 1 is a cross-cutting change touching ~39 endpoints and risks breaking the
legacy single-user mode; it needs the tenancy decision (#1) first, then its own
spec→plan→implement cycle. This document is the audit; the fix is the follow-up.

## Already-good (for reference)
- `chat.py` scopes by `user_id`; `DELETE /chat/account` 403-refuses anonymous/legacy and
  only wipes the caller's own `user_id` (`account_deletion.py`).
- Auth + rate-limit are enabled (`AUTH_ENABLED`, `RATE_LIMIT_ENABLED`); see `DEPLOY-RUNBOOK.md`.
