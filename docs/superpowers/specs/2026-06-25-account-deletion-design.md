# Account deletion (full user data wipe) — design

**Date:** 2026-06-25 · **P4** (privacy / right-to-delete). Self-service: the
authenticated user erases all of their own data across Postgres + Elasticsearch +
the image filesystem.

## Context

User-owned data (all keyed on the JWT `user_id`):
- `Document.user_id` (`database/models.py`) → `Segment` (FK `document_id`) + ES
  `agentrag_segments`/StructMem (by document title) + extracted-image folder.
- `Conversation.user_id` → `ChatMessage` (FK `conversation_id`) + ES chat-memory
  (`agentrag_chat_entries`/`_synthesis`, by `conversation_id`).
- `EventLog.user_id` (activity feed).
- `AdapterChatFeedback.user_id` (`adapter/db.py`).

`delete_source` (`adapter/routers/sources.py:526`) already cascades ONE document →
`Segment` delete + `Document` delete + `ElasticsearchStore().delete_document(title)` +
`shutil.rmtree` of the image folder. The wipe reuses this pattern per document.

Auth: `get_identity(request)` → `AuthIdentity(user_id, is_legacy)`; `user_id` is a UUID
for real JWT users, `LEGACY_PASSWORD_USER_ID` for the legacy single-user mode, or
`"anonymous"`.

## Scope

In: a `delete_user_data(user_id)` service + a `DELETE` endpoint (auth-gated) + the user
row removal + a live integration test. Out: soft-delete/recovery window; an
admin-deletes-another-user flow.

## Design

### Service `src/agentrag/adapter/account_deletion.py`
`async def delete_user_data(user_id: str) -> dict[str, int]` — ordered, best-effort on
the search/file layers so a search hiccup can't half-delete Postgres:

1. **Documents** — `select(Document).where(Document.user_id == user_id)`; per doc:
   `delete(Segment).where(Segment.document_id == doc.id)`, `session.delete(doc)`; then
   (outside the txn, wrapped) `ElasticsearchStore().delete_document(doc.title)` +
   `shutil.rmtree(IMAGE_STORAGE_DIR / sanitize(title), ignore_errors=True)`.
2. **Conversations** — collect ids `select(Conversation.id).where(user_id==me)`;
   `delete(ChatMessage).where(ChatMessage.conversation_id.in_(ids))`;
   `delete(Conversation).where(user_id==me)`; best-effort ES delete-by-query on the
   chat-memory indices for those `conversation_id`s (wrapped).
3. **Feedback** — `delete(AdapterChatFeedback).where(user_id==str(user_id))`.
4. **Events** — `delete(EventLog).where(EventLog.user_id == user_id)`.
5. **User row** — `delete(User).where(User.id == user_id)` LAST (steps 1-4 clear the
   FK references first).

Returns counts: `{"documents","segments","conversations","messages","feedback","events"}`.
Each store-purge step is wrapped in try/except (logged) — Postgres is the source of
truth and must fully delete even if ES/files error.

### Endpoint (in the existing chat router, `adapter/routers/chat.py`)
`DELETE …/account` — `identity = get_identity(request)`; if `identity is None`, or
`identity.user_id in ("anonymous", "")`, or `identity.is_legacy` → `HTTPException(403,
"account deletion requires an authenticated user")`. Else
`counts = await delete_user_data(identity.user_id)`; return `{"deleted": counts}`.
Only ever deletes rows matching the caller's own `user_id`.

## Error handling
PG deletions are authoritative; ES + filesystem purges are best-effort (wrapped). The
endpoint refuses anonymous/legacy so a shared/legacy identity can't wipe shared data.

## Testing
- **Unit** (`tests/adapter/test_account_deletion.py`): the endpoint guard rejects
  `anonymous`, legacy, and missing identity with 403 (mock `get_identity`).
- **Live integration** (stack up): seed a throwaway `User` + a `Document`(+`Segment`) +
  `Conversation`(+`ChatMessage`) + `AdapterChatFeedback` + `EventLog`; call
  `delete_user_data(uid)`; assert every row is gone and counts match. (Destructive op →
  verify against the real DB.)
