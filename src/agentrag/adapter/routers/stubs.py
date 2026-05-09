"""Stub endpoints for open-notebook features not yet implemented in AgentRag.

Real endpoints live in their own router modules:
  - models.py            (models/*)
  - transformations.py   (transformations/*)
  - insights.py          (sources/{id}/insights/*, insights/*)
  - auth.py              (auth/*)

These stubs cover the long tail of UI buttons (podcasts, credentials,
commands, episode/speaker profiles, embedding rebuild) so the UI loads
without errors. They return well-formed empty responses.
"""
from __future__ import annotations

from fastapi import APIRouter

router = APIRouter()


# ── Credentials ───────────────────────────────────────────────────────────────


@router.get("/credentials")
async def list_credentials(provider: str | None = None):
    return []


@router.get("/credentials/status")
async def credentials_status():
    return {}


@router.get("/credentials/env-status")
async def credentials_env_status():
    return {"configured_providers": []}


@router.get("/credentials/by-provider/{provider}")
async def credentials_by_provider(provider: str):
    return []


@router.post("/credentials")
async def create_credential(body: dict):
    return {
        "id": "stub",
        "provider": body.get("provider", ""),
        "created": "2025-01-01T00:00:00Z",
        "updated": "2025-01-01T00:00:00Z",
    }


@router.get("/credentials/{credential_id}")
async def get_credential(credential_id: str):
    return {"id": credential_id}


@router.put("/credentials/{credential_id}")
async def update_credential(credential_id: str, body: dict):
    return {"id": credential_id}


@router.delete("/credentials/{credential_id}")
async def delete_credential(credential_id: str, migrate_to: str | None = None):
    return {"message": "Credential deleted"}


@router.post("/credentials/{credential_id}/test")
async def test_credential(credential_id: str):
    return {"success": True}


@router.post("/credentials/{credential_id}/discover")
async def discover_credential_models(credential_id: str):
    return {"models": []}


@router.post("/credentials/{credential_id}/register-models")
async def register_credential_models(credential_id: str, body: dict):
    return {"registered": 0}


@router.post("/credentials/migrate-from-provider-config")
@router.post("/credentials/migrate-from-env")
async def migrate_credentials(body: dict = {}):
    return {"migrated": 0}


# ── Podcasts (out of scope for AgentRag) ─────────────────────────────────────


@router.get("/podcasts/episodes")
async def list_episodes():
    return []


@router.delete("/podcasts/episodes/{episode_id}")
async def delete_episode(episode_id: str):
    return {"message": "Podcast feature is not enabled in this deployment."}


@router.post("/podcasts/generate")
async def generate_podcast(body: dict):
    return {
        "job_id": "unsupported",
        "status": "unsupported",
        "message": "Podcast generation is not enabled in this deployment.",
    }


@router.get("/podcasts/jobs/{job_id}")
@router.get("/commands/jobs/{job_id}")
async def get_job(job_id: str):
    return {"job_id": job_id, "status": "completed", "result": None, "error": None}


# ── Commands (background-job registry) ───────────────────────────────────────


@router.get("/commands/jobs")
async def list_jobs():
    return []


@router.post("/commands/jobs")
async def create_job(body: dict):
    return {"job_id": "stub", "status": "queued"}


@router.delete("/commands/jobs/{job_id}")
async def cancel_job(job_id: str):
    return {"job_id": job_id, "cancelled": True}


@router.get("/commands/registry/debug")
async def commands_debug():
    return {"command_count": 0, "registry": {}}


# ── Embedding (single-doc reindex hooks) ─────────────────────────────────────


@router.post("/embed")
async def embed_item(body: dict):
    return {"status": "completed"}


@router.post("/embeddings/rebuild")
async def rebuild_embeddings(body: dict):
    return {
        "command_id": "stub",
        "status": "queued",
        "message": "Use the ingestion pipeline to re-embed sources.",
    }


@router.get("/embeddings/rebuild/{command_id}/status")
async def rebuild_status(command_id: str):
    return {"command_id": command_id, "status": "completed"}


# ── Episode / Speaker profiles ────────────────────────────────────────────────


@router.get("/episode-profiles")
@router.get("/speaker-profiles")
async def list_profiles():
    return []


@router.delete("/episode-profiles/{profile_id}")
@router.delete("/speaker-profiles/{profile_id}")
async def delete_profile(profile_id: str):
    return {"message": "Profile feature is not enabled in this deployment."}
