"""Runtime overrides for model defaults.

The UI (`PUT /on/api/models/defaults`) lets users switch the active provider/model
without editing .env. Overrides are persisted to a JSON file so the change
survives restarts. Loaded once at process startup and re-applied whenever the
file is updated via :func:`save_overrides`.

Scope: only the four model-pair settings are overridable. Everything else still
flows through .env. Worker processes pick up overrides on next start.
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Mapping

log = logging.getLogger(__name__)

OVERRIDE_PATH = Path(os.environ.get("AGENTRAG_OVERRIDES_FILE", "data/model_overrides.json"))

# Fields exposed to PUT /models/defaults. Anything not in this set is ignored.
ALLOWED_FIELDS = {
    "EMBEDDING_PROVIDER", "EMBEDDING_MODEL",
    "EXTRACTION_PROVIDER", "EXTRACTION_MODEL",
    "AGENT_PROVIDER", "AGENT_MODEL",
    "VISION_PROVIDER", "VISION_MODEL",
}


def load_overrides() -> dict[str, Any]:
    if not OVERRIDE_PATH.exists():
        return {}
    try:
        with OVERRIDE_PATH.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return {k: v for k, v in data.items() if k in ALLOWED_FIELDS and v not in (None, "")}
    except (OSError, json.JSONDecodeError) as exc:
        log.warning("failed to load %s: %s — ignoring overrides", OVERRIDE_PATH, exc)
        return {}


def save_overrides(updates: Mapping[str, Any]) -> dict[str, Any]:
    current = load_overrides()
    for k, v in updates.items():
        if k not in ALLOWED_FIELDS:
            continue
        if v in (None, ""):
            current.pop(k, None)
        else:
            current[k] = v
    OVERRIDE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OVERRIDE_PATH.open("w", encoding="utf-8") as f:
        json.dump(current, f, indent=2, sort_keys=True)
    return current


def apply_overrides(settings_obj) -> dict[str, Any]:
    """Mutate the in-memory settings object with persisted overrides."""
    overrides = load_overrides()
    for k, v in overrides.items():
        try:
            setattr(settings_obj, k, v)
        except (ValueError, TypeError) as exc:
            log.warning("override %s=%r rejected: %s", k, v, exc)
    return overrides
