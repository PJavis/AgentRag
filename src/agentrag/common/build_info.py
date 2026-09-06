"""Identity of the code that is actually running.

Why this exists: `docker-compose.fullstack.yml` and `docker-compose.deploy.yml`
are both invoked with `-f`, which disables auto-merge of
`docker-compose.override.yml` and therefore of the optional `./src:/app/src`
bind mount. A stale image on such a path does not error — it produces a
complete, plausible-looking result set from the wrong code. Documentation does
not protect anyone who does not read it; a line in every log and every result
artefact does.

Three identities, deliberately separate:

  ``image_git_sha``     the commit the image was BUILT from   (baked at build)
  ``image_build_id``    which build produced the image        (baked at build)
  ``source_sha``        hash of the .py files being IMPORTED  (computed now)

`baked_source_sha` is `source_sha` computed at build time and frozen into the
image. When the running `source_sha` differs from it, the code executing is not
the code the image shipped — a bind mount is shadowing it, or someone edited in
place. That is the signal; the git SHA alone cannot see it, because a mount
brings no `.git` with it.

Everything degrades to "unknown" rather than raising: this must never be the
reason a run fails.
"""

from __future__ import annotations

import hashlib
import logging
import os
from functools import lru_cache
from pathlib import Path

logger = logging.getLogger(__name__)

UNKNOWN = "unknown"
#: Cap so a pathological tree cannot stall startup.
MAX_FILES_HASHED = 5000
#: Written by the Dockerfile at build time. Lives OUTSIDE /app/src on purpose:
#: the bind mount covers /app/src only, so this file keeps recording what the
#: IMAGE shipped even while a mount changes what actually runs.
BAKED_SHA_FILE = Path("/app/.build-source-sha")


def _package_root() -> Path:
    """The directory holding the imported `agentrag` package."""
    return Path(__file__).resolve().parents[1]


@lru_cache(maxsize=8)
def source_sha(root: str | None = None) -> str:
    """Content hash of the `.py` files under `root` (default: the live package).

    Order-stable and path-relative, so the same source tree hashes identically
    inside and outside a container.
    """
    base = Path(root) if root else _package_root()
    try:
        files = sorted(
            p for p in base.rglob("*.py") if "__pycache__" not in p.parts
        )[:MAX_FILES_HASHED]
    except OSError:
        return UNKNOWN
    if not files:
        return UNKNOWN
    digest = hashlib.sha256()
    for path in files:
        try:
            body = path.read_bytes()
        except OSError:
            continue
        digest.update(str(path.relative_to(base)).encode("utf-8"))
        digest.update(hashlib.sha256(body).digest())
    return digest.hexdigest()[:12]


def _read_baked_sha() -> str:
    """Source hash the image was built with, written by the Dockerfile."""
    try:
        return BAKED_SHA_FILE.read_text(encoding="utf-8").strip() or UNKNOWN
    except OSError:
        return UNKNOWN


@lru_cache(maxsize=1)
def build_info() -> dict[str, str | bool | None]:
    """Provenance of the running code. Never raises."""
    image_git_sha = os.environ.get("AGENTRAG_GIT_SHA") or UNKNOWN
    image_build_id = os.environ.get("AGENTRAG_BUILD_ID") or UNKNOWN
    built_at = os.environ.get("AGENTRAG_BUILT_AT") or UNKNOWN
    baked = os.environ.get("AGENTRAG_SOURCE_SHA") or _read_baked_sha()

    try:
        running = source_sha()
    except Exception:  # noqa: BLE001 — provenance must never break a run
        running = UNKNOWN

    if baked == UNKNOWN or running == UNKNOWN:
        matches: bool | None = None
    else:
        matches = baked == running

    return {
        "image_git_sha": image_git_sha,
        "image_build_id": image_build_id,
        "image_built_at": built_at,
        "baked_source_sha": baked,
        "running_source_sha": running,
        "source_matches_image": matches,
    }


def format_build_banner() -> str:
    """One line for the startup log; second line only when something is off."""
    info = build_info()
    head = (
        f"build: git={info['image_git_sha']} "
        f"build_id={info['image_build_id']} "
        f"built_at={info['image_built_at']} "
        f"source={info['running_source_sha']}"
    )
    matches = info["source_matches_image"]
    if matches is True:
        return head + " (source matches image)"
    if matches is False:
        return (
            head
            + f" (SOURCE DOES NOT MATCH IMAGE: image shipped "
            f"{info['baked_source_sha']}, running {info['running_source_sha']} "
            "— a bind mount is shadowing /app/src, or the image is stale. Any "
            "result produced now belongs to the running source, not to "
            f"git {info['image_git_sha']}.)"
        )
    return head + " (provenance unverified: image was built without build args)"


def log_build_banner(where: str) -> None:
    info = build_info()
    line = f"[{where}] {format_build_banner()}"
    if info["source_matches_image"] is False:
        logger.warning(line)
    else:
        logger.info(line)
