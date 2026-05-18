"""PPTX → PDF (libreoffice) → MinerU pipeline.

Used when `INGEST_USE_MINERU_FOR_PPTX=true`. Requires libreoffice on PATH
(apt: `libreoffice` or `libreoffice-core` + `libreoffice-impress`) and
the `mineru` CLI installed.

Falls back to None on any missing dep so the pipeline can route to
MarkItDown without raising.
"""
from __future__ import annotations

import logging
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from src.agentrag.ingestion.parsers import mineru_parser

logger = logging.getLogger(__name__)


def libreoffice_bin() -> str | None:
    for name in ("libreoffice", "soffice"):
        path = shutil.which(name)
        if path:
            return path
    return None


def is_available() -> bool:
    return libreoffice_bin() is not None and mineru_parser.is_available()


def parse_pptx(file_path: str) -> dict[str, Any] | None:
    """Convert PPTX → PDF via libreoffice, then run MinerU. Returns the
    standard parser dict, or None on failure (caller falls back to
    MarkItDown)."""
    bin_path = libreoffice_bin()
    if not bin_path:
        logger.warning(
            "libreoffice not on PATH; install with: apt install libreoffice-impress"
        )
        return None
    if not mineru_parser.is_available():
        logger.warning("mineru CLI missing — cannot route PPTX through MinerU")
        return None

    src = Path(file_path)
    if not src.exists():
        raise FileNotFoundError(file_path)

    with tempfile.TemporaryDirectory(prefix="pptx2pdf_") as tmp:
        tmp_dir = Path(tmp)
        cmd = [
            bin_path,
            "--headless",
            "--convert-to", "pdf",
            "--outdir", str(tmp_dir),
            str(src),
        ]
        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300,
                check=False,
            )
        except (subprocess.TimeoutExpired, OSError) as exc:
            logger.warning("libreoffice convert failed: %s", exc)
            return None
        if proc.returncode != 0:
            logger.warning(
                "libreoffice returned %d: %s",
                proc.returncode,
                (proc.stderr or proc.stdout)[:500],
            )
            return None

        pdfs = list(tmp_dir.glob("*.pdf"))
        if not pdfs:
            logger.warning("libreoffice produced no PDF for %s", src.name)
            return None
        result = mineru_parser.parse_pdf(str(pdfs[0]))
        if result is not None:
            # Annotate source so downstream telemetry can distinguish PPTX-via-MinerU.
            for entry in result.get("page_data") or []:
                entry["source"] = "mineru_pptx"
        return result
