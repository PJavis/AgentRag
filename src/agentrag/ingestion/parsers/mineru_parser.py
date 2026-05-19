"""MinerU parser shim (opt-in alternative to Tesseract + vision LLM).

Invokes the `mineru` CLI (https://github.com/opendatalab/mineru) — layout
detection + OCR + formula-to-LaTeX + table-to-HTML over the whole PDF in
one pass. Produces a single markdown file with per-page block metadata.

We map MinerU's output back into the PDFParser contract:
  {parsed_content, pages, page_data: [{page_num, text, source='mineru'}]}

Activated when `PDF_PARSER_BACKEND=mineru` AND MinerU CLI is installed.
Falls back to Tesseract path on missing CLI / parse failure.

Install:
  pip install -U mineru          # CPU
  pip install -U "mineru[all]"   # + GPU + table-rec + formula models

Models download lazily on first run (~3-5GB). Subsequent calls cache.
"""
from __future__ import annotations

import json
import logging
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from src.agentrag.config import settings
from src.agentrag.ingestion.parsers.pdf_parser import make_page_marker

logger = logging.getLogger(__name__)


def is_available() -> bool:
    """Cheap check: is `mineru` on PATH?"""
    return shutil.which("mineru") is not None


def parse_pdf(file_path: str) -> dict[str, Any] | None:
    """Run MinerU on a PDF. Returns parser dict or None on failure."""
    if not is_available():
        logger.warning("mineru CLI not found; install with: pip install -U mineru")
        return None

    src = Path(file_path)
    if not src.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    out_root = Path(settings.MINERU_OUTPUT_DIR).expanduser().resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    # Map legacy/short lang codes to the new mineru CLI vocab.
    # Newer mineru (≥0.x) accepts only: ch|en|korean|japan|chinese_cht|
    # ta|te|ka|th|el|latin|arabic|east_slavic|cyrillic|devanagari.
    # Vietnamese uses Latin script → fall back to 'latin'.
    _LANG_ALIASES = {
        "vie": "latin",
        "vi": "latin",
        "vi-vn": "latin",
        "vn": "latin",
    }
    lang = (settings.MINERU_LANG or "").strip().lower()
    lang = _LANG_ALIASES.get(lang, lang or "ch")
    backend = (getattr(settings, "MINERU_BACKEND", "hybrid-auto-engine") or "hybrid-auto-engine").strip()

    # Use a per-call workdir so concurrent ingests don't clash.
    with tempfile.TemporaryDirectory(dir=out_root, prefix="mineru_") as tmp:
        tmp_dir = Path(tmp)
        cmd = [
            "mineru",
            "-p", str(src),
            "-o", str(tmp_dir),
            "-l", lang,
            "-b", backend,
        ]
        try:
            proc = subprocess.run(
                cmd,
                check=False,
                capture_output=True,
                text=True,
                timeout=900,  # 15 min cap per PDF
            )
        except (subprocess.TimeoutExpired, FileNotFoundError, OSError) as exc:
            logger.warning("mineru subprocess failed: %s", exc)
            return None
        if proc.returncode != 0:
            logger.warning(
                "mineru returned %d: %s",
                proc.returncode,
                (proc.stderr or proc.stdout)[:500],
            )
            return None

        # MinerU writes <tmp>/<doc-stem>/auto/<doc-stem>.md and
        # <doc-stem>_content_list.json (page-aware blocks).
        return _collect_mineru_output(tmp_dir, src.stem)


def _collect_mineru_output(tmp_dir: Path, stem: str) -> dict[str, Any] | None:
    """Read MinerU outputs from its conventional tmp layout."""
    # Try the documented layout first; fall back to glob search.
    candidates = list(tmp_dir.rglob(f"{stem}.md"))
    if not candidates:
        candidates = list(tmp_dir.rglob("*.md"))
    if not candidates:
        logger.warning("mineru produced no markdown output in %s", tmp_dir)
        return None
    md_path = candidates[0]
    md = md_path.read_text(encoding="utf-8", errors="replace")

    # content_list.json sits next to the .md
    page_data: list[dict[str, Any]] = []
    page_text: dict[int, list[str]] = {}
    cl_path = md_path.parent / f"{stem}_content_list.json"
    if not cl_path.exists():
        # Newer MinerU versions name it differently — best-effort glob.
        cl_candidates = list(md_path.parent.glob("*_content_list.json"))
        cl_path = cl_candidates[0] if cl_candidates else None
    if cl_path and cl_path.exists():
        try:
            blocks = json.loads(cl_path.read_text(encoding="utf-8"))
            if isinstance(blocks, list):
                for block in blocks:
                    page = block.get("page_idx")
                    if page is None:
                        continue
                    page = int(page) + 1  # mineru is 0-indexed; align with PyMuPDF
                    chunk_text = block.get("text") or block.get("md") or ""
                    if isinstance(chunk_text, list):
                        chunk_text = "\n".join(str(t) for t in chunk_text)
                    if chunk_text.strip():
                        page_text.setdefault(page, []).append(str(chunk_text))
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("mineru content_list parse failed: %s", exc)

    # Build per-page records + assembled content with page markers
    parts: list[str] = []
    if page_text:
        for page_num in sorted(page_text):
            text = "\n\n".join(page_text[page_num]).strip()
            if not text:
                continue
            page_data.append({"page_num": page_num, "text": text, "source": "mineru"})
            parts.append(f"{make_page_marker(page_num)}\n{text}")
        parsed_content = "\n\n".join(parts)
    else:
        # No structured json — single-page best effort, dump full md.
        page_data.append({"page_num": 1, "text": md, "source": "mineru"})
        parsed_content = f"{make_page_marker(1)}\n{md}"

    logger.info(
        "MinerU: parsed %s → %d pages, %d chars",
        stem,
        len(page_data),
        len(parsed_content),
    )
    return {
        "parsed_content": parsed_content,
        "pages": max(len(page_data), 1),
        "page_data": page_data,
    }
