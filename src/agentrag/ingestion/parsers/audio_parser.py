"""Audio parser — transcribe mp3/wav/m4a/... via Whisper.

Two backends:
1. `faster-whisper` (default) — runs locally on CPU/GPU. Auto-downloads
   model on first run (~150MB for small, ~1.5GB for large-v3). Free.
2. `openai_whisper` — calls OpenAI's /audio/transcriptions endpoint.
   Used when `AUDIO_TRANSCRIBE_PROVIDER=openai` and `OPENAI_API_KEY` set.

Output: same shape as MarkItDownParser so the rest of the ingestion
pipeline (chunker + embedder + StructMem) can consume it unchanged.
The transcript is prefixed with `[hh:mm:ss]` markers per segment so the
chunker can preserve approximate timestamps when relevant.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from src.agentrag.config import settings

logger = logging.getLogger(__name__)


def _format_ts(seconds: float) -> str:
    s = int(seconds)
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{sec:02d}" if h else f"{m:02d}:{sec:02d}"


class AudioParser:
    """Transcribe audio files to text chunks."""

    def __init__(self) -> None:
        self._fw_model = None  # lazy faster-whisper instance

    def _load_faster_whisper(self):
        if self._fw_model is not None:
            return self._fw_model
        try:
            from faster_whisper import WhisperModel  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "faster-whisper not installed. Run: uv add faster-whisper"
            ) from exc
        model_name = getattr(settings, "AUDIO_WHISPER_MODEL", "small")
        device = getattr(settings, "AUDIO_WHISPER_DEVICE", "auto")
        compute_type = getattr(settings, "AUDIO_WHISPER_COMPUTE_TYPE", "auto")
        logger.info(
            "AudioParser: loading faster-whisper model=%s device=%s compute_type=%s",
            model_name, device, compute_type,
        )
        # Resolve "auto" defaults — let faster-whisper pick CPU/GPU.
        kwargs: dict[str, Any] = {"model_size_or_path": model_name}
        if device != "auto":
            kwargs["device"] = device
        if compute_type != "auto":
            kwargs["compute_type"] = compute_type
        self._fw_model = WhisperModel(**kwargs)
        return self._fw_model

    def parse(self, file_path: str, title: str) -> dict[str, Any]:
        """Parse an audio file → markdown-ish text + page_data analog.

        Returns the same dict shape as other parsers:
            {"content": "<full transcript with [ts] markers>",
             "metadata": {"duration_s": float, "language": str, "segments": list}}
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(file_path)

        provider = getattr(settings, "AUDIO_TRANSCRIBE_PROVIDER", "faster_whisper")
        if provider == "openai":
            return self._parse_openai(path, title)
        return self._parse_faster_whisper(path, title)

    def _parse_faster_whisper(self, path: Path, title: str) -> dict[str, Any]:
        model = self._load_faster_whisper()
        language = getattr(settings, "AUDIO_WHISPER_LANGUAGE", None)
        beam_size = int(getattr(settings, "AUDIO_WHISPER_BEAM_SIZE", 5))
        segments_iter, info = model.transcribe(
            str(path),
            language=language,
            beam_size=beam_size,
            vad_filter=True,
        )
        lines: list[str] = []
        segments_data: list[dict[str, Any]] = []
        for seg in segments_iter:
            ts = _format_ts(seg.start)
            text = (seg.text or "").strip()
            if not text:
                continue
            lines.append(f"[{ts}] {text}")
            segments_data.append({
                "start": float(seg.start),
                "end": float(seg.end),
                "text": text,
            })
        content = "\n".join(lines)
        logger.info(
            "AudioParser(faster-whisper): %s → %d segments, language=%s, %.1fs",
            path.name, len(segments_data), getattr(info, "language", "?"),
            getattr(info, "duration", 0.0),
        )
        return {
            "content": content,
            "metadata": {
                "duration_s": float(getattr(info, "duration", 0.0)),
                "language": getattr(info, "language", None),
                "segments": segments_data,
                "title": title,
                "media_type": "audio",
            },
        }

    def _parse_openai(self, path: Path, title: str) -> dict[str, Any]:
        if not settings.OPENAI_API_KEY:
            raise ValueError(
                "OPENAI_API_KEY required for AUDIO_TRANSCRIBE_PROVIDER=openai"
            )
        from openai import OpenAI

        client = OpenAI(api_key=settings.OPENAI_API_KEY)
        with path.open("rb") as f:
            resp = client.audio.transcriptions.create(
                model=getattr(settings, "AUDIO_OPENAI_MODEL", "whisper-1"),
                file=f,
                response_format="verbose_json",
                language=getattr(settings, "AUDIO_WHISPER_LANGUAGE", None),
            )
        segments_data: list[dict[str, Any]] = []
        lines: list[str] = []
        for seg in getattr(resp, "segments", None) or []:
            text = getattr(seg, "text", "").strip()
            if not text:
                continue
            start = float(getattr(seg, "start", 0.0))
            end = float(getattr(seg, "end", 0.0))
            lines.append(f"[{_format_ts(start)}] {text}")
            segments_data.append({"start": start, "end": end, "text": text})
        content = "\n".join(lines) or (getattr(resp, "text", "") or "")
        return {
            "content": content,
            "metadata": {
                "duration_s": float(getattr(resp, "duration", 0.0)),
                "language": getattr(resp, "language", None),
                "segments": segments_data,
                "title": title,
                "media_type": "audio",
            },
        }
