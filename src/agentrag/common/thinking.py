"""Strip <think>...</think> blocks from reasoning-model outputs.

Adapted from open-notebook (open_notebook/utils/text_utils.py).

Reasoning models like DeepSeek-R1, QwQ, and DeepHermes wrap their internal
chain-of-thought in <think>...</think> tags. For end-user answers we want
just the final response without that scratch work.

Some models (Nemotron etc.) emit malformed output: `content</think>` with no
opening tag. parse_thinking_content() handles both cases.
"""
from __future__ import annotations

import re

# Match both <think>…</think> (DeepSeek-R1, QwQ) and <thought>…</thought>
# (Gemma 4 via Gemini OpenAI-compat endpoint emits chain-of-thought this way).
_THINK_PATTERN = re.compile(r"<(?:think|thought)>(.*?)</(?:think|thought)>", re.DOTALL)
_THINK_PATTERN_NO_OPEN = re.compile(r"^(.*?)</(?:think|thought)>", re.DOTALL)

# Skip very large content (>100KB) to avoid pathological regex backtracking.
_MAX_BYTES = 100_000


def parse_thinking_content(content: str) -> tuple[str, str]:
    """Return (thinking_block, cleaned_content). Handles malformed `…</think>`."""
    if not isinstance(content, str):
        return "", "" if content is None else str(content)
    if len(content) > _MAX_BYTES:
        return "", content

    matches = _THINK_PATTERN.findall(content)
    if matches:
        thinking = "\n\n".join(m.strip() for m in matches)
        cleaned = _THINK_PATTERN.sub("", content)
        cleaned = re.sub(r"\n\s*\n\s*\n", "\n\n", cleaned).strip()
        return thinking, cleaned

    m = _THINK_PATTERN_NO_OPEN.match(content)
    if m:
        return m.group(1).strip(), content[m.end():].strip()

    return "", content


def clean_thinking_content(content: str) -> str:
    """Strip <think>...</think> blocks, return only the user-visible content."""
    _, cleaned = parse_thinking_content(content)
    return cleaned
