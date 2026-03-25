from __future__ import annotations

import hashlib
import re
from typing import Dict, List

try:
    import tiktoken
except ModuleNotFoundError:
    tiktoken = None


class SimpleTokenizer:
    def encode(self, text: str) -> list[str]:
        return re.findall(r"\S+\s*", text)

    def decode(self, tokens: list[str]) -> str:
        return "".join(tokens)


class HybridChunker:
    def __init__(
        self,
        max_tokens: int = 1024,
        overlap_tokens: int = 0,
        tokenizer_model: str = "text-embedding-3-large",
        split_on_headings: bool = True,
    ):
        self.max_tokens = max_tokens
        self.overlap_tokens = overlap_tokens
        self.split_on_headings = split_on_headings
        if tiktoken is None:
            self.tokenizer = SimpleTokenizer()
        else:
            try:
                self.tokenizer = tiktoken.encoding_for_model(tokenizer_model)
            except KeyError:
                self.tokenizer = tiktoken.get_encoding("cl100k_base")

    def chunk(self, content: str, metadata: Dict | None = None) -> List[Dict]:
        chunks = []
        chunk_id = 0
        sections = self._split_sections(content)

        for section_name, section_text in sections:
            tokens = self.tokenizer.encode(section_text)
            position = 0

            while position < len(tokens):
                end = min(position + self.max_tokens, len(tokens))
                chunk_tokens = tokens[position:end]
                chunk_text = self.tokenizer.decode(chunk_tokens).strip()
                if not chunk_text:
                    position = self._next_position(position, end)
                    continue

                chunks.append(
                    {
                        "content": chunk_text,
                        "content_hash": hashlib.sha256(
                            chunk_text.encode("utf-8")
                        ).hexdigest(),
                        "segment_type": "text",
                        "section_path": section_name,
                        "position": chunk_id,
                        "metadata": metadata or {},
                    }
                )
                position = self._next_position(position, end)
                chunk_id += 1

        return chunks

    def _split_sections(self, content: str) -> list[tuple[str, str]]:
        if not self.split_on_headings:
            return [("chunk_0", content)]

        sections: list[tuple[str, str]] = []
        current_heading = "intro"
        current_lines: list[str] = []

        for line in content.splitlines():
            if re.match(r"^#{1,6}\s+", line):
                if current_lines:
                    sections.append(
                        (self._normalize_section_name(current_heading), "\n".join(current_lines))
                    )
                    current_lines = []
                current_heading = line.lstrip("#").strip()
            current_lines.append(line)

        if current_lines:
            sections.append(
                (self._normalize_section_name(current_heading), "\n".join(current_lines))
            )

        return sections or [("chunk_0", content)]

    def _normalize_section_name(self, heading: str) -> str:
        normalized = re.sub(r"[^a-zA-Z0-9]+", "_", heading).strip("_").lower()
        return normalized or "section"

    def _next_position(self, current: int, end: int) -> int:
        if self.overlap_tokens <= 0:
            return end
        next_position = max(end - self.overlap_tokens, 0)
        if next_position <= current:
            return end
        return next_position
