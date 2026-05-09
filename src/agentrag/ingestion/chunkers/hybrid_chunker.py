from __future__ import annotations

import hashlib
import re
from typing import Dict, List

# Page markers injected by PDFParser: \x00P{page_num}\x00
# Resolved here so page_start/page_end is set on each chunk without any
# structural changes to the heading/token chunking logic.
_PAGE_MARKER_RE = re.compile(r"\x00P(\d+)\x00")

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
        split_on_paragraphs: bool = False,
    ):
        self.max_tokens = max_tokens
        self.overlap_tokens = overlap_tokens
        self.split_on_headings = split_on_headings
        self.split_on_paragraphs = split_on_paragraphs
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
            section_chunks = self._chunk_section(section_text)
            for i, chunk_text in enumerate(section_chunks):
                # For chunks after the first in a section, prepend the section
                # heading so every chunk is self-contained (critical for Excel
                # sheets where only the first chunk would otherwise carry the
                # sheet name, making later row chunks context-free).
                if i > 0 and section_name not in ("intro", "section", "chunk_0"):
                    contextualized = f"{section_name}\n\n{chunk_text}"
                else:
                    contextualized = chunk_text
                chunks.append(
                    {
                        "content": contextualized,
                        "content_hash": hashlib.sha256(
                            contextualized.encode("utf-8")
                        ).hexdigest(),
                        "segment_type": "text",
                        "section_path": section_name,
                        "position": chunk_id,
                        "metadata": metadata or {},
                    }
                )
                chunk_id += 1

        self._resolve_page_numbers(chunks)
        return chunks

    @staticmethod
    def _resolve_page_numbers(chunks: list[Dict]) -> None:
        """Strip embedded page markers and assign page_start / page_end per chunk.

        Markers (\x00P{N}\x00) are injected by PDFParser. For non-PDF content
        (no markers) this is a no-op — page_start and page_end stay None.
        """
        last_page: int | None = None
        for chunk in chunks:
            content = chunk["content"]
            page_nums = [int(m) for m in _PAGE_MARKER_RE.findall(content)]
            if page_nums:
                chunk["page_start"] = page_nums[0]
                chunk["page_end"] = page_nums[-1]
                last_page = page_nums[-1]
            elif last_page is not None:
                # Chunk is entirely within the page containing the last marker
                chunk["page_start"] = last_page
                chunk["page_end"] = last_page
            else:
                chunk["page_start"] = None
                chunk["page_end"] = None

            # Strip markers from content and recompute hash
            if page_nums:
                clean = _PAGE_MARKER_RE.sub("", content).strip()
                chunk["content"] = clean
                chunk["content_hash"] = hashlib.sha256(
                    clean.encode("utf-8")
                ).hexdigest()

    def _chunk_section(self, section_text: str) -> list[str]:
        if self.split_on_paragraphs:
            return self._chunk_section_by_paragraph(section_text)
        return self._chunk_section_by_tokens(section_text)

    def _chunk_section_by_paragraph(self, section_text: str) -> list[str]:
        paragraphs = [
            paragraph.strip()
            for paragraph in re.split(r"\n\s*\n", section_text)
            if paragraph.strip()
        ]
        if not paragraphs:
            return []

        chunks: list[str] = []
        current_parts: list[str] = []
        current_tokens = 0

        for paragraph in paragraphs:
            paragraph_tokens = self.tokenizer.encode(paragraph)
            paragraph_len = len(paragraph_tokens)

            if paragraph_len > self.max_tokens:
                if current_parts:
                    chunks.append("\n\n".join(current_parts).strip())
                    current_parts = []
                    current_tokens = 0
                chunks.extend(self._chunk_section_by_tokens(paragraph))
                continue

            projected = current_tokens + paragraph_len
            if current_parts and projected > self.max_tokens:
                chunks.append("\n\n".join(current_parts).strip())
                current_parts = [paragraph]
                current_tokens = paragraph_len
                continue

            current_parts.append(paragraph)
            current_tokens = projected

        if current_parts:
            chunks.append("\n\n".join(current_parts).strip())

        return [chunk for chunk in chunks if chunk]

    def _chunk_section_by_tokens(self, section_text: str) -> list[str]:
        chunks: list[str] = []
        tokens = self.tokenizer.encode(section_text)
        position = 0

        while position < len(tokens):
            end = min(position + self.max_tokens, len(tokens))
            chunk_tokens = tokens[position:end]
            chunk_text = self.tokenizer.decode(chunk_tokens).strip()
            if not chunk_text:
                position = self._next_position(position, end)
                continue
            chunks.append(chunk_text)
            position = self._next_position(position, end)

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
        # Strip only control characters and null bytes; preserve Unicode (Vietnamese, CJK, etc.)
        normalized = re.sub(r"[\x00-\x1f\x7f\|]+", " ", heading).strip()
        return normalized[:200] or "section"

    def _next_position(self, current: int, end: int) -> int:
        if self.overlap_tokens <= 0:
            return end
        next_position = max(end - self.overlap_tokens, 0)
        if next_position <= current:
            return end
        return next_position
