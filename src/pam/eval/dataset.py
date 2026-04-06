"""
Golden dataset: cấu trúc dữ liệu và generator tự động từ tài liệu.

Schema golden dataset (data/eval/<doc_title>.json):
{
  "document_title": "achievement-system",
  "generated_at": "2026-04-06T...",
  "questions": [
    {
      "id": "q001",
      "question": "What are the main features?",
      "question_type": "factual|comparison|aggregation|multi_hop",
      "difficulty": "easy|medium|hard",
      "expected_answer": "The system provides: ...",
      "relevant_sections": ["overview"],        # section_path phải có trong top-K
      "relevant_keywords": ["progress tracking", "reward"],
      "language": "en"
    }
  ]
}
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

logger = logging.getLogger(__name__)

QuestionType = Literal["factual", "comparison", "aggregation", "multi_hop", "procedural"]
Difficulty   = Literal["easy", "medium", "hard"]


@dataclass
class GoldenQuestion:
    id: str
    question: str
    question_type: QuestionType
    difficulty: Difficulty
    expected_answer: str
    relevant_sections: list[str]        # section_path values that MUST appear in retrieval
    relevant_keywords: list[str]        # keywords that MUST appear in retrieved content
    language: str = "en"
    notes: str = ""


@dataclass
class GoldenDataset:
    document_title: str
    generated_at: str
    questions: list[GoldenQuestion] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "document_title": self.document_title,
            "generated_at": self.generated_at,
            "questions": [
                {
                    "id": q.id,
                    "question": q.question,
                    "question_type": q.question_type,
                    "difficulty": q.difficulty,
                    "expected_answer": q.expected_answer,
                    "relevant_sections": q.relevant_sections,
                    "relevant_keywords": q.relevant_keywords,
                    "language": q.language,
                    "notes": q.notes,
                }
                for q in self.questions
            ],
        }

    @classmethod
    def from_dict(cls, data: dict) -> "GoldenDataset":
        questions = [
            GoldenQuestion(
                id=q["id"],
                question=q["question"],
                question_type=q["question_type"],
                difficulty=q["difficulty"],
                expected_answer=q["expected_answer"],
                relevant_sections=q.get("relevant_sections", []),
                relevant_keywords=q.get("relevant_keywords", []),
                language=q.get("language", "en"),
                notes=q.get("notes", ""),
            )
            for q in data.get("questions", [])
        ]
        return cls(
            document_title=data["document_title"],
            generated_at=data.get("generated_at", ""),
            questions=questions,
        )

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
        logger.info("Saved %d questions to %s", len(self.questions), path)

    @classmethod
    def load(cls, path: Path) -> "GoldenDataset":
        return cls.from_dict(json.loads(path.read_text(encoding="utf-8")))


# ── Generator ────────────────────────────────────────────────────────────────

_GENERATE_SYSTEM = """You are an expert at creating evaluation datasets for RAG (retrieval-augmented generation) systems.
Given document content, generate diverse test questions that cover different aspects and difficulty levels.
Return a JSON array of question objects."""

_GENERATE_USER_TEMPLATE = """Document title: {title}

Document content (excerpt):
{content}

Generate {n_questions} test questions for evaluating a RAG system on this document.
Cover these question types: factual, comparison, aggregation, procedural.
Mix difficulty levels: easy (30%), medium (50%), hard (20%).

For each question, provide:
- question: the question text (write some in Vietnamese, some in English to test cross-lingual retrieval)
- question_type: factual | comparison | aggregation | multi_hop | procedural
- difficulty: easy | medium | hard
- expected_answer: a concise correct answer based on the document
- relevant_sections: list of section names where the answer can be found (use simple lowercase names)
- relevant_keywords: 2-4 keywords that must appear in relevant retrieved chunks
- language: "en" or "vi"

Return ONLY a JSON array, no other text:
[
  {{
    "question": "...",
    "question_type": "...",
    "difficulty": "...",
    "expected_answer": "...",
    "relevant_sections": ["section_name"],
    "relevant_keywords": ["keyword1", "keyword2"],
    "language": "en"
  }},
  ...
]"""


async def generate_golden_dataset(
    document_title: str,
    document_content: str,
    llm_gateway,
    n_questions: int = 15,
    content_preview_chars: int = 8000,
) -> GoldenDataset:
    """
    Dùng LLM để tự động sinh golden QA pairs từ nội dung tài liệu.
    """
    content_preview = document_content[:content_preview_chars]

    user_prompt = _GENERATE_USER_TEMPLATE.format(
        title=document_title,
        content=content_preview,
        n_questions=n_questions,
    )

    raw, _latency = await llm_gateway.json_response(
        system_prompt=_GENERATE_SYSTEM,
        user_prompt=user_prompt,
        task="eval_generate",
    )

    # raw có thể là list hoặc dict với key "questions"
    if isinstance(raw, list):
        raw_questions = raw
    elif isinstance(raw, dict):
        raw_questions = raw.get("questions") or raw.get("items") or []
    else:
        raw_questions = []

    questions: list[GoldenQuestion] = []
    for i, item in enumerate(raw_questions):
        if not isinstance(item, dict) or not item.get("question"):
            continue
        questions.append(
            GoldenQuestion(
                id=f"q{i + 1:03d}",
                question=item["question"],
                question_type=item.get("question_type", "factual"),
                difficulty=item.get("difficulty", "medium"),
                expected_answer=item.get("expected_answer", ""),
                relevant_sections=item.get("relevant_sections", []),
                relevant_keywords=item.get("relevant_keywords", []),
                language=item.get("language", "en"),
            )
        )

    return GoldenDataset(
        document_title=document_title,
        generated_at=datetime.now(timezone.utc).isoformat(),
        questions=questions,
    )
