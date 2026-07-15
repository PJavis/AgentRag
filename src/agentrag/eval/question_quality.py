"""Reject context-dependent / meta eval questions before they enter an eval set.

The 2026-07-14 home run surfaced that ~6/40 synthetic questions were unanswerable
by ANY retriever — they reference context the standalone question does not carry:
dangling demonstratives ("bệnh nhân **này**", "môn học **này**") and meta-references
to the source artifact ("câu 8 đến câu 12 trong **đề thi**", "trong **đoạn văn**").
On these, oracle scores ~1.0 (it is handed the gold context) while the system scores
0 (nothing to retrieve on) — inflating oracle−system with a fake "headroom" that no
graph/retrieval work can close. `build_prod_evalset.py` filters them so the measured
headroom reflects real system capability. Pure functions; no I/O."""
from __future__ import annotations

import re

# Meta-references to the eval artifact itself — a standalone question that talks
# about "the exam"/"the passage"/"question N" has no medical anchor to retrieve on.
_META_PATTERNS = [
    (re.compile(r"\bđề thi\b", re.I), "meta-reference: 'đề thi' (the exam)"),
    (re.compile(r"\bđoạn văn\b", re.I), "meta-reference: 'đoạn văn' (the passage)"),
    (re.compile(r"\bđoạn trích\b", re.I), "meta-reference: 'đoạn trích' (the excerpt)"),
    (re.compile(r"\bcâu hỏi số\b", re.I), "meta-reference: 'câu hỏi số N'"),
    (re.compile(r"\bcâu\s+\d+\s+(đến|tới)\s+câu?\s*\d+", re.I), "meta-reference: 'câu N đến câu M'"),
    # Bare "câu N" is an exam-item reference (e.g. "Câu 6 hỏi gì?", "trong câu 12")
    # — a standalone medical question never numbers itself. High precision.
    (re.compile(r"\bcâu\s+\d+\b", re.I), "meta-reference: 'câu N' (exam item)"),
    (re.compile(r"\btình huống (trên|này|đó|sau)\b", re.I), "meta-reference: 'tình huống trên/này' (the scenario)"),
    (re.compile(r"ngữ cảnh (được cung cấp|hiện có|sau)", re.I), "meta-reference: 'ngữ cảnh được cung cấp'"),
    (re.compile(r"\btrong ngữ cảnh\b", re.I), "meta-reference: 'trong ngữ cảnh'"),
]

# Dangling demonstrative: a noun that carries an antecedent from OUTSIDE the
# question, modified by này/đó/trên/kia. The noun list is limited to the ones the
# synthetic builder produces from case/exam/document chunks, so real disease
# phrases ("trong bệnh gan") are not touched.
_DANGLING = re.compile(
    r"\b(bệnh nhân|người bệnh|môn học|tài liệu|đoạn văn|hình ảnh|trường hợp|"
    r"ca bệnh|bài|chương|đề bài|văn bản|hình|bảng|biểu đồ|tình huống|kịch bản)\s+(này|đó|trên|kia)\b",
    re.I,
)


def is_context_dependent(question: str | None) -> tuple[bool, str]:
    """(is_bad, reason). is_bad=True → drop this question from the eval set."""
    q = (question or "").strip()
    if len(q) < 8:
        return True, "empty or too short"
    for pat, reason in _META_PATTERNS:
        if pat.search(q):
            return True, reason
    m = _DANGLING.search(q)
    if m:
        return True, f"dangling demonstrative: '{m.group(0)}'"
    return False, ""
