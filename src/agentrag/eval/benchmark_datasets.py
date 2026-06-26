"""HF benchmark dataset loaders → normalized eval examples.

Two sources, one shape:
  - VN  : sailor2/Vietnamese_RAG   configs BKAI_RAG, LegalRAG  (question/answer/context)
  - EN  : galileo-ai/ragbench      configs covidqa, pubmedqa   (question/documents/response)

All normalize to `EvalExample`. `gold_contexts` are the dataset's reference
passages — ingested into our index so our retrieval has something to find, and
fed to DeepEval as the recall/precision reference.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterator


@dataclass
class EvalExample:
    id: str
    question: str
    reference_answer: str
    gold_contexts: list[str] = field(default_factory=list)
    lang: str = "en"
    source: str = ""


# Registry: name → (hf_path, config, lang, field-mapping kind)
DATASETS: dict[str, dict[str, Any]] = {
    "vn_bkai":    {"path": "sailor2/Vietnamese_RAG", "config": "BKAI_RAG",  "lang": "vi", "kind": "qac"},
    "vn_legal":   {"path": "sailor2/Vietnamese_RAG", "config": "LegalRAG",  "lang": "vi", "kind": "qac"},
    "en_covidqa": {"path": "galileo-ai/ragbench",    "config": "covidqa",   "lang": "en", "kind": "ragbench"},
    "en_pubmedqa":{"path": "galileo-ai/ragbench",    "config": "pubmedqa",  "lang": "en", "kind": "ragbench"},
}

# Default suites per the medical-focused, small-scale plan.
SUITES: dict[str, list[str]] = {
    "vn": ["vn_bkai", "vn_legal"],
    "en": ["en_covidqa", "en_pubmedqa"],
    "both": ["vn_bkai", "vn_legal", "en_covidqa", "en_pubmedqa"],
}


def _as_context_list(value: Any) -> list[str]:
    """Coerce a `context`/`documents` field into a list of non-empty strings."""
    if value is None:
        return []
    if isinstance(value, str):
        return [value] if value.strip() else []
    if isinstance(value, (list, tuple)):
        out: list[str] = []
        for v in value:
            if isinstance(v, str) and v.strip():
                out.append(v)
            elif isinstance(v, dict):
                t = v.get("text") or v.get("content") or ""
                if t.strip():
                    out.append(t)
        return out
    return []


def normalize_row(row: dict[str, Any], *, kind: str, lang: str, source: str, idx: int) -> EvalExample:
    """Map one raw dataset row → EvalExample. Pure; unit-tested."""
    if kind == "qac":  # sailor2 BKAI_RAG / LegalRAG
        return EvalExample(
            id=f"{source}-{idx}",
            question=str(row.get("question", "")).strip(),
            reference_answer=str(row.get("answer", "")).strip(),
            gold_contexts=_as_context_list(row.get("context")),
            lang=lang,
            source=source,
        )
    if kind == "ragbench":  # galileo-ai/ragbench
        return EvalExample(
            id=str(row.get("id") or f"{source}-{idx}"),
            question=str(row.get("question", "")).strip(),
            reference_answer=str(row.get("response", "")).strip(),
            gold_contexts=_as_context_list(row.get("documents")),
            lang=lang,
            source=source,
        )
    raise ValueError(f"unknown dataset kind: {kind}")


def load_eval_examples(name: str, n: int = 30) -> list[EvalExample]:
    """Stream the first `n` valid rows of a registered dataset → EvalExamples.

    Streaming avoids downloading the (large) full dataset. Rows missing a
    question or all contexts are skipped (they can't be scored).
    """
    if name not in DATASETS:
        raise ValueError(f"unknown dataset {name!r}; choices: {sorted(DATASETS)}")
    spec = DATASETS[name]
    from datasets import load_dataset

    stream: Iterator[dict[str, Any]] = load_dataset(
        spec["path"], spec["config"], split="train", streaming=True
    )
    out: list[EvalExample] = []
    for idx, row in enumerate(stream):
        ex = normalize_row(row, kind=spec["kind"], lang=spec["lang"], source=name, idx=idx)
        if not ex.question or not ex.gold_contexts:
            continue
        out.append(ex)
        if len(out) >= n:
            break
    return out


def load_suite(suite: str, n: int = 30) -> list[EvalExample]:
    """Load `n` examples from each dataset in a named suite (vn|en|both)."""
    if suite not in SUITES:
        raise ValueError(f"unknown suite {suite!r}; choices: {sorted(SUITES)}")
    examples: list[EvalExample] = []
    for name in SUITES[suite]:
        examples.extend(load_eval_examples(name, n=n))
    return examples


def load_local_jsonl(path: str, n: int | None = None) -> list[EvalExample]:
    """Load EvalExamples from a local JSONL file (prod-corpus eval set).

    Each line is a JSON object with at least `question` and `gold_contexts`.
    Rows missing either are skipped (they can't be scored). `n` caps the count.
    """
    import json as _json

    out: list[EvalExample] = []
    with open(path, encoding="utf-8") as f:
        for idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                row = _json.loads(line)
            except _json.JSONDecodeError:
                continue
            question = str(row.get("question", "")).strip()
            contexts = _as_context_list(row.get("gold_contexts"))
            if not question or not contexts:
                continue
            out.append(EvalExample(
                id=str(row.get("id") or f"local-{idx}"),
                question=question,
                reference_answer=str(row.get("reference_answer", "")).strip(),
                gold_contexts=contexts,
                lang=str(row.get("lang", "en")),
                source=str(row.get("source", "local")),
            ))
            if n is not None and len(out) >= n:
                break
    return out
