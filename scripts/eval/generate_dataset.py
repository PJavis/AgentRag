"""
Tự động sinh golden QA dataset từ một tài liệu đã được ingest.

Usage:
  python scripts/eval/generate_dataset.py achievement-system
  python scripts/eval/generate_dataset.py achievement-system --n 20 --lang vi
  python scripts/eval/generate_dataset.py achievement-system --file data/test_docs/achievement-system.md
"""
from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.pam.config import settings
from src.pam.config_validation import validate_settings
from src.pam.eval.dataset import generate_golden_dataset
from src.pam.services.llm_gateway import LLMGateway


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate golden evaluation dataset from a document")
    p.add_argument("document_title", help="Title of the ingested document (e.g. achievement-system)")
    p.add_argument("--file", help="Path to source document file (to read content)")
    p.add_argument("--n", type=int, default=15, help="Number of questions to generate (default: 15)")
    p.add_argument("--out", help="Output JSON path (default: data/eval/<title>.json)")
    return p.parse_args()


async def main() -> None:
    args = parse_args()
    validate_settings(settings)

    # Tìm file nội dung
    content = ""
    if args.file:
        content = Path(args.file).read_text(encoding="utf-8")
    else:
        # Tìm trong data/ hoặc data/test_docs/
        for search_dir in ["data/test_docs", "data/docs", "data"]:
            candidates = list((ROOT / search_dir).glob(f"{args.document_title}*"))
            if candidates:
                content = candidates[0].read_text(encoding="utf-8")
                print(f"Found document: {candidates[0]}")
                break

    if not content:
        print(f"[ERROR] Could not find content for '{args.document_title}'.")
        print("  Use --file path/to/document.md to specify the source file.")
        sys.exit(1)

    out_path = Path(args.out) if args.out else ROOT / "data" / "eval" / f"{args.document_title}.json"

    print(f"Generating {args.n} golden questions for '{args.document_title}'...")
    print(f"  Content length: {len(content):,} chars")

    llm = LLMGateway()
    dataset = await generate_golden_dataset(
        document_title=args.document_title,
        document_content=content,
        llm_gateway=llm,
        n_questions=args.n,
    )

    dataset.save(out_path)
    print(f"\n✓ Generated {len(dataset.questions)} questions → {out_path}")

    # Preview
    print("\nSample questions:")
    for q in dataset.questions[:3]:
        print(f"  [{q.difficulty}] [{q.question_type}] {q.question}")
        print(f"    sections: {q.relevant_sections}")
        print(f"    keywords: {q.relevant_keywords}")


if __name__ == "__main__":
    asyncio.run(main())
