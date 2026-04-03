from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from src.pam.agents.insight_agent import InsightResult
from src.pam.services.llm_gateway import LLMGateway


@dataclass
class ReportResult:
    title: str
    sections: list[dict[str, Any]]   # [{"heading": str, "content": str, "citations": list}]
    all_citations: list[dict[str, Any]]
    summary: str


_REPORT_SYSTEM = """\
You are a report composer. Given multiple insights from different sub-questions, \
compose a structured report.

Return JSON:
{
  "title": "report title",
  "summary": "executive summary paragraph",
  "sections": [
    {"heading": "section heading", "content": "section content"}
  ]
}

Use the same language as the insights. Be comprehensive but concise.
"""


class ReportAgent:
    """
    Specialized worker: compose multi-section report từ nhiều InsightResult.
    """

    def __init__(self, llm_gateway: LLMGateway) -> None:
        self._llm = llm_gateway

    async def run(self, insights: list[InsightResult], report_title: str = "Analysis Report") -> ReportResult:
        all_citations: list[dict[str, Any]] = []
        seen_hashes: set[str] = set()
        for insight in insights:
            for c in insight.citations:
                h = c.get("content_hash", "")
                if h and h not in seen_hashes:
                    seen_hashes.add(h)
                    all_citations.append(c)

        insights_payload = [
            {
                "question": ins.question,
                "insight": ins.insight,
                "key_findings": ins.key_findings,
            }
            for ins in insights
        ]

        user_prompt = json.dumps(
            {"report_title": report_title, "insights": insights_payload},
            ensure_ascii=False,
        )

        try:
            raw, _latency = await self._llm.json_response(
                system_prompt=_REPORT_SYSTEM,
                user_prompt=user_prompt,
                task="report",
            )
            sections_raw = raw.get("sections") or []
            sections = [
                {
                    "heading": str(s.get("heading", "")),
                    "content": str(s.get("content", "")),
                    "citations": [],
                }
                for s in sections_raw
            ]
            return ReportResult(
                title=str(raw.get("title", report_title)),
                sections=sections,
                all_citations=all_citations,
                summary=str(raw.get("summary", "")),
            )
        except Exception as exc:
            # Fallback: composite từ insights thô
            sections = [
                {"heading": ins.question, "content": ins.insight, "citations": ins.citations}
                for ins in insights
            ]
            return ReportResult(
                title=report_title,
                sections=sections,
                all_citations=all_citations,
                summary=f"Report generation encountered an error: {exc}",
            )
