from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

from src.agentrag.config import settings

if TYPE_CHECKING:
    from src.agentrag.services.llm_gateway import LLMGateway

QueryIntent = Literal["semantic", "structured"]
QueryType = Literal["comparison", "aggregation", "ranking", "multi_filter", "multi_hop"]


@dataclass
class ClassifierOutput:
    intent: QueryIntent
    query_type: QueryType | None
    confidence: float
    reasoning: str
    method: Literal["rule", "llm", "default"]


_FLAG = re.IGNORECASE | re.UNICODE


class QueryIntentClassifier:
    """
    Phân loại câu hỏi thành 2 nhánh xử lý:
      - "semantic"    → nhánh RAG cũ (hybrid_kg + LLM answer)
      - "structured"  → nhánh SQL reasoning mới (ADR 0002)

    L1 (rule-based): nhanh, không cần LLM call.
    L2 (LLM-based):  fallback khi L1 không match và STRUCTURED_CLASSIFIER_METHOD != "rule".
    """

    # ── L1 compiled patterns ─────────────────────────────────────────────────

    _COMPARISON: list[re.Pattern] = [
        re.compile(p, _FLAG)
        for p in [
            r"\bso sánh\b",
            r"\bcompare\b",
            r"\bvs\.?\b",
            r"\bversus\b",
            r"\bkhác nhau\b",
            r"\bdiff(erence)?\b",
            r"\bgiống hay khác\b",
            r"\bcontrast\b",
        ]
    ]

    _AGGREGATION: list[re.Pattern] = [
        re.compile(p, _FLAG)
        for p in [
            r"\bbao nhiêu\b",
            r"\bhow many\b",
            r"\btổng\b",
            r"\btotal\b",
            r"\b\bsum\b\b",
            r"\btrung bình\b",
            r"\baverage\b",
            r"\bcount\b",
            r"\bsố lượng\b",
            r"\bpercentage\b",
            r"\bphần trăm\b",
            # Price × quantity calculation patterns
            r"\b\d+\s+\w[\w\s]{0,40}\bgiá\b",          # "10 mythic shard giá"
            r"\bgiá\b.{0,30}\b\d+\b",                   # "giá của 10 X"
            r"\b\d+\b.{0,20}\b(cost|price)\b",           # "10 ... cost/price"
            r"\bhow much.{0,30}\b\d+\b",                 # "how much do 10 X"
            r"\btính (giá|tiền|chi phí)\b",              # "tính giá / tính tiền"
            r"\b(giá|tiền|chi phí).{0,20}\btính\b",     # "giá ... tính"
        ]
    ]

    _RANKING: list[re.Pattern] = [
        re.compile(p, _FLAG)
        for p in [
            r"\btop\s*\d+\b",
            r"\bbest\b",
            r"\bworst\b",
            r"\bhighest\b",
            r"\blowest\b",
            r"\blớn nhất\b",
            r"\bnhỏ nhất\b",
            r"\btốt nhất\b",
            r"\bkém nhất\b",
            r"\branking\b",
            r"\bxếp hạng\b",
            r"\bmost\b",
            r"\bleast\b",
        ]
    ]

    _MULTI_FILTER: list[re.Pattern] = [
        re.compile(p, _FLAG)
        for p in [
            r"\btất cả\b.{0,40}\bvà\b",
            r"\ball\b.{0,40}\band\b",
            r"\bfind all\b",
            r"\btìm tất cả\b",
            r"\blist all\b",
            r"\bliệt kê tất cả\b",
        ]
    ]

    _MULTI_HOP: list[re.Pattern] = [
        re.compile(p, _FLAG)
        for p in [
            r"\bqua\b.{0,20}\bquan hệ\b",
            r"\bthrough\b",
            r"\bvia\b",
            r"\bchain\b",
            r"\bmulti.?hop\b",
            r"\bconnected to\b",
            r"\bliên kết\b.{0,20}\bvới\b",
        ]
    ]

    _PATTERN_MAP: list[tuple[QueryType, list[re.Pattern]]] = [
        ("comparison", _COMPARISON),
        ("aggregation", _AGGREGATION),
        ("ranking", _RANKING),
        ("multi_filter", _MULTI_FILTER),
        ("multi_hop", _MULTI_HOP),
    ]

    # ── Few-shot examples cho L2 ──────────────────────────────────────────────

    _L2_EXAMPLES = [
        {"question": "So sánh tính năng của A và B", "intent": "structured", "query_type": "comparison"},
        {"question": "Có bao nhiêu sản phẩm loại X?", "intent": "structured", "query_type": "aggregation"},
        {"question": "Top 5 công ty có doanh thu cao nhất", "intent": "structured", "query_type": "ranking"},
        {"question": "Tìm tất cả nhân viên có lương > 10M và là senior", "intent": "structured", "query_type": "multi_filter"},
        {"question": "10 mythic shard có giá bao nhiêu?", "intent": "structured", "query_type": "aggregation"},
        {"question": "Tính giá của 5 legendary pack", "intent": "structured", "query_type": "aggregation"},
        {"question": "How much do 20 epic trooper shards cost?", "intent": "structured", "query_type": "aggregation"},
        {"question": "Tính năng chính của sản phẩm là gì?", "intent": "semantic", "query_type": None},
        {"question": "Mô tả quy trình onboarding", "intent": "semantic", "query_type": None},
    ]

    def __init__(self, llm_gateway: LLMGateway | None = None) -> None:
        self._llm_gateway = llm_gateway

    async def classify(
        self,
        question: str,
        document_title: str | None = None,
        chat_history: list[dict[str, Any]] | None = None,
    ) -> ClassifierOutput:
        # L1 luôn chạy trước
        result = self._classify_l1(question)
        if result is not None:
            return result

        # L2: chỉ chạy nếu có llm_gateway và method cho phép
        method = settings.STRUCTURED_CLASSIFIER_METHOD
        if self._llm_gateway is not None and method in ("llm", "rule+llm"):
            return await self._classify_l2(question, chat_history)

        # Default: semantic
        return ClassifierOutput(
            intent="semantic",
            query_type=None,
            confidence=0.5,
            reasoning="No rule matched; defaulting to semantic path.",
            method="default",
        )

    def _classify_l1(self, question: str) -> ClassifierOutput | None:
        """
        Scan compiled regex patterns theo thứ tự ưu tiên.
        Trả None nếu không có pattern nào match.
        """
        for query_type, patterns in self._PATTERN_MAP:
            for pattern in patterns:
                if pattern.search(question):
                    return ClassifierOutput(
                        intent="structured",
                        query_type=query_type,
                        confidence=0.95,
                        reasoning=f"L1 rule matched pattern for query_type='{query_type}'",
                        method="rule",
                    )
        return None

    async def _classify_l2(
        self,
        question: str,
        chat_history: list[dict[str, Any]] | None = None,
    ) -> ClassifierOutput:
        """
        LLM fallback — được implement đầy đủ ở M6.
        Trả semantic default nếu llm_gateway chưa set.
        """
        if self._llm_gateway is None:
            return ClassifierOutput(
                intent="semantic",
                query_type=None,
                confidence=0.5,
                reasoning="L2 skipped: no llm_gateway configured.",
                method="default",
            )

        system_prompt = (
            "Classify the user question into one of two intents: 'semantic' or 'structured'.\n"
            "Structured intents require comparison, counting, aggregation, ranking, or multi-hop reasoning.\n"
            "Semantic intents are descriptive or explanatory questions.\n"
            "Return JSON: {\"intent\": str, \"query_type\": str|null, \"confidence\": float, \"reasoning\": str}\n"
            "query_type must be one of: comparison, aggregation, ranking, multi_filter, multi_hop, or null.\n\n"
            "Examples:\n"
            + json.dumps(self._L2_EXAMPLES, ensure_ascii=False, indent=2)
        )
        user_prompt = json.dumps({"question": question}, ensure_ascii=False)

        try:
            result, _latency = await self._llm_gateway.json_response(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                task="classify",
            )
            intent = result.get("intent", "semantic")
            if intent not in ("semantic", "structured"):
                intent = "semantic"
            query_type = result.get("query_type")
            if query_type not in ("comparison", "aggregation", "ranking", "multi_filter", "multi_hop", None):
                query_type = None
            return ClassifierOutput(
                intent=intent,
                query_type=query_type,
                confidence=float(result.get("confidence", 0.7)),
                reasoning=result.get("reasoning", ""),
                method="llm",
            )
        except Exception as exc:
            return ClassifierOutput(
                intent="semantic",
                query_type=None,
                confidence=0.5,
                reasoning=f"L2 LLM call failed: {exc}",
                method="default",
            )


if __name__ == "__main__":
    import asyncio

    classifier = QueryIntentClassifier()

    cases = [
        ("So sánh A và B theo tiêu chí X", "structured", "comparison"),
        ("Compare product A and B", "structured", "comparison"),
        ("Tính năng X là gì?", "semantic", None),
        ("What are the features?", "semantic", None),
        ("Top 3 sản phẩm bán chạy nhất", "structured", "ranking"),
        ("Bao nhiêu user đăng ký tháng này?", "structured", "aggregation"),
        ("How many records are there?", "structured", "aggregation"),
        ("Tìm tất cả đơn hàng và trả về danh sách", "structured", "multi_filter"),
        ("Find all items with status A and category B", "structured", "multi_filter"),
        ("Mô tả quy trình triển khai", "semantic", None),
    ]

    all_pass = True
    for question, expected_intent, expected_type in cases:
        out = asyncio.run(classifier.classify(question))
        ok = out.intent == expected_intent and (expected_type is None or out.query_type == expected_type)
        status = "OK" if ok else "FAIL"
        if not ok:
            all_pass = False
        print(f"[{status}] {question!r}")
        print(f"       → intent={out.intent} query_type={out.query_type} method={out.method}")

    print()
    print("[ALL PASS]" if all_pass else "[SOME FAILED]")
