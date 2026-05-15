"""SLM-driven classifier: free-text query → set of medical domains.

Uses LLMGateway.json_response with task="domain_router" so users can
route this call to a cheap model via LLM_TASK_MODEL_MAP.

Threshold behaviour (settings):
  - confidence >= DOMAIN_ROUTER_CONFIDENCE_THRESHOLD → top-1 system
  - else → up to DOMAIN_ROUTER_TOP_K systems (broaden federation)
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field

from src.agentrag.config import settings
from src.agentrag.services.llm_gateway import LLMGateway


SYSTEM_PROMPT = """\
You are a medical domain router for Vietnamese medical content.
Given a query, identify which body system(s) and clinical specialty(s)
are most relevant. Vietnamese taxonomy:

systems: tim_mach, ho_hap, tieu_hoa, than_kinh, noi_tiet, co_xuong_khop,
huyet_hoc, tiet_nieu, sinh_duc, da_lieu, mat_tmh, tam_than, mien_dich,
nhi_khoa, da_he

specialties: noi, ngoai, san, nhi, cap_cuu, hoi_suc, truyen_nhiem,
ung_buou, chan_doan_hinh_anh, xet_nghiem, duoc_ly, giai_phau,
sinh_ly_benh, general

Return JSON exactly:
{
  "systems":    ["...up to 3 most relevant..."],
  "specialties":["...up to 3..."],
  "confidence": 0.0
}
Confidence ~1.0 = single clear domain; ~0.5 = multi-system / ambiguous.
Use ONLY taxonomy values above. Return ONLY JSON, no markdown fences.
"""


@dataclass
class DomainRoute:
    systems: list[str]
    specialties: list[str]
    confidence: float
    raw: dict = field(default_factory=dict)


class DomainRouter:
    def __init__(self) -> None:
        self._gateway = LLMGateway()

    async def classify(self, query: str) -> DomainRoute:
        try:
            payload, _ = await self._gateway.json_response(
                system_prompt=SYSTEM_PROMPT,
                user_prompt=json.dumps({"query": query}, ensure_ascii=False),
                task="domain_router",
            )
        except Exception:
            payload = {}
        if not isinstance(payload, dict):
            payload = {}
        raw_systems = [
            s for s in (payload.get("systems") or []) if isinstance(s, str)
        ]
        raw_specs = [
            s for s in (payload.get("specialties") or []) if isinstance(s, str)
        ]
        try:
            conf = float(payload.get("confidence", 0.0))
        except (TypeError, ValueError):
            conf = 0.0
        threshold = settings.DOMAIN_ROUTER_CONFIDENCE_THRESHOLD
        top_k = settings.DOMAIN_ROUTER_TOP_K
        if conf >= threshold and raw_systems:
            systems = raw_systems[:1]
        else:
            systems = raw_systems[:top_k]
        return DomainRoute(
            systems=systems,
            specialties=raw_specs[:top_k],
            confidence=conf,
            raw=payload,
        )
