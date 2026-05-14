"""Transformations: user-defined prompts that run against text input.

Open-notebook UI uses these to generate summaries, key points, study notes,
etc. We persist transformations in `adapter_transformations` and execute via
the LLMGateway.
"""
from __future__ import annotations

import uuid
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from sqlalchemy import select

from src.agentrag.adapter.db import AdapterTransformation
from src.agentrag.database import AsyncSessionLocal
from src.agentrag.services.llm_gateway import LLMGateway

router = APIRouter(prefix="/transformations")


# ── Defaults ──────────────────────────────────────────────────────────────────
# Seeded once on first request so the UI always has something to pick.

_DEFAULT_TRANSFORMATIONS = [
    {
        "name": "summary",
        "title": "Tóm tắt",
        "description": "Tóm tắt toàn bộ nội dung trong 5-8 câu",
        "prompt": (
            "Hãy tóm tắt nội dung sau bằng tiếng Việt trong 5-8 câu, "
            "giữ nguyên các thuật ngữ y khoa quan trọng. "
            "Trả về một đoạn văn tóm tắt rõ ràng, súc tích."
        ),
        "apply_default": True,
    },
    {
        "name": "key_points",
        "title": "Điểm chính",
        "description": "Trích xuất 5-10 điểm quan trọng nhất dạng bullet",
        "prompt": (
            "Hãy trích xuất 5-10 điểm quan trọng nhất từ nội dung sau, "
            "viết dạng bullet (-), mỗi điểm 1-2 câu, ưu tiên thông tin lâm sàng, "
            "thuật ngữ chuyên môn, công thức, quy trình. Trả về thẳng dạng markdown."
        ),
        "apply_default": False,
    },
    {
        "name": "study_note",
        "title": "Ghi chú học tập",
        "description": "Cấu trúc ghi chú học tập y khoa",
        "prompt": (
            "Tạo ghi chú học tập y khoa từ nội dung dưới đây. "
            "Cấu trúc: Định nghĩa, Sinh lý bệnh, Triệu chứng, Chẩn đoán, Điều trị. "
            "Mỗi mục viết ngắn gọn, dùng bullet, in đậm thuật ngữ quan trọng. "
            "Trả về markdown."
        ),
        "apply_default": False,
    },
    {
        "name": "questions",
        "title": "Câu hỏi ôn tập",
        "description": "Sinh 5 câu hỏi ôn tập kèm gợi ý đáp án",
        "prompt": (
            "Sinh 5 câu hỏi ôn tập đa dạng (định nghĩa, lâm sàng, so sánh) "
            "từ nội dung sau. Mỗi câu hỏi đi kèm gợi ý đáp án ngắn. "
            "Định dạng markdown."
        ),
        "apply_default": False,
    },
]


async def _ensure_seeded():
    async with AsyncSessionLocal() as session:
        existing = (await session.execute(select(AdapterTransformation))).scalars().all()
        if existing:
            return
        for d in _DEFAULT_TRANSFORMATIONS:
            session.add(AdapterTransformation(**d))
        await session.commit()


def _fmt(t: AdapterTransformation) -> dict[str, Any]:
    return {
        "id": str(t.id),
        "name": t.name,
        "title": t.title,
        "description": t.description,
        "prompt": t.prompt,
        "apply_default": bool(t.apply_default),
        "created": t.created_at.isoformat() if t.created_at else "",
        "updated": t.updated_at.isoformat() if t.updated_at else "",
    }


# ── Request bodies ────────────────────────────────────────────────────────────


class TransformationCreate(BaseModel):
    name: str
    title: str | None = None
    description: str | None = None
    prompt: str
    apply_default: bool = False


class TransformationUpdate(BaseModel):
    name: str | None = None
    title: str | None = None
    description: str | None = None
    prompt: str | None = None
    apply_default: bool | None = None


class TransformationExecuteRequest(BaseModel):
    transformation_id: str | None = None
    name: str | None = None
    input_text: str | None = None
    source_id: str | None = None         # auto-fetch full text from Document segments
    model_id: str | None = None


# ── Routes ────────────────────────────────────────────────────────────────────


@router.get("")
async def list_transformations():
    await _ensure_seeded()
    async with AsyncSessionLocal() as session:
        rows = (
            await session.execute(
                select(AdapterTransformation).order_by(AdapterTransformation.name.asc())
            )
        ).scalars().all()
        return [_fmt(t) for t in rows]


@router.get("/default-prompt")
async def get_default_prompt():
    return {
        "transformation_instructions": (
            "Dùng tiếng Việt nếu nội dung là tiếng Việt; giữ thuật ngữ y khoa nguyên gốc; "
            "trả lời ngắn gọn, có cấu trúc rõ ràng."
        )
    }


@router.put("/default-prompt")
async def update_default_prompt(body: dict):
    return {"transformation_instructions": body.get("transformation_instructions", "")}


async def _load_source_text(source_id: str) -> str:
    """Fetch all segments of a Document and concat into a single input blob."""
    from sqlalchemy import select as _sel
    from src.agentrag.database.models import Segment

    raw = source_id.removeprefix("source:")
    try:
        sid = uuid.UUID(raw)
    except (ValueError, TypeError):
        raise HTTPException(400, "Invalid source_id")
    async with AsyncSessionLocal() as session:
        result = await session.execute(
            _sel(Segment.content, Segment.section_path, Segment.position)
            .where(Segment.document_id == sid)
            .order_by(Segment.position.asc())
        )
        rows = result.all()
    if not rows:
        raise HTTPException(404, "Source has no indexed segments yet")
    parts: list[str] = []
    for content, section_path, _pos in rows:
        if section_path:
            parts.append(f"[{section_path}]\n{content}")
        else:
            parts.append(content)
    return "\n\n---\n\n".join(parts)


@router.post("/execute")
async def execute_transformation(body: TransformationExecuteRequest):
    # Resolve input_text: explicit OR fetched from source_id
    input_text = body.input_text
    if not input_text and body.source_id:
        input_text = await _load_source_text(body.source_id)
    if not input_text:
        raise HTTPException(400, "input_text or source_id is required")

    transformation: AdapterTransformation | None = None
    async with AsyncSessionLocal() as session:
        if body.transformation_id:
            try:
                tid = uuid.UUID(body.transformation_id)
            except (ValueError, TypeError):
                raise HTTPException(400, "Invalid transformation_id")
            transformation = await session.get(AdapterTransformation, tid)
        elif body.name:
            row = await session.execute(
                select(AdapterTransformation).where(AdapterTransformation.name == body.name)
            )
            transformation = row.scalar_one_or_none()
    if transformation is None:
        raise HTTPException(404, "Transformation not found")

    llm = LLMGateway()
    text = await llm.text_response(
        system_prompt=transformation.prompt,
        user_prompt=input_text,
        task="transformation",
    ) if hasattr(llm, "text_response") else None

    if text is None:
        # Fallback: ask for JSON {"output": "..."}
        data, _ = await llm.json_response(
            system_prompt=transformation.prompt
            + "\n\nTrả về một JSON object chính xác có dạng: {\"output\": <kết quả>}.",
            user_prompt=input_text,
            task="transformation",
        )
        text = (data.get("output") or "").strip() if isinstance(data, dict) else str(data)

    # Optionally persist as a SourceInsight if source-derived
    if body.source_id:
        from src.agentrag.adapter.db import AdapterSourceInsight
        raw = body.source_id.removeprefix("source:")
        try:
            sid = uuid.UUID(raw)
            async with AsyncSessionLocal() as session:
                session.add(AdapterSourceInsight(
                    source_id=sid,
                    insight_type=transformation.name,
                    content=text,
                ))
                await session.commit()
        except (ValueError, TypeError):
            pass

    return {
        "output": text,
        "transformation_id": str(transformation.id),
        "model_id": body.model_id,
    }


@router.post("")
async def create_transformation(body: TransformationCreate):
    async with AsyncSessionLocal() as session:
        t = AdapterTransformation(
            name=body.name,
            title=body.title or body.name,
            description=body.description,
            prompt=body.prompt,
            apply_default=body.apply_default,
        )
        session.add(t)
        await session.commit()
        await session.refresh(t)
        return _fmt(t)


@router.get("/{transformation_id}")
async def get_transformation(transformation_id: str):
    async with AsyncSessionLocal() as session:
        try:
            tid = uuid.UUID(transformation_id)
        except (ValueError, TypeError):
            raise HTTPException(400, "Invalid id")
        t = await session.get(AdapterTransformation, tid)
        if not t:
            raise HTTPException(404, "Transformation not found")
        return _fmt(t)


@router.put("/{transformation_id}")
async def update_transformation(transformation_id: str, body: TransformationUpdate):
    async with AsyncSessionLocal() as session:
        t = await session.get(AdapterTransformation, uuid.UUID(transformation_id))
        if not t:
            raise HTTPException(404, "Transformation not found")
        for field in ("name", "title", "description", "prompt", "apply_default"):
            v = getattr(body, field)
            if v is not None:
                setattr(t, field, v)
        await session.commit()
        await session.refresh(t)
        return _fmt(t)


@router.delete("/{transformation_id}")
async def delete_transformation(transformation_id: str):
    async with AsyncSessionLocal() as session:
        t = await session.get(AdapterTransformation, uuid.UUID(transformation_id))
        if not t:
            raise HTTPException(404, "Transformation not found")
        await session.delete(t)
        await session.commit()
        return {"message": "Deleted"}
