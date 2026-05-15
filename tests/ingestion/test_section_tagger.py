"""SectionTagger tests. Requires ontology_terms seeded."""
from __future__ import annotations

import pytest

from src.agentrag.ingestion.section_tagger import SectionTagger


@pytest.mark.asyncio
async def test_tagger_uses_section_path():
    tagger = SectionTagger()
    chunk = {
        "section_path": "Chương 3 / Hệ tim mạch / Tim",
        "content": "Tim gồm bốn buồng...",
    }
    out = await tagger.tag_chunk(chunk)
    assert out["system_tag"] == "tim_mach"
    assert "Tim" in out["canonical_terms"]


@pytest.mark.asyncio
async def test_tagger_generic_heading_falls_back_to_content():
    tagger = SectionTagger()
    chunk = {
        "section_path": "Tổng quan",
        "content": "Bệnh nhân được chẩn đoán nhồi máu cơ tim cấp.",
    }
    out = await tagger.tag_chunk(chunk)
    assert out["system_tag"] == "tim_mach"


@pytest.mark.asyncio
async def test_tagger_no_match_returns_none_tag():
    tagger = SectionTagger()
    chunk = {
        "section_path": "Chapter X / Section Y",
        "content": "Some unrelated English text about programming",
    }
    out = await tagger.tag_chunk(chunk)
    assert out["system_tag"] is None
    assert out["specialty_tag"] == []
    assert out["canonical_terms"] == []


@pytest.mark.asyncio
async def test_tagger_aggregates_specialties():
    tagger = SectionTagger()
    chunk = {
        "section_path": "Bệnh án / Nhồi máu cơ tim",
        "content": "...",
    }
    out = await tagger.tag_chunk(chunk)
    # NMCT entry: specialty_tags=[noi, cap_cuu]
    assert "noi" in out["specialty_tag"]
    assert "cap_cuu" in out["specialty_tag"]
