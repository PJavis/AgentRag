"""Static taxonomy lookup endpoints for UI dropdowns (S5).

Keep in sync with the closed sets used by DomainRouter prompt + SectionTagger.
"""
from __future__ import annotations

from fastapi import APIRouter

router = APIRouter(prefix="/ontology")

_SYSTEMS: list[tuple[str, str]] = [
    ("tim_mach",       "Tim mạch"),
    ("ho_hap",         "Hô hấp"),
    ("tieu_hoa",       "Tiêu hóa"),
    ("than_kinh",      "Thần kinh"),
    ("noi_tiet",       "Nội tiết"),
    ("co_xuong_khop",  "Cơ - Xương - Khớp"),
    ("huyet_hoc",      "Huyết học"),
    ("tiet_nieu",      "Tiết niệu"),
    ("sinh_duc",       "Sinh dục"),
    ("da_lieu",        "Da liễu"),
    ("mat_tmh",        "Mắt - TMH"),
    ("tam_than",       "Tâm thần"),
    ("mien_dich",      "Miễn dịch / Dị ứng"),
    ("nhi_khoa",       "Nhi"),
    ("da_he",          "Đa hệ thống"),
]

_SPECIALTIES: list[tuple[str, str]] = [
    ("noi",                 "Nội"),
    ("ngoai",               "Ngoại"),
    ("san",                 "Sản phụ khoa"),
    ("nhi",                 "Nhi"),
    ("cap_cuu",             "Cấp cứu"),
    ("hoi_suc",             "Hồi sức tích cực"),
    ("truyen_nhiem",        "Truyền nhiễm"),
    ("ung_buou",            "Ung bướu"),
    ("chan_doan_hinh_anh",  "Chẩn đoán hình ảnh"),
    ("xet_nghiem",          "Xét nghiệm"),
    ("duoc_ly",             "Dược lý"),
    ("giai_phau",           "Giải phẫu"),
    ("sinh_ly_benh",        "Sinh lý bệnh"),
    ("general",             "Chung"),
]


@router.get("/systems")
async def list_systems() -> list[dict[str, str]]:
    return [{"value": v, "label": l} for v, l in _SYSTEMS]


@router.get("/specialties")
async def list_specialties() -> list[dict[str, str]]:
    return [{"value": v, "label": l} for v, l in _SPECIALTIES]
