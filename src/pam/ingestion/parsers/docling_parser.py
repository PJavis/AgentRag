# src/pam/ingestion/parsers/docling_parser.py
"""
DoclingParser: chuyển PDF / DOCX / Word thành markdown.

Nếu VISION_ENABLED=true và ImageDescriber được truyền vào,
các ảnh trong tài liệu sẽ được mô tả và chèn vào markdown:
  <!-- image_3: Biểu đồ tăng trưởng doanh thu Q1-Q4 2024... -->

Các mô tả này sẽ trở thành chunk với segment_type="image_description"
(được filter bởi pipeline).
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .image_describer import ImageDescriber

logger = logging.getLogger(__name__)


class DoclingParser:
    def __init__(self, image_describer: "ImageDescriber | None" = None):
        from docling.document_converter import DocumentConverter
        self.converter = DocumentConverter()
        self.image_describer = image_describer

    async def parse(self, file_path: str) -> dict[str, Any]:
        """
        Returns:
          {
            "parsed_content": str,    # markdown text (với image descriptions nếu vision bật)
            "structure": dict,
            "pages": int,
            "images_described": int,  # số ảnh được mô tả
          }
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        result = self.converter.convert(path)
        doc = result.document

        parsed_text = doc.export_to_markdown()

        structure = {
            "title": doc.title if hasattr(doc, "title") else path.stem,
            "num_pages": len(doc.pages) if hasattr(doc, "pages") else 1,
            "format": doc.format if hasattr(doc, "format") else "unknown",
        }

        images_described = 0
        if self.image_describer is not None:
            parsed_text, images_described = await self._inject_image_descriptions(
                doc, parsed_text
            )

        return {
            "parsed_content": parsed_text,
            "structure": structure,
            "pages": structure["num_pages"],
            "images_described": images_described,
        }

    async def _inject_image_descriptions(
        self, doc: Any, markdown_text: str
    ) -> tuple[str, int]:
        """
        Extract images từ document, gọi VLM mô tả, chèn vào cuối markdown.
        Trả về (updated_markdown, images_described_count).
        """
        pictures = []
        if hasattr(doc, "pictures"):
            pictures = list(doc.pictures)

        max_images = getattr(self.image_describer, "_max_images", 20)
        pictures = pictures[:max_images]

        if not pictures:
            return markdown_text, 0

        # Lấy bytes của từng ảnh
        image_bytes_list: list[bytes | None] = []
        for pic in pictures:
            try:
                img_bytes = _extract_picture_bytes(pic)
                image_bytes_list.append(img_bytes)
            except Exception as exc:
                logger.debug("Could not extract image bytes: %s", exc)
                image_bytes_list.append(None)

        # Lấy context text xung quanh ảnh (nếu có caption/label)
        contexts: list[str] = []
        for pic in pictures:
            ctx = ""
            if hasattr(pic, "caption") and pic.caption:
                ctx = str(pic.caption)
            elif hasattr(pic, "text") and pic.text:
                ctx = str(pic.text)
            contexts.append(ctx)

        # Gọi VLM song song cho các ảnh có bytes
        descriptions: list[str] = []
        valid_indices: list[int] = []
        valid_bytes: list[bytes] = []
        valid_contexts: list[str] = []

        for i, img_bytes in enumerate(image_bytes_list):
            if img_bytes is not None:
                valid_indices.append(i)
                valid_bytes.append(img_bytes)
                valid_contexts.append(contexts[i])

        if valid_bytes:
            raw_descriptions = await self.image_describer.describe_batch(
                valid_bytes, valid_contexts
            )
        else:
            raw_descriptions = []

        desc_map: dict[int, str] = {}
        for i, desc in zip(valid_indices, raw_descriptions):
            if desc:
                desc_map[i] = desc

        if not desc_map:
            return markdown_text, 0

        # Chèn mô tả vào cuối markdown dưới dạng chú thích
        image_section_lines = ["\n\n---\n\n## Image Descriptions\n"]
        for i, desc in sorted(desc_map.items()):
            # Format: HTML comment để không hiển thị nhưng vẫn được index
            image_section_lines.append(f"<!-- image_{i + 1}: {desc} -->")
            image_section_lines.append(f"\n**Image {i + 1}:** {desc}\n")

        updated = markdown_text + "\n".join(image_section_lines)
        return updated, len(desc_map)


def _extract_picture_bytes(picture: Any) -> bytes:
    """Lấy raw bytes của ảnh từ Docling PictureItem."""
    # Docling có thể lưu ảnh theo nhiều cách tùy version
    if hasattr(picture, "image") and picture.image is not None:
        img = picture.image
        if hasattr(img, "tobytes"):
            return img.tobytes()
        if hasattr(img, "getvalue"):
            return img.getvalue()
    if hasattr(picture, "data") and picture.data:
        if isinstance(picture.data, (bytes, bytearray)):
            return bytes(picture.data)
    raise ValueError("Cannot extract bytes from picture item")
