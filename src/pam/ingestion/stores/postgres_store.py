# src/pam/ingestion/stores/postgres_store.py
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, delete
from sqlalchemy.exc import IntegrityError
from src.pam.database.models import Document, Segment, Project
from src.pam.database import AsyncSessionLocal  # tạo sau nếu chưa có
from typing import List, Dict
import uuid

class PostgresStore:
    async def save_document_and_segments(
        self,
        session: AsyncSession,
        doc_data: Dict,
        chunks: List[Dict],
        project_id: uuid.UUID = None,
    ):
        try:
            # Tìm hoặc tạo Project nếu cần (tạm dùng project mặc định hoặc tạo mới)
            if project_id is None:
                # Tạo project default nếu chưa có (cho MVP)
                project = Project(name="Default Project", description="Auto created")
                session.add(project)
                await session.flush()
                project_id = project.id

            # 1. Tìm tất cả documents có cùng source_id
            stmt = select(Document).where(Document.source_id == doc_data["source_id"])
            result = await session.execute(stmt)
            existing_docs = result.scalars().all()  # Dùng .all() thay vì .scalar_one_or_none()

            if existing_docs:
                # 2. Kiểm tra xem có bản ghi nào trùng khớp content_hash không
                for doc in existing_docs:
                    if doc.content_hash == doc_data["content_hash"]:
                        # Hash giống nhau -> Không có thay đổi -> Bỏ qua
                        return doc.id, "skipped"
                
                # 3. Hash khác nhau (hoặc toàn là rác) -> Xóa TẤT CẢ các bản ghi cũ để re-ingest
                for doc in existing_docs:
                    # Xóa segments liên kết
                    await session.execute(
                        delete(Segment).where(Segment.document_id == doc.id)
                    )
                    # Xóa document
                    await session.delete(doc)
                
                # Flush để đẩy lệnh xóa xuống DB ngay
                await session.flush()
            # Tạo Document mới
            doc = Document(
                project_id=project_id,
                source_type=doc_data["source_type"],
                source_id=doc_data["source_id"],
                title=doc_data["title"],
                content_hash=doc_data["content_hash"],
                graph_synced=False,
                graph_status="pending",
                graph_last_error=None,
            )
            session.add(doc)
            await session.flush()  # lấy doc.id

            # Tạo Segments
            for chunk in chunks:
                segment = Segment(
                    document_id=doc.id,
                    content=chunk["content"],
                    content_hash=chunk["content_hash"],
                    segment_type=chunk["segment_type"],
                    section_path=chunk["section_path"],
                    position=chunk["position"],
                    extra_metadata=chunk["metadata"],
                    version=1,
                )
                session.add(segment)

            await session.commit()
            return doc.id, "ingested"

        except IntegrityError as e:
            await session.rollback()
            raise ValueError(f"Database integrity error: {str(e)}")
        except Exception as e:
            await session.rollback()
            raise RuntimeError(f"Error saving to Postgres: {str(e)}")
