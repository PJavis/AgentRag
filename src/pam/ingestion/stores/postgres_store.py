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

            # Kiểm tra document có tồn tại bằng content_hash + source_id
            stmt = select(Document).where(
                Document.source_id == doc_data["source_id"],
                Document.content_hash == doc_data["content_hash"]
            )
            result = await session.execute(stmt)
            existing_doc = result.scalar_one_or_none()

            if existing_doc:
                # Skip nếu hash giống (không thay đổi)
                return existing_doc.id, "skipped"

            # Xóa document cũ nếu có (re-ingest)
            if existing_doc:
                await session.execute(
                    delete(Segment).where(Segment.document_id == existing_doc.id)
                )
                await session.delete(existing_doc)
                await session.flush()

            # Tạo Document mới
            doc = Document(
                project_id=project_id,
                source_type=doc_data["source_type"],
                source_id=doc_data["source_id"],
                title=doc_data["title"],
                content_hash=doc_data["content_hash"],
                graph_synced=False,  # sẽ sync graph sau
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
                    extra_metadata=chunk["metadata"],  # dùng tên đã đổi
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
