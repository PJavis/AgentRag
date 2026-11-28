# src/pam/ingestion/stores/postgres_store.py
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, delete
from sqlalchemy.exc import IntegrityError
from src.agentrag.database.models import Document, Segment, Project
from src.agentrag.database import AsyncSessionLocal  # tạo sau nếu chưa có
from typing import List, Dict
import uuid

class PostgresStore:
    @staticmethod
    async def _has_segments(session: AsyncSession, document_id) -> bool:
        """A skeleton has none yet; an in-flight run past its save has some."""
        found = await session.execute(
            select(Segment.id).where(Segment.document_id == document_id).limit(1)
        )
        return found.scalar_one_or_none() is not None

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
            # In-flight rows first: a matching "done" row must not short-circuit
            # to "skipped" while an upload skeleton under this source_id waits.
            existing_docs = sorted(
                existing_docs,
                key=lambda d: getattr(d, "graph_status", None) not in ("pending", "parsing"),
            )

            if existing_docs:
                # 2. Kiểm tra xem có bản ghi nào trùng khớp content_hash không
                for doc in existing_docs:
                    # A pending/parsing row with no segments under this
                    # source_id is the upload skeleton: populate it whatever
                    # its hash says. Its hash is computed in the api process;
                    # if the api's parse settings differ from the worker's, a
                    # mismatch used to delete the skeleton and mint a new id
                    # (dropping its user_id). A row WITH segments is another
                    # run in flight: unchanged behaviour (hash decides).
                    if doc.content_hash == doc_data["content_hash"] or (
                        doc.graph_status in ("pending", "parsing")
                        and not await self._has_segments(session, doc.id)
                    ):
                        # Skeleton (pre-created via /sources upload) → populate it
                        # instead of creating a new doc. "parsing" = a skeleton the
                        # pipeline reports progress on (pipeline._mark_ingest_progress).
                        # A pending/parsing row only exists while an ingest is in
                        # flight, so one that already HAS segments is a run cut off
                        # after this save (ARQ timeout) — repopulate it too. It used
                        # to answer "skipped": no ES index, no failure recorded, the
                        # row stuck in pending/parsing for good.
                        is_skeleton = doc.graph_status in ("pending", "parsing")
                        if doc.graph_status == "failed" or is_skeleton:
                            # Reset + (re)populate. Clear any stale segments first.
                            await session.execute(
                                delete(Segment).where(Segment.document_id == doc.id)
                            )
                            doc.title = doc_data.get("title") or doc.title
                            doc.source_type = doc_data.get("source_type") or doc.source_type
                            doc.content_hash = doc_data["content_hash"]
                            doc.source_bytes_sha = doc_data.get("source_bytes_sha") or getattr(doc, "source_bytes_sha", None)
                            doc.graph_status = "pending"
                            doc.graph_synced = False
                            doc.graph_last_error = None
                            doc.graph_failed_chunks = 0
                            await session.flush()
                            for chunk in chunks:
                                session.add(Segment(
                                    document_id=doc.id,
                                    content=chunk["content"],
                                    content_hash=chunk["content_hash"],
                                    segment_type=chunk["segment_type"],
                                    section_path=chunk["section_path"],
                                    position=chunk["position"],
                                    extra_metadata=chunk["metadata"],
                                    version=1,
                                ))
                            await session.commit()
                            return doc.id, "ingested"
                        # Hash same, extraction already done or in-progress — skip
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
                source_bytes_sha=doc_data.get("source_bytes_sha"),
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
