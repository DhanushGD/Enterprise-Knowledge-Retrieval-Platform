"""Postgres schema and the registry that tracks every document and chunk we have seen."""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict
from datetime import datetime, timezone
from functools import lru_cache

from sqlalchemy import (
    BigInteger, Boolean, DateTime, Integer, String, Text, create_engine, delete, false, func, select,
    text, update,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker

from core.errors import SyncAlreadyRunning  # noqa: F401  (re-exported: existing imports keep working)
from core.settings import settings
from ingestion.records import DocRecord


def _now() -> datetime:
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    pass


class Document(Base):
    __tablename__ = "documents"

    doc_id: Mapped[str] = mapped_column(String(32), primary_key=True)
    source_name: Mapped[str] = mapped_column(String(100), index=True)
    source_path: Mapped[str] = mapped_column(Text)
    s3_key: Mapped[str] = mapped_column(Text)
    content_hash: Mapped[str] = mapped_column(String(64))
    size_bytes: Mapped[int] = mapped_column(BigInteger)
    etag: Mapped[str | None] = mapped_column(String(100), nullable=True)
    last_modified: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    department: Mapped[str] = mapped_column(String(50), index=True)
    access_level: Mapped[str] = mapped_column(String(50), index=True)
    product: Mapped[str] = mapped_column(String(50))
    status: Mapped[str] = mapped_column(String(20), index=True, default="new")
    version: Mapped[int] = mapped_column(Integer, default=1)
    first_seen_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now, onupdate=_now)


class ChunkRow(Base):
    """One retrievable passage. Metadata is copied from the document so retrieval
    can filter on it without a join."""

    __tablename__ = "chunks"

    chunk_id: Mapped[str] = mapped_column(String(48), primary_key=True)   # "<doc_id>-0003"
    doc_id: Mapped[str] = mapped_column(String(32), index=True)
    doc_version: Mapped[int] = mapped_column(Integer)
    chunk_index: Mapped[int] = mapped_column(Integer)
    doc_title: Mapped[str] = mapped_column(Text)
    source_path: Mapped[str] = mapped_column(Text)
    section: Mapped[str] = mapped_column(Text)
    text: Mapped[str] = mapped_column(Text)
    token_count: Mapped[int] = mapped_column(Integer)
    page: Mapped[int | None] = mapped_column(Integer, nullable=True)
    department: Mapped[str] = mapped_column(String(50))
    access_level: Mapped[str] = mapped_column(String(50))
    product: Mapped[str] = mapped_column(String(50))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)


class SyncRun(Base):
    """One row per ingestion run: an audit trail you can show in the README."""

    __tablename__ = "sync_runs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    source_name: Mapped[str] = mapped_column(String(100))
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    finished_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    new_count: Mapped[int] = mapped_column(Integer, default=0)
    changed_count: Mapped[int] = mapped_column(Integer, default=0)
    unchanged_count: Mapped[int] = mapped_column(Integer, default=0)
    failed_count: Mapped[int] = mapped_column(Integer, default=0)
    deleted_count: Mapped[int] = mapped_column(Integer, default=0, server_default="0")


class User(Base):
    """An API user. Only a salted scrypt hash is stored, never the password."""

    __tablename__ = "users"

    username: Mapped[str] = mapped_column(String(64), primary_key=True)
    password_hash: Mapped[str] = mapped_column(Text)
    role: Mapped[str] = mapped_column(String(50))
    disabled: Mapped[bool] = mapped_column(Boolean, default=False, server_default=false())
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)


@lru_cache
def get_engine():
    return create_engine(settings.postgres_dsn, pool_pre_ping=True)




def init_db(engine=None) -> None:
    """Create tables if they do not exist and apply small additive migrations.
    Safe to run repeatedly. (Alembic is the right tool once the schema churns more.)"""
    engine = engine or get_engine()
    Base.metadata.create_all(engine)
    if engine.dialect.name == "postgresql":
        with engine.begin() as conn:
            conn.execute(text(
                "ALTER TABLE sync_runs ADD COLUMN IF NOT EXISTS deleted_count INTEGER NOT NULL DEFAULT 0"
            ))


def _to_record(row: Document) -> DocRecord:
    return DocRecord(
        doc_id=row.doc_id, source_name=row.source_name, source_path=row.source_path,
        s3_key=row.s3_key, content_hash=row.content_hash, size_bytes=row.size_bytes,
        etag=row.etag, last_modified=row.last_modified, department=row.department,
        access_level=row.access_level, product=row.product, status=row.status,
        version=row.version,
    )


class SqlRegistry:
    def __init__(self, engine=None):
        self.engine = engine or get_engine()
        self._session = sessionmaker(self.engine, expire_on_commit=False)

    # ---- documents -------------------------------------------------------
    def get(self, doc_id: str) -> DocRecord | None:
        with self._session() as s:
            row = s.get(Document, doc_id)
            return _to_record(row) if row else None

    def upsert(self, rec: DocRecord) -> None:
        data = asdict(rec)
        with self._session.begin() as s:
            row = s.get(Document, rec.doc_id)
            if row is None:
                s.add(Document(**data))
            else:
                for key, value in data.items():
                    setattr(row, key, value)

    def list_by_status(self, statuses: list[str]) -> list[DocRecord]:
        with self._session() as s:
            rows = s.scalars(
                select(Document)
                .where(Document.status.in_(statuses))
                .order_by(Document.source_name, Document.source_path)
            )
            return [_to_record(r) for r in rows]

    def set_status(self, doc_id: str, status: str) -> None:
        with self._session.begin() as s:
            row = s.get(Document, doc_id)
            if row is not None:
                row.status = status

    def reset_status(self, from_status: str, to_status: str) -> int:
        """Bulk move documents between statuses (e.g. indexed -> chunked to re-embed all)."""
        with self._session.begin() as s:
            result = s.execute(
                update(Document).where(Document.status == from_status).values(status=to_status)
            )
            return result.rowcount

    def list_by_source(self, source_name: str) -> list[DocRecord]:
        with self._session() as s:
            rows = s.scalars(select(Document).where(Document.source_name == source_name))
            return [_to_record(r) for r in rows]

    def mark_deleted(self, doc_id: str) -> None:
        """Soft-delete: keep the document row as an audit trail, drop its chunks."""
        with self._session.begin() as s:
            s.execute(delete(ChunkRow).where(ChunkRow.doc_id == doc_id))
            row = s.get(Document, doc_id)
            if row is not None:
                row.status = "deleted"

    def purge_document(self, doc_id: str) -> None:
        """Hard-delete: remove the document and its chunks entirely."""
        with self._session.begin() as s:
            s.execute(delete(ChunkRow).where(ChunkRow.doc_id == doc_id))
            s.execute(delete(Document).where(Document.doc_id == doc_id))

    @contextmanager
    def sync_lock(self, key: int = 724_001):
        """Cross-process mutex (Postgres advisory lock), so a manual run and the
        scheduler can never sync at the same time."""
        conn = self.engine.connect()
        acquired = False
        try:
            acquired = bool(conn.execute(text("select pg_try_advisory_lock(:k)"), {"k": key}).scalar())
            conn.commit()
            if not acquired:
                raise SyncAlreadyRunning("another sync is already running")
            yield
        finally:
            if acquired:
                conn.execute(text("select pg_advisory_unlock(:k)"), {"k": key})
                conn.commit()
            conn.close()

    def status_counts(self) -> dict[str, int]:
        with self._session() as s:
            rows = s.execute(select(Document.status, func.count()).group_by(Document.status))
            return {status: n for status, n in rows}

    def indexed_chunk_count(self) -> int:
        """Chunks that belong to documents marked indexed: what the vector index should hold."""
        with self._session() as s:
            return s.scalar(
                select(func.count()).select_from(ChunkRow)
                .join(Document, Document.doc_id == ChunkRow.doc_id)
                .where(Document.status == "indexed")
            ) or 0

    def corpus_version(self) -> str:
        """Changes whenever any document is added, edited, re-indexed or deleted. Cached
        answers are keyed on it, so a sync can never leave stale (or deleted) content behind."""
        with self._session() as s:
            n, newest = s.execute(select(func.count(), func.max(Document.updated_at))).one()
            return f"{n}:{newest.isoformat() if newest else 'empty'}"

    # ---- chunks ----------------------------------------------------------
    def replace_chunks(self, rec: DocRecord, doc_title: str, chunks: list) -> None:
        """Swap in the full chunk set for one document (old chunks removed first)."""
        with self._session.begin() as s:
            s.execute(delete(ChunkRow).where(ChunkRow.doc_id == rec.doc_id))
            s.add_all(
                ChunkRow(
                    chunk_id=f"{rec.doc_id}-{c.chunk_index:04d}",
                    doc_id=rec.doc_id, doc_version=rec.version, chunk_index=c.chunk_index,
                    doc_title=doc_title, source_path=rec.source_path, section=c.section,
                    text=c.text, token_count=c.token_count, page=c.page,
                    department=rec.department, access_level=rec.access_level,
                    product=rec.product,
                )
                for c in chunks
            )

    def get_chunks(self, doc_id: str) -> list[ChunkRow]:
        with self._session() as s:
            return list(s.scalars(
                select(ChunkRow).where(ChunkRow.doc_id == doc_id).order_by(ChunkRow.chunk_index)
            ))

    def all_chunks(self) -> list[ChunkRow]:
        with self._session() as s:
            return list(s.scalars(select(ChunkRow).order_by(ChunkRow.source_path, ChunkRow.chunk_index)))

    def chunks_for(self, path_contains: str) -> list[ChunkRow]:
        with self._session() as s:
            return list(s.scalars(
                select(ChunkRow)
                .where(ChunkRow.source_path.contains(path_contains))
                .order_by(ChunkRow.source_path, ChunkRow.chunk_index)
            ))

    def chunk_stats(self) -> dict:
        with self._session() as s:
            total, avg, mn, mx = s.execute(
                select(func.count(), func.avg(ChunkRow.token_count),
                       func.min(ChunkRow.token_count), func.max(ChunkRow.token_count))
            ).one()
            return {"chunks": total, "avg_tokens": round(float(avg or 0)),
                    "min_tokens": mn or 0, "max_tokens": mx or 0}

    # ---- runs ------------------------------------------------------------
    def last_sync_at(self) -> datetime | None:
        """When the most recent ingestion run (any source) finished."""
        with self._session() as s:
            return s.scalar(select(func.max(SyncRun.finished_at)))

    def log_run(self, source_name: str, result, started_at: datetime, finished_at: datetime) -> None:
        with self._session.begin() as s:
            s.add(SyncRun(
                source_name=source_name, started_at=started_at, finished_at=finished_at,
                new_count=result.new, changed_count=result.changed,
                unchanged_count=result.unchanged, failed_count=result.failed,
                deleted_count=getattr(result, "deleted", 0),
            ))
