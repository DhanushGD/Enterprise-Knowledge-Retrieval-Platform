"""Step 7: embed chunked documents and upsert them into the vector index.

    chunked  --embed+upsert-->  indexed        (index_failed on error, retried next run)

Idempotent: point ids are derived from chunk ids, so re-indexing overwrites instead of
duplicating. When a document shrinks, its leftover points are deleted *after* the new
ones are in, so searches never see a half-empty document.
"""
from __future__ import annotations

import uuid

from ingestion.chunking import build_embed_text
from ingestion.pipeline import StageResult
from ingestion.records import IndexPoint

INDEX_QUEUE = ["chunked", "index_failed"]


def point_id(chunk_id: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"rag-chunk:{chunk_id}"))


class IndexingService:
    def __init__(self, registry, embedder, index):
        self.registry = registry
        self.embedder = embedder
        self.index = index

    def index_pending(self, limit: int | None = None) -> StageResult:
        result = StageResult()
        self.index.ensure(self.embedder.dimension)

        for rec in self.registry.list_by_status(INDEX_QUEUE)[:limit]:
            try:
                chunks = self.registry.get_chunks(rec.doc_id)
                if not chunks:
                    raise ValueError("document has no chunks")
                embeddings = self.embedder.embed_documents(
                    [build_embed_text(c.doc_title, c.section, c.text) for c in chunks]
                )
                points = [
                    IndexPoint(
                        point_id=point_id(c.chunk_id),
                        embedding=emb,
                        payload={
                            "chunk_id": c.chunk_id, "doc_id": c.doc_id,
                            "doc_version": c.doc_version, "chunk_index": c.chunk_index,
                            "doc_title": c.doc_title, "source_name": rec.source_name,
                            "source_path": c.source_path, "section": c.section,
                            "text": c.text, "page": c.page, "token_count": c.token_count,
                            "department": c.department, "access_level": c.access_level,
                            "product": c.product,
                        },
                    )
                    for c, emb in zip(chunks, embeddings)
                ]
                self.index.upsert(points)
                self.index.delete_stale(rec.doc_id, [p.point_id for p in points])
                self.registry.set_status(rec.doc_id, "indexed")
                result.ok += 1
            except Exception as exc:
                self.registry.set_status(rec.doc_id, "index_failed")
                result.failed += 1
                result.failures.append(f"{rec.source_path}: {type(exc).__name__}: {exc}")
        return result
