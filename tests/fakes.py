"""In-memory stand-ins for MinIO and Postgres, so pipeline tests need no Docker."""
import hashlib
from contextlib import contextmanager
from dataclasses import replace
from types import SimpleNamespace

from ingestion.records import Embedding


class FakeStore:
    def __init__(self):
        self.objects: dict[str, bytes] = {}

    def ensure(self):
        pass

    def put(self, key, data, content_type=None):
        self.objects[key] = data

    def get(self, key):
        return self.objects[key]

    def delete(self, keys):
        for key in keys:
            self.objects.pop(key, None)


class FakeRegistry:
    def __init__(self):
        self.lock_held = False
        self.rows = {}
        self.chunks: dict[str, tuple[str, list]] = {}   # doc_id -> (title, chunks)
        self.chunk_recs = {}
        self.runs = []

    def get(self, doc_id):
        row = self.rows.get(doc_id)
        return replace(row) if row else None

    def upsert(self, rec):
        self.rows[rec.doc_id] = rec

    def log_run(self, source_name, result, started, finished):
        self.runs.append((source_name, result))

    def list_by_status(self, statuses):
        return [replace(r) for r in sorted(self.rows.values(), key=lambda r: r.source_path)
                if r.status in statuses]

    def set_status(self, doc_id, status):
        self.rows[doc_id].status = status

    def replace_chunks(self, rec, title, chunks):
        self.chunks[rec.doc_id] = (title, list(chunks))
        self.chunk_recs[rec.doc_id] = replace(rec)

    def list_by_source(self, source_name):
        return [replace(r) for r in self.rows.values() if r.source_name == source_name]

    def mark_deleted(self, doc_id):
        self.rows[doc_id].status = "deleted"
        self.chunks.pop(doc_id, None)
        self.chunk_recs.pop(doc_id, None)

    @contextmanager
    def sync_lock(self, key=0):
        from core.errors import SyncAlreadyRunning
        if self.lock_held:
            raise SyncAlreadyRunning("another sync is already running")
        self.lock_held = True
        try:
            yield
        finally:
            self.lock_held = False

    def status_counts(self):
        counts = {}
        for row in self.rows.values():
            counts[row.status] = counts.get(row.status, 0) + 1
        return counts

    def indexed_chunk_count(self):
        return sum(len(self.chunks[d][1]) for d, r in self.rows.items()
                   if r.status == "indexed" and d in self.chunks)

    def corpus_version(self):
        return f"{len(self.rows)}:{sum(r.version for r in self.rows.values())}"

    def get_chunks(self, doc_id):
        """Same shape as the SQL ChunkRow rows."""
        title, chunks = self.chunks[doc_id]
        rec = self.chunk_recs[doc_id]
        return [
            SimpleNamespace(
                chunk_id=f"{doc_id}-{c.chunk_index:04d}", doc_id=doc_id, doc_version=rec.version,
                chunk_index=c.chunk_index, doc_title=title, source_path=rec.source_path,
                section=c.section, text=c.text, token_count=c.token_count, page=c.page,
                department=rec.department, access_level=rec.access_level, product=rec.product,
            )
            for c in chunks
        ]


class FakeEmbedder:
    dimension = 8

    def __init__(self, fail_on: str | None = None):
        self.seen: list[str] = []
        self.fail_on = fail_on

    def embed_documents(self, texts):
        self.seen.extend(texts)
        if self.fail_on and any(self.fail_on in t for t in texts):
            raise RuntimeError("model exploded")
        return [self.embed_query(t) for t in texts]

    def embed_query(self, text):
        digest = hashlib.sha256(text.encode()).digest()
        words = sorted(set(text.lower().split()))
        return Embedding(
            dense=[b / 255 for b in digest[:8]],
            sparse_indices=[int(hashlib.md5(w.encode()).hexdigest()[:6], 16) for w in words],
            sparse_values=[1.0] * len(words),
        )


class FakeIndex:
    def __init__(self):
        self.points = {}
        self.dimension = None

    def ensure(self, dimension):
        self.dimension = dimension

    def healthy(self):
        pass

    def upsert(self, points, batch_size=64):
        for p in points:
            self.points[p.point_id] = p

    def delete_stale(self, doc_id, keep_ids):
        keep = set(keep_ids)
        for pid in [k for k, p in self.points.items() if p.payload["doc_id"] == doc_id and k not in keep]:
            del self.points[pid]

    def delete_document(self, doc_id):
        for pid in [k for k, p in self.points.items() if p.payload["doc_id"] == doc_id]:
            del self.points[pid]

    def count(self):
        return len(self.points)
