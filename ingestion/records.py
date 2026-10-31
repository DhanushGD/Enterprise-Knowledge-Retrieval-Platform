"""Plain data objects shared by the landing code and the database layer."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime


@dataclass
class DocRecord:
    doc_id: str
    source_name: str
    source_path: str
    s3_key: str
    content_hash: str
    size_bytes: int
    etag: str | None
    last_modified: datetime | None
    department: str
    access_level: str
    product: str
    status: str = "new"   # new | changed | parsed | indexed | deleted
    version: int = 1


@dataclass
class Embedding:
    """Both vectors for one piece of text."""
    dense: list[float]
    sparse_indices: list[int]
    sparse_values: list[float]


@dataclass
class IndexPoint:
    point_id: str          # deterministic UUID derived from the chunk id
    embedding: Embedding
    payload: dict
