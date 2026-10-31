"""Wires the real components (Postgres, MinIO/S3, Qdrant, local models) into a SyncService.

Kept separate from ingestion/sync.py so the orchestration logic stays free of heavy
imports and can be unit-tested with fakes.
"""
from __future__ import annotations

from functools import lru_cache


@lru_cache
def get_embedder():
    """Loaded once per process: the scheduler reuses it across runs."""
    from core.settings import settings
    from ingestion.embeddings import FastEmbedEmbedder

    return FastEmbedEmbedder(
        settings.embedding_model, settings.sparse_model, settings.resolved_model_cache
    )


class LazyEmbedder:
    """Loads the embedding model on first real use. Land/parse/chunk stages never embed, so a
    process that only runs those (an Airflow task, say) skips the model load and its memory."""

    @property
    def dimension(self) -> int:
        return get_embedder().dimension

    def embed_documents(self, texts):
        return get_embedder().embed_documents(texts)

    def embed_query(self, text):
        return get_embedder().embed_query(text)


def build_sync_service(source: str | None = None):
    from pathlib import Path

    from connectors.factory import build_connectors
    from core.db import SqlRegistry, init_db
    from core.settings import ROOT, company, settings
    from core.storage import RawStore
    from core.vectorstore import QdrantIndex
    from ingestion.indexer import IndexingService
    from ingestion.sync import SyncService

    init_db()
    registry, store, index = SqlRegistry(), RawStore(), QdrantIndex()

    clone_root = ROOT / settings.resolved_clone_dir
    connectors = build_connectors(company["source"], ROOT, Path(clone_root))
    if source:
        connectors = [c for c in connectors if c.name == source]
        if not connectors:
            raise ValueError(f"No source named '{source}' in config/company.yaml")

    sync_cfg = company.get("sync", {})
    service = SyncService(
        connectors, store, registry, index,
        IndexingService(registry, LazyEmbedder(), index),
        company["metadata"],
        chunking=company.get("chunking"),
        max_delete_fraction=sync_cfg.get("max_delete_fraction", 0.3),
        delete_floor=sync_cfg.get("delete_floor", 5),
    )
    return service, registry


def run_sync_once(limit: int | None = None, source: str | None = None,
                  allow_mass_delete: bool = False):
    service, registry = build_sync_service(source)
    with registry.sync_lock():          # raises SyncAlreadyRunning if another sync is active
        return service.run(limit=limit, allow_mass_delete=allow_mass_delete)
