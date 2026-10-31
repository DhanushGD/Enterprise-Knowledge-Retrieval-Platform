"""Step 7: embed chunks and upsert them into Qdrant.

Run:  python scripts/run_indexing.py
      python scripts/run_indexing.py --limit 10
      python scripts/run_indexing.py --recreate   # drop the collection and re-embed everything
                                                  # (needed after changing the embedding model)

The first run downloads the embedding models (~150 MB) into ~/.cache/rag-models.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.db import SqlRegistry, init_db
from core.settings import settings
from core.vectorstore import QdrantIndex
from ingestion.embeddings import FastEmbedEmbedder
from ingestion.indexer import IndexingService

parser = argparse.ArgumentParser()
parser.add_argument("--limit", type=int, help="index at most N documents")
parser.add_argument("--recreate", action="store_true", help="drop the collection and re-embed all")
args = parser.parse_args()

init_db()
registry, index = SqlRegistry(), QdrantIndex()

print("Loading embedding models (first run downloads ~150 MB)...")
embedder = FastEmbedEmbedder(
    settings.embedding_model, settings.sparse_model, settings.resolved_model_cache
)
print(f"Dense model: {settings.embedding_model} ({embedder.dimension} dims) + sparse: {settings.sparse_model}")

if args.recreate:
    index.recreate(embedder.dimension)
    moved = registry.reset_status("indexed", "chunked")
    print(f"Collection recreated; {moved} documents queued for re-embedding")

result = IndexingService(registry, embedder, index).index_pending(args.limit)
print(f"index  ok={result.ok}  failed={result.failed}")
for failure in result.failures[:10]:
    print(f"  ! {failure}")

print("\nDocuments by status:", registry.status_counts())
print(f"Points in Qdrant collection '{settings.qdrant_collection}': {index.count()}")
sys.exit(1 if result.failed else 0)
