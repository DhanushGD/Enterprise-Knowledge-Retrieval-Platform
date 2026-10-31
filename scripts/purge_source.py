"""Completely remove one source: its vectors, chunks, database rows and stored files.

Use it to clear out the fictional sample documents once you move to a real corpus, or to
drop a source you no longer want. (A normal sync never touches sources that are no longer
listed in config/company.yaml, so this is the explicit way to remove them.)

Run:  python scripts/purge_source.py --source opendesk-sample-docs
      python scripts/purge_source.py --source opendesk-sample-docs --yes
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.db import SqlRegistry, init_db
from core.storage import RawStore
from core.vectorstore import QdrantIndex
from ingestion.pipeline import parsed_key

parser = argparse.ArgumentParser()
parser.add_argument("--source", required=True)
parser.add_argument("--yes", action="store_true", help="skip the confirmation prompt")
args = parser.parse_args()

init_db()
registry, store, index = SqlRegistry(), RawStore(), QdrantIndex()
docs = registry.list_by_source(args.source)
if not docs:
    sys.exit(f"No documents recorded for source '{args.source}'.")

print(f"Source '{args.source}': {len(docs)} documents will be permanently removed")
print("  (Qdrant vectors, Postgres chunks + rows, and files under raw/ and parsed/)")
if not args.yes and input("Type the source name to confirm: ").strip() != args.source:
    sys.exit("Cancelled.")

index.delete_source(args.source)
keys = []
for doc in docs:
    keys += [doc.s3_key, parsed_key(doc.source_name, doc.source_path)]
    registry.purge_document(doc.doc_id)
store.delete(keys)
print(f"Done. Removed {len(docs)} documents.")
print("Documents by status:", registry.status_counts())
print("Points left in Qdrant:", index.count())
