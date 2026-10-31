"""Smoke-test the index: run a query as a given role and see what each retriever returns.

Run:  python scripts/search_demo.py "how do I refund a customer"
      python scripts/search_demo.py "how many vacation days do we get" --role agent
      python scripts/search_demo.py "how many vacation days do we get" --role admin

Dense and sparse results are shown separately on purpose: Step 9 fuses them. If dense finds
the right document for paraphrases and sparse finds it for exact keywords, the index is good.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.settings import company, settings
from core.vectorstore import DENSE, SPARSE, QdrantIndex
from ingestion.embeddings import FastEmbedEmbedder

parser = argparse.ArgumentParser()
parser.add_argument("query")
parser.add_argument("--role", default="agent", choices=list(company["roles"]))
parser.add_argument("--limit", type=int, default=3)
args = parser.parse_args()

role = company["roles"][args.role]
print(f'Query: "{args.query}"   role={args.role}')
print(f"  may see access_level in {role['allowed_access_levels']}, department in {role['allowed_departments']}\n")

embedder = FastEmbedEmbedder(
    settings.embedding_model, settings.sparse_model, settings.resolved_model_cache
)
query = embedder.embed_query(args.query)
index = QdrantIndex()

for name, label in ((DENSE, "DENSE (meaning)"), (SPARSE, "SPARSE (keywords)")):
    print(f"--- {label} ---")
    hits = index.search(
        query, using=name, limit=args.limit,
        access_levels=role["allowed_access_levels"], departments=role["allowed_departments"],
    )
    if not hits:
        print("  (no results visible to this role)")
    for hit in hits:
        p = hit.payload
        where = f"{p['doc_title']} > {p['section']}" if p["section"] else p["doc_title"]
        print(f"  {hit.score:7.3f}  [{p['access_level']}/{p['department']}]  {where}")
        print(f"           {p['text'][:110].replace(chr(10), ' ')}...")
    print()
