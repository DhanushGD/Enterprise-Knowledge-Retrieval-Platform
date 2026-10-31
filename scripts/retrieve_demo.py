"""Run one query through the full Step 9 pipeline as a given role.

Run:  python scripts/retrieve_demo.py "how do I escalate a ticket"
      python scripts/retrieve_demo.py "how do I escalate a ticket" --mode hybrid
      python scripts/retrieve_demo.py "leave policy" --role admin
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.settings import company
from serving.retrieval import MODES, build_retriever

parser = argparse.ArgumentParser()
parser.add_argument("query")
parser.add_argument("--role", default="agent", choices=list(company["roles"]))
parser.add_argument("--mode", choices=MODES, help="default: hybrid_rerank if enabled")
parser.add_argument("--top-k", type=int, default=5)
args = parser.parse_args()

retriever = build_retriever(rerank=None if args.mode in (None, "hybrid_rerank") else False)
result = retriever.retrieve(args.query, args.role, mode=args.mode, top_k=args.top_k)

role = company["roles"][args.role]
print(f'Query: "{args.query}"  role={args.role}  mode={result.mode}')
print(f"  role may see access_level={role['allowed_access_levels']} department={role['allowed_departments']}")
print(f"  {result.candidates} candidates -> {len(result.hits)} shown   timings(ms)={result.timings}")
if result.degraded:
    print(f"  ! DEGRADED: {result.degraded}")
print()
for i, h in enumerate(result.hits, 1):
    rr = f"  rerank={h.rerank_score:.3f}" if h.rerank_score is not None else ""
    print(f"{i}. [{h.access_level}/{h.department}] {h.location}")
    print(f"   {h.source_path}   score={h.score:.4f}{rr}")
    print(f"   {h.text[:140].replace(chr(10), ' ')}...\n")
if not result.hits:
    print("(nothing visible to this role)")
