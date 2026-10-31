"""Compare retrieval strategies on a labelled question set.

Questions whose target page the role may not see are left out of the scores (and checked for
permission leaks instead), so the numbers measure retrieval, not the permission system.

Run:  python scripts/run_retrieval_eval.py                       # as an agent
      python scripts/run_retrieval_eval.py --role admin          # admin sees every page
      python scripts/run_retrieval_eval.py --show-misses
      python scripts/run_retrieval_eval.py --reranker Xenova/ms-marco-MiniLM-L-6-v2 --candidates 20
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core.db import SqlRegistry
from core.settings import company
from eval.retrieval_eval import HIDDEN, NO_DOC, classify, evaluate, format_scores, load_cases
from serving.retrieval import MODES, build_retriever

parser = argparse.ArgumentParser()
parser.add_argument("--cases", default=str(ROOT / "eval" / "retrieval_cases.yaml"))
parser.add_argument("--role", default="agent", choices=list(company["roles"]))
parser.add_argument("--modes", nargs="+", default=list(MODES), choices=MODES)
parser.add_argument("--reranker", help="reranker model name (default: RERANKER_MODEL in .env)")
parser.add_argument("--candidates", type=int, help="chunks sent to the reranker (default 30)")
parser.add_argument("--show-misses", action="store_true")
args = parser.parse_args()

cases = load_cases(args.cases)
docs = SqlRegistry().list_by_status(["indexed"])
role_cfg = company["roles"][args.role]
kinds = {c.question: classify(c, docs, role_cfg) for c in cases}

retriever = build_retriever(
    rerank="hybrid_rerank" in args.modes, reranker_model=args.reranker, candidates=args.candidates
)
print(f"{len(cases)} questions, role={args.role}, "
      f"reranker={args.reranker or 'default'}, candidates={retriever.candidates}\n")
scores = evaluate(retriever, cases, args.role, args.modes, classifier=lambda c: kinds[c.question])
print(format_scores(scores))

first = next(iter(scores.values()))
for label, kind in (("hidden from this role", HIDDEN), ("matches no indexed page", NO_DOC)):
    names = [c.question for c in cases if kinds[c.question] == kind]
    if names and args.show_misses:
        print(f"\n--- skipped: {label} ({len(names)}) ---")
        for q in names:
            print(f"  {q}")

for mode, s in scores.items():
    for case, top in s.leaks:
        print(f"\n!! LEAK in {mode}: {case.question!r} returned {top}")
if args.show_misses:
    for mode, s in scores.items():
        if s.misses:
            print(f"\n--- {mode}: answerable but not in top 5 ({len(s.misses)}) ---")
            for case, top in s.misses:
                print(f"  Q: {case.question}\n     wanted: {case.expected}\n     got:    {top}")
