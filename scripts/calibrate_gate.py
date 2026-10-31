"""Pick `generation.min_dense_score`: how similar must the best chunk be before we try to answer?

Compares the best dense similarity for questions we SHOULD answer (eval/retrieval_cases.yaml,
only those the role can see) with questions we should REFUSE (eval/out_of_scope.yaml).

Run:  python scripts/calibrate_gate.py            # as admin (sees everything)
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import yaml

from core.db import SqlRegistry
from core.settings import company
from eval.retrieval_eval import ANSWERABLE, classify, load_cases
from serving.retrieval import build_retriever

role = "admin"
retriever = build_retriever(rerank=False)
docs = SqlRegistry().list_by_status(["indexed"])
good, bad = [], []

for case in load_cases(ROOT / "eval" / "retrieval_cases.yaml"):
    if classify(case, docs, company["roles"][role]) == ANSWERABLE:
        score = retriever.retrieve(case.question, role, mode="hybrid").dense_top
        good.append((score, case.question))
oos = yaml.safe_load((ROOT / "eval" / "out_of_scope.yaml").read_text(encoding="utf-8"))["questions"]
for q in oos:
    bad.append((retriever.retrieve(q, role, mode="hybrid").dense_top, q))

print("SHOULD ANSWER (best dense similarity, lowest first):")
for s, q in sorted(good):
    print(f"  {s:.3f}  {q}")
print("\nSHOULD REFUSE (highest first):")
for s, q in sorted(bad, reverse=True):
    print(f"  {s:.3f}  {q}")

lowest_good, highest_bad = min(s for s, _ in good), max(s for s, _ in bad)
print(f"\nlowest answerable = {lowest_good:.3f}   highest out-of-scope = {highest_bad:.3f}")
if lowest_good > highest_bad:
    print(f"Clean gap. Suggested min_dense_score: {(lowest_good + highest_bad) / 2:.2f}")
else:
    print("The ranges overlap: no threshold separates them perfectly. Pick the highest value that "
          "still answers the questions you care about; the model's own refusal and the citation "
          "check catch the rest.")
