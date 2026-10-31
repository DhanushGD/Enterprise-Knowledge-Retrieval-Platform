"""Retrieval-only evaluation: does the right DOCUMENT come back near the top?

No LLM involved, so it's fast, free and deterministic - the right tool for comparing
dense vs sparse vs hybrid vs hybrid+rerank, or tuning chunk size / candidates / max_per_doc.
(Step 12 adds answer-quality metrics with RAGAS on top of this.)

A case is a question plus one or more substrings of `source_path` that identify a document
that answers it. Matching is at document level, so re-chunking doesn't invalidate labels.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import yaml


@dataclass
class Case:
    question: str
    expected: list[str]            # any of these substrings in source_path counts as a hit
    note: str = ""


ANSWERABLE, HIDDEN, NO_DOC = "answerable", "hidden", "no_doc"


@dataclass
class ModeScore:
    mode: str
    n: int = 0
    hidden: int = 0                 # cases whose page exists but this role may not see it
    no_doc: int = 0                 # cases whose label matches no indexed page
    leaks: list[tuple[Case, list[str]]] = field(default_factory=list)   # hidden page RETRIEVED
    hit1: int = 0
    hit3: int = 0
    hit5: int = 0
    reciprocal_ranks: list[float] = field(default_factory=list)
    latency_ms: list[float] = field(default_factory=list)
    misses: list[tuple[Case, list[str]]] = field(default_factory=list)   # (case, top paths)

    @property
    def mrr(self) -> float:
        return sum(self.reciprocal_ranks) / self.n if self.n else 0.0

    @property
    def avg_ms(self) -> float:
        return sum(self.latency_ms) / len(self.latency_ms) if self.latency_ms else 0.0

    def pct(self, hits: int) -> float:
        return 100 * hits / self.n if self.n else 0.0


def load_cases(path: str | Path) -> list[Case]:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    cases = []
    for item in raw.get("cases", []):
        expected = item["expected"]
        cases.append(Case(
            question=item["question"],
            expected=[expected] if isinstance(expected, str) else list(expected),
            note=item.get("note", ""),
        ))
    return cases


def rank_of(case: Case, source_paths: list[str]) -> int | None:
    """1-based rank of the first matching DOCUMENT (repeat chunks of one doc count once)."""
    seen: list[str] = []
    for path in source_paths:
        if path not in seen:
            seen.append(path)
    wanted = [e.lower() for e in case.expected]
    for rank, path in enumerate(seen, start=1):
        if any(w in path.lower() for w in wanted):
            return rank
    return None


def classify(case: Case, docs, role_cfg: dict) -> str:
    """Can this role be expected to find the answer?

    `docs` are registry rows (source_path, access_level, department, status). A page the role
    may not see makes the case HIDDEN: it must be left out of the quality score (not finding it
    is correct), and retrieving it is a permission leak.
    """
    wanted = [e.lower() for e in case.expected]
    matches = [d for d in docs
               if d.status == "indexed" and any(w in d.source_path.lower() for w in wanted)]
    if not matches:
        return NO_DOC
    levels, depts = role_cfg["allowed_access_levels"], role_cfg["allowed_departments"]
    if any(d.access_level in levels and d.department in depts for d in matches):
        return ANSWERABLE
    return HIDDEN


def evaluate(retriever, cases: list[Case], role: str, modes: list[str], depth: int = 10,
             classifier: Callable[[Case], str] | None = None) -> dict[str, ModeScore]:
    scores = {m: ModeScore(m) for m in modes}
    for case in cases:
        kind = classifier(case) if classifier else ANSWERABLE
        for mode in modes:
            result = retriever.retrieve(case.question, role, mode=mode, top_k=depth)
            paths = [h.source_path for h in result.hits]
            rank = rank_of(case, paths)
            s = scores[mode]
            if kind == NO_DOC:
                s.no_doc += 1
                continue
            if kind == HIDDEN:
                s.hidden += 1
                if rank is not None:
                    s.leaks.append((case, paths[:3]))
                continue
            s.n += 1
            s.latency_ms.append(result.timings.get("total_ms", 0.0))
            s.reciprocal_ranks.append(1 / rank if rank else 0.0)
            s.hit1 += bool(rank and rank <= 1)
            s.hit3 += bool(rank and rank <= 3)
            s.hit5 += bool(rank and rank <= 5)
            if not rank or rank > 5:
                s.misses.append((case, paths[:3]))
    return scores


def format_scores(scores: dict[str, ModeScore]) -> str:
    lines = [f"{'mode':15} {'hit@1':>7} {'hit@3':>7} {'hit@5':>7} {'MRR':>7} {'avg ms':>8}"]
    for s in scores.values():
        lines.append(
            f"{s.mode:15} {s.pct(s.hit1):6.0f}% {s.pct(s.hit3):6.0f}% {s.pct(s.hit5):6.0f}% "
            f"{s.mrr:7.3f} {s.avg_ms:8.0f}"
        )
    first = next(iter(scores.values())) if scores else None
    if first:
        lines.append(f"({first.n} answerable questions; hit@k = right document in the top k "
                     f"distinct documents)")
        if first.hidden:
            lines.append(f"{first.hidden} more questions target pages this role may not see - "
                         f"excluded from the scores, checked for permission leaks below")
        if first.no_doc:
            lines.append(f"{first.no_doc} questions match no indexed page - fix their labels")
        leaks = sum(len(s.leaks) for s in scores.values())
        lines.append("PERMISSION LEAKS: " + (f"{leaks} !!" if leaks else "0 (hidden pages were never returned)"))
    return "\n".join(lines)
