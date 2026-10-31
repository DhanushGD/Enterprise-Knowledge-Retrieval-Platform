"""Draft golden-set candidates from the indexed pages, with checks that keep the drafts honest.

A 3B model writes a question and picks phrases from a passage. It is a decent first draft and an
unreliable final answer, so every candidate must pass mechanical checks (the phrases really occur
in the passage; the question does not just copy the passage) and then a PERSON reads it before it
joins eval/golden.yaml. This module is the checks; scripts/build_golden.py does the model calls.
"""
from __future__ import annotations

import json
import re

PROMPT = """Below is a passage from a support handbook.
Write ONE question that a support engineer might ask and that this passage answers.
Rules: do not copy phrases of more than 4 words from the passage; do not mention "passage" or "text".
Then list 2 or 3 short exact phrases (1 to 4 words each) copied from the passage that a correct answer must contain.
Reply with JSON only: {{"question": "...", "facts": ["...", "..."]}}

Passage:
{passage}"""

MAX_COPIED_RUN = 6          # longest run of consecutive words allowed to be copied from the passage


def _words(text: str) -> list[str]:
    return re.findall(r"[a-z0-9']+", text.lower())


def parse_candidate(raw: str) -> dict | None:
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not match:
        return None
    try:
        data = json.loads(match.group(0))
    except ValueError:
        return None
    if not isinstance(data, dict) or not isinstance(data.get("question"), str):
        return None
    facts = data.get("facts")
    if not isinstance(facts, list):
        return None
    return {"question": data["question"].strip(),
            "facts": [f.strip() for f in facts if isinstance(f, str) and f.strip()]}


def longest_copied_run(question: str, passage: str) -> int:
    q, p = _words(question), _words(passage)
    grams = {}
    for i in range(len(p)):
        grams.setdefault(p[i], []).append(i)
    best = 0
    for i in range(len(q)):
        for j in grams.get(q[i], ()):
            k = 0
            while i + k < len(q) and j + k < len(p) and q[i + k] == p[j + k]:
                k += 1
            best = max(best, k)
    return best


def validate_candidate(cand: dict | None, passage: str) -> dict | None:
    """Return the cleaned candidate, or None when it fails a check."""
    if not cand:
        return None
    question = cand["question"]
    if not 15 <= len(question) <= 220 or "\n" in question:
        return None
    if re.search(r"\b(passage|the text|the above|this section)\b", question.lower()):
        return None
    if longest_copied_run(question, passage) > MAX_COPIED_RUN:
        return None
    lowered = passage.lower()
    facts = []
    for fact in cand["facts"]:
        if 1 <= len(fact.split()) <= 4 and fact.lower() in lowered and fact.lower() not in facts:
            facts.append(fact.lower())
    if not facts:
        return None
    return {"question": question if question.endswith("?") else question.rstrip(".") + "?",
            "facts": facts[:3]}
