"""Measure chunk sizes with the REAL embedding-model tokenizer (read-only).

The chunker sizes chunks with a fast word+punctuation count. The embedding model
(bge-small) sees WordPiece tokens and silently truncates anything past 512, and URLs,
code and snake_case identifiers split into many pieces. This script shows how far the
two counts drift on your data, which chunks would be truncated, and which are tiny.

Run:  python scripts/chunk_report.py
"""
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tokenizers import Tokenizer

from core.db import SqlRegistry
from core.settings import settings
from ingestion.chunking import build_embed_text

MODEL_LIMIT = 512

chunks = SqlRegistry().all_chunks()
if not chunks:
    sys.exit("No chunks found. Run scripts/run_processing.py first.")

print(f"Loading tokenizer for {settings.embedding_model} ...")
tokenizer = Tokenizer.from_pretrained(settings.embedding_model)
tokenizer.no_truncation()
tokenizer.no_padding()

rows = []
for c in chunks:
    embedded = build_embed_text(c.doc_title, c.section, c.text)      # exactly what gets embedded
    rows.append((len(tokenizer.encode(embedded).ids), c))

real = [n for n, _ in rows]
approx = [c.token_count for _, c in rows]
over = [(n, c) for n, c in rows if n > MODEL_LIMIT]

print(f"\nChunks: {len(rows)}")
print(f"Chunker's own token count : avg {sum(approx)/len(rows):.0f}, max {max(approx)}")
print(f"Real WordPiece tokens     : avg {sum(real)/len(rows):.0f}, max {max(real)}   "
      f"(ratio real/approx = {sum(real)/sum(approx):.2f})")

buckets = Counter()
for n in real:
    key = ("<30" if n < 30 else "30-99" if n < 100 else "100-249" if n < 250
           else "250-399" if n < 400 else "400-512" if n <= 512 else ">512 (TRUNCATED)")
    buckets[key] += 1
print("\nSize distribution (real tokens):")
for key in ["<30", "30-99", "100-249", "250-399", "400-512", ">512 (TRUNCATED)"]:
    print(f"  {key:18} {buckets[key]:5}  {'#' * min(60, buckets[key] * 60 // len(rows))}")

print(f"\nChunks over the {MODEL_LIMIT}-token model limit: {len(over)} ({100 * len(over) / len(rows):.1f}%)")
if over:
    print("  Their tails are invisible to the dense vector. Fix: lower `chunking.max_tokens` in")
    print("  config/company.yaml (try 320), then: python scripts/run_processing.py --rechunk")
    for n, c in sorted(over, key=lambda x: -x[0])[:5]:
        print(f"    {n:4} tokens  {c.source_path} > {c.section[:60]}")
        print(f"          starts: {c.text[:160]!r}")

tiny = sorted(rows, key=lambda x: x[0])[:8]
print("\nSmallest chunks:")
for n, c in tiny:
    print(f"  {n:3} tokens  {c.source_path}  |  {c.text[:70]!r}")
