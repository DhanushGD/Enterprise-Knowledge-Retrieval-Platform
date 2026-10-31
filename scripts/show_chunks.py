"""Print the chunks stored for documents whose path contains a substring.
Use it to eyeball chunk quality before embedding anything.

Run:  python scripts/show_chunks.py password-reset
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.db import SqlRegistry

if len(sys.argv) < 2:
    sys.exit("usage: python scripts/show_chunks.py <part-of-document-path>")

rows = SqlRegistry().chunks_for(sys.argv[1])
if not rows:
    sys.exit(f"No chunks found for '{sys.argv[1]}'. Did you run scripts/run_processing.py?")

for row in rows:
    print("=" * 78)
    print(f"{row.chunk_id}  [{row.token_count} tokens]  {row.department}/{row.access_level}/{row.product}")
    print(f"{row.doc_title} > {row.section}" if row.section else row.doc_title)
    print("-" * 78)
    print(row.text)
print("=" * 78)
