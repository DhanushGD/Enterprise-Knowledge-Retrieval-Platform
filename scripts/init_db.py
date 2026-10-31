"""Create the Postgres tables. Safe to run any number of times.

Run:  python scripts/init_db.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.db import Base, get_engine, init_db

init_db()
print("Tables ready:", ", ".join(sorted(Base.metadata.tables)))
