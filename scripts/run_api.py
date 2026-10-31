"""Start the API + chat page.   Run:  python scripts/run_api.py     then open http://localhost:8000

One worker on purpose: the embedding model and LLM client live in this process, and the
generation gate protecting the CPU is per-process.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import logging

import uvicorn

parser = argparse.ArgumentParser()
parser.add_argument("--host", default="127.0.0.1", help="use 0.0.0.0 only behind a reverse proxy / TLS")
parser.add_argument("--port", type=int, default=8000)
args = parser.parse_args()

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
uvicorn.run("serving.app:create_app", factory=True, host=args.host, port=args.port, workers=1)
