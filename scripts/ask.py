"""Ask the assistant a question (full pipeline: retrieve -> generate -> cited answer).

Run:  python scripts/ask.py "how do I handle a ticket in a language I don't speak" --role team_lead
      python scripts/ask.py "what is the capital of Australia"        # should decline, fast
      python scripts/ask.py "..." --no-stream --warm                  # load the model first

On a CPU the first question after starting Ollama also pays a one-off model load (5-15 s).
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.settings import company
from serving.generation import build_answer_service
from serving.llm import LLMError

parser = argparse.ArgumentParser()
parser.add_argument("question")
parser.add_argument("--role", default="agent", choices=list(company["roles"]))
parser.add_argument("--warm", action="store_true", help="load the model before timing the answer")
parser.add_argument("--no-stream", action="store_true")
args = parser.parse_args()

service = build_answer_service()
if args.warm:
    print("Warming up the model ...", flush=True)
    try:
        service.llm.warm_up()
    except LLMError as exc:
        sys.exit(f"LLM problem: {exc}")

print(f'Q [{args.role}]: {args.question}\n')
final = None
for event in service.stream(args.question, args.role):
    if event["type"] == "sources":
        print("Retrieved:")
        for s in event["sources"]:
            print(f"  [{s.n}] {s.location}   ({s.source_path})")
        print("\nAnswer (streaming; final text is validated below):", flush=True)
    elif event["type"] == "token" and not args.no_stream:
        print(event["text"], end="", flush=True)
    elif event["type"] == "done":
        final = event["answer"]

a = final
print(f"\n\n=== {a.status.upper()}" + (f"  ({a.reason})" if a.reason else "") + " ===")
print(a.text)
if a.citations:
    print("\nCitations:")
    for s in a.citations:
        print(f"  [{s.n}] {s.location}  -  {s.source_path}")
elif a.status in ("ungrounded", "llm_unavailable"):
    print("\nRelated pages:")
    for s in a.sources:
        print(f"  [{s.n}] {s.location}  -  {s.source_path}")
print(f"\ndense_top={a.dense_top}  timings(ms)={a.timings}  tokens: prompt={a.prompt_tokens} "
      f"answer={a.completion_tokens}")
