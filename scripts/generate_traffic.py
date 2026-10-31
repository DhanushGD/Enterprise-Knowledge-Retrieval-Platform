"""Create some traffic so the Grafana panels and Prometheus series have something to show.

Run:  python scripts/generate_traffic.py --user alice --password <alice-password>
      python scripts/generate_traffic.py --user alice --password <pw> --only logins
Scenarios (default: all, in this order): logins, refusals, cache, ratelimit.
The API must be running (python scripts/run_api.py). Needs no extra packages beyond requests.
"""
import argparse
import time

import requests

parser = argparse.ArgumentParser()
parser.add_argument("--user", required=True)
parser.add_argument("--password", required=True)
parser.add_argument("--base-url", default="http://localhost:8000")
parser.add_argument("--only", choices=["logins", "refusals", "cache", "ratelimit"])
args = parser.parse_args()
base = args.base_url.rstrip("/")


def login(password):
    return requests.post(f"{base}/auth/login", json={"username": args.user, "password": password}, timeout=30)


def ask(token, question):
    began = time.time()
    r = requests.post(f"{base}/ask", json={"question": question},
                      headers={"Authorization": f"Bearer {token}"}, timeout=300)
    body = r.json()
    detail = body.get("status") or body.get("error", {}).get("code")
    return r.status_code, detail, body.get("cached"), time.time() - began


def run(name):
    return args.only in (None, name)


r = login(args.password)
if r.status_code != 200:
    raise SystemExit(f"Login failed ({r.status_code}): check the username/password. {r.text[:200]}")
token = r.json()["access_token"]

if run("logins"):
    print("\n[logins] 3 wrong passwords, then a good one  -> 'Logins' panel: failed x3, ok x1")
    for _ in range(3):
        print("  wrong password ->", login("definitely-wrong").status_code)
    print("  right password ->", login(args.password).status_code)

if run("refusals"):
    print("\n[refusals] off-topic questions (fast, no model)  -> 'Why questions were refused': low_relevance")
    for q in ("What is the capital of Australia?", "Write me a poem about the ocean.",
              "Give me a recipe for chocolate chip cookies."):
        print(" ", ask(token, q))

if run("cache"):
    print("\n[cache] the same real question twice  -> first is a cache miss (slow), second a hit (instant)")
    q = "How do I publish a package to the NPM registry on GitLab?"
    for _ in range(2):
        print(" ", ask(token, q))

if run("ratelimit"):
    print("\n[ratelimit] 14 quick off-topic questions  -> 'Requests turned away': ask / rate_limited")
    for i in range(14):
        code, detail, _, _ = ask(token, f"What is the capital of Australia? ({i})")
        print(f"  {i + 1:>2}: HTTP {code} {detail}")
print("\nDone. Give Prometheus ~15-30 s, then refresh Grafana.")
