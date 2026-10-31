# Reference: how each part works and how to run it

Detailed notes per component, in build order. The overview is in the [README](../README.md);
design and decisions in [ARCHITECTURE.md](ARCHITECTURE.md). Commands are PowerShell unless stated.

## Quick start (original, with make)

Requirements: Docker, Python 3.11+.

```bash
make setup            # venv, dependencies, .env
source .venv/bin/activate
make up               # starts Postgres, Qdrant, Redis, MinIO, Prometheus, Grafana
make check            # verifies every service (creates the S3 bucket if missing)
make init-db          # creates Postgres tables
make land             # Steps 3+4: pull documents into raw/ and register them
python scripts/run_processing.py   # Steps 5+6: parse and chunk (no make target yet)
python scripts/run_indexing.py     # Step 7: embed and upsert into Qdrant
make test             # unit tests, no Docker needed
```

Without `make` (Windows): `python scripts/check_infra.py`, `python scripts/init_db.py`,
`python scripts/run_landing.py`, `python scripts/run_processing.py`, `python -m pytest -q`.
Inspect chunks with `python scripts/show_chunks.py <part-of-path>`; smoke-test the index with
`python scripts/search_demo.py "how do I refund a customer" --role agent`.

## Choosing a document source

Edit `source.type` in `config/company.yaml`:

| type | what it reads | use it for |
|---|---|---|
| `local` | `data/sample_docs/` (bundled, 5 fictional docs) | first end-to-end run, tests |
| `git` | sparse clones of the repos under `source.repos` | the open-source corpus |
| `s3` | `s3://<bucket>/<prefix>` | company-specific data (upload files, point here) |

Re-running `make land` is safe: documents are only re-landed when their SHA-256
changes. Raw files go to `raw/<source>/<path>` in the bucket; one row per document
goes to the Postgres `documents` table (hash, version, status, department,
access level, product).

## Running on the real open-source corpus

`config/company.yaml` ships with `source.type: git` (GitLab handbook + GitLab docs).

```powershell
python scripts/run_landing.py --source gitlab-handbook-sample --limit 50   # small first run
python scripts/run_processing.py
python scripts/run_indexing.py
python scripts/search_demo.py "how do I escalate a ticket" --role agent
```

- Clones live in `~/.cache/rag-sources` (override with `CLONE_DIR`), **outside OneDrive**, because
  OneDrive locks files inside `.git`.
- `max_documents` keeps a hash-stable sample spread across all folders, not the first N files.
- A wrong `subpath` fails with a list of the folders that do exist.

## Keeping the index fresh (Step 8)

One command does land -> detect deletions -> parse -> chunk -> index. Only new or changed
documents are re-embedded; a run that finds nothing new does almost no work.

```powershell
python scripts/run_sync.py                       # everything in config/company.yaml
python scripts/run_sync.py --source gitlab-handbook-sample
python scripts/run_scheduler.py --run-now        # sync now, then nightly at 02:00 (sync.cron)
python scripts/purge_source.py --source opendesk-sample-docs   # remove a source entirely
python scripts/chunk_report.py                   # chunk sizes measured with the real tokenizer
```

**Deletion safety.** A failed or partial listing (clone error, `--limit`) never deletes anything.
An empty listing, or one that would remove more than 30% of a source (min. 5 documents), is
refused unless you pass `--force-deletes`. Deleted documents are removed from Qdrant, chunks and
object storage; the Postgres row stays as an audit trail (`status = deleted`) and is re-indexed
automatically if the file comes back.

**Other behaviour.** A Postgres advisory lock prevents overlapping syncs (manual + scheduled).
Failed runs retry with backoff (30s, 2 min). Valid pages with no body text (e.g. Hugo section
index pages) get `status = skipped` instead of failing on every run. Every run is logged in the
`sync_runs` table.

## Airflow ETL (Step 8b)

The same sync, run by Airflow instead of the APScheduler script: each stage is its own task with
retries, a timeout and a log, and the UI shows every run.

```
preflight -> sources -> land (one task per source) -> parse -> chunk -> index -> verify
```

| Task | What it does | Fails (and retries) when |
|---|---|---|
| `preflight` | Postgres, Qdrant and object storage reachable | any is down |
| `sources` | lists the sources in `config/company.yaml` | - |
| `land` x N | lists one source, lands new/changed files, applies deletions | source unreachable; >5% of files fail |
| `parse` / `chunk` / `index` | the shared stages (embedding runs here, on CPU) | >5% of documents fail (`etl.max_failed_fraction`) |
| `verify` | vectors in Qdrant == chunks of indexed documents; nothing stuck; index not empty | mismatch (**no retry**: a person looks) |

A blocked mass deletion also fails **without retrying**. To allow it, use *Trigger DAG w/ config*
with `force_deletes = true`. `limit = 20` gives a quick trial run (it never deletes).

**How it is built.** The DAG file (`orchestration/dags/ekrp_sync.py`) imports nothing from this
project. Each task runs `python -m orchestration.etl <stage>` in a separate virtualenv, so
Airflow's pinned libraries never fight with ours (SQLAlchemy 2, fastembed, qdrant-client). The
ETL code is plain functions (`orchestration/etl.py`), unit-tested without Airflow; the CLI prints a
JSON result and exits `0` ok / `10` retry / `20` don't retry.

Without Docker you can still run the identical stages: `python -m orchestration.etl all`.

### Run it (Windows, Docker Desktop with >= 4 GB RAM for Docker)

```powershell
docker compose up -d                                          # Step 2 stack must be running
docker compose -f docker-compose.airflow.yml build            # first time: 5-10 min
docker compose -f docker-compose.airflow.yml up airflow-init  # creates Airflow's own database
docker compose -f docker-compose.airflow.yml up -d
docker compose -f docker-compose.airflow.yml logs airflow-apiserver | Select-String -Pattern "password"
```

Open http://localhost:8080, log in as `admin` with the generated password from that log line,
unpause **ekrp_sync**, then *Trigger*.

Checks, in order:

```powershell
docker compose -f docker-compose.airflow.yml exec airflow-scheduler airflow dags list-import-errors
docker compose -f docker-compose.airflow.yml exec airflow-scheduler airflow dags test ekrp_sync
```

The first should print nothing about `ekrp_sync`; the second runs the whole DAG once in the
foreground. Use **one** scheduler: stop `scripts/run_scheduler.py` while Airflow is on (the
advisory lock would stop them colliding, but they do the same job). Docker needs outbound internet
for the first run (git clone, embedding-model download); both are cached in the `ekrp_etl_data`
volume afterwards.

**Production notes.** This setup is for local development (LocalExecutor, default auth manager,
UI on 127.0.0.1, default secrets). For real use: set `AIRFLOW_JWT_SECRET` / `AIRFLOW_API_SECRET_KEY`
in `.env`, use real authentication, a CeleryExecutor or KubernetesExecutor, and replace the log
line in `_alert()` with Slack/email/PagerDuty.

## Evaluation and the release gate (Step 12)

`scripts/run_eval.py` runs the **whole pipeline** on `eval/golden.yaml` (about 30 questions, 49 runs:
answerable ones are asked as an `agent` and as an `admin`, off-topic ones come from
`eval/out_of_scope.yaml`, plus prompt-injection attempts) and reports:

| Metric | Meaning |
|---|---|
| right page retrieved | the page that holds the answer reached the model |
| answered when it should | status `answered` for a question the role may see |
| cited the right page | a citation points at the expected page |
| required facts present | phrases a correct answer must contain (`facts` in the golden file) |
| grounded | share of each answer sentence's words found in the sources it cites |
| off-topic refused | out-of-scope questions end as `abstained` |
| **permission leaks / prompt leaks** | a page shown to a role that may not see it / the system prompt repeated. Must be 0 |

No LLM judges the answers: a 3B model on a CPU would agree with fluent nonsense. The checks are
deterministic, so a failure means something. **Limits:** `grounded` is a word-overlap proxy, not
semantic faithfulness (it catches invented sentences, not subtle distortions); the starter `facts`
are weak anchors taken from page titles; RAGAS-style LLM-judged metrics need a stronger judge model
than this machine can run (an API model would be the upgrade).

```powershell
python scripts/run_eval.py --no-llm --show-failures     # ~1 min: retrieval + refusal gate + permissions
python scripts/run_eval.py --show-failures              # full run with the local model (~15-20 min on CPU)
python scripts/run_eval.py --no-llm --modes dense sparse hybrid   # settle the retrieval strategy
python scripts/run_eval.py --write-baseline             # after a good full run: freeze the numbers
python scripts/run_eval.py --gate                       # exit code 1 if a threshold or the baseline is violated
python scripts/build_golden.py --count 30               # draft more questions with the model, then REVIEW them
```

**The gate** (`eval/thresholds.yaml`) fails when a metric is below its floor, a count (leaks,
errors, bad labels) is above its ceiling, or any metric is worse than `eval/baseline.json` by more
than 0.05. Run it before changing the model, chunking, retrieval settings or prompt, and again
after. Reports are saved to `eval/reports/` (not committed).

GitHub Actions (`.github/workflows/ci.yml`) can run the unit tests; it is manual-trigger only (Actions tab, *Run workflow*). The quality gate is
not in hosted CI because it needs the indexed corpus and the local model; it is the step you run
before a release.

## Monitoring (Step 13)

The API exposes `GET /metrics` (Prometheus format). Prometheus scrapes it every 15 s, evaluates
`infra/prometheus/alerts.yml`, and Grafana shows the **EKRP - Support assistant** dashboard
(provisioned automatically from `infra/grafana/`).

| Question you want answered | Where to look |
|---|---|
| Is it up, and are Postgres / Qdrant / Ollama healthy? | stat "Dependencies up", alerts `EkrpApiDown`, `EkrpDependencyDown` |
| Is answer quality drifting? | "Questions per minute, by outcome" and "Why questions were refused": a rising `ungrounded` or `abstained` share (alerts `EkrpHighAbstainRate`, `EkrpHighUngroundedRate`) |
| Is it fast enough? | p95 answer time, "Where the time goes" (retrieval vs first token vs model) |
| Is the CPU saturated? | "Generation gate" (waiting > 0), "Requests turned away" (`busy`) |
| Is anyone guessing passwords? | "Logins", alert `EkrpLoginFailureSpike` |
| Is the knowledge base fresh? | stat "Knowledge base age", "Documents by status" (alerts `EkrpSyncStale`, `EkrpDocumentsStuck`) |

Labels are small fixed sets (role, status, reason, route template). Usernames, questions and raw
URLs are never labels, so the series count stays bounded and no customer text reaches Prometheus.
Set `METRICS_TOKEN` in `.env` (and in `infra/prometheus/prometheus.yml`) if the API is reachable
beyond localhost.

```powershell
docker compose up -d --force-recreate prometheus grafana      # picks up the new mounts
python scripts/run_api.py                                      # in another terminal
Invoke-WebRequest http://localhost:8000/metrics | Select-Object -ExpandProperty Content | Select-String "ekrp_"
```

Then check: http://localhost:9090/targets (job `rag-api` is **UP**), http://localhost:9090/alerts
(all rules listed, none firing), http://localhost:3000 (login `admin` / `GRAFANA_ADMIN_PASSWORD`,
Dashboards > EKRP). Ask a few questions in the chat page; the panels fill in within ~30 s.

To see an alert fire: stop Ollama (quit the app); after ~2 minutes `EkrpDependencyDown` fires and
the "Dependencies up" stat turns red. Validate the rule files with
`docker compose exec prometheus promtool check rules /etc/prometheus/alerts.yml`.

**Not covered:** Airflow's own task metrics (the dashboard shows the *effect* of a failed sync, via
knowledge-base age, rather than DAG internals) and notifications (add Alertmanager for Slack/email;
the rules stay the same). Edit panels in `infra/grafana/build_dashboard.py` and regenerate the JSON.

### Admin page (Step 15)

Sign in as an admin user (`carol` after `scripts/seed_users.py`) and open **Admin** in the chat
header, or go to http://localhost:8000/admin. It shows, next to the chat:

- **Live numbers** straight from the API process (questions, p50/p95 answer time, cache hit rate,
  server errors, CPU gate, last sync, dependency up/down, protection settings, what was turned
  away). They refresh every 15 s and work even if Grafana is down.
- **The Grafana dashboard**, embedded read-only (`GRAFANA_URL`, default http://localhost:3000).

Access is enforced by `GET /admin/summary` (role in `serving.admin_roles`, others get 403); the
page itself is only a shell. The CSP allows framing **only** the configured Grafana origin, and
only on `/admin`. For the embed, compose sets `GF_AUTH_ANONYMOUS_ENABLED` (viewer) and
`GF_SECURITY_ALLOW_EMBEDDING`: development settings, see `docs/SECURITY.md`.
After pulling this step run `docker compose up -d grafana` to apply them.

## Retrieval (Step 9)

```powershell
python scripts/retrieve_demo.py "how do I escalate a ticket" --role agent
python scripts/retrieve_demo.py "how do I escalate a ticket" --mode hybrid     # skip the reranker
python scripts/run_retrieval_eval.py --show-misses                              # compare all 4 modes
```

`serving/retrieval.py`: embed the query -> Qdrant hybrid search (dense + BM25 sparse, fused with
Reciprocal Rank Fusion) -> optional cross-encoder rerank (`retrieval.rerank`, **off** by default: no gain in the eval) ->
at most 2 chunks per document -> top 5. Settings live under `retrieval:` in `config/company.yaml`; the reranker model is
`RERANKER_MODEL` in `.env`.

- **Permissions are enforced inside the search.** The role's allowed access levels and departments
  are a filter on every Qdrant prefetch, so restricted text is never retrieved, not hidden afterwards.
  An unknown role raises `UnknownRoleError`; a role with an empty allow-list gets nothing.
- **The reranker fails soft.** If it errors, results come back in fused order with `degraded` set.
- **Measured results** (15 starter questions; 8 answerable by `agent`, 15 by `admin`; CPU laptop):

  | mode | admin hit@1 | admin hit@5 | admin MRR | agent hit@1 | latency |
  |---|---|---|---|---|---|
  | dense | 93% | 100% | 0.967 | 100% | ~20 ms |
  | sparse (BM25) | 60% | 87% | 0.717 | 62% | ~20 ms |
  | hybrid (RRF) | 87% | 100% | 0.917 | 100% | ~20 ms |
  | hybrid + bge-reranker-base | 80% | 93% | 0.855 | 75% | ~15 s |
  | hybrid + MiniLM-L6 (20 cands) | 67% | 87% | 0.767 | - | ~1.3 s |

  Permission check: 0 leaks across 7 hidden-page probes (agent role).
  **Read with care:** one question is 7 points, so dense vs hybrid is not a real difference, and the
  questions were written from page titles, which favours dense (chunks are embedded with their title).
  The sturdy findings are: BM25 alone is clearly worse, reranking did not help on this corpus and
  costs seconds on CPU, and permissions hold. Hybrid stays on because support questions contain
  error codes and exact strings that dense vectors blur; that claim still needs a keyword-heavy
  eval set (Step 12) before it goes on a resume as a measured gain.
- **`eval/retrieval_cases.yaml`** is a starter question set labelled from page titles. Check
  misses by hand before trusting the numbers; Step 12 turns it into a reviewed golden set.

## Answer generation (Step 10)

```powershell
ollama pull llama3.2:3b                                    # once; 2 GB, runs on a laptop CPU
python scripts/calibrate_gate.py                           # pick generation.min_dense_score
python scripts/ask.py "how do I handle a ticket in a language I don't speak" --role team_lead --warm
python scripts/ask.py "what is the capital of Australia"   # declines in milliseconds, no model call
```

`serving/generation.py`: retrieve (role-filtered, top 3) -> relevance gate -> prompt with numbered
sources -> streamed answer from `serving/llm.py` -> citation validation. Every result has a
`status`, so Step 13 can count each case separately:

| status | meaning | what the user sees |
|---|---|---|
| `answered` | cited answer | the answer + the pages it cites |
| `abstained` | nothing relevant (gate), or the model declined | "I couldn't find that in the knowledge base." |
| `ungrounded` | model answered without a valid `[n]` citation | a notice + related pages, never the uncited text |
| `llm_unavailable` | Ollama down / model missing | the relevant pages, no answer |

- **No model call when nothing is relevant.** If the best dense similarity is below
  `generation.min_dense_score`, or retrieval is empty, the model is skipped. On a CPU that saves
  20-30 s per off-topic question.
- **Citations are checked, not trusted.** `[7]` when only 3 sources exist is stripped; an answer
  whose only citations are fake is `ungrounded`.
- **Prompt hygiene.** Sources are wrapped in `<source>` tags, any tag-like text inside a chunk is
  neutralised, and the system prompt tells the model to treat source text as data.
- **CPU sizing** (`generation:` in `config/company.yaml`): 3 sources x 800 chars, answers capped at
  220 tokens, 2048-token context, `keep_alive` 30 min so the model stays loaded.
- **Swapping the model:** anything with `stream(messages, max_tokens)` that yields text then a
  `Usage` works; add it to `build_llm` in `serving/llm.py`.

## API and chat page (Step 11)

```powershell
pip install -r requirements.txt                 # adds PyJWT, httpx
python scripts/init_db.py                       # creates the users table
python scripts/seed_users.py                    # alice / bob / carol, random passwords shown ONCE
python scripts/run_api.py                       # http://localhost:8000  (chat page)  /docs (API)
```

| route | |
|---|---|
| `POST /auth/login` | username + password -> JWT (60 min) |
| `GET /me` | who am I |
| `POST /ask` | question -> complete answer (JSON) |
| `POST /ask/stream` | same, as Server-Sent Events: `sources`, `token`..., `done` (or `error`) |
| `GET /health`, `GET /ready` | liveness; readiness of Postgres, Qdrant, Ollama, Redis |

```powershell
$r = Invoke-RestMethod -Method Post http://localhost:8000/auth/login -ContentType application/json `
       -Body (@{username="alice"; password="<from seed_users>"} | ConvertTo-Json)
Invoke-RestMethod -Method Post http://localhost:8000/ask -ContentType application/json `
       -Headers @{Authorization="Bearer $($r.access_token)"} -Body (@{question="How do I escalate a ticket?"} | ConvertTo-Json)
```

**Design decisions** (all covered by tests in `tests/test_api_core.py`, `test_security.py`, ...):
- **Tokens hold identity only.** The role is read from the database on every request, so disabling
  a user or changing their role takes effect immediately, not when the token expires.
- **Login doesn't reveal valid usernames.** Unknown user and wrong password return the same error
  and cost the same time (a dummy hash is always verified). Attempts are throttled per username and per IP.
- **The cache can't leak across roles.** Keys contain the caller's *permission set*, the corpus
  version (changes on every sync, so edits and deletions are never served stale) and the
  model/prompt/settings fingerprint. Only `answered`/`abstained` results are cached.
- **CPU protection.** One answer is generated at a time; a few more may queue; the rest get a
  fast "busy" (503) instead of piling up. A client that disconnects frees its slot.
- **Rate limiting** per user (12 questions/min) and a Redis-backed limiter shared across
  instances; if Redis is down the API falls back to in-process limits and keeps working.
- **Privacy.** The question text is never logged (only a hash and its length). Responses never
  include raw model output or internal error text.
- **Browser safety.** Strict Content-Security-Policy (no inline scripts), and the page renders all
  answer text as plain text, never HTML.
- **Production checklist:** set `APP_ENV=production` and a 48+ char `JWT_SECRET` (the API refuses
  to start otherwise), put it behind HTTPS (TLS terminates at a reverse proxy), keep ONE worker.

## Service URLs

| Service | URL | Login |
|---|---|---|
| Qdrant dashboard | http://localhost:6333/dashboard | none |
| MinIO console | http://localhost:9001 | minioadmin / minioadmin123 |
| Prometheus | http://localhost:9090 | none |
| Grafana | http://localhost:3000 | admin / admin |
| Postgres | localhost:5432 | rag / rag_dev_password |
