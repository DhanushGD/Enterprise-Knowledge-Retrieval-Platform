# Operations runbook

Commands are PowerShell, run from the project folder with the virtualenv active. Items marked
*(untested here)* were written carefully but not run on your machine; try them on throw-away data first.

## Start, stop, check

```powershell
docker compose up -d                                   # Postgres, Qdrant, Redis, MinIO, Prometheus, Grafana
docker compose -f docker-compose.airflow.yml up -d     # Airflow (optional; replaces run_scheduler.py)
ollama serve                                           # or start the Ollama app; model: llama3.2:3b
python scripts/check_infra.py                          # every service reachable?
python scripts/run_api.py                              # API + chat page on :8000 (one worker)
```

Health: `GET /health` (process alive), `GET /ready` (Postgres, Qdrant, Ollama, Redis answer).
Stop everything: `docker compose down` (keeps data; add `-v` only to **delete** all data).

## Routine tasks

| Task | Command |
|---|---|
| Sync now (all sources) | `python scripts/run_sync.py`, or trigger `ekrp_sync` in Airflow |
| Quick trial sync | `python scripts/run_sync.py --limit 20` (never deletes) |
| Allow a refused mass deletion | `python scripts/run_sync.py --force-deletes`, or trigger the DAG with `force_deletes=true` |
| Add / reset / disable a user | `python scripts/seed_users.py` (new) / `--reset --only alice` / `--disable bob` |
| Change chunk size, re-split everything | edit `chunking:` then `python scripts/run_processing.py --rechunk` then `python scripts/run_indexing.py` |
| Change the embedding model | edit `.env`, then `python scripts/run_indexing.py --recreate` (re-embeds everything; re-run `scripts/calibrate_gate.py`) |
| Remove a source completely | `python scripts/purge_source.py --source <name>` |
| Re-run the quality checks | `python scripts/run_eval.py --no-llm` (1 min) or without `--no-llm` (~10 min) |
| Release gate before a change ships | `python scripts/run_eval.py --gate` |

## Admin page

`/admin` (admin role only) shows live numbers and the Grafana dashboard. Blank Grafana frame: check
`GRAFANA_URL` is the address your **browser** uses, and that Grafana was recreated after the compose
change (`docker compose up -d grafana`). "This page is for administrators": the account's role is not
in `serving.admin_roles`. The numbers reset when the API restarts; Prometheus keeps the history.

## Backups and restore

What matters: **Postgres** (documents, chunks, users, sync history) and **MinIO** (the original
bytes). **Qdrant and Redis are rebuildable**: Qdrant from Postgres + MinIO
(`run_indexing.py --recreate`), Redis is only a cache and rate-limit counters.
For the open-source corpus the originals can also be re-fetched from git; for company data they
may not be recoverable from the source, so back up MinIO.

```powershell
# Postgres (custom format; the file is written inside the container, then copied out)
New-Item -ItemType Directory -Force backups | Out-Null
docker exec rag-postgres pg_dump -U rag -Fc ragdb -f /tmp/ragdb.dump
docker cp rag-postgres:/tmp/ragdb.dump .\backups\ragdb-$(Get-Date -Format yyyyMMdd).dump

# MinIO data volume (name from `docker volume ls`; usually <folder>_minio_data)  (untested here)
docker run --rm -v ekrp_minio_data:/data -v ${PWD}\backups:/backup alpine tar czf /backup/minio.tgz -C /data .
```

Restore *(untested here; rehearse on a spare machine before you need it)*:

```powershell
docker cp .\backups\ragdb-YYYYMMDD.dump rag-postgres:/tmp/ragdb.dump
docker exec rag-postgres pg_restore -U rag -d ragdb --clean --if-exists /tmp/ragdb.dump
# restore the MinIO volume the same way (tar xzf), then rebuild the vector index:
python scripts/run_indexing.py --recreate
python scripts/run_eval.py --no-llm          # confirm retrieval and permissions are healthy
```

Frequency: nightly after the sync is a sensible default. A backup you have never restored is a hope, not a backup.

## When an alert fires

| Alert | Likely cause | First check | Fix |
|---|---|---|---|
| `EkrpApiDown` | API not running / not reachable from Docker | `python scripts/run_api.py` running? http://localhost:8000/health | start it; if only Prometheus cannot reach it, run with `--host 0.0.0.0` and set `METRICS_TOKEN` |
| `EkrpDependencyDown` | the named service is down | `docker compose ps`; `ollama list` | start it; the API recovers without restart |
| `EkrpModelUnavailable` | Ollama stopped or model missing | `ollama list` shows `llama3.2:3b`? | `ollama serve` / `ollama pull llama3.2:3b` |
| `EkrpSlowAnswers` | CPU contention, large prompts | dashboard "Where the time goes" and "Generation gate" | close other heavy apps; lower `context_chunks` or `max_answer_tokens` |
| `EkrpLoadShedding` | more concurrent users than one CPU serves | "Requests turned away" | raise `max_waiting`, or add capacity (see Scaling) |
| `EkrpHighAbstainRate` | users ask outside the corpus, or ingestion/retrieval broke | "Why questions were refused"; `ekrp_documents`; knowledge-base age | `run_eval.py --no-llm`; check the last sync; re-run `calibrate_gate.py` if the corpus changed |
| `EkrpHighUngroundedRate` | model/prompt change, or a degraded model | recent changes; `ollama list` | `run_eval.py --gate`; roll back the change |
| `EkrpLoginFailureSpike` | forgotten password or guessing | API audit log (`login_failed`, IPs) | reset the user; block the IP at the proxy |
| `EkrpSyncStale` | Airflow DAG failing/paused, or the scheduler stopped | Airflow UI > `ekrp_sync` > latest run, the failed task's log | fix the cause and trigger the DAG |
| `EkrpDocumentsStuck` | documents failing parse/chunk/index repeatedly | `python -m orchestration.etl verify`; the failed task's log | fix the document or parser; stuck ones retry on the next sync |
| `EkrpInternalErrors` / `EkrpHttp5xx` | bug or dependency failure mid-request | API log (`ask failed`) | fix; add a regression test |

## Airflow task failed

1. Open the run in the UI, click the red task, read the log (the JSON line at the end has `error`).
2. `Could not be listed` / `infrastructure not reachable` -> a dependency is down; clear the task to retry.
3. `refused to delete ... force_deletes` -> **do not retry blindly.** Check why the source shrank;
   if intended, trigger with `force_deletes=true`.
4. `vector index has N points but the registry expects M` -> run `python scripts/run_indexing.py`
   (re-indexes anything not marked indexed); if it persists, `--recreate`.
5. `SyncAlreadyRunning` -> a manual sync is in progress; wait, then clear the task.

## Scaling (what would have to change)

- More users per second: the 3B-on-CPU model is the limit (~1 answer per 15 s). Options: a GPU box
  or a hosted model behind the same `LLMClient` interface (`serving/llm.py`).
- More than one API process: move the generation gate and the in-process fallbacks to Redis; the
  rate limiter and answer cache already use it when available.
- More documents: Qdrant handles millions of chunks; the sync's per-document loops and the
  single-process embedding step are the first things to parallelise.

## Reproducible installs

`requirements.txt` uses minimum versions. To freeze exactly what works today:
`pip freeze > requirements.lock` and install from that file in CI/production.
