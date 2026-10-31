"""EKRP nightly knowledge-base sync, orchestrated by Airflow.

    preflight -> sources -> land[one task per source] -> parse -> chunk -> index -> verify

Every box calls `python -m orchestration.etl <stage>` inside the project's own virtualenv, so
this file only needs Airflow itself (it imports nothing from the project) and the ETL code stays
runnable and testable without Airflow. The CLI reports its outcome as a JSON line plus an exit
code: 0 = ok, 20 = retrying cannot help (a blocked mass deletion, an inconsistent index) and
anything else is retried with backoff.

Trigger it by hand from the UI ("Trigger DAG w/ config") to set:
  limit          land at most N documents per source (quick trial; also disables deletions)
  force_deletes  allow removing a large share of a source after a blocked-deletion failure

Do not run this together with `scripts/run_scheduler.py`: both do the same job.
"""
from __future__ import annotations

import json
import logging
import os
import subprocess
from datetime import timedelta
from pathlib import Path

import pendulum
import yaml

try:                                                     # Airflow 3
    from airflow.sdk import dag, task
except ImportError:                                      # Airflow 2.x
    from airflow.decorators import dag, task

try:
    from airflow.sdk import Param
except ImportError:
    from airflow.models.param import Param

try:
    from airflow.exceptions import AirflowFailException
except ImportError:                                      # pragma: no cover
    from airflow.sdk.exceptions import AirflowFailException

log = logging.getLogger(__name__)

PROJECT_DIR = os.environ.get("EKRP_PROJECT_DIR", "/opt/ekrp")
PYTHON = os.environ.get("EKRP_PYTHON", "/opt/airflow/ekrp-venv/bin/python")
EXIT_FATAL = 20


# ------------------------------------------------------------------ calling the ETL CLI

def _last_json_line(text: str) -> dict:
    for line in reversed((text or "").strip().splitlines()):
        line = line.strip()
        if line.startswith("{"):
            try:
                return json.loads(line)
            except ValueError:
                continue
    return {}


def _run_stage(stage: str, *args: str) -> dict:
    """Run one ETL stage in the project venv. Success returns its JSON result; failure raises
    AirflowFailException (no retry) or RuntimeError (Airflow retries)."""
    cmd = [PYTHON, "-m", "orchestration.etl", stage, *args]
    log.info("running: %s", " ".join(cmd))
    env = {**os.environ, "PYTHONPATH": PROJECT_DIR, "PYTHONUNBUFFERED": "1",
           "PYTHONDONTWRITEBYTECODE": "1"}
    # stderr is inherited, so the stage's logs stream into the Airflow task log as they happen;
    # stdout carries only the final JSON line.
    proc = subprocess.run(cmd, cwd=PROJECT_DIR, env=env, stdout=subprocess.PIPE, text=True)
    payload = _last_json_line(proc.stdout)
    if proc.returncode == 0:
        log.info("result: %s", json.dumps(payload))
        return payload
    message = payload.get("error") or f"{stage} exited with code {proc.returncode} (see log above)"
    if proc.returncode == EXIT_FATAL:
        raise AirflowFailException(message)
    raise RuntimeError(message)


# ------------------------------------------------------------------ schedule from config

def _config() -> dict:
    try:
        return yaml.safe_load((Path(PROJECT_DIR) / "config" / "company.yaml").read_text()) or {}
    except OSError:
        return {}


_SYNC = _config().get("sync", {})
CRON = _SYNC.get("cron", "0 2 * * *")
TIMEZONE = _SYNC.get("timezone", "UTC")


def _schedule():
    try:
        from airflow.timetables.trigger import CronTriggerTimetable

        return CronTriggerTimetable(CRON, timezone=TIMEZONE)
    except ImportError:                                  # pragma: no cover
        return CRON


def _alert(context) -> None:
    """Runs when a task fails after its last retry. Swap the log line for Slack/email/PagerDuty."""
    ti = context.get("task_instance")
    log.error("EKRP sync task failed: %s (try %s)", getattr(ti, "task_id", "?"),
              getattr(ti, "try_number", "?"))


# ------------------------------------------------------------------ the DAG

@dag(
    dag_id="ekrp_sync",
    description="Land, parse, chunk, embed and index the knowledge base (incremental).",
    schedule=_schedule(),
    start_date=pendulum.datetime(2026, 1, 1, tz=TIMEZONE),
    catchup=False,                  # a missed night is not replayed: the next run catches up anyway
    max_active_runs=1,              # never two syncs at once
    dagrun_timeout=timedelta(hours=6),
    params={
        "limit": Param(0, type="integer", minimum=0,
                       description="0 = everything. N = land at most N documents per source "
                                   "(quick trial; deletions are disabled)."),
        "force_deletes": Param(False, type="boolean",
                               description="Allow a deletion the safety check refused "
                                           "(more than 30% of a source, or an empty listing)."),
    },
    default_args={
        "owner": "ekrp",
        "retries": 2,
        "retry_delay": timedelta(minutes=2),
        "retry_exponential_backoff": True,
        "max_retry_delay": timedelta(minutes=20),
        "on_failure_callback": _alert,
    },
    tags=["ekrp", "etl", "rag"],
    doc_md=__doc__,
)
def ekrp_sync():
    @task(execution_timeout=timedelta(minutes=5))
    def preflight() -> dict:
        return _run_stage("preflight")

    @task(execution_timeout=timedelta(minutes=2))
    def sources() -> list[str]:
        return _run_stage("sources")["sources"]

    # One task per source. They run one at a time (the sync lock allows only one writer anyway),
    # and a source that cannot be reached fails on its own, with its own retries.
    @task(max_active_tis_per_dag=1, execution_timeout=timedelta(minutes=45))
    def land(source: str, params: dict | None = None) -> dict:
        params = params or {}
        args = ["--source", source]
        if int(params.get("limit") or 0) > 0:
            args += ["--limit", str(int(params["limit"]))]
        if params.get("force_deletes"):
            args.append("--force-deletes")
        return _run_stage("land", *args)

    @task(execution_timeout=timedelta(minutes=30))
    def parse() -> dict:
        return _run_stage("parse")

    @task(execution_timeout=timedelta(minutes=30))
    def chunk() -> dict:
        return _run_stage("chunk")

    @task(execution_timeout=timedelta(hours=3))      # embedding runs on CPU
    def index() -> dict:
        return _run_stage("index")

    @task(execution_timeout=timedelta(minutes=5), retries=0)
    def verify() -> dict:
        return _run_stage("verify")

    checked = preflight()
    names = sources()
    landed = land.expand(source=names)
    checked >> names
    landed >> parse() >> chunk() >> index() >> verify()


ekrp_sync()
