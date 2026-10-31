"""Run the incremental sync on a schedule (default: every night at 02:00 local time).

Run:  python scripts/run_scheduler.py                 # block and sync on the cron schedule
      python scripts/run_scheduler.py --run-now       # sync immediately, then keep the schedule
      python scripts/run_scheduler.py --cron "*/30 * * * *"

The schedule comes from `sync.cron` in config/company.yaml. A failed run is retried with
backoff (30s, 2min), overlapping runs are impossible, and one failure never stops the
scheduler. On Windows you can instead point Task Scheduler at `python scripts/run_sync.py`.
"""
import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger

from core.db import SyncAlreadyRunning
from core.settings import company
from ingestion.runner import run_sync_once
from ingestion.sync import format_report, run_with_retries

log = logging.getLogger("rag.scheduler")


def job() -> None:
    try:
        report = run_with_retries(run_sync_once, attempts=3, base_delay=30,
                                  no_retry=(SyncAlreadyRunning,))
    except SyncAlreadyRunning:
        log.warning("skipped this tick: a previous sync is still running")
    except Exception:
        log.exception("sync failed after all retries; will try again at the next tick")
    else:
        log.info("sync finished\n%s", format_report(report))


parser = argparse.ArgumentParser()
parser.add_argument("--cron", default=company.get("sync", {}).get("cron", "0 2 * * *"))
parser.add_argument("--run-now", action="store_true")
args = parser.parse_args()

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

scheduler = BlockingScheduler()
scheduler.add_job(job, CronTrigger.from_crontab(args.cron), id="sync",
                  max_instances=1, coalesce=True, misfire_grace_time=3600)
log.info("scheduler started; cron = '%s' (Ctrl+C to stop)", args.cron)
if args.run_now:
    job()
try:
    scheduler.start()
except (KeyboardInterrupt, SystemExit):
    log.info("scheduler stopped")
