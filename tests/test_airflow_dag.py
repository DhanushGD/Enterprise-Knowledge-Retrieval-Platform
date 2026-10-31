"""Step 8b: the Airflow DAG file, loaded against a tiny stand-in for Airflow.

Airflow is large and (on Windows) only runs in Docker, so these tests don't need it. They check
what is ours: the shape of the pipeline, how a stage's exit code becomes retry / no-retry, and
that the schedule comes from config. Whether Airflow itself accepts the file is checked inside
the container: `airflow dags list-import-errors` (README, "Airflow").
"""
import importlib.util
import subprocess
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
DAG_FILE = ROOT / "orchestration" / "dags" / "ekrp_sync.py"


class FailException(Exception):
    pass


def install_fake_airflow(monkeypatch):
    record = types.SimpleNamespace(dag_kwargs=None, tasks={}, edges=set(), mapped=[])

    class Ref:
        def __init__(self, name):
            self.name = name

        def __rshift__(self, other):
            record.edges.add((self.name, other.name))
            return other

    class Task:
        def __init__(self, fn, options):
            self.fn, self.options, self.name = fn, options, fn.__name__
            record.tasks[self.name] = self

        def __call__(self, *args, **kwargs):
            return Ref(self.name)

        def expand(self, **kwargs):
            record.mapped.append((self.name, sorted(kwargs)))
            for value in kwargs.values():
                record.edges.add((value.name, self.name))
            return Ref(self.name)

    def task(fn=None, **options):
        return Task(fn, {}) if fn else (lambda f: Task(f, options))

    def dag(**kwargs):
        record.dag_kwargs = kwargs
        return lambda fn: fn

    modules = {
        "airflow": types.ModuleType("airflow"),
        "airflow.sdk": types.SimpleNamespace(dag=dag, task=task, Param=lambda default, **kw: (default, kw)),
        "airflow.exceptions": types.SimpleNamespace(AirflowFailException=FailException),
    }
    if importlib.util.find_spec("pendulum") is None:
        modules["pendulum"] = types.SimpleNamespace(
            datetime=lambda *a, tz=None: ("datetime", a, tz))
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
    return record


@pytest.fixture
def loaded(monkeypatch):
    monkeypatch.setenv("EKRP_PROJECT_DIR", str(ROOT))
    record = install_fake_airflow(monkeypatch)
    spec = importlib.util.spec_from_file_location("ekrp_sync_under_test", DAG_FILE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, record


def test_pipeline_has_the_intended_shape(loaded):
    _, rec = loaded
    assert set(rec.tasks) == {"preflight", "sources", "land", "parse", "chunk", "index", "verify"}
    assert rec.edges == {
        ("preflight", "sources"), ("sources", "land"), ("land", "parse"),
        ("parse", "chunk"), ("chunk", "index"), ("index", "verify"),
    }
    assert rec.mapped == [("land", ["source"])]            # one land task per source


def test_dag_is_safe_by_default(loaded):
    _, rec = loaded
    kw = rec.dag_kwargs
    assert kw["dag_id"] == "ekrp_sync"
    assert kw["catchup"] is False and kw["max_active_runs"] == 1
    assert kw["default_args"]["retries"] >= 1 and kw["default_args"]["retry_exponential_backoff"]
    assert set(kw["params"]) == {"limit", "force_deletes"}
    assert kw["params"]["force_deletes"][0] is False       # never forces by default
    assert rec.tasks["land"].options["max_active_tis_per_dag"] == 1
    assert rec.tasks["verify"].options["retries"] == 0     # a failed gate is not retried
    for task in rec.tasks.values():
        assert "execution_timeout" in task.options         # nothing can hang forever


def test_schedule_and_timezone_come_from_config(loaded):
    module, _ = loaded
    assert module.CRON == "0 2 * * *"
    assert module.TIMEZONE == "Asia/Kolkata"


# ------------------------------------------------------------ exit code -> Airflow behaviour

def fake_run(monkeypatch, module, code, stdout):
    monkeypatch.setattr(
        module.subprocess, "run",
        lambda cmd, **kw: subprocess.CompletedProcess(cmd, code, stdout=stdout))


def test_success_returns_the_json_line_and_ignores_other_stdout(loaded, monkeypatch):
    module, _ = loaded
    fake_run(monkeypatch, module, 0, 'some noise\n{"ok": true, "stage": "land", "new": 3}\n')
    assert module._run_stage("land", "--source", "x") == {"ok": True, "stage": "land", "new": 3}


def test_retryable_failure_raises_a_normal_error(loaded, monkeypatch):
    module, _ = loaded
    fake_run(monkeypatch, module, 10, '{"ok": false, "error": "EtlError: qdrant down"}\n')
    with pytest.raises(RuntimeError, match="qdrant down") as info:
        module._run_stage("preflight")
    assert not isinstance(info.value, FailException)


def test_fatal_failure_raises_the_no_retry_exception(loaded, monkeypatch):
    module, _ = loaded
    fake_run(monkeypatch, module, 20, '{"ok": false, "error": "refused to delete 8 of 10"}\n')
    with pytest.raises(FailException, match="refused to delete"):
        module._run_stage("land", "--source", "docs")


def test_a_crash_with_no_json_still_fails_and_retries(loaded, monkeypatch):
    module, _ = loaded
    fake_run(monkeypatch, module, 1, "Traceback...\n")
    with pytest.raises(RuntimeError, match="exited with code 1"):
        module._run_stage("parse")


def test_land_passes_the_manual_trigger_options_to_the_cli(loaded, monkeypatch):
    module, rec = loaded
    seen = []
    monkeypatch.setattr(module, "_run_stage", lambda stage, *args: seen.append((stage, args)) or {})
    land = rec.tasks["land"].fn
    land("docs", params={"limit": 0, "force_deletes": False})
    land("docs", params={"limit": 5, "force_deletes": True})
    land("docs")
    assert seen == [
        ("land", ("--source", "docs")),
        ("land", ("--source", "docs", "--limit", "5", "--force-deletes")),
        ("land", ("--source", "docs")),
    ]


def test_subprocess_gets_the_project_on_its_path(loaded, monkeypatch):
    module, _ = loaded
    captured = {}

    def run(cmd, **kw):
        captured.update(cmd=cmd, **kw)
        return subprocess.CompletedProcess(cmd, 0, stdout='{"ok": true}')

    monkeypatch.setattr(module.subprocess, "run", run)
    module._run_stage("sources")
    assert captured["cmd"][1:4] == ["-m", "orchestration.etl", "sources"]
    assert captured["env"]["PYTHONPATH"] == str(ROOT) and captured["cwd"] == str(ROOT)
    assert captured["stdout"] == subprocess.PIPE and "stderr" not in captured   # logs stream through
