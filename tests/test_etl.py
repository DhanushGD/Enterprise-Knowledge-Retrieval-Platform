"""Step 8b: the ETL stages that Airflow calls. Fakes only - no Postgres, Qdrant or Airflow."""
import json

import pytest

from connectors.local_folder import LocalFolderConnector
from core.errors import SyncAlreadyRunning
from ingestion.indexer import IndexingService
from ingestion.sync import SyncService
from orchestration import etl
from tests.fakes import FakeEmbedder, FakeIndex, FakeRegistry, FakeStore
from tests.test_sync import TAXONOMY, md


class Env:
    def __init__(self, tmp_path, files, embedder=None):
        self.root = tmp_path
        self.store, self.registry, self.index = FakeStore(), FakeRegistry(), FakeIndex()
        for name, content in files.items():
            self.write(name, content)
        self.connector = LocalFolderConnector("docs", tmp_path, default_department="support")
        self.service = SyncService(
            [self.connector], self.store, self.registry, self.index,
            IndexingService(self.registry, embedder or FakeEmbedder(), self.index), TAXONOMY,
        )

    def write(self, name, content):
        (self.root / name).write_bytes(content)

    def remove(self, name):
        (self.root / name).unlink()

    def statuses(self):
        return {r.source_path: r.status for r in self.registry.rows.values()}


def docs(n, prefix="d"):
    return {f"{prefix}{i}.md": md(f"Doc{i}") for i in range(n)}


# ---------------------------------------------------------------- the happy path

def test_every_stage_in_order_gives_a_consistent_index(tmp_path):
    env = Env(tmp_path, docs(3))
    out = etl.run_all(env.service)
    steps = {s["stage"]: s for s in out["steps"]}
    assert [s["stage"] for s in out["steps"]] == ["preflight", "land", "parse", "chunk", "index", "verify"]
    assert (steps["land"]["new"], steps["parse"]["ok"], steps["index"]["ok"]) == (3, 3, 3)
    assert steps["verify"]["vectors"] == steps["verify"]["expected_vectors"] == env.index.count() > 0
    assert set(env.statuses().values()) == {"indexed"}


def test_stages_called_one_by_one_match_a_single_run(tmp_path):
    """Airflow runs each stage as its own process; state must flow through the registry only."""
    env = Env(tmp_path, docs(2))
    assert etl.sources(env.service)["sources"] == ["docs"]
    assert etl.land(env.service, "docs")["new"] == 2
    assert set(env.statuses().values()) == {"new"}
    etl.parse(env.service)
    assert set(env.statuses().values()) == {"parsed"}
    etl.chunk(env.service)
    assert set(env.statuses().values()) == {"chunked"}
    etl.index(env.service)
    assert set(env.statuses().values()) == {"indexed"}
    assert etl.verify(env.service)["vectors"] == env.index.count()


def test_a_second_run_finds_nothing_to_do_and_still_verifies(tmp_path):
    env = Env(tmp_path, docs(2))
    etl.run_all(env.service)
    again = {s["stage"]: s for s in etl.run_all(env.service)["steps"]}
    assert again["land"]["unchanged"] == 2 and again["land"]["new"] == 0
    assert (again["parse"]["ok"], again["chunk"]["ok"], again["index"]["ok"]) == (0, 0, 0)


def test_limit_is_a_partial_trial_that_deletes_nothing(tmp_path):
    env = Env(tmp_path, docs(3))
    etl.run_all(env.service)
    env.remove("d2.md")
    result = etl.land(env.service, "docs", limit=1)
    assert result["deleted"] == 0
    assert env.statuses()["d2.md"] == "indexed"


# ---------------------------------------------------------------- failures and their class

def test_unknown_source_is_a_usage_error_not_something_to_retry(tmp_path):
    env = Env(tmp_path, docs(1))
    with pytest.raises(ValueError):
        etl.land(env.service, "nope")
    assert etl.classify(ValueError("x")) == etl.EXIT_FATAL


def test_unlistable_source_is_retryable_and_deletes_nothing(tmp_path):
    env = Env(tmp_path, docs(2))
    etl.run_all(env.service)
    env.service.connectors.append(LocalFolderConnector("ghost", tmp_path / "missing"))
    with pytest.raises(etl.EtlError, match="ghost.*could not be listed") as info:
        etl.land(env.service, "ghost")
    assert etl.classify(info.value) == etl.EXIT_RETRY
    assert set(env.statuses().values()) == {"indexed"}


def test_blocked_mass_deletion_is_fatal_until_forced(tmp_path):
    env = Env(tmp_path, docs(10))
    etl.run_all(env.service)
    for i in range(8):
        env.remove(f"d{i}.md")

    with pytest.raises(etl.EtlFatal, match="force_deletes") as info:
        etl.land(env.service, "docs")
    assert etl.classify(info.value) == etl.EXIT_FATAL
    assert sum(s == "indexed" for s in env.statuses().values()) == 10

    assert etl.land(env.service, "docs", force_deletes=True)["deleted"] == 8
    etl.run_all(env.service)                                   # and the rest stays consistent


def test_infrastructure_down_is_reported_per_service(tmp_path):
    env = Env(tmp_path, docs(1))

    def boom():
        raise ConnectionError("refused")

    env.index.healthy = boom
    with pytest.raises(etl.EtlError, match=r"qdrant \(ConnectionError: refused\)") as info:
        etl.preflight(env.service)
    assert "postgres" not in str(info.value)
    assert etl.classify(info.value) == etl.EXIT_RETRY


def test_a_stage_fails_when_too_many_documents_fail_and_recovers_on_retry(tmp_path):
    flaky = FakeEmbedder(fail_on="Doc1")
    env = Env(tmp_path, docs(4), embedder=flaky)
    etl.land(env.service, "docs"); etl.parse(env.service); etl.chunk(env.service)

    with pytest.raises(etl.EtlError, match=r"index: 1 of 4 documents failed"):
        etl.index(env.service)
    assert sorted(env.statuses().values()) == ["index_failed", "indexed", "indexed", "indexed"]

    flaky.fail_on = None                                       # the transient problem is gone
    assert etl.index(env.service)["ok"] == 1                   # only the failed one is redone
    assert set(env.statuses().values()) == {"indexed"}


def test_failures_under_the_limit_do_not_fail_the_stage(tmp_path):
    env = Env(tmp_path, docs(4), embedder=FakeEmbedder(fail_on="Doc1"))
    etl.land(env.service, "docs"); etl.parse(env.service); etl.chunk(env.service)
    result = etl.index(env.service, max_fraction=0.5)
    assert (result["ok"], result["failed"]) == (3, 1) and result["failures"]


# ---------------------------------------------------------------- the final gate

def test_verify_catches_a_vector_that_vanished(tmp_path):
    env = Env(tmp_path, docs(3))
    etl.run_all(env.service)
    del env.index.points[next(iter(env.index.points))]
    with pytest.raises(etl.EtlFatal, match="points but the registry expects"):
        etl.verify(env.service)


def test_verify_catches_a_deleted_document_that_is_still_searchable(tmp_path):
    env = Env(tmp_path, docs(3))
    etl.run_all(env.service)
    ghost = next(iter(env.index.points.values()))
    env.index.points["ghost"] = ghost                          # an extra, orphan point
    with pytest.raises(etl.EtlFatal):
        etl.verify(env.service)


def test_verify_refuses_to_bless_an_empty_index(tmp_path):
    env = Env(tmp_path, {})
    with pytest.raises(etl.EtlFatal, match="no live documents"):
        etl.verify(env.service)


def test_verify_tolerates_a_few_stuck_documents_but_warns(tmp_path):
    env = Env(tmp_path, docs(30))
    etl.run_all(env.service)
    stuck = next(iter(env.registry.rows.values()))
    stuck.status = "parse_failed"                              # 1 of 30 = 3.3% < 5%
    result = etl.verify(env.service)
    assert "stuck" in result["warning"]


def test_verify_fails_when_too_many_are_stuck(tmp_path):
    env = Env(tmp_path, docs(10))
    etl.run_all(env.service)
    for row in list(env.registry.rows.values())[:3]:
        row.status = "chunk_failed"
    with pytest.raises(etl.EtlFatal, match="stuck in a failed state"):
        etl.verify(env.service)


# ---------------------------------------------------------------- locking

def test_a_running_sync_blocks_the_stage_and_the_lock_is_always_released(tmp_path):
    env = Env(tmp_path, docs(1))
    env.registry.lock_held = True                              # e.g. someone ran run_sync.py
    with pytest.raises(SyncAlreadyRunning):
        etl.land(env.service, "docs")
    assert env.registry.lock_held                              # the other holder is untouched

    env.registry.lock_held = False
    env.index.healthy = lambda: None
    etl.land(env.service, "docs")
    assert not env.registry.lock_held                          # released after success
    env.service.connectors.append(LocalFolderConnector("ghost", tmp_path / "missing"))
    with pytest.raises(etl.EtlError):
        etl.land(env.service, "ghost")
    assert not env.registry.lock_held                          # ...and after failure


# ---------------------------------------------------------------- the CLI contract

@pytest.fixture
def cli(monkeypatch, capsys):
    monkeypatch.setattr(etl, "_max_fraction", lambda: etl.DEFAULT_MAX_FAILED_FRACTION)

    def run(env, *argv):
        code = etl.main(list(argv), service_factory=lambda: env.service)
        out = capsys.readouterr().out.strip().splitlines()
        return code, json.loads(out[-1])

    return run


def test_cli_success_prints_json_on_the_last_stdout_line(tmp_path, cli):
    env = Env(tmp_path, docs(2))
    code, payload = cli(env, "land", "--source", "docs")
    assert code == 0 and payload["ok"] is True and payload["new"] == 2
    code, payload = cli(env, "all")
    assert code == 0 and payload["steps"][-1]["stage"] == "verify"


def test_cli_exit_codes_tell_the_orchestrator_whether_to_retry(tmp_path, cli):
    env = Env(tmp_path, docs(10))
    cli(env, "all")

    env.registry.lock_held = True
    code, payload = cli(env, "land", "--source", "docs")
    assert code == etl.EXIT_RETRY and "SyncAlreadyRunning" in payload["error"] and not payload["ok"]
    env.registry.lock_held = False

    code, payload = cli(env, "land")                           # missing --source: a usage error
    assert code == etl.EXIT_FATAL and "needs --source" in payload["error"]

    for i in range(8):
        env.remove(f"d{i}.md")
    code, payload = cli(env, "land", "--source", "docs")
    assert code == etl.EXIT_FATAL and "force_deletes" in payload["error"]
    code, payload = cli(env, "land", "--source", "docs", "--force-deletes")
    assert code == 0 and payload["deleted"] == 8


def test_cli_limit_flag_reaches_the_landing_stage(tmp_path, cli):
    env = Env(tmp_path, docs(5))
    code, payload = cli(env, "land", "--source", "docs", "--limit", "2")
    assert code == 0 and payload["new"] == 2


def test_unexpected_errors_default_to_retry(tmp_path, cli):
    env = Env(tmp_path, docs(1))
    env.registry.status_counts = lambda: (_ for _ in ()).throw(ConnectionError("db down"))
    code, payload = cli(env, "preflight")
    assert code == etl.EXIT_RETRY and "postgres" in payload["error"]


# ---------------------------------------------------------------- lazy embedder

def test_lazy_embedder_does_not_load_a_model_until_it_is_used(monkeypatch):
    from ingestion import runner

    loads = []

    class Model:
        dimension = 8

        def embed_documents(self, texts):
            return ["v"] * len(texts)

    monkeypatch.setattr(runner, "get_embedder", lambda: loads.append(1) or Model())
    lazy = runner.LazyEmbedder()
    assert loads == []
    assert lazy.dimension == 8 and lazy.embed_documents(["a", "b"]) == ["v", "v"]
    assert len(loads) == 2


@pytest.mark.parametrize("stage", ["parse", "chunk", "index"])
def test_every_writing_stage_respects_the_sync_lock(tmp_path, stage):
    env = Env(tmp_path, docs(1))
    env.registry.lock_held = True
    with pytest.raises(SyncAlreadyRunning):
        getattr(etl, stage)(env.service)
