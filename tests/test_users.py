import pytest

from serving.users import MemoryUserStore, SqlUserStore


def sqlite_store():
    sa = pytest.importorskip("sqlalchemy")
    from core.db import Base
    engine = sa.create_engine("sqlite://")
    Base.metadata.create_all(engine)
    return SqlUserStore(engine)


@pytest.fixture(params=["memory", "sql"])
def store(request):
    return MemoryUserStore() if request.param == "memory" else sqlite_store()


def test_missing_user_is_none(store):
    assert store.get("nobody") is None


def test_upsert_creates_then_updates_and_usernames_ignore_case_and_padding(store):
    store.upsert(" Alice ", "hash1", "agent")
    rec = store.get("ALICE")
    assert (rec.username, rec.password_hash, rec.role, rec.disabled) == ("alice", "hash1", "agent", False)
    store.upsert("alice", "hash2", "admin")
    rec = store.get("alice")
    assert (rec.password_hash, rec.role) == ("hash2", "admin")


def test_disable_and_enable_persist_and_survive_a_password_reset(store):
    store.upsert("bob", "h", "agent")
    assert store.set_disabled("bob", True) is True and store.get("bob").disabled is True
    store.upsert("bob", "h2", "agent")                         # resetting the password...
    assert store.get("bob").disabled is True                   # ...must not silently re-enable
    assert store.set_disabled("bob", False) and store.get("bob").disabled is False
    assert store.set_disabled("ghost", True) is False


def test_returned_records_are_copies(store):
    store.upsert("carol", "h", "admin")
    store.get("carol").role = "tampered"
    assert store.get("carol").role == "admin"
