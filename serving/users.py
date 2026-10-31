"""User storage behind a tiny interface: get / upsert / set_disabled."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class UserRecord:
    username: str
    password_hash: str
    role: str
    disabled: bool = False


def normalize_username(username: str) -> str:
    return username.strip().lower()


class MemoryUserStore:
    def __init__(self):
        self._users: dict[str, UserRecord] = {}

    def get(self, username: str) -> UserRecord | None:
        rec = self._users.get(normalize_username(username))
        return UserRecord(**vars(rec)) if rec else None

    def upsert(self, username: str, password_hash: str, role: str) -> None:
        name = normalize_username(username)
        existing = self._users.get(name)
        self._users[name] = UserRecord(name, password_hash, role, existing.disabled if existing else False)

    def set_disabled(self, username: str, disabled: bool) -> bool:
        rec = self._users.get(normalize_username(username))
        if not rec:
            return False
        rec.disabled = disabled
        return True


class SqlUserStore:
    def __init__(self, engine=None):
        from sqlalchemy.orm import sessionmaker

        from core.db import get_engine

        self.engine = engine or get_engine()
        self._session = sessionmaker(self.engine, expire_on_commit=False)

    def get(self, username: str) -> UserRecord | None:
        from core.db import User

        with self._session() as s:
            row = s.get(User, normalize_username(username))
            return UserRecord(row.username, row.password_hash, row.role, bool(row.disabled)) if row else None

    def upsert(self, username: str, password_hash: str, role: str) -> None:
        from core.db import User

        name = normalize_username(username)
        with self._session.begin() as s:
            row = s.get(User, name)
            if row:
                row.password_hash, row.role = password_hash, role
            else:
                s.add(User(username=name, password_hash=password_hash, role=role, disabled=False))

    def set_disabled(self, username: str, disabled: bool) -> bool:
        from core.db import User

        with self._session.begin() as s:
            row = s.get(User, normalize_username(username))
            if not row:
                return False
            row.disabled = disabled
            return True
