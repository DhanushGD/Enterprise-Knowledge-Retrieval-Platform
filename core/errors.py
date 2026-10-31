"""Exceptions shared across layers, kept free of heavy imports (SQLAlchemy, settings)."""


class SyncAlreadyRunning(RuntimeError):
    """Another sync process holds the lock."""
