"""Limit how many answers are generated at once.

A CPU running a 3B model serves roughly one answer at a time; ten simultaneous requests would
each take ten times as long and none would finish quickly. So at most `max_concurrent` run, a few
more may wait (`max_waiting`) up to `queue_timeout` seconds, and anything beyond that is turned
away immediately with a "busy" error instead of piling up.
"""
from __future__ import annotations

import threading
from contextlib import contextmanager


class Busy(Exception):
    """No generation slot became free in time."""


class GenerationGate:
    def __init__(self, max_concurrent: int = 1, max_waiting: int = 2, queue_timeout: float = 30.0):
        self.max_concurrent, self.max_waiting, self.queue_timeout = (
            max_concurrent, max_waiting, queue_timeout)
        self._sem = threading.Semaphore(max_concurrent)
        self._lock = threading.Lock()
        self.in_flight = 0
        self.waiting = 0

    def is_saturated(self) -> bool:
        """True when every slot is busy AND the waiting room is full: shed load right away."""
        with self._lock:
            return self.in_flight >= self.max_concurrent and self.waiting >= self.max_waiting

    @contextmanager
    def acquire(self):
        with self._lock:
            self.waiting += 1
        got = self._sem.acquire(timeout=self.queue_timeout)
        with self._lock:
            self.waiting -= 1
            if got:
                self.in_flight += 1
        if not got:
            raise Busy()
        try:
            yield
        finally:
            with self._lock:
                self.in_flight -= 1
            self._sem.release()
