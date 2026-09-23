"""Per-key coalescing for cache refreshes, used by both schema.py and policy.py.

Per-process only: one threading.Lock per key, created on first use and dropped once unused -
a striped (hash % N) table could deadlock the fetch executor on a stripe collision (see
test_refresh_gate.py's deadlock regression test).

A pure Python concurrency utility - stays here (not the Redis-backed a2a package) since it's
backend-agnostic and only used by schema.py/policy.py today.

Multi-replica: each replica has its own gate, so N replicas can redundantly refresh the same
stale key concurrently - wasted calls, not a correctness bug; self-corrects next refresh.

Not a new gap: the dogpile.cache this replaced had the same per-process-only behavior
invisibly (distributed_lock=True was never enabled in this project's prior config).
"""

import logging
import threading
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from contextlib import AbstractContextManager
from types import TracebackType

logger = logging.getLogger(__name__)


class RefreshGate:
    """One threading.Lock per key, refcounted and dropped once unused - bounded by keys
    actively in flight, not by key cardinality over the service's lifetime.

    Memory impact is negligible (~56 bytes/lock, <2MiB at 10k keys, measured via tracemalloc);
    every caller here is funneled through one shared 8-worker executor anyway.
    """

    def __init__(self) -> None:
        """Start with an empty table; locks are created lazily per key."""
        self._table_lock = threading.Lock()
        self._locks: dict[str, threading.Lock] = {}
        self._refcounts: dict[str, int] = {}

    def _checkout(self, key: str) -> threading.Lock:
        """Return key's lock, creating it on first use, and register one more interested caller."""
        with self._table_lock:
            if key not in self._locks:
                self._locks[key] = threading.Lock()
                self._refcounts[key] = 0
            self._refcounts[key] += 1
            return self._locks[key]

    def _checkin(self, key: str) -> None:
        """Unregister one caller, dropping key's table entry once nobody's left interested."""
        with self._table_lock:
            self._refcounts[key] -= 1
            if self._refcounts[key] == 0:
                del self._locks[key]
                del self._refcounts[key]

    def acquire_blocking(self, key: str) -> AbstractContextManager[None]:
        """Block until this key's lock is free, then hold it for the `with` block's duration -
        callers must re-check the cache immediately after acquiring, since another caller may
        have already populated it while this one was waiting.
        """
        return _BlockingAcquisition(self, key, self._checkout(key))

    def refresh_in_background_if_free(
        self, key: str, refresh: Callable[[], None], executor: ThreadPoolExecutor
    ) -> None:
        """Non-blocking: if this key's lock is free, acquire it and run `refresh` on `executor`,
        releasing when done, and return immediately without waiting on it. If it's already
        held (someone else is refreshing this key right now), do nothing - the caller already
        has a stale-but-usable value to return, and the in-flight refresh will satisfy the
        *next* read once it finishes.

        Must not raise: if executor.submit() itself raises (e.g. the executor already shut
        down), the lock is released here too, or every future call for this key would hang.
        Called synchronously from inside get_cached(), so raising here would otherwise turn a
        submission failure into a lost stale value.
        """
        lock = self._checkout(key)
        if not lock.acquire(blocking=False):
            self._checkin(key)
            return

        def _run() -> None:
            try:
                refresh()
            finally:
                lock.release()
                self._checkin(key)

        try:
            executor.submit(_run)
        except Exception:
            lock.release()
            self._checkin(key)
            logger.exception("Failed to submit background refresh for key %r - executor may be shutting down", key)


class _BlockingAcquisition(AbstractContextManager[None]):
    """The context manager acquire_blocking returns - a thin wrapper so RefreshGate itself
    doesn't need to be used as a context manager (it's a shared, long-lived registry, not a
    per-call resource).
    """

    def __init__(self, gate: "RefreshGate", key: str, lock: threading.Lock) -> None:
        self._gate = gate
        self._key = key
        self._lock = lock

    def __enter__(self) -> None:
        self._lock.acquire()

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self._lock.release()
        self._gate._checkin(self._key)  # noqa: SLF001 - RefreshGate's own helper, not external access
        return False
