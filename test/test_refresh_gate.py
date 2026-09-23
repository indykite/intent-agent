"""Tests for RefreshGate: per-key coalescing, and the lock-leak-on-submission-failure fix."""

import threading
from concurrent.futures import ThreadPoolExecutor

import pytest

from indykite.intent_agent.refresh_gate import RefreshGate


def test_acquire_blocking_lets_a_second_caller_through_once_the_first_releases():
    """The context manager form must actually serialize access to the same key - a second
    acquire must block until the first `with` block exits.
    """
    gate = RefreshGate()
    entered_first = threading.Event()
    release_first = threading.Event()
    second_acquired = threading.Event()

    def _hold_first():
        with gate.acquire_blocking("key"):
            entered_first.set()
            assert release_first.wait(timeout=5), "never told to release"

    def _try_second():
        assert entered_first.wait(timeout=5), "first never entered"
        with gate.acquire_blocking("key"):
            second_acquired.set()

    first_thread = threading.Thread(target=_hold_first)
    second_thread = threading.Thread(target=_try_second)
    first_thread.start()
    second_thread.start()

    assert entered_first.wait(timeout=5)
    assert not second_acquired.is_set(), "second must not acquire while first still holds it"
    release_first.set()
    second_thread.join(timeout=5)
    assert second_acquired.is_set()
    first_thread.join(timeout=5)


def test_refresh_in_background_if_free_runs_refresh_and_releases_the_lock():
    """The happy path: refresh runs on the executor, and the lock is free again afterward -
    proven by a second acquire_blocking succeeding without hanging.
    """
    gate = RefreshGate()
    executor = ThreadPoolExecutor(max_workers=1)
    ran = threading.Event()

    gate.refresh_in_background_if_free("key", ran.set, executor)
    executor.shutdown(wait=True)

    assert ran.is_set()
    with gate.acquire_blocking("key"):
        pass  # must not hang - proves the lock isn't still held from the refresh above


def test_refresh_in_background_if_free_does_nothing_when_already_held():
    """A second call for the same key, while the first refresh is still in flight, must not
    run its own refresh - the in-flight one is expected to satisfy the next read instead.
    """
    gate = RefreshGate()
    executor = ThreadPoolExecutor(max_workers=2)
    first_started = threading.Event()
    release_first = threading.Event()
    second_ran = threading.Event()

    def _slow_refresh():
        first_started.set()
        assert release_first.wait(timeout=5), "never told to release"

    gate.refresh_in_background_if_free("key", _slow_refresh, executor)
    assert first_started.wait(timeout=5)

    gate.refresh_in_background_if_free("key", second_ran.set, executor)

    release_first.set()
    executor.shutdown(wait=True)
    assert not second_ran.is_set()


def test_refresh_in_background_if_free_releases_the_lock_when_submission_fails():
    """If executor.submit() itself raises (e.g. the executor already shut down), the lock
    acquired just before must still be released, and this must not raise either.
    """
    gate = RefreshGate()
    executor = ThreadPoolExecutor(max_workers=1)
    executor.shutdown(wait=True)  # submit() now raises RuntimeError for any callable

    gate.refresh_in_background_if_free("key", lambda: None, executor)  # must not raise

    fresh_executor = ThreadPoolExecutor(max_workers=1)
    try:
        with gate.acquire_blocking("key"):
            pass  # must not hang - proves the lock was released despite the failed submission
    finally:
        fresh_executor.shutdown(wait=True)


@pytest.mark.parametrize("key", ["a", "b"])
def test_different_keys_do_not_contend(key):
    """Two different keys must have independent locks - acquiring one must never block on the
    other.
    """
    gate = RefreshGate()
    with gate.acquire_blocking(key), gate.acquire_blocking("other-key" if key == "a" else "a"):
        pass  # must not hang


def test_background_refresh_for_one_key_is_not_starved_by_blocked_callers_of_another():
    """A striped (hash % N) lock table could deadlock the shared fetch executor: cold callers
    for a key colliding onto the same stripe as a key mid-refresh would permanently occupy
    every worker blocked on that stripe, so the queued refresh (the only thing that could
    release it) never gets a worker to run on. Per-key locks must make this impossible -
    proven here by saturating every worker with callers for a *different* key than the one
    being refreshed, and confirming the refresh still completes.
    """
    gate = RefreshGate()
    executor = ThreadPoolExecutor(max_workers=2)
    release_other_callers = threading.Event()
    refresh_ran = threading.Event()

    def _other_key_caller():
        assert release_other_callers.wait(timeout=5), "never told to proceed"
        with gate.acquire_blocking("other-key"):
            pass

    # Occupy every worker with callers for a DIFFERENT key, paused just before they contend
    # for its lock - the executor already counts both workers as busy at this point.
    futures = [executor.submit(_other_key_caller) for _ in range(2)]

    gate.refresh_in_background_if_free("refreshing-key", refresh_ran.set, executor)

    # Only now let the other-key callers actually try to acquire - if locks were still
    # striped and collided with "refreshing-key", both would block forever on a lock only
    # the (unstarvable, by executor saturation) queued refresh could release.
    release_other_callers.set()

    assert refresh_ran.wait(timeout=5), "refresh was starved by unrelated callers"
    for future in futures:
        future.result(timeout=5)  # must not hang
    executor.shutdown(wait=True)
