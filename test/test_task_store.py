"""Tests for RedisTaskStore: save/get/delete, terminal-state TTL, list() filtering and
pagination, and index cleanup on delete - all against fakeredis, exercising the real
ZSET/transaction semantics with no live Redis needed.
"""

import asyncio
import threading

import pytest
from a2a.server.context import ServerCallContext
from a2a.types.a2a_pb2 import ListTasksRequest, Task, TaskState, TaskStatus
from a2a.utils.errors import InvalidParamsError
from a2a.utils.task import encode_page_token
from fakeredis import FakeAsyncRedis, FakeServer
from google.protobuf.timestamp_pb2 import Timestamp

from indykite.a2a.redis.task_store import RedisTaskStore
from indykite.redis.storage import RedisStorageBackend

_ENV_ID = "env-1"
_TERMINAL_TTL_SECONDS = 3600.0
_CONTEXT = ServerCallContext()
_EXPECTED_INDEX_SIZE = 3
_EXPECTED_CONTEXT_MATCH_COUNT = 2
_EXPECTED_TASK_COUNT_BEFORE_PRUNE = 2
_EXPECTED_INDEX_SIZE_BEFORE_TAIL_PRUNE = 21
_MIN_ATTEMPTS_PROVING_A_RETRY_HAPPENED = 2


def _timestamp(milliseconds: int) -> Timestamp:
    """Build a Timestamp for a task's status, for score-ordering/filtering tests."""
    ts = Timestamp()
    ts.FromMilliseconds(milliseconds)
    return ts


def _task(task_id, *, context_id="", state=TaskState.TASK_STATE_WORKING, ms=None):
    """Build a Task with the given id/context_id/status.state, and (optionally) an explicit
    status.timestamp for deterministic score ordering.
    """
    status = TaskStatus(state=state, timestamp=_timestamp(ms)) if ms is not None else TaskStatus(state=state)
    return Task(id=task_id, context_id=context_id, status=status)


def _store(*, terminal_ttl_seconds=_TERMINAL_TTL_SECONDS):
    """A fresh RedisTaskStore over a fakeredis-backed RedisStorageBackend."""
    backend = RedisStorageBackend(FakeAsyncRedis())
    return RedisTaskStore(backend, env_id=_ENV_ID, terminal_ttl_seconds=terminal_ttl_seconds), backend


def test_key_prefix_is_configurable_and_defaults_to_ikg():
    """key_prefix namespaces every Redis key this store writes - defaults to "ikg" (this repo's
    own usage), but a different consumer sharing the same Redis must be able to override it.
    """

    async def _run():
        backend = RedisStorageBackend(FakeAsyncRedis())
        store = RedisTaskStore(backend, env_id=_ENV_ID, terminal_ttl_seconds=_TERMINAL_TTL_SECONDS, key_prefix="other")
        await store.save(_task("t1", ms=1_000), _CONTEXT)

        assert await store.get("t1", _CONTEXT) is not None
        assert await backend.zcard("other_task_idx:{env-1}") == 1
        assert await backend.zcard("ikg_task_idx:{env-1}") == 0

    asyncio.run(_run())


def test_save_get_delete_round_trip():
    """save() then get() must return an equal task; delete() then get() must return None."""

    async def _run():
        store, _backend = _store()
        task = _task("t1", ms=1_000)

        await store.save(task, _CONTEXT)
        assert await store.get("t1", _CONTEXT) == task

        await store.delete("t1", _CONTEXT)
        assert await store.get("t1", _CONTEXT) is None

    asyncio.run(_run())


def test_aclose_closes_the_backend(mocker):
    """aclose() must delegate to the backend's own aclose()."""

    async def _run():
        store, backend = _store()
        backend.aclose = mocker.AsyncMock(wraps=backend.aclose)

        await store.aclose()

        backend.aclose.assert_awaited_once()

    asyncio.run(_run())


def test_get_returns_none_for_a_task_never_saved():
    """A task_id that was never saved must read back as None, not raise."""

    async def _run():
        store, _backend = _store()
        assert await store.get("missing", _CONTEXT) is None

    asyncio.run(_run())


def test_delete_is_a_noop_for_a_task_that_does_not_exist():
    """delete() on an absent task_id must not raise."""

    async def _run():
        store, _backend = _store()
        await store.delete("missing", _CONTEXT)

    asyncio.run(_run())


@pytest.mark.parametrize(
    "state",
    [
        TaskState.TASK_STATE_COMPLETED,
        TaskState.TASK_STATE_CANCELED,
        TaskState.TASK_STATE_FAILED,
        TaskState.TASK_STATE_REJECTED,
    ],
)
def test_save_applies_ttl_only_to_terminal_states(state):
    """A terminal-state task gets terminal_ttl_seconds; a non-terminal one gets no TTL."""

    async def _run():
        store, backend = _store()
        await store.save(_task("terminal", ms=1_000, state=state), _CONTEXT)
        ttl = await backend._client.ttl("ikg_task:{env-1}:terminal")
        assert 0 < ttl <= _TERMINAL_TTL_SECONDS

    asyncio.run(_run())


def test_save_sets_no_ttl_for_a_non_terminal_state():
    """A WORKING (non-terminal) task must have no TTL on its primary record."""

    async def _run():
        store, backend = _store()
        await store.save(_task("in-flight", ms=1_000, state=TaskState.TASK_STATE_WORKING), _CONTEXT)
        assert await backend._client.ttl("ikg_task:{env-1}:in-flight") == -1

    asyncio.run(_run())


def test_save_migrates_the_context_index_when_context_id_changes():
    """Re-saving task_id under a different context_id must remove it from the OLD context's
    index too, not just add it to the new one.
    """

    async def _run():
        store, backend = _store()
        await store.save(_task("t1", context_id="ctx-old", ms=1_000), _CONTEXT)

        await store.save(_task("t1", context_id="ctx-new", ms=2_000), _CONTEXT)

        assert await backend.zcard("ikg_task_idx:{env-1}:ctx:ctx-old") == 0
        assert await backend.zcard("ikg_task_idx:{env-1}:ctx:ctx-new") == 1
        old_ctx_listing = await store.list(ListTasksRequest(page_size=10, context_id="ctx-old"), _CONTEXT)
        assert old_ctx_listing.tasks == []

    asyncio.run(_run())


def test_save_removes_the_context_lookup_key_when_context_is_cleared():
    """Re-saving task_id with an empty context_id must delete the old context-lookup key too
    (review) - build_extra_sets only overwrites it when the new context_id is truthy, so a
    clear would otherwise leave it pointing at the old context forever (no TTL, non-terminal).
    """

    async def _run():
        store, backend = _store()
        await store.save(_task("t1", context_id="ctx-old", ms=1_000), _CONTEXT)
        assert await backend.get("ikg_task_ctx:{env-1}:t1") == b"ctx-old"

        await store.save(_task("t1", context_id="", ms=2_000), _CONTEXT)

        assert await backend.get("ikg_task_ctx:{env-1}:t1") is None

    asyncio.run(_run())


def test_delete_removes_task_from_both_env_and_context_indexes():
    """delete() must clean up the env-wide index and the per-context index too."""

    async def _run():
        store, backend = _store()
        task = _task("t1", context_id="ctx-1", ms=1_000)
        await store.save(task, _CONTEXT)

        await store.delete("t1", _CONTEXT)

        assert await backend.zcard("ikg_task_idx:{env-1}") == 0
        assert await backend.zcard("ikg_task_idx:{env-1}:ctx:ctx-1") == 0

    asyncio.run(_run())


def test_delete_removes_the_context_lookup_key_too():
    """delete() must also drop the context-lookup reverse-index key (review) - left behind, a
    non-terminal task's lookup key (no TTL of its own) would leak in Redis forever, since
    _prune_expired_candidate can only reach it via an index membership delete() already stripped.
    """

    async def _run():
        store, backend = _store()
        task = _task("t1", context_id="ctx-1", ms=1_000)
        await store.save(task, _CONTEXT)
        assert await backend.get("ikg_task_ctx:{env-1}:t1") is not None

        await store.delete("t1", _CONTEXT)

        assert await backend.get("ikg_task_ctx:{env-1}:t1") is None

    asyncio.run(_run())


def test_delete_re_validates_against_a_concurrent_save_before_committing(mocker):
    """delete() must not blind-delete-by-key: a concurrent save() landing between the read and
    the commit must trigger a retry against the fresh value (WATCH), not a commit based on the
    stale read - otherwise the concurrent writer's data is silently destroyed.

    The concurrent save is forced to land - via a background thread blocked on with .join() -
    in the middle of store1.delete("t1")'s own read step.
    """
    server = FakeServer()
    store1 = RedisTaskStore(
        RedisStorageBackend(FakeAsyncRedis(server=server)), env_id=_ENV_ID, terminal_ttl_seconds=_TERMINAL_TTL_SECONDS
    )
    replacement = _task("t1", context_id="ctx-new", ms=2_000, state=TaskState.TASK_STATE_COMPLETED)

    async def _run():
        await store1.save(_task("t1", context_id="ctx-old", ms=1_000), _CONTEXT)

        real_from_string = Task.FromString
        attempts = 0

        def _racy_from_string(encoded):
            nonlocal attempts
            attempts += 1
            if attempts == 1:

                def _concurrent_save():
                    async def _write():
                        writer_store = RedisTaskStore(
                            RedisStorageBackend(FakeAsyncRedis(server=server)),
                            env_id=_ENV_ID,
                            terminal_ttl_seconds=_TERMINAL_TTL_SECONDS,
                        )
                        await writer_store.save(replacement, _CONTEXT)

                    asyncio.run(_write())

                thread = threading.Thread(target=_concurrent_save)
                thread.start()
                thread.join()
            return real_from_string(encoded)

        mocker.patch("indykite.a2a.redis.task_store.Task.FromString", side_effect=_racy_from_string)

        await store1.delete("t1", _CONTEXT)

        assert attempts >= _MIN_ATTEMPTS_PROVING_A_RETRY_HAPPENED
        assert await store1.get("t1", _CONTEXT) is None
        assert await store1._backend.zcard("ikg_task_idx:{env-1}:ctx:ctx-new") == 0
        assert await store1._backend.zcard("ikg_task_idx:{env-1}") == 0

    asyncio.run(_run())


def test_list_does_not_prune_a_task_concurrently_recreated_during_its_own_scan(mocker):
    """list()'s inline pruning (get() returns None -> remove from index) must re-validate
    atomically before removing the index member: a concurrent save() recreating the same
    task_id between that get() and the removal must not have its fresh index entry destroyed.
    """
    server = FakeServer()
    store = RedisTaskStore(
        RedisStorageBackend(FakeAsyncRedis(server=server)), env_id=_ENV_ID, terminal_ttl_seconds=_TERMINAL_TTL_SECONDS
    )
    real_get = RedisTaskStore.get
    triggered = False

    async def _racy_get(self, task_id, context):
        nonlocal triggered
        result = await real_get(self, task_id, context)
        if task_id == "t1" and not triggered:
            triggered = True

            def _concurrent_save():
                async def _write():
                    writer_store = RedisTaskStore(
                        RedisStorageBackend(FakeAsyncRedis(server=server)),
                        env_id=_ENV_ID,
                        terminal_ttl_seconds=_TERMINAL_TTL_SECONDS,
                    )
                    await writer_store.save(_task("t1", ms=99_000), _CONTEXT)

                asyncio.run(_write())

            thread = threading.Thread(target=_concurrent_save)
            thread.start()
            thread.join()
        return result

    async def _run():
        await store.save(_task("t1", ms=1_000, state=TaskState.TASK_STATE_COMPLETED), _CONTEXT)
        await store._backend.delete("ikg_task:{env-1}:t1")  # simulate its primary key expiring

        mocker.patch.object(RedisTaskStore, "get", _racy_get)
        await store.list(ListTasksRequest(page_size=10), _CONTEXT)

        assert await store.get("t1", _CONTEXT) is not None
        assert await store._backend.zcard("ikg_task_idx:{env-1}") == 1

    asyncio.run(_run())


def test_prune_stale_tail_does_not_prune_a_concurrently_recreated_task(mocker):
    """_prune_stale_tail must carry the same protection as list()'s inline pruning - see
    test_list_does_not_prune_a_task_concurrently_recreated_during_its_own_scan.
    """
    server = FakeServer()
    store = RedisTaskStore(
        RedisStorageBackend(FakeAsyncRedis(server=server)), env_id=_ENV_ID, terminal_ttl_seconds=_TERMINAL_TTL_SECONDS
    )
    real_get = RedisTaskStore.get
    triggered = False

    async def _racy_get(self, task_id, context):
        nonlocal triggered
        result = await real_get(self, task_id, context)
        if task_id == "t1" and not triggered:
            triggered = True

            def _concurrent_save():
                async def _write():
                    writer_store = RedisTaskStore(
                        RedisStorageBackend(FakeAsyncRedis(server=server)),
                        env_id=_ENV_ID,
                        terminal_ttl_seconds=_TERMINAL_TTL_SECONDS,
                    )
                    await writer_store.save(_task("t1", ms=99_000), _CONTEXT)

                asyncio.run(_write())

            thread = threading.Thread(target=_concurrent_save)
            thread.start()
            thread.join()
        return result

    async def _run():
        await store.save(_task("t1", ms=1_000, state=TaskState.TASK_STATE_COMPLETED), _CONTEXT)
        await store._backend.delete("ikg_task:{env-1}:t1")

        mocker.patch.object(RedisTaskStore, "get", _racy_get)
        await store._prune_stale_tail("ikg_task_idx:{env-1}", _CONTEXT)

        assert await store.get("t1", _CONTEXT) is not None
        assert await store._backend.zcard("ikg_task_idx:{env-1}") == 1

    asyncio.run(_run())


def test_list_prunes_index_entries_whose_primary_record_has_already_expired():
    """A ZSET has no per-member TTL, so list() must prune a candidate whose primary record is
    already gone (simulated by deleting the primary key directly) from the index it's
    scanning, not just skip past it.
    """

    async def _run():
        store, backend = _store()
        await store.save(_task("live", ms=2_000), _CONTEXT)
        await store.save(_task("expired", ms=1_000), _CONTEXT)
        await backend.delete("ikg_task:{env-1}:expired")  # simulates the primary key's TTL firing
        assert await backend.zcard("ikg_task_idx:{env-1}") == _EXPECTED_TASK_COUNT_BEFORE_PRUNE

        response = await store.list(ListTasksRequest(), _CONTEXT)

        assert [task.id for task in response.tasks] == ["live"]
        assert await backend.zcard("ikg_task_idx:{env-1}") == 1

    asyncio.run(_run())


def test_list_does_not_skip_a_live_task_behind_a_pruned_entry_with_a_small_page_size():
    """Pruning a candidate shifts every later (lower-ranked) member's rank down by one -
    rank must shift with it, or the next zrevrange call starts one rank too far and silently
    skips a live task. page_size=1 forces the expired entry into its own window, so an
    unadjusted rank would resume past the live task entirely and return an empty page.
    """

    async def _run():
        store, backend = _store()
        await store.save(_task("expired", ms=2_000), _CONTEXT)  # highest score -> rank 0
        await store.save(_task("live", ms=1_000), _CONTEXT)  # rank 1, behind the pruned entry
        await backend.delete("ikg_task:{env-1}:expired")

        response = await store.list(ListTasksRequest(page_size=1), _CONTEXT)

        assert [task.id for task in response.tasks] == ["live"]

    asyncio.run(_run())


def test_list_survives_a_concurrent_delete_of_an_already_matched_task(mocker):
    """Replicas sharing one Redis: another replica's concurrent delete() of a task this
    list() call already matched must not silently drop a different, live task ("c" below).
    """
    server = FakeServer()
    store = RedisTaskStore(
        RedisStorageBackend(FakeAsyncRedis(server=server)), env_id=_ENV_ID, terminal_ttl_seconds=_TERMINAL_TTL_SECONDS
    )
    real_get = RedisTaskStore.get
    triggered = False

    async def _racy_get(self, task_id, context):
        nonlocal triggered
        result = await real_get(self, task_id, context)
        if task_id == "b" and not triggered:
            triggered = True

            def _concurrent_delete():
                async def _write():
                    writer_store = RedisTaskStore(
                        RedisStorageBackend(FakeAsyncRedis(server=server)),
                        env_id=_ENV_ID,
                        terminal_ttl_seconds=_TERMINAL_TTL_SECONDS,
                    )
                    await writer_store.delete("b", _CONTEXT)

                asyncio.run(_write())

            thread = threading.Thread(target=_concurrent_delete)
            thread.start()
            thread.join()
        return result

    async def _run():
        for task_id, ms in [("g1", 6_000), ("g2", 5_000), ("b", 4_000), ("c", 3_000), ("d", 2_000), ("e", 1_000)]:
            await store.save(_task(task_id, ms=ms), _CONTEXT)
        await store._backend.delete("ikg_task:{env-1}:g1")
        await store._backend.delete("ikg_task:{env-1}:g2")

        mocker.patch.object(RedisTaskStore, "get", _racy_get)
        response = await store.list(ListTasksRequest(page_size=3), _CONTEXT)

        all_ids = [task.id for task in response.tasks]
        token = response.next_page_token
        while token:
            next_response = await store.list(ListTasksRequest(page_size=3, page_token=token), _CONTEXT)
            all_ids += [task.id for task in next_response.tasks]
            token = next_response.next_page_token
            if not next_response.tasks:
                break

        assert "c" in all_ids

    asyncio.run(_run())


def test_list_prunes_a_stale_tail_even_when_fresh_tasks_satisfy_the_page():
    """A steady stream of fresh tasks always satisfies page_size from the high-score end, so
    the main scan never reaches the low-score (oldest) tail where expired entries actually
    pile up. list() must prune that whole tail directly on every call, independent of what the
    main scan already covered - not just a few of it, or the backlog still grows unbounded.
    """

    async def _run():
        store, backend = _store()
        for i in range(20):
            await store.save(_task(f"old{i}", ms=1_000 + i), _CONTEXT)
        for i in range(20):
            await backend.delete(f"ikg_task:{{env-1}}:old{i}")
        await store.save(_task("fresh", ms=99_000), _CONTEXT)  # highest score -> always rank 0

        assert await backend.zcard("ikg_task_idx:{env-1}") == _EXPECTED_INDEX_SIZE_BEFORE_TAIL_PRUNE

        response = await store.list(ListTasksRequest(page_size=1), _CONTEXT)

        assert [task.id for task in response.tasks] == ["fresh"]
        # The whole stale backlog must be gone in one call, not just reduced by a fixed amount.
        assert await backend.zcard("ikg_task_idx:{env-1}") == 1

    asyncio.run(_run())


def test_list_prunes_a_backlog_larger_than_any_fixed_sample_in_one_call():
    """A fixed-size sample per call only reduces the growth rate, it doesn't bound it: if
    expiry outpaces the sample size, the backlog still grows forever. Pruning must scale with
    the actual backlog - large or small - not a constant count, so even a 50-entry stale
    backlog is fully gone after a single list() call.
    """
    backlog_size = 50

    async def _run():
        store, backend = _store()
        for i in range(backlog_size):
            await store.save(_task(f"old{i}", ms=1_000 + i), _CONTEXT)
        for i in range(backlog_size):
            await backend.delete(f"ikg_task:{{env-1}}:old{i}")
        await store.save(_task("fresh", ms=99_000), _CONTEXT)

        await store.list(ListTasksRequest(page_size=1), _CONTEXT)

        assert await backend.zcard("ikg_task_idx:{env-1}") == 1

    asyncio.run(_run())


def test_list_prunes_the_env_wide_index_even_when_only_a_context_is_listed():
    """A context-specific scan finding a stale candidate must clean env-wide for it too, via
    the context-lookup reverse index - not just its own context index.
    """

    async def _run():
        store, backend = _store()
        await store.save(_task("old", context_id="ctx-0", state=TaskState.TASK_STATE_COMPLETED, ms=1_000), _CONTEXT)
        await backend.delete("ikg_task:{env-1}:old")
        await store.save(_task("fresh", context_id="ctx-0", ms=99_000), _CONTEXT)

        await store.list(ListTasksRequest(page_size=10, context_id="ctx-0"), _CONTEXT)

        assert await backend.zcard("ikg_task_idx:{env-1}:ctx:ctx-0") == 1
        assert await backend.zcard("ikg_task_idx:{env-1}") == 1

    asyncio.run(_run())


def test_list_prunes_a_context_index_even_when_only_env_wide_is_listed():
    """An env-wide scan finding a stale candidate must clean its context index too, via the
    reverse lookup - closing the gap a context-blind scan used to leave unrecoverable.
    """

    async def _run():
        store, backend = _store()
        await store.save(_task("old", context_id="ctx-1", state=TaskState.TASK_STATE_COMPLETED, ms=1_000), _CONTEXT)
        await backend.delete("ikg_task:{env-1}:old")
        await store.save(_task("fresh", ms=99_000), _CONTEXT)

        await store.list(ListTasksRequest(page_size=10), _CONTEXT)  # env-wide only

        assert await backend.zcard("ikg_task_idx:{env-1}:ctx:ctx-1") == 0
        assert await backend.zcard("ikg_task_idx:{env-1}") == 1

    asyncio.run(_run())


def test_list_does_not_prune_an_index_never_reached_by_any_scan():
    """Accepted residual limitation: pruning is reactive - a candidate whose context is never
    listed, and that env-wide never scans either, stays stale.
    """

    async def _run():
        store, backend = _store()
        await store.save(
            _task("old", context_id="ctx-unreached", state=TaskState.TASK_STATE_COMPLETED, ms=1_000), _CONTEXT
        )
        await backend.delete("ikg_task:{env-1}:old")
        # No list() call at all - nothing ever scans "old", via either index.

        assert await backend.zcard("ikg_task_idx:{env-1}:ctx:ctx-unreached") == 1

    asyncio.run(_run())


def test_list_filters_by_context_id():
    """list() with context_id set must only return tasks in that context."""

    async def _run():
        store, _backend = _store()
        await store.save(_task("t1", context_id="ctx-1", ms=1_000), _CONTEXT)
        await store.save(_task("t2", context_id="ctx-2", ms=2_000), _CONTEXT)
        await store.save(_task("t3", context_id="ctx-1", ms=3_000), _CONTEXT)

        response = await store.list(ListTasksRequest(context_id="ctx-1", page_size=10), _CONTEXT)

        assert {task.id for task in response.tasks} == {"t1", "t3"}
        assert response.total_size == _EXPECTED_CONTEXT_MATCH_COUNT

    asyncio.run(_run())


def test_list_filters_by_status():
    """list() with status set must only return tasks in that exact state."""

    async def _run():
        store, _backend = _store()
        await store.save(_task("working", ms=1_000, state=TaskState.TASK_STATE_WORKING), _CONTEXT)
        await store.save(_task("done", ms=2_000, state=TaskState.TASK_STATE_COMPLETED), _CONTEXT)

        response = await store.list(ListTasksRequest(status=TaskState.TASK_STATE_COMPLETED, page_size=10), _CONTEXT)

        assert [task.id for task in response.tasks] == ["done"]

    asyncio.run(_run())


def test_list_filters_by_status_timestamp_after():
    """list() with status_timestamp_after set must only return tasks at or after that bound."""

    async def _run():
        store, _backend = _store()
        await store.save(_task("old", ms=1_000), _CONTEXT)
        await store.save(_task("new", ms=5_000), _CONTEXT)

        response = await store.list(
            ListTasksRequest(status_timestamp_after=_timestamp(2_000), page_size=10),
            _CONTEXT,
        )

        assert [task.id for task in response.tasks] == ["new"]

    asyncio.run(_run())


def test_list_paginates_across_multiple_pages():
    """A page_size smaller than the total task count must page correctly: every task seen
    exactly once, most-recent-first.
    """

    async def _run():
        store, _backend = _store()
        for i in range(5):
            await store.save(_task(f"t{i}", ms=(i + 1) * 1_000), _CONTEXT)

        seen = []
        page_token = ""
        for _ in range(10):  # bounded loop, not an infinite one - a bug would just exhaust it
            request = ListTasksRequest(page_size=2, page_token=page_token)
            response = await store.list(request, _CONTEXT)
            seen.extend(task.id for task in response.tasks)
            if not response.next_page_token:
                break
            page_token = response.next_page_token

        assert seen == ["t4", "t3", "t2", "t1", "t0"]

    asyncio.run(_run())


def test_list_rejects_an_invalid_page_token():
    """A page_token naming a task_id not in the index must raise InvalidParamsError."""

    async def _run():
        store, _backend = _store()
        await store.save(_task("t1", ms=1_000), _CONTEXT)

        with pytest.raises(InvalidParamsError):
            await store.list(ListTasksRequest(page_token=encode_page_token("never-saved")), _CONTEXT)

    asyncio.run(_run())


@pytest.mark.parametrize("page_size", [-1, -100, 101, 1000])
def test_list_rejects_an_out_of_range_page_size(page_size):
    """A negative page_size is truthy, so `params.page_size or DEFAULT` lets it through instead
    of falling back - must be rejected, like one above MAX_LIST_TASKS_PAGE_SIZE.
    """

    async def _run():
        store, _backend = _store()
        await store.save(_task("t1", ms=1_000), _CONTEXT)

        with pytest.raises(InvalidParamsError):
            await store.list(ListTasksRequest(page_size=page_size), _CONTEXT)

    asyncio.run(_run())


def test_list_total_size_reflects_the_index_before_status_filtering():
    """total_size is the raw index size, not the post-status-filter count."""

    async def _run():
        store, _backend = _store()
        await store.save(_task("a", ms=1_000, state=TaskState.TASK_STATE_WORKING), _CONTEXT)
        await store.save(_task("b", ms=2_000, state=TaskState.TASK_STATE_COMPLETED), _CONTEXT)
        await store.save(_task("c", ms=3_000, state=TaskState.TASK_STATE_WORKING), _CONTEXT)

        response = await store.list(ListTasksRequest(status=TaskState.TASK_STATE_COMPLETED, page_size=10), _CONTEXT)

        assert len(response.tasks) == 1
        assert response.total_size == _EXPECTED_INDEX_SIZE

    asyncio.run(_run())
