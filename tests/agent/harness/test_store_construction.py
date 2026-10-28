from __future__ import annotations

import asyncio
import sqlite3
import threading

import pytest

from agent_runtime.harness import RolloutStore


@pytest.fixture
def connection_closes(monkeypatch):
    closes = []
    connect = sqlite3.connect

    class Connection(sqlite3.Connection):
        def close(self):
            super().close()
            closes.append(threading.get_ident())

    monkeypatch.setattr(sqlite3, "connect", lambda *a, **kw: connect(*a, **kw, factory=Connection))
    return closes


def test_bare_store_rejects_operations_and_closes_safely(tmp_path):
    store = RolloutStore()
    try:
        with pytest.raises(RuntimeError, match="not opened"):
            store.rows('threads')
        with pytest.raises(RuntimeError, match="not opened"):
            with store:
                pytest.fail("unopened store entered")
    finally:
        store.close()
        store.close()


def test_sync_factory_opens_on_owner_and_preserves_read_only(tmp_path):
    database = tmp_path / "run.db"
    with RolloutStore.open_sync(database) as store:
        assert store._owner_thread != threading.get_ident()
        def write():
            with store.transaction():
                store.put_metadata('construction-test', 'preserved')
        store.run_sync(write)
    with RolloutStore.open_sync(database, read_only=True) as reader:
        assert reader.metadata('construction-test') == 'preserved'
        with pytest.raises(sqlite3.OperationalError, match="readonly"):
            reader.run_sync(lambda: _readonly_write(reader))


def _readonly_write(store):
    with store.transaction():
        store.put_metadata('construction-test', 'must not commit')


@pytest.mark.anyio
async def test_sync_factory_rejects_event_loop_before_constructing(tmp_path, monkeypatch):
    def forbidden_init(self):
        pytest.fail("sync factory allocated a store on the event loop")

    monkeypatch.setattr(RolloutStore, "__init__", forbidden_init)
    with pytest.raises(RuntimeError, match=r"await RolloutStore.open"):
        RolloutStore.open_sync(tmp_path / "run.db")
    assert not (tmp_path / "run.db").exists()


@pytest.mark.anyio
async def test_sync_factory_is_allowed_in_worker_thread(tmp_path):
    store = await asyncio.to_thread(RolloutStore.open_sync, tmp_path / "run.db")
    try:
        assert (await store.run_async(store.verify)).valid
    finally:
        await store.aclose()


@pytest.mark.anyio
async def test_bare_store_rejects_async_context_and_closes_safely():
    store = RolloutStore()
    with pytest.raises(RuntimeError, match="not opened"):
        async with store:
            pytest.fail("unopened store entered")
    await store.aclose()
    await store.aclose()


@pytest.mark.anyio
@pytest.mark.parametrize("synchronous", [False, True])
async def test_factories_run_subclass_constructor(tmp_path, synchronous):
    class Store(RolloutStore):
        def __new__(cls):
            return super().__new__(cls)

        def __init__(self):
            super().__init__()
            self.constructed = True

    if synchronous:
        store = await asyncio.to_thread(Store.open_sync, tmp_path / "run.db")
    else:
        store = await Store.open(tmp_path / "run.db")
    async with store:
        assert isinstance(store, Store)
        assert store.constructed
        assert (await store.run_async(store.verify)).valid


@pytest.mark.anyio
@pytest.mark.parametrize("synchronous", [False, True])
async def test_failed_factory_closes_partially_opened_connection(tmp_path, synchronous, connection_closes):
    opened = []

    class Store(RolloutStore):
        def _initialize(self, path, *, read_only):
            super()._initialize(path, read_only=read_only)
            opened.append(self)
            raise OSError("initialization failed")

    with pytest.raises(OSError, match="initialization failed"):
        if synchronous:
            await asyncio.to_thread(Store.open_sync, tmp_path / "run.db")
        else:
            await Store.open(tmp_path / "run.db")
    [store] = opened
    assert store._closing.done()
    assert connection_closes == [store._owner_thread]


@pytest.mark.anyio
async def test_cancelled_open_drains_initialization_then_closes(tmp_path, connection_closes):
    entered, release = threading.Event(), threading.Event()
    opened = []

    class Store(RolloutStore):
        def _initialize(self, path, *, read_only):
            super()._initialize(path, read_only=read_only)
            opened.append(self)
            entered.set()
            assert release.wait(3)

    task = asyncio.create_task(Store.open(tmp_path / "run.db"))
    try:
        async with asyncio.timeout(3):
            while not entered.is_set():
                await asyncio.sleep(0.001)
            task.cancel()
            await asyncio.sleep(0)
            assert not task.done()
            release.set()
            with pytest.raises(asyncio.CancelledError):
                await task
        [store] = opened
        assert store._closing.done()
        assert connection_closes == [store._owner_thread]
    finally:
        release.set()
        await asyncio.gather(task, return_exceptions=True)
