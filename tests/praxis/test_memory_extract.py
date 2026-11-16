"""阶段4 T11:memory 状态代数 + 抽取流水线 + 作业队列(B4;R-PF-5;Q6)。

七测试场景(brief 钉死)逐项对应:
① 端到端 scripted 两阶段 → state 落库 + 包络 + 版本快照 + prune;
② 双实例并发 claim 恰一赢;
③ kill -9 于 maintenance dispatch 后 → 重开 = needs_reconciliation(B4 回归钉:
   旧 memory.py:178 `_unresolved` 查 {"running","unknown"} 漏 dispatched;正确集
   memory_identity.py:183-186 含 dispatched——praxis 以"未终态 turn + 未闭合
   事实窗口(compensation.unclosed_windows 同一实现)"结构消亡状态字符串集合);
④ batch 中途源 thread 新增 turn → 复检弃写(无脏合并);
⑤ budget_exhausted 路径(grant 打满)+ exhausted_batches 记录;
⑥ 幂等重放(同 batch_hash+version 不重复学);
⑦ R-PF-5 两例(父无限 24k parity / 父有界截断 + 不足 retry_after 重试)。

仓库无 pytest-asyncio:与 test_agents_core.py 同风格,asyncio.run 包装。
"""

from __future__ import annotations

import asyncio
import json
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any
from uuid import uuid4

import pytest

from praxis.adapters.models.scripted import ScriptedAdapter
from praxis.composition import build_session
from praxis.core import memory as memory_module
from praxis.core.agents import SubagentManager
from praxis.core.events import EventBus
from praxis.core.memory import (
    EXTRACT_INSTRUCTION,
    JOB_KEY,
    LEASE_KEY,
    MAX_ENTRY_CHARS,
    MAX_MEMORY_ENTRIES,
    MAX_SOURCE_BYTES,
    MEMORY_LEASE_SECONDS,
    MEMORY_RETRY_SECONDS,
    MEMORY_TOKEN_BUDGET,
    MEMORY_UPDATE_DEADLINE_SECONDS,
    STATE_KEY,
    claim_lease,
    initial_state,
    job_unresolved,
    record_revision,
    release_lease,
    run_update,
    validate_output,
)
from praxis.core.statemachine import TurnStatus, transition
from praxis.protocol.budget import Limits, limits_payload
from praxis.protocol.items import ItemKind
from praxis.protocol.json import JsonValue
from praxis.protocol.models import ModelResponse, Usage
from praxis.store.log import ItemEntry, Store
from praxis.store.memory import (
    MEMORY_ENTRY_HISTORY_KEEP,
    get_metadata,
    put_state_enveloped,
    read_state_verified,
)
from praxis.store.turns import create_turn, get_turn
from tests.praxis.conftest import make_turn

_ZERO_USAGE = Usage(input_tokens=0, output_tokens=0)
_KILL_SCRIPT = Path(__file__).resolve().parent / "fixtures" / "memory_kill_child.py"


# ---------------------------------------------------------------------------
# 脚手架
# ---------------------------------------------------------------------------


def _resp(content: str, *, usage: Usage = _ZERO_USAGE) -> ModelResponse:
    return ModelResponse(content=content, tool_calls=(), usage=usage, finish_reason="stop")


def _entry(key: str, ref: str, *, text: str = "a reusable fact", kind: str = "fact") -> dict[str, Any]:
    return {"key": key, "title": key.title(), "text": text, "source_item_ids": [ref], "kind": kind}


def _answer(entries: list[dict[str, Any]]) -> str:
    return json.dumps({"entries": entries}, ensure_ascii=False)


def _db_path(ws: Path) -> Path:
    return ws / ".praxis" / "praxis.sqlite"


def _preseed(ws: Path, thread_id: str, text: str) -> str:
    """build_session 之前先落一个已完成源 turn;返回 user_message item_id。"""
    path = _db_path(ws)
    path.parent.mkdir(parents=True, exist_ok=True)
    store = Store(path)
    try:
        _, item_id = _seed_turn(store, thread_id, text)
        return item_id
    finally:
        store.close()


def _seed_turn(store: Store, thread_id: str, text: str) -> tuple[str, str]:
    """一个已完成的源 turn + 一条 user_message;返回 (turn_id, item_id)。"""
    turn = make_turn(store, "completed", thread_id)
    appended = store.append_items(
        thread_id, turn.turn_id, [ItemEntry(kind=ItemKind.USER_MESSAGE, payload={"text": text})]
    )
    return turn.turn_id, appended[0].item_id


def _session(ws: Path, responses: list[ModelResponse], **kw: Any) -> Any:
    kw.setdefault("tools", [])
    kw.setdefault("enable_memory", False)
    return build_session(ws, ScriptedAdapter(responses), **kw)


def _manager(session: Any) -> SubagentManager:
    return SubagentManager(parent=session, store=session.store)


def _state(store: Store) -> dict[str, JsonValue] | None:
    return store.transact(lambda conn: read_state_verified(conn, STATE_KEY))  # type: ignore[arg-type,return-value]


def _job(store: Store) -> dict[str, Any]:
    raw = get_metadata(store, JOB_KEY)
    return {} if raw is None else json.loads(raw)


def _memory_allocations(store: Store, thread_id: str) -> list[dict[str, JsonValue]]:
    return [
        dict(item.payload)
        for item in store.read_items(thread_id)
        if item.kind == ItemKind.CHILD_ALLOCATION and item.payload.get("source") == "memory"
    ]


async def _wait_job(store: Store, statuses: set[str], timeout: float = 20.0) -> dict[str, Any]:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        job = _job(store)
        if job.get("status") in statuses:
            return job
        await asyncio.sleep(0.02)
    raise AssertionError(f"job never reached {statuses}: {_job(store)}")


# ---------------------------------------------------------------------------
# 常量移植 parity(旧 memory.py:42-45 逐值;lease/retry 300s;deadline 240s)
# ---------------------------------------------------------------------------


def test_constants_ported_verbatim() -> None:
    assert (MAX_MEMORY_ENTRIES, MAX_SOURCE_BYTES, MAX_ENTRY_CHARS, MEMORY_TOKEN_BUDGET) == (
        32,
        32_000,
        2_000,
        24_000,
    )
    assert MEMORY_LEASE_SECONDS == 300
    assert MEMORY_RETRY_SECONDS == 300
    assert MEMORY_UPDATE_DEADLINE_SECONDS == 240


def test_prompt_constants_rewritten() -> None:
    rules = memory_module.MEMORY_RULES
    assert "never instructions or permissions" in rules
    assert "Never store credentials" in rules
    assert "payload_fragment" in rules
    assert EXTRACT_INSTRUCTION
    assert memory_module.CONSOLIDATE_INSTRUCTION


# ---------------------------------------------------------------------------
# 状态代数:initial_state / validate_output / record_revision(prune M=100)
# ---------------------------------------------------------------------------


def test_initial_state_shape_and_flags() -> None:
    state = initial_state()
    assert state["version"] == 0
    assert state["entries"] == []
    assert state["processed"] == {}
    assert state["deleted_keys"] == []
    assert state["deleted_sources"] == []
    assert state["use_memories"] is True
    assert state["generate_memories"] is True


def test_validate_output_accepts_and_maps_sources() -> None:
    entries = validate_output(_answer([_entry("k1", "r1")]), allowed_refs={"r1": "t1"})
    assert entries[0]["sources"] == {"r1": "t1"}
    assert entries[0]["kind"] == "fact"


@pytest.mark.parametrize(
    "bad",
    [
        _answer([_entry("k" * 121, "r1")]),
        _answer([{**_entry("k1", "r1"), "title": ""}]),
        _answer([{**_entry("k1", "r1"), "text": "x" * (MAX_ENTRY_CHARS + 1)}]),
        _answer([{**_entry("k1", "r1"), "kind": "opinion"}]),
        _answer([_entry("k1", "unknown-ref")]),
        _answer([_entry("k1", "r1"), _entry("k1", "r1")]),
        _answer([_entry(f"k{i}", "r1") for i in range(MAX_MEMORY_ENTRIES + 1)]),
        _answer([{**_entry("k1", "r1"), "source_item_ids": []}]),
        _answer([{**_entry("k1", "r1"), "source_item_ids": [f"r{i}" for i in range(21)]}]),
        json.dumps({"entries": [], "extra": 1}),
        json.dumps(["not", "an", "object"]),
        "not json at all",
    ],
)
def test_validate_output_rejects(bad: str) -> None:
    refs = {f"r{i}": "t1" for i in range(21)} | {"r1": "t1"}
    with pytest.raises(ValueError):
        validate_output(bad, allowed_refs=refs)


def test_record_revision_stamps_and_prunes_entry_history_to_100() -> None:
    state = initial_state()
    state["entry_history"] = {"k1": [{"version": i, "action": "learn"} for i in range(105)]}
    stamped = record_revision(
        state,
        _entry("k1", "r1"),
        action="learn",
        origin="learned",
        previous={"entry_id": "memory_old"},
        kind_map={"r1": "user_message"},
    )
    assert stamped["entry_id"].startswith("memory_")
    assert stamped["origin"] == "learned"
    assert stamped["supersedes"] == "memory_old"
    assert stamped["evidence"] == ["user_message"]
    assert stamped["updated_at"]
    history = state["entry_history"]["k1"]
    assert len(history) == MEMORY_ENTRY_HISTORY_KEEP == 100
    assert history[-1]["action"] == "learn"
    assert history[-1]["entry_id"] == stamped["entry_id"]


# ---------------------------------------------------------------------------
# B4 判据:job_unresolved(未终态 turn + unclosed_windows 同一实现)
# ---------------------------------------------------------------------------


def test_job_unresolved_running_turn_is_in_flight(store: Store) -> None:
    turn = make_turn(store, "running", "t")
    assert job_unresolved(store, [turn.turn_id], {}) is True


def test_job_unresolved_missing_turn_conservative(store: Store) -> None:
    assert job_unresolved(store, ["no-such-turn"], {}) is True


def test_job_unresolved_settled_turn_is_closed(store: Store) -> None:
    turn = make_turn(store, "completed", "t")
    assert job_unresolved(store, [turn.turn_id], {}) is False


def test_job_unresolved_dispatched_tool_window_even_if_turn_terminal(store: Store) -> None:
    """dispatched-未 settle 的工具窗口:turn 已 FAILED 但窗口未闭合 → 仍在途。

    这是旧 _unresolved 漏 dispatched 病灶的工具面回归:判据只有事实窗口,
    不存在 {"running","unknown"} 之类的状态字符串集合。
    """
    turn = make_turn(store, "running", "t")
    store.append_items(
        "t",
        turn.turn_id,
        [
            ItemEntry(
                kind=ItemKind.MODEL_RESPONSE,
                payload={
                    "content": "",
                    "tool_calls": [{"call_id": "c1", "name": "toy", "arguments": {}}],
                },
            ),
            ItemEntry(kind=ItemKind.TOOL_CALL, payload={"call_id": "c1", "tool_name": "toy"}),
        ],
    )
    transition(store, EventBus(), turn.turn_id, TurnStatus.FAILED, reason="boom")
    assert job_unresolved(store, [turn.turn_id], {}) is True


# ---------------------------------------------------------------------------
# ② 双实例并发 claim 恰一赢 + lease 持有期 run_update busy
# ---------------------------------------------------------------------------


def test_concurrent_claim_exactly_one_winner(tmp_path: Path) -> None:
    db = tmp_path / "shared.db"
    stores = [Store(db), Store(db)]
    barrier = threading.Barrier(2)
    results: list[Any] = [None, None]

    def claim(index: int) -> None:
        barrier.wait()
        results[index] = claim_lease(stores[index], owner=f"owner-{index}", now=1000.0)

    threads = [threading.Thread(target=claim, args=(i,)) for i in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    try:
        winners = [r for r in results if r is not None]
        assert len(winners) == 1
        assert winners[0].expires_at == 1000.0 + MEMORY_LEASE_SECONDS
        # 租约未过期:再 claim 必败;过期后可接管。
        assert claim_lease(stores[0], owner="late", now=1000.0 + MEMORY_LEASE_SECONDS - 1) is None
        takeover = claim_lease(stores[0], owner="late", now=1000.0 + MEMORY_LEASE_SECONDS + 1)
        assert takeover is not None
        release_lease(stores[0], takeover)
        assert get_metadata(stores[0], LEASE_KEY) is None
    finally:
        for s in stores:
            s.close()


def test_run_update_busy_while_lease_held(workspace: Path) -> None:
    async def scenario() -> None:
        session = _session(workspace, [_resp("ok")], thread_id="src")
        try:
            _seed_turn(session.store, "src", "history")
            lease = claim_lease(session.store, owner="other-process", now=time.time())
            assert lease is not None
            outcome = await run_update(session.store, _manager(session))
            assert outcome.status == "busy"
            release_lease(session.store, lease)
        finally:
            await session.aclose()

    asyncio.run(scenario())


# ---------------------------------------------------------------------------
# ① 端到端 scripted 两阶段(Q6 后台 driver 路径)+ 包络/快照/prune
# ---------------------------------------------------------------------------


def test_end_to_end_background_two_stage(workspace: Path) -> None:
    async def scenario() -> None:
        ref = _preseed(workspace, "src", "I prefer neovim for editing.")
        session = _session(
            workspace,
            [
                _resp("parent done"),
                _resp(_answer([_entry("pref.editor", ref, kind="preference")])),
                _resp(_answer([_entry("pref.editor", ref, text="User prefers neovim.", kind="preference")])),
            ],
            thread_id="src",
            enable_memory=True,
        )
        try:
            result = await session.submit("hello")
            assert result.status is TurnStatus.COMPLETED
            job = await _wait_job(session.store, {"done"})
            assert job["source_thread"] == "src"
            assert len(job["turn_ids"]) == 2
            assert job["batch_hash"]

            state = _state(session.store)
            assert state is not None and state["version"] == 1
            assert state["use_memories"] is True and state["generate_memories"] is True
            entry = state["entries"][0]
            assert entry["key"] == "pref.editor"
            assert entry["origin"] == "learned"
            assert entry["supersedes"] is None
            assert entry["evidence"] == ["user_message"]
            assert entry["source_item_ids"] == [ref]
            assert state["processed"]["src"]
            assert state["entry_history"]["pref.editor"][-1]["action"] == "learn"

            # 包络 + 不可变版本快照 + 作业历史(live 键非数字后缀)。
            snapshot = get_metadata(session.store, f"{STATE_KEY}:version:1")
            assert snapshot is not None
            assert json.loads(snapshot)["version"] == 1
            assert get_metadata(session.store, f"{JOB_KEY}:1") is not None

            allocations = _memory_allocations(session.store, "src")
            assert len(allocations) == 2
            purposes = []
            for alloc in allocations:
                assert alloc["grant"] == {"tokens": MEMORY_TOKEN_BUDGET, "steps": 1}
                child_turn = get_turn(session.store, str(alloc["child_root_turn_id"]))
                assert child_turn is not None
                assert child_turn.binding.get("memory_maintenance") is True
                purposes.append(child_turn.binding.get("maintenance_purpose"))
            assert purposes == ["memory_extract", "memory_consolidate"]
        finally:
            await session.aclose()

    asyncio.run(scenario())


def test_version_snapshots_pruned_in_commit_tx(workspace: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """prune_versions 在 commit 同事务被消费(keep spy=2 → 第 3 版后 v1 消失)。"""
    real_prune = memory_module.prune_versions
    calls: list[str] = []

    def spy(conn: Any, key: str, keep: int = 20) -> int:
        calls.append(key)
        return real_prune(conn, key, keep=2)

    monkeypatch.setattr(memory_module, "prune_versions", spy)

    async def scenario() -> None:
        # 每 cycle 一个新源 turn(新 revision)+ 独立剧本会话(锚 = 既有终态 turn)。
        sessions = []
        try:
            for i in range(3):
                ref = _preseed(workspace, "src", f"fact number {i}")
                session = _session(
                    workspace,
                    [_resp(_answer([_entry(f"k{i}", ref)])), _resp(_answer([_entry(f"k{i}", ref)]))],
                    thread_id="src",
                )
                sessions.append(session)
                outcome = await run_update(session.store, _manager(session))
                assert outcome.status == "updated", outcome
                assert outcome.version == i + 1
            store = sessions[-1].store
            assert calls and all(key == STATE_KEY for key in calls)
            assert get_metadata(store, f"{STATE_KEY}:version:1") is None
            assert get_metadata(store, f"{STATE_KEY}:version:2") is not None
            assert get_metadata(store, f"{STATE_KEY}:version:3") is not None
            state = _state(store)
            assert state is not None and state["version"] == 3
        finally:
            for session in sessions:
                await session.aclose()

    asyncio.run(scenario())


# ---------------------------------------------------------------------------
# ⑥ 幂等重放 + 选源跳过(子线程/active/revision 已处理)
# ---------------------------------------------------------------------------


def test_idempotent_replay_same_revision_no_relearn(workspace: Path) -> None:
    async def scenario() -> None:
        ref = _preseed(workspace, "src", "stable fact")
        session = _session(
            workspace,
            [_resp("parent done"), _resp(_answer([_entry("k1", ref)])), _resp(_answer([_entry("k1", ref)]))],
            thread_id="src",
        )
        try:
            await session.submit("hello")
            manager = _manager(session)
            first = await run_update(session.store, manager)
            assert first.status == "updated" and first.version == 1
            allocations_before = len(_memory_allocations(session.store, "src"))
            second = await run_update(session.store, manager)
            assert second.status == "unchanged"
            assert second.maintenance_turn_ids == ()
            state = _state(session.store)
            assert state is not None and state["version"] == 1
            assert len(_memory_allocations(session.store, "src")) == allocations_before
        finally:
            await session.aclose()

    asyncio.run(scenario())


def test_source_selection_skips_child_and_active_threads(workspace: Path) -> None:
    async def scenario() -> None:
        session = _session(workspace, [_resp("ok")], thread_id="src")
        try:
            store = session.store
            # 子线程(binding.parent_turn_id 非 None)不得成为抽取源。
            store.ensure_thread("child-thread")
            child_turn = create_turn(
                store, turn_id=uuid4().hex, thread_id="child-thread", binding={"parent_turn_id": "p"}
            )
            store.append_items(
                "child-thread", child_turn.turn_id, [ItemEntry(kind=ItemKind.USER_MESSAGE, payload={"text": "child"})]
            )
            transition(store, EventBus(), child_turn.turn_id, TurnStatus.COMPLETED, reason="done")
            outcome = await run_update(store, _manager(session))
            assert outcome.status == "unchanged"
            # active turn 线程同样跳过。
            make_turn(store, "running", "active-thread")
            outcome = await run_update(store, _manager(session))
            assert outcome.status == "unchanged"
        finally:
            await session.aclose()

    asyncio.run(scenario())


def test_generate_memories_flag_gates_extraction(workspace: Path) -> None:
    async def scenario() -> None:
        session = _session(workspace, [], thread_id="src")
        try:
            _seed_turn(session.store, "src", "history")
            session.store.transact(
                lambda conn: put_state_enveloped(
                    conn, STATE_KEY, {**initial_state(), "generate_memories": False}
                )
            )
            outcome = await run_update(session.store, _manager(session))
            assert outcome.status == "unchanged"
            assert _memory_allocations(session.store, "src") == []
        finally:
            await session.aclose()

    asyncio.run(scenario())


def test_enable_memory_default_off_zero_disturbance(workspace: Path) -> None:
    async def scenario() -> None:
        session = _session(workspace, [_resp("ok")], thread_id="src")
        try:
            await session.submit("hello")
            assert get_metadata(session.store, JOB_KEY) is None
            assert _memory_allocations(session.store, "src") == []
        finally:
            await session.aclose()

    asyncio.run(scenario())


# ---------------------------------------------------------------------------
# ④ batch 中途源漂移 → 复检弃写(旧 :596-599 语义)
# ---------------------------------------------------------------------------


def test_source_drift_mid_batch_aborts_commit(workspace: Path) -> None:
    async def scenario() -> None:
        ref = _preseed(workspace, "src", "original history")
        # 慢剧本给 stage1 一个在途窗口(parent submit 同样慢 1s,无碍)。
        session = build_session(
            workspace,
            ScriptedAdapter(
                [
                    _resp("parent done"),
                    _resp(_answer([_entry("k1", ref)])),
                    _resp(_answer([_entry("k1", ref)])),
                ],
                delay_seconds=1.0,
            ),
            thread_id="src",
            tools=[],
        )
        try:
            await session.submit("hello")
            manager = _manager(session)
            task = asyncio.create_task(run_update(session.store, manager))
            # 等 stage1 在途(job running + turn_ids 已登记)再注入源漂移。
            await _wait_job(session.store, {"running"})
            deadline = time.monotonic() + 10
            while not _job(session.store).get("turn_ids"):
                if time.monotonic() > deadline:
                    raise AssertionError("stage1 never dispatched")
                await asyncio.sleep(0.02)
            drift = Store(_db_path(workspace))
            try:
                _seed_turn(drift, "src", "drift: new turn arrived mid-extraction")
            finally:
                drift.close()
            outcome = await task
            assert outcome.status == "failed"
            assert outcome.error_message and "source changed" in outcome.error_message
            # 弃写:state 未落库,job failed + retry_after(不脏合并)。
            assert _state(session.store) is None
            job = _job(session.store)
            assert job["status"] == "failed"
            assert job["retry_after"] >= time.time() + MEMORY_RETRY_SECONDS - 5
        finally:
            await session.aclose()

    asyncio.run(scenario())


# ---------------------------------------------------------------------------
# ⑤ budget_exhausted(grant 打满)+ exhausted_batches(新 revision 不回购)
# ---------------------------------------------------------------------------


def test_budget_exhausted_and_exhausted_batches(workspace: Path) -> None:
    async def scenario() -> None:
        ref = _preseed(workspace, "src", "expensive history")
        session = _session(
            workspace,
            [
                _resp("parent done"),
                _resp(
                    _answer([_entry("k1", ref)]),
                    usage=Usage(input_tokens=MEMORY_TOKEN_BUDGET, output_tokens=0),
                ),
            ],
            thread_id="src",
        )
        try:
            await session.submit("hello")
            manager = _manager(session)
            first = await run_update(session.store, manager)
            assert first.status == "budget_exhausted"
            assert len(first.maintenance_turn_ids) == 1
            job = _job(session.store)
            assert job["status"] == "budget_exhausted"
            first_hash = job["batch_hash"]
            assert _state(session.store) is None

            # 同 revision 的 exhausted 源被跳过(不回购)。
            again = await run_update(session.store, manager)
            assert again.status == "budget_exhausted"
            assert again.exhausted_sources == ("src",)
        finally:
            await session.aclose()

        # 新历史到达:exhausted_batches 记录旧批,游标越过旧批只学新增。
        ref2 = _preseed(workspace, "src", "cheap new history")
        session2 = _session(
            workspace,
            [_resp(_answer([_entry("k2", ref2)])), _resp(_answer([_entry("k2", ref2)]))],
            thread_id="src",
        )
        try:
            third = await run_update(session2.store, _manager(session2))
            assert third.status == "updated" and third.version == 1
            job = _job(session2.store)
            exhausted = job["sources"]["src"]["exhausted_batches"]
            assert len(exhausted) == 1
            assert exhausted[0]["batch_hash"] == first_hash
            assert job["batch_hash"] != first_hash
            state = _state(session2.store)
            assert state is not None
            assert [e["key"] for e in state["entries"]] == ["k2"]
            assert state["processed"]["src"]
        finally:
            await session2.aclose()

    asyncio.run(scenario())


# ---------------------------------------------------------------------------
# 分批(MAX_SOURCE_BYTES)+ 游标续跑(has_more)
# ---------------------------------------------------------------------------


def test_large_item_fragmented_into_batches_with_cursor(workspace: Path) -> None:
    async def scenario() -> None:
        big = "x" * (MAX_SOURCE_BYTES + 8_000)
        ref = _preseed(workspace, "src", big)
        session = _session(
            workspace,
            [
                _resp("parent done"),
                _resp(_answer([_entry("k1", ref)])),
                _resp(_answer([_entry("k1", ref)])),
                _resp(_answer([_entry("k1", ref)])),
                _resp(_answer([_entry("k1", ref)])),
            ],
            thread_id="src",
        )
        try:
            await session.submit("hello")
            manager = _manager(session)
            first = await run_update(session.store, manager)
            assert first.status == "updated" and first.has_more is True
            state = _state(session.store)
            assert state is not None and "src" not in state["processed"]
            job = _job(session.store)
            assert job["status"] == "pending"
            assert job["cursor"]["item_id"] == ref
            first_hash = job["batch_hash"]
            second = await run_update(session.store, manager)
            assert second.status == "updated" and second.has_more is False
            assert second.version == 2
            job = _job(session.store)
            assert job["batch_hash"] != first_hash
            state = _state(session.store)
            assert state is not None and state["processed"]["src"]
        finally:
            await session.aclose()

    asyncio.run(scenario())


# ---------------------------------------------------------------------------
# ⑦ R-PF-5:maintenance 份额无例外(父无限 parity / 父有界截断与 retry)
# ---------------------------------------------------------------------------


def test_rpf5_unlimited_parent_grant_parity(workspace: Path) -> None:
    async def scenario() -> None:
        ref = _preseed(workspace, "src", "history")
        session = _session(
            workspace,
            [_resp("parent done"), _resp(_answer([_entry("k1", ref)])), _resp(_answer([_entry("k1", ref)]))],
            thread_id="src",
        )
        try:
            await session.submit("hello")
            outcome = await run_update(session.store, _manager(session))
            assert outcome.status == "updated"
            for alloc in _memory_allocations(session.store, "src"):
                assert alloc["grant"] == limits_payload(Limits(tokens=MEMORY_TOKEN_BUDGET, steps=1))
                assert alloc["requested"] == limits_payload(Limits(tokens=MEMORY_TOKEN_BUDGET, steps=1))
        finally:
            await session.aclose()

    asyncio.run(scenario())


def test_rpf5_bounded_parent_truncates_share(workspace: Path) -> None:
    async def scenario() -> None:
        ref = _preseed(workspace, "src", "history")
        session = _session(
            workspace,
            [
                _resp("parent done", usage=Usage(input_tokens=2_000, output_tokens=0)),
                _resp(_answer([_entry("k1", ref)]), usage=Usage(input_tokens=100, output_tokens=0)),
                _resp(_answer([_entry("k1", ref)]), usage=Usage(input_tokens=100, output_tokens=0)),
            ],
            thread_id="src",
            budget=Limits(tokens=10_000),
        )
        try:
            await session.submit("hello")
            outcome = await run_update(session.store, _manager(session))
            assert outcome.status == "updated"
            grants = [a["grant"] for a in _memory_allocations(session.store, "src")]
            # 父剩余 8_000 → floor(8_000×0.5)=4_000 截断(无例外规则);
            # stage2 时父剩余再扣 stage1 已用 → 份额继续按算法收缩。
            assert grants[0] == {"tokens": 4_000, "steps": 1}
            assert grants[1]["tokens"] == 3_950
        finally:
            await session.aclose()

    asyncio.run(scenario())


def test_rpf5_exhausted_parent_retries_after_new_turn(workspace: Path) -> None:
    async def scenario() -> None:
        now = time.time()
        ref = _preseed(workspace, "src", "history")
        session = _session(
            workspace,
            [
                _resp("parent done", usage=Usage(input_tokens=20, output_tokens=0)),
                _resp(_answer([_entry("k1", ref)])),
                _resp(_answer([_entry("k1", ref)])),
            ],
            thread_id="src",
            budget=Limits(tokens=1),
        )
        try:
            await session.submit("hello")
            manager = _manager(session)
            first = await run_update(session.store, manager, now=now)
            assert first.status == "failed"
            job = _job(session.store)
            assert job["retry_after"] >= now + MEMORY_RETRY_SECONDS - 5
            # retry_after 未到:选源跳过(waiting),不 spawn。
            waiting = await run_update(session.store, manager, now=now + 1)
            assert waiting.status == "waiting"
            assert waiting.waiting_sources == ("src",)
            assert _memory_allocations(session.store, "src") == []
        finally:
            await session.aclose()

        # 父预算随新 turn 增长:新会话同 thread 大预算,终态后重试终可执行。
        session2 = _session(
            workspace,
            [
                _resp("second turn"),
                _resp(_answer([_entry("k1", ref)])),
                _resp(_answer([_entry("k1", ref)])),
            ],
            thread_id="src",
            budget=Limits(tokens=100_000),
        )
        try:
            await session2.submit("more budget")
            outcome = await run_update(session2.store, _manager(session2), now=now + MEMORY_RETRY_SECONDS + 1)
            assert outcome.status == "updated"
            grants = [a["grant"] for a in _memory_allocations(session2.store, "src")]
            assert grants[-1] == {"tokens": MEMORY_TOKEN_BUDGET, "steps": 1}
        finally:
            await session2.aclose()

    asyncio.run(scenario())


# ---------------------------------------------------------------------------
# Q6:aclose 取消 driver 不 join 死等(lease 交下次接管;子 driver 经收尾链收敛)
# ---------------------------------------------------------------------------


def test_aclose_cancels_driver_without_join(workspace: Path) -> None:
    async def scenario() -> None:
        ref = _preseed(workspace, "src", "history")
        session = build_session(
            workspace,
            ScriptedAdapter(
                [
                    _resp("parent done"),
                    _resp(_answer([_entry("k1", ref)])),
                    _resp(_answer([_entry("k1", ref)])),
                ],
                delay_seconds=5.0,
            ),
            thread_id="src",
            tools=[],
            enable_memory=True,
        )
        await session.submit("hello")
        await _wait_job(session.store, {"running"})
        started = time.monotonic()
        await session.aclose()
        elapsed = time.monotonic() - started
        # 不 join 死等:5s 剧本延迟下 aclose 必须快速返回。
        assert elapsed < 4.0
        # 子 driver 经 manager 收尾链收敛 CANCELLED(不留孤儿);job 终态化
        # 为 needs_reconciliation(子在途)而非 failed。aclose 后 store 已关,
        # 经新 Store 读事实。
        probe = Store(_db_path(workspace))
        try:
            job = _job(probe)
            assert job["status"] == "needs_reconciliation"
            for turn_id in job["turn_ids"]:
                turn = get_turn(probe, turn_id)
                assert turn is not None and turn.status == "cancelled"
        finally:
            probe.close()

    asyncio.run(scenario())


# ---------------------------------------------------------------------------
# ③ B4 回归钉:真 kill -9 于 maintenance dispatch 后 → needs_reconciliation
# ---------------------------------------------------------------------------


def test_kill9_after_maintenance_dispatch_needs_reconciliation(tmp_path: Path) -> None:
    workspace = tmp_path / "ws"
    workspace.mkdir()
    proc = subprocess.Popen(
        [sys.executable, str(_KILL_SCRIPT), str(workspace)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    sessions: list[Any] = []
    try:
        ready = workspace / "ready"
        deadline = time.monotonic() + 30.0
        while not ready.exists():
            if proc.poll() is not None:
                stderr = proc.stderr.read().decode(errors="replace") if proc.stderr else ""
                pytest.fail(f"kill child exited early rc={proc.returncode}: {stderr[:2000]}")
            if time.monotonic() > deadline:
                pytest.fail("maintenance never dispatched within 30s")
            time.sleep(0.05)

        probe = Store(_db_path(workspace))
        try:
            job = json.loads(get_metadata(probe, JOB_KEY) or "{}")
            assert job["status"] == "running" and job["turn_ids"]
            child_turn = get_turn(probe, job["turn_ids"][0])
            assert child_turn is not None and child_turn.status == "running"
            child_thread_id = child_turn.thread_id
            child_turn_id = child_turn.turn_id
            source_items = probe.read_items("kill-src")
            ref = next(i.item_id for i in source_items if i.kind == ItemKind.USER_MESSAGE)
        finally:
            probe.close()

        # 真 SIGKILL:无用户态清理(旧 memory.py:178 病灶回归钉)。
        os.kill(proc.pid, signal.SIGKILL)
        proc.wait()

        async def reopen() -> None:
            # 重开:sweep 把子 turn 标 interrupted;判据必须覆盖 dispatched-未
            # settle(memory_identity.py:183 含 dispatched)→ needs_reconciliation,
            # 绝不误标 failed。
            session2 = _session(
                workspace,
                [_resp("noop"), _resp(_answer([_entry("k1", ref)]))],
                thread_id="reopen",
            )
            sessions.append(session2)
            await session2.submit("noop")
            manager = _manager(session2)
            # 重开即判(租约未过期:只读 B4 判定,不启动工作,绝不误标 failed)。
            immediate = await run_update(session2.store, manager)
            assert immediate.status == "needs_reconciliation"
            assert immediate.status != "failed"
            # 租约过期接管(Q6:lease 交下次接管)→ 作业行终态化。
            outcome = await run_update(
                session2.store, manager, now=time.time() + MEMORY_LEASE_SECONDS + 1
            )
            assert outcome.status == "needs_reconciliation"
            job = _job(session2.store)
            assert job["status"] == "needs_reconciliation"
            turn = get_turn(session2.store, child_turn_id)
            assert turn is not None and turn.status == "interrupted"

            # 收敛路径:人工 confirm 续跑子 turn → 终态后重评可继续,
            # 且同 batch_hash+version 复用已完成 stage1 答案(不重复学)。
            child_session = _session(
                workspace,
                [_resp(_answer([_entry("k1", ref)]))],
                thread_id=child_thread_id,
            )
            sessions.append(child_session)
            resumed = await child_session.resume(child_turn_id, confirm=True)
            assert resumed.status is TurnStatus.COMPLETED
            allocations_before = len(_memory_allocations(session2.store, "reopen"))
            outcome2 = await run_update(session2.store, manager)
            assert outcome2.status == "updated" and outcome2.version == 1, outcome2
            # stage1 未重 spawn:只有 stage2 一条新分配。
            assert len(_memory_allocations(session2.store, "reopen")) == allocations_before + 1
            state = _state(session2.store)
            assert state is not None
            assert [e["key"] for e in state["entries"]] == ["k1"]

        asyncio.run(reopen())
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait()

        async def close_all() -> None:
            for s in sessions:
                await s.aclose()

        asyncio.run(close_all())