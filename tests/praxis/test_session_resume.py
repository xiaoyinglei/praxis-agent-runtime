"""Task 14:Session/组合根/SDK——人工确认续跑协议(设计 §4.1/§5.8/§5.12)。

全栈集成:ScriptedAdapter + 真实内置工具 + 真实 Store。"崩溃的进程"用第二个
Store 实例 + recovery.sweep 构造(running 且锁空闲 → interrupted);SDK 公开面
(open_session/ConversationBusy/ResumeNotConfirmed/adapters)是本文件唯一的装配
与异常入口——Session 构造只发生在 composition(CI 规则 7)。
仓库无 pytest-asyncio:与 test_turn.py 同风格,asyncio.run 包装。
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Iterator, Sequence
from pathlib import Path
from typing import Any
from uuid import uuid4

import pytest

from praxis.core import permission
from praxis.core.events import EventBus
from praxis.core.locks import thread_execution_lock
from praxis.core.recovery import sweep
from praxis.core.statemachine import TurnStatus
from praxis.interfaces.sdk import (
    ConversationBusy,
    ResumeNotConfirmed,
    ScriptedAdapter,
    open_session,
)
from praxis.protocol.items import ItemKind
from praxis.protocol.models import ModelResponse, ModelToolCall, Usage
from praxis.protocol.permission import Decision
from praxis.protocol.policy import PathRule, PermissionPolicy
from praxis.protocol.tools import args_digest
from praxis.store.log import ItemEntry, Store
from praxis.store.turns import create_turn, list_turns

_EMPTY_USAGE = {"input_tokens": 1, "output_tokens": 1}
_NOT_EXECUTED_TEXT = "[not executed: turn interrupted before dispatch]"
_OUTCOME_UNKNOWN_TEXT = "[outcome unknown: process died during execution]"


# ---------------------------------------------------------------------------
# 脚手架
# ---------------------------------------------------------------------------


def _resp(content: str, calls: tuple[tuple[str, str, dict[str, Any]], ...] = ()) -> ModelResponse:
    return ModelResponse(
        content=content,
        tool_calls=tuple(ModelToolCall(call_id=c, name=n, arguments=a) for c, n, a in calls),
        usage=Usage(input_tokens=3, output_tokens=5),
        finish_reason="tool_calls" if calls else "stop",
    )


def _model_response_entry(content: str, calls: tuple[tuple[str, str, dict[str, Any]], ...] = ()) -> ItemEntry:
    return ItemEntry(
        kind=ItemKind.MODEL_RESPONSE,
        payload={
            "content": content,
            "tool_calls": tuple({"call_id": c, "name": n, "arguments": a} for c, n, a in calls),
            "usage": _EMPTY_USAGE,
            "finish_reason": "tool_calls" if calls else "stop",
        },
    )


def _tool_call_entry(call_id: str, name: str, args: dict[str, Any]) -> ItemEntry:
    return ItemEntry(
        kind=ItemKind.TOOL_CALL,
        payload={
            "call_id": call_id,
            "tool_name": name,
            "arguments": args,
            "effects": (),
            "targets": (),
            "args_digest": args_digest(args),
            "idempotent": name != "bash",
        },
    )


@pytest.fixture
def open_ws(workspace: Path) -> Iterator[Callable[..., Any]]:
    """build_session 快捷入口;测试结束统一 close(写者线程不泄漏)。"""
    made: list[Any] = []

    def _open(adapter: Any, **kwargs: Any) -> Any:
        session = open_session(workspace, adapter, **kwargs)
        made.append(session)
        return session

    yield _open
    for session in reversed(made):
        session.close()


def _with_other_store(workspace: Path, fn: Callable[[Store], Any]) -> Any:
    """"另一进程"视角:独立的 Store 读写同一 sqlite 文件(WAL,提交即可见)。"""
    other = Store(workspace / ".praxis" / "praxis.sqlite")
    try:
        return fn(other)
    finally:
        other.close()


def _dead_turn(workspace: Path, thread_id: str, entries: Sequence[ItemEntry]) -> str:
    """构造崩溃遗留 turn:running + 已落 item → sweep 标 interrupted。"""

    def _make(other: Store) -> str:
        turn_id = f"dead-{uuid4().hex}"
        create_turn(other, turn_id=turn_id, thread_id=thread_id, binding={})
        if entries:
            other.append_items(thread_id, turn_id, list(entries))
        assert turn_id in sweep(other, EventBus(), workspace)
        return turn_id

    return _with_other_store(workspace, _make)


# ---------------------------------------------------------------------------
# brief 草图 1/7:submit 正常路径 + result 从 items 重建 + subscription 烟测
# ---------------------------------------------------------------------------


def test_submit_completes_and_result_reconstructs(workspace: Path, open_ws: Callable[..., Any]) -> None:
    adapter = ScriptedAdapter([_resp("hi there")])
    session = open_ws(adapter)
    sub = session.subscribe()

    result = asyncio.run(session.submit("hello"))

    assert result.status is TurnStatus.COMPLETED
    assert result.output == "hi there"
    assert result.reason == "model_finished"
    assert result.pending_approvals == ()
    assert result.unknown_calls == ()

    turn = session.status(result.turn_id)
    assert turn.status == "completed" and turn.thread_id == session.thread_id
    assert turn.binding == {"model": "scripted"}  # adapter spec id 进 binding

    rebuilt = session.result(result.turn_id)
    assert rebuilt.status is result.status
    assert rebuilt.output == result.output
    assert rebuilt.reason == result.reason

    # user_message 归因到本 turn(先于 create_turn 落笔,同 turn_id 标记)
    def _read(other: Store) -> list[str]:
        return [i.kind for i in other.read_turn_items(result.turn_id)]

    kinds = _with_other_store(workspace, _read)
    assert kinds[0] == ItemKind.USER_MESSAGE

    events: list[Any] = []
    while True:
        try:
            events.append(sub.queue.get_nowait())
        except asyncio.QueueEmpty:
            break
    assert any(type(e).__name__ == "ModelResponseReceived" for e in events)
    assert any(type(e).__name__ == "TurnStatusChanged" for e in events)


def test_submit_busy_raises_conversation_busy(workspace: Path, open_ws: Callable[..., Any]) -> None:
    """另一执行者持锁时 submit → ConversationBusy;锁前动作(append+create_turn)已发生。"""
    session = open_ws(ScriptedAdapter([_resp("never reached")]))
    with thread_execution_lock(workspace, session.thread_id):
        with pytest.raises(ConversationBusy):
            asyncio.run(session.submit("hi"))

    def _check(other: Store) -> tuple[str, list[str]]:
        turns = list_turns(other, thread_id=session.thread_id)
        return (turns[0].status, [i.kind for i in other.read_items(session.thread_id)])

    status, kinds = _with_other_store(workspace, _check)
    assert status == "running"  # brief 钉死的序:锁在 create_turn 之后,孤儿交 sweep
    assert kinds == [ItemKind.USER_MESSAGE]


# ---------------------------------------------------------------------------
# brief 草图 2:INTERRUPTED 人工确认守卫
# ---------------------------------------------------------------------------


def test_resume_interrupted_requires_confirm(workspace: Path, open_ws: Callable[..., Any]) -> None:
    session = open_ws(ScriptedAdapter([_resp("done after confirm")]))
    turn_id = _dead_turn(workspace, session.thread_id, [])

    with pytest.raises(ResumeNotConfirmed):
        asyncio.run(session.resume(turn_id))
    assert session.status(turn_id).status == "interrupted"  # 守卫失败不改状态

    result = asyncio.run(session.resume(turn_id, confirm=True))
    assert result.status is TurnStatus.COMPLETED and result.output == "done after confirm"


# ---------------------------------------------------------------------------
# brief 草图 3:not_started → not_executed,循环继续至 COMPLETED
# ---------------------------------------------------------------------------


def test_resume_not_started_writes_not_executed(workspace: Path, open_ws: Callable[..., Any]) -> None:
    session = open_ws(ScriptedAdapter([_resp("recovered")]))
    call = ("c1", "bash", {"command": "touch should_not_run"})
    turn_id = _dead_turn(workspace, session.thread_id, [_model_response_entry("running bash", (call,))])

    result = asyncio.run(session.resume(turn_id, confirm=True))

    assert result.status is TurnStatus.COMPLETED and result.output == "recovered"

    def _check(other: Store) -> list[Any]:
        items = other.read_turn_items(turn_id)
        results = [i for i in items if i.kind == ItemKind.TOOL_RESULT]
        calls = [i for i in items if i.kind == ItemKind.TOOL_CALL]
        return results, calls

    results, calls = _with_other_store(workspace, _check)
    assert len(results) == 1 and len(calls) == 0  # dispatch 前崩溃:没有 tool_call item
    payload = results[0].payload
    assert payload["call_id"] == "c1" and payload["tool_name"] == "bash"
    assert payload["status"] == "not_executed"
    assert payload["content"][0]["data"]["text"] == _NOT_EXECUTED_TEXT  # type: ignore[index]
    assert not (workspace / "should_not_run").exists()  # bash 绝不因续跑而执行


# ---------------------------------------------------------------------------
# brief 草图 4:非幂等 started → outcome_unknown 暂停;reconcile 续跑不重跑
# ---------------------------------------------------------------------------


def test_resume_non_idempotent_unknown_pauses_then_reconcile(
    workspace: Path, open_ws: Callable[..., Any]
) -> None:
    session = open_ws(ScriptedAdapter([_resp("reconciled and done")]))
    call = ("c1", "bash", {"command": "touch marker.txt"})
    turn_id = _dead_turn(
        workspace, session.thread_id, [_model_response_entry("running bash", (call,)), _tool_call_entry(*call)]
    )

    paused = asyncio.run(session.resume(turn_id, confirm=True))
    assert paused.status is TurnStatus.PAUSED
    assert paused.reason == "outcome_unknown"
    assert paused.unknown_calls == ("c1",)
    assert session.status(turn_id).status == "paused"
    assert not (workspace / "marker.txt").exists()

    def _first_result(other: Store) -> Any:
        return [i for i in other.read_turn_items(turn_id) if i.kind == ItemKind.TOOL_RESULT][0].payload

    unknown_payload = _with_other_store(workspace, _first_result)
    assert unknown_payload["status"] == "outcome_unknown"
    assert unknown_payload["content"][0]["data"]["text"] == _OUTCOME_UNKNOWN_TEXT

    done = asyncio.run(session.resume(turn_id, reconcile={"c1": "failed"}))
    assert done.status is TurnStatus.COMPLETED and done.output == "reconciled and done"

    def _check(other: Store) -> tuple[list[Any], Any]:
        items = other.read_turn_items(turn_id)
        results = [i.payload for i in items if i.kind == ItemKind.TOOL_RESULT]
        turn = list_turns(other, status="completed")
        return results, turn

    results, completed_turns = _with_other_store(workspace, _check)
    assert len(results) == 2  # outcome_unknown 留在日志(审计),reconcile 是第二条
    assert results[1]["status"] == "error"
    assert results[1]["content"][0]["data"]["text"] == "reconciled by user: failed"  # type: ignore[index]
    assert results[1]["metadata"] == {"reconciled": True}
    assert any(t.turn_id == turn_id for t in completed_turns)
    assert not (workspace / "marker.txt").exists()  # reconcile 绝不重跑 bash
    rebuilt = session.result(turn_id)
    assert rebuilt.unknown_calls == ()  # 最新结果已非 outcome_unknown


def test_resume_reconcile_with_ok_outcome(workspace: Path, open_ws: Callable[..., Any]) -> None:
    """reconcile ok → tool_result status=ok;投影只有 ok(R10)由 test_context 钉死。"""
    session = open_ws(ScriptedAdapter([_resp("ok path done")]))
    call = ("c9", "bash", {"command": "true"})
    turn_id = _dead_turn(
        workspace, session.thread_id, [_model_response_entry("running bash", (call,)), _tool_call_entry(*call)]
    )
    asyncio.run(session.resume(turn_id, confirm=True))  # → PAUSED/outcome_unknown
    done = asyncio.run(session.resume(turn_id, reconcile={"c9": "ok"}))
    assert done.status is TurnStatus.COMPLETED

    def _check(other: Store) -> list[Any]:
        return [i.payload for i in other.read_turn_items(turn_id) if i.kind == ItemKind.TOOL_RESULT]

    results = _with_other_store(workspace, _check)
    assert results[-1]["status"] == "ok"
    assert results[-1]["metadata"] == {"reconciled": True}
    assert results[-1]["is_error"] is False


def test_resume_approved_started_non_idempotent_marks_unknown_not_replay(
    workspace: Path, open_ws: Callable[..., Any]
) -> None:
    """R12 安全不变式回归:approve→dispatch→crash 后 started+非幂等仍走 outcome_unknown。

    审批决策(decision item + matching digest 在场)不得成为未知副作用的自动重放
    通道:确认续跑必须暂停等对账,reconcile 后才收尾,期间 bash 绝不重新执行。
    """
    session = open_ws(ScriptedAdapter([_resp("reconciled after crash")]))
    args = {"command": "touch replay_marker.txt"}
    call = ("c1", "bash", args)
    digest = args_digest(args)
    turn_id = _dead_turn(
        workspace,
        session.thread_id,
        [
            _model_response_entry("running bash", (call,)),
            ItemEntry(
                kind=ItemKind.APPROVAL_REQUEST,
                payload={
                    "call_id": "c1",
                    "tool_name": "bash",
                    "arguments": args,
                    "effects": ("execute_process",),
                    "targets": (),
                    "args_digest": digest,
                    "policy_revision": "0" * 64,
                },
            ),
            ItemEntry(
                kind=ItemKind.APPROVAL_DECISION,
                payload={"call_id": "c1", "verdict": "allow", "args_digest": digest, "destination": "once"},
            ),
            _tool_call_entry(*call),  # 崩溃点在 dispatch 之后:tool_call 已落库、无 result
        ],
    )

    paused = asyncio.run(session.resume(turn_id, confirm=True))
    assert paused.status is TurnStatus.PAUSED
    assert paused.reason == "outcome_unknown"
    assert paused.unknown_calls == ("c1",)
    assert not (workspace / "replay_marker.txt").exists()  # 确认续跑绝不重跑 bash

    def _counts(other: Store) -> tuple[int, list[Any]]:
        items = other.read_turn_items(turn_id)
        calls = sum(1 for i in items if i.kind == ItemKind.TOOL_CALL)
        results = [i.payload for i in items if i.kind == ItemKind.TOOL_RESULT]
        return calls, results

    call_count, results = _with_other_store(workspace, _counts)
    assert call_count == 1  # 重放没有经管道二次 dispatch
    assert results[-1]["status"] == "outcome_unknown"
    assert results[-1]["tool_name"] == "bash"

    done = asyncio.run(session.resume(turn_id, reconcile={"c1": "ok"}))
    assert done.status is TurnStatus.COMPLETED and done.output == "reconciled after crash"
    assert not (workspace / "replay_marker.txt").exists()  # reconcile 同样不执行
    assert session.result(turn_id).unknown_calls == ()


# ---------------------------------------------------------------------------
# brief 草图 5:幂等 started → 自动重跑
# ---------------------------------------------------------------------------


def test_resume_idempotent_retries(workspace: Path, open_ws: Callable[..., Any]) -> None:
    (workspace / "note.txt").write_text("contents here", encoding="utf-8")
    session = open_ws(ScriptedAdapter([_resp("read done")]))
    call = ("c1", "read", {"path": "note.txt"})
    turn_id = _dead_turn(
        workspace, session.thread_id, [_model_response_entry("reading", (call,)), _tool_call_entry(*call)]
    )

    result = asyncio.run(session.resume(turn_id, confirm=True))

    assert result.status is TurnStatus.COMPLETED and result.output == "read done"

    def _counts(other: Store) -> tuple[int, list[Any]]:
        items = other.read_turn_items(turn_id)
        return (
            sum(1 for i in items if i.kind == ItemKind.TOOL_CALL),
            [i.payload for i in items if i.kind == ItemKind.TOOL_RESULT],
        )

    call_count, results = _with_other_store(workspace, _counts)
    assert call_count == 2  # 原 tool_call + 重放的 tool_call(经唯一管道)
    assert len(results) == 1
    assert results[0]["status"] == "ok"
    assert "contents here" in results[0]["content"][0]["data"]["text"]  # type: ignore[index]


# ---------------------------------------------------------------------------
# brief 草图 6:steer 不取执行锁(§5.8)
# ---------------------------------------------------------------------------


def test_steer_needs_no_lock(workspace: Path, open_ws: Callable[..., Any]) -> None:
    session = open_ws(ScriptedAdapter([_resp("unused")]))
    with thread_execution_lock(workspace, session.thread_id):
        session.steer("mid-flight note")  # 另一执行者持锁,steer 依然成功

    def _check(other: Store) -> list[Any]:
        return [i for i in other.read_items(session.thread_id) if i.kind == ItemKind.STEERING]

    steering = _with_other_store(workspace, _check)
    assert len(steering) == 1
    assert steering[0].payload == {"text": "mid-flight note"}
    assert steering[0].turn_id is None


# ---------------------------------------------------------------------------
# 审批暂停的续跑(Task 13 交接:resume 先转 RUNNING,再带 resume_calls 重入)
# ---------------------------------------------------------------------------


def test_resume_approval_allow_executes_once_then_completes(
    workspace: Path, open_ws: Callable[..., Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(permission, "decide", lambda *a: Decision.ask("test"))
    args = {"path": "f.txt", "content": "approved content"}
    session = open_ws(ScriptedAdapter([_resp("", (("c1", "write", args),)), _resp("all done")]))

    paused = asyncio.run(session.submit("write it"))
    assert paused.status is TurnStatus.PAUSED and paused.reason == "approval_required"
    assert len(paused.pending_approvals) == 1
    assert paused.pending_approvals[0].call_id == "c1"
    assert not (workspace / "f.txt").exists()  # ask 不得执行

    with pytest.raises(ValueError, match="approval or reconcile"):
        asyncio.run(session.resume(paused.turn_id))

    done = asyncio.run(session.resume(paused.turn_id, approval={"c1": "allow"}))
    assert done.status is TurnStatus.COMPLETED and done.output == "all done"
    assert (workspace / "f.txt").read_text(encoding="utf-8") == "approved content"

    def _check(other: Store) -> tuple[Any, int, int]:
        items = other.read_turn_items(paused.turn_id)
        decisions = [i.payload for i in items if i.kind == ItemKind.APPROVAL_DECISION]
        calls = sum(1 for i in items if i.kind == ItemKind.TOOL_CALL)
        results = sum(1 for i in items if i.kind == ItemKind.TOOL_RESULT)
        return decisions, calls, results

    decisions, call_count, result_count = _with_other_store(workspace, _check)
    assert len(decisions) == 1
    assert decisions[0]["verdict"] == "allow"
    assert decisions[0]["args_digest"] == args_digest(args)  # 摘要绑定进 decision item
    assert decisions[0]["destination"] == "once"
    # R5(b):批准时刻的 policy revision 落 decision item(默认策略的 revision)
    assert decisions[0]["policy_revision"] == PermissionPolicy().revision
    assert call_count == 1 and result_count == 1  # 批准重放恰好一次
    assert session.result(paused.turn_id).pending_approvals == ()


def test_resume_approval_denies_writes_denied_result(
    workspace: Path, open_ws: Callable[..., Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(permission, "decide", lambda *a: Decision.ask("test"))
    args = {"path": "f.txt", "content": "nope"}
    session = open_ws(ScriptedAdapter([_resp("", (("c1", "write", args),)), _resp("denied path")]))

    paused = asyncio.run(session.submit("write it"))
    assert paused.status is TurnStatus.PAUSED

    done = asyncio.run(session.resume(paused.turn_id, approval={"c1": "deny"}))
    assert done.status is TurnStatus.COMPLETED and done.output == "denied path"
    assert not (workspace / "f.txt").exists()

    def _check(other: Store) -> list[Any]:
        return [i.payload for i in other.read_turn_items(paused.turn_id) if i.kind == ItemKind.TOOL_RESULT]

    results = _with_other_store(workspace, _check)
    assert len(results) == 1 and results[0]["status"] == "denied"


def test_resume_approval_stamps_current_revision_not_request_revision(
    workspace: Path, open_ws: Callable[..., Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """R5(b):decision 的 policy_revision 取批准时刻的当前策略,不是 request 的旧
    revision;重入按 decision item 校验——跨策略续跑时批准依然有效且绑定新 revision。"""
    monkeypatch.setattr(permission, "decide", lambda *a: Decision.ask("test"))
    args = {"path": "f.txt", "content": "revision bound"}
    first = open_ws(ScriptedAdapter([_resp("", (("c1", "write", args),)), _resp("不可达")]))
    paused = asyncio.run(first.submit("write it"))
    assert paused.status is TurnStatus.PAUSED

    # 另一策略(revision 不同)的会话续跑同一 turn:批准落笔必须 stamp 新 revision
    new_policy = PermissionPolicy(filesystem=(PathRule(path="extra", access="read"),))
    assert new_policy.revision != PermissionPolicy().revision
    second = open_ws(ScriptedAdapter([_resp("all done")]), policy=new_policy, thread_id=first.thread_id)
    done = asyncio.run(second.resume(paused.turn_id, approval={"c1": "allow"}))
    assert done.status is TurnStatus.COMPLETED and done.output == "all done"
    assert (workspace / "f.txt").read_text(encoding="utf-8") == "revision bound"

    def _check(other: Store) -> list[Any]:
        return [i.payload for i in other.read_turn_items(paused.turn_id) if i.kind == ItemKind.APPROVAL_DECISION]

    decisions = _with_other_store(workspace, _check)
    assert len(decisions) == 1
    assert decisions[0]["policy_revision"] == new_policy.revision


# ---------------------------------------------------------------------------
# 守卫与错误面
# ---------------------------------------------------------------------------


def test_resume_and_result_guards(workspace: Path, open_ws: Callable[..., Any]) -> None:
    session = open_ws(ScriptedAdapter([_resp("fin")]))

    with pytest.raises(ValueError, match="unknown turn_id"):
        asyncio.run(session.resume("nope"))
    with pytest.raises(ValueError, match="unknown turn_id"):
        session.status("nope")
    with pytest.raises(ValueError, match="unknown turn_id"):
        session.result("nope")

    done = asyncio.run(session.submit("hello"))
    assert done.status is TurnStatus.COMPLETED
    with pytest.raises(ValueError, match="not resumable"):
        asyncio.run(session.resume(done.turn_id, confirm=True))

    def _make(other: Store) -> str:
        other.ensure_thread("foreign-thread")
        foreign_turn = f"dead-{uuid4().hex}"
        create_turn(other, turn_id=foreign_turn, thread_id="foreign-thread", binding={})
        return foreign_turn

    foreign_turn = _with_other_store(workspace, _make)
    with pytest.raises(ValueError, match="different thread"):
        asyncio.run(session.resume(foreign_turn, confirm=True))


# ---------------------------------------------------------------------------
# 评审修复 #2:resume 的映射校验必须发生在 transition(RUNNING) 之前——
# 非法 approval/reconcile 输入不得把 turn 留在无执行者的 RUNNING(下次
# open_session 的 sweep 会把它误标 interrupted,伪造"进程死亡"现场)。
# ---------------------------------------------------------------------------


def _turn_item_counts(other: Store, turn_id: str) -> tuple[int, int, int, int]:
    """(总条数, turn_status, approval_decision, tool_result)——失败 resume 须四项全不动。"""
    items = other.read_turn_items(turn_id)
    return (
        len(items),
        sum(1 for i in items if i.kind == ItemKind.TURN_STATUS),
        sum(1 for i in items if i.kind == ItemKind.APPROVAL_DECISION),
        sum(1 for i in items if i.kind == ItemKind.TOOL_RESULT),
    )


def _facts(workspace: Path, turn_id: str) -> tuple[int, int, int, int]:
    return _with_other_store(workspace, lambda other: _turn_item_counts(other, turn_id))


def test_resume_paused_unknown_approval_id_raises_before_transition(
    workspace: Path, open_ws: Callable[..., Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(permission, "decide", lambda *a: Decision.ask("test"))
    args = {"path": "f.txt", "content": "x"}
    session = open_ws(ScriptedAdapter([_resp("", (("c1", "write", args),)), _resp("all done")]))
    paused = asyncio.run(session.submit("write it"))
    assert paused.status is TurnStatus.PAUSED and paused.reason == "approval_required"

    before = _facts(workspace, paused.turn_id)

    with pytest.raises(ValueError, match="no approval_request"):
        asyncio.run(session.resume(paused.turn_id, approval={"nonexistent_call_id": "allow"}))
    assert session.status(paused.turn_id).status == "paused"  # 绝不滞留 RUNNING
    assert _facts(workspace, paused.turn_id) == before  # 零 transition、零写入

    with pytest.raises(ValueError, match="allow/deny"):
        asyncio.run(session.resume(paused.turn_id, approval={"c1": "maybe"}))  # type: ignore[dict-item]
    assert session.status(paused.turn_id).status == "paused"
    assert _facts(workspace, paused.turn_id) == before

    # 失败输入不排队的合法续跑仍然工作
    done = asyncio.run(session.resume(paused.turn_id, approval={"c1": "allow"}))
    assert done.status is TurnStatus.COMPLETED and done.output == "all done"


def test_resume_paused_bad_reconcile_raises_before_transition(
    workspace: Path, open_ws: Callable[..., Any]
) -> None:
    session = open_ws(ScriptedAdapter([_resp("reconciled and done")]))
    call = ("c1", "bash", {"command": "true"})
    turn_id = _dead_turn(
        workspace, session.thread_id, [_model_response_entry("running bash", (call,)), _tool_call_entry(*call)]
    )
    asyncio.run(session.resume(turn_id, confirm=True))  # → PAUSED/outcome_unknown
    assert session.status(turn_id).status == "paused"

    before = _facts(workspace, turn_id)

    with pytest.raises(ValueError, match="not found in turn"):
        asyncio.run(session.resume(turn_id, reconcile={"nonexistent_call_id": "ok"}))
    assert session.status(turn_id).status == "paused"
    assert _facts(workspace, turn_id) == before

    with pytest.raises(ValueError, match="ok/failed"):
        asyncio.run(session.resume(turn_id, reconcile={"c1": "bogus"}))  # type: ignore[dict-item]
    assert session.status(turn_id).status == "paused"
    assert _facts(workspace, turn_id) == before

    done = asyncio.run(session.resume(turn_id, reconcile={"c1": "ok"}))
    assert done.status is TurnStatus.COMPLETED and done.output == "reconciled and done"


def test_resume_interrupted_bad_reconcile_stays_interrupted(
    workspace: Path, open_ws: Callable[..., Any]
) -> None:
    """INTERRUPTED 路径同理:校验失败绝不先转 RUNNING(否则 sweep 场景失真)。"""
    session = open_ws(ScriptedAdapter([_resp("never reached via bad input")]))
    call = ("c1", "bash", {"command": "true"})
    turn_id = _dead_turn(
        workspace, session.thread_id, [_model_response_entry("running bash", (call,)), _tool_call_entry(*call)]
    )
    assert session.status(turn_id).status == "interrupted"
    before = _facts(workspace, turn_id)

    with pytest.raises(ValueError, match="not found in turn"):
        asyncio.run(session.resume(turn_id, confirm=True, reconcile={"typo_id": "ok"}))
    assert session.status(turn_id).status == "interrupted"
    assert _facts(workspace, turn_id) == before

    paused = asyncio.run(session.resume(turn_id, confirm=True))  # 合法确认仍走 outcome_unknown 矩阵
    assert paused.status is TurnStatus.PAUSED and paused.unknown_calls == ("c1",)


# ---------------------------------------------------------------------------
# 评审修复 #3:PAUSED 守卫要求**非空**映射——approval={} 等空映射不是人工决策,
# 放行会静默丢弃待决审批(transition RUNNING → classify 写 not_executed → 完成)。
# ---------------------------------------------------------------------------


def test_resume_paused_empty_maps_rejected(
    workspace: Path, open_ws: Callable[..., Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(permission, "decide", lambda *a: Decision.ask("test"))
    args = {"path": "f.txt", "content": "approved content"}
    session = open_ws(ScriptedAdapter([_resp("", (("c1", "write", args),)), _resp("all done")]))
    paused = asyncio.run(session.submit("write it"))
    assert paused.status is TurnStatus.PAUSED

    before = _facts(workspace, paused.turn_id)

    with pytest.raises(ValueError, match="approval or reconcile"):
        asyncio.run(session.resume(paused.turn_id, approval={}))
    assert session.status(paused.turn_id).status == "paused"
    assert _facts(workspace, paused.turn_id) == before

    with pytest.raises(ValueError, match="approval or reconcile"):
        asyncio.run(session.resume(paused.turn_id, reconcile={}))
    assert session.status(paused.turn_id).status == "paused"
    assert _facts(workspace, paused.turn_id) == before
    assert not (workspace / "f.txt").exists()  # 审批未被静默丢弃,工具未执行

    done = asyncio.run(session.resume(paused.turn_id, approval={"c1": "allow"}))  # 非空映射照常
    assert done.status is TurnStatus.COMPLETED and done.output == "all done"
    assert (workspace / "f.txt").read_text(encoding="utf-8") == "approved content"


# ---------------------------------------------------------------------------
# composition:thread_id 边界校验 + 打开即 sweep
# ---------------------------------------------------------------------------


def test_build_session_validates_thread_id(workspace: Path) -> None:
    for bad in ("", "a/b", "..", "x\x00y", "a\\b"):
        with pytest.raises(ValueError):
            open_session(workspace, ScriptedAdapter([]), thread_id=bad)


def test_build_session_sweeps_dead_turn_on_open(workspace: Path) -> None:
    """打开即 sweep:同 thread 的 dead running turn → interrupted;confirm 后可续跑。"""
    first = open_session(workspace, ScriptedAdapter([]), thread_id="alpha")
    turn_id = _dead_turn(workspace, "alpha", [_model_response_entry("was running", ())])
    first.close()

    second = open_session(workspace, ScriptedAdapter([_resp("swept and resumed")]), thread_id="alpha")
    try:
        assert second.status(turn_id).status == "interrupted"
        result = asyncio.run(second.resume(turn_id, confirm=True))
        assert result.status is TurnStatus.COMPLETED
    finally:
        second.close()
