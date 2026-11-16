"""Task 14:审批窗口化——R-T7-2 真正闭合 + 子审批面同窗(设计 §5.4/§5.9)。

绑定键升级:(call_id, args_digest, policy_revision) →
**(window_seq, call_id, args_digest, policy_revision)**。window_seq = 声明该调用的
model_response item 的 seq(模型路径);None = CLI/工具直调域(天然无跨窗继承面)。

五测试矩阵(brief 钉死):
① 主案:turn 内轮 1 用 call_0(digest A)获批并执行,轮 2 复用 call_0+digest A
   → 必须重新 ask(R-T7-2 once 沿用自此关闭;B32 抑制升级 (window,call_id,digest));
② 同窗重入(resume 审批后管道重跑)仍不重复 ask(K-2 零回退);
③ 子审批:父不得有替子批通道(Session.resume thread 校验 + 子面端到端);
④ policy_revision/digest/window 三维各单独漂移 → 全 re-ask(2³ 穷举);
⑤ 旧库无 window 键历史 item → fail-closed 视作不匹配(读端 fail-open 只保护
   渲染,决策面缺维=无效——与阶段 2 缺 revision 同纪律)。

仓库无 pytest-asyncio:与 test_pipeline.py 同风格,asyncio.run 包装。
"""

from __future__ import annotations

import asyncio
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any
from uuid import uuid4

import pytest

from praxis.core import permission
from praxis.core.agents import ChildHandle, ChildStatus, SubagentManager
from praxis.core.events import EventBus
from praxis.core.pipeline import ExecCtx, execute
from praxis.core.recovery import sweep
from praxis.core.statemachine import TurnStatus, transition
from praxis.core.turn import pending_approval_summaries
from praxis.interfaces.sdk import ScriptedAdapter, open_session
from praxis.protocol.items import Item, ItemKind
from praxis.protocol.json import JsonValue
from praxis.protocol.models import ModelResponse, ModelToolCall, Usage
from praxis.protocol.permission import Decision, PermissionContext
from praxis.protocol.policy import PermissionPolicy
from praxis.protocol.sandbox import ToolExecContext
from praxis.protocol.tools import (
    CancellationMode,
    InterruptBehavior,
    ResolvedUse,
    ResultStatus,
    Tool,
    ToolCall,
    ToolContentBlock,
    ToolDefinition,
    ToolEffect,
    ToolResult,
    args_digest,
)
from praxis.store.log import ItemEntry, Store
from praxis.store.turns import create_turn, get_turn
from praxis.tools.builtins.filesystem import create_read_tool

ARGS_A = {"path": "f.txt"}
ARGS_B = {"path": "OTHER.txt"}
WINDOW_1 = 11
WINDOW_2 = 12


# ---------------------------------------------------------------------------
# 管道级脚手架(与 test_pipeline.py 同口径:turn 以 running 起步)
# ---------------------------------------------------------------------------


def _ctx(store: Store, workspace: Path) -> ExecCtx:
    (workspace / "f.txt").write_text("hello window", encoding="utf-8")
    policy = PermissionPolicy()
    turn = create_turn(store, turn_id="turn-1", thread_id="thread-1", binding={})
    return ExecCtx(
        store=store,
        bus=EventBus(),
        workspace=workspace,
        tools={"read": create_read_tool(workspace)},
        permissions=PermissionContext(workspace=str(workspace), policy=policy, mode="on-request"),
        turn_id=turn.turn_id,
        thread_id="thread-1",
        policy_revision=policy.revision,
    )


@pytest.fixture
def ctx(store: Store, workspace: Path) -> ExecCtx:
    return _ctx(store, workspace)


def _put_decision(
    ctx: ExecCtx,
    *,
    digest: str,
    window_seq: int | None = WINDOW_1,
    revision: str | None = None,
    verdict: str = "allow",
    with_window: bool = True,
) -> None:
    """模拟 Session._write_approval_decisions 落笔;with_window=False 构造 legacy 缺维 item。"""
    payload: dict[str, JsonValue] = {
        "call_id": "c1",
        "verdict": verdict,
        "args_digest": digest,
        "destination": "once",
        "policy_revision": ctx.policy_revision if revision is None else revision,
    }
    if with_window:
        payload["window_seq"] = window_seq
    ctx.store.append_items(
        ctx.thread_id, ctx.turn_id, [ItemEntry(kind=ItemKind.APPROVAL_DECISION, payload=payload)]
    )


def _resume(ctx: ExecCtx) -> None:
    transition(ctx.store, ctx.bus, ctx.turn_id, TurnStatus.RUNNING, reason="resumed")


def _kinds(ctx: ExecCtx) -> list[str]:
    return [i.kind for i in ctx.store.read_turn_items(ctx.turn_id)]


def _requests(ctx: ExecCtx) -> list[Item]:
    return [i for i in ctx.store.read_turn_items(ctx.turn_id) if i.kind == ItemKind.APPROVAL_REQUEST]


# ---------------------------------------------------------------------------
# ① 主案:跨窗复用同 call_id + 同 digest → 必须重新 ask(R-T7-2 闭合)
# ---------------------------------------------------------------------------


def test_cross_window_same_call_id_same_digest_reasks(
    ctx: ExecCtx, monkeypatch: pytest.MonkeyPatch
) -> None:
    """轮 1(window 11)获批并执行;轮 2(window 12)同 id 同 digest → re-ask。

    旧语义(窗口无关绑定)会沿用轮 1 的 once 批准直接执行——本测试钉死闭合。
    """
    monkeypatch.setattr(permission, "decide", lambda *a: Decision.ask("test"))
    digest = args_digest(ARGS_A)
    call_1 = ToolCall("c1", "read", dict(ARGS_A), window_seq=WINDOW_1)
    assert asyncio.run(execute(call_1, ctx)).status is ResultStatus.PAUSED
    assert _requests(ctx)[0].payload["window_seq"] == WINDOW_1  # request 携带声明窗口
    _put_decision(ctx, digest=digest, window_seq=WINDOW_1)
    _resume(ctx)
    assert asyncio.run(execute(call_1, ctx)).status is ResultStatus.OK

    # 轮 2:新声明窗口,同 call_id、同参数、同 revision——不得继承轮 1 批准。
    call_2 = ToolCall("c1", "read", dict(ARGS_A), window_seq=WINDOW_2)
    assert asyncio.run(execute(call_2, ctx)).status is ResultStatus.PAUSED
    reqs = _requests(ctx)
    assert len(reqs) == 2  # B32 抑制升级 (window, call_id, digest):新窗口必须写新请求
    assert reqs[0].payload["window_seq"] == WINDOW_1
    assert reqs[1].payload["window_seq"] == WINDOW_2
    assert reqs[1].payload["args_digest"] == digest
    assert _kinds(ctx).count(ItemKind.TOOL_CALL) == 1  # 轮 2 未 dispatch

    # 轮 2 获批后在同窗执行恰好一次。
    _put_decision(ctx, digest=digest, window_seq=WINDOW_2)
    _resume(ctx)
    assert asyncio.run(execute(call_2, ctx)).status is ResultStatus.OK
    assert _kinds(ctx).count(ItemKind.TOOL_CALL) == 2
    assert len(_requests(ctx)) == 2  # 同窗重入不重复写请求


def _resp(content: str, calls: tuple[tuple[str, str, dict[str, Any]], ...] = ()) -> ModelResponse:
    return ModelResponse(
        content=content,
        tool_calls=tuple(ModelToolCall(call_id=c, name=n, arguments=a) for c, n, a in calls),
        usage=Usage(input_tokens=1, output_tokens=1),
        finish_reason="tool_calls" if calls else "stop",
    )


def test_session_cross_window_reuse_reasks_end_to_end(
    workspace: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """① 端到端:submit → resume(批准) → 轮 2 同 id 同 digest 必须再 PAUSED 且可批。"""
    monkeypatch.setattr(permission, "decide", lambda *a: Decision.ask("test"))
    args = {"path": "a.txt", "content": "same-content"}
    digest = args_digest(args)
    adapter = ScriptedAdapter(
        [
            _resp("", (("call_0", "write", args),)),
            _resp("", (("call_0", "write", args),)),
            _resp("all done"),
        ]
    )
    session = open_session(workspace, adapter)
    try:
        first = asyncio.run(session.submit("go"))
        assert first.status is TurnStatus.PAUSED
        assert [(p.call_id, p.args_digest) for p in first.pending_approvals] == [("call_0", digest)]

        second = asyncio.run(session.resume(first.turn_id, approval={"call_0": "allow"}))
        # R-T7-2 闭合核心:轮 1 已执行,轮 2 同 id 同 digest 不得沿用 once 批准。
        assert second.status is TurnStatus.PAUSED
        assert second.reason == "approval_required"
        assert [(p.call_id, p.args_digest) for p in second.pending_approvals] == [("call_0", digest)]

        items = session.store.read_turn_items(first.turn_id)
        reqs = [i for i in items if i.kind == ItemKind.APPROVAL_REQUEST]
        decisions = [i for i in items if i.kind == ItemKind.APPROVAL_DECISION]
        assert len(reqs) == 2 and len(decisions) == 1
        assert reqs[0].payload["window_seq"] != reqs[1].payload["window_seq"]
        assert decisions[0].payload["window_seq"] == reqs[0].payload["window_seq"]
        assert sum(1 for i in items if i.kind == ItemKind.TOOL_CALL) == 1  # 只有轮 1 执行

        third = asyncio.run(session.resume(first.turn_id, approval={"call_0": "allow"}))
        assert third.status is TurnStatus.COMPLETED and third.output == "all done"
        assert (workspace / "a.txt").read_text(encoding="utf-8") == "same-content"
        items = session.store.read_turn_items(first.turn_id)
        assert sum(1 for i in items if i.kind == ItemKind.TOOL_CALL) == 2
        assert sum(1 for i in items if i.kind == ItemKind.APPROVAL_DECISION) == 2
        assert session.result(first.turn_id).pending_approvals == ()
    finally:
        session.close()


# ---------------------------------------------------------------------------
# ② 同窗重入:K-2 零回退
# ---------------------------------------------------------------------------


def test_same_window_reentry_executes_without_reask(
    ctx: ExecCtx, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(permission, "decide", lambda *a: Decision.ask("test"))
    call = ToolCall("c1", "read", dict(ARGS_A), window_seq=WINDOW_1)
    assert asyncio.run(execute(call, ctx)).status is ResultStatus.PAUSED
    _put_decision(ctx, digest=args_digest(ARGS_A), window_seq=WINDOW_1)
    _resume(ctx)
    assert asyncio.run(execute(call, ctx)).status is ResultStatus.OK
    assert len(_requests(ctx)) == 1
    assert _kinds(ctx).count(ItemKind.TOOL_CALL) == 1


def test_same_window_ask_reentry_dedups_request(
    ctx: ExecCtx, monkeypatch: pytest.MonkeyPatch
) -> None:
    """未批准的同窗重入仍幂等去重(K-2):不重复写 request。"""
    monkeypatch.setattr(permission, "decide", lambda *a: Decision.ask("test"))
    call = ToolCall("c1", "read", dict(ARGS_A), window_seq=WINDOW_1)
    assert asyncio.run(execute(call, ctx)).status is ResultStatus.PAUSED
    _resume(ctx)
    assert asyncio.run(execute(call, ctx)).status is ResultStatus.PAUSED
    assert len(_requests(ctx)) == 1


def test_none_domain_repeated_call_reuses_approval(
    ctx: ExecCtx, monkeypatch: pytest.MonkeyPatch
) -> None:
    """None = CLI/工具直调域:无声明窗口,天然无跨窗继承面——同域同 id 同 digest 沿用。"""
    monkeypatch.setattr(permission, "decide", lambda *a: Decision.ask("test"))
    call = ToolCall("c1", "read", dict(ARGS_A))  # window_seq 缺省 None
    assert asyncio.run(execute(call, ctx)).status is ResultStatus.PAUSED
    assert _requests(ctx)[0].payload["window_seq"] is None
    _put_decision(ctx, digest=args_digest(ARGS_A), window_seq=None)
    _resume(ctx)
    assert asyncio.run(execute(call, ctx)).status is ResultStatus.OK
    assert asyncio.run(execute(call, ctx)).status is ResultStatus.OK
    assert len(_requests(ctx)) == 1
    assert _kinds(ctx).count(ItemKind.TOOL_CALL) == 2


# ---------------------------------------------------------------------------
# ③ 子审批面:父不得有替子批通道(thread 校验 + 子面端到端同窗绑定)
# ---------------------------------------------------------------------------


def _toy(name: str, run: Any, *, effects: frozenset[ToolEffect] = frozenset()) -> Tool:
    return Tool(
        definition=ToolDefinition(name=name, description=f"{name} toy", input_schema={"type": "object"}),
        validate_input=lambda arguments: arguments,
        run=run,
        static_effects=effects,
        resolve_use=lambda arguments: ResolvedUse(effects=effects, targets=()),
        idempotent=True,
        concurrency_safe=True,
        cancellation_mode=CancellationMode.COOPERATIVE,
        interrupt_behavior=InterruptBehavior.CANCEL,
        timeout_seconds=30,
        max_model_output_bytes=64_000,
    )


def test_parent_cannot_approve_child_turn(workspace: Path) -> None:
    """③ 子 PAUSED 待批:父 Session.resume(子 turn) 必须被 thread 校验拒绝(零写入);
    人类对子 session 直接批准续跑,子的 decision 绑定子自己的声明窗口。"""
    holder: list[SubagentManager] = []
    handles: list[ChildHandle] = []
    statuses: list[ChildStatus] = []

    async def spawn_run(arguments: Mapping[str, JsonValue], exec_ctx: ToolExecContext) -> ToolResult:
        handle = await holder[0].spawn("child task")
        handles.append(handle)
        statuses.append(await holder[0].wait(handle.child_thread_id, timeout=5.0))
        return ToolResult(
            tool_call_id="pending",
            tool_name="spawn_probe",
            content=(ToolContentBlock(type="text", data={"text": handle.child_thread_id}),),
            status=ResultStatus.OK,
        )

    async def ask_run(arguments: Mapping[str, JsonValue], exec_ctx: ToolExecContext) -> ToolResult:
        return ToolResult(
            tool_call_id="pending",
            tool_name="ask_probe",
            content=(ToolContentBlock(type="text", data={"text": "approved run"}),),
            status=ResultStatus.OK,
        )

    # NETWORK 无 url target → allow_rules 覆盖不成立 → fallback ask(子面暂停)。
    tools = [_toy("spawn_probe", spawn_run), _toy("ask_probe", ask_run, effects=frozenset({ToolEffect.NETWORK}))]
    adapter = ScriptedAdapter(
        [
            _resp("", (("s1", "spawn_probe", {}),)),
            _resp("", (("a1", "ask_probe", {}),)),
            _resp("parent done"),
            _resp("child done after approval"),
        ]
    )

    async def scenario() -> None:
        parent = open_session(workspace, adapter, tools=tools, thread_id="parent")
        manager = SubagentManager(parent=parent, store=parent.store)
        holder.append(manager)
        try:
            result = await parent.submit("go")
            assert result.status is TurnStatus.COMPLETED
            handle = handles[0]
            status = statuses[0]
            assert status.state == "paused"
            assert len(status.pending_approvals) == 1
            call_id = status.pending_approvals[0].call_id

            child_items_before = parent.store.read_items(handle.child_thread_id)
            with pytest.raises(ValueError, match="different thread"):
                await parent.resume(handle.child_root_turn_id, approval={call_id: "allow"})
            # 父替子批被拒 = 零写入:子 turn 仍 paused,item 一条不多。
            assert get_turn(parent.store, handle.child_root_turn_id).status == "paused"  # type: ignore[union-attr]
            assert parent.store.read_items(handle.child_thread_id) == child_items_before

            # 人类直面子 session:独立打开同库子 thread,批准续跑(同窗绑定)。
            child = open_session(
                workspace, adapter, thread_id=handle.child_thread_id, db_path=parent.store.db_path
            )
            try:
                resumed = await child.resume(handle.child_root_turn_id, approval={call_id: "allow"})
                assert resumed.status is TurnStatus.COMPLETED
                assert resumed.output == "child done after approval"
            finally:
                await child.aclose()

            items = parent.store.read_turn_items(handle.child_root_turn_id)
            declaration = next(
                i.seq for i in items if i.kind == ItemKind.MODEL_RESPONSE and i.payload.get("tool_calls")
            )
            request = next(i for i in items if i.kind == ItemKind.APPROVAL_REQUEST)
            decision = next(i for i in items if i.kind == ItemKind.APPROVAL_DECISION)
            assert request.payload["window_seq"] == declaration
            assert decision.payload["window_seq"] == declaration  # 子审批绑定子自己的声明窗口
            assert (await manager.wait(handle.child_thread_id, timeout=5.0)).state == "done"
        finally:
            await manager.aclose()
            await parent.aclose()

    asyncio.run(scenario())


# ---------------------------------------------------------------------------
# ④ 2³ 穷举:policy_revision / digest / window 任一漂移 → re-ask
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("window_drift", [False, True], ids=["window-same", "window-drift"])
@pytest.mark.parametrize("digest_drift", [False, True], ids=["digest-same", "digest-drift"])
@pytest.mark.parametrize("revision_drift", [False, True], ids=["revision-same", "revision-drift"])
def test_drift_matrix_exhaustive(
    ctx: ExecCtx,
    monkeypatch: pytest.MonkeyPatch,
    window_drift: bool,
    digest_drift: bool,
    revision_drift: bool,
) -> None:
    """三维各单独漂移及组合(8 例):只有全匹配才执行,任一漂移一律 re-ask。"""
    monkeypatch.setattr(permission, "decide", lambda *a: Decision.ask("test"))
    _put_decision(
        ctx,
        digest=args_digest(ARGS_B if digest_drift else ARGS_A),
        window_seq=WINDOW_2 if window_drift else WINDOW_1,
        revision="0" * 64 if revision_drift else None,
    )
    call = ToolCall("c1", "read", dict(ARGS_A), window_seq=WINDOW_1)
    result = asyncio.run(execute(call, ctx))
    if window_drift or digest_drift or revision_drift:
        assert result.status is ResultStatus.PAUSED
        assert all(i.kind != ItemKind.TOOL_CALL for i in ctx.store.read_turn_items(ctx.turn_id))
    else:
        assert result.status is ResultStatus.OK
        assert _kinds(ctx).count(ItemKind.TOOL_CALL) == 1


# ---------------------------------------------------------------------------
# ⑤ legacy 缺 window 键:fail-closed(决策面缺维=无效;读端 fail-open 只保护渲染)
# ---------------------------------------------------------------------------


def test_legacy_decision_missing_window_fail_closed(
    ctx: ExecCtx, monkeypatch: pytest.MonkeyPatch
) -> None:
    """旧库 decision 无 window_seq 键:即使 digest/revision 全匹配也作废 → re-ask。"""
    monkeypatch.setattr(permission, "decide", lambda *a: Decision.ask("test"))
    call = ToolCall("c1", "read", dict(ARGS_A), window_seq=WINDOW_1)
    assert asyncio.run(execute(call, ctx)).status is ResultStatus.PAUSED
    _put_decision(ctx, digest=args_digest(ARGS_A), with_window=False)
    _resume(ctx)
    assert asyncio.run(execute(call, ctx)).status is ResultStatus.PAUSED
    assert all(i.kind != ItemKind.TOOL_CALL for i in ctx.store.read_turn_items(ctx.turn_id))
    assert len(_requests(ctx)) == 1  # 同窗同 digest 同 revision:请求去重仍成立


def test_legacy_decision_missing_window_invalid_in_none_domain(
    ctx: ExecCtx, monkeypatch: pytest.MonkeyPatch
) -> None:
    """缺维=无效是无条件的:None 域调用也不得继承无 window 键的 legacy 批准。"""
    monkeypatch.setattr(permission, "decide", lambda *a: Decision.ask("test"))
    call = ToolCall("c1", "read", dict(ARGS_A))  # window_seq=None
    assert asyncio.run(execute(call, ctx)).status is ResultStatus.PAUSED
    _put_decision(ctx, digest=args_digest(ARGS_A), with_window=False)
    _resume(ctx)
    assert asyncio.run(execute(call, ctx)).status is ResultStatus.PAUSED
    assert all(i.kind != ItemKind.TOOL_CALL for i in ctx.store.read_turn_items(ctx.turn_id))


def test_legacy_request_missing_window_not_deduped_and_approvable(
    ctx: ExecCtx, monkeypatch: pytest.MonkeyPatch
) -> None:
    """legacy request 无 window 键:不得被去重吞掉(否则新批准永远绑不上 → B32 同族卡死)。

    re-ask 必须写出带 window 键的新 request;批准携带 window 后同窗执行成功。
    """
    monkeypatch.setattr(permission, "decide", lambda *a: Decision.ask("test"))
    ctx.store.append_items(
        ctx.thread_id,
        ctx.turn_id,
        [
            ItemEntry(
                kind=ItemKind.APPROVAL_REQUEST,
                payload={
                    "call_id": "c1",
                    "tool_name": "read",
                    "arguments": dict(ARGS_A),
                    "effects": ("read_workspace",),
                    "targets": (),
                    "args_digest": args_digest(ARGS_A),
                    "policy_revision": ctx.policy_revision,
                },
            )
        ],
    )
    call = ToolCall("c1", "read", dict(ARGS_A))  # None 域
    assert asyncio.run(execute(call, ctx)).status is ResultStatus.PAUSED
    reqs = _requests(ctx)
    assert len(reqs) == 2  # legacy 请求不算重复:新请求带 window 键
    assert "window_seq" not in reqs[0].payload
    assert reqs[1].payload["window_seq"] is None
    _put_decision(ctx, digest=args_digest(ARGS_A), window_seq=None)
    _resume(ctx)
    assert asyncio.run(execute(call, ctx)).status is ResultStatus.OK


def _approval_items(store: Store, entries: Sequence[ItemEntry]) -> list[Item]:
    create_turn(store, turn_id="turn-x", thread_id="thread-x", binding={})
    store.append_items("thread-x", "turn-x", list(entries))
    return store.read_turn_items("turn-x")


def test_pending_summaries_legacy_decision_never_suppresses(store: Store) -> None:
    """展示面同口径:缺 window 键的 legacy decision 不得抑制带窗请求(B32 同族防卡死)。"""
    revision = PermissionPolicy().revision
    digest = args_digest(ARGS_A)
    items = _approval_items(
        store,
        [
            ItemEntry(
                kind=ItemKind.APPROVAL_REQUEST,
                payload={
                    "call_id": "c1",
                    "tool_name": "read",
                    "args_digest": digest,
                    "effects": ("read_workspace",),
                    "policy_revision": revision,
                    "window_seq": WINDOW_1,
                },
            ),
            ItemEntry(
                kind=ItemKind.APPROVAL_DECISION,
                payload={
                    "call_id": "c1",
                    "verdict": "allow",
                    "args_digest": digest,
                    "destination": "once",
                    "policy_revision": revision,
                },
            ),
        ],
    )
    summaries = pending_approval_summaries(items, policy_revision=revision)
    assert [(s.call_id, s.args_digest) for s in summaries] == [("c1", digest)]


def test_pending_summaries_window_drift_shows_new_request(store: Store) -> None:
    """轮 1 已批(window 11),轮 2 同 id 同 digest 新窗请求(window 12)必须呈现。"""
    revision = PermissionPolicy().revision
    digest = args_digest(ARGS_A)
    items = _approval_items(
        store,
        [
            ItemEntry(
                kind=ItemKind.APPROVAL_REQUEST,
                payload={
                    "call_id": "c1",
                    "tool_name": "read",
                    "args_digest": digest,
                    "effects": ("read_workspace",),
                    "policy_revision": revision,
                    "window_seq": WINDOW_1,
                },
            ),
            ItemEntry(
                kind=ItemKind.APPROVAL_DECISION,
                payload={
                    "call_id": "c1",
                    "verdict": "allow",
                    "args_digest": digest,
                    "destination": "once",
                    "policy_revision": revision,
                    "window_seq": WINDOW_1,
                },
            ),
            ItemEntry(
                kind=ItemKind.APPROVAL_REQUEST,
                payload={
                    "call_id": "c1",
                    "tool_name": "read",
                    "args_digest": digest,
                    "effects": ("read_workspace",),
                    "policy_revision": revision,
                    "window_seq": WINDOW_2,
                },
            ),
        ],
    )
    summaries = pending_approval_summaries(items, policy_revision=revision)
    assert [(s.call_id, s.args_digest) for s in summaries] == [("c1", digest)]
    # 同窗决策则抑制(②的展示面):把轮 2 请求的窗口改回 11 即无可批项。
    same_window = [
        Item(
            item_id=i.item_id,
            thread_id=i.thread_id,
            turn_id=i.turn_id,
            seq=i.seq,
            kind=i.kind,
            payload={**i.payload, "window_seq": WINDOW_1} if i.kind == ItemKind.APPROVAL_REQUEST else i.payload,
            created_at=i.created_at,
        )
        for i in items
    ]
    assert pending_approval_summaries(same_window, policy_revision=revision) == ()


def _model_response_entry(call_id: str, name: str, args: dict[str, Any]) -> ItemEntry:
    return ItemEntry(
        kind=ItemKind.MODEL_RESPONSE,
        payload={
            "content": "write it",
            "tool_calls": ({"call_id": call_id, "name": name, "arguments": args},),
            "usage": {"input_tokens": 1, "output_tokens": 1},
            "finish_reason": "tool_calls",
        },
    )


def test_session_legacy_items_reask_then_approvable(
    workspace: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """⑤ 端到端:旧库 request+decision 均无 window 键(同 revision 同 digest)→
    resume(confirm) 必须 re-ask 且待批可见;再批准后续跑成功(fail-closed 不卡死)。"""
    monkeypatch.setattr(permission, "decide", lambda *a: Decision.ask("test"))
    args = {"path": "f.txt", "content": "legacy window"}
    digest = args_digest(args)
    revision = PermissionPolicy().revision
    session = open_session(workspace, ScriptedAdapter([_resp("all done")]))
    try:
        other = Store(workspace / ".praxis" / "praxis.sqlite")
        try:
            turn_id = f"dead-{uuid4().hex}"
            create_turn(other, turn_id=turn_id, thread_id=session.thread_id, binding={})
            other.append_items(
                session.thread_id,
                turn_id,
                [
                    _model_response_entry("c1", "write", args),
                    ItemEntry(
                        kind=ItemKind.APPROVAL_REQUEST,
                        payload={
                            "call_id": "c1",
                            "tool_name": "write",
                            "arguments": args,
                            "effects": ("write_workspace",),
                            "targets": (),
                            "args_digest": digest,
                            "policy_revision": revision,
                        },
                    ),
                    ItemEntry(
                        kind=ItemKind.APPROVAL_DECISION,
                        payload={
                            "call_id": "c1",
                            "verdict": "allow",
                            "args_digest": digest,
                            "destination": "once",
                            "policy_revision": revision,
                        },
                    ),
                ],
            )
            assert turn_id in sweep(other, EventBus(), workspace)
        finally:
            other.close()

        paused = asyncio.run(session.resume(turn_id, confirm=True))
        assert paused.status is TurnStatus.PAUSED
        assert [(p.call_id, p.args_digest) for p in paused.pending_approvals] == [("c1", digest)]
        items = session.store.read_turn_items(turn_id)
        reqs = [i for i in items if i.kind == ItemKind.APPROVAL_REQUEST]
        assert len(reqs) == 2  # legacy 请求 + 带窗新请求
        assert "window_seq" not in reqs[0].payload
        declaration = next(i.seq for i in items if i.kind == ItemKind.MODEL_RESPONSE)
        assert reqs[1].payload["window_seq"] == declaration

        done = asyncio.run(session.resume(turn_id, approval={"c1": "allow"}))
        assert done.status is TurnStatus.COMPLETED and done.output == "all done"
        assert (workspace / "f.txt").read_text(encoding="utf-8") == "legacy window"
        decisions = [
            i.payload for i in session.store.read_turn_items(turn_id) if i.kind == ItemKind.APPROVAL_DECISION
        ]
        assert decisions[-1]["window_seq"] == declaration  # 新批准携带声明窗口
    finally:
        session.close()