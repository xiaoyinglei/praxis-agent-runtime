"""Task 12:唯一工具管道 —— 路径 + 审批摘要绑定回归(设计 §5.4;硬规则 3)。

仓库无 pytest-asyncio:与 test_tools_builtin.py 同风格,asyncio.run 包装。
monkeypatch praxis.core.permission.decide 证明管道真实经过唯一决策入口
——阶段 2 换 decide 实现,不换接线(Review Focus 4:digest 绑定无懈可击)。
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from praxis.core import permission
from praxis.core.events import EventBus
from praxis.core.pipeline import ExecCtx, execute
from praxis.core.statemachine import TurnStatus, transition
from praxis.protocol.items import ItemKind
from praxis.protocol.permission import Decision, PermissionContext
from praxis.protocol.policy import NetworkPolicy, PermissionPolicy
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
    ToolTarget,
    args_digest,
    json_schema_input,
)
from praxis.store.log import ItemEntry, Store
from praxis.store.turns import create_turn, get_turn
from praxis.tools.builtins.command import create_bash_tool
from praxis.tools.builtins.filesystem import create_read_tool

ARGS_A = {"path": "f.txt"}
ARGS_B = {"path": "OTHER.txt"}


def make_ctx(store: Store, bus: EventBus, workspace: Path, tools: dict[str, Tool]) -> ExecCtx:
    """turn 一律以 running 起步:transition(RUNNING→PAUSED) 才合法。"""
    policy = PermissionPolicy()
    turn = create_turn(store, turn_id="turn-1", thread_id="thread-1", binding={})
    return ExecCtx(
        store=store,
        bus=bus,
        workspace=workspace,
        tools=tools,
        permissions=PermissionContext(
            workspace=str(workspace), policy=policy, mode="on-request"
        ),
        turn_id=turn.turn_id,
        thread_id="thread-1",
        policy_revision=policy.revision,
    )


@pytest.fixture
def ctx(store: Store, workspace: Path) -> ExecCtx:
    (workspace / "f.txt").write_text("hello pipeline")
    return make_ctx(store, EventBus(), workspace, {"read": create_read_tool(workspace)})


def _decision_item(ctx: ExecCtx, verdict: str, digest: str, *, revision: str | None = None) -> None:
    """模拟 Task 14 的审批落笔:approval_decision item(payload 约定 T12 定义)。

    Task 7 R5(b) 迁移:决策必须携带批准时刻的 policy_revision(缺省=当前 ctx
    revision,与 Session._write_approval_decisions 的真实落笔一致);revision
    参数用于构造过期批准的负例。
    """
    ctx.store.append_items(
        ctx.thread_id,
        ctx.turn_id,
        [
            ItemEntry(
                kind=ItemKind.APPROVAL_DECISION,
                payload={
                    "call_id": "c1",
                    "verdict": verdict,
                    "args_digest": digest,
                    "destination": "session",
                    "policy_revision": ctx.policy_revision if revision is None else revision,
                },
            )
        ],
    )


def _resume(ctx: ExecCtx) -> None:
    """恢复语义(Task 13):批准回合回到 running 后才重放 execute。"""
    transition(ctx.store, ctx.bus, ctx.turn_id, TurnStatus.RUNNING, reason="resumed")


# ---------------------------------------------------------------------------
# brief Step 1:六个路径测试
# ---------------------------------------------------------------------------


def test_execute_allow_appends_call_then_result(ctx: ExecCtx) -> None:
    r = asyncio.run(execute(ToolCall("c1", "read", dict(ARGS_A)), ctx))
    items = ctx.store.read_turn_items(ctx.turn_id)
    kinds = [i.kind for i in items]
    assert kinds == [ItemKind.TOOL_CALL, ItemKind.TOOL_RESULT]  # call 先于执行落库
    assert r.status is ResultStatus.OK
    # 占位符替换:工具返回 "pending",管道落库/返回前换成真实 call id
    assert r.tool_call_id == "c1" and r.tool_name == "read"
    call_item, result_item = items
    assert call_item.payload["call_id"] == "c1"
    assert call_item.payload["args_digest"] == args_digest(ARGS_A)
    assert list(call_item.payload["effects"]) == ["read_workspace"]
    assert call_item.payload["idempotent"] is True
    assert result_item.payload["status"] == "ok"
    assert result_item.payload["content"][0]["type"] == "text"
    assert result_item.payload["content"][0]["data"]["text"] == "hello pipeline"
    turn = get_turn(ctx.store, ctx.turn_id)
    assert turn is not None and turn.status == "running"  # allow 不动状态机


def test_ask_pauses_and_writes_request(ctx: ExecCtx, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(permission, "decide", lambda *a: Decision.ask("test"))
    r = asyncio.run(execute(ToolCall("c1", "read", dict(ARGS_A)), ctx))
    assert r.status is ResultStatus.PAUSED
    assert r.tool_call_id == "c1"
    turn = get_turn(ctx.store, ctx.turn_id)
    assert turn is not None and turn.status == "paused"
    items = ctx.store.read_turn_items(ctx.turn_id)
    reqs = [i for i in items if i.kind == ItemKind.APPROVAL_REQUEST]
    assert len(reqs) == 1
    payload = reqs[0].payload
    assert payload["call_id"] == "c1"
    assert payload["tool_name"] == "read"
    assert payload["args_digest"] == args_digest(ARGS_A)
    assert payload["policy_revision"] == ctx.policy_revision
    assert list(payload["effects"]) == ["read_workspace"]
    assert payload["targets"][0]["kind"] == "path"  # type: ignore[index]
    assert all(i.kind != ItemKind.TOOL_CALL for i in items)  # ask 不得执行,更不落 tool_call


def test_ask_reentry_does_not_duplicate_request(
    ctx: ExecCtx, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(permission, "decide", lambda *a: Decision.ask("test"))
    assert asyncio.run(execute(ToolCall("c1", "read", dict(ARGS_A)), ctx)).status is ResultStatus.PAUSED
    _resume(ctx)
    assert asyncio.run(execute(ToolCall("c1", "read", dict(ARGS_A)), ctx)).status is ResultStatus.PAUSED
    reqs = [i for i in ctx.store.read_turn_items(ctx.turn_id) if i.kind == ItemKind.APPROVAL_REQUEST]
    assert len(reqs) == 1  # 同 call_id + 同 digest 幂等重入


def test_reentry_with_changed_args_reasks(ctx: ExecCtx, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(permission, "decide", lambda *a: Decision.ask("test"))
    assert asyncio.run(execute(ToolCall("c1", "read", dict(ARGS_A)), ctx)).status is ResultStatus.PAUSED
    _decision_item(ctx, "allow", args_digest(ARGS_A))
    _resume(ctx)
    r = asyncio.run(execute(ToolCall("c1", "read", dict(ARGS_B)), ctx))
    assert r.status is ResultStatus.PAUSED  # Review Focus 4:不得沿用旧批准
    items = ctx.store.read_turn_items(ctx.turn_id)
    reqs = [i for i in items if i.kind == ItemKind.APPROVAL_REQUEST]
    assert len(reqs) == 2
    assert reqs[-1].payload["args_digest"] == args_digest(ARGS_B)  # 新 ask 用新 digest
    assert all(i.kind != ItemKind.TOOL_CALL for i in items)


def test_reentry_with_matching_digest_executes(ctx: ExecCtx, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(permission, "decide", lambda *a: Decision.ask("test"))
    assert asyncio.run(execute(ToolCall("c1", "read", dict(ARGS_A)), ctx)).status is ResultStatus.PAUSED
    _decision_item(ctx, "allow", args_digest(ARGS_A))
    _resume(ctx)
    r = asyncio.run(execute(ToolCall("c1", "read", dict(ARGS_A)), ctx))
    assert r.status is ResultStatus.OK
    items = ctx.store.read_turn_items(ctx.turn_id)
    assert len([i for i in items if i.kind == ItemKind.TOOL_CALL]) == 1  # 只执行一次
    assert len([i for i in items if i.kind == ItemKind.TOOL_RESULT]) == 1


def test_deny_returns_denied_result(ctx: ExecCtx, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(permission, "decide", lambda *a: Decision.deny("rule"))
    r = asyncio.run(execute(ToolCall("c1", "read", dict(ARGS_A)), ctx))
    assert r.status is ResultStatus.DENIED and r.error_code == "permission_denied"
    assert r.is_error and r.error_message == "rule"
    items = ctx.store.read_turn_items(ctx.turn_id)
    assert [i.kind for i in items] == [ItemKind.TOOL_RESULT]
    assert items[0].payload["error_message"] == "rule"


def test_invalid_input_is_result_not_exception(ctx: ExecCtx) -> None:
    r = asyncio.run(execute(ToolCall("c1", "read", {}), ctx))  # 缺 path
    assert r.status is ResultStatus.ERROR and r.error_code == "invalid_input"
    assert isinstance(r.error_message, str) and r.error_message  # 有界消息非空
    items = ctx.store.read_turn_items(ctx.turn_id)
    assert [i.kind for i in items] == [ItemKind.TOOL_RESULT]  # 无 result 才叫 pending:此处有 result
    assert items[0].payload["is_error"] is True


def test_unknown_tool_is_result_not_exception(ctx: ExecCtx) -> None:
    r = asyncio.run(execute(ToolCall("c1", "nope", {}), ctx))
    assert r.status is ResultStatus.ERROR and r.error_code == "unknown_tool"
    assert [i.kind for i in ctx.store.read_turn_items(ctx.turn_id)] == [ItemKind.TOOL_RESULT]


# ---------------------------------------------------------------------------
# Step 8:占位符替换 + 模型输出字节上限
# ---------------------------------------------------------------------------


def _echo_tool(max_bytes: int) -> Tool:
    schema = {"type": "object", "properties": {}, "additionalProperties": False}
    validate = json_schema_input(schema)

    async def run(arguments: dict, exec_ctx: ToolExecContext) -> ToolResult:
        return ToolResult(
            tool_call_id="pending",  # documented convention:管道负责替换
            tool_name="echo",
            content=(ToolContentBlock(type="text", data={"text": "x" * 100}),),
            status=ResultStatus.OK,
        )

    return Tool(
        definition=ToolDefinition(name="echo", description="echo toy", input_schema=schema),
        validate_input=validate,
        run=run,
        static_effects=frozenset(),
        resolve_use=lambda arguments: ResolvedUse(effects=frozenset(), targets=()),
        idempotent=True,
        concurrency_safe=True,
        cancellation_mode=CancellationMode.COOPERATIVE,
        interrupt_behavior=InterruptBehavior.CANCEL,
        timeout_seconds=30,
        max_model_output_bytes=max_bytes,
    )


def test_output_bounded_to_model_limit(store: Store, workspace: Path) -> None:
    ctx = make_ctx(store, EventBus(), workspace, {"echo": _echo_tool(10)})
    r = asyncio.run(execute(ToolCall("c1", "echo", {}), ctx))
    assert r.tool_call_id == "c1"  # pending 占位符已替换
    assert r.truncated and len(r.text.encode("utf-8")) <= 10
    result_item = ctx.store.read_turn_items(ctx.turn_id)[-1]
    assert result_item.payload["truncated"] is True
    assert result_item.payload["content"][0]["data"]["text"] == "x" * 10  # type: ignore[index]


# ---------------------------------------------------------------------------
# Task 7 R5:审批重入硬化 —— 新鲜 deny 终局 / revision 绑定 / 执行时地板守卫
# ---------------------------------------------------------------------------


def _floor_write_tool(target: Path, runs: list[int]) -> Tool:
    """resolve_use 直指任意 target 的写工具(不经 checked_path 围栏):
    地板目标由此进入决策/守卫,模拟 ask 规则放行后批准落笔的重入现场。"""
    schema: dict = {"type": "object", "properties": {}, "additionalProperties": False}

    async def run(arguments: dict, exec_ctx: ToolExecContext) -> ToolResult:
        runs.append(1)
        return ToolResult(
            tool_call_id="pending",
            tool_name="floorwriter",
            content=(ToolContentBlock(type="text", data={"text": "ran"}),),
            status=ResultStatus.OK,
        )

    return Tool(
        definition=ToolDefinition(name="floorwriter", description="floor target toy", input_schema=schema),
        validate_input=json_schema_input(schema),
        run=run,
        static_effects=frozenset({ToolEffect.WRITE_WORKSPACE}),
        resolve_use=lambda arguments: ResolvedUse(
            effects=frozenset({ToolEffect.WRITE_WORKSPACE}),
            targets=(ToolTarget(kind="path", value=str(target)),),
        ),
        idempotent=False,
        concurrency_safe=False,
        cancellation_mode=CancellationMode.COOPERATIVE,
        interrupt_behavior=InterruptBehavior.CANCEL,
        timeout_seconds=30,
        max_model_output_bytes=4096,
    )


def test_reentry_with_stale_revision_reasks(ctx: ExecCtx, monkeypatch: pytest.MonkeyPatch) -> None:
    """R5(b):digest 一致但 policy_revision 过期 → 批准作废,走新鲜 decide(re-ask)。"""
    monkeypatch.setattr(permission, "decide", lambda *a: Decision.ask("test"))
    assert asyncio.run(execute(ToolCall("c1", "read", dict(ARGS_A)), ctx)).status is ResultStatus.PAUSED
    _decision_item(ctx, "allow", args_digest(ARGS_A), revision="0" * 64)  # 非当前 revision
    _resume(ctx)
    r = asyncio.run(execute(ToolCall("c1", "read", dict(ARGS_A)), ctx))
    assert r.status is ResultStatus.PAUSED  # 过期批准不得执行
    items = ctx.store.read_turn_items(ctx.turn_id)
    assert all(i.kind != ItemKind.TOOL_CALL for i in items)
    reqs = [i for i in items if i.kind == ItemKind.APPROVAL_REQUEST]
    assert len(reqs) == 1  # 同 call_id+digest 幂等,不重复写请求


def test_reentry_with_legacy_missing_revision_reasks(ctx: ExecCtx, monkeypatch: pytest.MonkeyPatch) -> None:
    """R5(b):legacy 决策 item 缺 policy_revision → fail-closed 作废,re-ask。"""
    monkeypatch.setattr(permission, "decide", lambda *a: Decision.ask("test"))
    assert asyncio.run(execute(ToolCall("c1", "read", dict(ARGS_A)), ctx)).status is ResultStatus.PAUSED
    ctx.store.append_items(
        ctx.thread_id,
        ctx.turn_id,
        [
            ItemEntry(
                kind=ItemKind.APPROVAL_DECISION,
                payload={
                    "call_id": "c1",
                    "verdict": "allow",
                    "args_digest": args_digest(ARGS_A),
                    "destination": "session",
                },
            )
        ],
    )
    _resume(ctx)
    r = asyncio.run(execute(ToolCall("c1", "read", dict(ARGS_A)), ctx))
    assert r.status is ResultStatus.PAUSED
    assert all(i.kind != ItemKind.TOOL_CALL for i in ctx.store.read_turn_items(ctx.turn_id))


def test_fresh_deny_not_overridden_by_bound_allow(ctx: ExecCtx, monkeypatch: pytest.MonkeyPatch) -> None:
    """R5(a):新鲜 decide deny(规则)是终局——digest/revision 全匹配的 bound allow 不得翻案。"""
    monkeypatch.setattr(permission, "decide", lambda *a: Decision.deny("rule: hard deny"))
    _decision_item(ctx, "allow", args_digest(ARGS_A))
    r = asyncio.run(execute(ToolCall("c1", "read", dict(ARGS_A)), ctx))
    assert r.status is ResultStatus.DENIED and r.error_code == "permission_denied"
    assert r.error_message == "rule: hard deny"
    assert [i.kind for i in ctx.store.read_turn_items(ctx.turn_id)] == [
        ItemKind.APPROVAL_DECISION,
        ItemKind.TOOL_RESULT,
    ]


def test_fresh_floor_deny_not_overridden_by_bound_allow(store: Store, workspace: Path) -> None:
    """R5(a):新鲜地板 deny 同样终局(真实 decide,非 monkeypatch)。"""
    (workspace / ".git").mkdir()
    runs: list[int] = []
    tool = _floor_write_tool(workspace / ".git" / "config", runs)
    ctx = make_ctx(store, EventBus(), workspace, {"floorwriter": tool})
    _decision_item(ctx, "allow", args_digest({}))
    r = asyncio.run(execute(ToolCall("c1", "floorwriter", {}), ctx))
    assert r.status is ResultStatus.DENIED and r.error_code == "permission_denied"
    assert isinstance(r.error_message, str) and r.error_message.startswith("floor:")
    assert runs == []
    assert all(i.kind != ItemKind.TOOL_CALL for i in ctx.store.read_turn_items(ctx.turn_id))


def test_fresh_sandbox_deny_not_overridden_by_bound_allow(store: Store, workspace: Path) -> None:
    """R5(a):沙箱不可用的新鲜 deny 终局(bound allow 不得把 bash 放进无沙箱执行)。"""
    policy = PermissionPolicy(network=NetworkPolicy(decision="allow"))
    turn = create_turn(store, turn_id="turn-1", thread_id="thread-1", binding={})
    ctx = ExecCtx(
        store=store,
        bus=EventBus(),
        workspace=workspace,
        tools={"bash": create_bash_tool(workspace)},
        permissions=PermissionContext(workspace=str(workspace), policy=policy, mode="on-request"),
        turn_id=turn.turn_id,
        thread_id="thread-1",
        policy_revision=policy.revision,
    )
    args = {"command": "echo should-not-run"}
    _decision_item(ctx, "allow", args_digest(args))
    r = asyncio.run(execute(ToolCall("c1", "bash", args), ctx))
    assert r.status is ResultStatus.DENIED and r.error_message == "sandbox_unavailable"
    assert all(i.kind != ItemKind.TOOL_CALL for i in ctx.store.read_turn_items(ctx.turn_id))


def test_floor_guard_denies_approved_floor_target(
    store: Store, workspace: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """R5(c):ask 规则 + 地板目标 → 批准 → 执行时守卫仍 deny(地板不可批准)。"""
    monkeypatch.setattr(permission, "decide", lambda *a: Decision.ask("test"))
    (workspace / ".git").mkdir()
    runs: list[int] = []
    tool = _floor_write_tool(workspace / ".git" / "config", runs)
    ctx = make_ctx(store, EventBus(), workspace, {"floorwriter": tool})

    assert asyncio.run(execute(ToolCall("c1", "floorwriter", {}), ctx)).status is ResultStatus.PAUSED
    _decision_item(ctx, "allow", args_digest({}))
    _resume(ctx)
    r = asyncio.run(execute(ToolCall("c1", "floorwriter", {}), ctx))

    assert r.status is ResultStatus.DENIED and r.error_code == "permission_denied"
    assert isinstance(r.error_message, str) and r.error_message.startswith("floor:")
    assert runs == []  # 从未 dispatch
    items = ctx.store.read_turn_items(ctx.turn_id)
    assert all(i.kind != ItemKind.TOOL_CALL for i in items)
    assert len([i for i in items if i.kind == ItemKind.TOOL_RESULT]) == 1


def test_floor_guard_checks_real_symlink_candidate(
    store: Store, workspace: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """R5(c) 双候选:词法路径干净、realpath 命中地板 → 守卫仍 deny(symlink 不转移豁免)。"""
    monkeypatch.setattr(permission, "decide", lambda *a: Decision.ask("test"))
    (workspace / ".git").mkdir()
    (workspace / "link").symlink_to(workspace / ".git")
    runs: list[int] = []
    tool = _floor_write_tool(workspace / "link" / "config", runs)
    ctx = make_ctx(store, EventBus(), workspace, {"floorwriter": tool})

    assert asyncio.run(execute(ToolCall("c1", "floorwriter", {}), ctx)).status is ResultStatus.PAUSED
    _decision_item(ctx, "allow", args_digest({}))
    _resume(ctx)
    r = asyncio.run(execute(ToolCall("c1", "floorwriter", {}), ctx))

    assert r.status is ResultStatus.DENIED and r.error_code == "permission_denied"
    assert isinstance(r.error_message, str) and r.error_message.startswith("floor:")
    assert runs == []
    assert all(i.kind != ItemKind.TOOL_CALL for i in ctx.store.read_turn_items(ctx.turn_id))
