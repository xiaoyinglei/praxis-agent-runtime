"""阶段4 T5:子代理原语——§5.13 四条已拍板语义 + B2 份额 + B14 级联。

四条语义逐条对应(控制器裁决 R-T5-1..5 一并钉死):
- 同库独立 thread:子 items 写自己的 thread,但库是父库(read_only 与 write 桩
  两种工作区都断言父 store 可读子事实);执行锁按子 thread 独立持有;
- 显式级联:cascade_for 是唯一实现——Session CANCELLED/FAILED 终态挂钩与
  recovery sweep INTERRUPTED 路径都调它;幂等;只到直接子(孙代逐跳);
- resume 不连带子:父 resume(confirm=True) 后子仍 cancelled/interrupted 原状;
- 预算树结算:grant = limits_merge_grant(B2:默认永不吸干父);子终态 →
  settle 推导(未用份额归还);interrupted 子继续占用份额(R-T5-1)。

R-PF-6:同事务 kill 矩阵的真 SIGKILL 归 T9;本文件用异常注入证
"transact 内抛错 → 回滚 → allocation/子根 turn 零残留"。
R-T5-4:wait 过期不取消 driver(asyncio.wait 旧先例 subagents.py:199-201)。

仓库无 pytest-asyncio:与 test_turn.py 同风格,asyncio.run 包装。
"""

from __future__ import annotations

import asyncio
import os
import subprocess
from collections.abc import Mapping
from pathlib import Path
from typing import Any
from uuid import uuid4

import pytest

from praxis.adapters.sandbox.base import probe_backend
from praxis.adapters.sandbox.compile import compile_sandbox, finalize_sandbox
from praxis.composition import build_child_session
from praxis.core import agents as agents_module
from praxis.core.agents import (
    AGENT_TOOL_NAMES,
    ChildGrantError,
    ChildHandle,
    ChildLimitError,
    ChildStatus,
    SubagentManager,
    cascade_for,
)
from praxis.core.budget import build_tree, exposure, remaining
from praxis.core.events import EventBus
from praxis.core.recovery import sweep
from praxis.core.session import ResumeNotConfirmed
from praxis.core.statemachine import TurnStatus
from praxis.interfaces.sdk import ScriptedAdapter, open_session
from praxis.protocol.budget import UNLIMITED, Limits, limits_payload
from praxis.protocol.items import ItemKind
from praxis.protocol.json import JsonValue
from praxis.protocol.models import ModelResponse, ModelToolCall, Usage
from praxis.protocol.policy import PermissionPolicy, path_access, resolve_policy
from praxis.protocol.sandbox import ToolExecContext
from praxis.protocol.tools import (
    CancellationMode,
    InterruptBehavior,
    ResolvedUse,
    ResultStatus,
    Tool,
    ToolContentBlock,
    ToolDefinition,
    ToolEffect,
    ToolResult,
    ToolTarget,
)
from praxis.protocol.workspaces import ChildWorkspaceRequiresGit
from praxis.store.log import ItemEntry, Store
from praxis.store.turns import create_turn, get_turn, list_turns

_ZERO_USAGE = Usage(input_tokens=0, output_tokens=0)
_SANDBOX = probe_backend()


# ---------------------------------------------------------------------------
# 脚手架
# ---------------------------------------------------------------------------


def _resp(
    content: str,
    calls: tuple[tuple[str, str, dict[str, Any]], ...] = (),
    *,
    usage: Usage = _ZERO_USAGE,
) -> ModelResponse:
    return ModelResponse(
        content=content,
        tool_calls=tuple(ModelToolCall(call_id=c, name=n, arguments=a) for c, n, a in calls),
        usage=usage,
        finish_reason="tool_calls" if calls else "stop",
    )


def _tool(
    name: str,
    run: Any,
    *,
    effects: frozenset[ToolEffect] = frozenset(),
    targets: tuple[ToolTarget, ...] = (),
) -> Tool:
    """空 effects 玩具经 allow_rules 的 policy-grant 自动放行(与 test_turn_budget 同口径)。"""
    return Tool(
        definition=ToolDefinition(name=name, description=f"{name} toy", input_schema={"type": "object"}),
        validate_input=lambda arguments: arguments,
        run=run,
        static_effects=effects,
        resolve_use=lambda arguments: ResolvedUse(effects=effects, targets=targets),
        idempotent=True,
        concurrency_safe=True,
        cancellation_mode=CancellationMode.COOPERATIVE,
        interrupt_behavior=InterruptBehavior.CANCEL,
        timeout_seconds=30,
        max_model_output_bytes=64_000,
    )


def _ok(name: str, text: str) -> ToolResult:
    return ToolResult(
        tool_call_id="pending",
        tool_name=name,
        content=(ToolContentBlock(type="text", data={"text": text}),),
        status=ResultStatus.OK,
    )


def _hang_tool(name: str, event: asyncio.Event) -> Tool:
    async def run(arguments: Mapping[str, JsonValue], exec_ctx: ToolExecContext) -> ToolResult:
        await event.wait()
        return _ok(name, f"{name}:released")

    return _tool(name, run)


def _spawn_probe(
    manager_holder: list[SubagentManager],
    results: list[Any],
    *,
    task: str = "child task",
    context_summary: str = "",
    wait_child: bool = True,
    requested: Limits | None = None,
    depth: int = 0,
    workspace_mode: str = "read_only",
    hold_event: asyncio.Event | None = None,
    second_spawn: bool = False,
) -> Tool:
    """父 turn 内的 spawn 探针工具:T6 模型工具面的测试替身(同一条 manager 路径)。

    results 依次收集:handle/异常、(可选)wait 的 ChildStatus、二次 spawn 结局。
    hold_event 非空时 spawn 后挂起父 turn(给测试窗口取消父)。
    """

    async def run(arguments: Mapping[str, JsonValue], exec_ctx: ToolExecContext) -> ToolResult:
        manager = manager_holder[0]
        try:
            handle = await manager.spawn(
                task,
                context_summary=context_summary,
                requested=requested,
                depth=depth,
                workspace_mode=workspace_mode,
            )
        except (ChildGrantError, ChildLimitError, ValueError, RuntimeError) as exc:
            results.append(exc)
            return ToolResult(
                tool_call_id="pending",
                tool_name="spawn_probe",
                is_error=True,
                status=ResultStatus.ERROR,
                error_code="spawn_failed",
                error_message=str(exc),
            )
        results.append(handle)
        if second_spawn:
            try:
                second = await manager.spawn("second child")
                results.append(second)
            except (ChildGrantError, ChildLimitError) as exc:
                results.append(exc)
        if hold_event is not None:
            results.append(manager_holder[0].parent.active_turn_id)
            await hold_event.wait()
        if wait_child:
            results.append(await manager.wait(handle.child_thread_id, timeout=5.0))
        return _ok("spawn_probe", handle.child_thread_id)

    return _tool("spawn_probe", run)


def _allocation_payload(
    *,
    allocation_id: str,
    parent_turn_id: str,
    child_thread_id: str,
    child_root_turn_id: str,
    grant: Limits,
    workspace: str = "",
) -> dict[str, JsonValue]:
    return {
        "allocation_id": allocation_id,
        "parent_turn_id": parent_turn_id,
        "child_thread_id": child_thread_id,
        "child_root_turn_id": child_root_turn_id,
        "workspace_mode": "read_only",
        "workspace": workspace,
        "grant": limits_payload(grant),
        "requested": {},
        "source": "model",
    }


def _manual_family(
    store: Store, workspace: Path, *, layers: int = 2
) -> tuple[list[str], list[str]]:
    """手工构造 running 孤儿家族(无执行锁 = 原主已死):thread/turn 逐层各一。

    返回 (thread_ids, turn_ids);allocation 链 pt0 → pt1 → ...。
    """
    threads: list[str] = []
    turns: list[str] = []
    for level in range(layers):
        thread = store.create_thread()
        turn_id = f"turn-{level}"
        create_turn(
            store,
            turn_id=turn_id,
            thread_id=thread,
            binding={"budget": limits_payload(Limits(tokens=1000 >> level))},
        )
        threads.append(thread)
        turns.append(turn_id)
    for level in range(layers - 1):
        store.append_items(
            threads[level],
            turns[level],
            [
                ItemEntry(
                    kind=ItemKind.CHILD_ALLOCATION,
                    payload=_allocation_payload(
                        allocation_id=f"alloc-{level}",
                        parent_turn_id=turns[level],
                        child_thread_id=threads[level + 1],
                        child_root_turn_id=turns[level + 1],
                        grant=Limits(tokens=500 >> level),
                        workspace=str(workspace),
                    ),
                )
            ],
        )
    return threads, turns


class _StubProvisioner:
    """T7 前的注入桩:prepare 返回独立目录;diff/apply/discard 未实装。"""

    def __init__(self, target: Path) -> None:
        self._target = target
        self.prepared: list[tuple[Path, str, str]] = []

    def prepare(self, parent_root: Path, child_thread_id: str, mode: str) -> Path:
        self.prepared.append((parent_root, child_thread_id, mode))
        self._target.mkdir(parents=True, exist_ok=True)
        return self._target

    def diff(self, ws: Path) -> bytes:
        raise NotImplementedError

    def apply(self, parent_root: Path, patch: bytes, *, preimages: Mapping[str, str | None]) -> Any:
        raise NotImplementedError

    def discard(self, ws: Path) -> None:
        raise NotImplementedError


# ---------------------------------------------------------------------------
# ① B2 份额:默认 grant = floor(remaining × 0.5),父 remaining 被 reserved
# ---------------------------------------------------------------------------


def test_spawn_default_grant_reserves_half_of_parent_remaining(workspace: Path) -> None:
    child_release = asyncio.Event()
    holder: list[SubagentManager] = []
    results: list[Any] = []

    async def scenario() -> None:
        tools = [
            _spawn_probe(holder, results, wait_child=False),
            _hang_tool("child_hang", child_release),
        ]
        # 剧本序钉死:P1 → 子 C1(挂起)→ P2 →(释放后)子 C2。
        adapter = ScriptedAdapter(
            [
                _resp("", (("s1", "spawn_probe", {}),)),
                _resp("", (("h1", "child_hang", {}),)),
                _resp("parent done"),
                _resp("child done"),
            ]
        )
        parent = open_session(
            workspace, adapter, tools=tools, thread_id="parent", budget=Limits(tokens=1000)
        )
        manager = SubagentManager(parent=parent, store=parent.store)
        holder.append(manager)
        try:
            result = await parent.submit("go")
            assert result.status is TurnStatus.COMPLETED
            handle: ChildHandle = results[0]
            # B2 逐字:父剩 1000 → grant.tokens ≤ 500(默认份额上限)。
            assert handle.grant.tokens == 500
            node = build_tree(
                parent.store,
                thread_id="parent",
                turn_id=result.turn_id,
                root_limits=Limits(tokens=1000),
                unit_costs=None,
            )
            child = node.children[handle.allocation_id]
            assert child.state == "active"
            assert child.child_thread_id == handle.child_thread_id
            # 父 remaining 相应被 reserved:1000 − 500(零 usage 事实)。
            assert remaining(node).tokens == 500
            assert node.reserved.tokens == 500
            # children 注册表 = item 折叠(无表列);TurnResult 注记字段透出。
            assert len(result.children_summary) == 1
            assert result.children_summary[0].child_thread_id == handle.child_thread_id
            assert result.children_summary[0].state == "running"
            allocations = [
                item for item in parent.store.read_items("parent") if item.kind == ItemKind.CHILD_ALLOCATION
            ]
            assert len(allocations) == 1
            assert allocations[0].turn_id == result.turn_id
            assert allocations[0].payload["grant"] == {"tokens": 500}
            # 同库独立 thread:子根 turn binding 含 budget/parent_turn_id/root_turn_id。
            child_turn = get_turn(parent.store, handle.child_root_turn_id)
            assert child_turn is not None
            assert child_turn.thread_id == handle.child_thread_id
            assert child_turn.binding["budget"] == {"tokens": 500}
            assert child_turn.binding["parent_turn_id"] == result.turn_id
            assert child_turn.binding["root_turn_id"] == handle.child_root_turn_id
            # 子 items 写自己的 thread,但父 store 同库可读。
            child_items = parent.store.read_items(handle.child_thread_id)
            assert any(item.kind == ItemKind.USER_MESSAGE for item in child_items)
            child_release.set()
            status = await manager.wait(handle.child_thread_id, timeout=5.0)
            assert status.state == "done"
            assert status.conclusion == "child done"
        finally:
            await manager.aclose()
            await parent.aclose()

    asyncio.run(scenario())


def test_spawn_requested_truncated_to_share_and_small_request_honored(workspace: Path) -> None:
    holder: list[SubagentManager] = []
    results: list[Any] = []

    async def scenario() -> None:
        tools = [_spawn_probe(holder, results, requested=Limits(tokens=900))]
        adapter = ScriptedAdapter(
            [
                _resp("", (("s1", "spawn_probe", {}),)),
                _resp("child done"),
                _resp("parent done"),
            ]
        )
        parent = open_session(
            workspace, adapter, tools=tools, thread_id="parent", budget=Limits(tokens=1000)
        )
        manager = SubagentManager(parent=parent, store=parent.store)
        holder.append(manager)
        try:
            result = await parent.submit("go")
            assert result.status is TurnStatus.COMPLETED
            handle: ChildHandle = results[0]
            # requested 900 超份额 → 截断到 500,不报错(B2:上限是算法)。
            assert handle.grant.tokens == 500
            status: ChildStatus = results[1]
            assert status.state == "done"
        finally:
            await manager.aclose()
            await parent.aclose()

    asyncio.run(scenario())


def test_spawn_small_request_granted_as_requested(workspace: Path) -> None:
    holder: list[SubagentManager] = []
    results: list[Any] = []

    async def scenario() -> None:
        tools = [_spawn_probe(holder, results, requested=Limits(tokens=100))]
        adapter = ScriptedAdapter(
            [
                _resp("", (("s1", "spawn_probe", {}),)),
                _resp("child done"),
                _resp("parent done"),
            ]
        )
        parent = open_session(
            workspace, adapter, tools=tools, thread_id="parent", budget=Limits(tokens=1000)
        )
        manager = SubagentManager(parent=parent, store=parent.store)
        holder.append(manager)
        try:
            await parent.submit("go")
            assert results[0].grant.tokens == 100
        finally:
            await manager.aclose()
            await parent.aclose()

    asyncio.run(scenario())


def test_spawn_grant_exhausted_dimension_rejected_zero_write(workspace: Path) -> None:
    holder: list[SubagentManager] = []
    results: list[Any] = []

    async def scenario() -> None:
        tools = [_spawn_probe(holder, results)]
        adapter = ScriptedAdapter(
            [_resp("", (("s1", "spawn_probe", {}),)), _resp("parent done")]
        )
        # 父 tokens=1 → floor(1×0.5)=0 → grant.tokens=0 → ChildGrantError。
        parent = open_session(
            workspace, adapter, tools=tools, thread_id="parent", budget=Limits(tokens=1)
        )
        manager = SubagentManager(parent=parent, store=parent.store)
        holder.append(manager)
        try:
            result = await parent.submit("go")
            assert result.status is TurnStatus.COMPLETED
            assert isinstance(results[0], ChildGrantError)
            assert "tokens" in str(results[0])
            # 拒绝 = 零写入:无 allocation、无子 turn。
            assert not any(
                item.kind == ItemKind.CHILD_ALLOCATION for item in parent.store.read_items("parent")
            )
            assert len(list_turns(parent.store)) == 1
        finally:
            await manager.aclose()
            await parent.aclose()

    asyncio.run(scenario())


# ---------------------------------------------------------------------------
# ② settle 推导:子终态 → 未用份额归还(exposure 回落)
# ---------------------------------------------------------------------------


def test_child_completion_settles_unused_share_back_to_parent(workspace: Path) -> None:
    child_release = asyncio.Event()
    holder: list[SubagentManager] = []
    results: list[Any] = []

    async def scenario() -> None:
        tools = [
            _spawn_probe(holder, results, wait_child=False),
            _hang_tool("child_hang", child_release),
        ]
        adapter = ScriptedAdapter(
            [
                _resp("", (("s1", "spawn_probe", {}),)),
                _resp("", (("h1", "child_hang", {}),), usage=Usage(input_tokens=3, output_tokens=5)),
                _resp("parent done"),
                _resp("child done"),
            ]
        )
        parent = open_session(
            workspace, adapter, tools=tools, thread_id="parent", budget=Limits(tokens=1000)
        )
        manager = SubagentManager(parent=parent, store=parent.store)
        holder.append(manager)
        try:
            result = await parent.submit("go")
            handle: ChildHandle = results[0]
            before = build_tree(
                parent.store,
                thread_id="parent",
                turn_id=result.turn_id,
                root_limits=Limits(tokens=1000),
                unit_costs=None,
            )
            # 子已用 8(C1 usage),active 份额占用 = 500 − 8 = 492。
            assert before.children[handle.allocation_id].used.tokens == 8
            assert before.reserved.tokens == 492
            assert exposure(before).tokens == 500
            child_release.set()
            status = await manager.wait(handle.child_thread_id, timeout=5.0)
            assert status.state == "done"
            assert status.steps_used == 1
            after = build_tree(
                parent.store,
                thread_id="parent",
                turn_id=result.turn_id,
                root_limits=Limits(tokens=1000),
                unit_costs=None,
            )
            # settle 是推导不是写:终态 → reserved 归零,exposure 只剩已用事实。
            assert after.children[handle.allocation_id].state == "done"
            assert after.reserved.tokens == 0
            assert exposure(after).tokens == 8
            assert remaining(after).tokens == 992
            # 无 settle item(守恒由事实只增保证)。
            assert not any(
                item.kind == "settle" for item in parent.store.read_items(handle.child_thread_id)
            )
        finally:
            await manager.aclose()
            await parent.aclose()

    asyncio.run(scenario())


# ---------------------------------------------------------------------------
# ③ 显式级联:父 CANCELLED(session 挂钩)与父 INTERRUPTED(sweep 挂钩)
# ---------------------------------------------------------------------------


def test_parent_cancelled_cascades_cancel_request_to_child(workspace: Path) -> None:
    child_release = asyncio.Event()
    parent_hold = asyncio.Event()
    holder: list[SubagentManager] = []
    results: list[Any] = []

    async def scenario() -> None:
        tools = [
            _spawn_probe(holder, results, wait_child=False, hold_event=parent_hold),
            _hang_tool("child_hang", child_release),
        ]
        adapter = ScriptedAdapter(
            [
                _resp("", (("s1", "spawn_probe", {}),)),
                _resp("", (("h1", "child_hang", {}),)),
                _resp("不可达"),
            ]
        )
        parent = open_session(workspace, adapter, tools=tools, thread_id="parent")
        manager = SubagentManager(parent=parent, store=parent.store)
        holder.append(manager)
        try:
            submit_task = asyncio.create_task(parent.submit("go"))
            await asyncio.sleep(0.05)  # 父进入 hold,子挂起在 child_hang
            parent_turn_id = results[1]
            assert parent.request_cancel(parent_turn_id) == "signalled"
            result = await submit_task
            assert result.status is TurnStatus.CANCELLED
            handle: ChildHandle = results[0]
            # 级联事实:子 thread 收到 source=parent 的 cancel_request。
            cancels = [
                item
                for item in parent.store.read_items(handle.child_thread_id)
                if item.kind == ItemKind.CANCEL_REQUEST
            ]
            assert len(cancels) == 1
            assert cancels[0].payload["source"] == "parent"
            assert cancels[0].payload["parent_turn_id"] == parent_turn_id
            assert cancels[0].turn_id == handle.child_root_turn_id
            # 子执行者步边界收敛:释放挂起工具后子收敛为 CANCELLED(同进程)。
            child_release.set()
            status = await manager.wait(handle.child_thread_id, timeout=5.0)
            assert status.state == "cancelled"
            child_turn = get_turn(parent.store, handle.child_root_turn_id)
            assert child_turn is not None and child_turn.status == "cancelled"
            # B14 对称性:级联走 item 缝,挂起工具被释放后正常结算——步边界
            # 收敛时窗口全闭合,常态零补偿 entries(在途取消的 outcome_unknown
            # 补偿由 request_cancel signalled 路径的测试钉)。
            child_items = parent.store.read_turn_items(handle.child_root_turn_id)
            assert any(
                item.kind == ItemKind.TOOL_RESULT and item.payload.get("status") == "ok"
                for item in child_items
            )
            assert not any(
                item.kind == ItemKind.TOOL_RESULT and item.payload.get("status") == "outcome_unknown"
                for item in child_items
            )
        finally:
            parent_hold.set()
            child_release.set()
            await manager.aclose()
            await parent.aclose()

    asyncio.run(scenario())


def test_sweep_interrupted_parent_cascades_and_interrupted_child_keeps_share(
    workspace: Path, store: Store
) -> None:
    """③b + R-T5-1:sweep 标 interrupted 后级联;子 interrupted 份额仍 reserved。"""
    bus = EventBus()
    threads, turns = _manual_family(store, workspace, layers=2)
    marked = sweep(store, bus, workspace)
    assert set(marked) == set(turns)
    # 级联同样发生(recovery 挂钩):子收到 source=parent 取消事实。
    cancels = [
        item for item in store.read_items(threads[1]) if item.kind == ItemKind.CANCEL_REQUEST
    ]
    assert len(cancels) == 1
    assert cancels[0].payload["source"] == "parent"
    assert cancels[0].payload["parent_turn_id"] == turns[0]
    # R-T5-1:子 interrupted(尚未收敛终态)→ 折叠 active,份额继续占用。
    node = build_tree(
        store,
        thread_id=threads[0],
        turn_id=turns[0],
        root_limits=Limits(tokens=1000),
        unit_costs=None,
    )
    assert node.children["alloc-0"].state == "active"
    assert node.reserved.tokens == 500
    # 子有自己的 interrupted 标记:resume 需各自确认(§5.2 守卫)。
    child = open_session(workspace, ScriptedAdapter([]), thread_id=threads[1], db_path=store.db_path)

    async def scenario() -> None:
        try:
            with pytest.raises(ResumeNotConfirmed):
                await child.resume(turns[1])
            result = await child.resume(turns[1], confirm=True)
            # 步边界读到级联取消事实 → 收敛 CANCELLED(不请求模型)。
            assert result.status is TurnStatus.CANCELLED
            settled = build_tree(
                store,
                thread_id=threads[0],
                turn_id=turns[0],
                root_limits=Limits(tokens=1000),
                unit_costs=None,
            )
            assert settled.children["alloc-0"].state == "cancelled"
            assert settled.reserved.tokens == 0
        finally:
            await child.aclose()

    asyncio.run(scenario())


# ---------------------------------------------------------------------------
# ④ resume 不连带子:父续跑后子仍 cancelled/interrupted 原状
# ---------------------------------------------------------------------------


def test_parent_resume_does_not_revive_cancelled_child(workspace: Path, store: Store) -> None:
    bus = EventBus()
    threads, turns = _manual_family(store, workspace, layers=2)
    sweep(store, bus, workspace)
    child = open_session(workspace, ScriptedAdapter([]), thread_id=threads[1], db_path=store.db_path)
    parent = open_session(
        workspace, ScriptedAdapter([_resp("parent resumed")]), thread_id=threads[0], db_path=store.db_path
    )

    async def scenario() -> None:
        try:
            cancelled = await child.resume(turns[1], confirm=True)
            assert cancelled.status is TurnStatus.CANCELLED
            child_items_before = len(store.read_items(threads[1]))
            result = await parent.resume(turns[0], confirm=True)
            assert result.status is TurnStatus.COMPLETED
            # 子不自动复活:状态原状、零新 item。
            child_turn = get_turn(store, turns[1])
            assert child_turn is not None and child_turn.status == "cancelled"
            assert len(store.read_items(threads[1])) == child_items_before
        finally:
            await child.aclose()
            await parent.aclose()

    asyncio.run(scenario())


def test_parent_resume_leaves_interrupted_child_untouched(workspace: Path, store: Store) -> None:
    """子自身 interrupted(未被级联):父 resume 不连带;份额仍 reserved(R-T5-1)。"""
    bus = EventBus()
    threads, turns = _manual_family(store, workspace, layers=2)
    # 父 PAUSED(不是 running 孤儿 → sweep 不标它、不级联);子是 running 孤儿。
    from praxis.core.statemachine import transition

    transition(store, bus, turns[0], TurnStatus.PAUSED, reason="approval_required")
    marked = sweep(store, bus, workspace)
    assert marked == [turns[1]]
    assert not any(item.kind == ItemKind.CANCEL_REQUEST for item in store.read_items(threads[1]))
    node = build_tree(
        store, thread_id=threads[0], turn_id=turns[0], root_limits=Limits(tokens=1000), unit_costs=None
    )
    assert node.children["alloc-0"].state == "active"
    assert node.reserved.tokens == 500
    parent = open_session(
        workspace, ScriptedAdapter([_resp("parent resumed")]), thread_id=threads[0], db_path=store.db_path
    )

    async def scenario() -> None:
        try:
            result = await parent.resume(turns[0], extend={"tokens": 1})
            assert result.status is TurnStatus.COMPLETED
            child_turn = get_turn(store, turns[1])
            assert child_turn is not None and child_turn.status == "interrupted"
        finally:
            await parent.aclose()

    asyncio.run(scenario())


# ---------------------------------------------------------------------------
# ⑤ 孙代逐跳收敛 + depth 门(max_depth=2 第三层被拒)
# ---------------------------------------------------------------------------


def test_cascade_reaches_only_direct_children_grandchildren_hop_by_hop(
    workspace: Path, store: Store
) -> None:
    threads, turns = _manual_family(store, workspace, layers=3)
    # 直接级联:只到直接子;孙代此时零取消事实。
    assert cascade_for(store, threads[0], turns[0]) == 1
    assert not any(item.kind == ItemKind.CANCEL_REQUEST for item in store.read_items(threads[2]))
    # 幂等:已有未处理 cancel item → 跳过。
    assert cascade_for(store, threads[0], turns[0]) == 0
    # 逐跳:子自己这一跳的级联把取消传给孙代。
    assert cascade_for(store, threads[1], turns[1]) == 1
    grand_cancels = [
        item for item in store.read_items(threads[2]) if item.kind == ItemKind.CANCEL_REQUEST
    ]
    assert grand_cancels[0].payload["parent_turn_id"] == turns[1]
    assert grand_cancels[0].payload["source"] == "parent"


def test_sweep_three_layer_family_converges_hop_by_hop(workspace: Path, store: Store) -> None:
    bus = EventBus()
    threads, turns = _manual_family(store, workspace, layers=3)
    marked = sweep(store, bus, workspace)
    assert set(marked) == set(turns)
    # sweep 对每个被标 interrupted 的父都调 cascade_for → 孙代经子这一跳收到取消。
    assert any(
        item.kind == ItemKind.CANCEL_REQUEST
        and item.payload["parent_turn_id"] == turns[1]
        for item in store.read_items(threads[2])
    )
    child = open_session(workspace, ScriptedAdapter([]), thread_id=threads[1], db_path=store.db_path)
    grandchild = open_session(workspace, ScriptedAdapter([]), thread_id=threads[2], db_path=store.db_path)

    async def scenario() -> None:
        try:
            child_result = await child.resume(turns[1], confirm=True)
            assert child_result.status is TurnStatus.CANCELLED
            # 子终态挂钩再级联:孙代已有未处理 cancel → 幂等跳过,不重复写。
            grand_cancels = [
                item
                for item in store.read_items(threads[2])
                if item.kind == ItemKind.CANCEL_REQUEST
            ]
            assert len(grand_cancels) == 1
            grand_result = await grandchild.resume(turns[2], confirm=True)
            assert grand_result.status is TurnStatus.CANCELLED
        finally:
            await child.aclose()
            await grandchild.aclose()

    asyncio.run(scenario())


def test_spawn_depth_gate_rejects_third_layer(workspace: Path) -> None:
    holder: list[SubagentManager] = []
    results: list[Any] = []

    async def scenario() -> None:
        tools = [_spawn_probe(holder, results, depth=2)]
        adapter = ScriptedAdapter(
            [_resp("", (("s1", "spawn_probe", {}),)), _resp("parent done")]
        )
        parent = open_session(workspace, adapter, tools=tools, thread_id="parent")
        manager = SubagentManager(
            parent=parent, store=parent.store, limits_cfg={"max_depth": 2}
        )
        holder.append(manager)
        try:
            await parent.submit("go")
            # depth=2 的父再生子 = 第三层子代理 → 拒(max_depth=2)。
            assert isinstance(results[0], ChildLimitError)
            assert "depth" in str(results[0])
            assert not any(
                item.kind == ItemKind.CHILD_ALLOCATION for item in parent.store.read_items("parent")
            )
        finally:
            await manager.aclose()
            await parent.aclose()

    asyncio.run(scenario())


def test_spawn_depth_two_allowed_and_binding_carries_depth(workspace: Path) -> None:
    holder: list[SubagentManager] = []
    results: list[Any] = []

    async def scenario() -> None:
        tools = [_spawn_probe(holder, results, depth=1)]
        adapter = ScriptedAdapter(
            [
                _resp("", (("s1", "spawn_probe", {}),)),
                _resp("child done"),
                _resp("parent done"),
            ]
        )
        parent = open_session(workspace, adapter, tools=tools, thread_id="parent")
        manager = SubagentManager(parent=parent, store=parent.store)
        holder.append(manager)
        try:
            await parent.submit("go")
            handle: ChildHandle = results[0]
            child_turn = get_turn(parent.store, handle.child_root_turn_id)
            assert child_turn is not None
            assert child_turn.binding["depth"] == 2
        finally:
            await manager.aclose()
            await parent.aclose()

    asyncio.run(scenario())


# ---------------------------------------------------------------------------
# ⑥ 并发/总数 guard:超限明确拒绝
# ---------------------------------------------------------------------------


def test_concurrent_spawn_over_limit_rejected(workspace: Path) -> None:
    child_release = asyncio.Event()
    holder: list[SubagentManager] = []
    results: list[Any] = []

    async def scenario() -> None:
        tools = [
            _spawn_probe(holder, results, wait_child=False, second_spawn=True),
            _hang_tool("child_hang", child_release),
        ]
        adapter = ScriptedAdapter(
            [
                _resp("", (("s1", "spawn_probe", {}),)),
                _resp("", (("h1", "child_hang", {}),)),
                _resp("parent done"),
                _resp("child done"),
            ]
        )
        parent = open_session(workspace, adapter, tools=tools, thread_id="parent")
        manager = SubagentManager(
            parent=parent, store=parent.store, limits_cfg={"max_concurrent": 1}
        )
        holder.append(manager)
        try:
            result = await parent.submit("go")
            assert result.status is TurnStatus.COMPLETED
            assert isinstance(results[0], ChildHandle)
            assert isinstance(results[1], ChildLimitError)
            assert "concurrent" in str(results[1])
            allocations = [
                item for item in parent.store.read_items("parent") if item.kind == ItemKind.CHILD_ALLOCATION
            ]
            assert len(allocations) == 1
            child_release.set()
            status = await manager.wait(results[0].child_thread_id, timeout=5.0)
            assert status.state == "done"
        finally:
            child_release.set()
            await manager.aclose()
            await parent.aclose()

    asyncio.run(scenario())


def test_total_spawn_over_limit_rejected_even_after_child_done(workspace: Path) -> None:
    holder: list[SubagentManager] = []
    results: list[Any] = []

    async def scenario() -> None:
        tools = [_spawn_probe(holder, results, second_spawn=True)]
        adapter = ScriptedAdapter(
            [
                _resp("", (("s1", "spawn_probe", {}),)),
                _resp("child done"),
                _resp("parent done"),
            ]
        )
        parent = open_session(workspace, adapter, tools=tools, thread_id="parent")
        manager = SubagentManager(
            parent=parent, store=parent.store, limits_cfg={"max_total": 1}
        )
        holder.append(manager)
        try:
            await parent.submit("go")
            assert isinstance(results[0], ChildHandle)
            # 探针序:handle → 二次 spawn 结局 → 第一个子的 wait 状态。
            assert isinstance(results[1], ChildLimitError)
            assert "total" in str(results[1])
            assert isinstance(results[2], ChildStatus)
            assert results[2].state == "done"
        finally:
            await manager.aclose()
            await parent.aclose()

    asyncio.run(scenario())


# ---------------------------------------------------------------------------
# ⑦ 同事务(R-PF-6 异常注入):transact 内抛错 → 回滚 → 零孤儿
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("fail_point", ["create_turn_tx", "append_items_tx"])
def test_spawn_transaction_failure_leaves_no_orphan(
    workspace: Path, monkeypatch: pytest.MonkeyPatch, fail_point: str
) -> None:
    holder: list[SubagentManager] = []
    results: list[Any] = []

    real = getattr(agents_module, fail_point)

    def boom(*args: Any, **kwargs: Any) -> Any:
        real(*args, **kwargs)
        raise RuntimeError("injected kill inside transaction")

    monkeypatch.setattr(agents_module, fail_point, boom)

    async def scenario() -> None:
        tools = [_spawn_probe(holder, results)]
        adapter = ScriptedAdapter(
            [_resp("", (("s1", "spawn_probe", {}),)), _resp("不可达")]
        )
        parent = open_session(workspace, adapter, tools=tools, thread_id="parent")
        manager = SubagentManager(parent=parent, store=parent.store)
        holder.append(manager)
        try:
            result = await parent.submit("go")
            # 注入的 RuntimeError 被探针工具捕获为 error 结果,父 turn 正常收尾。
            assert result.status is TurnStatus.COMPLETED
            assert isinstance(results[0], RuntimeError)
            assert "injected kill" in str(results[0])
            # 零残留:无 allocation item、无子根 turn(回滚整组)。
            assert not any(
                item.kind == ItemKind.CHILD_ALLOCATION for item in parent.store.read_items("parent")
            )
            assert len(list_turns(parent.store)) == 1
        finally:
            await manager.aclose()
            await parent.aclose()

    asyncio.run(scenario())


def test_spawn_success_writes_allocation_and_child_turn_atomically(workspace: Path) -> None:
    """成功路径的同事务证据:allocation 与子根 turn 同时存在且身份互指。"""
    holder: list[SubagentManager] = []
    results: list[Any] = []

    async def scenario() -> None:
        tools = [_spawn_probe(holder, results)]
        adapter = ScriptedAdapter(
            [
                _resp("", (("s1", "spawn_probe", {}),)),
                _resp("child done"),
                _resp("parent done"),
            ]
        )
        parent = open_session(workspace, adapter, tools=tools, thread_id="parent")
        manager = SubagentManager(parent=parent, store=parent.store)
        holder.append(manager)
        try:
            result = await parent.submit("go")
            handle: ChildHandle = results[0]
            allocations = [
                item for item in parent.store.read_items("parent") if item.kind == ItemKind.CHILD_ALLOCATION
            ]
            assert len(allocations) == 1
            payload = allocations[0].payload
            assert payload["child_root_turn_id"] == handle.child_root_turn_id
            assert payload["child_thread_id"] == handle.child_thread_id
            assert payload["allocation_id"] == handle.allocation_id
            child_turn = get_turn(parent.store, handle.child_root_turn_id)
            assert child_turn is not None
            assert child_turn.binding["parent_turn_id"] == result.turn_id
        finally:
            await manager.aclose()
            await parent.aclose()

    asyncio.run(scenario())


# ---------------------------------------------------------------------------
# wait/send/request_cancel(R-T5-4:wait 过期不取消)
# ---------------------------------------------------------------------------


def test_wait_timeout_does_not_cancel_driver_and_clamps(workspace: Path) -> None:
    child_release = asyncio.Event()
    holder: list[SubagentManager] = []
    results: list[Any] = []

    async def scenario() -> None:
        tools = [
            _spawn_probe(holder, results, wait_child=False),
            _hang_tool("child_hang", child_release),
        ]
        adapter = ScriptedAdapter(
            [
                _resp("", (("s1", "spawn_probe", {}),)),
                _resp("", (("h1", "child_hang", {}),)),
                _resp("parent done"),
                _resp("child done"),
            ]
        )
        parent = open_session(workspace, adapter, tools=tools, thread_id="parent")
        manager = SubagentManager(parent=parent, store=parent.store)
        holder.append(manager)
        try:
            await parent.submit("go")
            handle: ChildHandle = results[0]
            # 过期不取消:短超时返回 running,driver 仍活着。
            status = await manager.wait(handle.child_thread_id, timeout=0.05)
            assert status.state == "running"
            child_turn = get_turn(parent.store, handle.child_root_turn_id)
            assert child_turn is not None and child_turn.status == "running"
            # 钳位:负值 → 0(立即返回,不拒绝)。
            assert (await manager.wait(handle.child_thread_id, timeout=-5)).state == "running"
            # 钳位:超大 → 60(不拒绝);子已终结时立即返回 done。
            child_release.set()
            assert (await manager.wait(handle.child_thread_id, timeout=999)).state == "done"
        finally:
            child_release.set()
            await manager.aclose()
            await parent.aclose()

    asyncio.run(scenario())


def test_send_appends_steering_to_child_thread(workspace: Path) -> None:
    child_release = asyncio.Event()
    holder: list[SubagentManager] = []
    results: list[Any] = []

    async def scenario() -> None:
        tools = [
            _spawn_probe(holder, results, wait_child=False),
            _hang_tool("child_hang", child_release),
        ]
        adapter = ScriptedAdapter(
            [
                _resp("", (("s1", "spawn_probe", {}),)),
                _resp("", (("h1", "child_hang", {}),)),
                _resp("parent done"),
                _resp("child done"),
            ]
        )
        parent = open_session(workspace, adapter, tools=tools, thread_id="parent")
        manager = SubagentManager(parent=parent, store=parent.store)
        holder.append(manager)
        try:
            await parent.submit("go")
            handle: ChildHandle = results[0]
            await manager.send(handle.child_thread_id, "focus on tests")
            steering = [
                item for item in parent.store.read_items(handle.child_thread_id) if item.kind == ItemKind.STEERING
            ]
            assert len(steering) == 1
            assert steering[0].payload["text"] == "focus on tests"
            assert steering[0].turn_id is None  # thread 级,与 Session.steer 同形
            child_release.set()
            assert (await manager.wait(handle.child_thread_id, timeout=5.0)).state == "done"
            with pytest.raises(ValueError):
                await manager.send("no-such-child", "hello")
        finally:
            child_release.set()
            await manager.aclose()
            await parent.aclose()

    asyncio.run(scenario())


def test_request_cancel_child_signalled_and_terminal_rejected(workspace: Path) -> None:
    child_release = asyncio.Event()
    holder: list[SubagentManager] = []
    results: list[Any] = []

    async def scenario() -> None:
        tools = [
            _spawn_probe(holder, results, wait_child=False),
            _hang_tool("child_hang", child_release),
        ]
        adapter = ScriptedAdapter(
            [
                _resp("", (("s1", "spawn_probe", {}),)),
                _resp("", (("h1", "child_hang", {}),)),
                _resp("parent done"),
                _resp("不可达"),
            ]
        )
        parent = open_session(workspace, adapter, tools=tools, thread_id="parent")
        manager = SubagentManager(parent=parent, store=parent.store)
        holder.append(manager)
        try:
            await parent.submit("go")
            handle: ChildHandle = results[0]
            assert manager.request_cancel(handle.child_thread_id) == "signalled"
            status = await manager.wait(handle.child_thread_id, timeout=5.0)
            assert status.state == "cancelled"
            cancels = [
                item
                for item in parent.store.read_items(handle.child_thread_id)
                if item.kind == ItemKind.CANCEL_REQUEST
            ]
            assert cancels[0].payload["source"] == "user"
            assert cancels[0].payload["parent_turn_id"] == handle_parent_turn(handle, parent)
            # 终态再取消 → ValueError(与 Session.request_cancel 同守卫)。
            with pytest.raises(ValueError):
                manager.request_cancel(handle.child_thread_id)
            with pytest.raises(ValueError):
                manager.request_cancel("no-such-child")
        finally:
            child_release.set()
            await manager.aclose()
            await parent.aclose()

    def handle_parent_turn(handle: ChildHandle, parent: Any) -> str:
        allocations = [
            item for item in parent.store.read_items("parent") if item.kind == ItemKind.CHILD_ALLOCATION
        ]
        return str(allocations[0].payload["parent_turn_id"])

    asyncio.run(scenario())


# ---------------------------------------------------------------------------
# read_only 权限收窄 + 结构深度门(子工具表减 agent 五件)
# ---------------------------------------------------------------------------


def test_read_only_child_denies_write_tool(workspace: Path) -> None:
    holder: list[SubagentManager] = []
    results: list[Any] = []
    executed: list[str] = []

    async def write_run(arguments: Mapping[str, JsonValue], exec_ctx: ToolExecContext) -> ToolResult:
        executed.append("write")
        return _ok("write_probe", "written")

    async def scenario() -> None:
        write_tool = _tool(
            "write_probe",
            write_run,
            effects=frozenset({ToolEffect.WRITE_WORKSPACE}),
            targets=(ToolTarget(kind="path", value=str(workspace / "f.txt")),),
        )
        tools = [_spawn_probe(holder, results), write_tool]
        adapter = ScriptedAdapter(
            [
                _resp("", (("s1", "spawn_probe", {}),)),
                _resp("", (("w1", "write_probe", {}),)),
                _resp("child done"),
                _resp("parent done"),
            ]
        )
        parent = open_session(workspace, adapter, tools=tools, thread_id="parent")
        manager = SubagentManager(parent=parent, store=parent.store)
        holder.append(manager)
        try:
            await parent.submit("go")
            handle: ChildHandle = results[0]
            child_results = [
                item
                for item in parent.store.read_turn_items(handle.child_root_turn_id)
                if item.kind == ItemKind.TOOL_RESULT
            ]
            assert len(child_results) == 1
            assert child_results[0].payload["status"] == "denied"
            assert executed == []  # 拒绝发生在 dispatch 之前
            status: ChildStatus = results[1]
            assert status.state == "done"
        finally:
            await manager.aclose()
            await parent.aclose()

    asyncio.run(scenario())


def test_read_only_child_ceiling_vetoes_write_path_but_not_read(workspace: Path) -> None:
    """R-T5-6 承重层(全平台可跑):read_only ceiling 对写路径的 path_access
    否决——compile_sandbox 的 allow_write 逐 ceiling AND 此项;读不受影响。

    空工具表也恒追加 ceiling:bash 无静态 WRITE 声明,收窄不能依赖工具表组成。
    """

    async def scenario() -> None:
        parent = open_session(workspace, ScriptedAdapter([]), tools=[], thread_id="parent")
        child = build_child_session(
            parent=parent,
            workspace=workspace,
            thread_id=uuid4().hex,
            limits=UNLIMITED,
            tool_subset=(),
        )
        try:
            readonly = [c for c in child.permissions.ceilings if c.profile == "read-only"]
            assert len(readonly) == 1
            resolved = resolve_policy(readonly[0], workspace)
            # 写否决:工作区根本体与子路径都拒(allow_write 的 AND 项 = False)。
            assert path_access(resolved, workspace, write=True) is False
            assert path_access(resolved, workspace / "f.txt", write=True) is False
            # 读保留:收窄只否决写,不伤读。
            assert path_access(resolved, workspace, write=False) is True
            assert path_access(resolved, workspace / "f.txt", write=False) is True
            # 父策略本身不受影响(ceiling 只约束子)。
            assert path_access(resolve_policy(parent.permissions.policy, workspace), workspace, write=True) is True
        finally:
            await child.aclose()
            await parent.aclose()

    asyncio.run(scenario())


@pytest.mark.skipif(not _SANDBOX.available, reason="no sandbox backend (bwrap deferred)")
def test_read_only_child_bash_write_denied_by_sandbox(workspace: Path) -> None:
    """R-T5-6 实证:read_only 子的 bash `echo x > f.txt` 被沙箱拒,读仍允许。

    编译层断言(离线):profile 对工作区子路径渲染 deny file-write*、无
    allow file-write*;强制层(真沙箱后端):写命令非零退出且文件不存在,
    cat 既有文件成功。child.permissions 经公开工厂构造——与生产 driver
    注入管道的 compile_sandbox 是同一上下文同一编译入口。
    """
    (workspace / "readable.txt").write_text("orig", encoding="utf-8")

    async def scenario() -> None:
        parent = open_session(workspace, ScriptedAdapter([]), tools=[], thread_id="parent")
        child = build_child_session(
            parent=parent,
            workspace=workspace,
            thread_id=uuid4().hex,
            limits=UNLIMITED,
            tool_subset=(),
        )
        try:
            use = ResolvedUse(
                effects=frozenset({ToolEffect.EXECUTE_PROCESS}),
                targets=(ToolTarget(kind="path", value=str(workspace)),),
            )
            spec = compile_sandbox(child.permissions, use, workspace)
            try:
                text = spec.profile_text or ""
                realpath = os.path.realpath(workspace)
                assert f'(allow file-write* (require-all (subpath "{realpath}")' not in text
                assert f'(deny file-write* (require-all (subpath "{realpath}")' in text
                write = subprocess.run(
                    [*spec.argv_prefix, "/bin/bash", "-lc", "echo hacked > f.txt"],
                    cwd=workspace,
                    capture_output=True,
                    text=True,
                    timeout=30,
                )
                assert write.returncode != 0
                assert not (workspace / "f.txt").exists()
                read = subprocess.run(
                    [*spec.argv_prefix, "/bin/bash", "-lc", "cat readable.txt"],
                    cwd=workspace,
                    capture_output=True,
                    text=True,
                    timeout=30,
                )
                assert read.returncode == 0, read.stderr
                assert "orig" in read.stdout
            finally:
                finalize_sandbox(spec, workspace)
        finally:
            await child.aclose()
            await parent.aclose()

    asyncio.run(scenario())


def test_child_tool_table_excludes_agent_five(workspace: Path) -> None:
    holder: list[SubagentManager] = []
    results: list[Any] = []

    async def scenario() -> None:
        # 父表含一个 agent 同名玩具;子表必须结构性减去它(depth 结构门)。
        agent_toy = _tool("spawn_subagent", lambda a, c: _ok("spawn_subagent", "no"))
        tools = [_spawn_probe(holder, results), agent_toy]
        adapter = ScriptedAdapter(
            [
                _resp("", (("s1", "spawn_probe", {}),)),
                _resp("", (("a1", "spawn_subagent", {}),)),
                _resp("child done"),
                _resp("parent done"),
            ]
        )
        parent = open_session(workspace, adapter, tools=tools, thread_id="parent")
        manager = SubagentManager(parent=parent, store=parent.store)
        holder.append(manager)
        try:
            await parent.submit("go")
            handle: ChildHandle = results[0]
            child_results = [
                item
                for item in parent.store.read_turn_items(handle.child_root_turn_id)
                if item.kind == ItemKind.TOOL_RESULT
            ]
            assert child_results[0].payload["error_code"] == "unknown_tool"
            assert "spawn_subagent" in AGENT_TOOL_NAMES
        finally:
            await manager.aclose()
            await parent.aclose()

    asyncio.run(scenario())


# ---------------------------------------------------------------------------
# workspace_mode:write 无 provisioner 明确拒绝;桩 provisioner 同库独立目录
# ---------------------------------------------------------------------------


def test_write_mode_without_provisioner_rejected(workspace: Path) -> None:
    holder: list[SubagentManager] = []
    results: list[Any] = []

    async def scenario() -> None:
        tools = [_spawn_probe(holder, results, workspace_mode="write")]
        adapter = ScriptedAdapter(
            [_resp("", (("s1", "spawn_probe", {}),)), _resp("parent done")]
        )
        parent = open_session(workspace, adapter, tools=tools, thread_id="parent")
        manager = SubagentManager(parent=parent, store=parent.store)
        holder.append(manager)
        try:
            await parent.submit("go")
            assert isinstance(results[0], ValueError)
            assert "provisioner" in str(results[0])
            assert not any(
                item.kind == ItemKind.CHILD_ALLOCATION for item in parent.store.read_items("parent")
            )
        finally:
            await manager.aclose()
            await parent.aclose()

    asyncio.run(scenario())


def test_write_mode_stub_provisioner_child_stays_in_parent_db(
    workspace: Path, tmp_path: Path
) -> None:
    holder: list[SubagentManager] = []
    results: list[Any] = []
    child_ws = tmp_path / "child-workspace"
    provisioner = _StubProvisioner(child_ws)

    async def scenario() -> None:
        tools = [_spawn_probe(holder, results, workspace_mode="write")]
        adapter = ScriptedAdapter(
            [
                _resp("", (("s1", "spawn_probe", {}),)),
                _resp("child done"),
                _resp("parent done"),
            ]
        )
        parent = open_session(workspace, adapter, tools=tools, thread_id="parent")
        manager = SubagentManager(parent=parent, store=parent.store, provisioner=provisioner)
        holder.append(manager)
        try:
            await parent.submit("go")
            handle: ChildHandle = results[0]
            assert handle.workspace == child_ws
            assert provisioner.prepared == [(workspace, handle.child_thread_id, "write")]
            allocations = [
                item for item in parent.store.read_items("parent") if item.kind == ItemKind.CHILD_ALLOCATION
            ]
            assert allocations[0].payload["workspace"] == str(child_ws)
            assert allocations[0].payload["workspace_mode"] == "write"
            # 同库:子 items 经父 store 可读(独立目录 ≠ 独立库)。
            assert any(
                item.kind == ItemKind.MODEL_RESPONSE
                for item in parent.store.read_items(handle.child_thread_id)
            )
            # 执行锁按子 thread 在子工作区独立持有(锁文件族在子 ws)。
            assert (child_ws / ".praxis" / "locks" / f"{handle.child_thread_id}.lock").exists()
            status: ChildStatus = results[1]
            assert status.state == "done"
            assert status.workspace == str(child_ws)
        finally:
            await manager.aclose()
            await parent.aclose()

    asyncio.run(scenario())


def test_provisioner_prepare_failure_writes_no_allocation(workspace: Path) -> None:
    holder: list[SubagentManager] = []
    results: list[Any] = []

    class _FailingProvisioner(_StubProvisioner):
        def prepare(self, parent_root: Path, child_thread_id: str, mode: str) -> Path:
            raise ChildWorkspaceRequiresGit("not a git repository")

    async def scenario() -> None:
        tools = [_spawn_probe(holder, results, workspace_mode="write")]
        adapter = ScriptedAdapter(
            [_resp("", (("s1", "spawn_probe", {}),)), _resp("parent done")]
        )
        parent = open_session(workspace, adapter, tools=tools, thread_id="parent")
        manager = SubagentManager(
            parent=parent,
            store=parent.store,
            provisioner=_FailingProvisioner(workspace / "unused"),
        )
        holder.append(manager)
        try:
            result = await parent.submit("go")
            assert result.status is TurnStatus.COMPLETED
            assert isinstance(results[0], ChildWorkspaceRequiresGit)
            assert not any(
                item.kind == ItemKind.CHILD_ALLOCATION for item in parent.store.read_items("parent")
            )
        finally:
            await manager.aclose()
            await parent.aclose()

    asyncio.run(scenario())


# ---------------------------------------------------------------------------
# cascade_for 单元性质 + 零既有破坏
# ---------------------------------------------------------------------------


def test_cascade_skips_terminal_and_missing_children(workspace: Path, store: Store) -> None:
    bus = EventBus()
    threads, turns = _manual_family(store, workspace, layers=2)
    from praxis.core.statemachine import transition

    # 子已终态 → 跳过(幂等/终态不级联)。
    transition(store, bus, turns[1], TurnStatus.CANCELLED, reason="cancelled_by_user")
    assert cascade_for(store, threads[0], turns[0]) == 0
    assert not any(item.kind == ItemKind.CANCEL_REQUEST for item in store.read_items(threads[1]))
    # allocation 指向不存在的子 turn → fail-open 跳过,不抛。
    store.append_items(
        threads[0],
        turns[0],
        [
            ItemEntry(
                kind=ItemKind.CHILD_ALLOCATION,
                payload=_allocation_payload(
                    allocation_id="alloc-ghost",
                    parent_turn_id=turns[0],
                    child_thread_id="ghost-thread",
                    child_root_turn_id="ghost-turn",
                    grant=Limits(tokens=10),
                ),
            )
        ],
    )
    assert cascade_for(store, threads[0], turns[0]) == 0


def test_turn_result_children_summary_default_empty(workspace: Path) -> None:
    async def scenario() -> None:
        parent = open_session(
            workspace, ScriptedAdapter([_resp("plain")]), tools=[], thread_id="plain"
        )
        try:
            result = await parent.submit("go")
            assert result.status is TurnStatus.COMPLETED
            assert result.children_summary == ()
        finally:
            await parent.aclose()

    asyncio.run(scenario())


def test_manager_limits_cfg_validation(workspace: Path) -> None:
    async def scenario() -> None:
        parent = open_session(
            workspace, ScriptedAdapter([]), tools=[], thread_id="cfg"
        )
        try:
            with pytest.raises(ValueError):
                SubagentManager(parent=parent, store=parent.store, limits_cfg={"bogus": 1})
            with pytest.raises(ValueError):
                SubagentManager(parent=parent, store=parent.store, limits_cfg={"max_depth": -1})
            with pytest.raises(ValueError):
                SubagentManager(parent=parent, store=parent.store, limits_cfg={"max_depth": True})
        finally:
            await parent.aclose()

    asyncio.run(scenario())


def test_spawn_outside_executing_turn_rejected(workspace: Path) -> None:
    async def scenario() -> None:
        parent = open_session(
            workspace, ScriptedAdapter([]), tools=[], thread_id="idle"
        )
        manager = SubagentManager(parent=parent, store=parent.store)
        try:
            # 无执行中父 turn(active_turn_id=None)→ 明确拒绝,零写入。
            with pytest.raises(RuntimeError, match="executing parent turn"):
                await manager.spawn("orphan")
            assert not any(
                item.kind == ItemKind.CHILD_ALLOCATION for item in parent.store.read_items("idle")
            )
        finally:
            await manager.aclose()
            await parent.aclose()

    asyncio.run(scenario())


def test_context_summary_prepended_to_child_task(workspace: Path) -> None:
    holder: list[SubagentManager] = []
    results: list[Any] = []

    async def scenario() -> None:
        tools = [_spawn_probe(holder, results, context_summary="parent context facts")]
        adapter = ScriptedAdapter(
            [
                _resp("", (("s1", "spawn_probe", {}),)),
                _resp("child done"),
                _resp("parent done"),
            ]
        )
        parent = open_session(workspace, adapter, tools=tools, thread_id="parent")
        manager = SubagentManager(parent=parent, store=parent.store)
        holder.append(manager)
        try:
            await parent.submit("go")
            handle: ChildHandle = results[0]
            messages = [
                item
                for item in parent.store.read_items(handle.child_thread_id)
                if item.kind == ItemKind.USER_MESSAGE
            ]
            assert len(messages) == 1
            text = messages[0].payload["text"]
            assert isinstance(text, str)
            assert "parent context facts" in text
            assert "child task" in text
        finally:
            await manager.aclose()
            await parent.aclose()

    asyncio.run(scenario())


def test_paused_child_visible_via_wait_and_approved_directly(workspace: Path) -> None:
    """Q3 直面:子 PAUSED 待批 → wait 如实展示 paused + pending_approvals;
    人类对子 session 直接 resume(approval) 续跑——父不替子审批。"""
    holder: list[SubagentManager] = []
    results: list[Any] = []

    async def scenario() -> None:
        # NETWORK 无 url target 的玩具 → allow_rules 覆盖不成立 → fallback ask。
        ask_tool = _tool(
            "ask_probe",
            lambda a, c: _ok("ask_probe", "approved run"),
            effects=frozenset({ToolEffect.NETWORK}),
        )
        tools = [_spawn_probe(holder, results), ask_tool]
        adapter = ScriptedAdapter(
            [
                _resp("", (("s1", "spawn_probe", {}),)),
                _resp("", (("a1", "ask_probe", {}),)),
                _resp("parent done"),
                _resp("child done after approval"),
            ]
        )
        parent = open_session(workspace, adapter, tools=tools, thread_id="parent")
        manager = SubagentManager(parent=parent, store=parent.store)
        holder.append(manager)
        try:
            result = await parent.submit("go")
            assert result.status is TurnStatus.COMPLETED
            handle: ChildHandle = results[0]
            status: ChildStatus = results[1]
            assert status.state == "paused"
            assert len(status.pending_approvals) == 1
            assert status.pending_approvals[0].tool_name == "ask_probe"
            # 父结局注记如实:子 paused(可见但不代批)。
            assert result.children_summary[0].state == "paused"
            # 人类直面子 session:独立打开同库子 thread,直接批准续跑。
            # 共享同一剧本适配器:子的续作响应在剧本末位(消费序确定)。
            child = open_session(
                workspace,
                adapter,
                thread_id=handle.child_thread_id,
                db_path=parent.store.db_path,
            )
            try:
                call_id = status.pending_approvals[0].call_id
                resumed = await child.resume(
                    handle.child_root_turn_id, approval={call_id: "allow"}
                )
                assert resumed.status is TurnStatus.COMPLETED
                assert resumed.output == "child done after approval"
            finally:
                await child.aclose()
            assert (await manager.wait(handle.child_thread_id, timeout=5.0)).state == "done"
        finally:
            await manager.aclose()
            await parent.aclose()

    asyncio.run(scenario())
