"""阶段4 T6:子代理模型工具面五件套(Q4)+ T5 转记安全/生命周期收口。

覆盖面:
- schema 校验边界逐值(task/context_summary 20_000/20_001;wait timeout ≤60);
- 开关默认关=零扰动;enable_subagents=True → 15+5=20(口径同步见
  test_default_table3b);子工具表不含五件(结构深度门);
- spawn→wait 端到端(scripted 父子,真实管道);错误码面
  (grant_exhausted/child_limit_reached/child_not_found/child_still_active;
  integrate 实装语义见 test_integrate.py);
- session 域 ToolRule allow(never 模式仍放行四件编排;显式 deny 仍胜;
  allow 不落盘);integrate DESTRUCTIVE 显式审批不可豁免(S5);
- `ps` 无关:spawn→wait 全程零新进程(subprocess.Popen 探针);
- R-T5-7:guard 全家族口径——父跨 turn 逐 turn spawn 仍受全家族
  max_total/max_concurrent 约束;
- R-T5-8:read_only 子网络决策=父决策、外部读授权保留、写否决承重不变;
- ㉖:子 always 不污染父 permissions.json(read_only 子内存生效+披露;
  write 子落自己的隔离文件);
- ㉒:driver 收尾竞态——teardown 窗口内二次 cancel 不漏 Store 写者线程;
- ③:manager.aclose 挂 Session 收尾链——Session 关闭后子 driver 与
  Store 写者线程无泄漏。

仓库无 pytest-asyncio:与 test_agents_core.py 同风格,asyncio.run 包装。
"""

from __future__ import annotations

import asyncio
import json
import subprocess
import threading
from collections.abc import AsyncIterator, Callable, Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any
from uuid import uuid4

import pytest

from praxis.composition import build_child_session
from praxis.core.agents import SubagentManager
from praxis.core.permission import decide, network_verdict
from praxis.core.statemachine import TurnStatus
from praxis.interfaces.sdk import ScriptedAdapter, open_session
from praxis.protocol.agents import (
    AGENT_TOOL_NAMES,
    ChildActiveError,
    ChildGrantError,
    ChildHandle,
    ChildLimitError,
    ChildStatus,
    SubagentSurface,
)
from praxis.protocol.budget import UNLIMITED, Limits
from praxis.protocol.items import ItemKind
from praxis.protocol.json import JsonValue
from praxis.protocol.models import (
    ModelEvent,
    ModelMessage,
    ModelResponse,
    ModelSettings,
    ModelToolCall,
    ModelToolCallDelta,
    Usage,
    WireRequest,
    assemble_model_events,
    collect_events,
)
from praxis.protocol.permission import PermissionUpdate
from praxis.protocol.policy import (
    NetworkPolicy,
    PathRule,
    PermissionPolicy,
    ToolRule,
    path_access,
    resolve_policy,
    tool_decision,
)
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
    ToolValidationError,
)
from praxis.protocol.workspaces import ApplyResult, WorkspaceInspection, WorkspaceMode
from praxis.store.log import Store
from praxis.store.turns import get_turn
from praxis.tools.builtins.agent_tools import create_agent_tools

_ZERO_USAGE = Usage(input_tokens=0, output_tokens=0)


# ---------------------------------------------------------------------------
# 脚手架
# ---------------------------------------------------------------------------


def _resp(
    content: str,
    calls: tuple[tuple[str, str, dict[str, Any]], ...] = (),
) -> ModelResponse:
    return ModelResponse(
        content=content,
        tool_calls=tuple(ModelToolCall(call_id=c, name=n, arguments=a) for c, n, a in calls),
        usage=_ZERO_USAGE,
        finish_reason="tool_calls" if calls else "stop",
    )


def _ok(name: str, text: str) -> ToolResult:
    return ToolResult(
        tool_call_id="pending",
        tool_name=name,
        content=(ToolContentBlock(type="text", data={"text": text}),),
        status=ResultStatus.OK,
    )


def _toy_tool(name: str, run: Any) -> Tool:
    return Tool(
        definition=ToolDefinition(name=name, description=f"{name} toy", input_schema={"type": "object"}),
        validate_input=lambda arguments: arguments,
        run=run,
        static_effects=frozenset(),
        resolve_use=lambda arguments: ResolvedUse(frozenset(), ()),
        idempotent=True,
        concurrency_safe=True,
        cancellation_mode=CancellationMode.COOPERATIVE,
        interrupt_behavior=InterruptBehavior.CANCEL,
        timeout_seconds=30,
        max_model_output_bytes=64_000,
    )


def _hang_tool(name: str, event: asyncio.Event) -> Tool:
    async def run(arguments: Mapping[str, JsonValue], exec_ctx: ToolExecContext) -> ToolResult:
        await event.wait()
        return _ok(name, f"{name}:released")

    return _toy_tool(name, run)


class _StubSurface:
    """SubagentSurface 测试替身:记录调用,可注入异常/ canned 结果。"""

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.exc: Exception | None = None
        self.handle = ChildHandle(
            child_thread_id="child-stub",
            child_root_turn_id="root-stub",
            allocation_id="alloc-stub",
            workspace=Path("/tmp/stub"),
            workspace_mode="read_only",
            grant=Limits(tokens=100),
        )
        self.status = ChildStatus(
            state="done",
            conclusion="stub conclusion",
            pending_approvals=(),
            unknown_calls=(),
            steps_used=1,
            cost_used=0,
            workspace="/tmp/stub",
        )

    async def spawn(
        self,
        task: str,
        *,
        context_summary: str = "",
        requested: Limits | None = None,
        workspace_mode: WorkspaceMode = "read_only",
        depth: int = 0,
    ) -> ChildHandle:
        self.calls.append(("spawn", {"task": task, "context_summary": context_summary,
                                     "requested": requested, "workspace_mode": workspace_mode}))
        if self.exc is not None:
            raise self.exc
        return self.handle

    async def wait(self, child_thread_id: str, *, timeout: float = 10.0) -> ChildStatus:
        self.calls.append(("wait", {"child_thread_id": child_thread_id, "timeout": timeout}))
        if self.exc is not None:
            raise self.exc
        return self.status

    async def send(self, child_thread_id: str, message: str) -> None:
        self.calls.append(("send", {"child_thread_id": child_thread_id, "message": message}))
        if self.exc is not None:
            raise self.exc

    def request_cancel(self, child_thread_id: str) -> str:
        self.calls.append(("request_cancel", {"child_thread_id": child_thread_id}))
        if self.exc is not None:
            raise self.exc
        return "signalled"

    def integrate_inspect(
        self, child_thread_id: str, *, paths: Sequence[str] | None = None
    ) -> WorkspaceInspection:
        self.calls.append(("integrate_inspect", {"child_thread_id": child_thread_id, "paths": paths}))
        if self.exc is not None:
            raise self.exc
        return WorkspaceInspection(
            paths=(), preimages={}, postimages={}, diff_preview="", diff_truncated=False, diff_bytes=0
        )

    def integrate_apply(
        self, child_thread_id: str, *, paths: Sequence[str] | None = None
    ) -> ApplyResult:
        self.calls.append(("integrate_apply", {"child_thread_id": child_thread_id, "paths": paths}))
        if self.exc is not None:
            raise self.exc
        return ApplyResult(applied=[], conflicts=[])

    def integrate_discard(self, child_thread_id: str) -> None:
        self.calls.append(("integrate_discard", {"child_thread_id": child_thread_id}))
        if self.exc is not None:
            raise self.exc


def _tools_by_name(surface: SubagentSurface | None = None) -> dict[str, Tool]:
    stub = surface if surface is not None else _StubSurface()
    return {tool.definition.name: tool for tool in create_agent_tools(lambda: stub)}


class _ChildIdRewritingAdapter:
    """ScriptedAdapter 代理:把 wait/send/cancel 剧本里的占位 child id
    重写为 store 中最近一条 child_allocation 的真实 child_thread_id——
    剧本是静态的,而 spawn 的 id 是运行期 uuid;其余帧原样透传。
    store 经 provider 延迟绑定(Session 在适配器之后构造;不另开 Store)。"""

    def __init__(
        self, inner: ScriptedAdapter, store_provider: Callable[[], Store], parent_thread: str
    ) -> None:
        self._inner = inner
        self._store_provider = store_provider
        self._parent_thread = parent_thread

    def _latest_child_id(self) -> str:
        child_id = ""
        for item in self._store_provider().read_items(self._parent_thread):
            if item.kind == ItemKind.CHILD_ALLOCATION:
                raw = item.payload.get("child_thread_id")
                if isinstance(raw, str):
                    child_id = raw
        return child_id

    def build_request(
        self,
        messages: Sequence[ModelMessage],
        tools: Sequence[ToolDefinition],
        settings: ModelSettings,
    ) -> WireRequest:
        return self._inner.build_request(messages, tools, settings)

    async def stream(self, req: WireRequest) -> AsyncIterator[ModelEvent]:
        async for event in self._inner.stream(req):
            if (
                isinstance(event, ModelToolCallDelta)
                and event.name_fragment in ("wait_subagent", "send_to_subagent", "cancel_subagent")
            ):
                args = json.loads(event.arguments_fragment)
                args["child_thread_id"] = self._latest_child_id()
                event = replace(event, arguments_fragment=json.dumps(args))
            yield event

    async def complete(self, req: WireRequest) -> ModelResponse:
        return assemble_model_events(await collect_events(self.stream(req)))


def _result_items(store: Store, turn_id: str, tool_name: str) -> list[Mapping[str, JsonValue]]:
    return [
        item.payload
        for item in store.read_turn_items(turn_id)
        if item.kind == ItemKind.TOOL_RESULT and item.payload.get("tool_name") == tool_name
    ]


def _writer_threads() -> list[threading.Thread]:
    return [t for t in threading.enumerate() if t.name == "praxis-store-writer"]


# ---------------------------------------------------------------------------
# ① schema 校验边界逐值(20_000/20_001;timeout ≤60)
# ---------------------------------------------------------------------------


def test_spawn_schema_task_and_context_summary_bounds() -> None:
    spawn = _tools_by_name()["spawn_subagent"]
    ok = spawn.validate_input({"task": "x" * 20_000, "context_summary": "y" * 20_000})
    assert ok["task"] == "x" * 20_000
    with pytest.raises(ToolValidationError):
        spawn.validate_input({"task": "x" * 20_001})
    with pytest.raises(ToolValidationError):
        spawn.validate_input({"task": "t", "context_summary": "y" * 20_001})
    with pytest.raises(ToolValidationError):
        spawn.validate_input({"task": ""})
    with pytest.raises(ToolValidationError):
        spawn.validate_input({})
    with pytest.raises(ToolValidationError):
        spawn.validate_input({"task": "t", "bogus": 1})


def test_spawn_schema_budget_and_workspace_mode_fields() -> None:
    spawn = _tools_by_name()["spawn_subagent"]
    ok = spawn.validate_input(
        {"task": "t", "max_tokens": 100, "max_cost_micros": 5, "max_steps": 3, "workspace_mode": "write"}
    )
    assert ok["max_tokens"] == 100
    for bad in ({"max_tokens": 0}, {"max_tokens": -1}, {"max_steps": True}, {"max_cost_micros": 1.5}):
        with pytest.raises(ToolValidationError):
            spawn.validate_input({"task": "t", **bad})
    with pytest.raises(ToolValidationError):
        spawn.validate_input({"task": "t", "workspace_mode": "read-only"})


def test_wait_schema_timeout_bounds() -> None:
    wait = _tools_by_name()["wait_subagent"]
    assert wait.validate_input({"child_thread_id": "c", "timeout": 60})["timeout"] == 60
    assert wait.validate_input({"child_thread_id": "c", "timeout": 0})["timeout"] == 0
    for bad in (60.5, -1, "10"):
        with pytest.raises(ToolValidationError):
            wait.validate_input({"child_thread_id": "c", "timeout": bad})
    with pytest.raises(ToolValidationError):
        wait.validate_input({})


def test_send_cancel_integrate_schemas() -> None:
    tools = _tools_by_name()
    assert tools["send_to_subagent"].validate_input({"child_thread_id": "c", "message": "m"})
    with pytest.raises(ToolValidationError):
        tools["send_to_subagent"].validate_input({"child_thread_id": "c", "message": ""})
    assert tools["cancel_subagent"].validate_input({"child_thread_id": "c"})
    integrate = tools["integrate_child_workspace"]
    assert integrate.validate_input({"child_thread_id": "c", "action": "inspect"})
    assert integrate.validate_input({"child_thread_id": "c", "action": "apply", "paths": ["a.txt"]})
    with pytest.raises(ToolValidationError):
        integrate.validate_input({"child_thread_id": "c", "action": "merge"})
    with pytest.raises(ToolValidationError):
        integrate.validate_input({"child_thread_id": "c", "action": "apply", "paths": [1]})


def test_five_names_and_effects_declarations() -> None:
    tools = _tools_by_name()
    assert set(tools) == AGENT_TOOL_NAMES
    for name in ("spawn_subagent", "wait_subagent", "send_to_subagent", "cancel_subagent"):
        assert tools[name].static_effects == frozenset()
    integrate = tools["integrate_child_workspace"]
    assert integrate.static_effects == frozenset({ToolEffect.DESTRUCTIVE, ToolEffect.WRITE_WORKSPACE})
    # spawn 非幂等(两次 spawn 两个子);wait 纯观察幂等。
    assert tools["spawn_subagent"].idempotent is False
    assert tools["wait_subagent"].idempotent is True
    # wait 的工具超时必须覆盖 schema 允许的最长等待(60s)。
    assert tools["wait_subagent"].timeout_seconds > 60


# ---------------------------------------------------------------------------
# ② 开关默认关=零扰动;session 域 allow;S5 integrate 显式审批
# ---------------------------------------------------------------------------


def test_default_off_no_agent_tools_and_no_session_rules(workspace: Path) -> None:
    async def scenario() -> None:
        session = open_session(workspace, ScriptedAdapter([]), thread_id="off")
        try:
            assert not (set(session.tools) & AGENT_TOOL_NAMES)
            assert session.permissions.session_rules == ()
        finally:
            await session.aclose()

    asyncio.run(scenario())


def test_orchestration_tools_allowed_in_never_mode_but_explicit_deny_wins(workspace: Path) -> None:
    async def scenario() -> None:
        session = open_session(
            workspace, ScriptedAdapter([]), tools=[], thread_id="never",
            enable_subagents=True, mode="never",
        )
        try:
            spawn = session.tools["spawn_subagent"]
            decision = decide(spawn, spawn.resolve_use({"task": "t"}), session.permissions)
            assert decision.is_allow  # 预算份额+子限额才是闸门,审批不是
            integrate = session.tools["integrate_child_workspace"]
            use = integrate.resolve_use({"child_thread_id": "c", "action": "apply"})
            assert decide(integrate, use, session.permissions).is_deny  # S5+never
        finally:
            await session.aclose()
        denied = open_session(
            workspace, ScriptedAdapter([]), tools=[], thread_id="denied",
            enable_subagents=True,
            policy=PermissionPolicy(tools=(ToolRule(pattern="spawn_subagent", decision="deny"),)),
        )
        try:
            spawn = denied.tools["spawn_subagent"]
            decision = decide(spawn, spawn.resolve_use({"task": "t"}), denied.permissions)
            assert decision.is_deny  # 显式 deny 跨源最严,session allow 不能放宽
        finally:
            await denied.aclose()

    asyncio.run(scenario())


def test_session_domain_allow_never_persisted(workspace: Path) -> None:
    async def scenario() -> None:
        session = open_session(
            workspace, ScriptedAdapter([]), tools=[], thread_id="persist", enable_subagents=True
        )
        try:
            session.update_permission(
                PermissionUpdate(rule=ToolRule(pattern="read", decision="allow"), destination="always")
            )
            text = (workspace / ".praxis" / "permissions.json").read_text(encoding="utf-8")
            assert "spawn_subagent" not in text  # 自动 allow 是 session 域,绝不进文件
        finally:
            await session.aclose()

    asyncio.run(scenario())


# ---------------------------------------------------------------------------
# ③ spawn→wait 端到端(scripted 父子,真实管道)+ 子表结构深度门
# ---------------------------------------------------------------------------


def test_spawn_wait_end_to_end_via_pipeline(workspace: Path) -> None:
    async def scenario() -> None:
        inner = ScriptedAdapter(
            [
                _resp("", (("s1", "spawn_subagent", {"task": "child task", "context_summary": "ctx facts"}),)),
                _resp("child done"),
                _resp("", (("w1", "wait_subagent", {"child_thread_id": "PLACEHOLDER", "timeout": 5}),)),
                _resp("parent done"),
            ]
        )
        store_holder: list[Store] = []
        parent = open_session(
            workspace, _ChildIdRewritingAdapter(inner, lambda: store_holder[0], "parent"),
            thread_id="parent", enable_subagents=True, budget=Limits(tokens=1000),
        )
        store_holder.append(parent.store)
        try:
            result = await parent.submit("go")
            assert result.status is TurnStatus.COMPLETED
            spawn_results = _result_items(parent.store, result.turn_id, "spawn_subagent")
            assert len(spawn_results) == 1
            assert spawn_results[0]["status"] == "ok"
            child_id = spawn_results[0]["metadata"]["child_thread_id"]
            # T15(Q8):默认表含 web 工具 → 父 web 32/browser 64 逐维切半(16/32)。
            assert spawn_results[0]["metadata"]["grant"] == {
                "tokens": 500, "web_requests": 16, "browser_requests": 32,
            }
            wait_results = _result_items(parent.store, result.turn_id, "wait_subagent")
            assert len(wait_results) == 1
            assert wait_results[0]["status"] == "ok"
            assert wait_results[0]["metadata"]["state"] == "done"
            assert wait_results[0]["metadata"]["conclusion"] == "child done"
            # 子 items 写自己的 thread;context_summary 前置进子任务文本。
            child_items = parent.store.read_items(str(child_id))
            messages = [i for i in child_items if i.kind == ItemKind.USER_MESSAGE]
            assert "ctx facts" in str(messages[0].payload["text"])
            assert "child task" in str(messages[0].payload["text"])
            assert result.children_summary[0].state == "done"
        finally:
            await parent.aclose()

    asyncio.run(scenario())


def test_child_tool_table_excludes_agent_five_structural_gate(workspace: Path) -> None:
    async def scenario() -> None:
        inner = ScriptedAdapter(
            [
                _resp("", (("s1", "spawn_subagent", {"task": "child task"}),)),
                _resp("", (("a1", "spawn_subagent", {"task": "grandchild"}),)),
                _resp("child done"),
                _resp("", (("w1", "wait_subagent", {"child_thread_id": "PLACEHOLDER", "timeout": 5}),)),
                _resp("parent done"),
            ]
        )
        store_holder: list[Store] = []
        parent = open_session(
            workspace, _ChildIdRewritingAdapter(inner, lambda: store_holder[0], "parent"),
            thread_id="parent", enable_subagents=True,
        )
        store_holder.append(parent.store)
        try:
            result = await parent.submit("go")
            assert result.status is TurnStatus.COMPLETED
            child_id = str(
                _result_items(parent.store, result.turn_id, "spawn_subagent")[0]["metadata"]["child_thread_id"]
            )
            child_results = [
                item.payload
                for item in parent.store.read_items(child_id)
                if item.kind == ItemKind.TOOL_RESULT
            ]
            assert child_results[0]["error_code"] == "unknown_tool"
            assert "spawn_subagent" in str(child_results[0]["error_message"])
        finally:
            await parent.aclose()

    asyncio.run(scenario())


def test_spawn_wait_spawns_no_new_process(workspace: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """`ps` 无关:子是进程内 asyncio driver,不是新进程(Popen 探针)。"""

    def no_process(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("subagent spawn must not create a new process")

    monkeypatch.setattr(subprocess, "Popen", no_process)

    async def scenario() -> None:
        inner = ScriptedAdapter(
            [
                _resp("", (("s1", "spawn_subagent", {"task": "child task"}),)),
                _resp("child done"),
                _resp("", (("w1", "wait_subagent", {"child_thread_id": "PLACEHOLDER", "timeout": 5}),)),
                _resp("parent done"),
            ]
        )
        store_holder: list[Store] = []
        parent = open_session(
            workspace, _ChildIdRewritingAdapter(inner, lambda: store_holder[0], "parent"),
            thread_id="parent", enable_subagents=True,
        )
        store_holder.append(parent.store)
        try:
            result = await parent.submit("go")
            assert result.status is TurnStatus.COMPLETED
            assert _result_items(parent.store, result.turn_id, "wait_subagent")[0]["metadata"]["state"] == "done"
        finally:
            await parent.aclose()

    asyncio.run(scenario())


# ---------------------------------------------------------------------------
# ④ 错误码面(stub surface 直达 + 真实管道 grant/limit)
# ---------------------------------------------------------------------------


def test_error_code_mapping_on_stub_surface() -> None:
    async def scenario() -> None:
        stub = _StubSurface()
        tools = _tools_by_name(stub)
        ctx = ToolExecContext()
        stub.exc = ChildGrantError("parent budget exhausted in dimension 'tokens'")
        result = await tools["spawn_subagent"].run({"task": "t"}, ctx)
        assert result.error_code == "grant_exhausted"
        stub.exc = ChildLimitError("subagent total count limit exceeded (max_total=16)")
        result = await tools["spawn_subagent"].run({"task": "t"}, ctx)
        assert result.error_code == "child_limit_reached"
        stub.exc = ValueError("unknown child thread 'ghost' (no allocation in parent thread)")
        for name, args in (
            ("wait_subagent", {"child_thread_id": "ghost"}),
            ("send_to_subagent", {"child_thread_id": "ghost", "message": "m"}),
            ("cancel_subagent", {"child_thread_id": "ghost"}),
        ):
            result = await tools[name].run(args, ctx)
            assert result.error_code == "child_not_found", name
        stub.exc = ValueError("unknown child thread 'ghost' (no allocation in parent thread)")
        result = await tools["integrate_child_workspace"].run(
            {"child_thread_id": "ghost", "action": "inspect"}, ctx
        )
        assert result.error_code == "child_not_found"
        stub.exc = ChildActiveError("child 'c' is still active; integrate requires a terminal child")
        result = await tools["integrate_child_workspace"].run(
            {"child_thread_id": "c", "action": "apply"}, ctx
        )
        assert result.error_code == "child_still_active"
        assert result.status is ResultStatus.ERROR
        stub.exc = None
        result = await tools["integrate_child_workspace"].run(
            {"child_thread_id": "c", "action": "discard"}, ctx
        )
        assert result.status is ResultStatus.OK
        # 终态子的 cancel ValueError 不是 child_not_found:如实上抛交管道归一。
        stub.exc = ValueError("child turn 'r' is completed; terminal turns cannot be cancelled")
        with pytest.raises(ValueError, match="terminal"):
            await tools["cancel_subagent"].run({"child_thread_id": "c"}, ctx)

    asyncio.run(scenario())


def test_spawn_success_and_observability_results_on_stub_surface() -> None:
    async def scenario() -> None:
        stub = _StubSurface()
        tools = _tools_by_name(stub)
        ctx = ToolExecContext()
        result = await tools["spawn_subagent"].run(
            {"task": "t", "max_tokens": 100, "max_steps": 3, "workspace_mode": "read_only"}, ctx
        )
        assert result.status is ResultStatus.OK
        assert result.metadata["child_thread_id"] == "child-stub"
        assert stub.calls[0][1]["requested"] == Limits(tokens=100, steps=3)
        result = await tools["wait_subagent"].run({"child_thread_id": "child-stub", "timeout": 7}, ctx)
        assert result.metadata["state"] == "done"
        assert result.metadata["conclusion"] == "stub conclusion"
        assert stub.calls[1][1]["timeout"] == 7.0
        result = await tools["send_to_subagent"].run({"child_thread_id": "child-stub", "message": "focus"}, ctx)
        assert result.status is ResultStatus.OK
        result = await tools["cancel_subagent"].run({"child_thread_id": "child-stub"}, ctx)
        assert result.metadata["convergence"] == "signalled"

    asyncio.run(scenario())


def test_grant_exhausted_via_real_pipeline(workspace: Path) -> None:
    async def scenario() -> None:
        adapter = ScriptedAdapter(
            [_resp("", (("s1", "spawn_subagent", {"task": "t"}),)), _resp("parent done")]
        )
        parent = open_session(
            workspace, adapter, thread_id="parent", enable_subagents=True, budget=Limits(tokens=1)
        )
        try:
            result = await parent.submit("go")
            assert result.status is TurnStatus.COMPLETED
            payload = _result_items(parent.store, result.turn_id, "spawn_subagent")[0]
            assert payload["error_code"] == "grant_exhausted"
            assert not any(
                item.kind == ItemKind.CHILD_ALLOCATION for item in parent.store.read_items("parent")
            )
        finally:
            await parent.aclose()

    asyncio.run(scenario())


def test_child_limit_reached_via_real_pipeline(workspace: Path) -> None:
    async def scenario() -> None:
        adapter = ScriptedAdapter(
            [_resp("", (("s1", "spawn_subagent", {"task": "t"}),)), _resp("parent done")]
        )
        parent = open_session(
            workspace, adapter, thread_id="parent", enable_subagents=True,
            subagent_limits={"max_concurrent": 0},
        )
        try:
            result = await parent.submit("go")
            assert result.status is TurnStatus.COMPLETED
            payload = _result_items(parent.store, result.turn_id, "spawn_subagent")[0]
            assert payload["error_code"] == "child_limit_reached"
            assert "concurrent" in str(payload["error_message"])
        finally:
            await parent.aclose()

    asyncio.run(scenario())


def test_child_not_found_via_real_pipeline(workspace: Path) -> None:
    async def scenario() -> None:
        adapter = ScriptedAdapter(
            [
                _resp("", (("w1", "wait_subagent", {"child_thread_id": "ghost"}),)),
                _resp("parent done"),
            ]
        )
        parent = open_session(workspace, adapter, thread_id="parent", enable_subagents=True)
        try:
            result = await parent.submit("go")
            assert result.status is TurnStatus.COMPLETED
            payload = _result_items(parent.store, result.turn_id, "wait_subagent")[0]
            assert payload["error_code"] == "child_not_found"
        finally:
            await parent.aclose()

    asyncio.run(scenario())


# ---------------------------------------------------------------------------
# ⑤ R-T5-7:guard 全家族口径(跨 turn)
# ---------------------------------------------------------------------------


def test_cross_turn_spawn_still_bounded_by_family_max_total(workspace: Path) -> None:
    async def scenario() -> None:
        inner = ScriptedAdapter(
            [
                _resp("", (("s1", "spawn_subagent", {"task": "c1"}),)),
                _resp("child one done"),
                _resp("", (("w1", "wait_subagent", {"child_thread_id": "PLACEHOLDER", "timeout": 5}),)),
                _resp("turn one done"),
                _resp("", (("s2", "spawn_subagent", {"task": "c2"}),)),
                _resp("turn two done"),
            ]
        )
        store_holder: list[Store] = []
        parent = open_session(
            workspace, _ChildIdRewritingAdapter(inner, lambda: store_holder[0], "parent"),
            thread_id="parent", enable_subagents=True, subagent_limits={"max_total": 1},
        )
        store_holder.append(parent.store)
        try:
            first = await parent.submit("go")
            assert first.status is TurnStatus.COMPLETED
            assert _result_items(parent.store, first.turn_id, "wait_subagent")[0]["metadata"]["state"] == "done"
            # 第二个 turn:第一个子已终态,但全家族 max_total 累计口径仍拒。
            second = await parent.submit("again")
            assert second.status is TurnStatus.COMPLETED
            payload = _result_items(parent.store, second.turn_id, "spawn_subagent")[0]
            assert payload["error_code"] == "child_limit_reached"
            assert "total" in str(payload["error_message"])
        finally:
            await parent.aclose()

    asyncio.run(scenario())


def test_cross_turn_spawn_still_bounded_by_family_max_concurrent(workspace: Path) -> None:
    child_release = asyncio.Event()

    async def scenario() -> None:
        inner = ScriptedAdapter(
            [
                _resp("", (("s1", "spawn_subagent", {"task": "c1"}),)),
                _resp("", (("h1", "child_hang", {}),)),
                _resp("turn one done"),
                _resp("", (("s2", "spawn_subagent", {"task": "c2"}),)),
                _resp("turn two done"),
                _resp("child done"),
            ]
        )
        store_holder: list[Store] = []
        parent = open_session(
            workspace,
            _ChildIdRewritingAdapter(inner, lambda: store_holder[0], "parent"),
            thread_id="parent", enable_subagents=True,
            subagent_limits={"max_concurrent": 1},
            extra_tools=(_hang_tool("child_hang", child_release),),
        )
        store_holder.append(parent.store)
        try:
            first = await parent.submit("go")
            assert first.status is TurnStatus.COMPLETED
            assert first.children_summary[0].state == "running"
            # turn 1 的子仍 active(跨 turn 占用并发份额)→ turn 2 spawn 拒。
            second = await parent.submit("again")
            assert second.status is TurnStatus.COMPLETED
            payload = _result_items(parent.store, second.turn_id, "spawn_subagent")[0]
            assert payload["error_code"] == "child_limit_reached"
            assert "concurrent" in str(payload["error_message"])
            child_release.set()
            child_id = str(
                _result_items(parent.store, first.turn_id, "spawn_subagent")[0]["metadata"]["child_thread_id"]
            )
            allocations = [
                item for item in parent.store.read_items("parent") if item.kind == ItemKind.CHILD_ALLOCATION
            ]
            root_turn = str(allocations[0].payload["child_root_turn_id"])
            for _ in range(500):
                turn = get_turn(parent.store, root_turn)
                if turn is not None and turn.status == "completed":
                    break
                await asyncio.sleep(0.01)
            else:
                pytest.fail("child did not converge after release")
            assert child_id
        finally:
            child_release.set()
            await parent.aclose()

    asyncio.run(scenario())


# ---------------------------------------------------------------------------
# ⑥ R-T5-8:read_only 子网络=父决策、外部读保留、写否决承重不变
# ---------------------------------------------------------------------------


def test_read_only_child_network_and_external_read_inherit_parent(workspace: Path, tmp_path: Path) -> None:
    external = tmp_path / "external"
    external.mkdir()
    (external / "data.txt").write_text("ext", encoding="utf-8")

    async def scenario() -> None:
        policy = PermissionPolicy(
            network=NetworkPolicy(decision="allow"),
            filesystem=(PathRule(path=str(external), access="read"),),
        )
        parent = open_session(
            workspace, ScriptedAdapter([]), tools=[], thread_id="parent", policy=policy
        )
        child = build_child_session(
            parent=parent, workspace=workspace, thread_id=uuid4().hex,
            limits=UNLIMITED, tool_subset=(),
        )
        try:
            url = "https://example.com/path"
            assert network_verdict(parent.permissions, url) == "allow"
            # 网络决策=父决策(不再被 ceiling 默认 ask 拖窄)。
            assert network_verdict(child.permissions, url) == "allow"
            readonly = [c for c in child.permissions.ceilings if c.profile == "read-only"]
            assert len(readonly) == 1
            resolved = resolve_policy(readonly[0], workspace)
            # 外部读授权保留;写否决(R-T5-6 承重)不变。
            assert path_access(resolved, external / "data.txt", write=False) is True
            assert path_access(resolved, external / "data.txt", write=True) is False
            assert path_access(resolved, workspace, write=True) is False
            assert path_access(resolved, workspace / "f.txt", write=False) is True
        finally:
            await child.aclose()
            await parent.aclose()

    asyncio.run(scenario())


def test_read_only_child_network_ask_and_deny_follow_parent(workspace: Path) -> None:
    async def scenario() -> None:
        for decision in ("ask", "deny"):
            parent = open_session(
                workspace, ScriptedAdapter([]), tools=[], thread_id=f"parent-{decision}",
                policy=PermissionPolicy(network=NetworkPolicy(decision=decision)),
            )
            child = build_child_session(
                parent=parent, workspace=workspace, thread_id=uuid4().hex,
                limits=UNLIMITED, tool_subset=(),
            )
            try:
                url = "https://example.com/path"
                assert network_verdict(child.permissions, url) == decision
                assert network_verdict(child.permissions, url) == network_verdict(parent.permissions, url)
            finally:
                await child.aclose()
                await parent.aclose()

    asyncio.run(scenario())


def test_read_only_child_write_grant_in_parent_policy_becomes_read_only(workspace: Path, tmp_path: Path) -> None:
    """父对外部路径的 write 授权:子保留读、否决写(profile 承重)。"""
    external = tmp_path / "external-w"
    external.mkdir()

    async def scenario() -> None:
        parent = open_session(
            workspace, ScriptedAdapter([]), tools=[], thread_id="parent",
            policy=PermissionPolicy(filesystem=(PathRule(path=str(external), access="write"),)),
        )
        child = build_child_session(
            parent=parent, workspace=workspace, thread_id=uuid4().hex,
            limits=UNLIMITED, tool_subset=(),
        )
        try:
            readonly = [c for c in child.permissions.ceilings if c.profile == "read-only"][0]
            resolved = resolve_policy(readonly, workspace)
            assert path_access(resolved, external, write=False) is True
            assert path_access(resolved, external, write=True) is False
        finally:
            await child.aclose()
            await parent.aclose()

    asyncio.run(scenario())


# ---------------------------------------------------------------------------
# ⑦ ㉖:子 always 不污染父 permissions.json
# ---------------------------------------------------------------------------


def test_read_only_child_always_never_writes_parent_permissions(workspace: Path) -> None:
    async def scenario() -> None:
        parent = open_session(workspace, ScriptedAdapter([]), tools=[], thread_id="parent")
        declared = workspace / ".praxis" / "permissions.json"
        declared.write_text(PermissionPolicy().model_dump_json(indent=2), encoding="utf-8")
        before = declared.read_bytes()
        child = build_child_session(
            parent=parent, workspace=workspace, thread_id=uuid4().hex,
            limits=UNLIMITED, tool_subset=(),
        )
        try:
            notice = child.update_permission(
                PermissionUpdate(rule=ToolRule(pattern="read", decision="allow"), destination="always")
            )
            # 内存生效(子自身 lifetime),但父文件字节不动。
            assert tool_decision(resolve_policy(child.permissions.policy, workspace), "read") == "allow"
            assert declared.read_bytes() == before
            assert notice is not None and notice.kind == "always_memory_only"
        finally:
            await child.aclose()
            await parent.aclose()

    asyncio.run(scenario())


def test_write_child_always_persists_to_its_own_isolated_file(workspace: Path, tmp_path: Path) -> None:
    child_ws = tmp_path / "child-ws"
    child_ws.mkdir()

    async def scenario() -> None:
        parent = open_session(workspace, ScriptedAdapter([]), tools=[], thread_id="parent")
        declared = workspace / ".praxis" / "permissions.json"
        declared.write_text(PermissionPolicy().model_dump_json(indent=2), encoding="utf-8")
        before = declared.read_bytes()
        child = build_child_session(
            parent=parent, workspace=child_ws, thread_id=uuid4().hex,
            limits=UNLIMITED, tool_subset=(), workspace_mode="write",
        )
        try:
            child.update_permission(
                PermissionUpdate(rule=ToolRule(pattern="read", decision="allow"), destination="always")
            )
            assert declared.read_bytes() == before
            child_file = child_ws / ".praxis" / "permissions.json"
            assert child_file.exists()
            assert "read" in child_file.read_text(encoding="utf-8")
        finally:
            await child.aclose()
            await parent.aclose()

    asyncio.run(scenario())


# ---------------------------------------------------------------------------
# ⑧ ㉒:driver 收尾竞态(shield);③:manager.aclose 挂 Session 收尾链
# ---------------------------------------------------------------------------


def test_cancel_during_child_teardown_does_not_leak_store_writer(workspace: Path) -> None:
    """读 RUNNING 后子恰终结:二次 cancel 落在 teardown 窗口——子 Store
    写者线程必须仍收敛(shield 承重),不得被打断成孤儿。"""
    child_release = asyncio.Event()
    holder: list[SubagentManager] = []
    results: list[Any] = []

    async def spawn_probe(arguments: Mapping[str, JsonValue], exec_ctx: ToolExecContext) -> ToolResult:
        manager = holder[0]
        handle = await manager.spawn("child task")
        results.append(handle)
        return _ok("spawn_probe", handle.child_thread_id)

    async def scenario() -> None:
        inner = ScriptedAdapter(
            [
                _resp("", (("s1", "spawn_probe", {}),)),
                _resp("", (("h1", "child_hang", {}),)),
                _resp("parent done"),
                _resp("不可达"),
            ]
        )
        parent = open_session(
            workspace, inner, tools=[_toy_tool("spawn_probe", spawn_probe), _hang_tool("child_hang", child_release)],
            thread_id="parent",
        )
        manager = SubagentManager(parent=parent, store=parent.store)
        holder.append(manager)
        try:
            await parent.submit("go")
            handle: ChildHandle = results[0]
            driver = manager._drivers[handle.child_thread_id]  # noqa: SLF001 — 竞态窗口注入点
            real_aclose = driver.session.aclose
            teardown_started = asyncio.Event()

            async def slow_aclose() -> None:
                teardown_started.set()
                await asyncio.sleep(0.1)
                await real_aclose()

            driver.session.aclose = slow_aclose  # type: ignore[method-assign]
            driver.task.cancel()  # 第一次:run_turn 收敛 CANCELLED
            await asyncio.wait_for(teardown_started.wait(), 5)
            driver.task.cancel()  # 第二次:恰落在 teardown 窗口
            await asyncio.gather(driver.task, return_exceptions=True)
            # 子 Store 写者线程已收敛:只剩父的那一条。
            assert len(_writer_threads()) == 1
            turn = get_turn(parent.store, handle.child_root_turn_id)
            assert turn is not None and turn.status == "cancelled"
        finally:
            child_release.set()
            await manager.aclose()
            await parent.aclose()

    asyncio.run(scenario())


def test_session_close_converges_child_drivers_and_store_writers(workspace: Path) -> None:
    """③:Session.aclose 收尾链含 manager.aclose——关闭后无子 driver、
    无 Store 写者线程泄漏;子 turn 收敛 CANCELLED。"""
    child_release = asyncio.Event()  # 永不释放:关闭时子仍在跑

    async def scenario() -> None:
        inner = ScriptedAdapter(
            [
                _resp("", (("s1", "spawn_subagent", {"task": "child task"}),)),
                _resp("", (("h1", "child_hang", {}),)),
                _resp("parent done"),
                _resp("不可达"),
            ]
        )
        parent = open_session(
            workspace, inner, thread_id="parent", enable_subagents=True,
            extra_tools=(_hang_tool("child_hang", child_release),),
        )
        try:
            result = await parent.submit("go")
            assert result.status is TurnStatus.COMPLETED
            assert result.children_summary[0].state == "running"
            allocations = [
                item for item in parent.store.read_items("parent") if item.kind == ItemKind.CHILD_ALLOCATION
            ]
            root_turn = str(allocations[0].payload["child_root_turn_id"])
        finally:
            await parent.aclose()
        assert _writer_threads() == []
        assert asyncio.all_tasks() == {asyncio.current_task()}
        verify = Store(workspace / ".praxis" / "praxis.sqlite")
        try:
            turn = get_turn(verify, root_turn)
            assert turn is not None and turn.status == "cancelled"
        finally:
            verify.close()
        assert _writer_threads() == []

    asyncio.run(scenario())
