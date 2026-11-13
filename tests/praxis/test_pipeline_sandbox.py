"""Task 7:管道沙箱接线 —— fake compiler 单测 + 真实 seatbelt 管道级验收。

fake compiler 单测(全平台):
- 步骤序 compile → tool_call item → run → finalize(ToolExecContext.sandbox 透传);
- SandboxRefusal 是结果不是异常(ERROR/sandbox_refused,无 tool_call,未 dispatch);
- finalize 在超时/外部取消路径也执行(finally);finalize 异常归一 tool_error;
- 沙箱不可用 → decide 第 7 步 deny,compiler 零调用(bound allow 不得翻案,R5a);
- compile_sandbox=None → 无沙箱直跑(单测路径);编译只对 EXECUTE_PROCESS 发生
  (read/write 进程内工具不编译,R9)。

真实验收(darwin seatbelt,经唯一组合根 composition):
- bash 写 workspace 成功、写 .git 在 syscall 层失败(S2 管道级);
- TMPDIR=scratch 且 finalize 后 scratch 已清理;
- trusted-local + bash → SandboxRefusal → sandbox_refused 结果(R9 fail-closed),
  同策略下 read/write 进程内工具不受影响。

仓库无 pytest-asyncio:与 test_pipeline.py 同风格,asyncio.run 包装。
"""

from __future__ import annotations

import asyncio
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pytest

from praxis.adapters.models.scripted import ScriptedAdapter
from praxis.adapters.sandbox.base import probe_backend
from praxis.adapters.sandbox.compile import compile_sandbox, finalize_sandbox
from praxis.core.events import EventBus
from praxis.core.pipeline import ExecCtx, execute
from praxis.core.statemachine import TurnStatus
from praxis.interfaces.sdk import open_session
from praxis.protocol.items import ItemKind
from praxis.protocol.json import JsonValue
from praxis.protocol.models import ModelResponse, ModelToolCall, Usage
from praxis.protocol.permission import PermissionContext, SandboxAvailability
from praxis.protocol.policy import NetworkPolicy, PermissionPolicy
from praxis.protocol.sandbox import SandboxRefusal, SandboxSpec, ToolExecContext
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
from praxis.store.turns import create_turn
from praxis.tools.builtins.filesystem import create_read_tool

_AVAILABLE = probe_backend()
_EMPTY_SCHEMA: dict[str, Any] = {"type": "object", "properties": {}, "additionalProperties": False}


# ---------------------------------------------------------------------------
# 脚手架:记录序的 Store 包装、fake EXECUTE_PROCESS 工具、fake compiler/finalizer
# ---------------------------------------------------------------------------


class _RecordingStore:
    """委托真实 Store,只把 tool_call item 的写入记进 order(步骤序断言用)。"""

    def __init__(self, inner: Store, order: list[str]) -> None:
        self._inner = inner
        self._order = order

    def append_items(
        self, thread_id: str, turn_id: str | None, entries: Sequence[ItemEntry]
    ) -> Any:
        for entry in entries:
            if entry.kind == ItemKind.TOOL_CALL:
                self._order.append("tool_call")
        return self._inner.append_items(thread_id, turn_id, entries)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


def _process_tool(
    workspace: Path,
    order: list[str],
    seen: list[ToolExecContext],
    *,
    delay: float = 0.0,
    timeout: float = 30.0,
) -> Tool:
    """fake EXECUTE_PROCESS 工具:记录 run 序与收到的 exec_ctx;delay>0 时睡眠。"""

    async def run(arguments: Mapping[str, JsonValue], exec_ctx: ToolExecContext) -> ToolResult:
        order.append("run")
        seen.append(exec_ctx)
        if delay > 0:
            await asyncio.sleep(delay)
        return ToolResult(
            tool_call_id="pending",
            tool_name="proc",
            content=(ToolContentBlock(type="text", data={"text": "ran"}),),
            status=ResultStatus.OK,
        )

    return Tool(
        definition=ToolDefinition(name="proc", description="fake process tool", input_schema=_EMPTY_SCHEMA),
        validate_input=json_schema_input(_EMPTY_SCHEMA),
        run=run,
        static_effects=frozenset({ToolEffect.EXECUTE_PROCESS}),
        resolve_use=lambda arguments: ResolvedUse(
            effects=frozenset({ToolEffect.EXECUTE_PROCESS}),
            targets=(ToolTarget(kind="path", value=str(workspace)),),
        ),
        idempotent=False,
        concurrency_safe=False,
        cancellation_mode=CancellationMode.MANAGED_PROCESS,
        interrupt_behavior=InterruptBehavior.CANCEL,
        timeout_seconds=timeout,
        max_model_output_bytes=4096,
    )


def _make_ctx(
    store: Store,
    workspace: Path,
    tools: dict[str, Tool],
    *,
    available: bool = True,
    compiler: Any = None,
    finalizer: Any = None,
) -> ExecCtx:
    # network allow:真实 decide 下 EXECUTE_PROCESS 才能走到第 7 步沙箱裁决
    policy = PermissionPolicy(network=NetworkPolicy(decision="allow"))
    turn = create_turn(store, turn_id="turn-1", thread_id="thread-1", binding={})
    return ExecCtx(
        store=store,
        bus=EventBus(),
        workspace=workspace,
        tools=tools,
        permissions=PermissionContext(
            workspace=str(workspace),
            policy=policy,
            mode="on-request",
            sandbox=SandboxAvailability(backend="seatbelt" if available else None, available=available),
        ),
        turn_id=turn.turn_id,
        thread_id="thread-1",
        policy_revision=policy.revision,
        compile_sandbox=compiler,
        finalize_sandbox=finalizer,
    )


def _fake_spec(tmp_path: Path) -> SandboxSpec:
    return SandboxSpec(
        backend="seatbelt",
        argv_prefix=("/usr/bin/sandbox-exec", "-p", "(version 1)"),
        scratch_dir=str(tmp_path / "scratch"),
    )


def _resp(content: str, calls: tuple[tuple[str, str, dict[str, Any]], ...] = ()) -> ModelResponse:
    return ModelResponse(
        content=content,
        tool_calls=tuple(ModelToolCall(call_id=c, name=n, arguments=a) for c, n, a in calls),
        usage=Usage(input_tokens=3, output_tokens=5),
        finish_reason="tool_calls" if calls else "stop",
    )


# ---------------------------------------------------------------------------
# fake compiler 单测:步骤序 / refusal / finalize / 不可用 / 无编译器
# ---------------------------------------------------------------------------


def test_execute_compiles_before_dispatch(store: Store, workspace: Path, tmp_path: Path) -> None:
    order: list[str] = []
    seen: list[ToolExecContext] = []
    compiled_with: list[tuple[Any, Any, Any]] = []
    finalized_with: list[tuple[Any, Any]] = []
    spec = _fake_spec(tmp_path)

    def compiler(permissions: PermissionContext, use: ResolvedUse, root: Path) -> SandboxSpec:
        order.append("compile")
        compiled_with.append((permissions, use, root))
        return spec

    def finalizer(compiled: SandboxSpec, root: Path) -> None:
        order.append("finalize")
        finalized_with.append((compiled, root))

    tool = _process_tool(workspace, order, seen)
    ctx = _make_ctx(
        _RecordingStore(store, order), workspace, {"proc": tool}, compiler=compiler, finalizer=finalizer
    )
    # _RecordingStore 只替换写序观测;turn 读取仍走真实 Store
    r = asyncio.run(execute(ToolCall("c1", "proc", {}), ctx))

    assert r.status is ResultStatus.OK
    assert order == ["compile", "tool_call", "run", "finalize"]
    assert seen[0].sandbox is spec  # ToolExecContext 透传编译产物
    permissions, use, root = compiled_with[0]
    assert permissions is ctx.permissions and root == workspace
    assert ToolEffect.EXECUTE_PROCESS in use.effects
    assert finalized_with == [(spec, workspace)]


def test_compile_refusal_is_result_not_exception(store: Store, workspace: Path) -> None:
    order: list[str] = []
    seen: list[ToolExecContext] = []

    def compiler(permissions: PermissionContext, use: ResolvedUse, root: Path) -> SandboxSpec:
        order.append("compile")
        raise SandboxRefusal("boundary not representable")

    def finalizer(compiled: SandboxSpec, root: Path) -> None:
        order.append("finalize")

    tool = _process_tool(workspace, order, seen)
    ctx = _make_ctx(store, workspace, {"proc": tool}, compiler=compiler, finalizer=finalizer)
    r = asyncio.run(execute(ToolCall("c1", "proc", {}), ctx))

    assert r.status is ResultStatus.ERROR and r.error_code == "sandbox_refused"
    assert r.error_message == "boundary not representable"
    assert order == ["compile"]  # 未 dispatch、未 finalize(没有编译产物)
    assert seen == []
    items = store.read_turn_items("turn-1")
    assert [i.kind for i in items] == [ItemKind.TOOL_RESULT]  # 无 tool_call:未 dispatch 语义同 invalid_input
    assert items[0].payload["is_error"] is True


def test_finalize_runs_on_timeout(store: Store, workspace: Path, tmp_path: Path) -> None:
    order: list[str] = []
    seen: list[ToolExecContext] = []
    spec = _fake_spec(tmp_path)
    finalized: list[SandboxSpec] = []

    def compiler(permissions: PermissionContext, use: ResolvedUse, root: Path) -> SandboxSpec:
        order.append("compile")
        return spec

    def finalizer(compiled: SandboxSpec, root: Path) -> None:
        order.append("finalize")
        finalized.append(compiled)

    tool = _process_tool(workspace, order, seen, delay=10.0, timeout=0.1)
    ctx = _make_ctx(
        _RecordingStore(store, order), workspace, {"proc": tool}, compiler=compiler, finalizer=finalizer
    )
    r = asyncio.run(execute(ToolCall("c1", "proc", {}), ctx))

    assert r.status is ResultStatus.TIMEOUT and r.error_code == "timeout"
    assert finalized == [spec]  # 超时路径 finalize 仍执行(commit/清理不丢)
    assert order == ["compile", "tool_call", "run", "finalize"]


def test_finalize_runs_on_external_cancel(store: Store, workspace: Path, tmp_path: Path) -> None:
    order: list[str] = []
    seen: list[ToolExecContext] = []
    spec = _fake_spec(tmp_path)
    finalized: list[SandboxSpec] = []

    def compiler(permissions: PermissionContext, use: ResolvedUse, root: Path) -> SandboxSpec:
        return spec

    def finalizer(compiled: SandboxSpec, root: Path) -> None:
        finalized.append(compiled)

    tool = _process_tool(workspace, order, seen, delay=30.0)
    ctx = _make_ctx(store, workspace, {"proc": tool}, compiler=compiler, finalizer=finalizer)

    async def main() -> None:
        task = asyncio.create_task(execute(ToolCall("c1", "proc", {}), ctx))
        for _ in range(200):
            await asyncio.sleep(0.01)
            if seen:
                break
        assert seen  # run 已开始
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(main())
    assert finalized == [spec]  # 取消路径 finalize 仍执行
    kinds = [i.kind for i in store.read_turn_items("turn-1")]
    assert ItemKind.TOOL_CALL in kinds and ItemKind.TOOL_RESULT not in kinds  # 取消语义不变


def test_finalize_exception_normalized_to_tool_error(store: Store, workspace: Path, tmp_path: Path) -> None:
    order: list[str] = []
    seen: list[ToolExecContext] = []
    spec = _fake_spec(tmp_path)

    def compiler(permissions: PermissionContext, use: ResolvedUse, root: Path) -> SandboxSpec:
        return spec

    def finalizer(compiled: SandboxSpec, root: Path) -> None:
        raise RuntimeError("cleanup boom")

    tool = _process_tool(workspace, order, seen)
    ctx = _make_ctx(store, workspace, {"proc": tool}, compiler=compiler, finalizer=finalizer)
    r = asyncio.run(execute(ToolCall("c1", "proc", {}), ctx))

    assert r.status is ResultStatus.ERROR and r.error_code == "tool_error"
    assert isinstance(r.error_message, str) and "cleanup boom" in r.error_message
    items = store.read_turn_items("turn-1")
    assert [i.kind for i in items] == [ItemKind.TOOL_CALL, ItemKind.TOOL_RESULT]  # 不炸穿,result 落库


def test_sandbox_unavailable_denies_before_compile(store: Store, workspace: Path) -> None:
    order: list[str] = []
    seen: list[ToolExecContext] = []
    compiled: list[int] = []

    def compiler(permissions: PermissionContext, use: ResolvedUse, root: Path) -> SandboxSpec:
        compiled.append(1)
        raise AssertionError("compiler must not be called when sandbox unavailable")

    tool = _process_tool(workspace, order, seen)
    ctx = _make_ctx(store, workspace, {"proc": tool}, available=False, compiler=compiler)
    r = asyncio.run(execute(ToolCall("c1", "proc", {}), ctx))

    assert r.status is ResultStatus.DENIED and r.error_code == "permission_denied"
    assert r.error_message == "sandbox_unavailable"
    assert compiled == [] and seen == []  # decide 第 7 步先 deny,compiler 零调用
    assert [i.kind for i in store.read_turn_items("turn-1")] == [ItemKind.TOOL_RESULT]


def test_bound_allow_cannot_override_sandbox_unavailable(store: Store, workspace: Path) -> None:
    """R5(a):新鲜 deny(sandbox_unavailable)是终局,bound allow 不得翻案。"""
    order: list[str] = []
    seen: list[ToolExecContext] = []
    tool = _process_tool(workspace, order, seen)
    ctx = _make_ctx(store, workspace, {"proc": tool}, available=False)
    store.append_items(
        "thread-1",
        "turn-1",
        [
            ItemEntry(
                kind=ItemKind.APPROVAL_DECISION,
                payload={
                    "call_id": "c1",
                    "verdict": "allow",
                    "args_digest": args_digest({}),
                    "destination": "once",
                    "policy_revision": ctx.policy_revision,
                },
            )
        ],
    )
    r = asyncio.run(execute(ToolCall("c1", "proc", {}), ctx))

    assert r.status is ResultStatus.DENIED and r.error_message == "sandbox_unavailable"
    assert seen == []
    assert [i.kind for i in store.read_turn_items("turn-1")] == [
        ItemKind.APPROVAL_DECISION,
        ItemKind.TOOL_RESULT,
    ]


def test_no_compiler_runs_without_sandbox(store: Store, workspace: Path) -> None:
    order: list[str] = []
    seen: list[ToolExecContext] = []
    tool = _process_tool(workspace, order, seen)
    ctx = _make_ctx(store, workspace, {"proc": tool})  # compile_sandbox=None:单测路径
    r = asyncio.run(execute(ToolCall("c1", "proc", {}), ctx))

    assert r.status is ResultStatus.OK
    assert seen[0].sandbox is None


def test_compile_only_for_execute_process(store: Store, workspace: Path) -> None:
    """R9:read/write 进程内工具不编译——trusted-local 等拒绝形态不得波及它们。"""
    (workspace / "f.txt").write_text("in-process", encoding="utf-8")
    compiled: list[int] = []

    def compiler(permissions: PermissionContext, use: ResolvedUse, root: Path) -> SandboxSpec:
        compiled.append(1)
        raise AssertionError("read tool must not trigger sandbox compile")

    ctx = _make_ctx(store, workspace, {"read": create_read_tool(workspace)}, compiler=compiler)
    r = asyncio.run(execute(ToolCall("c1", "read", {"path": "f.txt"}), ctx))

    assert r.status is ResultStatus.OK and r.text == "in-process"
    assert compiled == []


# ---------------------------------------------------------------------------
# 真实 seatbelt 管道级验收(darwin only;经唯一组合根 composition)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _AVAILABLE.available, reason="no sandbox backend")
def test_real_bash_under_sandbox_via_composition(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / ".git").mkdir()
    (workspace / ".git" / "config").write_text("orig", encoding="utf-8")
    adapter = ScriptedAdapter(
        [
            _resp(
                "",
                (("c1", "bash", {"command": "echo sandbox-ok > note.txt && cat note.txt && echo TMPDIR=$TMPDIR"}),),
            ),
            _resp("", (("c2", "bash", {"command": "echo pwned > .git/config"}),)),
            _resp("done"),
        ]
    )
    session = open_session(
        workspace, adapter, policy=PermissionPolicy(network=NetworkPolicy(decision="allow"))
    )
    try:
        result = asyncio.run(session.submit("real sandbox acceptance"))
        items = session.items()
    finally:
        session.close()

    assert result.status is TurnStatus.COMPLETED
    # workspace 写成功;TMPDIR 指向编译期 scratch(praxis-scratch- 前缀)
    assert (workspace / "note.txt").read_text(encoding="utf-8") == "sandbox-ok\n"
    results = {i.payload["call_id"]: i.payload for i in items if i.kind == ItemKind.TOOL_RESULT}
    c1_text: str = results["c1"]["content"][0]["data"]["text"]  # type: ignore[index]
    assert "sandbox-ok" in c1_text
    tmpdir_line = next(line for line in c1_text.splitlines() if line.startswith("TMPDIR="))
    scratch = Path(tmpdir_line[len("TMPDIR=") :])
    assert "praxis-scratch-" in scratch.name
    assert not scratch.exists()  # finalize 已清理 scratch
    # .git 写在 syscall 层失败:exit_code 非 0、内容不变(seatbelt 强制,非管道拦截)
    assert results["c2"]["status"] == "ok"  # 命令确实执行了,非零退出不是 is_error
    assert results["c2"]["metadata"]["exit_code"] != 0  # type: ignore[index]
    c2_text: str = results["c2"]["content"][0]["data"]["text"]  # type: ignore[index]
    assert "Operation not permitted" in c2_text
    assert (workspace / ".git" / "config").read_text(encoding="utf-8") == "orig"


@pytest.mark.skipif(not _AVAILABLE.available, reason="no sandbox backend")
def test_real_trusted_local_bash_refused_read_write_unaffected(tmp_path: Path) -> None:
    """R9:trusted-local + 写授权 → broad-root-read 不可沙箱化 → bash sandbox_refused;
    read/write 进程内工具同策略下照常工作。"""
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    adapter = ScriptedAdapter(
        [
            _resp("", (("c1", "write", {"path": "note.txt", "content": "in-process ok"}),)),
            _resp("", (("c2", "bash", {"command": "echo should-not-run > marker.txt"}),)),
            _resp("", (("c3", "read", {"path": "note.txt"}),)),
            _resp("done"),
        ]
    )
    session = open_session(
        workspace,
        adapter,
        policy=PermissionPolicy(profile="trusted-local", network=NetworkPolicy(decision="allow")),
    )
    try:
        result = asyncio.run(session.submit("trusted-local refusal"))
        items = session.items()
    finally:
        session.close()

    assert result.status is TurnStatus.COMPLETED
    assert (workspace / "note.txt").read_text(encoding="utf-8") == "in-process ok"
    assert not (workspace / "marker.txt").exists()  # bash 未 dispatch
    results = {i.payload["call_id"]: i.payload for i in items if i.kind == ItemKind.TOOL_RESULT}
    assert results["c2"]["status"] == "error"
    assert results["c2"]["error_code"] == "sandbox_refused"
    assert results["c3"]["content"][0]["data"]["text"] == "in-process ok"  # type: ignore[index]
    calls = [i.payload["call_id"] for i in items if i.kind == ItemKind.TOOL_CALL]
    assert calls == ["c1", "c3"]  # c2 无 tool_call item:未 dispatch


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-v"]))