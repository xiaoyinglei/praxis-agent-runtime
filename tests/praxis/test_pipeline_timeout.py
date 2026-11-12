"""Task 12:超时/取消语义移植回归(Review Focus 5;旧 executor :1564-1657 语义)。

四条路径:
- 同步工具线程不可杀:FINISH_CURRENT 超时后 drain,交付真实结果;
- async 工具 CANCEL:超时 → cancel+drain → TIMEOUT;
- 外部取消:持有(own)工具任务直到停止,再上抛 CancelledError
  ——tool_call 已落库、无 tool_result(留给 recovery 判 pending);
- 连续 cancel:_drain_task 循环持有,任务不失控,结局确定(回归测试)。

仓库无 pytest-asyncio:asyncio.run 包装,同 test_tools_builtin.py 风格。
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import pytest

from praxis.core.events import EventBus
from praxis.core.pipeline import ExecCtx, execute, run_with_timeout
from praxis.protocol.items import ItemKind
from praxis.protocol.permission import PermissionContext
from praxis.protocol.policy import PermissionPolicy
from praxis.protocol.tools import (
    CancellationMode,
    InterruptBehavior,
    ResolvedUse,
    ResultStatus,
    Tool,
    ToolCall,
    ToolContentBlock,
    ToolDefinition,
    ToolResult,
    json_schema_input,
)
from praxis.store.log import Store
from praxis.store.turns import create_turn

_SCHEMA: dict[str, Any] = {"type": "object", "properties": {}, "additionalProperties": False}


def _ok(text: str) -> ToolResult:
    return ToolResult(
        tool_call_id="pending",  # documented convention:管道占位符
        tool_name="toy",
        content=(ToolContentBlock(type="text", data={"text": text}),),
        status=ResultStatus.OK,
    )


def _toy_tool(run: Callable[[Mapping[str, Any]], Any], *, interrupt: InterruptBehavior, timeout: float) -> Tool:
    return Tool(
        definition=ToolDefinition(name="toy", description="timeout toy", input_schema=_SCHEMA),
        validate_input=json_schema_input(_SCHEMA),
        run=run,
        static_effects=frozenset(),
        resolve_use=lambda arguments: ResolvedUse(effects=frozenset(), targets=()),
        idempotent=True,
        concurrency_safe=True,
        cancellation_mode=CancellationMode.COOPERATIVE,
        interrupt_behavior=interrupt,
        timeout_seconds=timeout,
        max_model_output_bytes=4_096,
    )


def sync_tool(delay: float, interrupt: InterruptBehavior, timeout: float) -> Tool:
    def run(arguments: Mapping[str, Any]) -> ToolResult:
        time.sleep(delay)
        return _ok("sync-done")

    return _toy_tool(run, interrupt=interrupt, timeout=timeout)


def async_tool(
    delay: float,
    interrupt: InterruptBehavior,
    timeout: float,
    state: dict[str, int],
    *,
    cleanup: float = 0.0,
) -> Tool:
    """cleanup>0:取消处理里再 await 一次——多步收尾让"未持有即上抛"的泄漏现形。"""

    async def run(arguments: Mapping[str, Any]) -> ToolResult:
        state["started"] = state.get("started", 0) + 1
        try:
            await asyncio.sleep(delay)
        except asyncio.CancelledError:
            state["cancelled"] = state.get("cancelled", 0) + 1
            await asyncio.sleep(cleanup)
            raise
        return _ok("async-done")

    return _toy_tool(run, interrupt=interrupt, timeout=timeout)


def _live_tasks() -> list[asyncio.Task[Any]]:
    """execute/直调任务浮出终态的当下,循环里不应再有未完成的他人任务。"""
    return [t for t in asyncio.all_tasks() if t is not asyncio.current_task() and not t.done()]


# ---------------------------------------------------------------------------
# Step 5 的四个形态
# ---------------------------------------------------------------------------


def test_sync_thread_timeout_drains_not_hangs() -> None:
    tool = sync_tool(0.3, InterruptBehavior.FINISH_CURRENT, timeout=0.05)
    started = time.monotonic()
    r = asyncio.run(run_with_timeout(tool, {}))
    elapsed = time.monotonic() - started
    assert r.status is ResultStatus.OK and r.text == "sync-done"  # 线程真实完成的真实结果
    assert elapsed >= 0.3  # 线程不可杀:等它跑完,这是如实的限制
    assert elapsed < 5.0  # 绝不挂起


def test_async_tool_timeout_cancels() -> None:
    state: dict[str, int] = {}
    tool = async_tool(10.0, InterruptBehavior.CANCEL, timeout=0.1, state=state)
    started = time.monotonic()
    r = asyncio.run(run_with_timeout(tool, {}))
    assert r.status is ResultStatus.TIMEOUT
    assert r.is_error and r.error_code == "timeout"
    assert r.tool_call_id == "pending"  # run_with_timeout 保持占位,execute 替换
    assert state.get("cancelled") == 1  # 工具任务确被取消
    assert time.monotonic() - started < 5.0


def _ctx_with_tool(store: Store, workspace: Path, tool: Tool) -> ExecCtx:
    policy = PermissionPolicy()
    turn = create_turn(store, turn_id="turn-1", thread_id="thread-1", binding={})
    return ExecCtx(
        store=store,
        bus=EventBus(),
        workspace=workspace,
        tools={"toy": tool},
        permissions=PermissionContext(workspace=str(workspace), policy=policy, mode="on-request"),
        turn_id=turn.turn_id,
        thread_id="thread-1",
        policy_revision=policy.revision,
    )


def test_external_cancellation_drains_then_propagates(store: Store, workspace: Path) -> None:
    state: dict[str, int] = {}
    tool = async_tool(10.0, InterruptBehavior.CANCEL, timeout=30.0, state=state)

    async def main() -> None:
        task = asyncio.create_task(execute(ToolCall("c1", "toy", {}), _ctx_with_tool(store, workspace, tool)))
        for _ in range(200):
            await asyncio.sleep(0.01)
            if state.get("started"):
                break
        assert state.get("started") == 1
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(main())
    assert state.get("cancelled") == 1  # 外部取消后,工具任务先被持有到停止
    kinds = [i.kind for i in store.read_turn_items("turn-1")]
    assert ItemKind.TOOL_CALL in kinds  # call 已落库(执行前写入)
    assert ItemKind.TOOL_RESULT not in kinds  # 无 result → recovery 判 pending


def test_repeated_cancellation_still_owned(store: Store, workspace: Path) -> None:
    """两次真实投递的取消(CANCEL 模式):_drain_task 循环必须持有到 runner 终止。

    关键:cancel() 之间必须 await——同一步内背靠背两次 cancel 在 CPython 3.12
    只投递 1 次 CancelledError,测不到重复取消;sleep(0) 隔开才有第 2 次投递。
    cleanup=0.05 让 runner 的终止晚于"第一次取消到达",若管道在收到第 2 次
    投递时直接上抛(mutated _drain_task),此刻 runner 仍活着 → _live_tasks 非空。
    """
    state: dict[str, int] = {}
    tool = async_tool(0.5, InterruptBehavior.CANCEL, timeout=30.0, state=state, cleanup=0.05)

    async def main() -> None:
        task = asyncio.create_task(execute(ToolCall("c1", "toy", {}), _ctx_with_tool(store, workspace, tool)))
        for _ in range(200):
            await asyncio.sleep(0.01)
            if state.get("started"):
                break
        assert state.get("started") == 1
        task.cancel()
        await asyncio.sleep(0)  # 隔开:让第一次取消真正投递
        task.cancel()  # 第二次 → 投递进 drain 的 shield await,考验循环持有
        with pytest.raises(asyncio.CancelledError):
            await task
        assert _live_tasks() == []  # 上抛的当下,runner 任务必已终止——不泄漏

    asyncio.run(main())
    assert state.get("cancelled") == 1  # 恰好一次投递给工具,结局确定
    kinds = [i.kind for i in store.read_turn_items("turn-1")]
    assert ItemKind.TOOL_CALL in kinds
    assert ItemKind.TOOL_RESULT not in kinds


def test_finish_current_repeated_cancellation() -> None:
    """FINISH_CURRENT + 两次隔开的外部取消:drain 等线程真实跑完,然后才上抛。

    线程不可杀在此是硬约束:第 2 次取消投递进 drain 时,shield 循环必须继续
    持有直到 to_thread worker 自然完成(state["done"]),再按 brief 语义上抛。
    """
    state: dict[str, int] = {}

    def run(arguments: Mapping[str, Any]) -> ToolResult:
        state["enter"] = 1
        time.sleep(0.3)
        state["done"] = 1
        return _ok("sync-done")

    tool = _toy_tool(run, interrupt=InterruptBehavior.FINISH_CURRENT, timeout=30.0)

    async def main() -> tuple[float, list[asyncio.Task[Any]]]:
        task = asyncio.create_task(run_with_timeout(tool, {}))
        for _ in range(100):
            await asyncio.sleep(0.01)
            if state.get("enter"):
                break
        assert state.get("enter") == 1
        started = time.monotonic()
        task.cancel()
        await asyncio.sleep(0)  # 隔开两次投递
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        return time.monotonic() - started, _live_tasks()

    elapsed, live = asyncio.run(main())
    assert live == []
    assert state.get("done") == 1  # 线程真实完成发生在传播之前,而非之后
    assert elapsed >= 0.25  # 0.3s 线程睡眠扣去轮询开销;mutated drain ≈0.00s
