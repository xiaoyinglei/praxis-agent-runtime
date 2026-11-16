"""kill -9 续跑矩阵的子进程驱动(Task 16;设计 §5.8;旧 §13 纪律)。

由父测试用 ``sys.executable`` 以独立 OS 进程启动,等着被 SIGKILL:

    python _child_run_turn.py <workspace> <responses.json> <thread_id> <variant>

variant 三档(与 test_kill9_resume.py 一一对应):

a  无注入:bash 执行 ``touch marker && sleep 30`` 本身就慢。父进程等
   "tool_call item 落库且 marker 真实出现"(touch 在 sleep 之前)才 kill,
   此刻回合停在 started+非幂等 的执行中。
b  故障注入缝(patch 执行循环对管道唯一 ``execute`` 的调用点)。注意:
   turn.py 是 ``from praxis.core.pipeline import execute``——按名导入,
   运行时查找的是 ``praxis.core.turn.execute`` 这个模块属性,patch 必须
   打在查找处才生效(brief 所述"patch pipeline.execute"的落点即此)。
   首次调用先睡 _HOLD_SECONDS 再进原函数:kill 恰好落在 model_response
   item 已落库之后、管道写 tool_call item 之前(§5.8"已生成未分发")。
c  故障注入缝(dataclasses.replace 换掉 read 工具的 run):协程先睡
   _HOLD_SECONDS 再真实读。tool_call item 已写、幂等工具执行中,kill
   落在执行中途;resume 应自动重跑(重放安全性由 read.idempotent=True 给出)。
d  Task 18 后台命令飞行中:首个 bash 带 yield_time_seconds 起长命令并
   立即返回 running tool_result;第二次 ``execute``(后续 read)前睡
   _HOLD_SECONDS,kill 落在"后台命令仍在跑、turn 仍在执行"的窗口。
   清扫由父测试以真实 sweep_orphan_command_jobs 验证。

阶段4 T9 崩溃矩阵变体(父+子同进程真 Session,子 driver 是 asyncio task——
进程级孤儿只可能来自 provisioner git 调用,父测试以 ps 复查 pgid 无残留):
responses.json 改形为 {"parent": [spec...], "child": [spec...],
"budget": {Limits 维度}, "mode": "read_only"|"write", "extend": {增量}}。
父剧本首帧 spawn_subagent(task 含标记 ``[child]``),wait/integrate 的
child_thread_id 用 ``PLACEHOLDER-1`` 由路由重写为真实 id。

alloc-before    allocation 事务**前**:缝 = agents.append_items_tx 查找处,
                CHILD_ALLOCATION 批次写入前睡(BEGIN IMMEDIATE 已开、未写;
                kill → 整事务回滚,allocation 与子根 turn 都不在)。
alloc-after     allocation 事务**后**:缝 = Store.transact,提交后检测到新增
                CHILD_ALLOCATION 即睡(driver task 尚未创建,子 turn running
                零 items)。
dispatch-before 子首 dispatch **前**:缝 = turn.execute,首个非父 thread 调用
                前睡(子 model_response 已落库、tool_call 未写)。
dispatch-after  子首 dispatch **后**:缝 = composition.create_write_tool
                工厂(父表与子重构同一查找处),write.run 先睡再真实写;
                子 tool_call 已写、side.txt 尚未出现(write 子,真 git worktree)。
settle-before   子终态 turn_status 写**前**:缝 = turn.transition,首个终态
                转换前睡(子先于父终结——父 wait 子)。
settle-after    子终态 turn_status 写**后**:同一缝,真实转换提交后睡
                (子已 completed、父仍 running;settle 是推导不是写)。
extend-before   extend item 写**前**:缝 = store.log.append_items_tx 查找处,
                BUDGET_EXTENSION 批次写入前睡(turn 已 PAUSED→RUNNING、
                事务已开未写;kill → 回滚,extension 不落库)。
extend-after    extend item 写**后**:缝 = Store.transact,提交后检测到新增
                BUDGET_EXTENSION 即睡(extension 已落库、续跑尚未推进)。

responses.json 形状(剧本每条一个对象,与 model_response item payload 同构):
    {"content": str,
     "tool_calls": [{"call_id": str, "name": str, "arguments": {}}],
     "usage": {"input_tokens": int, "output_tokens": int},
     "finish_reason": str}

子进程绝不自求多福地清理:SIGKILL 不留任何用户态收尾正是被验收的性质。
若它侥幸跑完 submit(缝失效=测试装置失败),打印结局并以退出码 1 报给父测试。
"""

from __future__ import annotations

import asyncio
import dataclasses
import json
import sys
import time
from collections.abc import AsyncIterator, Mapping, Sequence
from pathlib import Path
from typing import Any

# 脚本位于 tests/praxis/acceptance/,parents[3] = 仓库根;先修 path 再 import praxis。
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import praxis.composition as composition_module  # noqa: E402
import praxis.core.agents as agents_module  # noqa: E402
import praxis.core.turn as turn_module  # noqa: E402
import praxis.store.log as store_log_module  # noqa: E402
from praxis.adapters.models.scripted import ScriptedAdapter  # noqa: E402
from praxis.core.statemachine import TurnStatus  # noqa: E402
from praxis.core.turn import execute as _real_execute  # noqa: E402
from praxis.interfaces.sdk import open_session  # noqa: E402
from praxis.protocol.budget import Limits  # noqa: E402
from praxis.protocol.items import ItemKind  # noqa: E402
from praxis.protocol.json import JsonValue  # noqa: E402
from praxis.protocol.models import (  # noqa: E402
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
from praxis.protocol.policy import NetworkPolicy, PermissionPolicy  # noqa: E402
from praxis.protocol.sandbox import ToolExecContext  # noqa: E402
from praxis.protocol.tools import (  # noqa: E402
    CancellationMode,
    InterruptBehavior,
    ResolvedUse,
    ResultStatus,
    Tool,
    ToolContentBlock,
    ToolDefinition,
    ToolResult,
)
from praxis.store.log import ItemEntry, Store  # noqa: E402
from praxis.tools.builtins.command import create_bash_tool  # noqa: E402
from praxis.tools.builtins.filesystem import create_read_tool, create_write_tool  # noqa: E402

_HOLD_SECONDS = 45.0  # 必须远大于父进程轮询间隔:保证 kill 落在窗口正中

# T9 崩溃矩阵变体(见模块 docstring);responses.json 为 parent/child 双剧本形。
MATRIX_VARIANTS = frozenset(
    {
        "alloc-before",
        "alloc-after",
        "dispatch-before",
        "dispatch-after",
        "settle-before",
        "settle-after",
        "extend-before",
        "extend-after",
    }
)
_CHILD_MARKER = "[child]"


def _load_specs(path: str) -> list[Mapping[str, Any]]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, list) or not all(isinstance(entry, Mapping) for entry in raw):
        raise ValueError(f"responses.json must be a list of objects: {path}")
    return list(raw)


def _to_response(spec: Mapping[str, Any]) -> ModelResponse:
    calls = tuple(
        ModelToolCall(call_id=c["call_id"], name=c["name"], arguments=dict(c["arguments"]))
        for c in spec.get("tool_calls", [])
    )
    usage = spec.get("usage", {"input_tokens": 1, "output_tokens": 1})
    return ModelResponse(
        content=spec["content"],
        tool_calls=calls,
        usage=Usage(input_tokens=int(usage["input_tokens"]), output_tokens=int(usage["output_tokens"])),
        finish_reason=str(spec.get("finish_reason", "tool_calls" if calls else "stop")),
    )


def _install_dispatch_seam() -> None:
    """变体 b:turn 循环执行首个工具调用前挂起(见模块 docstring 的落点说明)。"""
    state = {"first": True}

    async def _seam_execute(call: Any, ctx: Any) -> Any:
        if state["first"]:
            state["first"] = False
            await asyncio.sleep(_HOLD_SECONDS)
        return await _real_execute(call, ctx)

    import praxis.core.turn as turn_module

    turn_module.execute = _seam_execute


def _install_second_dispatch_seam() -> None:
    """变体 d:第二次 execute(后台命令已 running 后的后续工具)前挂起。"""
    state = {"count": 0}

    async def _seam_execute(call: Any, ctx: Any) -> Any:
        state["count"] += 1
        if state["count"] == 2:
            await asyncio.sleep(_HOLD_SECONDS)
        return await _real_execute(call, ctx)

    import praxis.core.turn as turn_module

    turn_module.execute = _seam_execute


def _slow_read_tools(workspace: Path) -> list[Tool]:
    """变体 c:read.run 换成"睡 _HOLD_SECONDS 再真实读"的协程,其余字段原样。"""
    read_tool = create_read_tool(workspace)
    real_run = read_tool.run

    async def _slow_run(arguments: Mapping[str, JsonValue], exec_ctx: ToolExecContext) -> Any:
        await asyncio.sleep(_HOLD_SECONDS)
        return await real_run(arguments, exec_ctx)

    slow_read = dataclasses.replace(read_tool, run=_slow_run)
    return [slow_read, create_write_tool(workspace), create_bash_tool(workspace)]


# ---------------------------------------------------------------------------
# T9 崩溃矩阵:父+子双剧本路由与八个故障注入缝
# ---------------------------------------------------------------------------


class _MatrixRouter:
    """标记路由(子 user_message 含 [child])+ PLACEHOLDER-1 → 真实 child id。"""

    def __init__(
        self,
        parent: Sequence[ModelResponse],
        child: Sequence[ModelResponse],
        holder: list[Any],
        thread_id: str,
    ) -> None:
        self._parent = ScriptedAdapter(parent)
        self._child = ScriptedAdapter(child)
        self._holder = holder
        self._thread = thread_id

    def _latest_child_id(self) -> str:
        child_id = ""
        for item in self._holder[0].store.read_items(self._thread):
            if item.kind == ItemKind.CHILD_ALLOCATION:
                raw = item.payload.get("child_thread_id")
                if isinstance(raw, str):
                    child_id = raw
        return child_id

    def build_request(
        self, messages: Sequence[ModelMessage], tools: Sequence[Any], settings: ModelSettings
    ) -> WireRequest:
        return self._parent.build_request(messages, tools, settings)

    async def stream(self, req: WireRequest) -> AsyncIterator[ModelEvent]:
        text = " ".join(message.content for message in req.messages)
        adapter = self._child if _CHILD_MARKER in text else self._parent
        async for event in adapter.stream(req):
            if isinstance(event, ModelToolCallDelta) and event.name_fragment in (
                "wait_subagent",
                "integrate_child_workspace",
            ):
                args = json.loads(event.arguments_fragment)
                args["child_thread_id"] = self._latest_child_id()
                event = dataclasses.replace(event, arguments_fragment=json.dumps(args))
            yield event

    async def complete(self, req: WireRequest) -> ModelResponse:
        return assemble_model_events(await collect_events(self.stream(req)))


def _count_kind(store: Store, kind: str) -> int:
    return sum(
        1 for tid in store.list_threads() for item in store.read_items(tid) if item.kind == kind
    )


def _install_alloc_before_seam() -> None:
    """alloc-before:allocation 批次写入前睡(事务已开未写;kill → 回滚)。"""
    real = agents_module.append_items_tx

    def seam(
        conn: Any, thread_id: str, turn_id: str | None, entries: Sequence[ItemEntry]
    ) -> Any:
        if any(entry.kind == ItemKind.CHILD_ALLOCATION for entry in entries):
            time.sleep(_HOLD_SECONDS)  # 写者线程内、COMMIT 前:kill 即整事务回滚
        return real(conn, thread_id, turn_id, entries)

    agents_module.append_items_tx = seam


def _install_post_commit_seam(kind: str) -> None:
    """alloc-after / extend-after:Store.transact 提交后检测到新增 kind 即睡。"""
    real = Store.transact

    def seam(self: Store, fn: Any) -> Any:
        before = _count_kind(self, kind)
        result = real(self, fn)
        if _count_kind(self, kind) > before:
            time.sleep(_HOLD_SECONDS)  # 已 COMMIT;driver/续跑尚未推进
        return result

    Store.transact = seam  # type: ignore[method-assign]


def _install_child_dispatch_before_seam(parent_thread_id: str) -> None:
    """dispatch-before:首个非父 thread 的 execute(子首 dispatch)前睡。"""
    state = {"held": False}

    async def seam(call: Any, ctx: Any) -> Any:
        if not state["held"] and ctx.thread_id != parent_thread_id:
            state["held"] = True
            await asyncio.sleep(_HOLD_SECONDS)  # 子 model_response 已落、tool_call 未写
        return await _real_execute(call, ctx)

    turn_module.execute = seam


def _install_slow_write_seam() -> None:
    """dispatch-after:write.run 先睡再真实写(父表与子重构共用工厂查找处)。"""
    real_create = composition_module.create_write_tool

    def seam(root: Path) -> Tool:
        tool = real_create(root)
        real_run = tool.run

        async def slow_run(arguments: Mapping[str, JsonValue], exec_ctx: ToolExecContext) -> Any:
            await asyncio.sleep(_HOLD_SECONDS)  # tool_call 已写、文件尚未出现
            return await real_run(arguments, exec_ctx)

        return dataclasses.replace(tool, run=slow_run)

    composition_module.create_write_tool = seam


def _install_settle_seam(variant: str) -> None:
    """settle-before/after:首个终态 transition(子先终结)前/后睡。"""
    real = turn_module.transition
    state = {"held": False}
    terminal = (TurnStatus.COMPLETED, TurnStatus.FAILED, TurnStatus.CANCELLED)

    def seam(store: Any, bus: Any, turn_id: str, to: TurnStatus, *, reason: str) -> Any:
        first_terminal = not state["held"] and to in terminal
        if first_terminal and variant == "settle-before":
            state["held"] = True
            time.sleep(_HOLD_SECONDS)  # 子终态 turn_status 尚未写
        result = real(store, bus, turn_id, to, reason=reason)
        if first_terminal and variant == "settle-after":
            state["held"] = True
            time.sleep(_HOLD_SECONDS)  # 子终态已提交;父仍 running
        return result

    turn_module.transition = seam


def _install_extend_before_seam() -> None:
    """extend-before:budget_extension 批次写入前睡(turn 已 RUNNING、事务已开未写)。

    落点 = Store.append_items 对 append_items_tx 的模块查找处(公开函数,与
    alloc-before 缝同纪律);kill → 整事务回滚,extension 不落库、不凭空多预算。
    """
    real = store_log_module.append_items_tx

    def seam(
        conn: Any, thread_id: str, turn_id: str | None, entries: Sequence[ItemEntry]
    ) -> Any:
        if any(entry.kind == ItemKind.BUDGET_EXTENSION for entry in entries):
            time.sleep(_HOLD_SECONDS)
        return real(conn, thread_id, turn_id, entries)

    store_log_module.append_items_tx = seam


def _child_hold_tool(workspace: Path) -> Tool:
    """extend 变体的子保持缝:轮询 hold-marker 文件(测试永不创建),子持续 running。

    纯编排测试缝(_barrier_tool 同先例):零工作区写入;idempotent=False——
    保持中的调用被 kill/取消后按 started+非幂等走 outcome_unknown 对账。
    """

    async def run(arguments: Mapping[str, JsonValue], exec_ctx: ToolExecContext) -> ToolResult:
        marker = workspace / "hold-marker"
        deadline = time.monotonic() + 50.0
        while time.monotonic() < deadline:
            if marker.exists():
                break
            await asyncio.sleep(0.05)  # 与父 turn 共享事件循环:绝不能阻塞父推进
        return ToolResult(
            tool_call_id="pending",
            tool_name="child_hold",
            content=(ToolContentBlock(type="text", data={"text": "hold finished"}),),
            status=ResultStatus.OK,
        )

    return Tool(
        definition=ToolDefinition(
            name="child_hold",
            description="TEST SEAM: hold the child running until a marker file appears.",
            input_schema={"type": "object"},
        ),
        validate_input=lambda arguments: arguments,
        run=run,
        static_effects=frozenset(),
        resolve_use=lambda arguments: ResolvedUse(frozenset(), ()),
        idempotent=False,
        concurrency_safe=True,
        cancellation_mode=CancellationMode.COOPERATIVE,
        interrupt_behavior=InterruptBehavior.CANCEL,
        timeout_seconds=60,
        max_model_output_bytes=4_000,
    )


async def _amain_matrix(workspace: Path, scripts: Mapping[str, Any], thread_id: str, variant: str) -> int:
    if variant == "alloc-before":
        _install_alloc_before_seam()
    elif variant == "alloc-after":
        _install_post_commit_seam(ItemKind.CHILD_ALLOCATION)
    elif variant == "dispatch-before":
        _install_child_dispatch_before_seam(thread_id)
    elif variant == "dispatch-after":
        _install_slow_write_seam()
    elif variant in ("settle-before", "settle-after"):
        _install_settle_seam(variant)
    elif variant == "extend-before":
        _install_extend_before_seam()
    elif variant == "extend-after":
        _install_post_commit_seam(ItemKind.BUDGET_EXTENSION)
    holder: list[Any] = []
    router = _MatrixRouter(
        [_to_response(spec) for spec in scripts["parent"]],
        [_to_response(spec) for spec in scripts["child"]],
        holder,
        thread_id,
    )
    extra = (_child_hold_tool(workspace),) if variant.startswith("extend-") else ()
    session = open_session(
        workspace,
        router,
        thread_id=thread_id,
        enable_subagents=True,
        budget=Limits(**scripts.get("budget", {})),
        extra_tools=extra,
        policy=PermissionPolicy(network=NetworkPolicy(decision="allow")),
    )
    holder.append(session)
    try:
        result = await session.submit(f"acceptance: budget conserve fault matrix {variant}")
        if variant.startswith("extend-"):
            if result.status is not TurnStatus.PAUSED or result.reason != "budget_exhausted":
                print(json.dumps({"stage": "submit", "status": result.status.value, "reason": result.reason}))
                return 1
            # 补预算续跑 → 缝 hold → 等 SIGKILL(extend item 前/后)。
            result = await session.resume(result.turn_id, extend=dict(scripts["extend"]))
    finally:
        session.close()
    print(json.dumps({"turn_id": result.turn_id, "status": result.status.value, "reason": result.reason}))
    return 1


async def _amain(workspace: Path, specs: list[Mapping[str, Any]], thread_id: str, variant: str) -> int:
    if variant == "b":
        _install_dispatch_seam()
    if variant == "d":
        _install_second_dispatch_seam()
    tools = _slow_read_tools(workspace) if variant == "c" else None
    adapter = ScriptedAdapter([_to_response(spec) for spec in specs])
    # Task 4 迁移:真实 decide 下 bash 需 network allow(默认 ask 会第 2 步暂停);
    # Task 7:composition 已接真实 probe_backend + compile/finalize 注入(darwin seatbelt)。
    session = open_session(
        workspace,
        adapter,
        thread_id=thread_id,
        tools=tools,
        policy=PermissionPolicy(network=NetworkPolicy(decision="allow")),
    )
    try:
        result = await session.submit(f"acceptance: kill -9 resume matrix, variant {variant}")
    finally:
        session.close()
    # 走到这里说明没被 kill 住:装置失败,如实报告结局。
    print(json.dumps({"turn_id": result.turn_id, "status": result.status.value}))
    return 1


def main() -> None:
    if len(sys.argv) != 5:
        raise SystemExit(f"usage: {sys.argv[0]} <workspace> <responses.json> <thread_id> <variant>")
    workspace = Path(sys.argv[1])
    variant = sys.argv[4]
    if variant in MATRIX_VARIANTS:
        scripts = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
        exit_code = asyncio.run(_amain_matrix(workspace, scripts, sys.argv[3], variant))
    else:
        specs = _load_specs(sys.argv[2])
        exit_code = asyncio.run(_amain(workspace, specs, sys.argv[3], variant))
    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
