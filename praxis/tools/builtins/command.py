"""bash 内置工具(移植旧 agent_runtime shell.py:653-840 的子进程/进程组语义,修正 EOF)。

执行算法(brief Step 3,修正 asyncio 的 EOF 语义):
1. ``/bin/bash -lc command``,cwd=workspace,stdout/stderr 管道,``start_new_session=True``
   (独立进程组,超时才能整组回收);
2. 读管道任务与主进程监视并发启动——边跑边读,大输出不会把管道灌满造成死锁;
3. 主进程退出的判据是 ``proc.returncode``(子进程回收事件),**不是** ``proc.wait()``:
   CPython 的 ``wait()`` 把"进程退出"与"所有管道 EOF"混为一谈,后台孙进程持写端
   时 wait() 会拖到孙进程自然退出——这正是 B20 形态要修正的点;
4. ``wait_for`` 主进程退出;超时 → SIGTERM 整组 → 5s 宽限 → SIGKILL 整组;
5. 主进程退出后,每个管道只再给 0.5s 收尾读取(读到多少算多少);若 wait() 仍未完成,
   整组 SIGTERM(组已空则跳过宽限)→ SIGKILL → 2s 短探;仍等不到 EOF
   (setsid 逃逸到组外的孙进程持管道 fd)→ 取消挂起任务直接返回——逃逸进程
   Phase 1 不追杀(manage_command 清理是 Phase 3),subprocess transport 由引用
   消亡时(GC,循环尚存活)惰性关闭。
6. 每路输出缓冲有硬上限(``_MAX_STREAM_BUFFER_BYTES``,对齐旧
   ``_MAX_STREAM_BYTES=16_000``):超限字节继续读但丢弃——不能停止读取,否则管道
   灌满反而阻塞子进程;超限计入 truncated。
7. 非零退出码不是 is_error(命令确实执行了),exit_code 放 metadata。

取消(MANAGED_PROCESS/CANCEL):run() 的**所有** await 阶段共用一个
CancelledError 出口——SIGTERM 整组 → 有界退出宽限 → SIGKILL 整组(宽限期间被
二次 cancel 立即升级 KILL)→ 取消并回收三个任务 → 透传 CancelledError。
不用 ``call_later`` 延迟致命信号:事件循环可能先于定时器关闭。

ToolResult 占位符约定同 filesystem.py:``tool_call_id="pending"``,管道替换。
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import signal
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

from praxis.protocol.json import JsonValue
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
    json_schema_input,
)
from praxis.tools.builtins.filesystem import PENDING_TOOL_CALL_ID

_DEFAULT_TIMEOUT_SECONDS = 120.0
_MAX_TIMEOUT_SECONDS = 600.0
_TERM_GRACE_SECONDS = 5.0
_POST_KILL_WAIT_SECONDS = 2.0
_DRAIN_DEADLINE_SECONDS = 0.5
_MAX_OUTPUT_BYTES = 30_000
_MAX_STREAM_BUFFER_BYTES = 16_000


def _signal_group(pgid: int, sig: signal.Signals) -> bool:
    """start_new_session=True ⇒ 子进程组 id == 主进程 pid;整组收发信号。

    返回"组内可能仍有活体":False 仅当 ESRCH(组已空,持管道者必在组外)。
    """
    try:
        os.killpg(pgid, sig)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


async def _await_main_exit(proc: asyncio.subprocess.Process, timeout: float | None) -> int:
    """等主进程"退出"这一事件(returncode 由子进程回收线程置位),不等管道 EOF。"""
    async def poll() -> int:
        while proc.returncode is None:
            await asyncio.sleep(0.02)
        return proc.returncode

    return await asyncio.wait_for(poll(), timeout)


async def _drain_to(stream: asyncio.StreamReader, buffer: list[bytes], discarded: list[int]) -> None:
    """读到 EOF;已收字节留在 buffer,任务被取消也不丢。

    buffer 至多 ``_MAX_STREAM_BUFFER_BYTES``;超限字节仍继续读但只计入 discarded
    ——停止读取会让管道灌满、阻塞子进程,那是另一种绑架。
    """
    buffered = 0
    while True:
        chunk = await stream.read(65536)
        if not chunk:
            return
        room = _MAX_STREAM_BUFFER_BYTES - buffered
        if room >= len(chunk):
            buffer.append(chunk)
        elif room > 0:
            buffer.append(chunk[:room])
            discarded[0] += len(chunk) - room
        else:
            discarded[0] += len(chunk)
        buffered += len(chunk)


async def _finish_drain(task: asyncio.Task[None]) -> None:
    """主进程已退出:再给 0.5s 收尾,超时即取消——后台孙进程不得绑架 EOF(B20)。"""
    if task.done():
        return
    try:
        await asyncio.wait_for(asyncio.shield(task), _DRAIN_DEADLINE_SECONDS)
    except TimeoutError:
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task
    except asyncio.CancelledError:
        task.cancel()
        raise


async def _reap_main_exit(proc: asyncio.subprocess.Process) -> None:
    """超时收敛:SIGTERM 整组 → 5s 宽限 → SIGKILL 整组 → 等主进程回收。"""
    _signal_group(proc.pid, signal.SIGTERM)
    try:
        await _await_main_exit(proc, _TERM_GRACE_SECONDS)
    except TimeoutError:
        _signal_group(proc.pid, signal.SIGKILL)
        await _await_main_exit(proc, None)


async def _abandon_wait(wait_task: asyncio.Task[int]) -> None:
    """EOF 被组外逃逸进程绑架(Finding 1):run() 不陪孙进程寿命。

    取消挂起的 wait 任务即返回;transport 在 proc 引用消亡时(GC)关闭——
    此时事件循环仍在运行,关闭路径干净。setsid 逃逸的进程本身 Phase 1 不追杀。
    """
    wait_task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await wait_task


async def _converge_pipes(proc: asyncio.subprocess.Process, wait_task: asyncio.Task[int]) -> None:
    """主进程已退出但 wait() 未完成:有进程还持着管道写端。

    先 SIGTERM 整组(组内残留随信号退出、管道 EOF);组已空(ESRCH,持管道者
    必为 setsid 逃逸的组外进程)则跳过 TERM 宽限,只做 2s 短探;短探落空
    → 放弃等待(逃逸孙进程 Phase 1 不追杀)。
    """
    signaled = _signal_group(proc.pid, signal.SIGTERM)
    killed = False
    window = _TERM_GRACE_SECONDS if signaled else _POST_KILL_WAIT_SECONDS
    while True:
        try:
            await asyncio.wait_for(asyncio.shield(wait_task), window)
            return
        except TimeoutError:
            pass
        if signaled and not killed:
            killed = True
            _signal_group(proc.pid, signal.SIGKILL)
            window = _POST_KILL_WAIT_SECONDS
            continue
        await _abandon_wait(wait_task)
        return


async def _cancel_cleanup(proc: asyncio.subprocess.Process, tasks: tuple[asyncio.Task[Any], ...]) -> None:
    """任何阶段被取消的统一收敛(Finding 3):不留悬空任务/孤儿进程组。"""
    killed = False
    try:
        _signal_group(proc.pid, signal.SIGTERM)
        try:
            await _await_main_exit(proc, _TERM_GRACE_SECONDS)
        except TimeoutError:
            pass
        _signal_group(proc.pid, signal.SIGKILL)
        killed = True
        try:
            await _await_main_exit(proc, _POST_KILL_WAIT_SECONDS)
        except TimeoutError:
            pass
    except asyncio.CancelledError:
        if not killed:
            _signal_group(proc.pid, signal.SIGKILL)
        raise
    finally:
        for task in tasks:
            task.cancel()
        for task in tasks:
            with contextlib.suppress(BaseException):
                await task


def create_bash_tool(workspace: Path) -> Tool:
    validate_input = json_schema_input(
        {
            "type": "object",
            "properties": {
                "command": {"type": "string", "minLength": 1},
                "timeout_seconds": {
                    "type": "number",
                    "minimum": 1,
                    "maximum": _MAX_TIMEOUT_SECONDS,
                },
            },
            "required": ("command",),
            "additionalProperties": False,
        }
    )

    async def run(
        arguments: Mapping[str, JsonValue],
        exec_ctx: ToolExecContext,  # 沙箱接线(Task 7)在此消费 exec_ctx.sandbox;本任务忽略
    ) -> ToolResult:
        args = validate_input(arguments)
        command = cast("str", args["command"])
        timeout = cast("float", args.get("timeout_seconds", _DEFAULT_TIMEOUT_SECONDS))
        proc = await asyncio.create_subprocess_exec(
            "/bin/bash",
            "-lc",
            command,
            cwd=str(workspace),
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            start_new_session=True,
        )
        if proc.stdout is None or proc.stderr is None:
            raise RuntimeError("bash process pipes were not created")
        out_buf: list[bytes] = []
        err_buf: list[bytes] = []
        out_discarded: list[int] = [0]
        err_discarded: list[int] = [0]
        out_task = asyncio.create_task(_drain_to(proc.stdout, out_buf, out_discarded))
        err_task = asyncio.create_task(_drain_to(proc.stderr, err_buf, err_discarded))
        wait_task = asyncio.create_task(proc.wait())
        tasks = (out_task, err_task, wait_task)
        timed_out = False
        try:
            try:
                await _await_main_exit(proc, timeout)
            except TimeoutError:
                timed_out = True
                await _reap_main_exit(proc)
            await _finish_drain(out_task)
            await _finish_drain(err_task)
            if not wait_task.done():
                await _converge_pipes(proc, wait_task)
        except asyncio.CancelledError:
            await _cancel_cleanup(proc, tasks)
            raise

        stdout = b"".join(out_buf).decode("utf-8", errors="replace")
        stderr = b"".join(err_buf).decode("utf-8", errors="replace")
        text = stdout if not stderr else (f"{stdout}\n{stderr}" if stdout else stderr)
        truncated = out_discarded[0] > 0 or err_discarded[0] > 0
        encoded = text.encode("utf-8")
        if len(encoded) > _MAX_OUTPUT_BYTES:
            text = encoded[:_MAX_OUTPUT_BYTES].decode("utf-8", errors="ignore")
            truncated = True
        error_message = (
            "command timed out and its process group was terminated" if timed_out else None
        )
        return ToolResult(
            tool_call_id=PENDING_TOOL_CALL_ID,
            tool_name="bash",
            content=(ToolContentBlock(type="text", data={"text": text}),),
            status=ResultStatus.TIMEOUT if timed_out else ResultStatus.OK,
            is_error=timed_out,
            error_code="timeout" if timed_out else None,
            error_message=error_message,
            truncated=truncated,
            metadata={
                "exit_code": proc.returncode,
                "timed_out": timed_out,
            },
        )

    def resolve_use(arguments: Mapping[str, JsonValue]) -> ResolvedUse:
        return ResolvedUse(
            effects=frozenset({ToolEffect.EXECUTE_PROCESS}),
            targets=(ToolTarget(kind="path", value=str(workspace.resolve())),),
        )

    return Tool(
        definition=ToolDefinition(
            name="bash",
            description=(
                "Run a shell command with /bin/bash -lc in the workspace; "
                "returns combined stdout/stderr and exit_code."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "Shell command text."},
                    "timeout_seconds": {
                        "type": "number",
                        "description": (
                            f"Wall-clock timeout, 1..{_MAX_TIMEOUT_SECONDS:.0f} "
                            f"(default {_DEFAULT_TIMEOUT_SECONDS:.0f}); the whole process "
                            "group is killed on timeout."
                        ),
                    },
                },
                "required": ("command",),
                "additionalProperties": False,
            },
        ),
        validate_input=validate_input,
        run=run,
        static_effects=frozenset({ToolEffect.EXECUTE_PROCESS}),
        resolve_use=resolve_use,
        idempotent=False,
        concurrency_safe=False,
        cancellation_mode=CancellationMode.MANAGED_PROCESS,
        interrupt_behavior=InterruptBehavior.CANCEL,
        timeout_seconds=605,
        max_model_output_bytes=_MAX_OUTPUT_BYTES,
    )
