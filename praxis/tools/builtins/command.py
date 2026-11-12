"""bash 内置工具(移植旧 agent_runtime shell.py:653-840 的子进程/进程组语义,修正 EOF)。

执行算法(brief Step 3,修正 asyncio 的 EOF 语义):
1. ``/bin/bash -lc command``,cwd=workspace,stdout/stderr 管道,``start_new_session=True``
   (独立进程组,超时才能整组回收);
2. 读管道任务与主进程监视并发启动——边跑边读,大输出不会把管道灌满造成死锁;
3. 主进程退出的判据是 ``proc.returncode``(子进程回收事件),**不是** ``proc.wait()``:
   CPython 的 ``wait()`` 把"进程退出"与"所有管道 EOF"混为一谈,后台孙进程持写端
   时 wait() 会拖到孙进程自然退出——这正是 B20 形态要修正的点;
4. ``wait_for`` 主进程退出;超时 → SIGTERM 整组 → 5s 宽限 → SIGKILL 整组;
5. 主进程退出后,每个管道只再给 0.5s 收尾读取(读到多少算多少);若 wait() 仍未完成
   (孙进程持管道 fd),整组 SIGTERM/SIGKILL 收敛后收尸——后台孙进程不得绑架结果;
6. 非零退出码不是 is_error(命令确实执行了),exit_code 放 metadata。

取消(MANAGED_PROCESS/CANCEL):外层 cancel 时向进程组发 SIGTERM,并用
``call_later`` 挂一个宽限后的 SIGKILL,然后透传 CancelledError。

ToolResult 占位符约定同 filesystem.py:``tool_call_id="pending"``,管道替换。
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import signal
from collections.abc import Mapping
from pathlib import Path
from typing import cast

from praxis.protocol.json import JsonValue
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
_DRAIN_DEADLINE_SECONDS = 0.5
_MAX_OUTPUT_BYTES = 30_000


def _signal_group(pgid: int, sig: signal.Signals) -> None:
    """start_new_session=True ⇒ 子进程组 id == 主进程 pid;整组收发信号。"""
    with contextlib.suppress(ProcessLookupError, PermissionError):
        os.killpg(pgid, sig)


async def _await_main_exit(proc: asyncio.subprocess.Process, timeout: float | None) -> int:
    """等主进程"退出"这一事件(returncode 由子进程回收线程置位),不等管道 EOF。"""
    async def poll() -> int:
        while proc.returncode is None:
            await asyncio.sleep(0.02)
        return proc.returncode

    return await asyncio.wait_for(poll(), timeout)


async def _drain_to(stream: asyncio.StreamReader, buffer: list[bytes]) -> None:
    """读到 EOF;已收字节留在 buffer,任务被取消也不丢。"""
    while True:
        chunk = await stream.read(65536)
        if not chunk:
            return
        buffer.append(chunk)


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

    async def run(arguments: Mapping[str, JsonValue]) -> ToolResult:
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
        out_task = asyncio.create_task(_drain_to(proc.stdout, out_buf))
        err_task = asyncio.create_task(_drain_to(proc.stderr, err_buf))
        wait_task = asyncio.create_task(proc.wait())
        timed_out = False
        try:
            await _await_main_exit(proc, timeout)
        except TimeoutError:
            timed_out = True
            await _reap_main_exit(proc)
        except asyncio.CancelledError:
            _signal_group(proc.pid, signal.SIGTERM)
            loop = asyncio.get_running_loop()
            loop.call_later(_TERM_GRACE_SECONDS, _signal_group, proc.pid, signal.SIGKILL)
            out_task.cancel()
            err_task.cancel()
            wait_task.cancel()
            raise
        await _finish_drain(out_task)
        await _finish_drain(err_task)
        if not wait_task.done():
            # 孙进程仍持管道 fd:整组收敛后再收尸,确保 transport 干净关闭。
            _signal_group(proc.pid, signal.SIGTERM)
            try:
                await asyncio.wait_for(asyncio.shield(wait_task), _TERM_GRACE_SECONDS)
            except TimeoutError:
                _signal_group(proc.pid, signal.SIGKILL)
                await asyncio.shield(wait_task)

        stdout = b"".join(out_buf).decode("utf-8", errors="replace")
        stderr = b"".join(err_buf).decode("utf-8", errors="replace")
        text = stdout if not stderr else (f"{stdout}\n{stderr}" if stdout else stderr)
        truncated = False
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
