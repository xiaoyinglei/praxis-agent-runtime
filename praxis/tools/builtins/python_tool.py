"""run_python 内置工具(移植旧 ManagedPythonInput/Output 语义,Task 16)。

契约:
- 输入 ``code``(1..100_000) XOR ``script_path``(1..4096),二选一;
  ``working_dir`` 默认 "."、必须工作区内已存在目录;``timeout_seconds``
  默认 120、0 < x <= 600(**不接受 null**);
- 执行 ``[*spec.argv_prefix, sys.executable, "-I", <脚本路径>]``:``-I``
  隔离宿主 ``PYTHONPATH``/sitecustomize(与 search worker 同纪律);
  ``cwd=working_dir``、``start_new_session=True``;超时/取消对整组
  SIGTERM → 宽限 → SIGKILL(只组合 ``process.py`` 公开原语);
- R-T16-3 脚本落点:``code`` 模式写进沙箱 ``scratch_dir``(唯一可写区)
  的私有临时文件并在结束后删除;``script_path`` 经 ``checked_path`` 限制在
  工作区内。两者都不经过全局/可预测的世界可写路径;
- 语法错误在**执行前**用 ``compile(code, "<run_python>", "exec")`` 拒绝
  (``invalid_input``),不把 SyntaxError 当运行失败;
- 输出 stdout/stderr 各自 ≤16_000 头尾保留,截断标记逐字沿用旧文案;
  metadata = ``{exit_code, timed_out, duration_ms, python_version}``;
  非零退出 → ``python_failed``;超时 → ``TIMEOUT/timeout_cancelled``;
- 写授权与网络一律由 policy/decide 决定:本工具**不接收**
  ``workspace_write``/``network`` 布尔旗标(计划 D1 的有意差异,写进描述);
- 子进程环境由 ``child_env`` 唯一构造:宿主环境变量与凭据不可见(R-T16-2)。

本任务不把 ``run_python`` 注册进默认工具表(Task 17 负责)。
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import platform
import signal
import sys
import tempfile
import time
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import cast

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
    ToolValidationError,
    json_schema_input,
)
from praxis.tools.builtins.child_env import child_env, scratch_dir_for
from praxis.tools.builtins.filesystem import PENDING_TOOL_CALL_ID
from praxis.tools.builtins.paths import checked_path
from praxis.tools.builtins.process import await_main_exit, signal_group

_DEFAULT_TIMEOUT_SECONDS = 120.0
_MAX_TIMEOUT_SECONDS = 600.0
_MAX_CODE_CHARS = 100_000
_MAX_PATH_CHARS = 4_096
# 与 command.py 的 _MAX_STREAM_BUFFER_BYTES 同一上限;不跨模块借用私有名。
_MAX_STREAM_BUFFER_BYTES = 16_000
_TRUNCATION_MARKER = b"\n... output truncated; preserved head and tail ...\n"
_TERM_GRACE_SECONDS = 5.0
_POST_KILL_WAIT_SECONDS = 2.0
_DRAIN_DEADLINE_SECONDS = 0.5
_READ_CHUNK_BYTES = 65_536


@dataclass(frozen=True, slots=True)
class _Outcome:
    """一次 Python 执行的终态(两路文本已各自截断)。"""

    stdout: str
    stderr: str
    truncated: bool
    timed_out: bool
    exit_code: int | None
    duration_ms: float


class _HeadTailBuffer:
    """有界头 + 滚动尾(移植旧 _CommandStreamBuffer 的保留语义)。"""

    def __init__(self) -> None:
        self.data = bytearray()

    def append(self, chunk: bytes) -> None:
        limit = _MAX_STREAM_BUFFER_BYTES + 1
        self.data.extend(chunk)
        if len(self.data) > limit:
            head = limit // 2
            del self.data[head : len(self.data) - (limit - head)]

    def value(self) -> bytes:
        return bytes(self.data)


def _bounded_stream(value: bytes) -> tuple[str, bool]:
    """≤16_000 字节;超限保留头尾并插入旧截断标记(逐字一致)。"""
    truncated = len(value) > _MAX_STREAM_BUFFER_BYTES
    if not truncated:
        return value.decode("utf-8", errors="replace"), False
    available = _MAX_STREAM_BUFFER_BYTES - len(_TRUNCATION_MARKER)
    head_size = available // 2
    tail_size = available - head_size
    bounded = value[:head_size] + _TRUNCATION_MARKER + value[-tail_size:]
    return bounded.decode("utf-8", errors="replace"), True


async def _read_stream(stream: asyncio.StreamReader, buffer: _HeadTailBuffer) -> None:
    while True:
        chunk = await stream.read(_READ_CHUNK_BYTES)
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


async def _terminate_group(proc: asyncio.subprocess.Process) -> None:
    """超时收敛:SIGTERM 整组 → 宽限 → SIGKILL 整组 → 有界回收。"""
    signal_group(proc.pid, signal.SIGTERM)
    try:
        await await_main_exit(proc, _TERM_GRACE_SECONDS)
    except TimeoutError:
        signal_group(proc.pid, signal.SIGKILL)
        with contextlib.suppress(TimeoutError):
            await await_main_exit(proc, _POST_KILL_WAIT_SECONDS)


async def _cancel_execution(
    proc: asyncio.subprocess.Process, tasks: tuple[asyncio.Task[None], ...]
) -> None:
    """取消的统一收敛:TERM 整组 → 宽限 → KILL 整组 → 取消并回收读任务。"""
    killed = False
    try:
        signal_group(proc.pid, signal.SIGTERM)
        try:
            await await_main_exit(proc, _TERM_GRACE_SECONDS)
        except TimeoutError:
            pass
        signal_group(proc.pid, signal.SIGKILL)
        killed = True
        with contextlib.suppress(TimeoutError):
            await await_main_exit(proc, _POST_KILL_WAIT_SECONDS)
    except asyncio.CancelledError:
        if not killed:
            signal_group(proc.pid, signal.SIGKILL)
        raise
    finally:
        for task in tasks:
            task.cancel()
        for task in tasks:
            with contextlib.suppress(BaseException):
                await task


async def _execute(
    *,
    argv: tuple[str, ...],
    cwd: Path,
    env: Mapping[str, str],
    timeout: float,
) -> _Outcome:
    """spawn → 等主进程退出/超时收敛 → 收尾读取;取消路径不留孤儿进程组。

    spawn 半路失败(管道不变量或 create_task 失败)复用 ``CommandExecution.spawn``
    的同一收尸纪律:整组 SIGKILL → 取消已建任务 → 有界回收主进程 → 透传异常。
    不直接复用 ``CommandExecution`` 是因为它的 ``drain_stream`` 只保留头部
    (超出即丢弃),而 run_python 必须头尾都保留(``_HeadTailBuffer``)。
    """
    started = time.monotonic()
    proc = await asyncio.create_subprocess_exec(
        *argv,
        cwd=str(cwd),
        env=env,
        stdin=asyncio.subprocess.DEVNULL,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        start_new_session=True,
    )
    out_buffer = _HeadTailBuffer()
    err_buffer = _HeadTailBuffer()
    tasks: list[asyncio.Task[None]] = []
    try:
        if proc.stdout is None or proc.stderr is None:
            raise RuntimeError("run_python process pipes were not created")
        tasks.append(asyncio.create_task(_read_stream(proc.stdout, out_buffer)))
        tasks.append(asyncio.create_task(_read_stream(proc.stderr, err_buffer)))
    except BaseException:
        signal_group(proc.pid, signal.SIGKILL)
        for task in tasks:
            task.cancel()
        for task in tasks:
            with contextlib.suppress(BaseException):
                await task
        with contextlib.suppress(TimeoutError):
            await await_main_exit(proc, _POST_KILL_WAIT_SECONDS)
        raise
    timed_out = False
    try:
        try:
            await await_main_exit(proc, timeout)
        except TimeoutError:
            timed_out = True
            await _terminate_group(proc)
        await _finish_drain(tasks[0])
        await _finish_drain(tasks[1])
    except asyncio.CancelledError:
        await _cancel_execution(proc, tuple(tasks))
        raise
    out_text, out_truncated = _bounded_stream(out_buffer.value())
    err_text, err_truncated = _bounded_stream(err_buffer.value())
    return _Outcome(
        stdout=out_text,
        stderr=err_text,
        truncated=out_truncated or err_truncated,
        timed_out=timed_out,
        exit_code=proc.returncode,
        duration_ms=(time.monotonic() - started) * 1000,
    )


def _terminal_result(outcome: _Outcome) -> ToolResult:
    text = (
        outcome.stdout
        if not outcome.stderr
        else (f"{outcome.stdout}\n{outcome.stderr}" if outcome.stdout else outcome.stderr)
    )
    if outcome.timed_out:
        status = ResultStatus.TIMEOUT
        error_code: str | None = "timeout_cancelled"
        error_message: str | None = "Python code timed out and its process group was terminated"
    elif outcome.exit_code != 0:
        status = ResultStatus.ERROR
        error_code = "python_failed"
        error_message = (
            f"Python code exited with status {outcome.exit_code}; inspect stdout/stderr and "
            "change the code rather than retrying unchanged"
        )
    else:
        status = ResultStatus.OK
        error_code = None
        error_message = None
    return ToolResult(
        tool_call_id=PENDING_TOOL_CALL_ID,
        tool_name="run_python",
        content=(ToolContentBlock(type="text", data={"text": text}),),
        status=status,
        is_error=status is not ResultStatus.OK,
        error_code=error_code,
        error_message=error_message,
        truncated=outcome.truncated,
        metadata={
            "exit_code": outcome.exit_code,
            "timed_out": outcome.timed_out,
            "duration_ms": outcome.duration_ms,
            "python_version": platform.python_version(),
        },
    )


def _error_result(*, code: str, message: str) -> ToolResult:
    return ToolResult(
        tool_call_id=PENDING_TOOL_CALL_ID,
        tool_name="run_python",
        content=(ToolContentBlock(type="text", data={"text": f"{code}: {message}"}),),
        status=ResultStatus.ERROR,
        is_error=True,
        error_code=code,
        error_message=message,
    )


def _working_dir(workspace: Path, arguments: Mapping[str, JsonValue]) -> Path:
    raw = arguments.get("working_dir", ".")
    value = raw if isinstance(raw, str) and raw else "."
    target = checked_path(workspace, value, allow_root=True)
    if not target.is_dir():
        raise ToolValidationError(
            path="$.working_dir",
            message=f"working_dir must be an existing directory inside the workspace: {value!r}",
        )
    return target


def _syntax_message(exc: SyntaxError | ValueError) -> str:
    if isinstance(exc, SyntaxError):
        return f"Python syntax error at line {exc.lineno}: {exc.msg}"
    return f"Python source is not valid: {exc}"


def _write_temp_script(scratch: Path, code: str) -> Path:
    fd, name = tempfile.mkstemp(prefix="praxis-run-python-", suffix=".py", dir=scratch)
    with os.fdopen(fd, "w", encoding="utf-8") as stream:
        stream.write(code)
    return Path(name)


def create_run_python_tool(workspace: Path) -> Tool:
    validate_schema = json_schema_input(
        {
            "type": "object",
            "properties": {
                "code": {"type": "string", "minLength": 1, "maxLength": _MAX_CODE_CHARS},
                "script_path": {"type": "string", "minLength": 1, "maxLength": _MAX_PATH_CHARS},
                "working_dir": {"type": "string", "maxLength": _MAX_PATH_CHARS},
                "timeout_seconds": {
                    "type": "number",
                    "exclusiveMinimum": 0,
                    "maximum": _MAX_TIMEOUT_SECONDS,
                },
            },
            "additionalProperties": False,
        }
    )

    def validate_input(arguments: Mapping[str, JsonValue]) -> Mapping[str, JsonValue]:
        args = validate_schema(arguments)
        has_code = "code" in args
        has_script = "script_path" in args
        if has_code == has_script:
            raise ToolValidationError(
                path="$.code",
                message="exactly one of code/script_path is required",
            )
        return args

    async def run(
        arguments: Mapping[str, JsonValue],
        exec_ctx: ToolExecContext,
    ) -> ToolResult:
        args = validate_input(arguments)
        try:
            working_dir = _working_dir(workspace, args)
        except ToolValidationError as exc:
            return _error_result(code="invalid_input", message=exc.message)
        timeout = cast("float", args.get("timeout_seconds", _DEFAULT_TIMEOUT_SECONDS))
        spec = exec_ctx.sandbox
        scratch = scratch_dir_for(spec)
        code = cast("str | None", args.get("code"))
        temp_script: Path | None = None
        if code is not None:
            try:
                compile(code, "<run_python>", "exec")
            except (SyntaxError, ValueError) as exc:
                return _error_result(code="invalid_input", message=_syntax_message(exc))
            temp_script = _write_temp_script(scratch, code)
            script = temp_script
        else:
            try:
                script = checked_path(workspace, cast("str", args["script_path"]))
            except ToolValidationError as exc:
                return _error_result(code="invalid_input", message=exc.message)
            if not script.is_file():
                return _error_result(
                    code="invalid_input",
                    message=f"script_path is not a file inside the workspace: {script}",
                )
        prefix = spec.argv_prefix if spec is not None else ()
        try:
            outcome = await _execute(
                argv=(*prefix, sys.executable, "-I", str(script)),
                cwd=working_dir,
                env=child_env(scratch, workspace),
                timeout=timeout,
            )
        finally:
            if temp_script is not None:
                temp_script.unlink(missing_ok=True)
        return _terminal_result(outcome)

    def resolve_use(arguments: Mapping[str, JsonValue]) -> ResolvedUse:
        return ResolvedUse(
            effects=frozenset({ToolEffect.READ_WORKSPACE, ToolEffect.EXECUTE_PROCESS}),
            targets=(ToolTarget(kind="path", value=str(_working_dir(workspace, arguments))),),
        )

    return Tool(
        definition=ToolDefinition(
            name="run_python",
            description=(
                "Run Python code with the interpreter that is running Praxis "
                "(sys.executable), isolated with -I from the host PYTHONPATH and "
                "sitecustomize, and return combined stdout/stderr plus exit_code. "
                "Supply exactly one of code (run as a temporary script inside the "
                "sandbox scratch area) or script_path (a workspace-relative Python "
                "file). working_dir must be a directory inside the workspace. The "
                "subprocess receives only a minimal environment: host environment "
                "variables and credentials are not visible to it. Syntax errors are "
                "rejected before execution. Write access and network are decided by "
                "the permission policy; this tool has no workspace_write or network "
                "flags, so permissions cannot be widened from tool arguments. Use "
                "the read/write tools for file edits."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": (
                            "Python source executed as a temporary script. Pass plain "
                            "code, not a shell command or heredoc."
                        ),
                    },
                    "script_path": {
                        "type": "string",
                        "description": (
                            "Workspace-relative Python entry script. Supply exactly one "
                            "of code or script_path."
                        ),
                    },
                    "working_dir": {
                        "type": "string",
                        "description": (
                            "Workspace-relative working directory for the process "
                            "(default '.'); must exist."
                        ),
                    },
                    "timeout_seconds": {
                        "type": "number",
                        "description": (
                            f"Wall-clock timeout, 0 < x <= {_MAX_TIMEOUT_SECONDS:.0f} "
                            f"(default {_DEFAULT_TIMEOUT_SECONDS:.0f}); the whole process "
                            "group is killed on timeout."
                        ),
                    },
                },
                "additionalProperties": False,
            },
        ),
        validate_input=validate_input,
        run=run,
        static_effects=frozenset({ToolEffect.READ_WORKSPACE, ToolEffect.EXECUTE_PROCESS}),
        resolve_use=resolve_use,
        idempotent=False,
        concurrency_safe=False,
        cancellation_mode=CancellationMode.MANAGED_PROCESS,
        interrupt_behavior=InterruptBehavior.CANCEL,
        timeout_seconds=605,
        max_model_output_bytes=150_000,
    )
