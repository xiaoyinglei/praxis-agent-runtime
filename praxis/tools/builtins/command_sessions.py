"""后台命令会话(设计 §5.4;findings B20;裁决 D13)。

职责边界(刻意保持最小,durable worker 归阶段 5):
- ``CommandSessions``:进程/任务/容量/状态的所有权。同一会话最多
  ``_MAX_RUNNING`` 个 running(``command_session_busy``),最多保留
  ``_MAX_RETAINED`` 个**已取回终态**的会话(未取回终态永不驱逐;满了且无可
  驱逐者 → ``command_session_capacity``)。全部状态转换都在 ``threading.Lock`` 内。
- marker(崩溃清扫凭据):``command_jobs_dir()/<session_id>.json``(0600,目录
  0700),字段 ``{session_id,host_pid,pid,pgid,command_prefix,created_at}``。
  **只由宿主 Python 进程写**——该目录在安全地板私有集合内,沙箱内的子进程无
  写权限,这是设计而非缺陷。``host_pid`` 让清扫能区分"另一个活会话"与"崩溃
  遗留"(R-T15-6)。目录与文件都拒绝 symlink、校验属主/权限/普通文件
  (与 ``adapters/config/credentials.py`` 同一纪律,R-T15-9);会话进入终态并
  被取回后 ``unlink``。
- 执行驱动完全复用 ``command.CommandExecution``(spawn/wait_terminal/cancel),
  本模块不出现第二份 killpg 或 TERM→KILL 序列。

取消语义(R-T15-4):``start`` 的 yield 等待与 ``wait`` 用 ``asyncio.wait``——
调用方被 asyncio 取消时**不会**连带取消正在跑的命令;终止命令是 ``cancel``
动作的职责。``aclose`` 幂等,取消所有在跑会话(不留活口)。

错误码(公开口径):``command_session_busy`` / ``command_session_not_found`` /
``command_session_capacity`` / ``command_session_lost``(登记过但本进程没有
终态)/ ``command_session_unavailable``(注册表已关闭或 marker 无法安全登记)。

终态词汇统一为 ``running/completed/cancelled/lost``;bash 终态 metadata 的
``state`` 同样用 ``completed``(不再另说 ``exited``,R-T15-10)。

D13:后台任务的事实是 bash 返回的那条 ``tool_result`` 的 metadata
(``session_id/pid/pgid/state``),不新增 ItemKind,不建持久 worker。
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import secrets
import signal
import stat
import threading
import time
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast

from praxis.protocol.floor import command_jobs_dir
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
from praxis.tools.builtins.command import CommandExecution, CommandSessionError
from praxis.tools.builtins.filesystem import PENDING_TOOL_CALL_ID
from praxis.tools.builtins.process import signal_group

SessionStatus = Literal["running", "completed", "cancelled", "lost"]

_MAX_RUNNING = 1
_MAX_RETAINED = 32
_MAX_COMMAND_PREFIX_CHARS = 200
_JOBS_DIR_MODE = 0o700
_MARKER_MODE = 0o600
_MARKER_MAX_BYTES = 4096
_MAX_READ_BYTES = 64_000
_SESSION_ID_PATTERN = re.compile(r"cmd_[0-9a-f]{32}")

__all__ = [
    "CommandLogChunk",
    "CommandSessionError",
    "CommandSessionHandle",
    "CommandSessionView",
    "CommandSessions",
    "create_manage_command_tool",
]


@dataclass(frozen=True, slots=True)
class CommandSessionView:
    """一次会话查询的不可变快照(stdout_head/stderr_head 是缓冲头,不是增量)。"""

    session_id: str
    pid: int
    pgid: int
    status: SessionStatus
    exit_code: int | None
    stdout_head: str
    stderr_head: str
    truncated: bool


@dataclass(frozen=True, slots=True)
class CommandSessionHandle:
    """``start`` 的返回:窗口内退出时带终态输出,否则 ``status="running"``。"""

    session_id: str
    pid: int
    pgid: int
    status: SessionStatus
    exit_code: int | None
    stdout: str
    stderr: str
    truncated: bool
    timed_out: bool


@dataclass(frozen=True, slots=True)
class CommandLogChunk:
    """``read`` 的字节窗口(内存缓冲头;durable 日志续读归阶段 5)。"""

    data: str
    next_offset: int
    end_offset: int
    truncated: bool


@dataclass
class _CommandSession:
    session_id: str
    pid: int
    pgid: int
    started_at: float
    execution: CommandExecution
    marker: Path | None
    task: asyncio.Task[None] | None = None
    status: SessionStatus = "running"
    exit_code: int | None = None
    timed_out: bool = False
    observed: bool = False


def _bounded_command_prefix(command: str) -> str:
    """marker 的命令前缀:模型命令文本折叠空白后截到 200 字符。

    记录的是命令文本而不是完整 argv:bash 对简单命令会 tail-exec(ps 的 args
    只剩 ``sleep 20``),sandbox-exec 则保留完整 argv——两者共同点是命令文本
    作为 argv 尾部元素出现,所以清扫侧按"空格边界包含"匹配(见 recovery)。
    """
    return " ".join(command.split())[:_MAX_COMMAND_PREFIX_CHARS]


def _decode(data: bytes) -> str:
    return data.decode("utf-8", errors="replace")


class CommandSessions:
    """一个工作区的后台命令会话注册表(组合一次,注入 bash 与 manage_command)。"""

    def __init__(
        self,
        workspace: Path,
        *,
        clock: Callable[[], float] = time.monotonic,
        jobs_dir: Path | None = None,
    ) -> None:
        self.workspace = Path(workspace)
        self._clock = clock
        self._jobs_dir = Path(jobs_dir) if jobs_dir is not None else command_jobs_dir()
        self._sessions: dict[str, _CommandSession] = {}
        self._lock = threading.Lock()
        self._starting = False
        self._closed = False

    # -- 启动 ----------------------------------------------------------------

    async def start(
        self,
        spawn: Callable[[], Awaitable[CommandExecution]],
        *,
        command_prefix: str,
        timeout_seconds: float,
        yield_time_seconds: float,
    ) -> CommandSessionHandle:
        """占位 → spawn → 登记 → 等 yield 窗口。busy/capacity 抛 ``CommandSessionError``。

        spawn 契约:失败时必须自行收尸(``CommandExecution.spawn`` 已实现:整组
        SIGKILL + 回收),因为本方法在 spawn 抛出时拿不到任何半成品句柄(M2)。
        """
        with self._lock:
            if self._closed:
                raise CommandSessionError(
                    "command_session_unavailable", "command sessions are closed"
                )
            if self._running_locked():
                raise CommandSessionError(
                    "command_session_busy",
                    "a command is already running; use manage_command to wait or cancel it",
                )
            self._evict_locked()
            # 占位:spawn 是 await,并发 start 不得在窗口里双开(R-T15-3)。
            self._starting = True
        try:
            execution = await spawn()
        except BaseException:
            with self._lock:
                self._starting = False
            raise

        session_id = "cmd_" + secrets.token_hex(16)
        try:
            marker = self._write_marker(
                session_id,
                os.getpid(),
                execution.pid,
                execution.pid,
                _bounded_command_prefix(command_prefix),
                time.time(),
            )
        except OSError as exc:
            # marker 写不进 = 崩溃后无法登记孤儿;宁可取消刚起的进程也不留未登记活口。
            try:
                await execution.cancel()
            finally:
                with self._lock:
                    self._starting = False
            raise CommandSessionError(
                "command_session_unavailable", f"cannot register command job marker: {exc}"
            ) from exc

        entry = _CommandSession(
            session_id=session_id,
            pid=execution.pid,
            pgid=execution.pid,
            started_at=self._clock(),
            execution=execution,
            marker=marker,
        )
        with self._lock:
            self._starting = False
            # R-T15-8:spawn/marker 期间 aclose 可能已经完成关档。绝不把活会话
            # 登记进已关闭的注册表(否则 Session.close 已返回,没人再取消它)。
            closed = self._closed
            if not closed:
                self._sessions[session_id] = entry
                entry.task = asyncio.get_running_loop().create_task(
                    self._drive(entry, execution, timeout_seconds)
                )
        if closed:
            try:
                await execution.cancel()
            finally:
                self._unlink_marker(entry)
            raise CommandSessionError(
                "command_session_unavailable", "command sessions closed while starting"
            )

        await self._observe(entry, yield_time_seconds)
        with self._lock:
            terminal = entry.status != "running"
            if terminal:
                # 窗口内退出:未暴露过 session_id,无需保留可查询句柄。
                self._sessions.pop(session_id, None)
                entry.observed = True
            handle = self._handle_locked(entry)
        if terminal:
            self._unlink_marker(entry)
        return handle

    # -- 查询/管理 ------------------------------------------------------------

    def get(self, session_id: str) -> CommandSessionView:
        with self._lock:
            entry = self._require_locked(session_id)
            terminal = entry.status != "running"
            if terminal:
                entry.observed = True
            view = self._view_locked(entry)
        if terminal:
            self._unlink_marker(entry)
        return view

    async def wait(self, session_id: str, seconds: float) -> CommandSessionView:
        """等 seconds 秒或终态;等待本身被取消**不会**取消命令(R-T15-4)。"""
        with self._lock:
            entry = self._require_locked(session_id)
        await self._observe(entry, max(0.0, seconds))
        with self._lock:
            terminal = entry.status != "running"
            if terminal:
                entry.observed = True
            view = self._view_locked(entry)
        if terminal:
            self._unlink_marker(entry)
        return view

    async def cancel(self, session_id: str) -> CommandSessionView:
        """终止并回收整个进程组 + 删 marker;对已终态会话幂等。"""
        with self._lock:
            entry = self._require_locked(session_id)
        task = entry.task
        if task is not None and not task.done():
            task.cancel()
            try:
                await asyncio.shield(task)
            except asyncio.CancelledError:
                if not task.done():
                    raise
        with self._lock:
            entry.observed = True
            view = self._view_locked(entry)
        self._unlink_marker(entry)
        return view

    def read(
        self,
        session_id: str,
        *,
        stream: Literal["stdout", "stderr"],
        offset: int,
        max_bytes: int,
    ) -> CommandLogChunk:
        """按字节窗口读内存缓冲头(≤64_000/次;不存在的区间返回空块)。"""
        if stream not in {"stdout", "stderr"}:
            raise ValueError("stream must be 'stdout' or 'stderr'")
        if offset < 0 or not 1 <= max_bytes <= _MAX_READ_BYTES:
            raise ValueError(f"invalid read range: offset={offset}, max_bytes={max_bytes}")
        with self._lock:
            entry = self._require_locked(session_id)
            raw = (
                entry.execution.stdout_bytes()
                if stream == "stdout"
                else entry.execution.stderr_bytes()
            )
            chunk = raw[offset : offset + max_bytes]
            view = CommandLogChunk(
                data=_decode(chunk),
                next_offset=offset + len(chunk),
                end_offset=len(raw),
                truncated=entry.execution.truncated,
            )
            terminal = entry.status != "running"
            if terminal:
                entry.observed = True
        if terminal:
            self._unlink_marker(entry)
        return view

    # -- 关闭 ----------------------------------------------------------------

    async def aclose(self) -> None:
        """幂等收尾:取消所有在跑会话(TERM→有界→KILL 由执行器统一收敛)并删 marker。"""
        with self._lock:
            if self._closed:
                return
            self._closed = True
            self._starting = False
            entries = list(self._sessions.values())
            self._sessions.clear()
        for entry in entries:
            entry.observed = True
            task = entry.task
            if task is not None and not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
                except Exception:  # noqa: BLE001 — 关闭路径:跨循环时退化为硬 KILL,不留活口
                    signal_group(entry.pgid, signal.SIGKILL)
            self._unlink_marker(entry)

    # -- 内部 ----------------------------------------------------------------

    async def _drive(
        self, entry: _CommandSession, execution: CommandExecution, timeout: float
    ) -> None:
        """会话所有权任务:终态写回 entry;取消路径复用执行器的统一收敛。"""
        try:
            outcome = await execution.wait_terminal(timeout)
        except asyncio.CancelledError:
            try:
                await execution.cancel()
            except asyncio.CancelledError:
                pass
            finally:
                with self._lock:
                    entry.status = "cancelled"
                    entry.exit_code = None
            raise
        except Exception:  # noqa: BLE001 — 执行器自身失败:先证明组已收敛,再记 lost
            try:
                await execution.cancel()
            except Exception:  # noqa: BLE001 — 收敛失败也不能炸穿事件循环;marker 留待清扫
                pass
            with self._lock:
                entry.status = "lost"
                entry.exit_code = None
            return
        with self._lock:
            entry.status = "completed"
            entry.exit_code = outcome.exit_code
            entry.timed_out = outcome.timed_out

    async def _observe(self, entry: _CommandSession, seconds: float) -> None:
        """等待任务结束最多 seconds 秒;等待方被取消时任务继续跑(R-T15-4)。"""
        task = entry.task
        if task is None or task.done() or seconds <= 0:
            return
        await asyncio.wait({task}, timeout=seconds)

    def _running_locked(self) -> bool:
        if self._starting:
            return True
        running = sum(1 for entry in self._sessions.values() if entry.status == "running")
        return running >= _MAX_RUNNING

    def _evict_locked(self) -> None:
        """容量上限只淘汰**已取回终态**者;未取回终态永不驱逐(满则拒绝)。"""
        if len(self._sessions) < _MAX_RETAINED:
            return
        retired_id = next(
            (
                session_id
                for session_id, entry in self._sessions.items()
                if entry.observed and entry.status != "running"
            ),
            None,
        )
        if retired_id is None:
            raise CommandSessionError(
                "command_session_capacity",
                f"retain at most {_MAX_RETAINED} collected terminal sessions; "
                "collect an outstanding terminal result before starting more",
            )
        retired = self._sessions.pop(retired_id)
        self._unlink_marker(retired)

    def _require_locked(self, session_id: str) -> _CommandSession:
        entry = self._sessions.get(session_id)
        if entry is not None:
            return entry
        # marker 存在但本进程没有会话/终态 = 崩溃前登记过、here 无从得知结果。
        # R-T15-9:session_id 必须先通过形状校验才允许拼进路径。
        registered = False
        if _SESSION_ID_PATTERN.fullmatch(session_id):
            try:
                registered = (self._jobs_dir / f"{session_id}.json").exists()
            except OSError:
                registered = False
        if registered:
            raise CommandSessionError(
                "command_session_lost",
                f"command session {session_id!r} has no terminal result in this process",
            )
        raise CommandSessionError(
            "command_session_not_found", f"unknown command session: {session_id!r}"
        )

    def _view_locked(self, entry: _CommandSession) -> CommandSessionView:
        return CommandSessionView(
            session_id=entry.session_id,
            pid=entry.pid,
            pgid=entry.pgid,
            status=entry.status,
            exit_code=entry.exit_code,
            stdout_head=_decode(entry.execution.stdout_bytes()),
            stderr_head=_decode(entry.execution.stderr_bytes()),
            truncated=entry.execution.truncated,
        )

    def _handle_locked(self, entry: _CommandSession) -> CommandSessionHandle:
        return CommandSessionHandle(
            session_id=entry.session_id,
            pid=entry.pid,
            pgid=entry.pgid,
            status=entry.status,
            exit_code=entry.exit_code,
            stdout=_decode(entry.execution.stdout_bytes()),
            stderr=_decode(entry.execution.stderr_bytes()),
            truncated=entry.execution.truncated,
            timed_out=entry.timed_out,
        )

    def _write_marker(
        self,
        session_id: str,
        host_pid: int,
        pid: int,
        pgid: int,
        command_prefix: str,
        created_at: float,
    ) -> Path:
        """宿主进程写 marker(0600;目录 0700),拒绝 symlink 并校验属主(R-T15-9)。

        与 ``adapters/config/credentials.py`` 同一纪律:目录先拒 symlink 再
        mkdir/stat(uid + 目录类型),marker 用 ``O_NOFOLLOW`` 打开后 fstat 校验
        (普通文件、当前用户、组/其他无权限、``st_nlink == 1``)并 fchmod 0600。
        沙箱子进程无权写此目录(R-T15-1)。
        """
        if not _SESSION_ID_PATTERN.fullmatch(session_id):
            raise OSError(f"invalid command session id: {session_id!r}")
        directory = self._jobs_dir
        if directory.is_symlink():
            raise OSError(f"command jobs directory is a symlink: {directory}")
        directory.mkdir(parents=True, exist_ok=True, mode=_JOBS_DIR_MODE)
        metadata = directory.stat()
        if (
            not stat.S_ISDIR(metadata.st_mode)
            or metadata.st_uid != os.getuid()
        ):
            raise OSError(f"unsafe command jobs directory: {directory}")
        os.chmod(directory, _JOBS_DIR_MODE)

        marker = directory / f"{session_id}.json"
        payload = json.dumps(
            {
                "session_id": session_id,
                "host_pid": host_pid,
                "pid": pid,
                "pgid": pgid,
                "command_prefix": command_prefix,
                "created_at": created_at,
            },
            ensure_ascii=False,
        )
        descriptor = os.open(
            marker,
            os.O_WRONLY | os.O_CREAT | os.O_NOFOLLOW | os.O_TRUNC,
            _MARKER_MODE,
        )
        try:
            handle = os.fdopen(descriptor, "w", encoding="utf-8")
        except OSError:
            os.close(descriptor)
            raise
        with handle:
            file_metadata = os.fstat(handle.fileno())
            if (
                not stat.S_ISREG(file_metadata.st_mode)
                or file_metadata.st_uid != os.getuid()
                or file_metadata.st_mode & 0o077
                or file_metadata.st_nlink != 1
            ):
                raise OSError(f"unsafe command job marker: {marker}")
            os.fchmod(handle.fileno(), _MARKER_MODE)
            handle.write(payload)
        return marker

    @staticmethod
    def _unlink_marker(entry: _CommandSession) -> None:
        marker = entry.marker
        if marker is None:
            return
        try:
            marker.unlink(missing_ok=True)
        except OSError:
            pass


# ---------------------------------------------------------------------------
# manage_command(不授予新权限:READ_WORKSPACE + 工作区根 target,与旧一致)
# ---------------------------------------------------------------------------


def _error_result(code: str, message: str) -> ToolResult:
    return ToolResult(
        tool_call_id=PENDING_TOOL_CALL_ID,
        tool_name="manage_command",
        content=(ToolContentBlock(type="text", data={"text": f"{code}: {message}"}),),
        status=ResultStatus.ERROR,
        is_error=True,
        error_code=code,
        error_message=message,
    )


def _view_result(view: CommandSessionView) -> ToolResult:
    text = view.stdout_head
    if view.stderr_head:
        text = f"{text}\n{view.stderr_head}" if text else view.stderr_head
    if not text:
        text = f"command session {view.session_id} is {view.status}"
    return ToolResult(
        tool_call_id=PENDING_TOOL_CALL_ID,
        tool_name="manage_command",
        content=(ToolContentBlock(type="text", data={"text": text}),),
        status=ResultStatus.OK,
        metadata={
            "session_id": view.session_id,
            "pid": view.pid,
            "pgid": view.pgid,
            "status": view.status,
            "exit_code": view.exit_code,
            "truncated": view.truncated,
        },
    )


def _read_result(view: CommandSessionView, chunk: CommandLogChunk) -> ToolResult:
    """action=read 的结果:文本是本次字节窗口,metadata 带续读游标。"""
    text = chunk.data or (
        f"no data at offset {chunk.next_offset} "
        f"(end_offset={chunk.end_offset}, stream truncated={chunk.truncated})"
    )
    return ToolResult(
        tool_call_id=PENDING_TOOL_CALL_ID,
        tool_name="manage_command",
        content=(ToolContentBlock(type="text", data={"text": text}),),
        status=ResultStatus.OK,
        metadata={
            "session_id": view.session_id,
            "pid": view.pid,
            "pgid": view.pgid,
            "status": view.status,
            "exit_code": view.exit_code,
            "truncated": chunk.truncated,
            "next_offset": chunk.next_offset,
            "end_offset": chunk.end_offset,
        },
    )


def create_manage_command_tool(sessions: CommandSessions) -> Tool:
    validate_input = json_schema_input(
        {
            "type": "object",
            "properties": {
                "session_id": {"type": "string", "minLength": 1, "maxLength": 100},
                "action": {
                    "type": "string",
                    "enum": ("status", "wait", "cancel", "read"),
                    "default": "wait",
                },
                "wait_seconds": {
                    "type": "number",
                    "minimum": 0,
                    "maximum": 60,
                    "default": 10,
                },
                "stream": {"type": "string", "enum": ("stdout", "stderr"), "default": "stdout"},
                "offset": {"type": "integer", "minimum": 0, "default": 0},
                "max_bytes": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": _MAX_READ_BYTES,
                    "default": 16_000,
                },
            },
            "required": ("session_id",),
            "additionalProperties": False,
        }
    )

    async def run(
        arguments: Mapping[str, JsonValue],
        exec_ctx: ToolExecContext,
    ) -> ToolResult:
        args = validate_input(arguments)
        session_id = cast("str", args["session_id"])
        action = cast("str", args.get("action", "wait"))
        wait_seconds = cast("float", args.get("wait_seconds", 10.0))
        try:
            if action == "status":
                view = sessions.get(session_id)
            elif action == "cancel":
                view = await sessions.cancel(session_id)
            elif action == "read":
                stream = cast("Literal['stdout', 'stderr']", args.get("stream", "stdout"))
                chunk = sessions.read(
                    session_id,
                    stream=stream,
                    offset=cast("int", args.get("offset", 0)),
                    max_bytes=cast("int", args.get("max_bytes", 16_000)),
                )
                view = sessions.get(session_id)
                if view.status == "lost":
                    return _error_result(
                        "command_session_lost",
                        f"command session {session_id!r} lost its process before completion",
                    )
                return _read_result(view, chunk)
            else:
                view = await sessions.wait(session_id, wait_seconds)
        except CommandSessionError as exc:
            return _error_result(exc.code, exc.message)
        if view.status == "lost":
            return _error_result(
                "command_session_lost",
                f"command session {session_id!r} lost its process before completion",
            )
        return _view_result(view)

    def resolve_use(arguments: Mapping[str, JsonValue]) -> ResolvedUse:
        return ResolvedUse(
            effects=frozenset({ToolEffect.READ_WORKSPACE}),
            targets=(ToolTarget(kind="path", value=str(sessions.workspace.resolve())),),
        )

    return Tool(
        definition=ToolDefinition(
            name="manage_command",
            description=(
                "Manage a background command session started by bash with "
                "yield_time_seconds. action=status returns immediately; wait returns after "
                "the command completes or wait_seconds, and an expired wait leaves it "
                "running; cancel terminates and reaps its process group; read returns a "
                "bounded byte window of the retained stdout/stderr head. This tool grants "
                "no new path or network access and cannot alter the original command."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "session_id": {
                        "type": "string",
                        "description": "Session id returned by bash (cmd_<32 hex>).",
                    },
                    "action": {
                        "type": "string",
                        "enum": ("status", "wait", "cancel", "read"),
                        "default": "wait",
                        "description": (
                            "Query immediately, wait for completion, cancel, or read a log "
                            "window. Default wait."
                        ),
                    },
                    "wait_seconds": {
                        "type": "number",
                        "minimum": 0,
                        "maximum": 60,
                        "default": 10,
                        "description": "Response wait for action=wait, 0..60 (default 10).",
                    },
                    "stream": {
                        "type": "string",
                        "enum": ("stdout", "stderr"),
                        "default": "stdout",
                        "description": "Log stream for action=read.",
                    },
                    "offset": {
                        "type": "integer",
                        "minimum": 0,
                        "default": 0,
                        "description": "Byte offset in the retained log, for action=read.",
                    },
                    "max_bytes": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": _MAX_READ_BYTES,
                        "default": 16_000,
                        "description": (
                            "Maximum bytes returned by action=read, 1..64000 (default 16000)."
                        ),
                    },
                },
                "required": ("session_id",),
                "additionalProperties": False,
            },
        ),
        validate_input=validate_input,
        run=run,
        static_effects=frozenset({ToolEffect.READ_WORKSPACE}),
        resolve_use=resolve_use,
        idempotent=False,
        concurrency_safe=False,
        cancellation_mode=CancellationMode.MANAGED_PROCESS,
        interrupt_behavior=InterruptBehavior.CANCEL,
        timeout_seconds=65,
        max_model_output_bytes=150_000,
    )
