"""bash 内置工具(移植旧 agent_runtime shell.py:653-840 的子进程/进程组语义,修正 EOF)。

执行算法(brief Step 3,修正 asyncio 的 EOF 语义):
1. ``[*sandbox.argv_prefix, "/bin/bash", "-lc", command]``,cwd=workspace,
   stdout/stderr 管道,``start_new_session=True``(独立进程组,超时才能整组回收);
   子进程环境由 ``child_env`` 唯一构造(R-T16-2:固定 PATH、HOME/TMPDIR 指向
   沙箱可写 scratch、不继承任何宿主环境变量——凭据不可见)。run 只允许经
   管道调用:管道对 EXECUTE_PROCESS 恒编译沙箱(后端不可用已在 decide 第 7 步
   deny);``exec_ctx.sandbox=None`` 为单测路径直跑。trusted-local 等不可沙箱化
   策略在编译期 SandboxRefusal → 管道转 sandbox_refused 结果(R9 fail-closed);
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

Task 15(yield_time_seconds,D13):给出 yield 窗口且注入了 ``CommandSessions``
时,进程由会话注册表持有、marker 由**本 Python 进程**写(``command_jobs_dir()``
对沙箱是 deny,子进程内写必然失败,这是设计)。窗口内退出 → 直接返回终态;
否则返回 ``state="running"`` 与 ``session_id``,后台继续跑。未给 yield 时行为
与 Task 11 完全一致,只是 metadata 补 ``pid/pgid/state``。

``CommandExecution`` 是"一次命令执行"的唯一驱动(foreground 与后台会话共用):
spawn / wait_terminal / cancel 三者只组合 ``process.py`` 公开原语,不另写
killpg 或 TERM→KILL 序列。``CommandSessionError`` 定义在本模块,供
``command_sessions`` 单向依赖(避免 import 环)。

ToolResult 占位符约定同 filesystem.py:``tool_call_id="pending"``,管道替换。
"""

from __future__ import annotations

import asyncio
import contextlib
import signal
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

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
from praxis.tools.builtins.child_env import child_env, scratch_dir_for
from praxis.tools.builtins.filesystem import PENDING_TOOL_CALL_ID
from praxis.tools.builtins.process import await_main_exit, drain_stream, signal_group

if TYPE_CHECKING:
    from praxis.tools.builtins.command_sessions import CommandSessions

_DEFAULT_TIMEOUT_SECONDS = 120.0
_MAX_TIMEOUT_SECONDS = 600.0
_MAX_YIELD_SECONDS = 60.0
_TERM_GRACE_SECONDS = 5.0
_POST_KILL_WAIT_SECONDS = 2.0
_DRAIN_DEADLINE_SECONDS = 0.5
_MAX_OUTPUT_BYTES = 30_000
_MAX_STREAM_BUFFER_BYTES = 16_000


class CommandSessionError(RuntimeError):
    """后台命令会话的公开失败口径(``code`` 进 ``tool_result.error_code``)。

    定义在 command.py 而非 command_sessions.py,是为了让 command_sessions
    单向依赖执行器模块:bash 只需捕获本异常,不存在循环 import。
    """

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        self.message = message
        super().__init__(message)


@dataclass(frozen=True, slots=True)
class CommandOutcome:
    """一次命令执行的终态快照(文本已解码,truncated 含缓冲与总输出两层截断)。"""

    stdout: str
    stderr: str
    truncated: bool
    timed_out: bool
    exit_code: int | None


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
    signal_group(proc.pid, signal.SIGTERM)
    try:
        await await_main_exit(proc, _TERM_GRACE_SECONDS)
    except TimeoutError:
        signal_group(proc.pid, signal.SIGKILL)
        await await_main_exit(proc, None)


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
    signaled = signal_group(proc.pid, signal.SIGTERM)
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
            signal_group(proc.pid, signal.SIGKILL)
            window = _POST_KILL_WAIT_SECONDS
            continue
        await _abandon_wait(wait_task)
        return


class CommandExecution:
    """一次子进程执行的唯一驱动(foreground bash 与后台会话共用)。

    spawn 之后的三个阶段:
    - ``wait_terminal(timeout)``:等主进程退出/超时收敛,收尾读取,返回终态;
    - ``cancel()``:任何阶段取消的统一收敛(Finding 3),不留悬空任务/孤儿进程组;
    - ``stdout_bytes``/``stderr_bytes``:已缓冲的头(≤``_MAX_STREAM_BUFFER_BYTES``/路)。
    所有进程组信号都经 process.py 原语,本类不直接发送组信号。
    """

    def __init__(
        self,
        proc: asyncio.subprocess.Process,
        out_buf: list[bytes],
        err_buf: list[bytes],
        out_discarded: list[int],
        err_discarded: list[int],
        tasks: tuple[asyncio.Task[Any], ...],
        wait_task: asyncio.Task[int],
    ) -> None:
        self.proc = proc
        self._out_buf = out_buf
        self._err_buf = err_buf
        self._out_discarded = out_discarded
        self._err_discarded = err_discarded
        self._tasks = tasks
        self._wait_task = wait_task
        self._out_task = tasks[0]
        self._err_task = tasks[1]

    @classmethod
    async def spawn(
        cls,
        *,
        argv: Sequence[str],
        cwd: Path,
        env: Mapping[str, str] | None,
    ) -> CommandExecution:
        """起进程并建立读管道;spawn 半路失败时自行收尸(M2)。

        ``create_subprocess_exec`` 成功后任何一步抛出(管道不变量、create_task
        失败)都**不能**留下无主进程组:整组 SIGKILL → 取消已建任务 → 有界
        回收主进程,再把异常透传。调用方(``CommandSessions.start``)此时还没有
        任何句柄,这里必须是唯一的收尸点。
        """
        proc = await asyncio.create_subprocess_exec(
            *argv,
            cwd=str(cwd),
            env=env,
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            start_new_session=True,
        )
        tasks: list[asyncio.Task[Any]] = []
        try:
            if proc.stdout is None or proc.stderr is None:
                raise RuntimeError("bash process pipes were not created")
            out_buf: list[bytes] = []
            err_buf: list[bytes] = []
            out_discarded: list[int] = [0]
            err_discarded: list[int] = [0]
            tasks.append(
                asyncio.create_task(
                    drain_stream(
                        proc.stdout, out_buf, out_discarded, limit=_MAX_STREAM_BUFFER_BYTES
                    )
                )
            )
            tasks.append(
                asyncio.create_task(
                    drain_stream(
                        proc.stderr, err_buf, err_discarded, limit=_MAX_STREAM_BUFFER_BYTES
                    )
                )
            )
            tasks.append(asyncio.create_task(proc.wait()))
        except BaseException:
            signal_group(proc.pid, signal.SIGKILL)
            for task in tasks:
                task.cancel()
            for task in tasks:
                with contextlib.suppress(BaseException):
                    await task
            try:
                await await_main_exit(proc, _POST_KILL_WAIT_SECONDS)
            except TimeoutError:
                pass
            raise
        out_task, err_task, wait_task = tasks
        return cls(
            proc,
            out_buf,
            err_buf,
            out_discarded,
            err_discarded,
            (out_task, err_task, wait_task),
            wait_task,
        )

    @property
    def pid(self) -> int:
        return self.proc.pid

    @property
    def truncated(self) -> bool:
        return self._out_discarded[0] > 0 or self._err_discarded[0] > 0

    def stdout_bytes(self) -> bytes:
        return b"".join(self._out_buf)

    def stderr_bytes(self) -> bytes:
        return b"".join(self._err_buf)

    async def wait_terminal(self, timeout: float) -> CommandOutcome:
        """等到主进程退出(超时则整组收敛),再把管道读到可读边界。"""
        timed_out = False
        try:
            await await_main_exit(self.proc, timeout)
        except TimeoutError:
            timed_out = True
            await _reap_main_exit(self.proc)
        await _finish_drain(self._out_task)
        await _finish_drain(self._err_task)
        if not self._wait_task.done():
            await _converge_pipes(self.proc, self._wait_task)
        return CommandOutcome(
            stdout=self.stdout_bytes().decode("utf-8", errors="replace"),
            stderr=self.stderr_bytes().decode("utf-8", errors="replace"),
            truncated=self.truncated,
            timed_out=timed_out,
            exit_code=self.proc.returncode,
        )

    async def cancel(self) -> None:
        """任何阶段被取消的统一收敛:TERM 整组 → 有界宽限 → KILL 整组 → 回收任务。"""
        killed = False
        try:
            signal_group(self.proc.pid, signal.SIGTERM)
            try:
                await await_main_exit(self.proc, _TERM_GRACE_SECONDS)
            except TimeoutError:
                pass
            signal_group(self.proc.pid, signal.SIGKILL)
            killed = True
            try:
                await await_main_exit(self.proc, _POST_KILL_WAIT_SECONDS)
            except TimeoutError:
                pass
        except asyncio.CancelledError:
            if not killed:
                signal_group(self.proc.pid, signal.SIGKILL)
            raise
        finally:
            for task in self._tasks:
                task.cancel()
            for task in self._tasks:
                with contextlib.suppress(BaseException):
                    await task


def _terminal_result(
    *,
    stdout: str,
    stderr: str,
    truncated: bool,
    timed_out: bool,
    exit_code: int | None,
    pid: int,
    pgid: int,
) -> ToolResult:
    """把命令终态投影成 bash 的 ToolResult(foreground 与 yield 窗口内退出共用)。"""
    text = stdout if not stderr else (f"{stdout}\n{stderr}" if stdout else stderr)
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
            "exit_code": exit_code,
            "timed_out": timed_out,
            "pid": pid,
            "pgid": pgid,
            # R-T15-10:终态词汇与 CommandSessionView.status 统一,不再另说 "exited"。
            "state": "completed",
        },
    )


def _error_result(*, code: str, message: str) -> ToolResult:
    return ToolResult(
        tool_call_id=PENDING_TOOL_CALL_ID,
        tool_name="bash",
        content=(ToolContentBlock(type="text", data={"text": f"{code}: {message}"}),),
        status=ResultStatus.ERROR,
        is_error=True,
        error_code=code,
        error_message=message,
    )


def create_bash_tool(workspace: Path, *, sessions: CommandSessions | None = None) -> Tool:
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
                "yield_time_seconds": {
                    "type": "number",
                    "exclusiveMinimum": 0,
                    "maximum": _MAX_YIELD_SECONDS,
                },
            },
            "required": ("command",),
            "additionalProperties": False,
        }
    )

    async def spawn(argv: Sequence[str], env: Mapping[str, str] | None) -> CommandExecution:
        return await CommandExecution.spawn(argv=argv, cwd=workspace, env=env)

    async def run(
        arguments: Mapping[str, JsonValue],
        exec_ctx: ToolExecContext,
    ) -> ToolResult:
        args = validate_input(arguments)
        command = cast("str", args["command"])
        timeout = cast("float", args.get("timeout_seconds", _DEFAULT_TIMEOUT_SECONDS))
        yield_seconds = cast("float | None", args.get("yield_time_seconds"))
        # 沙箱接线(Task 7):argv_prefix 拼在真实命令之前(sandbox-exec -p profile)。
        # R-T16-2:子进程环境一律由 child_env 构造(固定 PATH、HOME/TMPDIR 指向
        # scratch、宿主变量与凭据不透传);sandbox=None 为单测路径直跑,用同一份
        # 最小环境(进程私有临时目录作 scratch),不存在透传 os.environ 的第二条路径。
        spec = exec_ctx.sandbox
        prefix = spec.argv_prefix if spec is not None else ()
        env = child_env(scratch_dir_for(spec), workspace)
        argv = (*prefix, "/bin/bash", "-lc", command)

        # Task 15:yield 窗口 + 会话注册表 → 后台会话(D13:返回的 tool_result
        # item 自身就是孤儿登记;marker 是宿主进程写的崩溃清扫凭据)。
        if yield_seconds is not None:
            # M6:没有 owner 时 yield 不可能兑现(会静默退化成前台阻塞),fail loud。
            if sessions is None:
                return _error_result(
                    code="command_session_unavailable",
                    message=(
                        "background command sessions are unavailable: this bash tool was "
                        "built without a CommandSessions owner"
                    ),
                )
            try:
                handle = await sessions.start(
                    lambda: spawn(argv, env),
                    command_prefix=command,
                    timeout_seconds=timeout,
                    yield_time_seconds=yield_seconds,
                )
            except CommandSessionError as exc:
                return _error_result(code=exc.code, message=exc.message)
            if handle.status != "running":
                if handle.status == "lost":
                    return _error_result(
                        code="command_session_lost",
                        message=f"command session {handle.session_id} lost its process before completion",
                    )
                return _terminal_result(
                    stdout=handle.stdout,
                    stderr=handle.stderr,
                    truncated=handle.truncated,
                    timed_out=handle.timed_out,
                    exit_code=handle.exit_code,
                    pid=handle.pid,
                    pgid=handle.pgid,
                )
            return ToolResult(
                tool_call_id=PENDING_TOOL_CALL_ID,
                tool_name="bash",
                content=(
                    ToolContentBlock(
                        type="text",
                        data={
                            "text": (
                                f"command is running in background session {handle.session_id} "
                                f"(pid {handle.pid}); use manage_command with this session_id "
                                "to status, wait, or cancel it"
                            )
                        },
                    ),
                ),
                status=ResultStatus.OK,
                metadata={
                    "session_id": handle.session_id,
                    "pid": handle.pid,
                    "pgid": handle.pgid,
                    "state": "running",
                },
            )

        execution = await spawn(argv, env)
        try:
            outcome = await execution.wait_terminal(timeout)
        except asyncio.CancelledError:
            await execution.cancel()
            raise
        return _terminal_result(
            stdout=outcome.stdout,
            stderr=outcome.stderr,
            truncated=outcome.truncated,
            timed_out=outcome.timed_out,
            exit_code=outcome.exit_code,
            pid=execution.pid,
            pgid=execution.pid,
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
                "returns combined stdout/stderr and exit_code. Commands run under the "
                "OS sandbox compiled from the permission policy; policies that cannot "
                "be sandboxed (e.g. trusted-local with write grants) refuse execution "
                "with sandbox_refused — use the read/write tools for file edits instead. "
                "The subprocess receives only a minimal environment: host environment "
                "variables and credentials are not visible to it. "
                "Set yield_time_seconds to return while a long command keeps running in "
                "the background; the result then carries a session_id for manage_command."
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
                    "yield_time_seconds": {
                        "type": "number",
                        "description": (
                            f"Return after this many seconds if the command is still running "
                            f"(0 < x <= {_MAX_YIELD_SECONDS:.0f}); the command continues in a "
                            "background session addressable with manage_command. Omit for the "
                            "synchronous behavior."
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
