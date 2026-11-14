"""Task 15:bash 后台会话 / CommandSessions / manage_command / 孤儿进程组清扫(B20/D13)。

分三层:
1. 纯逻辑(fake execution):并发 start 只开一个、容量上限只淘汰已取回终态者、
   spawn 半路失败收尸;
2. 真实子进程(无沙箱直跑):yield 返回 running + marker 形状/权限、cancel 整组回收;
3. seatbelt 真沙箱(``probe_backend().available`` 才跑)的 B20 三案:
   后台孙进程不绑架返回、yield 登记、cancel 后进程组真的消失。

测试纪律:每个用例结束必须 `sessions.aclose()`,不留任何活着的子进程组
(2026-11-13 孤儿 pytest 吃穿内存事故的教训)。等待驱动任务用 ``_wait_driver``,
不用固定时长的 sleep 当同步手段(M9)。
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import signal
import time
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from praxis import composition
from praxis.adapters.models.scripted import ScriptedAdapter
from praxis.adapters.sandbox.base import probe_backend
from praxis.adapters.sandbox.compile import compile_sandbox, finalize_sandbox
from praxis.core import recovery
from praxis.core.events import EventBus
from praxis.core.pipeline import ExecCtx, execute
from praxis.protocol.floor import command_jobs_dir
from praxis.protocol.permission import PermissionContext
from praxis.protocol.policy import NetworkPolicy, PermissionPolicy
from praxis.protocol.sandbox import ToolExecContext
from praxis.protocol.tools import ToolCall, ToolEffect
from praxis.store.log import Store
from praxis.store.turns import create_turn
from praxis.tools.builtins import command_sessions as command_sessions_module
from praxis.tools.builtins.command import CommandExecution, create_bash_tool
from praxis.tools.builtins.command_sessions import (
    CommandSessionError,
    CommandSessionHandle,
    CommandSessions,
    create_manage_command_tool,
)

_AVAILABLE = probe_backend().available
_SESSION_ID = re.compile(r"cmd_[0-9a-f]{32}")
_MARKER_ID = "cmd_" + "a" * 32


# ---------------------------------------------------------------------------
# 脚手架:fake execution(start 的并发/容量语义无需真实进程)
# ---------------------------------------------------------------------------


@dataclass
class _FakeExecution:
    """只实现 CommandSessions.start 需要的执行器表面(鸭子类型)。"""

    pid: int = 999_999
    delay: float = 0.2
    cancelled: list[bool] = field(default_factory=list)

    async def wait_terminal(self, timeout: float) -> SimpleNamespace:
        await asyncio.sleep(self.delay)
        return SimpleNamespace(
            stdout="fake\n",
            stderr="",
            truncated=False,
            timed_out=False,
            exit_code=0,
        )

    async def cancel(self) -> None:
        self.cancelled.append(True)

    def stdout_bytes(self) -> bytes:
        return b"fake\n"

    def stderr_bytes(self) -> bytes:
        return b""

    @property
    def truncated(self) -> bool:
        return False


def _fake_spawn(delay: float = 0.2) -> Any:
    async def spawn() -> _FakeExecution:
        return _FakeExecution(delay=delay)

    return spawn


async def _start_fake(
    sessions: CommandSessions,
    *,
    delay: float = 0.2,
    yield_time_seconds: float = 0.01,
) -> CommandSessionHandle:
    return await sessions.start(
        _fake_spawn(delay),
        command_prefix="sleep 1",
        timeout_seconds=10,
        yield_time_seconds=yield_time_seconds,
    )


async def _wait_driver(sessions: CommandSessions, session_id: str) -> None:
    """测试同步点:等驱动任务本身(M9;不经 get/wait,避免把会话标记为已取回)。"""
    entry = sessions._sessions[session_id]  # noqa: SLF001 — 测试同步点
    assert entry.task is not None
    await asyncio.wait({entry.task})


def _write_marker(directory: Path, name: str, **fields: Any) -> Path:
    directory.mkdir(parents=True, exist_ok=True, mode=0o700)
    payload: dict[str, Any] = {
        "session_id": name,
        "host_pid": 999_998,  # 默认视为已死宿主;存活用例显式覆盖
        "pid": 4242,
        "pgid": 4242,
        "command_prefix": "python -m http.server",
        "created_at": 1.0,
    }
    payload.update(fields)
    path = directory / f"{name}.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    os.chmod(path, 0o600)  # 与宿主写出的 marker 同权限;_read_marker 会拒绝宽松权限
    return path


def _kill_recorder() -> tuple[list[tuple[int, signal.Signals]], Any]:
    killed: list[tuple[int, signal.Signals]] = []

    def kill_group(pgid: int, sig: signal.Signals) -> bool:
        killed.append((pgid, sig))
        return True

    return killed, kill_group


def _group_alive(pgid: int) -> bool:
    try:
        os.killpg(pgid, 0)
    except ProcessLookupError:
        return False
    return True


def _wait_group_dead(pgid: int, budget: float = 5.0) -> None:
    deadline = time.monotonic() + budget
    while _group_alive(pgid):
        assert time.monotonic() < deadline, f"process group {pgid} survived cancellation"
        time.sleep(0.05)


# ---------------------------------------------------------------------------
# 常量与 schema
# ---------------------------------------------------------------------------


def test_capacity_constants_are_literal() -> None:
    assert command_sessions_module._MAX_RUNNING == 1
    assert command_sessions_module._MAX_RETAINED == 32


def test_bash_yield_schema_bounds(tmp_path: Path) -> None:
    tool = create_bash_tool(tmp_path)
    tool.validate_input({"command": "true", "yield_time_seconds": 0.001})
    tool.validate_input({"command": "true", "yield_time_seconds": 60})
    with pytest.raises(ValueError):
        tool.validate_input({"command": "true", "yield_time_seconds": 0})
    with pytest.raises(ValueError):
        tool.validate_input({"command": "true", "yield_time_seconds": 61})


def test_manage_command_policy_surface_and_schema_defaults(tmp_path: Path) -> None:
    sessions = CommandSessions(tmp_path)
    tool = create_manage_command_tool(sessions)
    assert tool.definition.name == "manage_command"
    assert tool.static_effects == frozenset({ToolEffect.READ_WORKSPACE})
    assert tool.timeout_seconds == 65
    assert tool.max_model_output_bytes == 150_000
    use = tool.resolve_use({"session_id": "cmd_x", "action": "status"})
    assert use.effects == frozenset({ToolEffect.READ_WORKSPACE})
    assert use.targets[0].value == str(tmp_path.resolve())
    properties = tool.definition.input_schema["properties"]
    assert properties["action"]["enum"] == ("status", "wait", "cancel", "read")  # type: ignore[index]
    assert properties["wait_seconds"]["default"] == 10  # type: ignore[index]
    assert properties["max_bytes"]["default"] == 16_000  # type: ignore[index]
    assert properties["stream"]["default"] == "stdout"  # type: ignore[index]
    assert properties["offset"]["default"] == 0  # type: ignore[index]


def test_yield_without_owner_fails_loud(tmp_path: Path) -> None:
    """M6:没有 CommandSessions owner 时不得静默退化成前台阻塞。"""

    async def scenario() -> Any:
        return await create_bash_tool(tmp_path).run(
            {"command": "sleep 20", "yield_time_seconds": 0.5}, ToolExecContext()
        )

    result = asyncio.run(scenario())
    assert result.is_error
    assert result.error_code == "command_session_unavailable"


def test_manage_command_marker_without_session_is_lost(tmp_path: Path) -> None:
    """marker 在(崩溃前登记过)但本进程无终态 → command_session_lost,不是 not_found。"""
    jobs = tmp_path / "jobs"
    name = "cmd_" + "a" * 32
    _write_marker(jobs, name)

    async def scenario() -> Any:
        sessions = CommandSessions(tmp_path, jobs_dir=jobs)
        try:
            return await create_manage_command_tool(sessions).run(
                {"session_id": name, "action": "status"}, ToolExecContext()
            )
        finally:
            await sessions.aclose()

    result = asyncio.run(scenario())
    assert result.is_error
    assert result.error_code == "command_session_lost"


def test_manage_command_unknown_session(tmp_path: Path) -> None:
    async def scenario() -> Any:
        sessions = CommandSessions(tmp_path, jobs_dir=tmp_path / "jobs")
        try:
            return await create_manage_command_tool(sessions).run(
                {"session_id": "cmd_" + "0" * 32, "action": "status"}, ToolExecContext()
            )
        finally:
            await sessions.aclose()

    result = asyncio.run(scenario())
    assert result.is_error
    assert result.error_code == "command_session_not_found"


# ---------------------------------------------------------------------------
# 并发/容量/关闭语义(fake execution;全部状态转换在锁内)
# ---------------------------------------------------------------------------


def test_concurrent_start_never_double_opens(tmp_path: Path) -> None:
    async def scenario() -> list[Any]:
        sessions = CommandSessions(tmp_path, jobs_dir=tmp_path / "jobs")
        try:
            return await asyncio.gather(
                _start_fake(sessions),
                _start_fake(sessions),
                return_exceptions=True,
            )
        finally:
            await sessions.aclose()

    results = asyncio.run(scenario())
    handles = [r for r in results if isinstance(r, CommandSessionHandle)]
    errors = [r for r in results if isinstance(r, CommandSessionError)]
    assert len(handles) == 1
    assert len(errors) == 1
    assert errors[0].code == "command_session_busy"


def test_capacity_rejects_when_all_unobserved(tmp_path: Path, monkeypatch: Any) -> None:
    monkeypatch.setattr(command_sessions_module, "_MAX_RETAINED", 2)

    async def scenario() -> str:
        sessions = CommandSessions(tmp_path, jobs_dir=tmp_path / "jobs")
        try:
            for _ in range(2):
                handle = await _start_fake(sessions, delay=0.02, yield_time_seconds=0.001)
                assert handle.status == "running"
                await _wait_driver(sessions, handle.session_id)  # 自然终态,但无人取回
            try:
                await _start_fake(sessions, delay=0.02, yield_time_seconds=0.001)
            except CommandSessionError as exc:
                return exc.code
            return "no-error"
        finally:
            await sessions.aclose()

    assert asyncio.run(scenario()) == "command_session_capacity"


def test_observed_terminal_is_evictable(tmp_path: Path, monkeypatch: Any) -> None:
    monkeypatch.setattr(command_sessions_module, "_MAX_RETAINED", 2)

    async def scenario() -> None:
        sessions = CommandSessions(tmp_path, jobs_dir=tmp_path / "jobs")
        try:
            for _ in range(2):
                handle = await _start_fake(sessions, delay=0.02, yield_time_seconds=0.001)
                await _wait_driver(sessions, handle.session_id)
                view = await sessions.wait(handle.session_id, 1.0)
                assert view.status == "completed"  # 取回 => observed
            third = await _start_fake(sessions, delay=0.02, yield_time_seconds=0.001)
            assert third.status == "running"
        finally:
            await sessions.aclose()

    asyncio.run(scenario())


def test_aclose_is_idempotent_and_cancels_running(tmp_path: Path) -> None:
    executions: list[_FakeExecution] = []

    async def spawn() -> _FakeExecution:
        execution = _FakeExecution(delay=10)
        executions.append(execution)
        return execution

    async def scenario() -> None:
        sessions = CommandSessions(tmp_path, jobs_dir=tmp_path / "jobs")
        handle = await sessions.start(
            spawn,
            command_prefix="sleep 10",
            timeout_seconds=10,
            yield_time_seconds=0.01,
        )
        assert handle.status == "running"
        await sessions.aclose()
        await sessions.aclose()  # 幂等:第二次立即返回
        assert executions[0].cancelled == [True]

    asyncio.run(scenario())


def test_start_racing_aclose_does_not_register_live_session(tmp_path: Path) -> None:
    """R-T15-8:spawn/marker 期间 aclose 完成 ⇒ 取消进程、删 marker、拒绝登记。"""
    jobs = tmp_path / "jobs"
    executions: list[_FakeExecution] = []
    spawned = asyncio.Event()
    release = asyncio.Event()

    async def spawn() -> _FakeExecution:
        execution = _FakeExecution(delay=10)
        executions.append(execution)
        spawned.set()
        await release.wait()
        return execution

    async def scenario() -> None:
        sessions = CommandSessions(tmp_path, jobs_dir=jobs)
        task = asyncio.create_task(
            sessions.start(
                spawn,
                command_prefix="sleep 10",
                timeout_seconds=10,
                yield_time_seconds=0.1,
            )
        )
        await spawned.wait()
        await sessions.aclose()  # 关档发生在登记之前
        release.set()
        with pytest.raises(CommandSessionError) as excinfo:
            await task
        assert excinfo.value.code == "command_session_unavailable"
        assert executions[0].cancelled == [True]
        assert list(jobs.glob("*.json")) == []
        assert sessions._sessions == {}  # noqa: SLF001 — 不得登记进已关闭注册表

    asyncio.run(scenario())


def test_driver_failure_cancels_execution_before_lost(tmp_path: Path) -> None:
    """M1:执行器异常标 lost 之前必须先取消进程组并保留/清理登记的完整性。"""
    executions: list[_FakeExecution] = []

    class _Boom(_FakeExecution):
        async def wait_terminal(self, timeout: float) -> SimpleNamespace:
            await asyncio.sleep(0.02)  # 先让 start 返回 running,再制造驱动异常
            raise RuntimeError("driver boom")

    async def spawn() -> _FakeExecution:
        execution = _Boom(delay=0)
        executions.append(execution)
        return execution

    async def scenario() -> str:
        sessions = CommandSessions(tmp_path, jobs_dir=tmp_path / "jobs")
        try:
            handle = await sessions.start(
                spawn,
                command_prefix="sleep 1",
                timeout_seconds=10,
                yield_time_seconds=0.001,
            )
            await _wait_driver(sessions, handle.session_id)
            return sessions.get(handle.session_id).status
        finally:
            await sessions.aclose()

    assert asyncio.run(scenario()) == "lost"
    assert executions[0].cancelled == [True]


# ---------------------------------------------------------------------------
# 真实子进程:yield 登记 / marker 形状 / 单开 / cancel 整组回收
# ---------------------------------------------------------------------------


def test_foreground_metadata_carries_pid_state(tmp_path: Path) -> None:
    async def scenario() -> Any:
        sessions = CommandSessions(tmp_path, jobs_dir=tmp_path / "jobs")
        try:
            return await create_bash_tool(tmp_path, sessions=sessions).run(
                {"command": "true"}, ToolExecContext()
            )
        finally:
            await sessions.aclose()

    result = asyncio.run(scenario())
    assert result.metadata["exit_code"] == 0
    assert result.metadata["state"] == "completed"  # R-T15-10:与视图 status 同一词汇
    assert isinstance(result.metadata["pid"], int)
    assert result.metadata["pgid"] == result.metadata["pid"]


def test_yield_returns_running_registers_marker_and_cancel_reaps(tmp_path: Path) -> None:
    jobs = tmp_path / "jobs"

    async def scenario() -> tuple[Any, Any, bool, bool]:
        sessions = CommandSessions(tmp_path, jobs_dir=jobs)
        try:
            bash = create_bash_tool(tmp_path, sessions=sessions)
            manage = create_manage_command_tool(sessions)
            result = await bash.run(
                {"command": "sleep 20", "yield_time_seconds": 0.5}, ToolExecContext()
            )
            session_id = result.metadata["session_id"]
            marker = jobs / f"{session_id}.json"
            pgid = int(result.metadata["pgid"])
            exists_before = marker.exists()
            cancelled = await manage.run(
                {"session_id": session_id, "action": "cancel"}, ToolExecContext()
            )
            _wait_group_dead(pgid)
            return result, cancelled, exists_before, marker.exists()
        finally:
            await sessions.aclose()

    result, cancelled, exists_before, exists_after = asyncio.run(scenario())
    assert result.metadata["state"] == "running"
    assert _SESSION_ID.fullmatch(str(result.metadata["session_id"]))
    assert exists_before
    assert cancelled.metadata["status"] == "cancelled"
    assert not exists_after


def test_marker_and_jobs_dir_have_private_modes(tmp_path: Path) -> None:
    jobs = tmp_path / "jobs"

    async def scenario() -> tuple[int, int, dict[str, Any]]:
        sessions = CommandSessions(tmp_path, jobs_dir=jobs)
        try:
            result = await create_bash_tool(tmp_path, sessions=sessions).run(
                {"command": "sleep 20", "yield_time_seconds": 0.3}, ToolExecContext()
            )
            marker = jobs / f"{result.metadata['session_id']}.json"
            payload = json.loads(marker.read_text(encoding="utf-8"))
            return (
                jobs.stat().st_mode & 0o777,
                marker.stat().st_mode & 0o777,
                payload,
            )
        finally:
            await sessions.aclose()

    dir_mode, marker_mode, payload = asyncio.run(scenario())
    assert dir_mode == 0o700
    assert marker_mode == 0o600
    assert set(payload) == {
        "session_id",
        "host_pid",
        "pid",
        "pgid",
        "command_prefix",
        "created_at",
    }
    assert _SESSION_ID.fullmatch(payload["session_id"])
    assert payload["host_pid"] == os.getpid()
    assert payload["pid"] == payload["pgid"]
    assert len(payload["command_prefix"]) <= 200


def test_terminal_within_yield_window_returns_exit_and_leaves_no_marker(tmp_path: Path) -> None:
    jobs = tmp_path / "jobs"

    async def scenario() -> Any:
        sessions = CommandSessions(tmp_path, jobs_dir=jobs)
        try:
            return await create_bash_tool(tmp_path, sessions=sessions).run(
                {"command": "echo fast; exit 3", "yield_time_seconds": 10}, ToolExecContext()
            )
        finally:
            await sessions.aclose()

    result = asyncio.run(scenario())
    assert result.metadata["state"] == "completed"
    assert result.metadata["exit_code"] == 3
    assert "fast" in result.text
    assert list(jobs.glob("*.json")) == []


def test_only_one_running_session_rejects_second_bash(tmp_path: Path) -> None:
    async def scenario() -> tuple[Any, Any]:
        sessions = CommandSessions(tmp_path, jobs_dir=tmp_path / "jobs")
        try:
            bash = create_bash_tool(tmp_path, sessions=sessions)
            first = await bash.run(
                {"command": "sleep 20", "yield_time_seconds": 0.2}, ToolExecContext()
            )
            second = await bash.run(
                {"command": "echo nope", "yield_time_seconds": 0.2}, ToolExecContext()
            )
            await create_manage_command_tool(sessions).run(
                {"session_id": first.metadata["session_id"], "action": "cancel"},
                ToolExecContext(),
            )
            return first, second
        finally:
            await sessions.aclose()

    first, second = asyncio.run(scenario())
    assert first.metadata["state"] == "running"
    assert second.is_error
    assert second.error_code == "command_session_busy"


def test_yield_wait_cancel_does_not_cancel_running_command(tmp_path: Path) -> None:
    """R-T15-4:取消等待(bash 的 yield 窗口)不得连带取消命令——那是 cancel 的职责。"""
    jobs = tmp_path / "jobs"

    async def scenario() -> tuple[Any, Any]:
        sessions = CommandSessions(tmp_path, jobs_dir=jobs)
        try:
            bash = create_bash_tool(tmp_path, sessions=sessions)
            manage = create_manage_command_tool(sessions)
            task = asyncio.create_task(
                bash.run({"command": "sleep 20", "yield_time_seconds": 30}, ToolExecContext())
            )
            deadline = time.monotonic() + 5
            while not list(jobs.glob("*.json")) and time.monotonic() < deadline:
                await asyncio.sleep(0.02)
            markers = list(jobs.glob("*.json"))
            assert markers, "background session did not register a marker"
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            session_id = markers[0].stem
            status = await manage.run(
                {"session_id": session_id, "action": "status"}, ToolExecContext()
            )
            cancelled = await manage.run(
                {"session_id": session_id, "action": "cancel"}, ToolExecContext()
            )
            return status, cancelled
        finally:
            await sessions.aclose()

    status, cancelled = asyncio.run(scenario())
    assert status.metadata["status"] == "running"
    assert cancelled.metadata["status"] == "cancelled"


def test_manage_wait_returns_running_then_terminal(tmp_path: Path) -> None:
    async def scenario() -> tuple[Any, Any, Path]:
        sessions = CommandSessions(tmp_path, jobs_dir=tmp_path / "jobs")
        try:
            bash = create_bash_tool(tmp_path, sessions=sessions)
            manage = create_manage_command_tool(sessions)
            started = await bash.run(
                {"command": "sleep 0.4", "yield_time_seconds": 0.05}, ToolExecContext()
            )
            session_id = started.metadata["session_id"]
            still_running = await manage.run(
                {"session_id": session_id, "action": "wait", "wait_seconds": 0.05},
                ToolExecContext(),
            )
            finished = await manage.run(
                {"session_id": session_id, "action": "wait", "wait_seconds": 5},
                ToolExecContext(),
            )
            return still_running, finished, tmp_path / "jobs" / f"{session_id}.json"
        finally:
            await sessions.aclose()

    still_running, finished, marker = asyncio.run(scenario())
    assert still_running.metadata["status"] == "running"
    assert finished.metadata["status"] == "completed"
    assert finished.metadata["exit_code"] == 0
    assert not marker.exists()


def test_manage_read_returns_window_and_cursor(tmp_path: Path) -> None:
    """R-T15-10:action=read 真接 CommandSessions.read(不再有无消费者的 schema)。"""

    async def scenario() -> tuple[Any, Any]:
        sessions = CommandSessions(tmp_path, jobs_dir=tmp_path / "jobs")
        try:
            bash = create_bash_tool(tmp_path, sessions=sessions)
            manage = create_manage_command_tool(sessions)
            started = await bash.run(
                {"command": "echo hello-read; sleep 20", "yield_time_seconds": 0.3},
                ToolExecContext(),
            )
            session_id = started.metadata["session_id"]
            first = await manage.run(
                {"session_id": session_id, "action": "read", "offset": 0, "max_bytes": 5},
                ToolExecContext(),
            )
            rest = await manage.run(
                {
                    "session_id": session_id,
                    "action": "read",
                    "offset": first.metadata["next_offset"],
                    "max_bytes": 16_000,
                },
                ToolExecContext(),
            )
            await manage.run(
                {"session_id": session_id, "action": "cancel"}, ToolExecContext()
            )
            return first, rest
        finally:
            await sessions.aclose()

    first, rest = asyncio.run(scenario())
    assert first.text == "hello"
    assert first.metadata["next_offset"] == 5
    assert rest.text.startswith("-read")
    assert isinstance(first.metadata["end_offset"], int)


# ---------------------------------------------------------------------------
# 文件系统卫生(R-T15-9)
# ---------------------------------------------------------------------------


def test_start_rejects_symlinked_jobs_dir(tmp_path: Path) -> None:
    target = tmp_path / "elsewhere"
    target.mkdir()
    link = tmp_path / "jobs"
    link.symlink_to(target, target_is_directory=True)
    executions: list[_FakeExecution] = []

    async def spawn() -> _FakeExecution:
        execution = _FakeExecution(delay=0)
        executions.append(execution)
        return execution

    async def scenario() -> None:
        sessions = CommandSessions(tmp_path, jobs_dir=link)
        with pytest.raises(CommandSessionError) as excinfo:
            await sessions.start(
                spawn,
                command_prefix="sleep 1",
                timeout_seconds=10,
                yield_time_seconds=0.01,
            )
        assert excinfo.value.code == "command_session_unavailable"
        assert executions[0].cancelled == [True]
        assert list(target.glob("*.json")) == []

    asyncio.run(scenario())


def test_sweep_rejects_symlink_marker(tmp_path: Path, monkeypatch: Any) -> None:
    name = "cmd_" + "a" * 32
    jobs = tmp_path / "jobs"
    outside = tmp_path / "outside"
    outside.mkdir()
    target = outside / "real.json"
    _write_marker(jobs, name)
    (jobs / f"{name}.json").rename(target)
    link = jobs / f"{name}.json"
    link.symlink_to(target)
    monkeypatch.setattr(recovery, "_host_alive", lambda pid: False)
    killed, kill_group = _kill_recorder()

    summaries = recovery.sweep_orphan_command_jobs(jobs, kill_group=kill_group)

    assert killed == []
    assert [(s.session_id, s.state) for s in summaries] == [(name, "corrupt")]
    assert not link.exists()  # symlink 本身被删
    assert target.exists()  # 目标文件未被跟随/删除


# ---------------------------------------------------------------------------
# spawn 半路失败收尸(M2)
# ---------------------------------------------------------------------------


def test_spawn_failure_after_process_start_kills_group(
    tmp_path: Path, monkeypatch: Any
) -> None:
    created: list[asyncio.subprocess.Process] = []
    real_exec = asyncio.create_subprocess_exec

    async def record_exec(*args: Any, **kwargs: Any) -> Any:
        process = await real_exec(*args, **kwargs)
        created.append(process)
        return process

    def failing_create_task(coro: Any, **kwargs: Any) -> Any:
        coro.close()
        raise RuntimeError("task creation failed")

    monkeypatch.setattr(asyncio, "create_subprocess_exec", record_exec)
    monkeypatch.setattr(asyncio, "create_task", failing_create_task)

    async def scenario() -> None:
        with pytest.raises(RuntimeError):
            await CommandExecution.spawn(
                argv=("/bin/bash", "-lc", "sleep 30"), cwd=tmp_path, env=None
            )

    asyncio.run(scenario())
    assert created
    assert created[0].returncode is not None  # 收尸完成:主进程已回收


# ---------------------------------------------------------------------------
# 孤儿清扫(PID 复用 / 宿主存活门;全部 monkeypatch,不碰真实进程)
# ---------------------------------------------------------------------------


def test_sweep_leaves_other_host_marker_untouched(tmp_path: Path) -> None:
    """R-T15-6:宿主仍活着 = 另一个活会话,绝不 killpg、不删 marker。"""
    marker = _write_marker(tmp_path, _MARKER_ID, host_pid=os.getpid())
    killed, kill_group = _kill_recorder()
    assert recovery._host_alive(os.getpid()) is True  # 真实实现:本进程必然存活

    summaries = recovery.sweep_orphan_command_jobs(tmp_path, kill_group=kill_group)

    assert killed == []
    assert summaries[0].state == "other_host"
    assert marker.exists()


def test_sweep_does_not_kill_recycled_pid(tmp_path: Path, monkeypatch: Any) -> None:
    marker = _write_marker(tmp_path, _MARKER_ID, host_pid=999_998)
    monkeypatch.setattr(recovery, "_host_alive", lambda pid: False)
    monkeypatch.setattr(recovery, "_probe_process", lambda pid: (4242, "/usr/bin/other-program"))
    killed, kill_group = _kill_recorder()

    summaries = recovery.sweep_orphan_command_jobs(tmp_path, kill_group=kill_group)

    assert killed == []
    assert len(summaries) == 1 and summaries[0].state == "stale"
    assert not marker.exists()


def test_sweep_pgid_mismatch_is_stale(tmp_path: Path, monkeypatch: Any) -> None:
    marker = _write_marker(tmp_path, _MARKER_ID, host_pid=999_998)
    monkeypatch.setattr(recovery, "_host_alive", lambda pid: False)
    monkeypatch.setattr(
        recovery, "_probe_process", lambda pid: (7777, "python -m http.server 8000")
    )
    killed, kill_group = _kill_recorder()

    summaries = recovery.sweep_orphan_command_jobs(tmp_path, kill_group=kill_group)

    assert killed == []
    assert summaries[0].state == "stale"
    assert not marker.exists()


def test_sweep_kills_matching_group_with_term_then_kill(
    tmp_path: Path, monkeypatch: Any
) -> None:
    marker = _write_marker(tmp_path, _MARKER_ID, host_pid=999_998)
    monkeypatch.setattr(recovery, "_host_alive", lambda pid: False)
    monkeypatch.setattr(
        recovery, "_probe_process", lambda pid: (4242, "python -m http.server 8000")
    )
    monkeypatch.setattr(recovery, "_ORPHAN_TERM_GRACE_SECONDS", 0.0)
    killed, kill_group = _kill_recorder()

    summaries = recovery.sweep_orphan_command_jobs(tmp_path, kill_group=kill_group)

    assert killed == [(4242, signal.SIGTERM), (4242, signal.SIGKILL)]
    assert summaries[0].state == "killed"
    assert not marker.exists()


def test_sweep_corrupt_marker_is_removed(tmp_path: Path, monkeypatch: Any) -> None:
    corrupt = tmp_path / "cmd_broken.json"
    corrupt.write_text("{not json", encoding="utf-8")
    monkeypatch.setattr(recovery, "_host_alive", lambda pid: False)
    monkeypatch.setattr(recovery, "_probe_process", lambda pid: (1, "unused"))
    killed, kill_group = _kill_recorder()

    summaries = recovery.sweep_orphan_command_jobs(tmp_path, kill_group=kill_group)

    assert killed == []
    assert [(s.session_id, s.state) for s in summaries] == [("cmd_broken", "corrupt")]
    assert not corrupt.exists()


def test_sweep_probe_failure_is_untouchable_and_keeps_marker(
    tmp_path: Path, monkeypatch: Any
) -> None:
    marker = _write_marker(tmp_path, _MARKER_ID, host_pid=999_998)
    monkeypatch.setattr(recovery, "_host_alive", lambda pid: False)

    def boom(pid: int) -> tuple[int, str]:
        raise OSError("ps unavailable")

    monkeypatch.setattr(recovery, "_probe_process", boom)
    killed, kill_group = _kill_recorder()

    summaries = recovery.sweep_orphan_command_jobs(tmp_path, kill_group=kill_group)

    assert killed == []
    assert summaries[0].state == "untouchable"
    assert marker.exists()  # 保留登记,下次清扫可重试


def test_sweep_missing_process_is_stale(tmp_path: Path, monkeypatch: Any) -> None:
    marker = _write_marker(tmp_path, _MARKER_ID, host_pid=999_998)
    monkeypatch.setattr(recovery, "_host_alive", lambda pid: False)
    monkeypatch.setattr(recovery, "_probe_process", lambda pid: None)
    killed, kill_group = _kill_recorder()

    summaries = recovery.sweep_orphan_command_jobs(tmp_path, kill_group=kill_group)

    assert killed == []
    assert summaries[0].state == "stale"
    assert not marker.exists()


def test_sweep_missing_jobs_dir_is_noop(tmp_path: Path) -> None:
    killed, kill_group = _kill_recorder()
    assert (
        recovery.sweep_orphan_command_jobs(tmp_path / "absent", kill_group=kill_group) == ()
    )


# ---------------------------------------------------------------------------
# composition 装配(M9):sessions 同时注入 bash / manage_command、打开时清扫、close 收尾
# ---------------------------------------------------------------------------


def test_composition_wires_sessions_sweep_and_close(tmp_path: Path, monkeypatch: Any) -> None:
    recorded: dict[str, Any] = {}

    def fake_sweep(**kwargs: Any) -> tuple[Any, ...]:
        recorded.update(kwargs)
        return ()

    monkeypatch.setattr(composition, "sweep_orphan_command_jobs", fake_sweep)
    session = composition.build_session(
        tmp_path, ScriptedAdapter([]), thread_id="t15-composition"
    )
    marker: Path | None = None
    pgid = 0
    try:
        tools = session._tools  # noqa: SLF001 — 装配覆盖:检查同一实例注入两个工具
        assert "bash" in tools and "manage_command" in tools
        assert recorded.get("kill_group") is not None

        async def drive() -> tuple[Any, Any]:
            started = await tools["bash"].run(
                {"command": "sleep 20", "yield_time_seconds": 0.2}, ToolExecContext()
            )
            status = await tools["manage_command"].run(
                {"session_id": started.metadata["session_id"], "action": "status"},
                ToolExecContext(),
            )
            return started, status

        started, status = asyncio.run(drive())
        assert status.metadata["status"] == "running"  # 两个工具共享同一注册表
        pgid = int(started.metadata["pgid"])
        marker = command_jobs_dir() / f"{started.metadata['session_id']}.json"
        assert marker.exists()
    finally:
        session.close()
    assert marker is not None and not marker.exists()  # Session.close -> aclose -> unlink
    _wait_group_dead(pgid)


# ---------------------------------------------------------------------------
# seatbelt 真沙箱三案(R-T15-5;B20 行为对照实测)
# ---------------------------------------------------------------------------


def _sandbox_ctx(store: Store, workspace: Path, sessions: CommandSessions) -> ExecCtx:
    policy = PermissionPolicy(network=NetworkPolicy(decision="allow"))
    turn = create_turn(store, turn_id="turn-1", thread_id="thread-1", binding={})
    tools = {
        "bash": create_bash_tool(workspace, sessions=sessions),
        "manage_command": create_manage_command_tool(sessions),
    }
    return ExecCtx(
        store=store,
        bus=EventBus(),
        workspace=workspace,
        tools=tools,
        permissions=PermissionContext(
            workspace=str(workspace),
            policy=policy,
            mode="bypass",
            sandbox=probe_backend(),
        ),
        turn_id=turn.turn_id,
        thread_id="thread-1",
        policy_revision=policy.revision,
        compile_sandbox=compile_sandbox,
        finalize_sandbox=finalize_sandbox,
    )


@pytest.mark.skipif(not _AVAILABLE, reason="real bash needs a sandbox backend")
def test_sandbox_background_grandchild_does_not_hold_return(
    store: Store, workspace: Path, tmp_path: Path
) -> None:
    """B20 主案:后台孙进程持管道,主命令退出即返回(旧实现拖到 EOF/硬超时)。"""

    async def scenario() -> tuple[Any, float]:
        sessions = CommandSessions(workspace, jobs_dir=tmp_path / "jobs")
        try:
            ctx = _sandbox_ctx(store, workspace, sessions)
            started = time.monotonic()
            result = await execute(
                ToolCall(
                    "c1",
                    "bash",
                    {"command": "sleep 30 & echo done", "timeout_seconds": 20},
                ),
                ctx,
            )
            return result, time.monotonic() - started
        finally:
            await sessions.aclose()

    result, elapsed = asyncio.run(scenario())
    assert result.metadata["exit_code"] == 0
    assert result.metadata["timed_out"] is False
    assert "done" in result.text
    assert elapsed < 8.0


@pytest.mark.skipif(not _AVAILABLE, reason="real bash needs a sandbox backend")
def test_sandbox_yield_returns_running_and_registers_marker(
    store: Store, workspace: Path, tmp_path: Path
) -> None:
    jobs = tmp_path / "jobs"

    async def scenario() -> tuple[Any, bool]:
        sessions = CommandSessions(workspace, jobs_dir=jobs)
        try:
            ctx = _sandbox_ctx(store, workspace, sessions)
            result = await execute(
                ToolCall("c1", "bash", {"command": "sleep 20", "yield_time_seconds": 0.5}),
                ctx,
            )
            session_id = result.metadata["session_id"]
            return result, (jobs / f"{session_id}.json").exists()
        finally:
            # aclose 取消在跑会话并清理 marker(不留活口)
            await sessions.aclose()

    result, existed = asyncio.run(scenario())
    assert result.metadata["state"] == "running"
    assert _SESSION_ID.fullmatch(str(result.metadata["session_id"]))
    assert existed
    assert list(jobs.glob("*.json")) == []  # aclose 后 marker 清空


@pytest.mark.skipif(not _AVAILABLE, reason="real bash needs a sandbox backend")
def test_sandbox_cancel_reaps_group_and_removes_marker(
    store: Store, workspace: Path, tmp_path: Path
) -> None:
    jobs = tmp_path / "jobs"

    async def scenario() -> tuple[Any, bool, bool]:
        sessions = CommandSessions(workspace, jobs_dir=jobs)
        try:
            ctx = _sandbox_ctx(store, workspace, sessions)
            started = await execute(
                ToolCall("c1", "bash", {"command": "sleep 20", "yield_time_seconds": 0.5}),
                ctx,
            )
            session_id = started.metadata["session_id"]
            pgid = int(started.metadata["pgid"])
            marker = jobs / f"{session_id}.json"
            assert marker.exists()
            cancelled = await execute(
                ToolCall(
                    "c2",
                    "manage_command",
                    {"session_id": session_id, "action": "cancel"},
                ),
                ctx,
            )
            _wait_group_dead(pgid)
            return cancelled, marker.exists(), _group_alive(pgid)
        finally:
            await sessions.aclose()

    cancelled, marker_exists, alive = asyncio.run(scenario())
    assert cancelled.metadata["status"] == "cancelled"
    assert marker_exists is False
    assert alive is False
