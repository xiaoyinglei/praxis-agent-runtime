from __future__ import annotations

import asyncio
import os
import shlex
import signal
import sys
from pathlib import Path

import pytest

from agent_runtime.tools.builtins import shell as shell_module
from agent_runtime.tools.builtins.shell import create_run_command_tool
from agent_runtime.tools.executor import ExecutionStatus, ToolExecutor
from agent_runtime.tools.permissions import ToolExecutionContext
from agent_runtime.tools.tool import ToolCall, ToolCallOrigin
from agent_runtime.workspace import WorkspaceRuntime, open_workspace


def _call(command: str, *, timeout_seconds: float = 5.0) -> ToolCall:
    return ToolCall(
        tool_call_id="call_run_command",
        tool_name="run_command",
        arguments={
            "command": command,
            "working_dir": ".",
            "timeout_seconds": timeout_seconds,
        },
        origin=ToolCallOrigin(
            request_id="req_process",
            toolset_revision="tools_process_v1",
            exposed_tool_names=("run_command",),
        ),
    )


def _context(workspace: WorkspaceRuntime) -> ToolExecutionContext:
    return ToolExecutionContext(
        workspace_root=workspace.root,
        cwd=workspace.root,
        allow_write_tools=True,
        allow_execute_tools=True,
    )


def _spawn_tree_command(
    *,
    pgid_file: Path,
    sentinel: Path,
    sentinel_delay: float,
) -> str:
    child_code = (
        "import pathlib,time;"
        f"time.sleep({sentinel_delay!r});"
        f"pathlib.Path({str(sentinel)!r}).write_text('late', encoding='utf-8')"
    )
    parent_code = (
        "import subprocess,sys,time;"
        f"subprocess.Popen([sys.executable, '-c', {child_code!r}]);"
        "time.sleep(5)"
    )
    return (
        f"echo $$ > {shlex.quote(str(pgid_file))}; "
        f"exec {shlex.quote(sys.executable)} -c {shlex.quote(parent_code)}"
    )


async def _wait_for_file(path: Path, *, timeout: float = 1.0) -> None:
    async with asyncio.timeout(timeout):
        while not path.exists():
            await asyncio.sleep(0.01)


def _kill_group_if_present(pgid: int | None) -> None:
    if pgid is None:
        return
    try:
        os.killpg(pgid, signal.SIGKILL)
    except ProcessLookupError:
        pass


def test_process_group_probe_treats_eperm_as_still_exiting(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def deny_probe(_process_group_id: int, _signal_value: int) -> None:
        raise PermissionError("transient process-group probe denial")

    monkeypatch.setattr(shell_module.os, "killpg", deny_probe)

    assert shell_module._process_group_exists(12345) is True


@pytest.mark.anyio
@pytest.mark.usefixtures("fake_sandbox_exec")
async def test_command_input_timeout_kills_the_complete_process_group(
    tmp_path: Path,
) -> None:
    workspace = open_workspace(tmp_path, create=True)
    pgid_file = workspace.root / "pgid.txt"
    sentinel = workspace.root / "sentinel.txt"
    command = _spawn_tree_command(
        pgid_file=pgid_file,
        sentinel=sentinel,
        sentinel_delay=0.6,
    )
    tool = create_run_command_tool(
        workspace,
        hard_timeout_seconds=2.0,
        termination_grace_seconds=0.1,
    )
    executor = ToolExecutor({"run_command": tool})
    pgid: int | None = None

    try:
        execution = await executor.execute(
            _call(command, timeout_seconds=0.2),
            context=_context(workspace),
        )
        await _wait_for_file(pgid_file)
        pgid = int(pgid_file.read_text(encoding="utf-8").strip())
        await asyncio.sleep(0.7)

        assert execution.result.error_code == "timeout_cancelled"
        assert execution.result.structured_content is not None
        assert execution.result.structured_content["timed_out"] is True
        assert sentinel.exists() is False
        with pytest.raises(ProcessLookupError):
            os.killpg(pgid, 0)
    finally:
        _kill_group_if_present(pgid)


@pytest.mark.anyio
@pytest.mark.usefixtures("fake_sandbox_exec")
async def test_executor_timeout_kills_and_reaps_the_command_process_group(
    tmp_path: Path,
) -> None:
    workspace = open_workspace(tmp_path, create=True)
    pgid_file = workspace.root / "pgid.txt"
    sentinel = workspace.root / "sentinel.txt"
    command = _spawn_tree_command(
        pgid_file=pgid_file,
        sentinel=sentinel,
        sentinel_delay=0.6,
    )
    tool = create_run_command_tool(
        workspace,
        hard_timeout_seconds=0.2,
        termination_grace_seconds=0.1,
    )
    executor = ToolExecutor({"run_command": tool})
    pgid: int | None = None

    try:
        execution = await executor.execute(
            _call(command),
            context=_context(workspace),
        )
        await _wait_for_file(pgid_file)
        pgid = int(pgid_file.read_text(encoding="utf-8").strip())
        await asyncio.sleep(0.7)

        assert execution.result.error_code == "timeout_cancelled"
        assert execution.record is not None
        assert execution.record.status is ExecutionStatus.FAILED
        assert sentinel.exists() is False
        with pytest.raises(ProcessLookupError):
            os.killpg(pgid, 0)
    finally:
        _kill_group_if_present(pgid)


@pytest.mark.anyio
@pytest.mark.usefixtures("fake_sandbox_exec")
async def test_user_cancellation_kills_the_command_process_group(
    tmp_path: Path,
) -> None:
    workspace = open_workspace(tmp_path, create=True)
    pgid_file = workspace.root / "pgid.txt"
    sentinel = workspace.root / "sentinel.txt"
    command = _spawn_tree_command(
        pgid_file=pgid_file,
        sentinel=sentinel,
        sentinel_delay=0.5,
    )
    tool = create_run_command_tool(
        workspace,
        hard_timeout_seconds=5.0,
        termination_grace_seconds=0.1,
    )
    executor = ToolExecutor({"run_command": tool})
    task = asyncio.create_task(
        executor.execute(_call(command), context=_context(workspace))
    )
    pgid: int | None = None

    try:
        await _wait_for_file(pgid_file)
        pgid = int(pgid_file.read_text(encoding="utf-8").strip())
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        await asyncio.sleep(0.6)

        assert sentinel.exists() is False
        with pytest.raises(ProcessLookupError):
            os.killpg(pgid, 0)
        traces = [
            trace
            for trace in executor.traces
            if trace.tool_call_id == "call_run_command"
        ]
        assert len(traces) == 1
        assert traces[0].error_code == "cancelled"
    finally:
        if not task.done():
            task.cancel()
        _kill_group_if_present(pgid)
