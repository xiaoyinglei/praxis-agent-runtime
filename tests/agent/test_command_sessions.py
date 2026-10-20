from __future__ import annotations

import asyncio
import shlex
import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

from agent_runtime.tools.builtins import shell
from agent_runtime.workspace import open_workspace


def test_command_deadline_is_separate_from_yield():
    request = shell.RunCommandInput(command="sleep 1", timeout_seconds=None, yield_time_seconds=0.01)
    assert request.timeout_seconds is None
    assert request.yield_time_seconds == 0.01
    with pytest.raises(ValidationError, match="yield_time_seconds"):
        shell.RunCommandInput(command="sleep 1", timeout_seconds=None)
    for value in (0, -1, 61, float("inf"), float("nan")):
        with pytest.raises(ValidationError):
            shell.RunCommandInput(command="true", yield_time_seconds=value)


@pytest.mark.anyio
@pytest.mark.usefixtures("fake_sandbox_exec")
async def test_yield_and_repeated_waits_keep_one_process(tmp_path):
    from agent_runtime.tools.builtins.command_sessions import CommandSessions

    workspace = open_workspace(tmp_path, create=True)
    sessions = CommandSessions(workspace, termination_grace_seconds=0.1)
    tool = shell.create_run_command_tool(workspace, command_sessions=sessions)
    try:
        started = await tool.run(dict(command="printf started; sleep 0.2; printf done",
                                      timeout_seconds=None, yield_time_seconds=0.01))
        assert started.status == "running"
        assert started.exit_code is None and started.session_id
        assert not shell._normalize_command_output(started).is_error
        first = await sessions.manage(started.session_id, action="wait", wait_seconds=0.01)
        assert first.status == "running"
        final = await sessions.manage(started.session_id, action="wait", wait_seconds=3)
        assert final.status == "completed" and final.exit_code == 0
        assert final.stdout == "starteddone"
        assert (await sessions.manage(started.session_id, action="status", wait_seconds=0)).stdout == final.stdout
        assert not list(workspace.scratch.glob("run-command-*"))
    finally:
        await sessions.aclose()


@pytest.fixture(params=["fake", "seatbelt"])
def command_backend(request, tmp_path):
    if request.param == "fake":
        request.getfixturevalue("fake_sandbox_exec")
    elif shell.sys.platform != "darwin" or not Path("/usr/bin/sandbox-exec").is_file():
        pytest.skip("requires real macOS Seatbelt")
    return open_workspace(tmp_path / "workspace", create=True)


def python_command(workspace, code):
    from agent_runtime.permission_policy import PathRule, PermissionPolicy

    # Declare the test interpreter's read dependencies instead of relying on an OS shim.
    if sys.platform == "darwin":
        roots = {Path(sys.prefix).resolve(), Path(sys.base_prefix).resolve(),
                 Path(sys.executable).resolve().parent.parent}
        policy = PermissionPolicy(filesystem=tuple(
            PathRule(path=str(root), access="read") for root in sorted(roots)
        )).resolve(workspace.root)
        workspace.permission_provider = lambda: policy
    return shlex.quote(sys.executable) + " -c " + shlex.quote(code)


async def start(sessions, command, **options):
    tool = shell.create_run_command_tool(sessions.workspace, command_sessions=sessions)
    return await tool.run({"command": command, "timeout_seconds": None, "yield_time_seconds": 0.01, **options})


async def wait_for_output(sessions, session_id, text):
    async def poll():
        while True:
            result = await sessions.manage(session_id, action="status", wait_seconds=0)
            if text in result.stdout or result.status != "running":
                return result
            await asyncio.sleep(0.01)
    result = await asyncio.wait_for(poll(), 5)
    assert text in result.stdout, result
    return result


@pytest.mark.anyio
async def test_real_lifecycle_wait_cancel_child_cleanup_and_private_temp(command_backend):
    from agent_runtime.tools.builtins.command_sessions import CommandSessions

    sessions = CommandSessions(command_backend, termination_grace_seconds=0.2)
    command = python_command(command_backend,
        'import os, subprocess, signal, time; '
        'child = subprocess.Popen(["/bin/sh", "-c", "trap \'\' TERM; while :; do sleep 1; done"]); '
        'print(str(child.pid) + " " + os.environ["TMPDIR"], flush=True); '
        'signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(60)'
    )
    try:
        result = await start(sessions, command)
        ready = await wait_for_output(sessions, result.session_id, "run-command-")
        child_pid, scratch = ready.stdout.split()
        assert Path(scratch).is_dir()
        still_running = await sessions.manage(result.session_id, action="wait", wait_seconds=0.02)
        assert still_running.status == "running"
        assert shell._process_group_exists(int(child_pid)) is False  # child is not the group leader
        cancelled = await sessions.manage(result.session_id, action="cancel", wait_seconds=0)
        assert cancelled.status == "cancelled"
        assert shell._normalize_command_output(cancelled).error_code == "command_cancelled"
        assert not Path(scratch).exists()
        with pytest.raises(ProcessLookupError):
            shell.os.kill(int(child_pid), 0)
    finally:
        await sessions.aclose()


@pytest.mark.anyio
@pytest.mark.parametrize("finish", ["exit", "timeout", "cancel", "close"])
async def test_files_survive_failure_deadline_cancel_and_close(command_backend, finish):
    from agent_runtime.tools.builtins.command_sessions import CommandSessions

    sessions = CommandSessions(command_backend, termination_grace_seconds=0.2)
    command = "printf partial > artifact; printf ready; " + ("exit 7" if finish == "exit" else "sleep 60")
    try:
        result = await start(sessions, command, workspace_write=True,
                             timeout_seconds=0.3 if finish == "timeout" else None)
        await wait_for_output(sessions, result.session_id, "ready")
        if finish == "cancel":
            result = await sessions.manage(result.session_id, action="cancel", wait_seconds=0)
            assert result.status == "cancelled"
        elif finish == "close":
            await sessions.aclose()
            assert not sessions.running
        else:
            result = await sessions.manage(result.session_id, action="wait", wait_seconds=3)
            assert result.status == "completed"
            assert result.timed_out is (finish == "timeout")
            assert shell._normalize_command_output(result).is_error
            if finish == "exit":
                assert result.exit_code == 7
        assert (command_backend.root / "artifact").read_text() == "partial"
        assert not list(command_backend.scratch.glob("run-command-*"))
    finally:
        await sessions.aclose()


@pytest.mark.anyio
async def test_cancelling_wait_reaps_the_command(command_backend):
    from agent_runtime.tools.builtins.command_sessions import CommandSessions

    sessions = CommandSessions(command_backend, termination_grace_seconds=0.2)
    try:
        result = await start(sessions, "printf ready; sleep 60")
        await wait_for_output(sessions, result.session_id, "ready")
        waiter = asyncio.create_task(sessions.manage(result.session_id, action="wait", wait_seconds=60))
        await asyncio.sleep(0.02)
        waiter.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(waiter, 3)
        assert not sessions.running
        assert (await sessions.manage(result.session_id, action="status", wait_seconds=0)).status == "cancelled"
        assert not list(command_backend.scratch.glob("run-command-*"))
    finally:
        await sessions.aclose()


@pytest.mark.anyio
async def test_session_output_keeps_tail_after_progress_budget(command_backend):
    from agent_runtime.tools.builtins.command_sessions import CommandSessions

    sessions = CommandSessions(command_backend)
    try:
        result = await start(sessions, python_command(command_backend,
            'import sys; print("HEAD", flush=True); '
            'sys.stdout.write("x" * 400000); sys.stderr.write("y" * 400000); '
            'print("TAIL", flush=True); print("ERRTAIL", file=sys.stderr, flush=True)'
        ))
        result = await sessions.manage(result.session_id, action="wait", wait_seconds=5)
        assert result.exit_code == 0, result.stderr
        assert result.stdout.startswith("HEAD") and result.stdout.endswith("TAIL\n")
        assert result.stderr.endswith("ERRTAIL\n")
        assert result.truncated
        assert len(result.stdout.encode()) <= shell._MAX_STREAM_BYTES
        assert len(result.stderr.encode()) <= shell._MAX_STREAM_BYTES
    finally:
        await sessions.aclose()


@pytest.mark.anyio
async def test_busy_and_owner_scoped_handles(command_backend):
    from agent_runtime.tools.builtins.command_sessions import CommandSessions

    owner = CommandSessions(command_backend)
    other = CommandSessions(command_backend)
    try:
        result = await start(owner, "sleep 60")
        assert (await start(owner, "touch unexpected", workspace_write=True)).sandbox_error == "command_session_busy"
        ordinary = shell.create_run_command_tool(command_backend, command_sessions=owner)
        assert (await ordinary.run({"command": "true"})).sandbox_error == "command_session_busy"
        assert (await other.manage(result.session_id, action="cancel", wait_seconds=0)).sandbox_error == (
            "command_session_not_found"
        )
        assert owner.running and not (command_backend.root / "unexpected").exists()
        await owner.manage(result.session_id, action="cancel", wait_seconds=0)
        next_command = await start(owner, "printf next")
        next_command = await owner.manage(next_command.session_id, action="wait", wait_seconds=3)
        assert next_command.exit_code == 0 and next_command.stdout == "next"
    finally:
        await owner.aclose()
        await other.aclose()


@pytest.mark.anyio
async def test_real_macos_large_files_and_external_writes_are_visible(command_backend):
    from agent_runtime.tools.builtins.command_sessions import CommandSessions

    sessions = CommandSessions(command_backend)
    try:
        result = await start(sessions, python_command(command_backend,
            'from pathlib import Path; import time; '
            'f = Path("large.bin").open("wb"); '
            '[f.write(b"x" * 1024 * 1024) for _ in range(65)]; f.close(); '
            'print("ready", flush=True); '
            'p = Path("host-signal"); '
            '\nwhile not p.exists(): time.sleep(0.01)\n'
            'print(p.read_text(), flush=True)'
        ), workspace_write=True)
        await wait_for_output(sessions, result.session_id, "ready")
        assert (command_backend.root / "large.bin").stat().st_size == 65 * 1024 * 1024
        # Direct Seatbelt writes are not candidate transactions. Host writes are
        # visible to the same running process; no snapshot publication is involved.
        (command_backend.root / "host-signal").write_text("external-change")
        result = await sessions.manage(result.session_id, action="wait", wait_seconds=5)
        assert result.exit_code == 0 and "external-change" in result.stdout
        second = await start(sessions, "wc -c < large.bin")
        second = await sessions.manage(second.session_id, action="wait", wait_seconds=3)
        assert int(second.stdout) == 65 * 1024 * 1024
        assert second.workspace_published is None
    finally:
        await sessions.aclose()


@pytest.mark.anyio
async def test_success_reaps_background_children_in_the_command_group(command_backend):
    from agent_runtime.tools.builtins.command_sessions import CommandSessions

    sessions = CommandSessions(command_backend, termination_grace_seconds=0.2)
    child_pid = None
    try:
        result = await start(sessions, "sleep 60 >/dev/null 2>&1 & echo $!")
        result = await sessions.manage(result.session_id, action="wait", wait_seconds=3)
        assert result.exit_code == 0, result
        child_pid = int(result.stdout)
        with pytest.raises(ProcessLookupError):
            shell.os.kill(child_pid, 0)
    finally:
        # Before the fix this test demonstrates a real orphan; do not leak it.
        if child_pid is not None:
            try:
                shell.os.kill(child_pid, shell.signal.SIGKILL)
            except ProcessLookupError:
                pass
        await sessions.aclose()


@pytest.mark.skipif(shell.sys.platform != "darwin", reason="requires real macOS Seatbelt")
@pytest.mark.anyio
@pytest.mark.parametrize("boundary", ["read_only", "protected", "symlink", "hardlink"])
async def test_real_session_keeps_existing_path_boundaries(tmp_path, boundary):
    from agent_runtime.tools.builtins.command_sessions import CommandSessions

    workspace = open_workspace(tmp_path / "workspace", create=True)
    outside = tmp_path / "outside"
    outside.write_text("secret")
    (workspace.root / ".git").mkdir()
    (workspace.root / ".git/config").write_text("protected")
    (workspace.root / "existing").write_text("before")
    (workspace.root / "alias").symlink_to(outside)
    if boundary == "hardlink":
        shell.os.link(outside, workspace.root / "hardlink")
    name = {"read_only": "existing", "protected": ".git/config", "symlink": "alias", "hardlink": "hardlink"}[boundary]
    sessions = CommandSessions(workspace)
    try:
        result = await start(sessions, "printf forbidden > " + name, workspace_write=boundary != "read_only")
        result = await sessions.manage(result.session_id, action="wait", wait_seconds=3)
        assert shell._normalize_command_output(result).is_error
        if boundary == "hardlink":
            assert result.sandbox_error == "workspace_hardlink_detected"
        assert outside.read_text() == "secret"
        assert (workspace.root / ".git/config").read_text() == "protected"
        assert (workspace.root / "existing").read_text() == "before"
    finally:
        await sessions.aclose()


@pytest.mark.anyio
@pytest.mark.usefixtures("fake_sandbox_exec")
async def test_uncollected_results_are_not_evicted(tmp_path, monkeypatch):
    from agent_runtime.tools.builtins import command_sessions

    monkeypatch.setattr(command_sessions, "_MAX_RETAINED_SESSIONS", 1)
    owner = command_sessions.CommandSessions(open_workspace(tmp_path, create=True))
    try:
        first = await start(owner, "sleep 0.05")
        pending = owner._sessions[first.session_id]
        await asyncio.wait_for(asyncio.shield(pending.task), 5)
        assert not pending.observed_terminal
        refused = await start(owner, "true")
        assert refused.sandbox_error == "command_session_capacity"
        assert (await owner.manage(first.session_id, action="status", wait_seconds=0)).exit_code == 0
        second = await start(owner, "printf next")
        assert second.session_id != first.session_id
        second = await owner.manage(second.session_id, action="wait", wait_seconds=2)
        assert second.stdout == "next"
    finally:
        await owner.aclose()


@pytest.mark.anyio
async def test_linux_sessions_fail_closed_without_bubblewrap(tmp_path, monkeypatch):
    from agent_runtime.tools.builtins.command_sessions import CommandSessions

    monkeypatch.setattr(shell, "_SANDBOX_PLATFORM", "linux")
    monkeypatch.setattr(shell, "_BWRAP_PATH", "/nonexistent/bwrap")
    owner = CommandSessions(open_workspace(tmp_path, create=True))
    result = await start(owner, "touch unexpected", workspace_write=True)
    assert result.sandbox_error == "sandbox_unavailable"
    assert not owner.running and not (tmp_path / "unexpected").exists()
    await owner.aclose()


@pytest.mark.skipif(shell.sys.platform != "darwin", reason="requires real macOS Seatbelt")
@pytest.mark.anyio
async def test_real_session_keeps_network_opt_in(tmp_path):
    import socket

    from agent_runtime.tools.builtins.command_sessions import CommandSessions

    sessions = CommandSessions(open_workspace(tmp_path, create=True))
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen()
        command = python_command(sessions.workspace,
            'import socket; s = socket.socket(); s.settimeout(1); '
            f's.connect(("127.0.0.1", {listener.getsockname()[1]})); print("connected", flush=True)'
        )
        try:
            blocked = await start(sessions, command)
            blocked = await sessions.manage(blocked.session_id, action="wait", wait_seconds=3)
            assert blocked.exit_code != 0 and not blocked.network_enabled
            allowed = await start(sessions, command, network=True)
            assert allowed.network_enabled
            allowed = await sessions.manage(allowed.session_id, action="wait", wait_seconds=3)
            assert allowed.exit_code == 0 and allowed.network_enabled
            assert allowed.stdout.strip() == "connected"
        finally:
            await sessions.aclose()
