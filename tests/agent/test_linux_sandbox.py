from __future__ import annotations

from pathlib import Path

import pytest

from agent_runtime.tools.builtins import shell
from agent_runtime.workspace import open_workspace


@pytest.mark.anyio
@pytest.mark.parametrize("python", [False, True])
async def test_linux_missing_bwrap_fails_closed(tmp_path, monkeypatch, python):
    monkeypatch.setattr(shell, "_SANDBOX_PLATFORM", "linux")
    monkeypatch.setattr(shell, "_BWRAP_PATH", str(tmp_path / "missing"), raising=False)
    workspace = open_workspace(tmp_path, create=True)
    if python:
        result = await shell._execute_managed_python(
            workspace,
            shell.ManagedPythonInput(code="raise AssertionError('must not run')"),
            termination_grace_seconds=0.1,
        )
    else:
        result = await shell._run_command(
            workspace,
            shell.RunCommandInput(command="touch must-not-exist"),
            termination_grace_seconds=0.1,
        )
    assert result.sandbox_error == "sandbox_unavailable"
    assert "bubblewrap" in result.stderr.lower()
    assert not (tmp_path / "must-not-exist").exists()


@pytest.mark.anyio
@pytest.mark.parametrize("python", [False, True])
@pytest.mark.parametrize("permission", ["workspace_write", "network"])
async def test_linux_unsupported_permissions_never_run(tmp_path, monkeypatch, python, permission):
    monkeypatch.setattr(shell, "_SANDBOX_PLATFORM", "linux")
    monkeypatch.setattr(shell, "_BWRAP_PATH", "/bin/sh", raising=False)
    workspace = open_workspace(tmp_path, create=True)
    if python:
        result = await shell._execute_managed_python(
            workspace,
            shell.ManagedPythonInput(code="raise AssertionError('must not run')", **{permission: True}),
            termination_grace_seconds=0.1,
        )
    else:
        result = await shell._run_command(
            workspace,
            shell.RunCommandInput(command="touch must-not-exist", **{permission: True}),
            termination_grace_seconds=0.1,
        )
    assert result.sandbox_error == "sandbox_policy_unsupported"
    assert permission in result.stderr
    assert not (tmp_path / "must-not-exist").exists()


def test_linux_argv_limits_host_mounts(tmp_path):
    workspace = tmp_path / "workspace"
    scratch = workspace / "scratch"
    scratch.mkdir(parents=True)
    argv = shell._linux_sandbox_argv(
        workspace_root=workspace,
        temporary_root=scratch,
        additional_read_roots=(),
    )
    assert "--unshare-user" in argv
    assert "--unshare-all" in argv
    assert "--cap-drop" in argv
    assert "--new-session" in argv
    assert "--share-net" not in argv
    mounts = [tuple(argv[i : i + 3]) for i, arg in enumerate(argv) if arg in {"--bind", "--ro-bind"}]
    assert ("--ro-bind", str(workspace), str(workspace)) in mounts
    assert ("--bind", str(scratch), str(scratch)) in mounts
    assert all(source != "/" for _, source, _ in mounts)
    assert all(source != str(Path.home()) for _, source, _ in mounts)


@pytest.mark.anyio
@pytest.mark.parametrize("python", [False, True])
async def test_linux_setup_failure_is_reported(tmp_path, monkeypatch, python):
    launcher = tmp_path / "bwrap"
    launcher.write_text('#!/bin/sh\nprintf "bwrap: namespace denied\\n" >&2\nexit 1\n')
    launcher.chmod(0o755)
    monkeypatch.setattr(shell, "_SANDBOX_PLATFORM", "linux")
    monkeypatch.setattr(shell, "_BWRAP_PATH", str(launcher))
    workspace = open_workspace(tmp_path / "workspace", create=True)
    if python:
        result = await shell._execute_managed_python(
            workspace,
            shell.ManagedPythonInput(code="print(42)"),
            termination_grace_seconds=0.1,
        )
    else:
        result = await shell._run_command(
            workspace,
            shell.RunCommandInput(command="echo 42"),
            termination_grace_seconds=0.1,
        )
    assert result.sandbox_error == "sandbox_start_failed"
    assert "namespace denied" in result.stderr


@pytest.mark.skipif(shell.sys.platform != "linux", reason="requires real Linux Bubblewrap")
@pytest.mark.anyio
@pytest.mark.parametrize("python", [False, True])
async def test_real_linux_execution_and_containment(tmp_path, monkeypatch, python):
    import shlex

    workspace = open_workspace(tmp_path / "workspace", create=True)
    (workspace.root / "input.txt").write_text("hello")
    outside = tmp_path / "secret.txt"
    outside.write_text("private")
    (workspace.root / "alias").symlink_to(outside)
    monkeypatch.setenv("PRAXIS_TEST_SECRET", "must-not-leak")
    code = """
import os, socket
from pathlib import Path
assert Path("input.txt").read_text() == "hello"
assert "PRAXIS_TEST_SECRET" not in os.environ
for name in ("input.txt", ".git/config", "new.txt"):
    try:
        Path(name).write_text("forbidden")
    except OSError:
        pass
    else:
        raise AssertionError("write escaped: " + name)
try:
    Path("alias").read_text()
except OSError:
    pass
else:
    raise AssertionError("host read escaped")
with socket.socket() as sock:
    sock.settimeout(0.2)
    assert sock.connect_ex(("1.1.1.1", 443)) != 0
Path(os.environ["TMPDIR"], "allowed").write_text("temporary")
print("linux-sandbox-ok")
"""
    if python:
        result = await shell._execute_managed_python(
            workspace,
            shell.ManagedPythonInput(code=code),
            termination_grace_seconds=0.1,
        )
    else:
        result = await shell._run_command(
            workspace,
            shell.RunCommandInput(command="python3 -c " + shlex.quote(code)),
            termination_grace_seconds=0.1,
        )
    assert result.exit_code == 0, result.stderr
    assert result.sandbox_error is None
    assert result.stdout.strip() == "linux-sandbox-ok"
    assert (workspace.root / "input.txt").read_text() == "hello"
    assert outside.read_text() == "private"


@pytest.mark.anyio
async def test_linux_pathname_socket_is_refused(tmp_path, monkeypatch):
    import socket

    monkeypatch.setattr(shell, "_SANDBOX_PLATFORM", "linux")
    monkeypatch.setattr(shell, "_BWRAP_PATH", "/bin/sh")
    import tempfile

    with tempfile.TemporaryDirectory(prefix="px-", dir="/tmp") as short_path:
        workspace = open_workspace(Path(short_path), create=True)
        with socket.socket(socket.AF_UNIX) as sock:
            sock.bind(str(workspace.root / "host.sock"))
            result = await shell._run_command(
                workspace,
                shell.RunCommandInput(command="echo must-not-run"),
                termination_grace_seconds=0.1,
            )
    assert result.sandbox_error == "workspace_socket_detected"


@pytest.mark.skipif(shell.sys.platform != "linux", reason="requires real Linux Bubblewrap")
@pytest.mark.anyio
@pytest.mark.parametrize("python", [False, True])
async def test_real_linux_timeout(tmp_path, python):
    workspace = open_workspace(tmp_path / "workspace", create=True)
    if python:
        result = await shell._execute_managed_python(
            workspace,
            shell.ManagedPythonInput(code="import time; time.sleep(60)", timeout_seconds=0.2),
            termination_grace_seconds=0.1,
        )
    else:
        result = await shell._run_command(
            workspace,
            shell.RunCommandInput(command="sleep 60 & wait", timeout_seconds=0.2),
            termination_grace_seconds=0.1,
        )
    assert result.sandbox_error is None, result.stderr
    assert result.timed_out
    assert result.exit_code != 0
    assert result.duration_ms < 5000


def test_runtime_under_tmp_remains_visible(tmp_path):
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    argv = shell._linux_sandbox_argv(
        workspace_root=tmp_path / "workspace",
        temporary_root=tmp_path / "workspace" / "scratch",
        additional_read_roots=(runtime,),
    )
    assert argv.index("--tmpfs") < argv.index(str(runtime))


@pytest.mark.anyio
async def test_linux_fifo_is_refused(tmp_path, monkeypatch):
    import os

    monkeypatch.setattr(shell, "_SANDBOX_PLATFORM", "linux")
    monkeypatch.setattr(shell, "_BWRAP_PATH", "/bin/sh")
    workspace = open_workspace(tmp_path, create=True)
    os.mkfifo(tmp_path / "host.fifo")
    result = await shell._run_command(
        workspace,
        shell.RunCommandInput(command="echo forbidden > host.fifo"),
        termination_grace_seconds=0.1,
    )
    assert result.sandbox_error == "workspace_special_file_detected"


@pytest.mark.skipif(shell.sys.platform != "linux", reason="requires real Linux Bubblewrap")
@pytest.mark.anyio
@pytest.mark.parametrize("python", [False, True])
async def test_real_linux_cancellation_reaps_launcher(tmp_path, monkeypatch, python):
    import asyncio
    import os

    workspace = open_workspace(tmp_path / "workspace", create=True)
    started = asyncio.Event()
    launcher = None
    original = asyncio.create_subprocess_exec

    async def capture(*args, **kwargs):
        nonlocal launcher
        launcher = await original(*args, **kwargs)
        started.set()
        return launcher

    monkeypatch.setattr(asyncio, "create_subprocess_exec", capture)
    if python:
        pending = shell._execute_managed_python(
            workspace,
            shell.ManagedPythonInput(code="import time; time.sleep(60)"),
            termination_grace_seconds=0.1,
        )
    else:
        pending = shell._run_command(
            workspace,
            shell.RunCommandInput(command="sleep 60 & wait"),
            termination_grace_seconds=0.1,
        )
    task = asyncio.create_task(pending)
    await asyncio.wait_for(started.wait(), timeout=5)
    await asyncio.sleep(0.1)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(task, timeout=5)
    assert launcher is not None and launcher.returncode is not None
    with pytest.raises(ProcessLookupError):
        os.killpg(launcher.pid, 0)
