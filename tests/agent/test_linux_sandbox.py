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


def test_linux_explicit_network_uses_filter_and_keeps_other_namespaces(tmp_path):
    argv = shell._linux_sandbox_argv(
        workspace_root=tmp_path, temporary_root=tmp_path / "scratch",
        additional_read_roots=(), allow_network=True, network_filter_fd=17,
    )
    assert "--unshare-all" in argv
    assert "--share-net" in argv
    assert argv[argv.index("--seccomp"):][:2] == ["--seccomp", "17"]
    assert ["--ro-bind", "/etc/resolv.conf", "/etc/resolv.conf"] == argv[
        argv.index("/etc/resolv.conf") - 1:][:3]


def test_linux_network_without_ipc_filter_fails_closed(tmp_path):
    with pytest.raises(ValueError, match="seccomp"):
        shell._linux_sandbox_argv(
            workspace_root=tmp_path, temporary_root=tmp_path / "scratch",
            additional_read_roots=(), allow_network=True,
        )


def test_linux_workspace_changes_publish_only_after_validation(tmp_path):
    workspace = open_workspace(tmp_path / "workspace", create=True)
    (workspace.root / "original.txt").write_text("before")
    view = tmp_path / "view"
    initial, protected = shell._prepare_linux_workspace_view(workspace, view)
    (view / "original.txt").write_text("after")
    (view / "nested").mkdir()
    (view / "nested/new.txt").write_bytes(b"new")
    assert (workspace.root / "original.txt").read_text() == "before"
    shell._commit_linux_workspace_view(workspace, view, initial, protected, workspace.permissions)
    assert (workspace.root / "original.txt").read_text() == "after"
    assert (workspace.root / "nested/new.txt").read_bytes() == b"new"


@pytest.mark.parametrize("name", [".git", ".agents", ".codex", ".aws", ".venv", "node_modules"])
def test_linux_new_protected_paths_reject_the_entire_publish(tmp_path, name):
    workspace = open_workspace(tmp_path / "workspace", create=True)
    (workspace.root / "original.txt").write_text("before")
    view = tmp_path / "view"
    initial, protected = shell._prepare_linux_workspace_view(workspace, view)
    (view / "original.txt").write_text("after")
    (view / "nested" / name).mkdir(parents=True)
    (view / "nested" / name / "config").write_text("forbidden")
    with pytest.raises(ValueError, match="protected"):
        shell._commit_linux_workspace_view(workspace, view, initial, protected, workspace.permissions)
    assert (workspace.root / "original.txt").read_text() == "before"
    assert not (workspace.root / "nested").exists()


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


def test_linux_candidate_refuses_concurrent_host_change(tmp_path):
    workspace = open_workspace(tmp_path / "workspace", create=True)
    (workspace.root / "original.txt").write_text("before")
    view = tmp_path / "view"
    initial, protected = shell._prepare_linux_workspace_view(workspace, view)
    (view / "original.txt").write_text("candidate")
    (workspace.root / "original.txt").write_text("concurrent")
    with pytest.raises(ValueError, match="changed while"):
        shell._commit_linux_workspace_view(workspace, view, initial, protected, workspace.permissions)
    assert (workspace.root / "original.txt").read_text() == "concurrent"


@pytest.mark.parametrize("hardlink", [False, True])
def test_linux_candidate_refuses_alias_output(tmp_path, hardlink):
    import os

    workspace = open_workspace(tmp_path / "workspace", create=True)
    outside = tmp_path / "outside"
    outside.write_text("private")
    view = tmp_path / "view"
    initial, protected = shell._prepare_linux_workspace_view(workspace, view)
    if hardlink:
        os.link(outside, view / "alias")
    else:
        (view / "alias").symlink_to(outside)
    with pytest.raises(ValueError, match="hardlink|symlink"):
        shell._commit_linux_workspace_view(workspace, view, initial, protected, workspace.permissions)
    assert not (workspace.root / "alias").exists()
    assert outside.read_text() == "private"


def test_linux_view_preserves_existing_directory_modes(tmp_path):
    workspace = open_workspace(tmp_path / "workspace", create=True)
    (workspace.root / "ordinary").mkdir(mode=0o700)
    view = tmp_path / "view"
    initial, protected = shell._prepare_linux_workspace_view(workspace, view)
    assert {name: entry[:2] for name, entry in shell._linux_view_snapshot(view, protected).items()} == {
        name: entry[:2] for name, entry in initial.items()}


def test_linux_view_file_data_limit_fails_closed(tmp_path, monkeypatch):
    workspace = open_workspace(tmp_path / "workspace", create=True)
    (workspace.root / "large").write_bytes(b"1234")
    monkeypatch.setattr(shell, "_LINUX_VIEW_MAX_BYTES", 3)
    with pytest.raises(ValueError, match="64 MiB"):
        shell._prepare_linux_workspace_view(workspace, tmp_path / "view")


@pytest.mark.anyio
@pytest.mark.parametrize("python", [False, True])
async def test_linux_missing_ipc_filter_never_launches_network(tmp_path, monkeypatch, python):
    monkeypatch.setattr(shell, "_SANDBOX_PLATFORM", "linux")
    monkeypatch.setattr(shell, "_BWRAP_PATH", "/bin/sh")
    def unavailable():
        raise ValueError("Linux network isolation requires libseccomp.so.2")
    monkeypatch.setattr(shell, "_linux_network_filter_file", unavailable)
    async def unexpected_spawn(*args, **kwargs):
        pytest.fail("missing IPC filter must fail before process launch")
    monkeypatch.setattr(shell.asyncio, "create_subprocess_exec", unexpected_spawn)
    workspace = open_workspace(tmp_path / "workspace", create=True)
    if python:
        result = await shell._execute_managed_python(
            workspace, shell.ManagedPythonInput(code="pass", network=True), termination_grace_seconds=0.1)
    else:
        result = await shell._run_command(
            workspace, shell.RunCommandInput(command="true", network=True), termination_grace_seconds=0.1)
    assert result.sandbox_error == "sandbox_policy_unsupported"
    assert "libseccomp" in result.stderr


@pytest.mark.skipif(shell.sys.platform != "linux", reason="requires real Linux Bubblewrap")
@pytest.mark.anyio
@pytest.mark.parametrize("python", [False, True])
@pytest.mark.parametrize("network", [False, True])
async def test_real_linux_explicit_network_and_host_ipc_boundary(tmp_path, python, network):
    import shlex
    import socket
    import threading
    from http.server import BaseHTTPRequestHandler, HTTPServer

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"network-ok")
        def log_message(self, *args):
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    abstract = "\0praxis-test-" + str(server.server_port)
    host_socket = socket.socket(socket.AF_UNIX)
    host_socket.bind(abstract)
    host_socket.listen()
    code = f'''
import socket, urllib.request
try:
    response = urllib.request.urlopen("http://localhost:{server.server_port}", timeout=1).read()
except OSError:
    assert {network!r} is False
else:
    assert {network!r} is True
    assert response == b"network-ok"
try:
    local = socket.socket(socket.AF_UNIX)
    local.settimeout(0.2)
    local.connect({abstract!r})
except OSError:
    pass
else:
    raise AssertionError("abstract Unix host IPC escaped")
finally:
    if 'local' in locals(): local.close()
if {network!r}:
    import asyncio
    async def fetch_async():
        assert await asyncio.to_thread(lambda: "worker-ok") == "worker-ok"
        reader, writer = await asyncio.open_connection("localhost", {server.server_port})
        writer.write(b"GET / HTTP/1.0\\r\\nHost: localhost\\r\\n\\r\\n")
        await writer.drain()
        response = await reader.read()
        writer.close()
        await writer.wait_closed()
        assert response.endswith(b"network-ok"), response
    asyncio.run(fetch_async())
    # Only anonymous, already-connected stream pairs may support asyncio's
    # self-pipe. Neither endpoint may be repurposed to reach host abstract IPC.
    for flags in (0, socket.SOCK_CLOEXEC, socket.SOCK_NONBLOCK, socket.SOCK_CLOEXEC | socket.SOCK_NONBLOCK):
        left, right = socket.socketpair(socket.AF_UNIX, socket.SOCK_STREAM | flags)
        try:
            left.settimeout(0.2)
            right.settimeout(0.2)
            left.sendall(b"pair-ok")
            assert right.recv(32) == b"pair-ok"
            for peer in (left, right):
                try:
                    peer.connect({abstract!r})
                except OSError: pass
                else: raise AssertionError("anonymous stream pair connected to host IPC")
                try:
                    sent = peer.sendto(b"pair-only", {abstract!r})
                except OSError: pass
                else:
                    other = right if peer is left else left
                    assert other.recv(32) == b"pair-only"[:sent]
        finally:
            left.close()
            right.close()
    for kind in (socket.SOCK_DGRAM, socket.SOCK_SEQPACKET):
        try:
            pair = socket.socketpair(socket.AF_UNIX, kind)
        except OSError: pass
        else:
            for peer in pair: peer.close()
            raise AssertionError("non-stream Unix socketpair was allowed")
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        try:
            listener.bind(("127.0.0.1", 0))
            listener.listen()
        except OSError:
            pass
        else:
            raise AssertionError("outbound grant exposed a host listener")
print("network-boundary-ok")
'''
    workspace = open_workspace(tmp_path / "workspace", create=True)
    try:
        if python:
            result = await shell._execute_managed_python(
                workspace, shell.ManagedPythonInput(code=code, network=network), termination_grace_seconds=0.1)
        else:
            result = await shell._run_command(
                workspace, shell.RunCommandInput(command="python3 -c " + shlex.quote(code), network=network),
                termination_grace_seconds=0.1)
        host_socket.settimeout(0.1)
        try:
            accepted, _ = host_socket.accept()
        except TimeoutError:
            pass
        else:
            accepted.close()
            raise AssertionError("sandbox reached host abstract socket")
    finally:
        host_socket.close()
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
    assert result.exit_code == 0, result.stderr
    assert result.network_enabled is network
    assert result.stdout.strip() == "network-boundary-ok"


@pytest.mark.skipif(shell.sys.platform != "linux", reason="requires real Linux Bubblewrap")
@pytest.mark.anyio
@pytest.mark.parametrize("python", [False, True])
@pytest.mark.parametrize("write", [False, True])
async def test_real_linux_workspace_writes_and_protected_boundaries(tmp_path, python, write):
    import shlex

    from agent_runtime.permission_policy import PathRule, PermissionPolicy

    workspace = open_workspace(tmp_path / "workspace", create=True)
    outside = tmp_path / "outside.txt"
    outside.write_text("private")
    readonly = tmp_path / "readonly.txt"
    readonly.write_text("authorized-read")
    permissions = PermissionPolicy(filesystem=(PathRule(path=str(readonly), access="read"),)).resolve(workspace.root)
    workspace.permission_provider = lambda: permissions
    (workspace.root / "original.txt").write_text("before")
    for name in (".git", ".agents", ".codex", ".aws", ".venv", "node_modules"):
        (workspace.root / name).mkdir()
        (workspace.root / name / "config").write_text("protected")
    code = f'''
from pathlib import Path
for name in (".git", ".agents", ".codex", ".aws", ".venv", "node_modules"):
    assert Path(name, "config").read_text() == "protected", "host metadata/toolchain became invisible: " + name
    try:
        Path(name, "config").write_text("forbidden")
    except OSError: pass
    else: raise AssertionError("protected write escaped: " + name)
# The host sibling is absent from the namespace. The same absolute pathname
# can be created in bwrap's private /tmp without touching the host inode.
private_name = Path({str(outside)!r})
try:
    private_name.read_text()
except OSError: pass
else: raise AssertionError("ungranted host sibling became readable")
private_name.write_text("namespace-only")
assert private_name.read_text() == "namespace-only"
# In contrast, this host inode is explicitly mounted read-only and must reject writes.
readonly_name = Path({str(readonly)!r})
assert readonly_name.read_text() == "authorized-read"
try:
    readonly_name.write_text("forbidden")
except OSError: pass
else: raise AssertionError("external read-only host grant became writable")
try:
    Path("original.txt").write_text("after")
    Path("nested").mkdir()
    Path("nested/new.bin").write_bytes(b"new")
except OSError:
    assert {write!r} is False
else:
    assert {write!r} is True
print("write-boundary-ok")
'''
    if python:
        result = await shell._execute_managed_python(
            workspace, shell.ManagedPythonInput(code=code, workspace_write=write,
                                                output_paths=["nested/new.bin"] if write else []),
            termination_grace_seconds=0.1)
    else:
        result = await shell._run_command(
            workspace, shell.RunCommandInput(command="python3 -c " + shlex.quote(code), workspace_write=write),
            termination_grace_seconds=0.1)
    # Check host evidence even when an assertion inside the child fails.
    assert outside.read_text() == "private"
    assert readonly.read_text() == "authorized-read"
    assert result.exit_code == 0, result.stderr
    assert (workspace.root / "original.txt").read_text() == ("after" if write else "before")
    assert (workspace.root / "nested/new.bin").exists() is write
    for name in (".git", ".agents", ".codex", ".aws", ".venv", "node_modules"):
        assert (workspace.root / name / "config").read_text() == "protected"


@pytest.mark.skipif(shell.sys.platform != "linux", reason="requires real Linux Bubblewrap")
@pytest.mark.anyio
@pytest.mark.parametrize("python", [False, True])
@pytest.mark.parametrize("failure", ["exit", "protected", "timeout"])
async def test_real_linux_failed_candidate_never_publishes(tmp_path, python, failure):
    import shlex

    workspace = open_workspace(tmp_path / "workspace", create=True)
    (workspace.root / "original.txt").write_text("before")
    code = 'from pathlib import Path; Path("original.txt").write_text("after"); '
    if failure == "exit":
        code += "raise RuntimeError('deliberate failure')"
    elif failure == "timeout":
        code += "import time; time.sleep(60)"
    else:
        code += 'Path("nested/.codex").mkdir(parents=True); Path("nested/.codex/config").write_text("forbidden")'
    if python:
        result = await shell._execute_managed_python(
            workspace, shell.ManagedPythonInput(code=code, workspace_write=True, output_paths=["original.txt"],
                                                timeout_seconds=0.5 if failure == "timeout" else 120),
            termination_grace_seconds=0.1)
    else:
        result = await shell._run_command(
            workspace, shell.RunCommandInput(command="python3 -c " + shlex.quote(code), workspace_write=True,
                                             timeout_seconds=0.5 if failure == "timeout" else 120),
            termination_grace_seconds=0.1)
    assert result.exit_code != 0, result.stderr
    assert (workspace.root / "original.txt").read_text() == "before"
    assert not (workspace.root / "nested").exists()
    if failure == "protected":
        assert result.sandbox_error == "sandbox_publish_denied"


@pytest.mark.parametrize("concurrent_change", ["content", "identity", "permission"])
def test_linux_publish_rechecks_each_target_after_tree_validation(tmp_path, monkeypatch, concurrent_change):
    from agent_runtime.permission_policy import PermissionPolicy

    workspace = open_workspace(tmp_path / "workspace", create=True)
    target = workspace.root / "original.txt"
    target.write_text("before")
    policy = PermissionPolicy().resolve(workspace.root)
    workspace.permission_provider = lambda: policy
    view = tmp_path / "view"
    initial, protected = shell._prepare_linux_workspace_view(workspace, view)
    (view / "original.txt").write_text("candidate")
    original_snapshot = shell._linux_view_snapshot
    def snapshot_then_concurrent_change(root, excluded):
        nonlocal policy
        result = original_snapshot(root, excluded)
        if root == workspace.root:
            if concurrent_change == "content":
                target.write_text("concurrent")
            elif concurrent_change == "identity":
                replacement = tmp_path / "replacement"
                replacement.write_text("before")
                shell.os.replace(replacement, target)
            else:
                policy = PermissionPolicy(profile="read-only").resolve(workspace.root)
        return result
    monkeypatch.setattr(shell, "_linux_view_snapshot", snapshot_then_concurrent_change)
    with pytest.raises(ValueError, match="changed|permission"):
        shell._commit_linux_workspace_view(workspace, view, initial, protected, workspace.permissions)
    assert target.read_text() == ("concurrent" if concurrent_change == "content" else "before")


def test_linux_publish_does_not_follow_concurrent_host_parent_symlink(tmp_path, monkeypatch):
    workspace = open_workspace(tmp_path / "workspace", create=True)
    (workspace.root / "nested").mkdir()
    (workspace.root / "nested/result.txt").write_text("before")
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "result.txt").write_text("private")
    view = tmp_path / "view"
    initial, protected = shell._prepare_linux_workspace_view(workspace, view)
    (view / "nested/result.txt").write_text("candidate")
    original_snapshot = shell._linux_view_snapshot
    def snapshot_then_parent_alias(root, excluded):
        result = original_snapshot(root, excluded)
        if root == workspace.root:
            (root / "nested").rename(root / "moved")
            (root / "nested").symlink_to(outside)
        return result
    monkeypatch.setattr(shell, "_linux_view_snapshot", snapshot_then_parent_alias)
    with pytest.raises((OSError, ValueError)):
        shell._commit_linux_workspace_view(workspace, view, initial, protected, workspace.permissions)
    assert (outside / "result.txt").read_text() == "private"
    assert (workspace.root / "moved/result.txt").read_text() == "before"


def test_linux_publish_uses_pinned_parent_at_replace_boundary(tmp_path, monkeypatch):
    workspace = open_workspace(tmp_path / "workspace", create=True)
    (workspace.root / "nested").mkdir()
    (workspace.root / "nested/result.txt").write_text("before")
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "result.txt").write_text("private")
    view = tmp_path / "view"
    initial, protected = shell._prepare_linux_workspace_view(workspace, view)
    (view / "nested/result.txt").write_text("candidate")
    original_replace = shell.os.replace
    def replace_after_parent_alias(source, destination, **kwargs):
        (workspace.root / "nested").rename(workspace.root / "moved")
        (workspace.root / "nested").symlink_to(outside)
        (outside / Path(source).name).write_text("injected")
        return original_replace(source, destination, **kwargs)
    monkeypatch.setattr(shell.os, "replace", replace_after_parent_alias)
    shell._commit_linux_workspace_view(workspace, view, initial, protected, workspace.permissions)
    assert (outside / "result.txt").read_text() == "private"



def test_linux_snapshot_does_not_read_file_replaced_with_symlink(tmp_path, monkeypatch):
    workspace = open_workspace(tmp_path / "workspace", create=True)
    target = workspace.root / "ordinary.txt"
    target.write_text("before")
    outside = tmp_path / "outside.txt"
    outside.write_text("secret")
    original_open = shell.os.open
    original_path_open = Path.open
    def alias():
        target.unlink()
        target.symlink_to(outside)
    def open_after_alias(path, flags, *args, **kwargs):
        if str(path) == "ordinary.txt" and kwargs.get("dir_fd") is not None:
            alias()
        return original_open(path, flags, *args, **kwargs)
    def path_open_after_alias(path, *args, **kwargs):
        if path == target:
            alias()
        return original_path_open(path, *args, **kwargs)
    monkeypatch.setattr(shell.os, "open", open_after_alias)
    monkeypatch.setattr(Path, "open", path_open_after_alias)
    with pytest.raises((OSError, ValueError)):
        shell._linux_view_snapshot(workspace.root, {".praxis", ".agent_memory"})
