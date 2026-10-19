from __future__ import annotations

import os

import pytest

from agent_runtime.tools.builtins import shell
from agent_runtime.workspace import open_workspace


@pytest.mark.parametrize("target", ["source.ml", "../source.ml", "../later.ml"])
def test_publish_relative_build_symlink(tmp_path, target):
    workspace = open_workspace(tmp_path / "workspace", create=True)
    (workspace.root / "source.ml").write_text("source")
    (workspace.root / "build").mkdir()
    view = tmp_path / "view"
    initial, protected = shell._prepare_linux_workspace_view(workspace, view)
    (view / "build/link.ml").symlink_to(target)
    shell._commit_linux_workspace_view(workspace, view, initial, protected, workspace.permissions)
    assert os.readlink(workspace.root / "build/link.ml") == target


@pytest.mark.parametrize("target", ["../../outside", "/etc/passwd", "../.git/config", "../alias/secret"])
def test_reject_escaping_build_symlink_before_any_publish(tmp_path, target):
    workspace = open_workspace(tmp_path / "workspace", create=True)
    (workspace.root / "source").write_text("before")
    (workspace.root / "alias").symlink_to(tmp_path)
    view = tmp_path / "view"
    initial, protected = shell._prepare_linux_workspace_view(workspace, view)
    (view / "source").write_text("after")
    (view / "build").mkdir()
    (view / "build/link").symlink_to(target)
    with pytest.raises(ValueError, match="symlink|protected"):
        shell._commit_linux_workspace_view(workspace, view, initial, protected, workspace.permissions)
    assert (workspace.root / "source").read_text() == "before"
    assert not (workspace.root / "build").exists()


def test_copy_and_publish_preserve_file_mtime_for_incremental_builds(tmp_path):
    workspace = open_workspace(tmp_path / "workspace", create=True)
    source = workspace.root / "source.c"
    source.write_text("source")
    timestamp = 1_600_000_000_000_000_000
    os.utime(source, ns=(timestamp, timestamp))
    view = tmp_path / "view"
    initial, protected = shell._prepare_linux_workspace_view(workspace, view)
    assert (view / "source.c").stat().st_mtime_ns == timestamp
    (view / "source.o").write_bytes(b"object")
    os.utime(view / "source.o", ns=(timestamp + 100, timestamp + 100))
    shell._commit_linux_workspace_view(workspace, view, initial, protected, workspace.permissions)
    assert (workspace.root / "source.o").stat().st_mtime_ns == timestamp + 100


def test_publish_timestamp_only_change_and_refuse_concurrent_touch(tmp_path):
    workspace = open_workspace(tmp_path / "workspace", create=True)
    source = workspace.root / "source"
    source.write_text("unchanged")
    view = tmp_path / "view"
    initial, protected = shell._prepare_linux_workspace_view(workspace, view)
    stamp = source.stat().st_mtime_ns + 1000
    os.utime(view / "source", ns=(stamp, stamp))
    shell._commit_linux_workspace_view(workspace, view, initial, protected, workspace.permissions)
    assert source.stat().st_mtime_ns == stamp
    view2 = tmp_path / "view2"
    initial, protected = shell._prepare_linux_workspace_view(workspace, view2)
    (view2 / "source").write_text("candidate")
    os.utime(source, ns=(stamp + 1000, stamp + 1000))
    with pytest.raises(ValueError, match="changed while"):
        shell._commit_linux_workspace_view(workspace, view2, initial, protected, workspace.permissions)
    assert source.read_text() == "unchanged"


@pytest.mark.parametrize("operation", ["retarget", "delete", "chain", "cycle", "absolute_inside"])
def test_link_replacement_and_graph_validation(tmp_path, operation):
    workspace = open_workspace(tmp_path / "workspace", create=True)
    (workspace.root / "old").write_text("old")
    (workspace.root / "link").symlink_to("old")
    view = tmp_path / "view"
    initial, protected = shell._prepare_linux_workspace_view(workspace, view)
    (view / "link").unlink()
    if operation == "retarget":
        (view / "new").write_text("new")
        (view / "link").symlink_to("new")
    elif operation == "chain":
        (view / "new").symlink_to("old")
        (view / "link").symlink_to("new")
    elif operation == "cycle":
        (view / "new").symlink_to("link")
        (view / "link").symlink_to("new")
    elif operation == "absolute_inside":
        (view / "link").symlink_to(workspace.root / "old")
    if operation in {"cycle", "absolute_inside"}:
        with pytest.raises(ValueError, match="symlink"):
            shell._commit_linux_workspace_view(workspace, view, initial, protected, workspace.permissions)
        assert os.readlink(workspace.root / "link") == "old"
    else:
        shell._commit_linux_workspace_view(workspace, view, initial, protected, workspace.permissions)
        if operation == "delete":
            assert not (workspace.root / "link").is_symlink()
        else:
            assert (workspace.root / "link").read_text() == ("new" if operation == "retarget" else "old")


def test_symlink_target_requires_read_permission(tmp_path):
    from agent_runtime.permission_policy import PathRule, PermissionPolicy

    workspace = open_workspace(tmp_path / "workspace", create=True)
    private = workspace.root / "private"
    private.write_text("secret")
    policy = PermissionPolicy(filesystem=(PathRule(path=str(private), access="deny"),)).resolve(workspace.root)
    workspace.permission_provider = lambda: policy
    view = tmp_path / "view"
    initial, protected = shell._prepare_linux_workspace_view(workspace, view)
    (view / "link").symlink_to("private")
    with pytest.raises(ValueError, match="permission"):
        shell._commit_linux_workspace_view(workspace, view, initial, protected, policy)
    assert not (workspace.root / "link").is_symlink()


@pytest.mark.anyio
@pytest.mark.parametrize("python", [False, True])
async def test_startup_failure_does_not_publish_with_opt_in(tmp_path, monkeypatch, python):
    launcher = tmp_path / "bwrap"
    launcher.write_text('#!/bin/sh\nprintf "bwrap: namespace denied\\n" >&2\nexit 1\n')
    launcher.chmod(0o755)
    monkeypatch.setattr(shell, "_SANDBOX_PLATFORM", "linux")
    monkeypatch.setattr(shell, "_BWRAP_PATH", str(launcher))
    workspace = open_workspace(tmp_path / "workspace", create=True)
    if python:
        result = await shell._execute_managed_python(workspace, shell.ManagedPythonInput(
            code="pass", workspace_write=True, publish_on_failure=True, output_paths=["result"],
        ), termination_grace_seconds=0.1)
    else:
        result = await shell._run_command(workspace, shell.RunCommandInput(
            command="true", workspace_write=True, publish_on_failure=True,
        ), termination_grace_seconds=0.1)
    assert result.sandbox_error == "sandbox_start_failed"
    assert result.workspace_published is False


@pytest.mark.skipif(shell.sys.platform != "linux", reason="requires real Linux Bubblewrap")
@pytest.mark.anyio
@pytest.mark.parametrize("python", [False, True])
@pytest.mark.parametrize("publish", [False, True])
@pytest.mark.parametrize("failure", ["exit", "timeout", "unsafe_link"])
async def test_real_linux_partial_publication_is_explicit_and_still_checked(tmp_path, python, publish, failure):
    import shlex

    workspace = open_workspace(tmp_path / "workspace", create=True)
    (workspace.root / "source").write_text("before")
    code = 'from pathlib import Path; Path("source").write_text("after"); '
    if failure == "exit":
        code += "raise SystemExit(7)"
    elif failure == "timeout":
        code += "import time; time.sleep(60)"
    else:
        code += 'Path("alias").symlink_to("/etc/passwd")'
    options = dict(workspace_write=True, publish_on_failure=publish,
                   timeout_seconds=1 if failure == "timeout" else 10)
    if python:
        result = await shell._execute_managed_python(
            workspace, shell.ManagedPythonInput(code=code, output_paths=["source"], **options),
            termination_grace_seconds=0.1,
        )
        normalized = shell._normalize_managed_python_output(result)
    else:
        result = await shell._run_command(
            workspace, shell.RunCommandInput(command="python3 -c " + shlex.quote(code), **options),
            termination_grace_seconds=0.1,
        )
        normalized = shell._normalize_command_output(result)
    published = publish and failure != "unsafe_link"
    assert result.workspace_published is published, result.stderr
    assert (workspace.root / "source").read_text() == ("after" if published else "before")
    assert normalized.is_error
    assert result.exit_code != 0
    if failure == "unsafe_link":
        assert result.sandbox_error == "sandbox_publish_denied"
        assert not (workspace.root / "alias").is_symlink()


@pytest.mark.skipif(shell.sys.platform != "linux", reason="requires real Linux Bubblewrap and make")
@pytest.mark.anyio
async def test_real_linux_configure_and_incremental_make_after_timeout(tmp_path):
    workspace = open_workspace(tmp_path / "workspace", create=True)
    (workspace.root / "source.in").write_text("source")
    (workspace.root / "Makefile").write_text(
        "all: first.o second.o\n"
        "first.o: source.in\n\tcp build/source.in first.o\n\techo first >> count\n"
        "second.o: source.in\n\tcp build/source.in second.o\n\techo second >> count\n"
    )
    first = await shell._run_command(workspace, shell.RunCommandInput(
        command="mkdir build && ln -s ../source.in build/source.in && make first.o && sleep 60",
        workspace_write=True, publish_on_failure=True, timeout_seconds=1,
    ), termination_grace_seconds=0.1)
    assert first.timed_out and first.workspace_published, first.stderr
    assert (workspace.root / "count").read_text().splitlines() == ["first"]
    second = await shell._run_command(workspace, shell.RunCommandInput(
        command="make all", workspace_write=True,
    ), termination_grace_seconds=0.1)
    assert second.exit_code == 0 and second.workspace_published, second.stderr
    assert (workspace.root / "count").read_text().splitlines() == ["first", "second"]


@pytest.mark.skipif(shell.sys.platform != "linux", reason="requires real Linux Bubblewrap")
@pytest.mark.anyio
@pytest.mark.parametrize("python", [False, True])
async def test_real_linux_cancellation_discards_even_with_failure_publication(tmp_path, python):
    import asyncio
    import shlex

    workspace = open_workspace(tmp_path / "workspace", create=True)
    (workspace.root / "source").write_text("before")
    ready = asyncio.Event()
    code = ('from pathlib import Path; import time; Path("source").write_text("after"); '
            'print("ready", flush=True); time.sleep(60)')
    if python:
        # Wait for a scratch signal because managed Python does not stream stdout.
        code = code.replace('print("ready", flush=True)',
                            'import os; Path(os.environ["TMPDIR"], "ready").touch()')
        pending = shell._execute_managed_python(workspace, shell.ManagedPythonInput(
            code=code, workspace_write=True, publish_on_failure=True, output_paths=["source"],
        ), termination_grace_seconds=0.1)
    else:
        async def progress(event):
            if "ready" in event.content:
                ready.set()
        pending = shell._run_command(workspace, shell.RunCommandInput(
            command="python3 -c " + shlex.quote(code), workspace_write=True, publish_on_failure=True,
        ), termination_grace_seconds=0.1, progress_sink=progress)
    task = asyncio.create_task(pending)
    try:
        if python:
            async def await_signal():
                while not list(workspace.scratch.glob("run-python-*/ready")):
                    if task.done():
                        pytest.fail(str(task.result()))
                    await asyncio.sleep(0.02)
            await asyncio.wait_for(await_signal(), 10)
        else:
            await asyncio.wait_for(ready.wait(), 10)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, 5)
    finally:
        if not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
    assert (workspace.root / "source").read_text() == "before"


@pytest.mark.skipif(shell.sys.platform != "linux", reason="requires real Linux Bubblewrap")
@pytest.mark.parametrize("entrypoint", ["sdk", "cli"])
def test_public_linux_approval_recovery_publishes_partial_build_as_tool_failure(tmp_path, monkeypatch, entrypoint):
    import asyncio

    from typer.testing import CliRunner

    from agent_runtime import Agent, cli
    from agent_runtime.harness import HarnessModelResponse, HarnessToolCall, RolloutStore
    from tests.agent.harness.test_public_agent_cutover import PatchThenAnswerModel

    class BuildOnce(PatchThenAnswerModel):
        async def dispatch(self, prepared, **kwargs):
            if prepared.request_ref["request_id"].endswith(":step:1"):
                return HarnessModelResponse(text="", provider_response_id=None,
                    usage={"input_tokens": 5, "output_tokens": 3}, tool_calls=(HarnessToolCall(
                        id="build", name="run_command", arguments={
                            "command": "printf artifact > built; exit 7",
                            "workspace_write": True, "publish_on_failure": True,
                        },
                    ),))
            return HarnessModelResponse(text="Build failed; partial artifact retained.", provider_response_id=None,
                                        usage={"input_tokens": 5, "output_tokens": 3})

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    database = tmp_path / "run.db"  # Runtime DB is outside the candidate workspace.
    if entrypoint == "sdk":
        first = Agent(workspace_path=workspace, checkpoint_db=database, enable_workspace_mcp=False)
        monkeypatch.setattr(first, "_model_factory", BuildOnce)
        paused = asyncio.run(first.run("Build", max_steps=3))
        assert paused.status == "paused"
        fresh = Agent(workspace_path=workspace, checkpoint_db=database, enable_workspace_mcp=False)
        monkeypatch.setattr(fresh, "_model_factory", BuildOnce)
        result = asyncio.run(fresh.resume(paused.turn_id, "allow_once"))
        assert result.status == "done"
        assert result.validation_status == "not_evaluated"
    else:
        original = cli._create_agent_facade
        def create(**kwargs):
            agent = original(**kwargs)
            monkeypatch.setattr(agent, "_model_factory", BuildOnce)
            return agent
        monkeypatch.setattr(cli, "_create_agent_facade", create)
        monkeypatch.chdir(workspace)
        runner = CliRunner()
        runner.invoke(cli.agent_app, ["run", "Build", "--max-steps", "3", "--disable-workspace-mcp",
                                     "--non-interactive", "--rollout-db", str(database)])
        with RolloutStore(database) as store:
            turn = store.list_turns()[-1]
            assert turn.status == "paused"
        result = runner.invoke(cli.agent_app, ["recover", turn.turn_id, "--action", "allow_once",
                                              "--rollout-db", str(database)])
        assert result.exit_code == 0, result.output
    assert (workspace / "built").read_text() == "artifact"
    with RolloutStore(database) as store:
        turn = store.list_turns()[-1]
        results = [i.payload for i in store.list_items(turn.turn_id) if i.kind == "tool_result"]
        assert len(results) == 1
        assert results[0]["is_error"] is True
        assert results[0]["error_code"] == "command_failed"
        assert "workspace_published" in results[0]["model_content"]
        assert store.verify().valid


def test_large_workspace_uses_streamed_file_fingerprints(tmp_path):
    import hashlib
    import tracemalloc

    workspace = open_workspace(tmp_path / "workspace", create=True)
    large = workspace.root / "large.bin"
    with large.open("wb") as stream:
        for _ in range(65):
            stream.write(b"x" * 1024 * 1024)
    tracemalloc.start()
    try:
        view = tmp_path / "view"
        initial, protected = shell._prepare_linux_workspace_view(workspace, view)
        assert (view / "large.bin").stat().st_size == 65 * 1024 * 1024
        # Full file bytes must not be retained in either tree snapshot.
        with large.open("rb") as stream:
            assert initial["large.bin"][1] == hashlib.file_digest(stream, "sha256").digest()
        (view / "small").write_text("changed")
        shell._commit_linux_workspace_view(workspace, view, initial, protected, workspace.permissions)
        assert (workspace.root / "small").read_text() == "changed"
        assert tracemalloc.get_traced_memory()[1] < 12 * 1024 * 1024
    finally:
        tracemalloc.stop()
