from __future__ import annotations

import asyncio
import os
import threading
import time
from dataclasses import replace
from pathlib import Path

import pytest

from agent_runtime.tools.builtins import data
from agent_runtime.tools.executor import ToolExecutor
from agent_runtime.tools.permissions import ToolExecutionContext
from agent_runtime.tools.tool import ToolCall, ToolCallOrigin
from agent_runtime.workspace import open_workspace


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.mark.anyio
@pytest.mark.parametrize("operation", ["cancel", "timeout"])
async def test_stopping_inspection_stops_parser_and_reaps_process(tmp_path, monkeypatch, operation):
    """Block inside the parser, not in the caller awaiting its result."""
    marker = tmp_path / "parser.pid"
    release = tmp_path / "release"
    finished = threading.Event()
    original_inspect = data._inspect_data_file

    def blocked_inspect(*args, **kwargs):
        marker.write_text(str(os.getpid()))
        try:
            while not release.exists():
                time.sleep(0.01)
            return original_inspect(*args, **kwargs)
        finally:
            finished.set()

    monkeypatch.setattr(data, "_inspect_data_file", blocked_inspect)
    original_spawn = asyncio.create_subprocess_exec
    children = []
    package_root = Path(__file__).resolve().parents[2]
    # Apply the same controlled blocking parser in the isolated worker. No sleep
    # is used to guess when parsing has started; the marker is the handshake.
    hook = (
        f"import sys; sys.path.insert(0, {str(package_root)!r})\n"
        "import os, time\n"
        "from pathlib import Path\n"
        "from agent_runtime.tools.builtins import data as inspection\n"
        "original_inspect = inspection._inspect_data_file\n"
        "def blocked_inspect(*args, **kwargs):\n"
        f"    Path({str(marker)!r}).write_text(str(os.getpid()))\n"
        f"    while not Path({str(release)!r}).exists(): time.sleep(0.01)\n"
        "    return original_inspect(*args, **kwargs)\n"
        "inspection._inspect_data_file = blocked_inspect\n"
    )

    async def spawn(*arguments, **kwargs):
        arguments = list(arguments)
        position = arguments.index("-c") + 1
        arguments[position] = hook + arguments[position]
        process = await original_spawn(*arguments, **kwargs)
        children.append(process)
        return process

    monkeypatch.setattr(asyncio, "create_subprocess_exec", spawn)
    (tmp_path / "sample.csv").write_text("name,value\nA,1\n")
    workspace = open_workspace(tmp_path)
    tool = replace(data.create_inspect_data_file_tool(workspace), timeout_seconds=3)
    call = ToolCall(
        tool_call_id="inspect", tool_name="inspect_data_file", arguments={"path": "sample.csv"},
        origin=ToolCallOrigin(request_id="inspection", toolset_revision="v1",
                              exposed_tool_names=("inspect_data_file",)),
    )
    task = asyncio.create_task(ToolExecutor({"inspect_data_file": tool}).execute(
        call, context=ToolExecutionContext(workspace_root=tmp_path, cwd=tmp_path),
    ))
    try:
        async with asyncio.timeout(5):
            while not marker.exists():
                if task.done():
                    pytest.fail(f"inspection ended before reaching parser: {await task}")
                await asyncio.sleep(0.01)
        if operation == "cancel":
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(task, timeout=2)
        else:
            execution = await asyncio.wait_for(task, timeout=5)
            assert execution.result.error_code == "timeout_cancelled"
        parser_pid = int(marker.read_text())
        if parser_pid == os.getpid():
            assert finished.is_set(), "cancelled inspection left its parser thread running"
        else:
            with pytest.raises(ProcessLookupError):
                os.kill(parser_pid, 0)
        assert all(child.returncode is not None for child in children)
    finally:
        release.touch()
        if not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        if marker.exists() and int(marker.read_text()) == os.getpid():
            await asyncio.to_thread(finished.wait, 5)
        for child in children:
            if child.returncode is None:
                child.kill()
                await child.wait()


@pytest.mark.anyio
async def test_inspection_worker_preserves_external_grants_and_denials(tmp_path):
    from agent_runtime.permission_policy import PathRule, PermissionPolicy

    workspace = open_workspace(tmp_path / "workspace", create=True)
    external = tmp_path / "external.csv"
    external.write_text("label,value\nexternal,42\n")
    denied = workspace.root / "denied.csv"
    denied.write_text("secret,value\nno,99\n")
    permissions = PermissionPolicy(filesystem=(
        PathRule(path=str(external), access="read"),
        PathRule(path=str(denied), access="deny"),
    )).resolve(workspace.root)
    workspace.permission_provider = lambda: permissions
    tool = data.create_inspect_data_file_tool(workspace)
    result = await tool.run({"path": str(external)})
    assert result.valid and result.tables[0].rows == [["label", "value"], ["external", "42"]]
    with pytest.raises(ValueError):
        await tool.run({"path": str(denied)})


@pytest.mark.anyio
async def test_inspection_worker_isolated_from_workspace_imports_and_host_secrets(tmp_path, monkeypatch):
    (tmp_path / "sample.csv").write_text("name,value\nA,1\n")
    (tmp_path / "csv.py").write_text("raise RuntimeError('workspace module executed')\n")
    monkeypatch.setenv("INSPECTION_TEST_SECRET", "must-not-inherit")
    monkeypatch.setenv("PYTHONPATH", str(tmp_path))
    original_spawn = asyncio.create_subprocess_exec
    observed = []

    async def spawn(*arguments, **kwargs):
        observed.append(kwargs)
        code_position = arguments.index("-c") + 1
        arguments = list(arguments)
        arguments[code_position] = (
            "import os; assert 'INSPECTION_TEST_SECRET' not in os.environ\n" + arguments[code_position]
        )
        return await original_spawn(*arguments, **kwargs)

    monkeypatch.setattr(asyncio, "create_subprocess_exec", spawn)
    result = await data.create_inspect_data_file_tool(open_workspace(tmp_path)).run({"path": "sample.csv"})
    assert result.valid
    assert observed, "inspection must not run on the host executor thread"
    assert all("INSPECTION_TEST_SECRET" not in options["env"] for options in observed)


@pytest.mark.anyio
async def test_cancellation_during_worker_cleanup_is_propagated(tmp_path, monkeypatch):
    """A first cancellation during reaping must not become a successful preview."""
    original_spawn = asyncio.create_subprocess_exec
    cleaning = asyncio.Event()
    release = asyncio.Event()
    children = []

    async def spawn(*arguments, **kwargs):
        process = await original_spawn(*arguments, **kwargs)
        children.append(process)
        original_communicate = process.communicate
        original_wait = process.wait

        async def wait_for_cleanup():
            cleaning.set()
            await release.wait()
            return await original_wait()

        async def communicate(payload):
            result = await original_communicate(payload)
            # communicate has finished reading and waiting for this real worker.
            # Only hold the explicit final cleanup await open for cancellation.
            process.wait = wait_for_cleanup
            return result

        process.communicate = communicate
        return process

    monkeypatch.setattr(asyncio, "create_subprocess_exec", spawn)
    (tmp_path / "sample.csv").write_text("name,value\nA,42\n")
    task = asyncio.create_task(data.create_inspect_data_file_tool(open_workspace(tmp_path)).run({
        "path": "sample.csv",
    }))
    try:
        await asyncio.wait_for(cleaning.wait(), timeout=5)
        task.cancel()
        await asyncio.sleep(0)
        assert not task.done(), "cancellation must wait for cleanup to finish"
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, timeout=2)
        assert all(child.returncode is not None for child in children)
    finally:
        release.set()
        if not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)


async def _execute_inspection(workspace, path):
    tool = data.create_inspect_data_file_tool(workspace)
    call = ToolCall(
        tool_call_id="inspect-error", tool_name="inspect_data_file", arguments={"path": path},
        origin=ToolCallOrigin(request_id="inspection-error", toolset_revision="v1",
                              exposed_tool_names=("inspect_data_file",)),
    )
    return await ToolExecutor({"inspect_data_file": tool}).execute(
        call, context=ToolExecutionContext(workspace_root=workspace.root, cwd=workspace.root),
    )


@pytest.mark.anyio
async def test_missing_inspection_file_keeps_typed_executor_failure(tmp_path):
    execution = await _execute_inspection(open_workspace(tmp_path), "absent.csv")
    assert execution.result.error_code == "file_not_found"
    assert execution.result.retryable is False
    diagnostic = execution.result.metadata["diagnostic"]
    assert diagnostic["exception_type"] == "FileNotFoundError"
    assert "absent.csv" in diagnostic["message"]


@pytest.mark.anyio
@pytest.mark.parametrize("exception_type", ["PermissionError", "WorkspacePathError", "RuntimeError"])
async def test_inspection_worker_preserves_controlled_errors_and_hides_unknown_details(
    tmp_path, monkeypatch, exception_type,
):
    (tmp_path / "sample.csv").write_text("label,value\nA,1\n")
    original_spawn = asyncio.create_subprocess_exec
    children = []
    package_root = Path(__file__).resolve().parents[2]
    detail = "controlled access denied" if exception_type != "RuntimeError" else "PRIVATE_WORKER_DETAILS"
    hook = (
        f"import sys; sys.path.insert(0, {str(package_root)!r})\n"
        "from agent_runtime.tools.builtins import data as inspection\n"
        "from agent_runtime.workspace import WorkspacePathError\n"
        "def failed_inspection(*args, **kwargs):\n"
        f"    raise {exception_type}({detail!r})\n"
        "inspection._inspect_data_file = failed_inspection\n"
    )

    async def spawn(*arguments, **kwargs):
        arguments = list(arguments)
        position = arguments.index("-c") + 1
        arguments[position] = hook + arguments[position]
        process = await original_spawn(*arguments, **kwargs)
        children.append(process)
        return process

    monkeypatch.setattr(asyncio, "create_subprocess_exec", spawn)
    execution = await _execute_inspection(open_workspace(tmp_path), "sample.csv")
    assert execution.result.error_code == "runner_failed"
    diagnostic = execution.result.metadata["diagnostic"]
    assert diagnostic["exception_type"] == exception_type
    if exception_type == "RuntimeError":
        assert diagnostic["message"] == "Data inspection worker failed; no preview was produced."
        assert detail not in str(execution)
    else:
        assert diagnostic["message"] == detail
    assert children and all(child.returncode is not None for child in children)
