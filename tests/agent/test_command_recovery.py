from __future__ import annotations

import asyncio
import json
import os
import shlex
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest

from agent_runtime.harness import HarnessModelResponse, HarnessToolCall
from agent_runtime.tools.builtins import command_worker as journal
from agent_runtime.tools.builtins import shell
from agent_runtime.tools.builtins.command_sessions import DurableCommandSessions
from agent_runtime.workspace import open_workspace
from tests.agent.harness.test_public_agent_cutover import PatchThenAnswerModel

pytestmark = pytest.mark.skipif(
    sys.platform not in {"darwin", "linux"}, reason="requires an actual supported OS sandbox",
)


def request_for(command):
    return shell.RunCommandInput(command=command, timeout_seconds=None, yield_time_seconds=0.01, workspace_write=True)


@pytest.mark.anyio
@pytest.mark.parametrize("released", [False, True])
async def test_sigkill_owner_then_reattach_exact_same_command(tmp_path, released):
    if sys.platform == "linux" and not Path(shell._BWRAP_PATH).is_file():
        pytest.skip("requires bubblewrap")
    workspace = open_workspace(tmp_path / "workspace", create=True)
    request = request_for("printf x >> counter; printf ready; sleep 0.8; printf done")
    script = '''
import asyncio, json, sys
from pathlib import Path
from agent_runtime.tools.builtins.command_sessions import DurableCommandSessions
from agent_runtime.tools.builtins.shell import RunCommandInput
from agent_runtime.workspace import open_workspace
async def main():
    owner = DurableCommandSessions(open_workspace(Path(sys.argv[1])))
    request = RunCommandInput.model_validate_json(sys.argv[2])
    result = await owner.start(request)
    if sys.argv[3] == 'True':
        await owner.after_commit({}, result.model_dump(), 'operation_test')
        while 'ready' not in (await owner.manage(result.session_id, action='status', wait_seconds=0)).stdout:
            await asyncio.sleep(.02)
    print(result.model_dump_json(), flush=True)
    await asyncio.Event().wait()
asyncio.run(main())
'''
    process = subprocess.Popen([sys.executable, "-c", script, str(workspace.root), request.model_dump_json(),
                                str(released)], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    owner = DurableCommandSessions(workspace)
    try:
        assert process.stdout is not None
        line = await asyncio.wait_for(asyncio.to_thread(process.stdout.readline), 10)
        assert line, process.stderr.read() if process.poll() is not None else "parent not ready"
        result = shell.RunCommandOutput.model_validate_json(line)
        process.kill()  # actual SIGKILL; aclose cannot run
        await asyncio.to_thread(process.wait)
        await owner.restore(result.session_id, request=request, digest=result.request_digest,
                            operation_id="operation_test")
        final = await owner.manage(result.session_id, action="wait", wait_seconds=10)
        assert final.exit_code == 0, final
        assert final.stdout == "readydone"
        assert (workspace.root / "counter").read_text() == "x"  # never replay
        assert final.workspace_published is (True if sys.platform == "linux" else None)
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()
        await owner.aclose()


@pytest.mark.anyio
async def test_durable_log_reads_preserve_middle_failure_and_offset(tmp_path):
    workspace = open_workspace(tmp_path / "workspace", create=True)
    owner = DurableCommandSessions(workspace)
    request = request_for(
        "awk 'BEGIN { for(i=0;i<18000;i++)printf \"A\"; "
        "printf \"IMPORTANT_MIDDLE_FAILURE\"; for(i=0;i<18000;i++)printf \"B\"; }'"
    )
    try:
        result = await owner.start(request)
        await owner.after_commit({}, result.model_dump(), "operation_log")
        final = await owner.manage(result.session_id, action="wait", wait_seconds=10)
        assert final.exit_code == 0, final
        assert final.truncated and "IMPORTANT_MIDDLE_FAILURE" not in final.stdout
        middle = await owner.read_log(result.session_id, "stdout", 17990, 100)
        assert "IMPORTANT_MIDDLE_FAILURE" in middle.log_chunk
        assert middle.log_next_offset == 18090 and middle.log_end_offset > 36000
    finally:
        await owner.aclose()


@pytest.mark.anyio
async def test_prepared_job_never_executes_before_durable_commit(tmp_path):
    workspace = open_workspace(tmp_path / "workspace", create=True)
    owner = DurableCommandSessions(workspace)
    try:
        result = await owner.start(request_for("touch unexpected"))
        await asyncio.sleep(0.1)
        path = journal.job_path(workspace.root, result.session_id)
        assert not (path / "started.json").exists()
        assert not (workspace.root / "unexpected").exists()
        await owner.aclose()
        assert not (workspace.root / "unexpected").exists()
    finally:
        await owner.aclose()


def test_deadline_units_are_seconds_without_guessing():
    for seconds in (900, 1000, 3600, 120000):
        request = shell.RunCommandInput(command="true", timeout_seconds=seconds, yield_time_seconds=0.01)
        assert request.timeout_seconds == seconds
    for seconds in (float("nan"), float("inf")):
        with pytest.raises(ValueError):
            shell.RunCommandInput(command="true", timeout_seconds=seconds, yield_time_seconds=0.01)


@pytest.mark.anyio
async def test_recovery_rejects_other_owner_and_request_tampering(tmp_path):
    workspace = open_workspace(tmp_path / "workspace", create=True)
    owner, second = DurableCommandSessions(workspace), DurableCommandSessions(workspace)
    request = request_for("sleep 2")
    try:
        result = await owner.start(request)
        await owner.after_commit({}, result.model_dump(), "operation_test")
        with pytest.raises(RuntimeError, match="another live Agent"):
            await second.restore(result.session_id, request=request, digest=result.request_digest,
                                 operation_id="operation_test")
        with pytest.raises(ValueError, match="does not match"):
            await second.restore(result.session_id, request=request_for("touch unexpected"),
                                 digest=result.request_digest, operation_id="operation_test")
    finally:
        await owner.aclose()
        await second.aclose()


@pytest.mark.anyio
@pytest.mark.parametrize("action", ["wait", "cancel"])
async def test_lost_worker_keeps_identity_and_is_never_replayed(tmp_path, action):
    workspace = open_workspace(tmp_path / "workspace", create=True)
    owner = DurableCommandSessions(workspace)
    request = request_for("touch unexpected")
    try:
        result = await owner.start(request)
        path = journal.job_path(workspace.root, result.session_id)
        # Simulate a worker's committed started marker and released run lock,
        # with no final result. No shell runs in this deterministic failure test.
        journal.write_json(path / "started.json", {})
        await owner.after_commit({}, result.model_dump(), "operation_test")
        await asyncio.sleep(0.1)
        final = await owner.manage(result.session_id, action=action, wait_seconds=2)
        assert final.session_id == result.session_id and final.request_digest == result.request_digest
        assert final.sandbox_error == "command_outcome_unknown"
        assert not (workspace.root / "unexpected").exists()
    finally:
        await owner.aclose()


@pytest.mark.anyio
async def test_policy_revocation_cancels_before_publication_and_hides_old_output(tmp_path):
    from agent_runtime.permission_policy import PermissionPolicy
    workspace = open_workspace(tmp_path / "workspace", create=True)
    policy = PermissionPolicy().resolve(workspace.root)
    workspace.permission_provider = lambda: policy
    owner = DurableCommandSessions(workspace)
    try:
        result = await owner.start(request_for("printf secret; sleep 2; printf new > artifact"))
        await owner.after_commit({}, result.model_dump(), "operation_test")
        policy = PermissionPolicy(profile="read-only").resolve(workspace.root)
        final = await owner.manage(result.session_id, action="wait", wait_seconds=5)
        assert final.sandbox_error == "command_permissions_changed", final
        assert final.session_id == result.session_id and final.stdout == ""
        assert not (workspace.root / "artifact").exists()
    finally:
        await owner.aclose()


@pytest.mark.anyio
async def test_worker_journal_is_hidden_under_explicit_parent_read_grant(tmp_path):
    from agent_runtime.permission_policy import PathRule, PermissionPolicy
    workspace = open_workspace(tmp_path / "workspace", create=True)
    private = journal.private_root(workspace.root)
    secret = private / ("secret_" + os.urandom(8).hex())
    secret.write_text("worker secret")
    secret.chmod(0o600)
    policy = PermissionPolicy(filesystem=(PathRule(path=str(private.parent), access="read"),)).resolve(workspace.root)
    workspace.permission_provider = lambda: policy
    owner = DurableCommandSessions(workspace)
    try:
        result = await owner.start(request_for("cat " + shlex.quote(str(secret))))
        await owner.after_commit({}, result.model_dump(), "operation_test")
        final = await owner.manage(result.session_id, action="wait", wait_seconds=5)
        assert final.exit_code != 0 and "worker secret" not in final.stdout, final
    finally:
        await owner.aclose()
        secret.unlink()


# Scripted provider, real public Session/rollout/sandbox; no model requests.


class CrashResumeModel(PatchThenAnswerModel):
    delay = 1.5
    wait_for_release = False

    def prepare(self, request):
        return replace(super().prepare(request), dispatch_payload=request)

    async def dispatch(self, prepared, **_kwargs):
        request = prepared.dispatch_payload
        outputs = [json.loads(m.content)["structured_content"] for m in request.messages if m.role == "tool"]
        if not outputs:
            wait = ("while [ ! -f command-release ]; do sleep 0.05; done"
                    if self.wait_for_release else f"sleep {self.delay}")
            call = HarnessToolCall(id="launch", name="run_command", arguments={
                "command": f"printf ready; {wait}; printf done", "timeout_seconds": None,
                "yield_time_seconds": 0.01,
            })
        elif len(outputs) == 1:
            call = HarnessToolCall(id="patch", name="apply_patch", arguments={
                "file_path": "source", "old_string": "before", "new_string": "after",
            })
        elif outputs[-1].get("status") != "completed":
            call = HarnessToolCall(id="wait-" + str(request.step), name="manage_command", arguments={
                "session_id": outputs[0]["session_id"], "action": "wait", "wait_seconds": 5,
            })
        else:
            assert outputs[-1]["exit_code"] == 0, outputs[-1]
            return HarnessModelResponse(text="Recovered original command", provider_response_id=None,
                                        usage={"input_tokens": 3, "output_tokens": 3})
        return HarnessModelResponse(text="", provider_response_id=None, tool_calls=(call,),
                                    usage={"input_tokens": 3, "output_tokens": 3})


@pytest.mark.anyio
@pytest.mark.parametrize("inspector", ["sdk", "cli"])
@pytest.mark.parametrize("abort", [False, True])
async def test_public_sdk_restores_ledger_authorized_job_after_sigkill(tmp_path, monkeypatch, inspector, abort):
    from agent_runtime import Agent
    from agent_runtime.harness import RolloutStore

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "source").write_text("before")
    database = tmp_path / "rollout.db"
    script = '''
import asyncio, json, sys
from pathlib import Path
from agent_runtime import Agent
from tests.agent.test_command_recovery import CrashResumeModel
async def main():
    agent = Agent(workspace_path=Path(sys.argv[1]), checkpoint_db=Path(sys.argv[2]), enable_workspace_mcp=False)
    CrashResumeModel.wait_for_release = sys.argv[3] == 'abort'
    if not CrashResumeModel.wait_for_release:
        CrashResumeModel.delay = float(sys.argv[3])
    agent._model_factory = CrashResumeModel
    async with agent.session(allow_execute_tools=True, max_steps=6) as session:
        result = await session.submit('Run a command then edit source')
        assert result.status == 'paused', result
        print(json.dumps({'turn_id': result.turn_id}), flush=True)
        await asyncio.Event().wait()
asyncio.run(main())
'''
    process = subprocess.Popen(
        [sys.executable, "-c", script, str(workspace), str(database), "abort" if abort else "1.5"],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )
    try:
        line = await asyncio.wait_for(asyncio.to_thread(process.stdout.readline), 15)
        assert line, process.stderr.read() if process.poll() is not None else "parent not ready"
        turn_id = json.loads(line)["turn_id"]
        process.kill()
        await asyncio.to_thread(process.wait)
        agent = Agent(workspace_path=workspace, checkpoint_db=database, enable_workspace_mcp=False)
        monkeypatch.setattr(agent, "_model_factory", CrashResumeModel)
        with RolloutStore(database) as store:
            launch = next(i for i in store.list_items(turn_id) if i.kind == "tool_result")
            key = launch.payload["structured_content"]["session_id"]
        if inspector == "sdk":
            status = await agent.command(turn_id, key)
            assert status.session_id == key
            if abort:
                assert status.status == "running"
        else:
            from typer.testing import CliRunner

            from agent_runtime import cli
            monkeypatch.chdir(workspace)
            inspected = await asyncio.to_thread(CliRunner().invoke, cli.agent_app, [
                "commands", turn_id, key, "--rollout-db", str(database),
            ])
            assert inspected.exit_code == 0, inspected.output
            assert json.loads(inspected.output)["session_id"] == key
            if abort:
                assert json.loads(inspected.output)["status"] == "running"
        if abort:
            result = await agent.resume(turn_id, "abort")
            assert result.status == "failed"  # Public AgentResult maps cancelled Turn to failed.
            with RolloutStore(database) as store:
                assert store.read_turn(turn_id).status == "cancelled"
                terminal = next(i for i in store.list_items(turn_id) if i.kind == "tool_continuation")
                assert terminal.payload["structured_content"]["status"] == "cancelled"
                assert store.verify().valid
            assert (workspace / "source").read_text() == "before"
            return
        result = await agent.resume(turn_id, "allow_once")
        assert result.status == "done", result
        with RolloutStore(database) as store:
            outputs = [i.payload for i in store.list_items(turn_id) if i.kind == "tool_result"]
            assert outputs[-1]["structured_content"]["stdout"] == "readydone"
            origin = outputs[-1]["metadata"]["runtime_workspace_origin_operation_id"]
            assert origin == outputs[0]["metadata"]["operation_id"]
            assert store.verify().valid
        assert (workspace / "source").read_text() == "after"
    finally:
        (workspace / "command-release").touch()
        if process.poll() is None:
            process.kill()
            process.wait()


@pytest.mark.anyio
async def test_completed_output_is_withheld_after_policy_change(tmp_path):
    from agent_runtime.permission_policy import PermissionPolicy
    workspace = open_workspace(tmp_path / "workspace", create=True)
    policy = PermissionPolicy().resolve(workspace.root)
    workspace.permission_provider = lambda: policy
    owner = DurableCommandSessions(workspace)
    try:
        started = await owner.start(request_for("printf private"))
        await owner.after_commit({}, started.model_dump(), "operation_policy")
        final = await owner.manage(started.session_id, action="wait", wait_seconds=5)
        assert final.stdout == "private"
        policy = PermissionPolicy(profile="read-only").resolve(workspace.root)
        refused = await owner.read_log(started.session_id, "stdout", 0, 100)
        assert refused.sandbox_error == "command_permissions_changed"
        assert refused.stdout == "" and refused.log_chunk is None
    finally:
        await owner.aclose()


def test_worker_code_snapshot_is_outside_editable_install_and_immutable(tmp_path, monkeypatch):
    source = tmp_path / "editable" / "agent_runtime" / "tools" / "builtins" / "command_worker.py"
    source.parent.mkdir(parents=True)
    source.write_text("original trusted executor")
    monkeypatch.setattr(journal, "__file__", str(source))
    frozen = journal.freeze_code(tmp_path / "editable")
    source.write_text("modified by workspace command")
    copied = frozen / "agent_runtime/tools/builtins/command_worker.py"
    assert copied.read_text() == "original trusted executor"
    assert not frozen.is_relative_to(tmp_path / "editable")


@pytest.mark.skipif(sys.platform != "linux", reason="requires real Linux bubblewrap")
@pytest.mark.anyio
@pytest.mark.parametrize("finish", ["success", "failure", "timeout"])
async def test_linux_large_workspace_background_build_publishes_checked_progress(tmp_path, finish):
    workspace = open_workspace(tmp_path / "workspace", create=True)
    with (workspace.root / "large.bin").open("wb") as output:
        for _ in range(65):
            output.write(b"x" * 1024 * 1024)
    owner = DurableCommandSessions(workspace)
    ending = {"success": "exit 0", "failure": "exit 7", "timeout": "sleep 10"}[finish]
    request = shell.RunCommandInput(
        command="printf partial > build-progress; " + ending,
        workspace_write=True, publish_on_failure=True, yield_time_seconds=0.01,
        timeout_seconds=0.2 if finish == "timeout" else None,
    )
    try:
        result = await owner.start(request)
        await owner.after_commit({}, result.model_dump(), "operation_large")
        result = await owner.manage(result.session_id, action="wait", wait_seconds=10)
        assert result.workspace_published is True, result
        assert result.timed_out is (finish == "timeout")
        assert (result.exit_code == 0) is (finish == "success")
        assert (workspace.root / "build-progress").read_text() == "partial"
        assert (workspace.root / "large.bin").stat().st_size == 65 * 1024 * 1024
    finally:
        await owner.aclose()


@pytest.mark.anyio
async def test_policy_change_at_wait_return_confirms_cancellation_before_detach(tmp_path, monkeypatch):
    from agent_runtime.permission_policy import PermissionPolicy
    from agent_runtime.tools.builtins.command_sessions import CommandSessions
    workspace = open_workspace(tmp_path / "workspace", create=True)
    policy = PermissionPolicy().resolve(workspace.root)
    workspace.permission_provider = lambda: policy
    owner = DurableCommandSessions(workspace)
    try:
        started = await owner.start(request_for("printf ready; sleep 30"))
        await owner.after_commit({}, started.model_dump(), "operation_revoke")
        await asyncio.sleep(0.1)

        async def return_running_then_revoke(self, session_id, **_kwargs):
            nonlocal policy
            policy = PermissionPolicy(profile="read-only").resolve(workspace.root)
            return self._sessions[session_id].snapshot()

        monkeypatch.setattr(CommandSessions, "manage", return_running_then_revoke)
        result = await owner.manage(started.session_id, action="wait", wait_seconds=0.01)
        assert result.sandbox_error == "command_permissions_changed"
        assert not owner.running  # terminal metadata cannot precede actual worker cleanup
        await owner.detach()
    finally:
        await owner.aclose()


@pytest.mark.anyio
async def test_unreadable_current_policy_cancels_instead_of_losing_monitor(tmp_path):
    from agent_runtime.permission_policy import PermissionPolicy
    workspace = open_workspace(tmp_path / "workspace", create=True)
    policy = PermissionPolicy().resolve(workspace.root)
    workspace.permission_provider = lambda: policy
    owner = DurableCommandSessions(workspace)
    try:
        result = await owner.start(request_for("sleep 30"))
        await owner.after_commit({}, result.model_dump(), "operation_bad_policy")

        def unavailable():
            raise ValueError("invalid policy file")

        workspace.permission_provider = unavailable
        result = await owner.manage(result.session_id, action="status", wait_seconds=0)
        assert result.sandbox_error == "command_permissions_changed"
        assert not owner.running
    finally:
        await owner.aclose()
