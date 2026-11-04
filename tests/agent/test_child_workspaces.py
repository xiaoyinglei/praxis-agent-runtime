"""Dirty source, isolated execution and conservative child integration."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from agent_runtime.worktrees import WorktreeError, WorktreeManager


def git(root, *args):
    return subprocess.run(["git", "-C", str(root), *args], check=True, capture_output=True).stdout


@pytest.fixture
def repo(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    git(root, "init", "-q")
    git(root, "config", "user.name", "Test")
    git(root, "config", "user.email", "test@example.com")
    (root / ".gitignore").write_text(".praxis/\n")
    (root / "code.py").write_text("a = 1\nb = 2\n")
    git(root, "add", ".")
    git(root, "commit", "-qm", "base")
    return root


def test_child_snapshot_preserves_dirty_source_index_and_full_working_content(repo, tmp_path):
    (repo / "code.py").write_text("a = 10\nb = 2\n")
    git(repo, "add", "code.py")
    (repo / "code.py").write_text("a = 10\nb = 20\n")
    (repo / "new.bin").write_bytes(b"\x00\xffnew")
    index_before = (repo / ".git/index").read_bytes()
    status_before = git(repo, "status", "--porcelain=v1", "-z")
    manager = WorktreeManager(repo, root=tmp_path / "trees")
    recorded = []
    report, snapshot = manager.create_child_snapshot(
        "child", identity={"child_turn_id": "child", "parent_turn_id": "parent"}, before_create=recorded.append
    )
    child = report.worktree.root
    assert recorded[0]["workspace"] == str(child)
    assert recorded[0]["source_root"] == str(repo)
    assert snapshot["source_head"] == git(repo, "rev-parse", "HEAD").decode().strip()
    assert (child / "code.py").read_text() == "a = 10\nb = 20\n"
    assert (child / "new.bin").read_bytes() == b"\x00\xffnew"
    assert (repo / ".git/index").read_bytes() == index_before
    assert git(repo, "status", "--porcelain=v1", "-z") == status_before
    assert git(child, "status", "--porcelain=v1", "-z") == b""
    with pytest.raises(WorktreeError, match="子任务|child"):
        manager.remove(str(child))
    assert manager.prune(keep=0) == []
    assert child.is_dir()


@pytest.mark.parametrize("scope", ["subtree", "workspace", "ancestor"])
def test_staged_source_patch_respects_declared_private_regions(repo, scope):
    from agent_runtime.workspace_snapshot import capture_snapshot

    (repo / ".praxis").mkdir()
    (repo / ".praxis/private.txt").write_text("PRIVATE_RUNTIME_CONTENT")
    (repo / "host-config").mkdir()
    (repo / "host-config/settings.json").write_text("PRIVATE_CONFIG_CONTENT")
    (repo / "code.py").write_text("public_staged_change = 1\n")
    git(repo, "add", "-f", ".praxis/private.txt", "host-config/settings.json", "code.py")
    original = (repo / ".git/index").read_bytes()
    excluded = {"subtree": repo / "host-config", "workspace": repo, "ancestor": repo.parent}[scope]
    snapshot = capture_snapshot(repo, excluded=(excluded,))
    staged = bytes.fromhex(snapshot["source_index_patch_hex"])
    if scope == "subtree":
        assert b"public_staged_change" in staged
    else:
        assert staged == b""
        assert snapshot["entries"] == {}
    assert b"PRIVATE_RUNTIME_CONTENT" not in staged
    assert b"PRIVATE_CONFIG_CONTENT" not in staged
    assert ".praxis/private.txt" not in snapshot["entries"]
    assert "host-config/settings.json" not in snapshot["entries"]
    assert bytes.fromhex(snapshot["source_index_hex"]) == original
    assert (repo / ".git/index").read_bytes() == original


@pytest.mark.parametrize("change", ["delete", "rename"])
def test_staged_removal_requires_source_read_authority(repo, change):
    from agent_runtime.workspace_snapshot import capture_snapshot

    denied = repo / "denied.txt"
    denied.write_text("DENIED_OLD_CONTENT")
    git(repo, "add", "denied.txt")
    git(repo, "commit", "-qm", "private source")
    if change == "delete":
        git(repo, "rm", "denied.txt")
    else:
        git(repo, "mv", "denied.txt", "renamed.txt")

    def authorize(path):
        if path == denied:
            raise PermissionError("source path denied")

    with pytest.raises(PermissionError, match="source path denied"):
        capture_snapshot(repo, authorize=authorize)


def test_snapshot_handles_deleted_renamed_executable_and_symlink(repo, tmp_path):
    (repo / "code.py").rename(repo / "renamed.py")
    (repo / "renamed.py").chmod(0o755)
    (repo / "link").symlink_to("renamed.py")
    report, _ = WorktreeManager(repo, root=tmp_path / "trees").create_child_snapshot(
        "child", identity={"child_turn_id": "c"}, before_create=lambda _: None
    )
    child = report.worktree.root
    assert not (child / "code.py").exists()
    assert (child / "renamed.py").stat().st_mode & 0o111
    assert (child / "link").is_symlink()
    assert (child / "link").readlink() == Path("renamed.py")


@pytest.mark.anyio
async def test_session_persists_child_environment_before_dispatch_and_integrates_real_diff(repo, tmp_path, monkeypatch):
    from agent_runtime.harness import Session, turn_ops
    from agent_runtime.harness.subagents import SubagentManager
    from tests.agent.harness.test_subagent_budget import AcceptAnswer, BlockingAnswerModel

    monkeypatch.setenv("PRAXIS_WORKTREE_ROOT", str(tmp_path / "trees"))
    (repo / "code.py").write_text("a = 10\nb = 2\n")
    git(repo, "add", "code.py")
    (repo / "code.py").write_text("a = 10\nb = 20\n")
    (repo / "untracked.txt").write_text("user work\n")
    index_before = (repo / ".git/index").read_bytes()
    model = BlockingAnswerModel()
    async with await Session.open(
        database=tmp_path / "runtime.db", workspace=repo, model=model, completion_gate=AcceptAnswer()
    ) as session:
        p = turn_ops.start_turn(
            session.store,
            thread_id=session.thread_id,
            user_message="parent",
            binding_manifest={"model_id": "test-model"},
        )
        handle = await session.subagents.spawn(parent_turn_id=p.turn_id, user_message="child", workspace_mode="write")
        child_id = str(handle["child_turn_id"])
        child = session.store.read_turn(child_id)
        environment = child.binding_manifest["child_environment"]
        checkout = Path(session.store.read_thread(child.thread_id).workspace)
        assert checkout != repo
        assert environment["workspace"] == str(checkout)
        assert environment["parent_turn_id"] == p.turn_id
        assert session.store.read_artifact(environment["source_artifact_id"])
        await model.started.wait()
        child_session = session._subagent_sessions[child_id]
        assert child_session.tool_execution_context.workspace_root == checkout
        assert child_session.tool_execution_context.cwd == checkout
        assert not child_session._resource_permissions().path_access(repo / "code.py", write=True)
        assert child_session._resource_permissions().path_access(checkout / "code.py", write=True)
        (checkout / "code.py").write_text("a = 10\nb = 30\n")
        (checkout / "artifact.bin").write_bytes(b"\x00\xffchild")
        model.release.set()
        await session.subagents.wait(parent_turn_id=p.turn_id, child_turn_id=child_id, timeout_seconds=5)
        assert (repo / "code.py").read_text() == "a = 10\nb = 20\n"
        manager = SubagentManager(session)  # No live task table needed.
        with pytest.raises(Exception, match="child|子任务"):
            turn_ops.complete_turn(session.store, turn_id=p.turn_id, answer="premature")
        review = await manager.integrate(parent_turn_id=p.turn_id, child_turn_id=child_id, action="inspect")
        assert "-b = 20" in review["diff"] and "+b = 30" in review["diff"]
        assert "-a = 1\n" not in review["diff"]
        applied = await manager.integrate(
            parent_turn_id=p.turn_id, child_turn_id=child_id, action="apply", artifact_id=review["artifact_id"]
        )
        assert applied["status"] == "integrated"
        assert applied["verification"] == "required"
        assert (repo / "code.py").read_text() == "a = 10\nb = 30\n"
        assert (repo / "artifact.bin").read_bytes() == b"\x00\xffchild"
        assert (repo / "untracked.txt").read_text() == "user work\n"
        assert (repo / ".git/index").read_bytes() == index_before
        assert session.store.verify().valid


@pytest.mark.anyio
async def test_conflict_keeps_both_workspaces_and_rejects_unreviewed_artifact(repo, tmp_path, monkeypatch):
    from agent_runtime.harness import Session, turn_ops
    from tests.agent.harness.test_subagent_budget import AcceptAnswer, SmallAnswerModel

    monkeypatch.setenv("PRAXIS_WORKTREE_ROOT", str(tmp_path / "trees"))
    async with await Session.open(
        database=tmp_path / "runtime.db", workspace=repo, model=SmallAnswerModel(), completion_gate=AcceptAnswer()
    ) as session:
        p = turn_ops.start_turn(session.store, thread_id=session.thread_id, user_message="parent", binding_manifest={})
        prepared = await session._prepare_child(
            parent_turn_id=p.turn_id,
            user_message="child",
            max_steps=None,
            max_tokens_total=None,
            workspace_mode="write",
        )
        child_id = prepared["turn_id"]
        checkout = Path(session.store.read_thread(prepared["thread_id"]).workspace)
        (checkout / "code.py").write_text("child\n")
        await session._drive_child(prepared)
        (repo / "code.py").write_text("concurrent parent\n")
        review = await session.subagents.integrate(parent_turn_id=p.turn_id, child_turn_id=child_id, action="inspect")
        with pytest.raises(ValueError, match="artifact"):
            await session.subagents.integrate(
                parent_turn_id=p.turn_id, child_turn_id=child_id, action="apply", artifact_id="wrong"
            )
        conflict = await session.subagents.integrate(
            parent_turn_id=p.turn_id, child_turn_id=child_id, action="apply", artifact_id=review["artifact_id"]
        )
        assert conflict["status"] == "conflict"
        assert conflict["conflicts"] == ["code.py"]
        assert (repo / "code.py").read_text() == "concurrent parent\n"
        assert (checkout / "code.py").read_text() == "child\n"
        assert checkout.is_dir()


@pytest.mark.anyio
async def test_read_only_child_cannot_mutate_and_needs_no_integration(repo, tmp_path, monkeypatch):
    from agent_runtime.harness import Session, turn_ops
    from tests.agent.harness.test_subagent_budget import AcceptAnswer, SmallAnswerModel

    monkeypatch.setenv("PRAXIS_WORKTREE_ROOT", str(tmp_path / "trees"))
    async with await Session.open(
        database=tmp_path / "runtime.db", workspace=repo, model=SmallAnswerModel(), completion_gate=AcceptAnswer()
    ) as session:
        p = turn_ops.start_turn(session.store, thread_id=session.thread_id, user_message="parent", binding_manifest={})
        handle = await session.subagents.spawn(
            parent_turn_id=p.turn_id, user_message="read", workspace_mode="read-only"
        )
        result = await session.subagents.wait(
            parent_turn_id=p.turn_id, child_turn_id=handle["child_turn_id"], timeout_seconds=5
        )
        assert result["status"] == "done"
        env = session.store.read_turn(handle["child_turn_id"]).binding_manifest["child_environment"]
        assert env["mode"] == "read-only"
        assert session.store.read_thread(handle["thread_id"]).workspace != str(repo)
        turn_ops.complete_turn(session.store, turn_id=p.turn_id, answer="read complete")


@pytest.mark.anyio
async def test_integration_requires_parent_verification_before_completion(repo, tmp_path, monkeypatch):
    from agent_runtime.harness import Session, turn_ops
    from tests.agent.harness.test_subagent_budget import AcceptAnswer, SmallAnswerModel

    monkeypatch.setenv("PRAXIS_WORKTREE_ROOT", str(tmp_path / "trees"))
    async with await Session.open(
        database=tmp_path / "runtime.db", workspace=repo, model=SmallAnswerModel(), completion_gate=AcceptAnswer()
    ) as session:
        p = turn_ops.start_turn(session.store, thread_id=session.thread_id, user_message="parent", binding_manifest={})
        handle = await session.subagents.spawn(parent_turn_id=p.turn_id, user_message="child", workspace_mode="write")
        await session.subagents.wait(parent_turn_id=p.turn_id, child_turn_id=handle["child_turn_id"], timeout_seconds=5)
        review = await session.subagents.integrate(parent_turn_id=p.turn_id, child_turn_id=handle["child_turn_id"])
        await session.subagents.integrate(
            parent_turn_id=p.turn_id,
            child_turn_id=handle["child_turn_id"],
            action="apply",
            artifact_id=review["artifact_id"],
        )
        with pytest.raises(Exception, match="child|子任务"):
            turn_ops.complete_turn(session.store, turn_id=p.turn_id, answer="unverified")
        from tests.agent.test_versioned_delivery import command

        result = await command(session.store, p.turn_id, repo, "versioned-parent-check", "true")
        assert not result.is_error
        verified = await session.subagents.integrate(
            parent_turn_id=p.turn_id, child_turn_id=handle["child_turn_id"], action="verify",
            verification_tool_call_id="versioned-parent-check",
        )
        assert verified["verification"] == "recorded"


def test_child_snapshot_keeps_raw_bytes_despite_checkout_filters(repo, tmp_path):
    (repo / ".gitattributes").write_text("code.py filter=example\n")
    git(repo, "config", "filter.example.smudge", "sed s/a/CHANGED/g")
    git(repo, "config", "filter.example.clean", "cat")
    git(repo, "add", ".gitattributes")
    git(repo, "commit", "-qm", "filter")
    report, _ = WorktreeManager(repo, root=tmp_path / "trees").create_child_snapshot(
        "filtered", identity={"child_turn_id": "c"}, before_create=lambda _: None
    )
    assert (report.worktree.root / "code.py").read_bytes() == (repo / "code.py").read_bytes()


@pytest.mark.parametrize("entrypoint", ["sdk", "cli"])
def test_public_child_patch_integrate_and_verify_in_real_parent_process(repo, tmp_path, monkeypatch, entrypoint):
    import asyncio
    import json

    from typer.testing import CliRunner

    from agent_runtime import Agent, cli
    from agent_runtime.harness import HarnessModelResponse, ModelToolCall, RuntimeStore
    from tests.agent.harness.test_subagent_path import ParentChildModel

    class CodingModel(ParentChildModel):
        async def dispatch(self, prepared, **kwargs):
            request = prepared.dispatch_payload
            results = {
                m.tool_call_id: json.loads(m.content)["structured_content"]
                for m in request.messages
                if m.role == "tool"
            }
            if request.messages[0].content.startswith("child task"):
                if "edit" not in results:
                    call = ModelToolCall(
                        id="edit",
                        name="apply_patch",
                        arguments={"file_path": "code.py", "old_string": "b = 20", "new_string": "b = 30"},
                    )
                else:
                    assert not results["edit"].get("error_code"), results["edit"]
                    return HarnessModelResponse.from_parts(
                        text="child changed code", usage={"input_tokens": 2, "output_tokens": 2}
                    )
            elif "discover-task" not in results:
                call = ModelToolCall(id="discover-task", name="find_tools", arguments={"query": "task", "limit": 1})
            elif "child" not in results:
                call = ModelToolCall(
                    id="child", name="task", arguments={"task": "child task: edit code", "max_turns": 3}
                )
            elif "discover-integrate" not in results:
                assert results["child"]["status"] == "done", results["child"]
                call = ModelToolCall(
                    id="discover-integrate", name="find_tools", arguments={"query": "integrate_agent", "limit": 1}
                )
            elif "review" not in results:
                call = ModelToolCall(
                    id="review",
                    name="integrate_agent",
                    arguments={"child_turn_id": results["child"]["child_turn_id"], "action": "inspect"},
                )
            elif "apply" not in results:
                assert "+b = 30" in results["review"]["diff"], results["review"]
                call = ModelToolCall(
                    id="apply",
                    name="integrate_agent",
                    arguments={
                        "child_turn_id": results["child"]["child_turn_id"],
                        "action": "apply",
                        "artifact_id": results["review"]["artifact_id"],
                    },
                )
            elif "check" not in results:
                call = ModelToolCall(
                    id="check",
                    name="run_command",
                    arguments={
                        "command": (
                            'python3 -c "from pathlib import Path; '
                            "assert 'b = 30' in Path('code.py').read_text(); print(Path.cwd())\""
                        ),
                        "working_dir": ".",
                        "workspace_write": True,
                        "yield_time_seconds": 1,
                    },
                )
            elif results.get("check", {}).get("status") == "running" and not any(
                k.startswith("observe") for k in results
            ):
                call = ModelToolCall(
                    id="observe",
                    name="manage_command",
                    arguments={"session_id": results["check"]["session_id"], "action": "wait", "wait_seconds": 5},
                )
            elif "verified" not in results:
                command_result = results.get("observe", results["check"])
                assert command_result["exit_code"] == 0, command_result
                assert str(repo) in command_result["stdout"]
                call = ModelToolCall(
                    id="verified",
                    name="integrate_agent",
                    arguments={
                        "child_turn_id": results["child"]["child_turn_id"],
                        "action": "verify",
                        "verification_tool_call_id": "observe" if "observe" in results else "check",
                    },
                )
            elif "delivery-plan" not in results:
                assert results["verified"]["verification"] == "recorded", results["verified"]
                call = ModelToolCall(id="delivery-plan", name="record_delivery", arguments={
                    "action": "plan", "scope": ["integrated code uses b = 30; user file preserved"],
                    "commands": [
                        '/usr/bin/python3 -c "from pathlib import Path; '
                        "assert 'b = 30' in Path('code.py').read_text(); "
                        "assert Path('user.txt').read_text() == 'keep user work'\""
                    ],
                })
            elif "delivery-check" not in results:
                call = ModelToolCall(id="delivery-check", name="run_command", arguments={
                    "command": results["delivery-plan"]["commands"][0], "verification_mode": "snapshot",
                    "verification_dependencies": [],
                })
            elif "delivery-record" not in results:
                assert results["delivery-check"]["exit_code"] == 0, results["delivery-check"]
                call = ModelToolCall(id="delivery-record", name="record_delivery", arguments={
                    "action": "record", "tool_call_ids": ["delivery-check"],
                    "review": "Inspected patch and integration; check covers code and preserved user input.",
                })
            else:
                assert results["delivery-record"]["status"] == "verified", results["delivery-record"]
                return HarnessModelResponse.from_parts(
                    text="integrated and checked in parent", usage={"input_tokens": 2, "output_tokens": 2}
                )
            return HarnessModelResponse.from_parts(tool_calls=(call,), usage={"input_tokens": 2, "output_tokens": 2})

    monkeypatch.setenv("PRAXIS_WORKTREE_ROOT", str(tmp_path / "trees"))
    (repo / "code.py").write_text("a = 10\nb = 2\n")
    git(repo, "add", "code.py")
    (repo / "code.py").write_text("a = 10\nb = 20\n")
    (repo / "user.txt").write_text("keep user work")
    original_index = (repo / ".git/index").read_bytes()
    database = tmp_path / "public.db"
    model = CodingModel()
    if entrypoint == "sdk":
        agent = Agent(workspace_path=repo, checkpoint_db=database, enable_workspace_mcp=False, approval_mode="full")
        monkeypatch.setattr(agent, "_model_factory", lambda: model)
        result = asyncio.run(
            agent.run("delegate code change", allow_write_tools=True, allow_execute_tools=True, max_steps=20)
        )
        assert result.status == "done", result
        assert result.delivery["status"] == "verified", result.delivery
    else:
        original = cli._create_agent_facade

        def create(**kwargs):
            agent = original(**kwargs)
            agent._model_factory = lambda: model
            return agent

        monkeypatch.setattr(cli, "_create_agent_facade", create)
        monkeypatch.chdir(repo)
        result = CliRunner().invoke(
            cli.agent_app,
            [
                "run",
                "delegate code change",
                "--approval-mode",
                "full",
                "--allow-write-tools",
                "--allow-execute-tools",
                "--max-steps",
                "20",
                "--non-interactive",
                "--disable-workspace-mcp",
                "--rollout-db",
                str(database),
            ],
        )
        assert result.exit_code == 0, result.output
        assert "交付: verified" in result.output
    assert (repo / "code.py").read_text() == "a = 10\nb = 30\n"
    assert (repo / "user.txt").read_text() == "keep user work"
    assert (repo / ".git/index").read_bytes() == original_index
    with RuntimeStore.open_sync(database) as store:
        assert store.verify().valid
        child = next(t for t in store.list_turns() if "child_environment" in t.binding_manifest)
        parent_turn = store.read_turn(child.binding_manifest["budget_parent_turn_id"])
        assert parent_turn.status == "completed"
        assert child.status == "completed"
        assert Path(store.read_thread(child.thread_id).workspace) != repo
        from agent_runtime.harness import delivery
        saved = delivery.inspect(store, parent_turn.turn_id)
        assert saved["status"] == "verified", saved
        (repo / "code.py").write_text("a = 10\nb = 40\n")
        stale = delivery.inspect(store, parent_turn.turn_id)
        assert stale["status"] != "verified" and stale["freshness"] == "stale"


@pytest.mark.anyio
async def test_host_recovers_reserved_child_in_recorded_isolated_environment(repo, tmp_path, monkeypatch):
    import asyncio

    from agent_runtime import Agent
    from agent_runtime.harness import Session, turn_ops
    from agent_runtime.harness.ownership import OWNERSHIP_REVISION
    from agent_runtime.host import BackgroundHost
    from tests.agent.harness.test_subagent_budget import SmallAnswerModel

    monkeypatch.setenv("PRAXIS_WORKTREE_ROOT", str(tmp_path / "trees"))
    database = tmp_path / "background.db"
    agent = Agent(workspace_path=repo, checkpoint_db=database, model_session_path=None, enable_workspace_mcp=False)
    handle = await agent.start_background("parent", autostart=False)
    async with await Session.open(database=database, workspace=repo, model=SmallAnswerModel()) as session:
        binding = dict(session.binding_provider.snapshot(thread_id=handle.thread_id, turn_id=handle.turn_id))
        binding["execution_ownership"] = OWNERSHIP_REVISION
        turn_ops.start_turn(
            session.store,
            thread_id=handle.thread_id,
            turn_id=handle.turn_id,
            user_message="parent",
            binding_manifest=binding,
        )
        # Use the canonical background Thread, as a driver would after opening it.
        session.thread_id = handle.thread_id
        prepared = await session._prepare_child(
            parent_turn_id=handle.turn_id,
            user_message="child",
            max_steps=3,
            max_tokens_total=None,
            workspace_mode="write",
        )
        turn_ops.pause_turn(
            session.store,
            turn_id=handle.turn_id,
            reason="child pending",
            reason_code="attached_children",
            input_revision=1,
        )
        await session.close(preserve_children=True)
    child_id = prepared["turn_id"]
    async with BackgroundHost(database, model=SmallAnswerModel()) as host:
        serving = asyncio.create_task(host.serve(idle_timeout=None))
        try:
            async with asyncio.timeout(5):
                while (await host.store.run_async(host.store.read_turn, child_id)).status != "completed":
                    await asyncio.sleep(0.02)
        finally:
            serving.cancel()
            await asyncio.gather(serving, return_exceptions=True)
        child = await host.store.run_async(host.store.read_turn, child_id)
        checkout = Path((await host.store.run_async(host.store.read_thread, child.thread_id)).workspace)
        assert checkout != repo
        assert child.binding_manifest["child_environment"]["workspace"] == str(checkout)
        assert checkout.is_dir()
        assert (await handle.status())["children"][0]["environment"]["workspace"] == str(checkout)


@pytest.mark.anyio
async def test_reserved_child_cannot_be_force_removed_before_driver_opens(repo, tmp_path, monkeypatch):
    from agent_runtime.harness import Session, turn_ops
    from tests.agent.harness.test_subagent_budget import SmallAnswerModel

    monkeypatch.setenv("PRAXIS_WORKTREE_ROOT", str(tmp_path / "trees"))
    async with await Session.open(
        database=tmp_path / "runtime.db", workspace=repo, model=SmallAnswerModel()
    ) as session:
        p = turn_ops.start_turn(session.store, thread_id=session.thread_id, user_message="parent", binding_manifest={})
        prepared = await session._prepare_child(
            parent_turn_id=p.turn_id, user_message="child", max_steps=3, max_tokens_total=None, workspace_mode="write"
        )
        checkout = Path(prepared["workspace"])
        with pytest.raises(WorktreeError, match="运行|active|未结束"):
            await __import__("asyncio").to_thread(WorktreeManager(repo).remove, str(checkout), force=True)
        assert checkout.is_dir()


@pytest.mark.anyio
async def test_interrupted_integration_is_durable_and_explicitly_reconciled(repo, tmp_path, monkeypatch):
    from agent_runtime.harness import Session, child_workspaces, turn_ops
    from tests.agent.harness.test_subagent_budget import AcceptAnswer, SmallAnswerModel

    monkeypatch.setenv("PRAXIS_WORKTREE_ROOT", str(tmp_path / "trees"))
    async with await Session.open(
        database=tmp_path / "runtime.db", workspace=repo, model=SmallAnswerModel(), completion_gate=AcceptAnswer()
    ) as session:
        p = turn_ops.start_turn(session.store, thread_id=session.thread_id, user_message="parent", binding_manifest={})
        prepared = await session._prepare_child(
            parent_turn_id=p.turn_id, user_message="child", max_steps=3, max_tokens_total=None, workspace_mode="write"
        )
        child_id = prepared["turn_id"]
        checkout = Path(prepared["workspace"])
        (checkout / "code.py").write_text("changed\n")
        (checkout / "new.txt").write_text("new\n")
        await session._drive_child(prepared)
        review = await session.subagents.integrate(parent_turn_id=p.turn_id, child_turn_id=child_id)
        real = child_workspaces.atomic_bytes
        count = 0

        def fail_second(*args, **kwargs):
            nonlocal count
            count += 1
            if count == 2:
                raise OSError("simulated interruption after first write")
            return real(*args, **kwargs)

        with monkeypatch.context() as patch:
            patch.setattr(child_workspaces, "atomic_bytes", fail_second)
            with pytest.raises(OSError, match="simulated interruption"):
                await session.subagents.integrate(
                    parent_turn_id=p.turn_id, child_turn_id=child_id, action="apply", artifact_id=review["artifact_id"]
                )
        assert (repo / "code.py").read_text() == "changed\n"
        assert not (repo / "new.txt").exists()
        with pytest.raises(RuntimeError, match="unknown"):
            await session.subagents.integrate(
                parent_turn_id=p.turn_id, child_turn_id=child_id, action="apply", artifact_id=review["artifact_id"]
            )
        reconciled = await session.subagents.integrate(
            parent_turn_id=p.turn_id, child_turn_id=child_id, action="reconcile", artifact_id=review["artifact_id"]
        )
        assert reconciled["status"] == "reviewed"
        final = await session.subagents.integrate(
            parent_turn_id=p.turn_id, child_turn_id=child_id, action="apply", artifact_id=review["artifact_id"]
        )
        assert final["status"] == "integrated"
        assert (repo / "new.txt").read_text() == "new\n"


@pytest.mark.anyio
async def test_parent_can_record_manual_conflict_resolution_without_overwriting_user_work(repo, tmp_path, monkeypatch):
    from agent_runtime.harness import Session, turn_ops
    from tests.agent.harness.test_subagent_budget import SmallAnswerModel

    monkeypatch.setenv("PRAXIS_WORKTREE_ROOT", str(tmp_path / "trees"))
    async with await Session.open(
        database=tmp_path / "runtime.db", workspace=repo, model=SmallAnswerModel()
    ) as session:
        p = turn_ops.start_turn(session.store, thread_id=session.thread_id, user_message="parent", binding_manifest={})
        prepared = await session._prepare_child(
            parent_turn_id=p.turn_id, user_message="child", max_steps=3, max_tokens_total=None, workspace_mode="write"
        )
        (Path(prepared["workspace"]) / "code.py").write_text("a = 1\nb = 30\n")
        await session._drive_child(prepared)
        child = prepared["turn_id"]
        review = await session.subagents.integrate(parent_turn_id=p.turn_id, child_turn_id=child)
        (repo / "code.py").write_text("a = 100\nb = 2\n")
        conflict = await session.subagents.integrate(
            parent_turn_id=p.turn_id, child_turn_id=child, action="apply", artifact_id=review["artifact_id"]
        )
        assert conflict["status"] == "conflict"
        (repo / "code.py").write_text("a = 100\nb = 30\n")
        result = await session.subagents.integrate(
            parent_turn_id=p.turn_id,
            child_turn_id=child,
            action="resolve",
            artifact_id=review["artifact_id"],
            resolution_note="Kept parent a and child b after reviewing both diffs",
        )
        assert result["status"] == "integrated"
        assert result["verification"] == "required"
        assert (repo / "code.py").read_text() == "a = 100\nb = 30\n"
        resolution = session.store.read_artifact(result["resolution_artifact_id"]).decode()
        assert "+a = 100" in resolution and "+b = 30" in resolution


@pytest.mark.anyio
async def test_large_child_diff_is_read_in_pages_before_apply(repo, tmp_path, monkeypatch):
    from agent_runtime.harness import Session, turn_ops
    from tests.agent.harness.test_subagent_budget import SmallAnswerModel

    monkeypatch.setenv("PRAXIS_WORKTREE_ROOT", str(tmp_path / "trees"))
    async with await Session.open(
        database=tmp_path / "runtime.db", workspace=repo, model=SmallAnswerModel()
    ) as session:
        p = turn_ops.start_turn(session.store, thread_id=session.thread_id, user_message="parent", binding_manifest={})
        prepared = await session._prepare_child(
            parent_turn_id=p.turn_id, user_message="child", max_steps=3, max_tokens_total=None, workspace_mode="write"
        )
        (Path(prepared["workspace"]) / "code.py").write_text("changed\n" * 50)
        await session._drive_child(prepared)
        child = prepared["turn_id"]
        page = await session.subagents.integrate(parent_turn_id=p.turn_id, child_turn_id=child, diff_limit=50)
        assert page["diff_next_offset"] == 50
        with pytest.raises(ValueError, match="read|review"):
            await session.subagents.integrate(
                parent_turn_id=p.turn_id, child_turn_id=child, action="apply", artifact_id=page["artifact_id"]
            )
        data = page["diff"]
        while page["diff_next_offset"] is not None:
            page = await session.subagents.integrate(
                parent_turn_id=p.turn_id, child_turn_id=child, diff_offset=page["diff_next_offset"], diff_limit=50
            )
            data += page["diff"]
        assert data.encode() == session.store.read_artifact(page["artifact_id"])
        assert (
            await session.subagents.integrate(
                parent_turn_id=p.turn_id, child_turn_id=child, action="apply", artifact_id=page["artifact_id"]
            )
        )["status"] == "integrated"


@pytest.mark.anyio
async def test_live_parent_permission_revocation_reaches_isolated_child(repo, tmp_path, monkeypatch):
    from agent_runtime.harness import Session, turn_ops
    from agent_runtime.permission_policy import PermissionPolicy
    from tests.agent.harness.test_subagent_budget import BlockingAnswerModel

    monkeypatch.setenv("PRAXIS_WORKTREE_ROOT", str(tmp_path / "trees"))
    model = BlockingAnswerModel()
    async with await Session.open(database=tmp_path / "runtime.db", workspace=repo, model=model) as session:
        p = turn_ops.start_turn(session.store, thread_id=session.thread_id, user_message="parent", binding_manifest={})
        handle = await session.subagents.spawn(parent_turn_id=p.turn_id, user_message="child", workspace_mode="write")
        await model.started.wait()
        child = session._subagent_sessions[handle["child_turn_id"]]
        assert child.permissions().path_access(child.workspace_path / "code.py", write=True)
        session.update_permissions(PermissionPolicy(profile="read-only"))
        assert not child.permissions().path_access(child.workspace_path / "code.py", write=True)
        model.release.set()
        await session.subagents.wait(parent_turn_id=p.turn_id, child_turn_id=handle["child_turn_id"], timeout_seconds=5)


@pytest.mark.anyio
async def test_parent_approval_downgrade_reaches_isolated_child(repo, tmp_path, monkeypatch):
    from agent_runtime.harness import Session, turn_ops
    from agent_runtime.tools.permissions import ToolExecutionContext
    from tests.agent.harness.test_subagent_budget import BlockingAnswerModel

    monkeypatch.setenv("PRAXIS_WORKTREE_ROOT", str(tmp_path / "trees"))
    model = BlockingAnswerModel()
    context = ToolExecutionContext(
        workspace_root=repo, cwd=repo, allow_write_tools=True, allow_execute_tools=True, approval_mode="full"
    )
    async with await Session.open(
        database=tmp_path / "runtime.db", workspace=repo, model=model, tool_execution_context=context
    ) as session:
        p = turn_ops.start_turn(session.store, thread_id=session.thread_id, user_message="parent", binding_manifest={})
        handle = await session.subagents.spawn(parent_turn_id=p.turn_id, user_message="child", workspace_mode="write")
        await model.started.wait()
        child = session._subagent_sessions[handle["child_turn_id"]]
        session.update_tool_policy(approval_mode="ask", allow_write_tools=False, allow_execute_tools=False)
        effective = child.tool_execution_context.refreshed()
        assert effective.approval_mode == "ask"
        assert not effective.allow_write_tools
        assert not effective.allow_execute_tools
        model.release.set()
        await session.subagents.wait(parent_turn_id=p.turn_id, child_turn_id=handle["child_turn_id"], timeout_seconds=5)


@pytest.mark.anyio
async def test_integration_never_overwrites_an_ignored_user_file(repo, tmp_path, monkeypatch):
    from agent_runtime.harness import Session, turn_ops
    from tests.agent.harness.test_subagent_budget import SmallAnswerModel

    monkeypatch.setenv("PRAXIS_WORKTREE_ROOT", str(tmp_path / "trees"))
    async with await Session.open(
        database=tmp_path / "runtime.db", workspace=repo, model=SmallAnswerModel()
    ) as session:
        p = turn_ops.start_turn(session.store, thread_id=session.thread_id, user_message="parent", binding_manifest={})
        prepared = await session._prepare_child(
            parent_turn_id=p.turn_id, user_message="child", max_steps=3, max_tokens_total=None, workspace_mode="write"
        )
        (Path(prepared["workspace"]) / "new.txt").write_text("child\n")
        await session._drive_child(prepared)
        review = await session.subagents.integrate(parent_turn_id=p.turn_id, child_turn_id=prepared["turn_id"])
        (repo / ".gitignore").write_text(".praxis/\nnew.txt\n")
        (repo / "new.txt").write_text("USER IRREPLACEABLE\n")
        result = await session.subagents.integrate(
            parent_turn_id=p.turn_id,
            child_turn_id=prepared["turn_id"],
            action="apply",
            artifact_id=review["artifact_id"],
        )
        assert result["status"] == "conflict"
        assert result["conflicts"] == ["new.txt"]
        assert (repo / "new.txt").read_text() == "USER IRREPLACEABLE\n"


@pytest.mark.anyio
async def test_low_level_child_keeps_the_explicit_parent_toolset(repo, tmp_path, monkeypatch):
    from agent_runtime.harness import Session, turn_ops
    from agent_runtime.tools.permissions import ToolExecutionContext
    from tests.agent.harness.test_subagent_budget import BlockingAnswerModel

    monkeypatch.setenv("PRAXIS_WORKTREE_ROOT", str(tmp_path / "trees"))
    model = BlockingAnswerModel()
    async with await Session.open(
        database=tmp_path / "runtime.db",
        workspace=repo,
        model=model,
        tools={},
        tool_execution_context=ToolExecutionContext(
            workspace_root=repo, cwd=repo, approval_mode="full", allow_write_tools=True, allow_execute_tools=True
        ),
    ) as session:
        p = turn_ops.start_turn(session.store, thread_id=session.thread_id, user_message="parent", binding_manifest={})
        handle = await session.subagents.spawn(parent_turn_id=p.turn_id, user_message="child", workspace_mode="write")
        await model.started.wait()
        child = session._subagent_sessions[handle["child_turn_id"]]
        assert set(child.tools) == set(session.tools)
        assert "run_command" not in child.tools
        model.release.set()
        await session.subagents.wait(parent_turn_id=p.turn_id, child_turn_id=handle["child_turn_id"], timeout_seconds=5)


@pytest.mark.anyio
async def test_host_restart_keeps_persistent_ancestor_revocation(repo, tmp_path, monkeypatch):
    import asyncio

    from agent_runtime import Agent
    from agent_runtime.harness import Session, turn_ops
    from agent_runtime.harness.ownership import OWNERSHIP_REVISION
    from agent_runtime.host import BackgroundHost
    from agent_runtime.permission_policy import PermissionPolicy
    from tests.agent.harness.test_subagent_budget import BlockingAnswerModel, SmallAnswerModel

    monkeypatch.setenv("PRAXIS_WORKTREE_ROOT", str(tmp_path / "trees"))
    database = tmp_path / "background.db"
    agent = Agent(workspace_path=repo, checkpoint_db=database, model_session_path=None, enable_workspace_mcp=False)
    handle = await agent.start_background("parent", autostart=False)
    async with await Session.open(database=database, workspace=repo, model=SmallAnswerModel()) as session:
        session.thread_id = handle.thread_id
        binding = dict(session.binding_provider.snapshot(thread_id=handle.thread_id, turn_id=handle.turn_id))
        binding["execution_ownership"] = OWNERSHIP_REVISION
        turn_ops.start_turn(
            session.store,
            thread_id=handle.thread_id,
            turn_id=handle.turn_id,
            user_message="parent",
            binding_manifest=binding,
        )
        prepared = await session._prepare_child(
            parent_turn_id=handle.turn_id,
            user_message="child",
            max_steps=3,
            max_tokens_total=None,
            workspace_mode="write",
        )
        session.update_permissions(PermissionPolicy(profile="read-only"))
        session.update_tool_policy(approval_mode="ask", allow_write_tools=False, allow_execute_tools=False)
        turn_ops.pause_turn(
            session.store,
            turn_id=handle.turn_id,
            reason="child pending",
            reason_code="attached_children",
            input_revision=1,
        )
        await session.close(preserve_children=True)
    model = BlockingAnswerModel()
    async with BackgroundHost(database, model=model) as host:
        serving = asyncio.create_task(host.serve(idle_timeout=None))
        try:
            await asyncio.wait_for(model.started.wait(), 5)
            child = host._sessions[prepared["turn_id"]]
            assert not child.permissions().path_access(child.workspace_path / "code.py", write=True)
            context = child.tool_orchestrator._context_for_turn(prepared["turn_id"]).refreshed()
            assert context.approval_mode == "ask"
            assert not context.allow_write_tools
            assert not context.allow_execute_tools
        finally:
            model.release.set()
            serving.cancel()
            await asyncio.gather(serving, return_exceptions=True)


@pytest.mark.anyio
@pytest.mark.parametrize("change", ["edit", "delete", "symlink"])
async def test_integration_binds_content_and_version_before_concurrent_editor_write(
    repo, tmp_path, monkeypatch, change
):
    from agent_runtime.harness import Session, child_workspaces, turn_ops
    from tests.agent.harness.test_subagent_budget import SmallAnswerModel

    monkeypatch.setenv("PRAXIS_WORKTREE_ROOT", str(tmp_path / "trees"))
    async with await Session.open(
        database=tmp_path / "runtime.db", workspace=repo, model=SmallAnswerModel()
    ) as session:
        p = turn_ops.start_turn(session.store, thread_id=session.thread_id, user_message="parent", binding_manifest={})
        prepared = await session._prepare_child(
            parent_turn_id=p.turn_id, user_message="child", max_steps=3, max_tokens_total=None, workspace_mode="write"
        )
        target = Path(prepared["workspace"]) / "code.py"
        if change == "edit":
            target.write_text("child\n")
        else:
            target.unlink()
            if change == "symlink":
                target.symlink_to(".gitignore")
        await session._drive_child(prepared)
        review = await session.subagents.integrate(parent_turn_id=p.turn_id, child_turn_id=prepared["turn_id"])
        real = child_workspaces.working_entries
        count = 0

        def racing(root, **kwargs):
            nonlocal count
            result = real(root, **kwargs)
            if root == repo and kwargs.get("paths") == ("code.py",):
                count += 1
                if count == 2:
                    (repo / "code.py").write_text("CONCURRENT USER EDIT\n")
            return result

        with monkeypatch.context() as patch:
            patch.setattr(child_workspaces, "working_entries", racing)
            with pytest.raises((PermissionError, RuntimeError), match="changed|version|preimage"):
                await session.subagents.integrate(
                    parent_turn_id=p.turn_id,
                    child_turn_id=prepared["turn_id"],
                    action="apply",
                    artifact_id=review["artifact_id"],
                )
        assert (repo / "code.py").read_text() == "CONCURRENT USER EDIT\n"


@pytest.mark.anyio
@pytest.mark.parametrize("direction", ["directory_to_file", "file_to_directory", "directory_to_symlink"])
async def test_child_file_directory_replacement_integrates_in_safe_order(repo, tmp_path, monkeypatch, direction):
    from agent_runtime.harness import Session, turn_ops
    from tests.agent.harness.test_subagent_budget import SmallAnswerModel

    node = repo / "node"
    if direction in {"directory_to_file", "directory_to_symlink"}:
        node.mkdir()
        (node / "f").write_text("base\n")
    else:
        node.write_text("base\n")
    git(repo, "add", ".")
    git(repo, "commit", "-qm", "structure")
    monkeypatch.setenv("PRAXIS_WORKTREE_ROOT", str(tmp_path / "trees"))
    async with await Session.open(
        database=tmp_path / "runtime.db", workspace=repo, model=SmallAnswerModel()
    ) as session:
        p = turn_ops.start_turn(session.store, thread_id=session.thread_id, user_message="parent", binding_manifest={})
        prepared = await session._prepare_child(
            parent_turn_id=p.turn_id, user_message="child", max_steps=3, max_tokens_total=None, workspace_mode="write"
        )
        child_node = Path(prepared["workspace"]) / "node"
        if direction in {"directory_to_file", "directory_to_symlink"}:
            (child_node / "f").unlink()
            child_node.rmdir()
            child_node.symlink_to("code.py") if direction == "directory_to_symlink" else child_node.write_text(
                "child file\n"
            )
        else:
            child_node.unlink()
            child_node.mkdir()
            (child_node / "f").write_text("child nested\n")
        await session._drive_child(prepared)
        review = await session.subagents.integrate(parent_turn_id=p.turn_id, child_turn_id=prepared["turn_id"])
        result = await session.subagents.integrate(
            parent_turn_id=p.turn_id,
            child_turn_id=prepared["turn_id"],
            action="apply",
            artifact_id=review["artifact_id"],
        )
        assert result["status"] == "integrated"
        if direction == "directory_to_symlink":
            assert node.is_symlink() and node.readlink() == Path("code.py")
        elif direction == "directory_to_file":
            assert node.read_text() == "child file\n"
        else:
            assert (node / "f").read_text() == "child nested\n"


@pytest.mark.anyio
async def test_directory_replacement_preserves_ignored_concurrent_contents(repo, tmp_path, monkeypatch):
    from agent_runtime.harness import Session, turn_ops
    from tests.agent.harness.test_subagent_budget import SmallAnswerModel

    (repo / "node").mkdir()
    (repo / "node/f").write_text("base\n")
    git(repo, "add", ".")
    git(repo, "commit", "-qm", "directory")
    monkeypatch.setenv("PRAXIS_WORKTREE_ROOT", str(tmp_path / "trees"))
    async with await Session.open(
        database=tmp_path / "runtime.db", workspace=repo, model=SmallAnswerModel()
    ) as session:
        p = turn_ops.start_turn(session.store, thread_id=session.thread_id, user_message="parent", binding_manifest={})
        prepared = await session._prepare_child(
            parent_turn_id=p.turn_id, user_message="child", max_steps=3, max_tokens_total=None, workspace_mode="write"
        )
        child = Path(prepared["workspace"])
        (child / "node/f").unlink()
        (child / "node").rmdir()
        (child / "node").write_text("child\n")
        await session._drive_child(prepared)
        review = await session.subagents.integrate(parent_turn_id=p.turn_id, child_turn_id=prepared["turn_id"])
        (repo / ".gitignore").write_text(".praxis/\nnode/local\n")
        (repo / "node/local").write_text("USER WORK\n")
        result = await session.subagents.integrate(
            parent_turn_id=p.turn_id,
            child_turn_id=prepared["turn_id"],
            action="apply",
            artifact_id=review["artifact_id"],
        )
        assert result["status"] == "conflict"
        assert (repo / "node/local").read_text() == "USER WORK\n"
        assert (repo / "node/f").read_text() == "base\n"


@pytest.mark.anyio
async def test_terminal_unknown_child_environment_cannot_be_force_removed(repo, tmp_path, monkeypatch):
    import asyncio

    from agent_runtime.budget import ResourceUsage
    from agent_runtime.harness import Session, model_ops, turn_ops
    from tests.agent.harness.test_subagent_budget import UnknownAnswerModel

    monkeypatch.setenv("PRAXIS_WORKTREE_ROOT", str(tmp_path / "trees"))
    async with await Session.open(
        database=tmp_path / "runtime.db", workspace=repo, model=UnknownAnswerModel()
    ) as session:
        p = turn_ops.start_turn(session.store, thread_id=session.thread_id, user_message="parent", binding_manifest={})
        prepared = await session._prepare_child(
            parent_turn_id=p.turn_id, user_message="child", max_steps=3, max_tokens_total=None, workspace_mode="write"
        )
        op = model_ops.prepare_model_operation(
            session.store,
            turn_id=prepared["turn_id"],
            request_hash="r",
            context_hash="c",
            tool_hash="t",
            wire_hash="w",
            request_ref={"request_id": "unknown", "input_revision": 1},
        )
        attempt = model_ops.dispatch_model_attempt(
            session.store,
            op.operation_id,
            resource_request=ResourceUsage(input_tokens=2, output_tokens=3, model_calls=1),
        )
        turn_ops.fail_turn(session.store, turn_id=prepared["turn_id"], reason_code="probe", message="terminal unknown")
        model_ops.mark_model_attempt_unknown(
            session.store, operation_id=op.operation_id, attempt_id=attempt.attempt_id, generation=attempt.generation
        )
        assert session.store.read_turn(prepared["turn_id"]).status == "failed"
        with pytest.raises(WorktreeError, match="unknown|未知|暴露"):
            await asyncio.to_thread(WorktreeManager(repo).remove, str(prepared["workspace"]), force=True)
        assert Path(prepared["workspace"]).exists()


@pytest.mark.anyio
@pytest.mark.parametrize("resource", ["custom_tool", "skill_runtime"])
async def test_rejected_custom_resource_has_terminal_canonical_result_instead_of_ghost_running(
    repo, tmp_path, monkeypatch, resource
):
    from dataclasses import replace

    from agent_runtime.harness import Session, turn_ops
    from agent_runtime.tools.builtins.filesystem import create_read_file_tool
    from agent_runtime.workspace import open_workspace
    from tests.agent.harness.test_skill_path import FakeSkillRuntime
    from tests.agent.harness.test_subagent_budget import SmallAnswerModel

    monkeypatch.setenv("PRAXIS_WORKTREE_ROOT", str(tmp_path / "trees"))
    custom = replace(create_read_file_tool(open_workspace(repo)), execution_revision="custom-wrapper-v1")
    options = (
        {"tools": {"read_file": custom}} if resource == "custom_tool" else {"skill_runtime": FakeSkillRuntime(repo)}
    )
    async with await Session.open(
        database=tmp_path / "runtime.db", workspace=repo, model=SmallAnswerModel(), **options
    ) as session:
        p = turn_ops.start_turn(session.store, thread_id=session.thread_id, user_message="parent", binding_manifest={})
        handle = await session.subagents.spawn(parent_turn_id=p.turn_id, user_message="child", workspace_mode="write")
        result = await session.subagents.wait(
            parent_turn_id=p.turn_id, child_turn_id=handle["child_turn_id"], timeout_seconds=5
        )
        assert result["status"] == "failed"
        assert result["stop_reason"] == "child_driver_failed"
        assert session.store.list_model_operations(handle["child_turn_id"]) == ()
        assert session.store.read_child_budget_allocation(handle["child_turn_id"])["status"] == "settled"
        assert Path(result["environment"]["workspace"]).is_dir()


@pytest.mark.anyio
async def test_low_level_child_preserves_unbound_planning_adapter(repo, tmp_path, monkeypatch):
    from agent_runtime.harness import Session, turn_ops
    from agent_runtime.tools.builtins.planning import create_update_plan_tool
    from tests.agent.harness.test_subagent_budget import BlockingAnswerModel

    monkeypatch.setenv("PRAXIS_WORKTREE_ROOT", str(tmp_path / "trees"))
    model = BlockingAnswerModel()
    planning = create_update_plan_tool(lambda _: {"accepted": True, "revision": 3, "message": "advisory plan"})
    async with await Session.open(
        database=tmp_path / "runtime.db", workspace=repo, model=model, tools={"update_plan": planning}
    ) as session:
        p = turn_ops.start_turn(session.store, thread_id=session.thread_id, user_message="parent", binding_manifest={})
        handle = await session.subagents.spawn(parent_turn_id=p.turn_id, user_message="child", workspace_mode="write")
        await model.started.wait()
        child = session._subagent_sessions[handle["child_turn_id"]]
        assert child.tools["update_plan"] is planning
        arguments = planning.validate_input({"plan": [{"step": "Inspect child snapshot", "status": "in_progress"}]})
        output = planning.normalize_output(planning.run(arguments))
        assert output.structured_content["revision"] == 3
        model.release.set()
        await session.subagents.wait(parent_turn_id=p.turn_id, child_turn_id=handle["child_turn_id"], timeout_seconds=5)


def test_child_snapshot_does_not_refresh_a_touched_but_unchanged_source_index(repo, tmp_path):
    import os

    original_index = (repo / ".git/index").read_bytes()
    file = repo / "code.py"
    info = file.stat()
    os.utime(file, ns=(info.st_atime_ns, info.st_mtime_ns + 10_000_000_000))
    WorktreeManager(repo, root=tmp_path / "trees").create_child_snapshot(
        "touch", identity={"child_turn_id": "c"}, before_create=lambda _: None
    )
    assert (repo / ".git/index").read_bytes() == original_index


def test_source_staged_provenance_uses_the_frozen_index(repo, tmp_path, monkeypatch):
    from agent_runtime import workspace_snapshot as snapshots

    (repo / "code.py").write_text("a = 10\nb = 2\n")
    git(repo, "add", "code.py")
    (repo / "code.py").write_text("a = 20\nb = 2\n")
    original_index = (repo / ".git/index").read_bytes()
    original = snapshots.git_bytes

    def editing_index(root, *args, **kwargs):
        if args[:2] != ("diff", "--cached"):
            return original(root, *args, **kwargs)
        (repo / "code.py").write_text("a = 999\nb = 2\n")
        git(repo, "add", "code.py")
        try:
            return original(root, *args, **kwargs)
        finally:
            (repo / ".git/index").write_bytes(original_index)
            (repo / "code.py").write_text("a = 20\nb = 2\n")

    monkeypatch.setattr(snapshots, "git_bytes", editing_index)
    snapshot = snapshots.capture_snapshot(repo)
    staged = bytes.fromhex(snapshot["source_index_patch_hex"]).decode()
    assert "+a = 10" in staged
    assert "+a = 999" not in staged
    assert (repo / ".git/index").read_bytes() == original_index


@pytest.mark.anyio
@pytest.mark.parametrize("boundary", ["before_claim", "after_claim", "after_observation", "search_leaf"])
async def test_persistent_ancestor_revocation_covers_dispatch_and_internal_file_access(
    repo, tmp_path, monkeypatch, boundary
):
    import asyncio

    from agent_runtime.harness import RuntimeStore, Session, turn_ops
    from agent_runtime.permission_policy import PathRule, PermissionPolicy
    from agent_runtime.tools import executor as executor_module
    from agent_runtime.tools.builtins.filesystem import create_apply_patch_tool
    from agent_runtime.tools.builtins.search import create_search_text_tool
    from agent_runtime.tools.executor import ExecutionStatus
    from agent_runtime.tools.permissions import ToolExecutionContext
    from agent_runtime.tools.tool import ToolCall, ToolCallOrigin
    from agent_runtime.workspace import open_workspace
    from tests.agent.harness.test_subagent_budget import BlockingAnswerModel

    monkeypatch.setenv("PRAXIS_WORKTREE_ROOT", str(tmp_path / "trees"))
    (repo / "secret.txt").write_text("CONFIDENTIAL SECRET\n")
    workspace = open_workspace(repo)
    tool = create_search_text_tool(workspace) if boundary == "search_leaf" else create_apply_patch_tool(workspace)
    model = BlockingAnswerModel()
    database = tmp_path / "runtime.db"
    context = ToolExecutionContext(
        workspace_root=repo,
        cwd=repo,
        allow_write_tools=True,
        approval_mode="full",
        permissions=PermissionPolicy().resolve(repo),
    )
    async with await Session.open(
        database=database,
        workspace=repo,
        model=model,
        tools={tool.definition.name: tool},
        tool_execution_context=context,
    ) as parent_session:
        p = turn_ops.start_turn(
            parent_session.store, thread_id=parent_session.thread_id, user_message="parent", binding_manifest={}
        )
        handle = await parent_session.subagents.spawn(
            parent_turn_id=p.turn_id, user_message="child", workspace_mode="write"
        )
        await model.started.wait()
        child = parent_session._subagent_sessions[handle["child_turn_id"]]
        revoked = False

        def revoke():
            nonlocal revoked
            if revoked:
                return
            with RuntimeStore.open_sync(database) as other:
                settings = dict(other.read_thread(parent_session.thread_id).settings)
                policy = dict(settings["tool_execution_policy"])
                declaration = (
                    PermissionPolicy(filesystem=(PathRule(path=str(repo / "secret.txt"), access="deny"),))
                    if boundary == "search_leaf"
                    else PermissionPolicy(profile="read-only")
                )
                policy["permissions"] = declaration.resolve(repo).model_dump(mode="json")
                settings["tool_execution_policy"] = policy
                turn_ops.update_session_settings(other, parent_session.thread_id, settings)
            revoked = True

        original_persist = child.tool_orchestrator._persist_execution_record

        async def at_claim(**kwargs):
            starts = kwargs["record"].status == ExecutionStatus.STARTED
            if starts and boundary == "before_claim":
                await asyncio.to_thread(revoke)
            result = await original_persist(**kwargs)
            if starts and boundary == "after_claim":
                await asyncio.to_thread(revoke)
            return result

        original_hash = executor_module.workspace_tree_sha256

        def at_observation(root):
            result = original_hash(root)
            if root == child.workspace_path and boundary == "after_observation":
                revoke()
            return result

        if boundary == "search_leaf":
            await asyncio.to_thread(revoke)
        name = tool.definition.name
        arguments = (
            {"path": ".", "pattern": "CONFIDENTIAL"}
            if boundary == "search_leaf"
            else {"file_path": "code.py", "old_string": "a = 1", "new_string": "a = 999"}
        )
        call = ToolCall(
            tool_call_id="boundary-check",
            tool_name=name,
            arguments=arguments,
            origin=ToolCallOrigin(request_id="direct-aci", toolset_revision="v1", exposed_tool_names=(name,)),
        )
        try:
            with monkeypatch.context() as patch:
                patch.setattr(child.tool_orchestrator, "_persist_execution_record", at_claim)
                patch.setattr(executor_module, "workspace_tree_sha256", at_observation)
                result = await child.tool_orchestrator.execute(turn_id=handle["child_turn_id"], call=call)
            assert revoked
            if boundary == "search_leaf":
                assert not result.is_error
                assert "CONFIDENTIAL SECRET" not in str(result.structured_content)
                assert all(match["file_path"] != "secret.txt" for match in result.structured_content["matches"])
            else:
                assert result.is_error and result.error_code == "permission_changed"
                assert (child.workspace_path / "code.py").read_text() == "a = 1\nb = 2\n"
        finally:
            model.release.set()
            await parent_session.subagents.wait(
                parent_turn_id=p.turn_id, child_turn_id=handle["child_turn_id"], timeout_seconds=5
            )


@pytest.mark.anyio
async def test_host_reloads_permission_file_in_original_source_domain(repo, tmp_path, monkeypatch):
    import asyncio
    import json

    from agent_runtime import Agent
    from agent_runtime.harness import Session, host_ops, turn_ops
    from agent_runtime.harness.ownership import OWNERSHIP_REVISION
    from agent_runtime.host import BackgroundHost
    from tests.agent.harness.test_subagent_budget import BlockingAnswerModel, SmallAnswerModel

    monkeypatch.setenv("PRAXIS_WORKTREE_ROOT", str(tmp_path / "trees"))
    policy = tmp_path / "permissions.json"
    policy.write_text(json.dumps({"profile": "workspace"}))
    (repo / "secret.txt").write_text("CONFIDENTIAL\n")
    database = tmp_path / "background.db"
    agent = Agent(
        workspace_path=repo,
        checkpoint_db=database,
        model_session_path=None,
        permissions=policy,
        enable_workspace_mcp=False,
    )
    handle = await agent.start_background("parent", autostart=False)
    agent._model_factory = SmallAnswerModel
    async with await Session.open(agent=agent, thread_id=handle.thread_id) as session:
        binding = dict(session.binding_provider.snapshot(thread_id=handle.thread_id, turn_id=handle.turn_id))
        binding["execution_ownership"] = OWNERSHIP_REVISION
        turn_ops.start_turn(
            session.store,
            thread_id=handle.thread_id,
            turn_id=handle.turn_id,
            user_message="parent",
            binding_manifest=binding,
        )
        prepared = await session._prepare_child(
            parent_turn_id=handle.turn_id,
            user_message="child",
            max_steps=3,
            max_tokens_total=None,
            workspace_mode="write",
        )
        turn_ops.pause_turn(
            session.store,
            turn_id=handle.turn_id,
            reason="child pending",
            reason_code="attached_children",
            input_revision=1,
        )
        previous = host_ops.claim_next(session.store, owner="previous-host")
        assert previous.turn_id == handle.turn_id
        host_ops.settle_command(session.store, previous, status="applied", result={"turn_id": handle.turn_id})
        await session.close(preserve_children=True)
    policy.write_text(
        json.dumps({"profile": "workspace", "filesystem": [{"path": str(repo / "secret.txt"), "access": "deny"}]})
    )
    model = BlockingAnswerModel()
    async with BackgroundHost(database, model=model) as host:
        serving = asyncio.create_task(host.serve(idle_timeout=None))
        try:
            await asyncio.wait_for(model.started.wait(), 5)
            child = host._sessions[prepared["turn_id"]]
            assert not child.permissions().path_access(child.workspace_path / "secret.txt")
        finally:
            model.release.set()
            serving.cancel()
            await asyncio.gather(serving, return_exceptions=True)
