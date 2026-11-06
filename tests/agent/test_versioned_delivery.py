import pytest

from agent_runtime.harness import RuntimeStore, delivery, turn_ops


@pytest.fixture
def context(tmp_path, monkeypatch):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "source.txt").write_text("user version\n")
    monkeypatch.setenv("PRAXIS_WORKTREE_ROOT", str(tmp_path / "trees"))
    with RuntimeStore.open_sync(tmp_path / "runtime.db") as store:
        thread = turn_ops.create_thread(store, workspace=workspace)
        turn = turn_ops.start_turn(
            store,
            thread_id=thread.thread_id,
            user_message="explain source",
            binding_manifest={"delivery_policy": {"revision": "delivery-v1"}},
        )
        delivery.begin(store, turn.turn_id)
        yield store, turn.turn_id, workspace


def test_missing_checks_are_unverified_and_survive_reopen(context):
    store, turn, workspace = context
    saved = delivery.record(store, turn)
    assert saved["status"] == "unverified"
    assert saved["next_actions"]
    assert delivery.record(store, turn)["delivery_id"] == saved["delivery_id"]
    assert delivery.inspect(store, turn)["delivery_id"] == saved["delivery_id"]
    assert store.verify().valid


def test_final_diff_distinguishes_user_baseline_and_untracked_changes(context):
    store, turn, workspace = context
    (workspace / "source.txt").write_text("task version\n")
    (workspace / "new.bin").write_bytes(b"\x00\xff")
    result = delivery.record(store, turn)
    assert result["changed_paths"] == ["new.bin", "source.txt"]
    assert result["baseline"]["entries"]["source.txt"] != result["final"]["entries"]["source.txt"]
    diff = store.artifact_blob_path(result["diff_artifact_id"]).read_bytes()
    assert b"new.bin" in diff and b"task version" in diff
    (workspace / "new.bin").write_bytes(b"changed")
    assert delivery.inspect(store, turn)["freshness"] == "stale"


def test_blocked_is_separate_from_turn_lifecycle(context):
    store, turn, workspace = context
    delivery.plan(store, turn, scope=["source explanation"], commands=[], omissions=["No process needed"])
    result = delivery.record(store, turn, blockers=["source unavailable"])
    assert result["status"] == "blocked"
    assert store.read_turn(turn).status == "running"


def test_unbacked_explanation_cannot_be_verified(context):
    store, turn, workspace = context
    delivery.plan(store, turn, scope=["source explanation"], commands=[], omissions=["No process needed"])
    with pytest.raises(ValueError, match="source"):
        delivery.record(store, turn, source_item_ids=["fabricated"])
    assert delivery.record(store, turn)["status"] == "unverified"


@pytest.mark.anyio
async def test_frozen_command_binds_untracked_inputs_and_keeps_parent_unchanged(tmp_path, monkeypatch):
    from agent_runtime.tools.builtins.shell import create_run_command_tool
    from agent_runtime.workspace import open_workspace

    root = tmp_path / "source"
    root.mkdir()
    (root / "new.txt").write_text("first")
    monkeypatch.setenv("PRAXIS_WORKTREE_ROOT", str(tmp_path / "trees"))
    workspace = open_workspace(root)
    tool = create_run_command_tool(workspace)
    output = await tool.run(
        {
            "command": "python3 -c \"from pathlib import Path; assert Path('new.txt').read_text() == 'first'\"",
            "verification_mode": "snapshot",
        }
    )
    assert output.exit_code == 0, output.stderr
    receipt = output.verification_receipt
    assert receipt["input_tree"] == delivery.snapshot(root)["tree"]
    assert receipt["inputs_unchanged"] is True
    assert receipt["cwd"] != str(root)
    assert receipt["started_at"] <= receipt["finished_at"]
    assert receipt["environment"]
    assert (root / "new.txt").read_text() == "first"


def explanation_record(store, turn):
    from tests.agent.harness.test_completion_gate import _commit_tool_result

    _commit_tool_result(
        store,
        turn_id=turn,
        call_id="source",
        tool_name="read_file",
        arguments={"path": "source.txt"},
        result={"text": "user version"},
    )
    source = store.list_tool_operations(turn)[-1].result_item_id
    delivery.plan(store, turn, scope=["explain the source"], commands=[], omissions=["No code changes"])
    return delivery.record(
        store,
        turn,
        source_item_ids=[source],
        review="Answer explains the source",
        answer="The source has the user version.",
    )


def test_new_plan_invalidates_old_scope(context):
    store, turn, workspace = context
    assert explanation_record(store, turn)["status"] == "verified"
    delivery.plan(store, turn, scope=["new coverage"], commands=["new check"])
    assert delivery.inspect(store, turn)["status"] == "unverified"


def test_missing_evidence_blob_blocks_delivery(context):
    store, turn, workspace = context
    saved = explanation_record(store, turn)
    assert saved["status"] == "verified"
    store.artifact_blob_path(saved["diff_artifact_id"]).unlink()
    assert delivery.inspect(store, turn)["status"] == "blocked"


def test_snapshot_write_cannot_masquerade_as_frozen_verification():
    from agent_runtime.tools.builtins.shell import RunCommandInput

    with pytest.raises(ValueError, match="read-only"):
        RunCommandInput(command="write then restore", verification_mode="snapshot", workspace_write=True)


async def command(store, turn, root, identity, text, *, dependencies=(), mode="snapshot"):
    from agent_runtime.harness.tool_orchestrator import ToolOrchestrator
    from agent_runtime.tools.builtins.shell import create_run_command_tool
    from agent_runtime.tools.permissions import ToolExecutionContext
    from agent_runtime.tools.tool import ToolCall, ToolCallOrigin
    from agent_runtime.workspace import open_workspace
    from agent_runtime.workspace_snapshot import runtime_excluded_paths

    tool = create_run_command_tool(open_workspace(root))
    orchestrator = ToolOrchestrator(
        store=store,
        tools={"run_command": tool},
        execution_context=ToolExecutionContext(
            workspace_root=root,
            cwd=root,
            allow_execute_tools=True,
            approval_mode="full",
            snapshot_excluded_paths=runtime_excluded_paths(store.persistence._path),
        ),
    )
    return await orchestrator.execute(
        turn_id=turn,
        call=ToolCall(
            tool_call_id=identity,
            tool_name="run_command",
            arguments={
                "command": text,
                "verification_mode": mode,
                "verification_dependencies": None if dependencies is None else list(dependencies),
            },
            origin=ToolCallOrigin(request_id=identity, toolset_revision="test", exposed_tool_names=("run_command",)),
        ),
    )


@pytest.mark.anyio
async def test_verified_stale_reverified_and_restart(context):
    store, turn, root = context
    check = "/usr/bin/grep version source.txt"
    delivery.plan(store, turn, scope=["source version marker"], commands=[check])
    assert not (await command(store, turn, root, "first", check)).is_error
    first = delivery.record(store, turn, tool_call_ids=["first"], review="Check covers the source version marker")
    assert first["status"] == "verified", first["next_actions"]
    (root / "source.txt").write_text("next version\n")
    assert delivery.inspect(store, turn)["status"] == "unverified"
    assert not (await command(store, turn, root, "second", check)).is_error
    second = delivery.record(store, turn, tool_call_ids=["second"], review="Checked the updated marker")
    assert second["status"] == "verified", second["next_actions"]
    assert second["delivery_id"] != first["delivery_id"]
    # Rebuild the canonical projections, preserving the same delivery identity.
    assert store.verify().valid
    database = store.persistence._path
    async with await RuntimeStore.open(database, read_only=True) as reader:
        restored = delivery.inspect(reader, turn)
        assert restored["status"] == "verified"
        assert restored["delivery_id"] == second["delivery_id"]


@pytest.mark.anyio
async def test_latest_failure_cannot_be_hidden_by_selecting_old_success(context):
    store, turn, root = context
    check = "/usr/bin/grep pass source.txt"
    (root / "source.txt").write_text("pass")
    delivery.plan(store, turn, scope=["contains pass"], commands=[check])
    assert not (await command(store, turn, root, "passing", check)).is_error
    assert delivery.record(store, turn, tool_call_ids=["passing"], review="Checked marker")["status"] == "verified"
    (root / "source.txt").write_text("fail")
    assert (await command(store, turn, root, "failing", check)).is_error
    # Even restoring the successful bytes must retain the later failure.
    (root / "source.txt").write_text("pass")
    assert delivery.record(store, turn, tool_call_ids=["passing"], review="Checked marker")["status"] == "blocked"


@pytest.mark.anyio
async def test_missing_dependency_declaration_is_unverified(context):
    store, turn, root = context
    delivery.plan(store, turn, scope=["command exit"], commands=["true"])
    assert not (await command(store, turn, root, "check", "true", dependencies=None)).is_error
    result = delivery.record(store, turn, tool_call_ids=["check"], review="Checked declared scope")
    assert result["status"] == "unverified"
    assert any("dependency" in reason for reason in result["next_actions"])


@pytest.mark.anyio
async def test_dependency_change_invalidates_the_proof(context, tmp_path):
    store, turn, root = context
    dependency = tmp_path / "dependency.txt"
    dependency.write_text("one")
    delivery.plan(store, turn, scope=["command exit with declared dependency"], commands=["true"])
    assert not (await command(store, turn, root, "check", "true", dependencies=[str(dependency)])).is_error
    saved = delivery.record(store, turn, tool_call_ids=["check"], review="Checked declared scope")
    assert saved["status"] == "verified"
    dependency.write_text("two")
    assert delivery.inspect(store, turn)["status"] == "unverified"


@pytest.mark.anyio
async def test_custom_database_in_workspace_is_excluded(context):
    # The source fixture uses an external DB; this independently exercises a
    # custom database whose basename has no special runtime naming convention.
    old, _, root = context
    async with await RuntimeStore.open(root / "my-runtime.sqlite") as store:
        thread = turn_ops.create_thread(store, workspace=root)
        turn = turn_ops.start_turn(store, thread_id=thread.thread_id, user_message="check", binding_manifest={})
        delivery.begin(store, turn.turn_id)
        delivery.plan(store, turn.turn_id, scope=["source marker"], commands=["true"])
        result = await command(store, turn.turn_id, root, "check", "true")
        assert not result.is_error
        receipt = result.structured_content["verification_receipt"]
        assert not any(p.startswith("my-runtime.sqlite") for p in receipt["input_entries"])
        saved = delivery.record(store, turn.turn_id, tool_call_ids=["check"], review="Checked declared scope")
        assert saved["status"] == "verified", saved["next_actions"]


def test_new_input_invalidates_requirement_version(context):
    store, turn, root = context
    assert explanation_record(store, turn)["status"] == "verified"
    turn_ops.receive_turn_input(
        store, thread_id=store.read_turn(turn).thread_id, turn_id=turn, text="also cover another requirement"
    )
    assert delivery.inspect(store, turn)["status"] == "unverified"


def test_different_final_answer_invalidates_explanation_review(context):
    from agent_runtime.harness import completion_ops

    store, turn, root = context
    assert explanation_record(store, turn)["status"] == "verified"
    completion_ops.record_final_proposal(store, turn_id=turn, answer="A different answer without source coverage")
    assert delivery.inspect(store, turn)["status"] == "unverified"


@pytest.mark.anyio
@pytest.mark.parametrize("mode", ["snapshot", "workspace"])
async def test_concurrent_parent_edit_and_restore_has_a_testable_boundary(context, monkeypatch, mode):
    import asyncio

    from agent_runtime.tools.builtins import shell

    store, turn, root = context
    entered, release = asyncio.Event(), asyncio.Event()
    actual = shell._run_command

    async def barrier(*args, **kwargs):
        entered.set()
        await release.wait()
        return await actual(*args, **kwargs)

    monkeypatch.setattr(shell, "_run_command", barrier)
    check = "/usr/bin/grep version source.txt"
    delivery.plan(store, turn, scope=["source marker"], commands=[check])
    pending = asyncio.create_task(command(store, turn, root, "check", check, mode=mode))
    await entered.wait()
    original = (root / "source.txt").read_bytes()
    (root / "source.txt").write_text("concurrent edit")
    (root / "source.txt").write_bytes(original)
    release.set()
    output = await pending
    assert not output.is_error
    saved = delivery.record(store, turn, tool_call_ids=["check"], review="Scope covers source marker")
    assert saved["status"] == ("verified" if mode == "snapshot" else "unverified"), saved["next_actions"]


def test_cli_reloads_current_delivery_without_model(context):
    import json

    from typer.testing import CliRunner

    from agent_runtime.cli import agent_app

    store, turn, root = context
    assert explanation_record(store, turn)["status"] == "verified"
    turn_ops.complete_turn(store, turn_id=turn, answer="The source has the user version.")
    value = CliRunner().invoke(
        agent_app, ["delivery", turn, "--workspace", str(root), "--rollout-db", str(store.persistence._path)]
    )
    assert value.exit_code == 0, value.output
    assert json.loads(value.output)["status"] == "verified"
    (root / "source.txt").write_text("changed after completion")
    value = CliRunner().invoke(
        agent_app, ["delivery", turn, "--workspace", str(root), "--rollout-db", str(store.persistence._path)]
    )
    assert value.exit_code == 0, value.output
    assert json.loads(value.output)["status"] == "unverified"


@pytest.mark.anyio
async def test_delivery_snapshot_honors_persistent_ancestor_read_revocation(tmp_path, monkeypatch):
    from agent_runtime.harness import Session
    from agent_runtime.permission_policy import PathRule, PermissionPolicy
    from tests.agent.harness.test_subagent_budget import SmallAnswerModel

    root = tmp_path / "source"
    root.mkdir()
    (root / "secret.txt").write_text("private input")
    monkeypatch.setenv("PRAXIS_WORKTREE_ROOT", str(tmp_path / "trees"))
    async with await Session.open(
        database=tmp_path / "runtime.db", workspace=root, model=SmallAnswerModel()
    ) as session:
        parent = turn_ops.start_turn(
            session.store, thread_id=session.thread_id, user_message="parent", binding_manifest={}
        )
        child = await session._prepare_child(
            parent_turn_id=parent.turn_id,
            user_message="child",
            max_steps=2,
            max_tokens_total=None,
            workspace_mode="read-only",
        )
        session.update_permissions(
            PermissionPolicy(filesystem=(PathRule(path=str(root / "secret.txt"), access="deny"),))
        )
        with pytest.raises(PermissionError, match="denied"):
            await session.store.run_async(delivery._capture, session.store, child["turn_id"])


def test_delivery_diff_rebuilds_from_canonical_archive_after_git_gc(context):
    from agent_runtime.workspace_snapshot import git_bytes

    store, turn, root = context
    baseline = delivery.begin(store, turn)
    repository = root
    if not (root / ".git").exists():
        baseline = next(i.payload for i in store.list_items(turn) if i.kind == "delivery_baseline")
        repository = __import__("pathlib").Path(baseline["repository"])
    git_bytes(repository, "prune", "--expire=now")
    (root / "source.txt").write_text("new task version")
    saved = delivery.record(store, turn)
    assert saved["final"] is not None
    assert saved["changed_paths"] == ["source.txt"]
    assert b"new task version" in store.artifact_blob_path(saved["diff_artifact_id"]).read_bytes()


def test_dependency_manifest_checks_each_descendant(tmp_path):
    from agent_runtime.workspace_snapshot import dependency_manifest

    root = tmp_path / "dependencies"
    root.mkdir()
    denied = root / "private.txt"
    denied.write_text("private")

    def authorize(path):
        if path == denied:
            raise PermissionError("private descendant denied")

    with pytest.raises(PermissionError, match="private descendant"):
        dependency_manifest((root,), authorize=authorize)


def test_unknown_operation_blocks_even_without_delivery_declaration(context):
    from agent_runtime.harness import tool_ops

    store, turn, root = context
    tool_ops.record_tool_call(
        store,
        turn_id=turn,
        tool_call_id="unknown",
        tool_name="run_command",
        arguments={"command": "external operation"},
        origin={"request_id": "r", "toolset_revision": "t", "exposed_tool_names": ["run_command"]},
    )
    values = dict(
        turn_id=turn,
        operation_id="uncertain-op",
        tool_call_id="unknown",
        tool_name="run_command",
        arguments_digest="digest",
        execution_revision="test",
        idempotent=False,
        attempt_count=0,
        error_code=None,
        requires_reconciliation=False,
        effects=("execute_process",),
        resources=(),
    )
    tool_ops.record_tool_execution_state(store, status="prepared", **values)
    tool_ops.record_tool_execution_state(store, status="ready", **values)
    claim = tool_ops.claim_tool_operation(store, operation_id="uncertain-op", worker_id="owner", lease_seconds=30)
    assert tool_ops.commit_tool_operation_outcome(
        store,
        operation_id="uncertain-op",
        claim_generation=claim.claim_generation,
        fencing_token=claim.fencing_token,
        status="unknown",
        attempt_count=1,
        error_code="owner_lost",
        requires_reconciliation=True,
    )
    assert delivery.inspect(store, turn)["status"] == "blocked"
    with pytest.raises(RuntimeError, match="not running"):
        delivery.record(store, turn)
    assert delivery.inspect(store, turn)["unknown_operations"] == ["uncertain-op"]


@pytest.mark.anyio
async def test_explicit_acceptance_delivery_requires_current_workspace_receipt(context):
    store, old, root = context
    thread = turn_ops.create_thread(store, workspace=root)
    turn = turn_ops.start_turn(
        store,
        thread_id=thread.thread_id,
        user_message="exact acceptance",
        binding_manifest={"completion_policy": {"acceptance_command": "true"}},
    ).turn_id
    delivery.begin(store, turn)
    delivery.plan(store, turn, scope=["configured acceptance exits successfully"], commands=["true"])
    assert not (await command(store, turn, root, "frozen", "true", mode="snapshot")).is_error
    assert (
        delivery.record(store, turn, tool_call_ids=["frozen"], review="Reviewed exact command")["status"]
        == "unverified"
    )
    assert not (await command(store, turn, root, "current", "true", mode="workspace")).is_error
    saved = delivery.record(store, turn, tool_call_ids=["current"], review="Reviewed current-workspace command")
    assert saved["status"] == "verified", saved["next_actions"]


@pytest.mark.parametrize("source_tool", ["read_file", "web_fetch", "web_search", "inspect_data_file"])
def test_explanation_sources_follow_trusted_read_effects_instead_of_tool_name(context, source_tool):
    from tests.agent.harness.test_completion_gate import _commit_tool_result

    store, turn, root = context
    _commit_tool_result(
        store,
        turn_id=turn,
        call_id="source-observation",
        tool_name=source_tool,
        arguments={},
        result={"content": "observed source"},
    )
    source = store.list_tool_operations(turn)[-1].result_item_id
    delivery.plan(store, turn, scope=["explain observed source"], commands=[], omissions=["No behavior change"])
    saved = delivery.record(
        store,
        turn,
        source_item_ids=[source],
        review="Answer covers the retained observation",
        answer="The observed source is recorded.",
    )
    assert saved["status"] == "verified", saved["next_actions"]


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("action", "acceptance", "stop_mode"),
    [
        ("record", None, None),
        ("submit", None, None),
        ("submit", "false", "allow"),
        ("submit", None, "block"),
        ("submit", None, "mutate"),
    ],
)
async def test_public_delivery_can_publish_exact_reviewed_answer(tmp_path, monkeypatch, action, acceptance, stop_mode):
    import hashlib

    from agent_runtime import Agent, HookDecision, LifecycleHook
    from agent_runtime.harness import HarnessModelResponse, ModelToolCall
    from tests.agent.harness.test_public_agent_cutover import PatchThenAnswerModel

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "source.txt").write_text("original evidence\n")
    database = tmp_path / "runtime.db"
    answer = "The file contains original evidence.\n\nNo changes were needed."
    different = "A regenerated final with different wording."
    monkeypatch.setenv("PRAXIS_WORKTREE_ROOT", str(tmp_path / "trees"))

    class ReviewModel(PatchThenAnswerModel):
        async def dispatch(self, prepared, *, event_sink=None):
            step = prepared.request_ref["request_id"].rsplit(":", 1)[-1]
            calls = {
                "1": ("record_delivery", {
                    "action": "plan", "scope": ["Explain source"], "omissions": ["No files changed"],
                }),
                "2": ("read_file", {"path": "source.txt"}),
                "3": ("record_delivery", {
                    "action": action, "source_item_ids": ["read"], "review": "Checked the source against the answer",
                    "answer": answer,
                }),
            }
            if step not in calls:
                return HarnessModelResponse.from_parts(text=different)
            name, arguments = calls[step]
            call_id = "read" if step == "2" else f"delivery-{step}"
            return HarnessModelResponse.from_parts(tool_calls=(ModelToolCall(call_id, name, arguments),))

    proposals = []
    reviews = []

    async def stop(request):
        proposals.append(request.answer)
        reviews.append(request)
        if request.answer != answer:
            return None
        if stop_mode == "mutate":
            (workspace / "source.txt").write_text("changed during Stop")
        if stop_mode == "block":
            return HookDecision(block=True, feedback="Revise the reviewed wording.")
        return None

    agent = Agent(
        workspace_path=workspace,
        checkpoint_db=database,
        enable_workspace_mcp=False,
        hooks=() if stop_mode is None else (LifecycleHook("Stop", "review", stop),),
    )
    monkeypatch.setattr(agent, "_model_factory", lambda: ReviewModel())
    result = await agent.run("Explain what source.txt contains.", acceptance_command=acceptance, max_steps=4)
    if stop_mode is not None:
        reviewed = reviews[0]
        assert reviewed.completion_revision
        assert reviewed.completion["delivery"]["delivery_id"] == result.delivery["delivery_id"]
        assert reviewed.completion["delivery"]["freshness"] == "current"
        assert reviewed.completion["delivery"]["source_item_ids"]
        if acceptance:
            assert reviewed.completion["decision"]["reason_code"] == "acceptance_command_missing"
        else:
            assert reviewed.completion["decision"]["action"] == "accept"
            assert reviewed.completion["delivery"]["status"] == "verified"
    if stop_mode in {"block", "mutate"}:
        assert proposals == [answer, different]
        assert result.status == "done"
        assert result.answer == different
        assert result.delivery["status"] == "unverified"
        async with await RuntimeStore.open(database, read_only=True) as store:
            items = await store.run_async(store.list_items, result.turn_id)
        decision = next(i for i in items if i.kind == "completion_decision")
        assert decision.payload["reason_code"] == (
            "stop_hook_blocked" if stop_mode == "block" else "delivery_evidence_stale"
        )
        if stop_mode == "mutate":
            from agent_runtime.workspace_snapshot import git_bytes
            from agent_runtime.worktrees import WorktreeManager

            repository = WorktreeManager(workspace)._child_repository()
            oid = git_bytes(repository, "hash-object", "--stdin", input=b"changed during Stop").decode().strip()
            with pytest.raises(RuntimeError, match="cat-file"):
                git_bytes(repository, "cat-file", "-e", oid)
        return
    if acceptance:
        assert proposals == [answer, different]
        assert result.status == "failed"
        assert result.answer != answer
        assert result.delivery["status"] == "unverified"
        async with await RuntimeStore.open(database, read_only=True) as store:
            items = await store.run_async(store.list_items, result.turn_id)
        proposal = next(i for i in items if i.kind == "final_proposal")
        decision = next(i for i in items if i.kind == "completion_decision")
        assert proposal.payload["text"] == answer
        assert decision.payload["proposal_item_id"] == proposal.item_id
        assert decision.payload["reason_code"] == "acceptance_command_missing"
        return
    assert result.status == "done"
    assert result.answer == (answer if action == "submit" else different)
    assert result.delivery["status"] == ("verified" if action == "submit" else "unverified")
    assert result.delivery["answer_sha256"] == hashlib.sha256(answer.encode()).hexdigest()
    restored = await agent.read_result(result.turn_id)
    assert restored.answer == result.answer
    assert dict(restored.delivery) == dict(result.delivery)
    if action == "submit":
        (workspace / "source.txt").write_text("new evidence\n")
        stale = await agent.read_result(result.turn_id)
        assert stale.answer == answer
        assert stale.delivery["status"] == "unverified"
        assert stale.delivery["freshness"] == "stale"


@pytest.mark.anyio
async def test_stop_discards_a_verdict_when_a_declared_dependency_changes(tmp_path, monkeypatch):
    from agent_runtime import Agent, LifecycleHook, PathRule, PermissionPolicy
    from agent_runtime.harness import HarnessModelResponse, ModelToolCall
    from tests.agent.harness.test_lifecycle_hooks import ScriptModel

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "source.txt").write_text("source")
    dependency = tmp_path / "dependency.txt"
    dependency.write_text("original")
    db = tmp_path / "run.db"
    command_text = "/usr/bin/true"
    model = ScriptModel([
        HarnessModelResponse.from_parts(tool_calls=(ModelToolCall("plan", "record_delivery", {
            "action": "plan", "scope": ["source"], "commands": [command_text],
        }),)),
        HarnessModelResponse.from_parts(tool_calls=(ModelToolCall("check", "run_command", {
            "command": command_text, "verification_mode": "workspace",
            "verification_dependencies": [str(dependency)],
        }),)),
        HarnessModelResponse.from_parts(tool_calls=(ModelToolCall("submit", "record_delivery", {
            "action": "submit", "tool_call_ids": ["check"], "review": "Checked declared scope", "answer": "first",
        }),)),
        HarnessModelResponse.from_parts(text="revised"),
    ])
    reviews = []

    async def stop(request):
        reviews.append(request)
        if request.answer == "first":
            dependency.write_text("changed")

    agent = Agent(
        workspace_path=workspace, checkpoint_db=db, enable_workspace_mcp=False,
        permissions=PermissionPolicy(filesystem=(PathRule(path=str(dependency), access="read"),)),
        hooks=(LifecycleHook("Stop", "review", stop),),
    )
    monkeypatch.setattr(agent, "_model_factory", lambda: model)
    result = await agent.run("Verify", acceptance_command=command_text, allow_execute_tools=True, max_steps=4)

    assert result.answer == "revised"
    before, after = reviews
    assert before.completion["acceptance"]["command_status"] == "passed"
    [receipt] = before.completion["delivery"]["receipt_evidence"]
    assert receipt["status"] == "current"
    assert receipt["source_result_item_id"] == before.completion["acceptance"]["result_item_id"]
    assert receipt["verification_item_id"]
    assert after.completion["delivery"]["freshness"] == "current"
    assert after.completion["delivery"]["receipt_evidence"][0]["status"] == "unverified"
    assert result.delivery["status"] == "unverified"
    async with await RuntimeStore.open(db, read_only=True) as store:
        items = await store.run_async(store.list_items, result.turn_id)
        rejected, accepted = [i for i in items if i.kind == "completion_decision"]
        assert rejected.payload["reason_code"] == "stop_evidence_changed"
        assert accepted.payload["hook_evidence_revision"] == after.completion_revision
        assert (await store.run_async(store.verify)).valid
