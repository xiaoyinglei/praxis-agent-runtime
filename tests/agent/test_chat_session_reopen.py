"""Durable chat selection/replay is independent from same-Turn task recovery."""

from contextlib import asynccontextmanager
from types import SimpleNamespace

import pytest

from agent_runtime import cli
from agent_runtime.harness import RolloutStore
from agent_runtime.history import list_chat_sessions, read_chat_messages
from tests.agent.test_agent_cli_resume import _result


def conversation(database, workspace, *, question="first question", answer="saved answer"):
    workspace.mkdir(exist_ok=True)
    with RolloutStore(database) as store:
        thread = store.create_thread(workspace=workspace)
        turn = store.start_turn(thread_id=thread.thread_id, user_message=question,
                                binding_manifest={"model_id": "saved-model"})
        store.complete_turn(turn_id=turn.turn_id, answer=answer)
    return thread.thread_id, turn.turn_id


def records(database):
    with RolloutStore(database, read_only=True) as store:
        return store.list_global_records()


def test_chat_list_groups_turns_and_isolates_workspace_without_mutation(tmp_path):
    database = tmp_path / "state.sqlite"
    workspace = tmp_path / "workspace"
    thread_id, _ = conversation(database, workspace)
    conversation(database, tmp_path / "other", question="private other workspace")
    with RolloutStore(database) as store:
        second = store.start_turn(thread_id=thread_id, user_message="second question", binding_manifest={})
        store.complete_turn(turn_id=second.turn_id, answer="second answer")
        child = store.create_thread(workspace=workspace)
        child_turn = store.start_turn(thread_id=child.thread_id, user_message="child agent task",
                                      binding_manifest={"budget_parent_turn_id": second.turn_id})
        store.complete_turn(turn_id=child_turn.turn_id, answer="child answer")
    before = records(database)
    summaries = list_chat_sessions(database, workspace=workspace)
    assert len(summaries) == 1
    assert summaries[0].thread_id == thread_id
    assert summaries[0].title == "first question"
    assert summaries[0].head_turn_id == second.turn_id
    assert summaries[0].status == "completed" and summaries[0].updated_at_ms > 0
    assert read_chat_messages(database, thread_id=thread_id, workspace=workspace) == (
        ("user", "first question"), ("answer", "saved answer"),
        ("user", "second question"), ("answer", "second answer"),
    )
    assert records(database) == before
    with pytest.raises(ValueError, match="其他工作区"):
        read_chat_messages(database, thread_id=thread_id, workspace=tmp_path / "other")


def test_fork_replay_uses_ancestor_cutoff_and_original_messages(tmp_path):
    database = tmp_path / "state.sqlite"
    thread_id, fork_point = conversation(database, tmp_path / "workspace")
    with RolloutStore(database) as store:
        later = store.start_turn(thread_id=thread_id, user_message="excluded later question", binding_manifest={})
        store.complete_turn(turn_id=later.turn_id, answer="excluded later answer")
        fork = store.fork_thread(from_turn_id=fork_point)
        turn = store.start_turn(thread_id=fork.thread_id, user_message="branch question", binding_manifest={})
        store.complete_turn(turn_id=turn.turn_id, answer="branch answer")
    messages = read_chat_messages(database, thread_id=fork.thread_id, workspace=tmp_path / "workspace")
    assert messages == (("user", "first question"), ("answer", "saved answer"),
                        ("user", "branch question"), ("answer", "branch answer"))


def test_no_saved_chats_does_not_create_database(tmp_path):
    database = tmp_path / "missing/state.sqlite"
    assert list_chat_sessions(database, workspace=tmp_path) == ()
    assert not database.parent.exists()


def test_replay_keeps_original_messages_after_context_compaction_and_deduplicates_answer(tmp_path):
    database = tmp_path / "state.sqlite"
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    with RolloutStore(database) as store:
        thread = store.create_thread(workspace=workspace)
        turn = store.start_turn(thread_id=thread.thread_id, user_message="original question", binding_manifest={})
        operation = store.prepare_model_operation(
            turn_id=turn.turn_id, request_hash="r", context_hash="c", tool_hash="t", wire_hash="w",
            request_ref={"request_id": "fixture"},
        )
        attempt = store.dispatch_model_attempt(operation.operation_id, now=1)
        assert store.complete_model_attempt(
            operation_id=operation.operation_id, attempt_id=attempt.attempt_id, generation=attempt.generation,
            text="final answer", provider_response_id=None, usage={}, now=2,
        )
        store.complete_turn(turn_id=turn.turn_id, answer="final answer")
        following = store.start_turn(thread_id=thread.thread_id, user_message="follow up", binding_manifest={})
        store.record_context_compaction(
            turn_id=following.turn_id, covered_item_ids=tuple(item.item_id for item in store.list_items(turn.turn_id)),
            summary="compressed summary", preserved_facts={name: [] for name in (
                "architecture_and_safety_constraints", "file_changes", "verification_results",
                "unresolved_work", "uncertain_side_effects",
            )}, context_version=1,
        )
        store.complete_turn(turn_id=following.turn_id, answer="second answer")
    before = records(database)
    messages = read_chat_messages(database, thread_id=thread.thread_id, workspace=workspace)
    assert messages == (("user", "original question"), ("answer", "final answer"),
                        ("user", "follow up"), ("answer", "second answer"))
    assert records(database) == before


@pytest.mark.anyio
async def test_reopen_and_copy_saved_answer_never_open_session(tmp_path, monkeypatch, capsys):
    database = tmp_path / "state.sqlite"
    workspace = tmp_path / "workspace"
    thread_id, _ = conversation(database, workspace)
    before = records(database)

    class Facade:
        checkpoint_db = database
        workspace_path = workspace

        def current_model(self):
            raise AssertionError("History browsing must work without an available model catalog")

        def session(self, **options):
            raise AssertionError("Browsing must not open runtime resources")

    commands = iter(["/copy", "/status", "/exit"])
    monkeypatch.setattr("builtins.input", lambda _prompt="": next(commands))
    await cli._chat_facade_loop(Facade(), max_tokens_total=None, resume_thread_id=thread_id)
    output = capsys.readouterr().out
    assert "first question" in output and output.count("saved answer") == 2
    assert thread_id in output and "模型: saved-model" in output
    assert records(database) == before


@pytest.mark.anyio
async def test_reopened_chat_submits_new_turn_in_original_thread(tmp_path, monkeypatch):
    database = tmp_path / "state.sqlite"
    workspace = tmp_path / "workspace"
    thread_id, previous_turn_id = conversation(database, workspace)
    opened = []

    class Facade:
        checkpoint_db = database
        workspace_path = workspace

        def current_model(self):
            return SimpleNamespace(id="saved-model")

        @asynccontextmanager
        async def session(self, **options):
            opened.append(options)
            assert options["thread_id"] == thread_id and options["previous_turn_id"] is None
            yield self

        async def submit(self, task, **options):
            with RolloutStore(database) as store:
                turn = store.start_turn(thread_id=thread_id, user_message=task, binding_manifest={})
                assert turn.predecessor_turn_id == previous_turn_id
                assert turn.turn_id != previous_turn_id
                assert [item.payload.get("text") for item in store.list_context_items(turn.turn_id)] == [
                    "first question", "saved answer", "follow up",
                ]
                store.complete_turn(turn_id=turn.turn_id, answer="new answer")
            return _result(turn_id=turn.turn_id, answer="new answer")

    commands = iter(["follow up", "/exit"])
    monkeypatch.setattr("builtins.input", lambda _prompt="": next(commands))
    await cli._chat_facade_loop(Facade(), max_tokens_total=None, resume_thread_id=thread_id)
    assert len(opened) == 1
    with RolloutStore(database, read_only=True) as store:
        assert len(store.list_threads()) == 1
        assert len(store.list_turns(thread_id)) == 2
        assert store.read_turn(previous_turn_id).status == "completed"


@pytest.mark.anyio
@pytest.mark.parametrize("tool_unknown", [False, True])
async def test_running_conversation_attempts_owned_recovery_when_user_sends(
    tmp_path, monkeypatch, capsys, tool_unknown,
):
    database = tmp_path / "state.sqlite"
    workspace = tmp_path / "workspace"
    thread_id, _ = conversation(database, workspace)
    with RolloutStore(database) as store:
        running = store.start_turn(thread_id=thread_id, user_message="unfinished", binding_manifest={})
    before = records(database)
    resumed = []

    class Facade:
        checkpoint_db = database
        workspace_path = workspace

        def current_model(self):
            return SimpleNamespace(id="caller-model")

        @asynccontextmanager
        async def session(self, **options):
            assert options["thread_id"] == thread_id
            yield self

        async def read_result(self, turn_id):
            assert turn_id == running.turn_id
            return _result(turn_id=turn_id, status="running")

        async def submit(self, task, **options):
            resumed.append(task)
            if tool_unknown:
                from agent_runtime.result import AgentPause
                return _result(turn_id=running.turn_id, status="paused", pause=AgentPause(
                    request_id="unknown", kind="tool_reconciliation", question="结果待核对", options=()))
            return _result(turn_id="turn_new", answer="continued")

        async def resume(self, *_args, **_options):
            raise AssertionError("Unknown tool effects must never be automatically aborted")

    commands = iter(["continue with my new requirement", "/exit"])
    monkeypatch.setattr("builtins.input", lambda _prompt="": next(commands))
    await cli._chat_facade_loop(Facade(), max_tokens_total=None, resume_thread_id=thread_id)
    output = capsys.readouterr().out
    assert resumed == ["continue with my new requirement"]
    assert "recover " not in output
    assert list_chat_sessions(database, workspace=workspace)[0].active_turn_id == running.turn_id
    assert records(database) == before


@pytest.mark.anyio
async def test_saved_clarification_continues_in_chat_after_user_reply(tmp_path, monkeypatch, capsys):
    from agent_runtime.agent import Agent

    database = tmp_path / "state.sqlite"
    workspace = tmp_path / "workspace"
    thread_id, _ = conversation(database, workspace)
    with RolloutStore(database) as store:
        turn = store.start_turn(thread_id=thread_id, user_message="original task", binding_manifest={})
        interaction = store.request_clarification(turn_id=turn.turn_id, question="Which destination?")
    before = records(database)
    recovery = []

    class Facade:
        checkpoint_db = database
        workspace_path = workspace

        def current_model(self):
            return SimpleNamespace(id="saved-model")

        async def read_result(self, turn_id):
            return await Agent(checkpoint_db=database, workspace_path=workspace).read_result(turn_id)

        def session(self, **options):
            raise AssertionError("Reply must resume the original Turn, not submit a new task")

        async def resume(self, turn_id, action, **options):
            # Inspection before the explicit reply has changed no durable facts.
            assert records(database) == before
            recovery.append((turn_id, action, options["user_input"]))
            with RolloutStore(database) as store:
                store.resolve_clarification(turn_id=turn_id, request_id=interaction.request_id,
                                            response=options["user_input"])
                store.complete_turn(turn_id=turn_id, answer="continued original task")
            return _result(turn_id=turn_id, answer="continued original task")

    commands = iter(["Shanghai", "/exit"])
    monkeypatch.setattr("builtins.input", lambda _prompt="": next(commands))
    await cli._chat_facade_loop(Facade(), max_tokens_total=None, resume_thread_id=thread_id)
    output = capsys.readouterr().out
    assert "Which destination?" in output and "recover " not in output
    assert recovery == [(turn.turn_id, "continue", "Shanghai")]


def test_resume_command_routes_thread_identity_without_task_recovery(monkeypatch):
    from typer.testing import CliRunner

    calls = []
    monkeypatch.setattr(cli, "agent_chat", lambda **options: calls.append(options))
    result = CliRunner().invoke(cli.agent_app, ["resume", "thread_saved"])
    assert result.exit_code == 0 and calls[0]["resume_thread_id"] == "thread_saved"
    calls.clear()
    result = CliRunner().invoke(cli.agent_app, ["resume"])
    assert result.exit_code == 0 and calls[0]["resume_thread_id"] == ""
    result = CliRunner().invoke(cli.agent_app, ["resume", "turn_old", "--action", "continue"])
    assert result.exit_code == 2  # Legacy task syntax is rejected, never silently recovered.


def test_continue_selects_recent_thread_in_current_workspace(tmp_path, monkeypatch):
    database = tmp_path / "state.sqlite"
    thread_id, _ = conversation(database, tmp_path / "workspace")
    conversation(database, tmp_path / "other")
    monkeypatch.chdir(tmp_path / "workspace")
    monkeypatch.setattr(cli, "_create_agent_facade", lambda **options: object())
    calls = []

    async def loop(facade, **options):
        calls.append(options)

    monkeypatch.setattr(cli, "_chat_facade_loop", loop)
    cli.agent_chat(checkpoint_db=database, continue_chat=True)
    assert calls[0]["resume_thread_id"] == thread_id and calls[0]["previous_turn_id"] is None


@pytest.mark.anyio
@pytest.mark.parametrize("action", ["continue", "retry", "abort"])
async def test_reopened_reconciliation_reaches_owned_recovery(tmp_path, monkeypatch, action):
    from agent_runtime.result import AgentPause
    database = tmp_path / "state.sqlite"
    workspace = tmp_path / "workspace"
    thread_id, _ = conversation(database, workspace)
    with RolloutStore(database) as store:
        turn = store.start_turn(thread_id=thread_id, user_message="interrupted command", binding_manifest={})
        store.pause_turn(turn_id=turn.turn_id, reason="command observation needs recovery")
    calls = []

    class Facade:
        checkpoint_db = database
        workspace_path = workspace

        def session(self, **options):
            raise AssertionError("A saved pending Turn must use the owned resume path")

        async def read_result(self, turn_id):
            return _result(turn_id=turn_id, status="paused", pause=AgentPause(
                request_id="pending-wait", kind="tool_reconciliation", question="needs recovery", options=()))

        async def resume(self, turn_id, action, **options):
            calls.append((turn_id, action, options.get("user_input")))
            return _result(turn_id=turn_id, answer="recovered original observation")

    commands = iter([action, "/exit"])
    monkeypatch.setattr("builtins.input", lambda _prompt="": next(commands))
    await cli._chat_facade_loop(Facade(), max_tokens_total=None, resume_thread_id=thread_id)
    assert calls == [(turn.turn_id, action, None)]


@pytest.mark.anyio
@pytest.mark.parametrize(("query", "reconciled", "driver"), [
    ("Stop the old work and explain the existing files", False, "agent"),
    ("Stop the old work and explain the existing files", True, "agent"),
    ("abort", False, "agent"),
    ("abort", True, "agent"),
    ("abort", False, "session"),
    ("abort", True, "session"),
])
async def test_reopened_reconciliation_preserves_new_request_in_owned_followup(
    tmp_path, monkeypatch, capsys, query, reconciled, driver,
):
    from agent_runtime import Agent
    from agent_runtime.harness import Session, ToolOrchestrator
    from agent_runtime.harness.ownership import OWNERSHIP_REVISION
    from tests.agent.harness.test_process_reopen import ReopenModel

    database = tmp_path / "state.sqlite"
    workspace = tmp_path / "workspace"
    thread_id, _ = conversation(database, workspace)
    with RolloutStore(database) as store:
        old = store.start_turn(thread_id=thread_id, user_message="interrupted operation",
                               binding_manifest={"execution_ownership": OWNERSHIP_REVISION})
        operation = store.prepare_model_operation(
            turn_id=old.turn_id, request_hash="r", context_hash="c", tool_hash="t", wire_hash="w",
            request_ref={"request_id": "old-request", "purpose": "agent_step"},
        )
        attempt = store.dispatch_model_attempt(operation.operation_id, worker_id="dead")
        assert store.complete_model_attempt(
            operation_id=operation.operation_id, attempt_id=attempt.attempt_id, generation=attempt.generation,
            text="", provider_response_id=None, usage={},
            tool_calls=({"id": "write-call", "name": "write_file", "arguments": {"path": "artifact"}},),
        )
        values = dict(turn_id=old.turn_id, operation_id="uncertain-write", tool_call_id="write-call",
                      tool_name="write_file", arguments_digest="args", execution_revision="write-v1",
                      idempotent=False, attempt_count=0, error_code=None, requires_reconciliation=False)
        store.record_tool_execution_state(status="prepared", **values)
        store.record_tool_execution_state(status="ready", **values)
        claim = store.claim_tool_operation(operation_id="uncertain-write", worker_id="dead", lease_seconds=30)
        assert store.commit_tool_operation_result(
            turn_id=old.turn_id, operation_id=claim.operation_id, claim_generation=claim.claim_generation,
            fencing_token=claim.fencing_token, status="unknown", attempt_count=1,
            error_code="execution_owner_lost", requires_reconciliation=True,
            result={"tool_call_id": "write-call", "tool_name": "write_file", "is_error": True,
                    "content": [], "attachments": []},
        )

    model = ReopenModel()
    observed = []

    async def observe(orchestrator, _commands, turn_id):
        # An independent receipt is available only in the recoverable case.
        pending = [item for item in orchestrator._store.list_interactions(turn_id)
                   if item.kind == "tool_reconciliation" and item.status == "pending"]
        observed.append(turn_id)
        if not reconciled or not pending:
            return False
        orchestrator._store.resolve_tool_reconciliation(
            turn_id=turn_id, status="succeeded", reconciler_revision="independent-receipt-v1",
            result={"tool_call_id": "write-call", "tool_name": "write_file", "is_error": False,
                    "content": [], "attachments": []},
        )
        return True

    monkeypatch.setattr(ToolOrchestrator, "reconcile_command_observation", observe)

    class Facade:
        checkpoint_db = database
        workspace_path = workspace

        async def read_result(self, turn_id):
            return await Agent(checkpoint_db=database, workspace_path=workspace).read_result(turn_id)

        async def resume(self, turn_id, action, **options):
            assert query == "abort" and action == "abort", "New requests must reach Session.submit"
            if reconciled:
                # A receipt can close the effect between the read and owned abort.
                with RolloutStore(database) as store:
                    store.resolve_tool_reconciliation(
                        turn_id=turn_id, status="succeeded", reconciler_revision="independent-receipt-v1",
                        result={"tool_call_id": "write-call", "tool_name": "write_file", "is_error": False,
                                "content": [], "attachments": []},
                    )
            if driver == "session":
                async with await Session.open(database=database, workspace=workspace, thread_id=thread_id,
                                              model=model, _background_memory=False) as session:
                    return await session.resume(turn_id, action, **options)
            return await Agent(checkpoint_db=database, workspace_path=workspace).resume(turn_id, action, **options)

        @asynccontextmanager
        async def session(self, **options):
            assert query != "abort", "Cancellation must not open a model Session"
            assert options["thread_id"] == thread_id
            async with await Session.open(database=database, workspace=workspace, thread_id=thread_id,
                                          model=model, _background_memory=False) as session:
                session._command_sessions = object()
                yield SimpleNamespace(current_model=lambda: SimpleNamespace(id="offline-model"),
                                      submit=session.submit)

    commands = iter([query, "/exit"])
    monkeypatch.setattr("builtins.input", lambda _prompt="": next(commands))
    await cli._chat_facade_loop(Facade(), max_tokens_total=None, resume_thread_id=thread_id)
    output = capsys.readouterr().out
    assert bool(observed) is (query != "abort")
    with RolloutStore(database, read_only=True) as store:
        assert store.verify().valid
        assert store.read_tool_operation("uncertain-write").attempt_count == 1
        turns = store.list_turns(thread_id)
        if reconciled:
            assert store.read_turn(old.turn_id).status == "cancelled"
            if query == "abort":
                assert len(turns) == 2 and not model.requests
            else:
                assert len(turns) == 3 and turns[-1].status == "completed"
                assert any(message.role == "user" and message.content == query
                           for message in model.requests[-1].messages)
        else:
            assert ("取消未完成" if query == "abort" else "输入未发送") in output
            assert store.read_turn(old.turn_id).status == "paused"
            assert store.read_tool_operation("uncertain-write").status == "unknown"
            assert any(item.kind == "tool_reconciliation" and item.status == "pending"
                       for item in store.list_interactions(old.turn_id))
            assert len(turns) == 2 and not model.requests
