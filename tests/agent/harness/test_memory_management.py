from __future__ import annotations

import json

import pytest

from agent_runtime.harness import RuntimeState


def test_explicit_correction_is_versioned_and_forget_can_be_explicitly_reversed(tmp_path):
    from agent_runtime.harness.memory import delete_memory
    from agent_runtime.harness.memory_management import edit_memory, explain_memory, remember_memory

    with RuntimeState.open_sync(tmp_path / "r.db") as store:
        first = remember_memory(
            store, tmp_path, key="tests", title="Test command", text="Use pytest.", kind="preference"
        )
        second = edit_memory(store, tmp_path, key="tests", text="Use uv run pytest.", expected_version=first["version"])
        old = store.read_memory_state(tmp_path, version=first["version"])["entries"][0]
        new = second["entries"][0]
        assert old["text"] == "Use pytest." and new["text"] == "Use uv run pytest."
        assert new["origin"] == "explicit" and new["supersedes"] == old["entry_id"]
        assert new["source_item_ids"] != old["source_item_ids"]
        assert new["evidence"] == ["user_message"] and new["kind"] == "preference"
        explanation = explain_memory(store, tmp_path, key="tests")
        assert [item["action"] for item in explanation["history"]] == ["edit", "remember"]
        deleted = delete_memory(store, tmp_path, "tests")
        assert (
            deleted["entries"] == []
            and explain_memory(store, tmp_path, key="tests")["history"][0]["action"] == "forget"
        )
        restored = remember_memory(store, tmp_path, key="tests", title="Test command", text="Use make test.")
        assert restored["entries"][0]["text"] == "Use make test."
        assert "tests" not in restored["deleted_keys"]


def test_stale_manual_edit_is_rejected_without_creating_audit_turn(tmp_path):
    from agent_runtime.harness.memory_management import edit_memory, remember_memory
    from agent_runtime.harness.protocol import ContextSourceChangedError

    with RuntimeState.open_sync(tmp_path / "r.db") as store:
        remember_memory(store, tmp_path, key="k", title="Title", text="first")
        before = store.list_turns()
        with pytest.raises(ContextSourceChangedError):
            edit_memory(store, tmp_path, key="k", text="stale", expected_version=0)
        assert store.list_turns() == before
        assert store.read_memory_state(tmp_path)["entries"][0]["text"] == "first"


@pytest.mark.anyio
async def test_automatic_history_cannot_overwrite_explicit_correction(tmp_path):
    from agent_runtime.harness.memory import update_memory
    from agent_runtime.harness.memory_management import remember_memory
    from tests.agent.harness.test_cross_session_memory import MemoryModel, seed

    async with (await RuntimeState.open(tmp_path / "r.db")) as store:
        remember_memory(store, tmp_path, key="environment", title="Command runner", text="Use make.", kind="preference")
        seed(store, tmp_path, text="old source: use uv")
        result = await update_memory(store, workspace=tmp_path, model=MemoryModel(), binding={})
        assert result["status"] == "updated"
        assert store.read_memory_state(tmp_path)["entries"][0]["text"] == "Use make."


def test_cli_management_and_sdk_explanation_require_no_model(tmp_path):
    from typer.testing import CliRunner

    from agent_runtime import Agent
    from agent_runtime.cli import agent_app

    database = tmp_path / "r.db"
    common = ["--workspace", str(tmp_path), "--checkpoint-db", str(database)]
    runner = CliRunner()
    first = runner.invoke(
        agent_app,
        [
            "memory",
            "remember",
            "--key",
            "k",
            "--title",
            "Testing",
            "--text",
            "Run pytest.",
            "--kind",
            "preference",
            *common,
        ],
    )
    assert first.exit_code == 0, first.output
    updated = runner.invoke(agent_app, ["memory", "edit", "--key", "k", "--text", "Run uv run pytest.", *common])
    assert updated.exit_code == 0, updated.output
    agent = Agent(workspace_path=tmp_path, checkpoint_db=database)
    detail = agent.explain_memory("k")
    assert detail["entry"]["text"] == "Run uv run pytest."
    assert detail["history"][0]["action"] == "edit"
    assert runner.invoke(agent_app, ["memory", "explain", "--key", "k", *common]).exit_code == 0


@pytest.mark.anyio
async def test_management_tool_uses_current_user_provenance_and_rejects_stale_write(tmp_path):
    from agent_runtime.harness.memory import memory_snapshot
    from agent_runtime.harness.memory_management import create_memory_management_tool
    from agent_runtime.harness.tool_orchestrator import ToolOrchestrator
    from agent_runtime.tools.permissions import ToolExecutionContext
    from agent_runtime.tools.tool import ToolCall, ToolCallOrigin

    async with (await RuntimeState.open(tmp_path / "r.db")) as store:
        thread = store.create_thread(workspace=tmp_path)
        turn = store.start_turn(
            thread_id=thread.thread_id,
            user_message="Remember the test runner is uv.",
            binding_manifest={"memory_snapshot": memory_snapshot(store, tmp_path, navigation_only=True)},
        )
        executor = ToolOrchestrator(
            store=store,
            tools={"manage_memory": create_memory_management_tool(store)},
            execution_context=ToolExecutionContext(workspace_root=tmp_path, cwd=tmp_path, allow_write_tools=True),
        )

        def call(identifier, version):
            return ToolCall(
                tool_call_id=identifier,
                tool_name="manage_memory",
                arguments={
                    "action": "remember",
                    "key": "runner",
                    "text": "Use uv.",
                    "kind": "preference",
                    "expected_version": version,
                },
                origin=ToolCallOrigin(request_id="r", toolset_revision="v", exposed_tool_names=("manage_memory",)),
            )

        output = await executor.execute(turn_id=turn.turn_id, call=call("write-1", 0))
        assert not output.is_error, output
        entry = json.loads(output.content[0].data["text"])["entry"]
        assert entry["source_item_ids"] == [store.list_items(turn.turn_id)[0].item_id]
        stale = await executor.execute(turn_id=turn.turn_id, call=call("write-2", 0))
        assert stale.is_error


@pytest.mark.anyio
async def test_readonly_tool_policy_cannot_mutate_memory(tmp_path):
    from agent_runtime.harness.memory import memory_snapshot
    from agent_runtime.harness.memory_management import create_memory_management_tool
    from agent_runtime.harness.tool_orchestrator import ToolApprovalRequiredError, ToolOrchestrator
    from agent_runtime.tools.permissions import ToolExecutionContext
    from agent_runtime.tools.tool import ToolCall, ToolCallOrigin

    async with (await RuntimeState.open(tmp_path / "r.db")) as store:
        thread = store.create_thread(workspace=tmp_path)
        turn = store.start_turn(
            thread_id=thread.thread_id,
            user_message="Remember uv.",
            binding_manifest={"memory_snapshot": memory_snapshot(store, tmp_path)},
        )
        executor = ToolOrchestrator(
            store=store,
            tools={"manage_memory": create_memory_management_tool(store)},
            execution_context=ToolExecutionContext(workspace_root=tmp_path, cwd=tmp_path, allow_write_tools=False),
        )
        with pytest.raises(ToolApprovalRequiredError):
            await executor.execute(
                turn_id=turn.turn_id,
                call=ToolCall(
                    tool_call_id="denied",
                    tool_name="manage_memory",
                    arguments={"action": "remember", "key": "k", "text": "uv", "expected_version": 0},
                    origin=ToolCallOrigin(request_id="r", toolset_revision="v", exposed_tool_names=("manage_memory",)),
                ),
            )
        assert store.read_memory_state(tmp_path)["entries"] == []


@pytest.mark.anyio
async def test_empty_memory_exposes_version_through_read_for_first_explicit_write(tmp_path):
    from agent_runtime.harness import HarnessModelResponse, HarnessToolCall, Session
    from agent_runtime.harness.memory import MemoryConfig
    from agent_runtime.tools.permissions import ToolExecutionContext
    from tests.agent.test_update_plan_surfaces import _UpdatePlanThenAnswerModel

    class Model(_UpdatePlanThenAnswerModel):
        def __init__(self):
            self.requests = []

        def prepare(self, request):
            self.requests.append(request)
            return super().prepare(request)

        async def dispatch(self, prepared, *, event_sink=None):
            step = prepared.request_ref["step"]
            if step == 1:
                call = HarnessToolCall(id="read", name="read_memory", arguments={})
            elif step == 2:
                detail = next(m.content for m in reversed(self.requests[-1].messages) if m.role == "tool")
                # Tool results use the normal ACI envelope; the version is not
                # a volatile navigation field in every model request.
                version = json.loads(json.loads(detail)["content"][0]["data"]["text"])["version"]
                call = HarnessToolCall(id="write", name="manage_memory", arguments={
                    "action": "remember", "key": "runner", "text": "Use uv.", "expected_version": version})
            else:
                return HarnessModelResponse(text="remembered", provider_response_id=None, usage={})
            return HarnessModelResponse(text="", provider_response_id=None, usage={}, tool_calls=(call,))

    model = Model()
    async with await Session.open(
        database=tmp_path / "r.db", workspace=tmp_path, model=model,
        memory=MemoryConfig(generate_memories=False), require_workspace_change=False,
        tool_execution_context=ToolExecutionContext(workspace_root=tmp_path, cwd=tmp_path, allow_write_tools=True),
    ) as session:
        result = await session.submit("Remember a new preference: use uv.")
        assert result.status == "done", result.diagnostics
        assert all(not call.is_error for call in result.tool_calls)
        assert session.store.read_memory_state(tmp_path)["entries"][0]["text"] == "Use uv."
    navigation = [
        message.content for message in model.requests[0].messages
        if message.content.startswith("Cross-chat memory")
    ]
    assert len(navigation) == 1
    snapshot = json.loads(navigation[0].split("\n", 1)[1])
    assert "version" not in snapshot
    assert "read_memory" in snapshot["recall"]
    assert snapshot["entries"] == []
