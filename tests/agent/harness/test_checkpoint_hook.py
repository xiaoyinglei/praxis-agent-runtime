"""The WRITE_WORKSPACE effect is the single checkpoint trigger at Turn scale."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import pytest

from agent_runtime.checkpoints import CheckpointService
from agent_runtime.harness import RolloutStore, ToolOrchestrator
from agent_runtime.harness.tool_orchestrator import ToolApprovalRequiredError
from agent_runtime.tools.permissions import ToolExecutionContext
from agent_runtime.tools.registry import ToolRegistry
from agent_runtime.tools.tool import (
    CancellationMode,
    InterruptBehavior,
    JsonValue,
    NormalizedToolOutput,
    ResolvedToolUse,
    Tool,
    ToolCall,
    ToolCallOrigin,
    ToolContentBlock,
    ToolDefinition,
    ToolEffect,
    ToolTarget,
    json_schema_input,
)

pytestmark = pytest.mark.anyio

_SCHEMA: Mapping[str, JsonValue] = {
    "type": "object",
    "properties": {
        "path": {"type": "string"},
        "content": {"type": "string"},
        "workspace_write": {"type": "boolean"},
    },
    "required": ("path", "content"),
    "additionalProperties": False,
}


def _file_tool(workspace: Path) -> Tool:
    def run(arguments: Mapping[str, JsonValue]) -> dict[str, str]:
        (workspace / str(arguments["path"])).write_text(str(arguments["content"]))
        return {"text": "written"}

    def resolve_use(arguments: Mapping[str, JsonValue]) -> ResolvedToolUse:
        effects = {ToolEffect.READ_WORKSPACE}
        if arguments.get("workspace_write") is True:
            effects.add(ToolEffect.WRITE_WORKSPACE)
        return ResolvedToolUse(
            effects=frozenset(effects),
            targets=(ToolTarget(kind="workspace_path", value=str(workspace)),),
        )

    return Tool(
        definition=ToolDefinition(
            name="write_file",
            description="Write one workspace file.",
            input_schema=_SCHEMA,
        ),
        validate_input=json_schema_input(_SCHEMA),
        run=run,
        normalize_output=lambda raw: NormalizedToolOutput(
            content=(ToolContentBlock(type="text", data=raw),),
        ),
        output_schema=None,
        static_effects=frozenset({ToolEffect.READ_WORKSPACE}),
        resolve_use=resolve_use,
        execution_revision="write-file-v1",
        idempotent=False,
        concurrency_safe=False,
        cancellation_mode=CancellationMode.COOPERATIVE,
        interrupt_behavior=InterruptBehavior.CANCEL,
        timeout_seconds=1.0,
        max_model_output_bytes=4_096,
    )


def _call(name: str, arguments: Mapping[str, JsonValue]) -> ToolCall:
    return ToolCall(
        tool_call_id=f"call-{name}",
        tool_name="write_file",
        arguments=dict(arguments),
        origin=ToolCallOrigin(
            request_id="request-1",
            toolset_revision="toolset-v1",
            exposed_tool_names=("write_file",),
        ),
    )


def _orchestrator(
    store: RolloutStore, workspace: Path, checkpoints: CheckpointService | None,
) -> ToolOrchestrator:
    registry = ToolRegistry()
    registry.register(_file_tool(workspace))
    return ToolOrchestrator(
        store=store,
        tools=registry.freeze(),
        execution_context=ToolExecutionContext(workspace_root=workspace, cwd=workspace),
        checkpoints=checkpoints,
    )


def _turn(store: RolloutStore, workspace: Path, message: str) -> str:
    thread = store.create_thread(workspace=workspace)
    return store.start_turn(
        thread_id=thread.thread_id,
        user_message=message,
        binding_manifest={"model_id": "test-model"},
    ).turn_id


def _pre_write_items(store: RolloutStore, turn_id: str) -> list[str]:
    return [
        str(item.payload.get("checkpoint_id"))
        for item in store.list_items(turn_id)
        if item.kind == "workspace_checkpoint" and item.payload.get("trigger") == "pre_write"
    ]


async def _execute_with_approval(
    orchestrator: ToolOrchestrator, turn_id: str, call: ToolCall,
) -> object:
    try:
        return (await orchestrator.execute_batch(turn_id=turn_id, calls=(call,)))[0]
    except ToolApprovalRequiredError:
        return await orchestrator.resume_approval(turn_id=turn_id, decision="approve")


async def test_write_call_snapshots_pre_turn_state(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "a.txt").write_text("before")
    service = CheckpointService(workspace)
    with RolloutStore(tmp_path / "rollout.sqlite3") as store:
        turn_id = _turn(store, workspace, "change the file")
        orchestrator = _orchestrator(store, workspace, service)
        result = await _execute_with_approval(orchestrator, turn_id, _call(
            "1", {"path": "a.txt", "content": "after", "workspace_write": True},
        ))
        assert getattr(result, "is_error", True) is False
        [checkpoint_id] = _pre_write_items(store, turn_id)
    # The snapshot holds the pre-write content: restoring it undoes the tool.
    result = await service.restore(checkpoint_id)
    assert (workspace / "a.txt").read_text() == "before"
    assert result.restored == ("a.txt",)


async def test_one_checkpoint_per_turn_across_batches(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    service = CheckpointService(workspace)
    with RolloutStore(tmp_path / "rollout.sqlite3") as store:
        turn_id = _turn(store, workspace, "two writes")
        orchestrator = _orchestrator(store, workspace, service)
        await _execute_with_approval(orchestrator, turn_id, _call(
            "1", {"path": "one.txt", "content": "1", "workspace_write": True},
        ))
        await _execute_with_approval(orchestrator, turn_id, _call(
            "2", {"path": "two.txt", "content": "2", "workspace_write": True},
        ))
        assert len(_pre_write_items(store, turn_id)) == 1


async def test_read_only_calls_do_not_checkpoint(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    service = CheckpointService(workspace)
    with RolloutStore(tmp_path / "rollout.sqlite3") as store:
        turn_id = _turn(store, workspace, "no writes")
        orchestrator = _orchestrator(store, workspace, service)
        results = await orchestrator.execute_batch(turn_id=turn_id, calls=(
            _call("1", {"path": "a.txt", "content": "ignored"}),
        ))
        assert results[0].is_error is False
        assert _pre_write_items(store, turn_id) == []
    assert service.shadow_root.exists() is False


async def test_invalid_arguments_do_not_checkpoint(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    service = CheckpointService(workspace)
    with RolloutStore(tmp_path / "rollout.sqlite3") as store:
        turn_id = _turn(store, workspace, "broken call")
        orchestrator = _orchestrator(store, workspace, service)
        results = await orchestrator.execute_batch(turn_id=turn_id, calls=(
            _call("1", {"path": 42, "content": "x", "workspace_write": True}),
        ))
        assert results[0].is_error is True
        assert _pre_write_items(store, turn_id) == []


async def test_unavailable_service_never_blocks_the_write(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    with RolloutStore(tmp_path / "rollout.sqlite3") as store:
        turn_id = _turn(store, workspace, "still writes")
        orchestrator = _orchestrator(store, workspace, None)
        result = await _execute_with_approval(orchestrator, turn_id, _call(
            "1", {"path": "a.txt", "content": "data", "workspace_write": True},
        ))
        assert getattr(result, "is_error", True) is False
        assert (workspace / "a.txt").read_text() == "data"
        assert _pre_write_items(store, turn_id) == []
        assert store.verify().valid is True


async def test_agent_facade_rewind_restores_workspace(tmp_path: Path) -> None:
    workspace = tmp_path / "project"
    workspace.mkdir()
    (workspace / "a.txt").write_text("original")
    service = CheckpointService(workspace)
    with RolloutStore(tmp_path / "rollout.sqlite3") as store:
        thread = store.create_thread(workspace=workspace)
        turn_id = store.start_turn(
            thread_id=thread.thread_id,
            user_message="mutate",
            binding_manifest={"model_id": "test-model"},
        ).turn_id
        orchestrator = _orchestrator(store, workspace, service)
        await _execute_with_approval(orchestrator, turn_id, _call(
            "1", {"path": "a.txt", "content": "mutated", "workspace_write": True},
        ))
        store.complete_turn(turn_id=turn_id, answer="done")
        [checkpoint_id] = _pre_write_items(store, turn_id)

    from agent_runtime.agent import Agent

    agent = Agent(workspace_path=workspace, checkpoint_db=tmp_path / "rollout.sqlite3")
    rows = await agent.list_checkpoints()
    assert [row["checkpoint_id"] for row in rows] == [checkpoint_id]
    result = await agent.rewind(checkpoint_id)
    assert result["restored"] == ["a.txt"]
    assert (workspace / "a.txt").read_text() == "original"


async def test_agent_facade_rewind_refuses_active_turn(tmp_path: Path) -> None:
    workspace = tmp_path / "project"
    workspace.mkdir()
    with RolloutStore(tmp_path / "rollout.sqlite3") as store:
        thread = store.create_thread(workspace=workspace)
        store.start_turn(
            thread_id=thread.thread_id,
            user_message="running",
            binding_manifest={"model_id": "test-model"},
        )

    from agent_runtime.agent import Agent

    agent = Agent(workspace_path=workspace, checkpoint_db=tmp_path / "rollout.sqlite3")
    with pytest.raises(RuntimeError, match="未结束"):
        await agent.rewind("0" * 40)


async def test_rewind_anchor_is_listed_and_undoable(tmp_path: Path) -> None:
    workspace = tmp_path / "project"
    workspace.mkdir()
    (workspace / "a.txt").write_text("v1")
    service = CheckpointService(workspace)
    with RolloutStore(tmp_path / "rollout.sqlite3") as store:
        thread = store.create_thread(workspace=workspace)
        turn_id = store.start_turn(
            thread_id=thread.thread_id,
            user_message="mutate",
            binding_manifest={"model_id": "test-model"},
        ).turn_id
        orchestrator = _orchestrator(store, workspace, service)
        await _execute_with_approval(orchestrator, turn_id, _call(
            "1", {"path": "a.txt", "content": "v2", "workspace_write": True},
        ))
        store.complete_turn(turn_id=turn_id, answer="done")
        [checkpoint_id] = _pre_write_items(store, turn_id)

    from agent_runtime.agent import Agent

    agent = Agent(workspace_path=workspace, checkpoint_db=tmp_path / "rollout.sqlite3")
    result = await agent.rewind(checkpoint_id)
    anchor_id = result["restore_anchor_id"]
    assert isinstance(anchor_id, str) and anchor_id
    rows = await agent.list_checkpoints()
    triggers = [row["trigger"] for row in rows]
    ids = [row["checkpoint_id"] for row in rows]
    assert "restore" in triggers
    assert anchor_id in ids
    # Undoing the rewind through the public listing restores the mutated state.
    undo = await agent.rewind(anchor_id)
    assert (workspace / "a.txt").read_text() == "v2"
    assert undo["restored"] == ["a.txt"]


async def test_broken_checkpoint_storage_never_blocks_the_write(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    blocker = tmp_path / "blocked-shadow"
    blocker.write_text("regular file blocking the shadow repository")
    service = CheckpointService(workspace, shadow_root=blocker)
    with RolloutStore(tmp_path / "rollout.sqlite3") as store:
        turn_id = _turn(store, workspace, "write despite broken safety net")
        orchestrator = _orchestrator(store, workspace, service)
        result = await _execute_with_approval(orchestrator, turn_id, _call(
            "1", {"path": "a.txt", "content": "written", "workspace_write": True},
        ))
        assert getattr(result, "is_error", True) is False
        assert (workspace / "a.txt").read_text() == "written"
        assert _pre_write_items(store, turn_id) == []
