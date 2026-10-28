from __future__ import annotations

import json
from pathlib import Path

import pytest

from agent_runtime.harness import RuntimeState
from agent_runtime.harness.memory import create_memory_tool, memory_snapshot
from agent_runtime.harness.memory_management import create_memory_management_tool, remember_memory
from agent_runtime.harness.tool_orchestrator import ToolOrchestrator
from agent_runtime.tools.permissions import ToolExecutionContext
from agent_runtime.tools.tool import ToolCall, ToolCallOrigin


def _call(tool_name, arguments):
    return ToolCall(
        tool_call_id="memory-call",
        tool_name=tool_name,
        arguments=arguments,
        origin=ToolCallOrigin(request_id="request", toolset_revision="v1", exposed_tool_names=(tool_name,)),
    )


@pytest.mark.anyio
@pytest.mark.parametrize("action", ["remember", "edit", "forget"])
async def test_old_turn_cannot_mutate_unrelated_memory_at_reused_workspace_path(tmp_path: Path, action: str) -> None:
    old, new = tmp_path / "old", tmp_path / "new"
    old.mkdir()
    new.mkdir()
    async with await RuntimeState.open(old / "host.db") as store:
        original = remember_memory(store, old, key="runner", title="Runner", text="Use uv.")
        thread = store.create_thread(workspace=old)
        turn = store.start_turn(
            thread_id=thread.thread_id,
            user_message=f"Please {action} the runner preference.",
            binding_manifest={"memory_snapshot": memory_snapshot(store, old, navigation_only=True)},
        )
        store.relocate_memory(previous_workspace=old, workspace=new)
        unrelated = remember_memory(store, old, key="runner", title="Runner", text="Use make.")
        assert original["version"] == unrelated["version"] == 1
        executor = ToolOrchestrator(
            store=store,
            tools={"manage_memory": create_memory_management_tool(store)},
            execution_context=ToolExecutionContext(workspace_root=old, cwd=old, allow_write_tools=True),
        )
        arguments = {"action": action, "key": "runner", "expected_version": original["version"]}
        if action != "forget":
            arguments["text"] = "Use pytest."
        if action == "remember":
            arguments["key"] = "new-key"
        output = await executor.execute(turn_id=turn.turn_id, call=_call("manage_memory", arguments))
        assert output.is_error, output
        assert store.read_memory_state(new) == original
        assert store.read_memory_state(old) == unrelated


@pytest.mark.anyio
@pytest.mark.parametrize("tool_name", ["read_memory", "manage_memory"])
async def test_old_turn_memory_reads_stay_in_original_scope_after_path_reuse(tmp_path: Path, tool_name: str) -> None:
    old, new = tmp_path / "old", tmp_path / "new"
    old.mkdir()
    new.mkdir()
    async with await RuntimeState.open(old / "host.db") as store:
        original = remember_memory(store, old, key="runner", title="Runner", text="Use uv.")
        thread = store.create_thread(workspace=old)
        turn = store.start_turn(
            thread_id=thread.thread_id,
            user_message="Explain the runner preference.",
            binding_manifest={"memory_snapshot": memory_snapshot(store, old, navigation_only=True)},
        )
        store.relocate_memory(previous_workspace=old, workspace=new)
        unrelated = remember_memory(store, old, key="runner", title="Runner", text="Use make.")
        tools = {"read_memory": create_memory_tool(store), "manage_memory": create_memory_management_tool(store)}
        executor = ToolOrchestrator(
            store=store,
            tools=tools,
            execution_context=ToolExecutionContext(workspace_root=old, cwd=old, allow_write_tools=True),
        )
        arguments = {"key": "runner"}
        if tool_name == "manage_memory":
            arguments["action"] = "explain"
        output = await executor.execute(turn_id=turn.turn_id, call=_call(tool_name, arguments))
        assert not output.is_error, output
        result = json.loads(output.content[0].data["text"])
        entry = result["entry"] if tool_name == "manage_memory" else result["entries"][0]
        assert entry == original["entries"][0]
        assert store.read_memory_state(new) == original
        assert store.read_memory_state(old) == unrelated
