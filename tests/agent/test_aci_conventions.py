from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import pytest

from agent_runtime.tools.builtins import (
    RESIDENT_CODING_TOOL_NAMES,
    create_resident_coding_tools,
)
from agent_runtime.tools.executor import ToolExecutor
from agent_runtime.tools.permissions import ToolExecutionContext
from agent_runtime.tools.selection import find_tools
from agent_runtime.tools.tool import Tool, ToolCall, ToolCallOrigin, ToolResult
from agent_runtime.workspace import WorkspaceRuntime, open_workspace


def _tools(workspace: WorkspaceRuntime) -> dict[str, Tool]:
    values = create_resident_coding_tools(
        workspace,
        plan_updater=lambda _arguments: {
            "accepted": True,
            "revision": 1,
            "message": "updated",
        },
    )
    return {tool.definition.name: tool for tool in values}


def _call(name: str, arguments: Mapping[str, object]) -> ToolCall:
    return ToolCall(
        tool_call_id=f"call_{name}",
        tool_name=name,
        arguments=arguments,
        origin=ToolCallOrigin(
            request_id="request_aci",
            toolset_revision="aci-v1",
            exposed_tool_names=(name,),
        ),
    )


def test_resident_aci_is_small_ordered_and_self_documenting(tmp_path: Path) -> None:
    tools = tuple(_tools(open_workspace(tmp_path, create=True)).values())

    assert tuple(tool.definition.name for tool in tools) == (
        RESIDENT_CODING_TOOL_NAMES
    )
    for tool in tools:
        assert len(tool.definition.description) >= 80
        properties = tool.definition.input_schema.get("properties")
        assert isinstance(properties, Mapping)
        assert properties
        assert all(
            isinstance(schema, Mapping) and schema.get("description")
            for schema in properties.values()
        )


def test_resident_aci_spells_out_observed_model_argument_pitfalls(
    tmp_path: Path,
) -> None:
    tools = _tools(open_workspace(tmp_path, create=True))

    read_properties = tools["read_file"].definition.input_schema["properties"]
    assert isinstance(read_properties, Mapping)
    max_bytes = read_properties["max_bytes"]
    assert isinstance(max_bytes, Mapping)
    assert "1,000,000" in str(max_bytes["description"])
    assert max_bytes["default"] == 16_000

    update = tools["update_plan"].definition
    assert '"step"' in update.description
    assert '"status"' in update.description
    assert "advisory" in update.description
    assert "never prove" in update.description

    assert "preferred first tool" in tools["search_text"].definition.description
    assert "not a code-search tool" in tools["list_files"].definition.description
    assert "focused verification" in tools["run_command"].definition.description


@pytest.mark.anyio
async def test_invalid_input_becomes_one_canonical_tool_result(
    tmp_path: Path,
) -> None:
    workspace = open_workspace(tmp_path, create=True)
    tool = _tools(workspace)["read_file"]

    execution = await ToolExecutor({"read_file": tool}).execute(
        _call("read_file", {}),
        context=ToolExecutionContext(
            workspace_root=workspace.root,
            cwd=workspace.root,
        ),
    )

    assert isinstance(execution.result, ToolResult)
    assert execution.result.is_error is True
    assert execution.result.error_code == "invalid_arguments"
    assert execution.result.content[0].type == "text"
    assert execution.result.tool_call_id == "call_read_file"


@pytest.mark.anyio
async def test_success_output_is_structured_once_without_formatter(
    tmp_path: Path,
) -> None:
    workspace = open_workspace(tmp_path, create=True)
    (workspace.root / "notes.txt").write_text("one\ntwo\n", encoding="utf-8")
    tool = _tools(workspace)["read_file"]

    execution = await ToolExecutor({"read_file": tool}).execute(
        _call("read_file", {"path": "notes.txt", "max_bytes": 100}),
        context=ToolExecutionContext(
            workspace_root=workspace.root,
            cwd=workspace.root,
        ),
    )

    assert execution.result.is_error is False
    output = execution.result.structured_content
    assert output is not None
    assert output["path"] == "notes.txt"
    assert output["content"] == "one\ntwo\n"
    assert output["size_bytes"] == 8
    assert output["truncated"] is False
    assert output["is_binary"] is False
    assert execution.result.content == ()


@pytest.mark.anyio
async def test_write_effect_is_approval_gated_by_execution_context(
    tmp_path: Path,
) -> None:
    workspace = open_workspace(tmp_path, create=True)
    (workspace.root / "notes.txt").write_text("before", encoding="utf-8")
    tool = _tools(workspace)["apply_patch"]
    call = _call(
        "apply_patch",
        {
            "file_path": "notes.txt",
            "old_string": "before",
            "new_string": "after",
        },
    )

    blocked = await ToolExecutor({"apply_patch": tool}).execute(
        call,
        context=ToolExecutionContext(
            workspace_root=workspace.root,
            cwd=workspace.root,
        ),
    )
    allowed = await ToolExecutor({"apply_patch": tool}).execute(
        call,
        context=ToolExecutionContext(
            workspace_root=workspace.root,
            cwd=workspace.root,
            allow_write_tools=True,
        ),
    )

    assert blocked.result.error_code == "approval_required"
    assert allowed.result.is_error is False
    assert (workspace.root / "notes.txt").read_text(encoding="utf-8") == "after"


def test_hidden_tool_search_is_deterministic_and_bounded(
    tmp_path: Path,
) -> None:
    snapshot = _tools(open_workspace(tmp_path, create=True))

    first = find_tools(
        snapshot,
        query="执行终端命令",
        discoverable_names=("run_command", "search_text"),
        limit=1,
    )
    second = find_tools(
        snapshot,
        query="执行终端命令",
        discoverable_names=("run_command", "search_text"),
        limit=1,
    )

    assert first == second
    assert tuple(match.name for match in first.matches) == ("run_command",)
    assert first.proposed_activation_names == ("run_command",)
