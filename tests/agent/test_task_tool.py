from __future__ import annotations

import ast
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest

from agent_runtime.tools.executor import ToolExecutor
from agent_runtime.tools.integrations import subagent as subagent_module
from agent_runtime.tools.integrations.subagent import create_subagent_tool
from agent_runtime.tools.permissions import ToolExecutionContext
from agent_runtime.tools.tool import Tool, ToolCall, ToolCallOrigin


def _call(arguments: Mapping[str, Any]) -> ToolCall:
    return ToolCall(
        tool_call_id="call_task",
        tool_name="task",
        arguments=arguments,
        origin=ToolCallOrigin(
            request_id="request_task",
            toolset_revision="subagent-v1",
            exposed_tool_names=("task",),
        ),
    )


@pytest.mark.anyio
async def test_subagent_tool_projects_an_injected_runner() -> None:
    calls: list[Mapping[str, Any]] = []

    async def run_child(arguments: Mapping[str, Any]) -> Mapping[str, Any]:
        calls.append(arguments)
        return {
            "conclusion": "child conclusion",
            "key_facts": ["fact one"],
            "evidence_refs": [],
            "citations": [],
            "status": "done",
            "child_turn_id": "child-1",
            "stop_reason": "finished",
        }

    tool = create_subagent_tool(
        run_child,
        execution_revision="child-v2",
    )
    execution = await ToolExecutor({"task": tool}).execute(
        _call(
            {
                "task": "Investigate the isolated question",
                "context_summary": "Only bounded parent context.",
            }
        ),
        context=ToolExecutionContext(
            approved_tool_call_ids=frozenset({"call_task"})
        ),
    )

    assert isinstance(tool, Tool)
    assert tool.execution_revision.endswith(":child-v2")
    assert calls[0]["task"] == "Investigate the isolated question"
    assert execution.result.is_error is False
    output = execution.result.structured_content
    assert output is not None
    assert output["status"] == "done"
    assert output["conclusion"] == "child conclusion"
    assert output["key_facts"] == ("fact one",)
    assert execution.result.metadata["child_turn_id"] == "child-1"


@pytest.mark.anyio
async def test_subagent_failure_is_a_canonical_error_result() -> None:
    tool = create_subagent_tool(
        lambda _arguments: {
            "conclusion": "child failed",
            "status": "failed",
            "child_turn_id": "child-failed",
        }
    )

    execution = await ToolExecutor({"task": tool}).execute(
        _call({"task": "Fail cleanly."}),
        context=ToolExecutionContext(
            approved_tool_call_ids=frozenset({"call_task"})
        ),
    )

    assert execution.result.is_error is True
    assert execution.result.error_code == "subagent_failed"
    assert execution.result.error_message == "child failed"


@pytest.mark.anyio
async def test_subagent_effect_requires_external_approval() -> None:
    tool = create_subagent_tool(
        lambda _arguments: {
            "conclusion": "done",
            "status": "done",
            "child_turn_id": "child-1",
        }
    )

    execution = await ToolExecutor({"task": tool}).execute(
        _call({"task": "Delegate."}),
        context=ToolExecutionContext(),
    )

    assert execution.result.error_code == "approval_required"


def test_subagent_adapter_does_not_own_the_child_loop() -> None:
    module_path = Path(subagent_module.__file__ or "")
    source = module_path.read_text(encoding="utf-8")
    imports = {
        node.module
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.ImportFrom) and node.module is not None
    }

    assert not any(module.startswith("agent_runtime.loop") for module in imports)
    assert "AgentLoop" not in source
    assert "AgentService" not in source
    assert "DelegatedAgentRunner" not in source
