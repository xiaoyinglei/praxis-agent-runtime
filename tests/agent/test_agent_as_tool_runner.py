from __future__ import annotations

from collections.abc import Mapping

import pytest

from agent_runtime.tools.executor import ToolExecutor
from agent_runtime.tools.integrations.subagent import (
    SubagentOutput,
    create_subagent_tool,
)
from agent_runtime.tools.permissions import ToolExecutionContext
from agent_runtime.tools.tool import JsonValue, ToolCall, ToolCallOrigin


def _call(arguments: Mapping[str, JsonValue]) -> ToolCall:
    return ToolCall(
        tool_call_id="call-subagent",
        tool_name="task",
        arguments=arguments,
        origin=ToolCallOrigin(
            request_id="request-subagent",
            toolset_revision="tools-subagent",
            exposed_tool_names=("task",),
        ),
    )


@pytest.mark.anyio
async def test_subagent_factory_is_an_ordinary_tool_and_normalizes_output() -> None:
    seen: list[Mapping[str, JsonValue]] = []

    async def run(arguments: Mapping[str, JsonValue]) -> object:
        seen.append(arguments)
        return SubagentOutput(
            conclusion="Child conclusion.",
            key_facts=["fact"],
            status="done",
            child_turn_id="child-1",
        )

    tool = create_subagent_tool(run)
    execution = await ToolExecutor({"task": tool}).execute(
        _call({"task": "Inspect module."}),
        context=ToolExecutionContext(
            approved_tool_call_ids=frozenset({"call-subagent"})
        ),
    )

    assert seen[0]["task"] == "Inspect module."
    assert execution.result.is_error is False
    assert execution.result.structured_content["conclusion"] == (
        "Child conclusion."
    )
    assert execution.result.metadata["child_turn_id"] == "child-1"


@pytest.mark.anyio
async def test_subagent_non_done_status_becomes_canonical_error_result() -> None:
    async def run(_arguments: Mapping[str, JsonValue]) -> object:
        return SubagentOutput(
            conclusion="Needs approval.",
            status="paused",
            child_turn_id="child-paused",
        )

    tool = create_subagent_tool(run)
    execution = await ToolExecutor({"task": tool}).execute(
        _call({"task": "Do bounded work."}),
        context=ToolExecutionContext(
            approved_tool_call_ids=frozenset({"call-subagent"})
        ),
    )

    assert execution.result.is_error is True
    assert execution.result.error_code == "subagent_paused"
    assert execution.result.retryable is True


@pytest.mark.anyio
async def test_subagent_network_effect_requires_approval() -> None:
    tool = create_subagent_tool(
        lambda _arguments: SubagentOutput(
            conclusion="done",
            status="done",
        )
    )

    execution = await ToolExecutor({"task": tool}).execute(
        _call({"task": "Delegate."}),
        context=ToolExecutionContext(),
    )

    assert execution.result.is_error is True
    assert execution.result.error_code == "approval_required"
    assert execution.record is None


def test_subagent_tool_has_bounded_schema_and_execution_contract() -> None:
    tool = create_subagent_tool(lambda _arguments: {})

    assert tool.definition.name == "task"
    assert tool.definition.input_schema["additionalProperties"] is False
    assert tool.idempotent is False
    assert tool.concurrency_safe is False
    assert tool.timeout_seconds == 180.0
