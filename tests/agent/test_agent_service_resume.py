from __future__ import annotations

import asyncio
from collections.abc import Mapping
from pathlib import Path

import pytest
from langgraph.checkpoint.memory import MemorySaver

from agent_runtime.core.checkpointing import agent_checkpoint_serde
from agent_runtime.core.definition import AgentRuntimePolicy
from agent_runtime.core.human_input import HumanInputRequest
from agent_runtime.core.turn_contracts import ToolCallPlan
from agent_runtime.loop.state import LoopState, ModelTurnDraft
from agent_runtime.service import AgentRunRequest, AgentService
from agent_runtime.tools.registry import ToolRegistry
from agent_runtime.tools.tool import (
    CancellationMode,
    InterruptBehavior,
    JsonValue,
    NormalizedToolOutput,
    ResolvedToolUse,
    Tool,
    ToolDefinition,
    ToolEffect,
    json_schema_input,
)
from agent_runtime.turns import (
    RuntimeBinding,
    TurnStateError,
    TurnStatus,
    TurnStore,
)
from agent_runtime.workspace import open_workspace


class _FinishProvider:
    async def next_turn(
        self,
        state: LoopState,
        *,
        definition: AgentRuntimePolicy,
        budget_remaining: int,
    ) -> ModelTurnDraft:
        del definition, budget_remaining
        if state["tool_results"]:
            payload = state["tool_results"][-1].structured_content
            if isinstance(payload, Mapping):
                text = payload.get("text")
                if isinstance(text, str):
                    return ModelTurnDraft(
                        action="finish",
                        final_answer=text,
                    )
        return ModelTurnDraft(action="finish", final_answer="done")


class _SlowToolCallingProvider:
    def __init__(self, call: ToolCallPlan) -> None:
        self._call = call

    async def next_turn(
        self,
        state: LoopState,
        *,
        definition: AgentRuntimePolicy,
        budget_remaining: int,
    ) -> ModelTurnDraft:
        del definition, budget_remaining
        if state["tool_results"]:
            return ModelTurnDraft(action="finish", final_answer="done")
        await asyncio.sleep(0.05)
        return ModelTurnDraft(action="execute", tool_calls=(self._call,))


def _tool(
    calls: list[str],
    *,
    execution_revision: str = "remote-v1",
) -> Tool:
    schema: Mapping[str, JsonValue] = {
        "type": "object",
        "properties": {"value": {"type": "string"}},
        "required": ["value"],
        "additionalProperties": False,
    }

    def run(arguments: Mapping[str, JsonValue]) -> object:
        value = str(arguments["value"])
        calls.append(value)
        return {"text": f"ran:{value}"}

    return Tool(
        definition=ToolDefinition(
            name="remote_write",
            description="Perform one remote write.",
            input_schema=schema,
        ),
        validate_input=json_schema_input(schema),
        run=run,
        normalize_output=lambda raw: NormalizedToolOutput(
            structured_content=raw  # type: ignore[arg-type]
        ),
        output_schema=None,
        static_effects=frozenset({ToolEffect.NETWORK}),
        resolve_use=lambda _arguments: ResolvedToolUse(
            effects=frozenset({ToolEffect.NETWORK}),
            targets=(),
        ),
        execution_revision=execution_revision,
        idempotent=False,
        concurrency_safe=False,
        cancellation_mode=CancellationMode.COOPERATIVE,
        interrupt_behavior=InterruptBehavior.CANCEL,
        timeout_seconds=1.0,
        max_model_output_bytes=4096,
    )


def _definition() -> AgentRuntimePolicy:
    return AgentRuntimePolicy.test_factory(
        system_prompt="Use the remote tool.",
        allowed_tools=["remote_write"],
        max_iterations=4,
    )


def _approval_id(request: HumanInputRequest) -> str:
    summary = request.tool_calls[0]
    return summary.approval_id or summary.tool_call_id


def _service(
    calls: list[str],
    *,
    turn_store: TurnStore,
    workspace_path: Path,
    checkpointer: MemorySaver | None = None,
    execution_revision: str = "remote-v1",
    model_turn_provider: _FinishProvider | _SlowToolCallingProvider | None = None,
    model_alias: str | None = None,
    bind_workspace: bool = True,
) -> AgentService:
    registry = ToolRegistry()
    registry.register(_tool(calls, execution_revision=execution_revision))
    return AgentService(
        definition=_definition(),
        tool_registry=registry,
        model_turn_provider=model_turn_provider or _FinishProvider(),
        checkpointer=checkpointer,
        workspace=open_workspace(workspace_path),
        turn_store=turn_store,
        runtime_binding=RuntimeBinding(
            model_alias=model_alias,
            workspace_path=str(workspace_path) if bind_workspace else None,
        ),
    )


async def _pause(
    service: AgentService,
    *,
    turn_store: TurnStore,
    value: str,
):
    call = ToolCallPlan.create("remote_write", {"value": value})
    result = await service.run(
        AgentRunRequest(
            message="Run one remote write.",
            pending_tool_calls=[call],
        ),
    )
    assert result.status == "paused"
    assert result.human_input_request is not None
    assert turn_store.get_turn(result.turn_id).status is TurnStatus.PAUSED
    return call, result


@pytest.mark.anyio
async def test_default_checkpointer_resumes_on_same_service_without_replay(
    tmp_path: Path,
) -> None:
    calls: list[str] = []
    turn_store = TurnStore(tmp_path / "agent.sqlite")
    service = _service(
        calls,
        turn_store=turn_store,
        workspace_path=tmp_path,
    )
    call, paused = await _pause(
        service,
        turn_store=turn_store,
        value="once",
    )
    assert _approval_id(paused.human_input_request) == call.tool_call_id

    resumed = await service.resume_turn(
        turn_id=paused.turn_id,
        action="allow_once",
        user_input=None,
    )

    assert resumed.status == "done"
    assert resumed.final_answer == "ran:once"
    assert calls == ["once"]
    assert resumed.tool_results[0].is_error is False
    assert turn_store.get_turn(paused.turn_id).status is TurnStatus.COMPLETED


@pytest.mark.anyio
async def test_resume_accepts_workspace_allocated_when_the_turn_started(
    tmp_path: Path,
) -> None:
    calls: list[str] = []
    turn_store = TurnStore(tmp_path / "agent.sqlite")
    service = _service(
        calls,
        turn_store=turn_store,
        workspace_path=tmp_path,
        model_alias="model-a",
        bind_workspace=False,
    )
    _call, paused = await _pause(
        service,
        turn_store=turn_store,
        value="allocated",
    )
    persisted = turn_store.get_turn(paused.turn_id)
    assert persisted.runtime == RuntimeBinding(
        model_alias="model-a",
        workspace_path=str(tmp_path),
    )

    resumed = await service.resume_turn(
        turn_id=paused.turn_id,
        action="allow_once",
        user_input=None,
    )

    assert resumed.status == "done"
    assert calls == ["allocated"]


@pytest.mark.anyio
async def test_resume_latency_is_cumulative_across_approval_pause(
    tmp_path: Path,
) -> None:
    calls: list[str] = []
    call = ToolCallPlan.create("remote_write", {"value": "timed"})
    turn_store = TurnStore(tmp_path / "agent.sqlite")
    service = _service(
        calls,
        turn_store=turn_store,
        workspace_path=tmp_path,
        model_turn_provider=_SlowToolCallingProvider(call),
    )

    paused = await service.run(
        AgentRunRequest(
            message="Run one remote write.",
        ),
    )
    assert paused.status == "paused"
    assert paused.latency_profile is not None
    assert paused.human_input_request is not None
    assert _approval_id(paused.human_input_request) == call.tool_call_id
    assert turn_store.get_turn(paused.turn_id).status is TurnStatus.PAUSED

    resumed = await service.resume_turn(
        turn_id=paused.turn_id,
        action="allow_once",
        user_input=None,
    )

    assert resumed.status == "done"
    assert resumed.latency_profile is not None
    assert resumed.latency_profile.total_ms >= paused.latency_profile.total_ms
    assert resumed.latency_profile.total_ms >= (
        resumed.latency_profile.model_latency_ms + resumed.latency_profile.tool_latency_ms
    )


@pytest.mark.anyio
async def test_resume_uses_same_codec_across_service_boundary(
    tmp_path: Path,
) -> None:
    checkpointer = MemorySaver(serde=agent_checkpoint_serde())
    calls: list[str] = []
    turn_store = TurnStore(tmp_path / "agent.sqlite")
    first = _service(
        calls,
        turn_store=turn_store,
        workspace_path=tmp_path,
        checkpointer=checkpointer,
    )
    _call, paused = await _pause(
        first,
        turn_store=turn_store,
        value="persisted",
    )

    second = _service(
        calls,
        turn_store=turn_store,
        workspace_path=tmp_path,
        checkpointer=checkpointer,
    )
    pending = second.pending_human_input_request(turn_id=paused.turn_id)
    assert pending.request_id == paused.human_input_request.request_id
    assert _approval_id(pending) == _approval_id(paused.human_input_request)
    resumed = await second.resume_turn(
        turn_id=paused.turn_id,
        action="allow_once",
        user_input=None,
    )

    assert resumed.status == "done"
    assert calls == ["persisted"]


@pytest.mark.anyio
async def test_resume_rejects_a_service_bound_to_a_different_model(
    tmp_path: Path,
) -> None:
    checkpointer = MemorySaver(serde=agent_checkpoint_serde())
    calls: list[str] = []
    turn_store = TurnStore(tmp_path / "agent.sqlite")
    first = _service(
        calls,
        turn_store=turn_store,
        workspace_path=tmp_path,
        checkpointer=checkpointer,
        model_alias="model-a",
    )
    _call, paused = await _pause(
        first,
        turn_store=turn_store,
        value="persisted",
    )
    wrong_model = _service(
        calls,
        turn_store=turn_store,
        workspace_path=tmp_path,
        checkpointer=checkpointer,
        model_alias="model-b",
    )

    with pytest.raises(TurnStateError, match="resume runtime does not match"):
        await wrong_model.resume_turn(
            turn_id=paused.turn_id,
            action="allow_once",
            user_input=None,
        )

    assert turn_store.get_turn(paused.turn_id).status is TurnStatus.PAUSED
    assert calls == []


@pytest.mark.anyio
async def test_resume_preserves_request_max_turns_across_service_boundary(
    tmp_path: Path,
) -> None:
    checkpointer = MemorySaver(serde=agent_checkpoint_serde())
    calls: list[str] = []
    call = ToolCallPlan.create("remote_write", {"value": "bounded"})
    turn_store = TurnStore(tmp_path / "agent.sqlite")
    first = _service(
        calls,
        turn_store=turn_store,
        workspace_path=tmp_path,
        checkpointer=checkpointer,
        model_turn_provider=_SlowToolCallingProvider(call),
    )

    paused = await first.run(
        AgentRunRequest(
            message="Use one model turn, then write.",
            max_turns=1,
        ),
    )

    assert paused.status == "paused"
    assert paused.iteration == 1
    assert paused.human_input_request is not None
    assert _approval_id(paused.human_input_request) == call.tool_call_id
    second = _service(
        calls,
        turn_store=turn_store,
        workspace_path=tmp_path,
        checkpointer=checkpointer,
    )
    resumed = await second.resume_turn(
        turn_id=paused.turn_id,
        action="allow_once",
        user_input=None,
    )

    assert calls == ["bounded"]
    assert resumed.status == "failed"
    assert resumed.stop_reason == "max_turns"
    assert resumed.iteration == 1


@pytest.mark.anyio
async def test_changed_pending_tool_definition_requires_reconciliation(
    tmp_path: Path,
) -> None:
    checkpointer = MemorySaver(serde=agent_checkpoint_serde())
    calls: list[str] = []
    turn_store = TurnStore(tmp_path / "agent.sqlite")
    first = _service(
        calls,
        turn_store=turn_store,
        workspace_path=tmp_path,
        checkpointer=checkpointer,
        execution_revision="remote-v1",
    )
    _call, paused = await _pause(
        first,
        turn_store=turn_store,
        value="drifted",
    )
    second = _service(
        calls,
        turn_store=turn_store,
        workspace_path=tmp_path,
        checkpointer=checkpointer,
        execution_revision="remote-v2",
    )

    result = await second.resume_turn(
        turn_id=paused.turn_id,
        action="allow_once",
        user_input=None,
    )

    assert result.status == "paused"
    assert result.needs_user_input == "tool_definition_changed"
    assert result.human_input_request.kind == "tool_reconciliation"
    reconciliation_request_id = result.human_input_request.request_id
    assert calls == []
    assert turn_store.get_turn(paused.turn_id).status is TurnStatus.PAUSED

    pending = await second.apending_human_input_request(turn_id=paused.turn_id)
    assert pending.request_id == reconciliation_request_id
    with pytest.raises(
        ValueError,
        match="only supports mark_failed or abort",
    ):
        await second.resume_turn(
            turn_id=paused.turn_id,
            action="mark_completed",
            user_input=None,
        )

    resolved = await second.resume_turn(
        turn_id=paused.turn_id,
        action="mark_failed",
        user_input=None,
    )

    assert resolved.status == "done"
    assert calls == []
    assert resolved.tool_results[-1].error_code == "reconciled_failed"
    assert resolved.tool_results[-1].metadata["reconciled"] is True
    assert turn_store.get_turn(paused.turn_id).status is TurnStatus.COMPLETED


@pytest.mark.anyio
async def test_changed_pending_tool_definition_can_be_aborted(
    tmp_path: Path,
) -> None:
    checkpointer = MemorySaver(serde=agent_checkpoint_serde())
    calls: list[str] = []
    turn_store = TurnStore(tmp_path / "agent.sqlite")
    first = _service(
        calls,
        turn_store=turn_store,
        workspace_path=tmp_path,
        checkpointer=checkpointer,
        execution_revision="remote-v1",
    )
    _call, paused = await _pause(
        first,
        turn_store=turn_store,
        value="abort-drifted",
    )
    second = _service(
        calls,
        turn_store=turn_store,
        workspace_path=tmp_path,
        checkpointer=checkpointer,
        execution_revision="remote-v2",
    )
    reconciliation = await second.resume_turn(
        turn_id=paused.turn_id,
        action="allow_once",
        user_input=None,
    )
    assert reconciliation.human_input_request is not None
    assert reconciliation.human_input_request.kind == "tool_reconciliation"

    aborted = await second.resume_turn(
        turn_id=paused.turn_id,
        action="abort",
        user_input=None,
    )

    assert aborted.status == "failed"
    assert aborted.stop_reason == "user_aborted"
    assert calls == []
    assert turn_store.get_turn(paused.turn_id).status is TurnStatus.FAILED


@pytest.mark.anyio
async def test_resume_rejects_unsupported_tool_approval_action(
    tmp_path: Path,
) -> None:
    turn_store = TurnStore(tmp_path / "agent.sqlite")
    service = _service(
        [],
        turn_store=turn_store,
        workspace_path=tmp_path,
    )
    _call, paused = await _pause(
        service,
        turn_store=turn_store,
        value="x",
    )

    with pytest.raises(
        ValueError,
        match="supports allow_once, deny, or abort",
    ):
        await service.resume_turn(
            turn_id=paused.turn_id,
            action="continue",
            user_input=None,
        )

    assert turn_store.get_turn(paused.turn_id).status is TurnStatus.PAUSED
