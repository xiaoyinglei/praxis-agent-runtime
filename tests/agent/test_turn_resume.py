from __future__ import annotations

import shlex
import shutil
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import replace

import pytest
from langgraph.checkpoint.memory import MemorySaver

from rag.agent.core.checkpointing import (
    LangGraphCheckpointStore,
    agent_checkpoint_serde,
)
from rag.agent.core.definition import AgentRuntimePolicy
from rag.agent.core.human_input import HumanInputRequest, ToolCallSummary
from rag.agent.core.messages import ModelMessage
from rag.agent.core.turn_contracts import ToolCallPlan
from rag.agent.loop.runtime import ModelTurnEnvelope
from rag.agent.loop.state import LoopPause, LoopState, ModelTurnDraft
from rag.agent.memory.compactor import LoopContextCompactor
from rag.agent.memory.models import MemoryPolicy
from rag.agent.service import AgentRunRequest, AgentService
from rag.agent.tools.builtins.shell import create_run_command_tool
from rag.agent.tools.executor import ExecutionStatus, ToolExecutionRecord
from rag.agent.tools.registry import ToolRegistry
from rag.agent.tools.tool import (
    CancellationMode,
    InterruptBehavior,
    JsonValue,
    NormalizedToolOutput,
    ResolvedToolUse,
    Tool,
    ToolDefinition,
    json_schema_input,
)
from rag.agent.turns import (
    RuntimeBinding,
    TurnStateError,
    TurnStatus,
    TurnStore,
)
from rag.agent.workspace import open_workspace


class _PauseProvider:
    async def next_turn(
        self,
        state: LoopState,
        *,
        definition: AgentRuntimePolicy,
        budget_remaining: int,
    ) -> ModelTurnDraft:
        del state, definition, budget_remaining
        return ModelTurnDraft(
            action="pause",
            pause_reason="Need more input.",
        )


class _FinishProvider:
    def __init__(self, answer: str = "done") -> None:
        self.answer = answer
        self.observed: list[LoopState] = []

    async def next_turn(
        self,
        state: LoopState,
        *,
        definition: AgentRuntimePolicy,
        budget_remaining: int,
    ) -> ModelTurnEnvelope:
        del definition, budget_remaining
        self.observed.append(deepcopy(state))
        return ModelTurnEnvelope(
            draft=ModelTurnDraft(
                action="finish",
                final_answer=self.answer,
            ),
            assistant_message=ModelMessage(
                role="assistant",
                content=self.answer,
            ),
        )


class _CommandProvider:
    def __init__(self, plan: ToolCallPlan) -> None:
        self.plan = plan
        self.observed: list[LoopState] = []

    async def next_turn(
        self,
        state: LoopState,
        *,
        definition: AgentRuntimePolicy,
        budget_remaining: int,
    ) -> ModelTurnDraft:
        del definition, budget_remaining
        self.observed.append(deepcopy(state))
        if state["tool_results"]:
            return ModelTurnDraft(action="finish", final_answer="command done")
        return ModelTurnDraft(action="execute", tool_calls=(self.plan,))


def _definition() -> AgentRuntimePolicy:
    return AgentRuntimePolicy.test_factory(
        system_prompt="Resume safely.",
        allowed_tools=[],
        max_iterations=3,
    )


def _service(
    tmp_path,
    *,
    store: TurnStore,
    checkpointer: MemorySaver,
    provider: object,
    registry: ToolRegistry | None = None,
    definition: AgentRuntimePolicy | None = None,
) -> AgentService:
    return AgentService(
        definition=definition or _definition(),
        tool_registry=registry or ToolRegistry(),
        model_turn_provider=provider,
        checkpointer=checkpointer,
        workspace=open_workspace(tmp_path),
        turn_store=store,
        runtime_binding=RuntimeBinding(workspace_path=str(tmp_path)),
    )


async def _save_paused_state(
    service: AgentService,
    checkpointer: MemorySaver,
    request: AgentRunRequest,
    human_request: HumanInputRequest,
) -> None:
    state = service.initial_state(request)
    state["status"] = "paused"
    state["approval_request"] = human_request
    state["pause"] = LoopPause(
        reason=human_request.question,
        request=human_request,
    )
    await LangGraphCheckpointStore(
        checkpointer,
        run_config=request.to_run_config(_definition()),
    ).save_snapshot(state, reason="test_pause")


@pytest.mark.anyio
async def test_paused_turn_cannot_receive_a_followup(
    tmp_path,
) -> None:
    store = TurnStore(tmp_path / "agent.sqlite")
    checkpointer = MemorySaver(serde=agent_checkpoint_serde())
    service = _service(
        tmp_path,
        store=store,
        checkpointer=checkpointer,
        provider=_PauseProvider(),
    )

    paused = await service.run(
        AgentRunRequest(message="first"),
    )

    assert paused.status == "paused"
    assert store.get_turn(paused.turn_id).status is TurnStatus.PAUSED
    with pytest.raises(TurnStateError, match="only terminal Turns"):
        await service.run(
            AgentRunRequest(
                message="must fail",
                previous_turn_id=paused.turn_id,
            ),
        )


@pytest.mark.anyio
async def test_resume_fails_loud_when_turn_checkpoint_is_missing(
    tmp_path,
) -> None:
    store = TurnStore(tmp_path / "agent.sqlite")
    checkpointer = MemorySaver(serde=agent_checkpoint_serde())
    turn = store.begin_turn(
        "missing checkpoint",
        RuntimeBinding(workspace_path=str(tmp_path)),
    )
    store.mark_paused(turn.turn_id)
    service = _service(
        tmp_path,
        store=store,
        checkpointer=checkpointer,
        provider=_FinishProvider(),
    )

    with pytest.raises(KeyError, match="No checkpoint found"):
        await service.resume_turn(
            turn_id=turn.turn_id,
            action="continue",
        )

    assert store.get_turn(turn.turn_id).status is TurnStatus.PAUSED


@pytest.mark.anyio
async def test_resume_interrupted_turn_hydrates_full_predecessor_history(
    tmp_path,
) -> None:
    store = TurnStore(tmp_path / "agent.sqlite")
    checkpointer = MemorySaver(serde=agent_checkpoint_serde())
    runtime = RuntimeBinding(workspace_path=str(tmp_path))
    first = store.begin_turn(
        "remember alpha",
        runtime,
    )
    store.sync_turn_messages(
        first.turn_id,
        (
            ModelMessage(role="user", content="remember alpha"),
            ModelMessage(role="assistant", content="remembered"),
        ),
    )
    store.mark_terminal(first.turn_id, TurnStatus.COMPLETED)
    second = store.begin_turn(
        "what did I ask you to remember?",
        runtime,
        previous_turn_id=first.turn_id,
        lease_owner="dead-worker",
    )
    seed = _service(
        tmp_path,
        store=store,
        checkpointer=checkpointer,
        provider=_FinishProvider(),
    )
    request = AgentRunRequest(
        message="what did I ask you to remember?",
        previous_turn_id=first.turn_id,
        turn_id=second.turn_id,
        conversation_history=[
            ModelMessage(role="user", content="remember alpha"),
            ModelMessage(role="assistant", content="remembered"),
        ],
    )
    state = seed.initial_state(request)
    await LangGraphCheckpointStore(
        checkpointer,
        run_config=request.to_run_config(_definition()),
    ).save_snapshot(state, reason="process_interrupted")
    store.mark_interrupted(second.turn_id)
    provider = _FinishProvider("alpha")
    restored = _service(
        tmp_path,
        store=store,
        checkpointer=checkpointer,
        provider=provider,
    )

    result = await restored.resume_turn(
        turn_id=second.turn_id,
        action="continue",
        user_input=None,
    )

    assert result.status == "done"
    assert provider.observed[0]["current_message"] == "what did I ask you to remember?"
    assert provider.observed[0]["conversation_history"] == [
        ModelMessage(role="user", content="remember alpha"),
        ModelMessage(role="assistant", content="remembered"),
    ]
    assert provider.observed[0]["turn_transcript"] == [
        ModelMessage(
            role="user",
            content="what did I ask you to remember?",
        ),
    ]
    assert store.history_through(second.turn_id) == (
        ModelMessage(role="user", content="remember alpha"),
        ModelMessage(role="assistant", content="remembered"),
        ModelMessage(
            role="user",
            content="what did I ask you to remember?",
        ),
        ModelMessage(role="assistant", content="alpha"),
    )
    assert store.get_turn(second.turn_id).status is TurnStatus.COMPLETED


@pytest.mark.anyio
async def test_resume_repairs_checkpoint_first_compaction_rewrite(
    tmp_path,
) -> None:
    store = TurnStore(tmp_path / "agent.sqlite")
    checkpointer = MemorySaver(serde=agent_checkpoint_serde())
    runtime = RuntimeBinding(workspace_path=str(tmp_path))
    turn = store.begin_turn(
        "compact before resume",
        runtime,
        lease_owner="dead-worker",
    )
    request = AgentRunRequest(
        message="compact before resume",
        turn_id=turn.turn_id,
        memory_policy=MemoryPolicy(
            message_compaction_min_count=99,
            reactive_compact_tail_count=2,
        ),
    )
    seed = _service(
        tmp_path,
        store=store,
        checkpointer=checkpointer,
        provider=_FinishProvider(),
    )
    state = seed.initial_state(request)
    original = [
        ModelMessage(role="user", content=request.message),
        *[
            ModelMessage(
                role="assistant" if index % 2 == 0 else "user",
                content=f"resume-{index}: " + (f"token-{index} " * 500),
            )
            for index in range(6)
        ],
    ]
    state["turn_transcript"] = list(original)
    store.sync_turn_messages(turn.turn_id, original)
    compaction = LoopContextCompactor().reactive_compact(state)
    assert compaction.changed is True
    compacted = tuple(state["turn_transcript"])
    await LangGraphCheckpointStore(
        checkpointer,
        run_config=state["run_config"],
    ).save_snapshot(
        state,
        reason="checkpoint_before_turn_store_sync",
    )
    store.mark_interrupted(turn.turn_id)

    provider = _FinishProvider("resumed after compaction")
    restored = _service(
        tmp_path,
        store=store,
        checkpointer=checkpointer,
        provider=provider,
    )
    result = await restored.resume_turn(
        turn_id=turn.turn_id,
        action="continue",
    )

    assert result.status == "done"
    assert provider.observed[0]["turn_transcript"] == list(compacted)
    persisted = store.turn_history(turn.turn_id)
    assert persisted[: len(compacted)] == compacted
    assert any(
        '"event_type":"context_compaction"' in message.content
        for message in persisted
    )


@pytest.mark.anyio
async def test_resume_clarification_appends_user_input_to_same_turn(
    tmp_path,
) -> None:
    store = TurnStore(tmp_path / "agent.sqlite")
    checkpointer = MemorySaver(serde=agent_checkpoint_serde())
    turn = store.begin_turn(
        "Which target?",
        RuntimeBinding(workspace_path=str(tmp_path)),
    )
    seed = _service(
        tmp_path,
        store=store,
        checkpointer=checkpointer,
        provider=_FinishProvider(),
    )
    run_request = AgentRunRequest(
        message="Which target?",
        turn_id=turn.turn_id,
    )
    human_request = HumanInputRequest(
        request_id="hir_clarify",
        kind="clarification",
        question="Which target should I use?",
        options=["continue", "abort"],
    )
    await _save_paused_state(
        seed,
        checkpointer,
        run_request,
        human_request,
    )
    store.mark_paused(turn.turn_id)
    provider = _FinishProvider("target accepted")
    restored = _service(
        tmp_path,
        store=store,
        checkpointer=checkpointer,
        provider=provider,
    )

    result = await restored.resume_turn(
        turn_id=turn.turn_id,
        action="continue",
        user_input="Use production.",
    )

    assert result.status == "done"
    assert provider.observed[0]["turn_transcript"] == [
        ModelMessage(role="user", content="Which target?"),
        ModelMessage(role="user", content="Use production."),
    ]
    assert store.history_through(turn.turn_id) == (
        ModelMessage(role="user", content="Which target?"),
        ModelMessage(role="user", content="Use production."),
        ModelMessage(role="assistant", content="target accepted"),
    )


@pytest.mark.anyio
async def test_resume_tool_approval_maps_action_to_pending_calls(
    tmp_path,
) -> None:
    store = TurnStore(tmp_path / "agent.sqlite")
    checkpointer = MemorySaver(serde=agent_checkpoint_serde())
    turn = store.begin_turn(
        "Approve it",
        RuntimeBinding(workspace_path=str(tmp_path)),
    )
    seed = _service(
        tmp_path,
        store=store,
        checkpointer=checkpointer,
        provider=_FinishProvider(),
    )
    run_request = AgentRunRequest(
        message="Approve it",
        turn_id=turn.turn_id,
    )
    human_request = HumanInputRequest(
        request_id="hir_approve",
        kind="tool_approval",
        question="Approve write?",
        tool_calls=[
            ToolCallSummary(
                tool_call_id="call_write",
                approval_id="call_write::network",
                tool_name="write_file",
                args_preview="path='result.txt'",
            )
        ],
        options=["allow_once", "deny", "abort"],
    )
    await _save_paused_state(
        seed,
        checkpointer,
        run_request,
        human_request,
    )
    store.mark_paused(turn.turn_id)
    provider = _FinishProvider("approved")
    restored = _service(
        tmp_path,
        store=store,
        checkpointer=checkpointer,
        provider=provider,
    )

    await restored.resume_turn(
        turn_id=turn.turn_id,
        action="allow_once",
        user_input=None,
    )

    assert provider.observed[0]["approved_tool_call_ids"] == ["call_write::network"]


@pytest.mark.anyio
@pytest.mark.skipif(
    shutil.which("sandbox-exec") is None,
    reason="Seatbelt sandbox-exec is not available on this platform",
)
async def test_run_command_network_uses_two_checkpointed_approvals(
    tmp_path,
) -> None:
    store = TurnStore(tmp_path / "agent.sqlite")
    checkpointer = MemorySaver(serde=agent_checkpoint_serde())
    workspace = open_workspace(tmp_path)
    sentinel = workspace.root / "approved-command.txt"
    command = f"printf executed > {shlex.quote(str(sentinel))}"
    plan = ToolCallPlan.create(
        "run_command",
        {
            "command": command,
            "working_dir": ".",
            "timeout_seconds": 2,
            "network": True,
            "workspace_write": True,
        },
    )
    definition = AgentRuntimePolicy.test_factory(
        system_prompt="Run the approved command.",
        allowed_tools=["run_command"],
        max_iterations=4,
    )

    def service(provider: _CommandProvider) -> AgentService:
        registry = ToolRegistry()
        registry.register(create_run_command_tool(workspace))
        return _service(
            tmp_path,
            store=store,
            checkpointer=checkpointer,
            provider=provider,
            registry=registry,
            definition=definition,
        )

    first = await service(_CommandProvider(plan)).run(
        AgentRunRequest(message="Run the command with network enabled."),
    )

    assert first.status == "paused"
    assert sentinel.exists() is False
    assert first.human_input_request is not None
    tool_approval = first.human_input_request
    assert tool_approval.context["approval_scope"] == "tool"
    assert tool_approval.context["workspace_write"] is True
    assert tool_approval.context["workspace_path"] == str(workspace.root)
    assert tool_approval.tool_calls[0].approval_id == plan.tool_call_id
    assert command in tool_approval.tool_calls[0].args_preview
    assert "workspace write: requested for entire workspace" in (
        tool_approval.tool_calls[0].args_preview
    )
    assert str(workspace.root) in tool_approval.tool_calls[0].args_preview
    assert "destructive operation" in tool_approval.tool_calls[0].reason
    assert "destructive write access to the entire workspace" in (
        tool_approval.question
    )

    second = await service(_CommandProvider(plan)).resume_turn(
        turn_id=first.turn_id,
        action="allow_once",
    )

    assert second.status == "paused"
    assert sentinel.exists() is False
    assert second.human_input_request is not None
    network_approval = second.human_input_request
    assert network_approval.context["approval_scope"] == "network"
    assert network_approval.tool_calls[0].approval_id == (f"{plan.tool_call_id}::network")

    final_provider = _CommandProvider(plan)
    final = await service(final_provider).resume_turn(
        turn_id=first.turn_id,
        action="allow_once",
    )

    assert final.status == "done"
    assert sentinel.read_text(encoding="utf-8") == "executed"
    assert final_provider.observed[-1]["approved_tool_call_ids"] == [
        plan.tool_call_id,
        f"{plan.tool_call_id}::network",
    ]


@pytest.mark.anyio
async def test_outcome_unknown_reconciliation_never_replays_side_effect(
    tmp_path,
) -> None:
    runner_calls = 0
    schema: Mapping[str, JsonValue] = {
        "type": "object",
        "properties": {"value": {"type": "string"}},
        "required": ["value"],
        "additionalProperties": False,
    }

    def runner(_arguments: Mapping[str, JsonValue]) -> object:
        nonlocal runner_calls
        runner_calls += 1
        return {"value": "must not run"}

    tool = Tool(
        definition=ToolDefinition(
            name="remote_write",
            description="Perform a non-idempotent write.",
            input_schema=schema,
        ),
        validate_input=json_schema_input(schema),
        run=runner,
        normalize_output=lambda _raw: NormalizedToolOutput(),
        output_schema=None,
        static_effects=frozenset(),
        resolve_use=lambda _arguments: ResolvedToolUse(
            effects=frozenset(),
            targets=(),
        ),
        execution_revision="remote-write-v1",
        idempotent=False,
        concurrency_safe=False,
        cancellation_mode=CancellationMode.REMOTE_BEST_EFFORT,
        interrupt_behavior=InterruptBehavior.CANCEL,
        timeout_seconds=1.0,
        max_model_output_bytes=4096,
    )
    registry = ToolRegistry()
    registry.register(tool)
    definition = AgentRuntimePolicy.test_factory(
        system_prompt="Reconcile safely.",
        allowed_tools=["remote_write"],
        max_iterations=3,
    )
    store = TurnStore(tmp_path / "agent.sqlite")
    checkpointer = MemorySaver(serde=agent_checkpoint_serde())
    turn = store.begin_turn(
        "Reconcile write",
        RuntimeBinding(workspace_path=str(tmp_path)),
    )
    seed = _service(
        tmp_path,
        store=store,
        checkpointer=checkpointer,
        provider=_FinishProvider(),
        registry=registry,
        definition=definition,
    )
    plan = ToolCallPlan.create("remote_write", {"value": "once"})
    run_request = AgentRunRequest(
        message="Reconcile write",
        turn_id=turn.turn_id,
        pending_tool_calls=[plan],
    )
    state = seed.initial_state(run_request)
    canonical = state["canonical_tool_calls"][plan.tool_call_id]
    state["tool_execution_records"][plan.tool_call_id] = replace(
        ToolExecutionRecord.prepare(canonical, tool),
        status=ExecutionStatus.OUTCOME_UNKNOWN,
        attempt_count=1,
        error_code="interrupted_outcome_unknown",
        requires_reconciliation=True,
    )
    await LangGraphCheckpointStore(
        checkpointer,
        run_config=run_request.to_run_config(definition),
    ).save_snapshot(state, reason="crash_after_started")
    store.mark_interrupted(turn.turn_id)
    provider = _FinishProvider("reconciled")
    restored = _service(
        tmp_path,
        store=store,
        checkpointer=checkpointer,
        provider=provider,
        registry=registry,
        definition=definition,
    )

    result = await restored.resume_turn(
        turn_id=turn.turn_id,
        action="mark_completed",
        user_input=None,
    )

    assert result.status == "done"
    assert runner_calls == 0
    assert result.tool_results[0].metadata["reconciled"] is True
