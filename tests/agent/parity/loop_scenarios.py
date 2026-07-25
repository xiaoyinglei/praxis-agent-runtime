from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from tempfile import TemporaryDirectory

from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver

from rag.agent.core.checkpointing import (
    LangGraphCheckpointStore,
    agent_checkpoint_serde,
)
from rag.agent.core.context import TurnRegistry
from rag.agent.core.definition import AgentRuntimePolicy
from rag.agent.core.finalization import FinishCandidateBuilder
from rag.agent.core.goal_contract import GoalDeliverable, GoalSpec
from rag.agent.core.human_input import HumanInputResponse
from rag.agent.core.messages import ModelMessage
from rag.agent.core.model_request import build_tool_manifest
from rag.agent.core.turn_contracts import ToolCallPlan
from rag.agent.loop.runtime import AgentLoop, LoopEventSink, ModelTurnEnvelope
from rag.agent.loop.state import LoopState, LoopTransition, ModelTurnDraft
from rag.agent.loop.stop_hooks import StopHookRunner, build_stop_hooks
from rag.agent.memory.compactor import LoopCompactionResult, LoopContextCompactor
from rag.agent.memory.models import MemoryPolicy
from rag.agent.memory.store import WorkspaceMemoryStore
from rag.agent.tools.executor import ToolExecutor
from rag.agent.tools.permissions import ToolExecutionContext
from rag.agent.tools.registry import ToolRegistry
from rag.agent.tools.tool import JsonValue, Tool, ToolEffect
from rag.agent.workspace import WorkspaceRuntime
from tests.agent.parity.fixtures import (
    PARITY_SCENARIO_NAMES,
    _definition,
    _state,
    _StructuredAnswer,
    _StructuredFinalizer,
    _tool,
)
from tests.agent.parity.normalize import normalize_loop_state


class _CandidateProvider:
    def __init__(
        self,
        *,
        direct_answer: str | None = None,
        pause_reason: str | None = None,
        pause_after_feedback: str | None = None,
        fallback: bool = False,
    ) -> None:
        self._direct_answer = direct_answer
        self._pause_reason = pause_reason
        self._pause_after_feedback = pause_after_feedback
        self._fallback = fallback

    async def next_turn(
        self,
        state: LoopState,
        *,
        definition: AgentRuntimePolicy,
        budget_remaining: int,
    ) -> ModelTurnDraft | ModelTurnEnvelope:
        del definition, budget_remaining
        if state["finish_state"].feedback and self._pause_after_feedback:
            draft = ModelTurnDraft(
                action="pause",
                pause_reason=self._pause_after_feedback,
            )
        elif self._pause_reason is not None:
            draft = ModelTurnDraft(action="pause", pause_reason=self._pause_reason)
        elif self._direct_answer is not None:
            draft = ModelTurnDraft(
                action="finish",
                final_answer=self._direct_answer,
            )
        else:
            raise AssertionError("scenario provider has no configured outcome")
        if not self._fallback:
            return draft
        return ModelTurnEnvelope(
            draft=draft,
            transitions=(
                LoopTransition(
                    reason="fallback",
                    iteration=state["iteration"],
                    detail={"provider": "fallback"},
                ),
            ),
        )


class _SequenceProvider:
    def __init__(self, turns: list[ModelTurnDraft]) -> None:
        self._turns = turns

    async def next_turn(
        self,
        state: LoopState,
        *,
        definition: AgentRuntimePolicy,
        budget_remaining: int,
    ) -> ModelTurnDraft:
        del state, definition, budget_remaining
        return self._turns.pop(0)


@dataclass
class _EventRecorder(LoopEventSink):
    reasons: list[str] = field(default_factory=list)

    async def emit(self, transition: LoopTransition) -> None:
        self.reasons.append(transition.reason)


class _NoCompaction:
    def prepare(self, state: LoopState) -> LoopCompactionResult:
        del state
        return LoopCompactionResult(changed=False)


@dataclass
class _RuntimeHarness:
    snapshot: Mapping[str, Tool]
    executor: ToolExecutor
    checkpoint_store: LangGraphCheckpointStore


def _runtime_harness(
    state: LoopState,
    registry: ToolRegistry,
) -> _RuntimeHarness:
    snapshot = registry.freeze()
    state["resident_tool_names"] = list(snapshot)
    state["tool_manifest"] = build_tool_manifest(
        tools=tuple(snapshot.values()),
        resident_tool_names=tuple(snapshot),
        explicit_tool_names=(),
        active_tool_names=(),
        provider_serializer_revision=state["provider_serializer_revision"],
    )
    return _RuntimeHarness(
        snapshot=snapshot,
        executor=ToolExecutor(snapshot),
        checkpoint_store=LangGraphCheckpointStore(
            MemorySaver(serde=agent_checkpoint_serde()),
            run_config=state["run_config"],
        ),
    )


async def _run_loop(
    *,
    definition: AgentRuntimePolicy,
    registry: ToolRegistry,
    state: LoopState,
    provider: object,
    harness: _RuntimeHarness | None = None,
    goal_spec: GoalSpec | None = None,
    output_finalizer: object | None = None,
    context_manager: object | None = None,
    memory_store: WorkspaceMemoryStore | None = None,
    event_recorder: _EventRecorder | None = None,
) -> tuple[LoopState, _RuntimeHarness]:
    if memory_store is not None:
        TurnRegistry.get(state["run_config"].turn_id).memory_store = memory_store
    runtime = harness or _runtime_harness(state, registry)
    loop = AgentLoop(
        definition=definition,
        model_provider=provider,  # type: ignore[arg-type]
        context_manager=context_manager or _NoCompaction(),  # type: ignore[arg-type]
        tool_executor=runtime.executor,
        registry_snapshot=runtime.snapshot,
        execution_context=ToolExecutionContext(
            workspace_root=Path.cwd(),
            cwd=Path.cwd(),
        ),
        checkpoint_store=runtime.checkpoint_store,
        stop_hook_runner=StopHookRunner(
            hooks=build_stop_hooks(
                definition=definition,
                output_finalizer=output_finalizer,  # type: ignore[arg-type]
                goal_spec=goal_spec,
            ),
            max_blocks=definition.max_stop_hook_blocks,
        ),
        finish_candidate_builder=FinishCandidateBuilder(),
        event_sink=event_recorder,
    )
    return await loop.run(state), runtime


def _registry(*tools: Tool) -> ToolRegistry:
    registry = ToolRegistry()
    for tool in tools:
        registry.register(tool)
    return registry


async def _plain_without_tools() -> dict[str, object]:
    result, _ = await _run_loop(
        definition=_definition("plain", ()),
        registry=_registry(),
        state=_state(
            "loop-parity-plain",
            task="Answer directly without tools.",
        ),
        provider=_CandidateProvider(direct_answer="Direct answer."),
    )
    return normalize_loop_state(result)


async def _single_tool() -> dict[str, object]:
    call = ToolCallPlan.create("answer_tool", {"text": "policy"})
    registry = _registry(
        _tool(
            "answer_tool",
            lambda payload: {"text": f"answer:{payload['text']}"},
        )
    )
    result, _ = await _run_loop(
        definition=_definition("single", ("answer_tool",)),
        registry=registry,
        state=_state(
            "loop-parity-single",
            task="Answer with one tool.",
            pending_tool_calls=(call,),
        ),
        provider=_CandidateProvider(direct_answer="answer:policy"),
    )
    return normalize_loop_state(result)


async def _multiple_tools() -> dict[str, object]:
    registry = _registry(
        _tool(
            "echo_tool",
            lambda payload: {"text": f"echo:{payload['text']}"},
        ),
        _tool(
            "calculate_tool",
            lambda payload: {
                "text": "42",
                "operation": "sum",
                "expression": payload["text"],
            },
        ),
    )
    calls = (
        ToolCallPlan.create("echo_tool", {"text": "first"}),
        ToolCallPlan.create("calculate_tool", {"text": "40 + 2"}),
    )
    result, _ = await _run_loop(
        definition=_definition("multiple", ("echo_tool", "calculate_tool")),
        registry=registry,
        state=_state(
            "loop-parity-multiple",
            task="Use multiple tools and calculate.",
            pending_tool_calls=calls,
        ),
        provider=_CandidateProvider(direct_answer="42"),
    )
    return normalize_loop_state(result)


async def _rag_grounding() -> dict[str, object]:
    answer = "Employees receive 15 days of annual leave."
    registry = _registry(
        _tool(
            "vector_search",
            lambda payload: {
                "text": "policy#leave",
                "query": payload["text"],
            },
        ),
        _tool(
            "rerank",
            lambda payload: {
                "text": "policy#leave:0.99",
                "query": payload["text"],
            },
        ),
        _tool(
            "rag_search_answer",
            lambda payload: {
                "text": answer,
                "query": payload["text"],
                "groundedness_flag": True,
            },
        ),
    )
    calls = tuple(
        ToolCallPlan.create(name, {"text": "annual leave"})
        for name in ("vector_search", "rerank", "rag_search_answer")
    )
    result, _ = await _run_loop(
        definition=_definition(
            "rag",
            ("vector_search", "rerank", "rag_search_answer"),
        ),
        registry=registry,
        state=_state(
            "loop-parity-rag",
            task="Answer the annual leave policy with citations.",
            pending_tool_calls=calls,
        ),
        provider=_CandidateProvider(direct_answer=answer),
    )
    return normalize_loop_state(result)


async def _approval_resume() -> dict[str, object]:
    calls: list[str] = []

    def write(payload: dict[str, JsonValue]) -> dict[str, JsonValue]:
        text = str(payload["text"])
        calls.append(text)
        return {"text": f"wrote:{text}"}

    registry = _registry(
        _tool(
            "write_tool",
            write,  # type: ignore[arg-type]
            effects=frozenset({ToolEffect.WRITE_WORKSPACE}),
            concurrency_safe=False,
        )
    )
    call = ToolCallPlan.create("write_tool", {"text": "approved"})
    state = _state(
        "loop-parity-approval",
        task="Run an approved mutation.",
        pending_tool_calls=(call,),
    )
    definition = _definition("approval", ("write_tool",))
    paused_state, harness = await _run_loop(
        definition=definition,
        registry=registry,
        state=state,
        provider=_CandidateProvider(direct_answer="wrote:approved"),
    )
    paused = normalize_loop_state(
        paused_state,
        observed={"runner_calls": tuple(calls)},
    )
    request = paused_state["approval_request"]
    assert request is not None
    resumed_state = await harness.checkpoint_store.apply_human_response(
        HumanInputResponse(
            request_id=request.request_id,
            decision="allow_once",
            approved_tool_call_ids=[call.tool_call_id],
        )
    )
    resumed, _ = await _run_loop(
        definition=definition,
        registry=registry,
        state=resumed_state,
        provider=_CandidateProvider(direct_answer="wrote:approved"),
        harness=harness,
    )
    return {
        "paused": paused,
        "resumed": normalize_loop_state(
            resumed,
            observed={"runner_calls": tuple(calls)},
        ),
    }


async def _tool_retry() -> dict[str, object]:
    attempts: list[str] = []

    def flaky(payload: dict[str, JsonValue]) -> dict[str, JsonValue]:
        text = str(payload["text"])
        attempts.append(text)
        if len(attempts) == 1:
            raise RuntimeError("transient failure")
        return {"text": f"recovered:{text}"}

    registry = _registry(_tool("retry_tool", flaky))  # type: ignore[arg-type]
    first = ToolCallPlan.create("retry_tool", {"text": "retry"})
    second = ToolCallPlan.create("retry_tool", {"text": "retry"})
    result, _ = await _run_loop(
        definition=_definition("retry", ("retry_tool",)),
        registry=registry,
        state=_state(
            "loop-parity-retry",
            task="Recover from a transient tool error.",
            pending_tool_calls=(first,),
        ),
        provider=_SequenceProvider(
            [
                ModelTurnDraft(action="execute", tool_calls=(second,)),
                ModelTurnDraft(
                    action="finish",
                    final_answer="recovered:retry",
                ),
            ]
        ),
    )
    return normalize_loop_state(result, observed={"attempts": tuple(attempts)})


async def _message_compaction() -> dict[str, object]:
    policy = MemoryPolicy(
        message_compaction_min_count=4,
        max_message_tail_count=2,
    )
    with TemporaryDirectory(prefix="loop-agent-parity-memory-") as directory:
        workspace = WorkspaceRuntime(
            root=Path(directory) / "workspace",
            is_temporary=True,
        )
        workspace.initialize()
        store = WorkspaceMemoryStore(workspace=workspace, policy=policy)
        state = _state(
            "loop-parity-compaction",
            task="Compact prior messages.",
            memory_policy=policy,
            messages=(
                HumanMessage(content=f"history {index}", id=f"msg-{index}")
                for index in range(6)
            ),
        )
        state["turn_transcript"] = [
            ModelMessage(
                role="user",
                content="Compact prior messages.",
            ),
            *[
                ModelMessage(
                    role="assistant" if index % 2 == 0 else "user",
                    content=(
                        f"canonical history {index}: "
                        + (f"token-{index} " * 500)
                    ),
                )
                for index in range(6)
            ],
        ]
        result, _ = await _run_loop(
            definition=_definition("compaction", ()),
            registry=_registry(),
            state=state,
            provider=_CandidateProvider(
                pause_reason="No model answer is configured."
            ),
            context_manager=LoopContextCompactor(store=store),
            memory_store=store,
        )
        return normalize_loop_state(result)


async def _structured_output() -> dict[str, object]:
    registry = _registry(
        _tool(
            "answer_tool",
            lambda payload: {"text": f"structured:{payload['text']}"},
        )
    )
    result, _ = await _run_loop(
        definition=_definition(
            "structured",
            ("answer_tool",),
            output_model=_StructuredAnswer,
        ),
        registry=registry,
        state=_state(
            "loop-parity-structured",
            task="Return a structured answer.",
            pending_tool_calls=(
                ToolCallPlan.create("answer_tool", {"text": "policy"}),
            ),
        ),
        provider=_CandidateProvider(direct_answer="structured:policy"),
        output_finalizer=_StructuredFinalizer(),
    )
    return normalize_loop_state(result)


async def _explicit_goal_spec() -> dict[str, object]:
    registry = _registry(
        _tool(
            "answer_tool",
            lambda payload: {"text": f"answer:{payload['text']}"},
        )
    )
    goal = GoalSpec(
        original_query="Answer with evidence.",
        deliverables=[
            GoalDeliverable(
                deliverable_id="answer",
                kind="answer",
                acceptance_rule="non_empty_answer",
            ),
            GoalDeliverable(
                deliverable_id="evidence",
                kind="evidence",
                acceptance_rule="traceable_evidence",
            ),
        ],
    )
    result, _ = await _run_loop(
        definition=_definition("goal", ("answer_tool",)),
        registry=registry,
        state=_state(
            "loop-parity-goal",
            task="Answer with evidence.",
            pending_tool_calls=(
                ToolCallPlan.create(
                    "answer_tool",
                    {"text": "without evidence"},
                ),
            ),
        ),
        provider=_CandidateProvider(
            direct_answer="answer:without evidence",
            pause_after_feedback=(
                "Explicit goal contract still requires traceable evidence."
            ),
        ),
        goal_spec=goal,
    )
    return normalize_loop_state(result)


async def _model_fallback() -> dict[str, object]:
    events = _EventRecorder()
    result, _ = await _run_loop(
        definition=_definition("model_fallback", ()),
        registry=_registry(),
        state=_state(
            "loop-parity-model-fallback",
            task="Use the fallback model.",
        ),
        provider=_CandidateProvider(
            pause_reason="Fallback model needs clarification.",
            fallback=True,
        ),
        event_recorder=events,
    )
    return normalize_loop_state(
        result,
        observed={"transition_reasons": tuple(events.reasons)},
    )


_SCENARIOS = {
    "approval_resume": _approval_resume,
    "explicit_goal_spec": _explicit_goal_spec,
    "message_compaction": _message_compaction,
    "model_fallback": _model_fallback,
    "multiple_tools": _multiple_tools,
    "plain_without_tools": _plain_without_tools,
    "rag_grounding": _rag_grounding,
    "single_tool": _single_tool,
    "structured_output": _structured_output,
    "tool_retry": _tool_retry,
}


async def run_loop_scenarios() -> dict[str, object]:
    return {name: await _SCENARIOS[name]() for name in PARITY_SCENARIO_NAMES}
