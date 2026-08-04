from __future__ import annotations

from collections.abc import Awaitable, Callable, Iterable, Mapping

from langchain_core.messages import BaseMessage
from pydantic import BaseModel

from agent_runtime.core.context import AgentRunConfig, TurnRegistry
from agent_runtime.core.definition import AgentRuntimePolicy
from agent_runtime.core.turn_contracts import ToolCallPlan
from agent_runtime.loop.state import LoopState, create_loop_state
from agent_runtime.memory.models import MemoryPolicy
from agent_runtime.tools.tool import (
    CancellationMode,
    InterruptBehavior,
    JsonValue,
    NormalizedToolOutput,
    ResolvedToolUse,
    Tool,
    ToolContentBlock,
    ToolDefinition,
    ToolEffect,
    ToolTarget,
    json_schema_input,
)

PARITY_SCENARIO_NAMES = (
    "approval_resume",
    "explicit_goal_spec",
    "message_compaction",
    "model_fallback",
    "multiple_tools",
    "plain_without_tools",
    "rag_grounding",
    "single_tool",
    "structured_output",
    "tool_retry",
)

_TEXT_SCHEMA: Mapping[str, JsonValue] = {
    "type": "object",
    "properties": {"text": {"type": "string"}},
    "required": ["text"],
    "additionalProperties": False,
}


class _StructuredAnswer(BaseModel):
    answer: str
    confidence: float


class _StructuredFinalizer:
    def finalize(
        self,
        *,
        definition: AgentRuntimePolicy,
        state: LoopState,
        candidate_text: str,
    ) -> _StructuredAnswer:
        del definition, state
        return _StructuredAnswer(answer=candidate_text, confidence=0.91)


def _config(
    run_id: str,
    *,
    memory_policy: MemoryPolicy | None = None,
) -> AgentRunConfig:
    config = AgentRunConfig(
        turn_id=run_id,
        llm_budget_total=20_000,
        memory_policy=memory_policy or MemoryPolicy(),
    )
    TurnRegistry.remove(run_id)
    TurnRegistry.get_or_create(config)
    return config


def _state(
    run_id: str,
    *,
    task: str,
    pending_tool_calls: Iterable[ToolCallPlan] = (),
    memory_policy: MemoryPolicy | None = None,
    messages: Iterable[BaseMessage] = (),
) -> LoopState:
    return create_loop_state(
        current_message=task,
        run_config=_config(run_id, memory_policy=memory_policy),
        pending_tool_calls=pending_tool_calls,
        messages=messages,
    )


def _definition(
    name: str,
    allowed_tools: Iterable[str],
    *,
    output_model: type[BaseModel] | None = None,
) -> AgentRuntimePolicy:
    return AgentRuntimePolicy.test_factory(
        system_prompt="Use canonical tools and return the requested result.",
        allowed_tools=list(allowed_tools),
        output_model=output_model,
        max_iterations=6,
    )


def _tool(
    name: str,
    runner: Callable[
        [Mapping[str, JsonValue]],
        object | Awaitable[object],
    ],
    *,
    effects: frozenset[ToolEffect] = frozenset(),
    concurrency_safe: bool = True,
) -> Tool:
    def normalize(raw: object) -> NormalizedToolOutput:
        if isinstance(raw, Mapping):
            structured = dict(raw)
            text = str(structured.get("text", structured))
        else:
            text = str(raw)
            structured = {"text": text}
        return NormalizedToolOutput(
            content=(ToolContentBlock(type="text", data={"text": text}),),
            structured_content=structured,
        )

    targets = (ToolTarget(kind="workspace_path", value="."),) if ToolEffect.WRITE_WORKSPACE in effects else ()
    return Tool(
        definition=ToolDefinition(
            name=name,
            description=f"Use {name} in the parity scenario.",
            input_schema=_TEXT_SCHEMA,
        ),
        validate_input=json_schema_input(_TEXT_SCHEMA),
        run=runner,
        normalize_output=normalize,
        output_schema=None,
        static_effects=effects,
        resolve_use=lambda _arguments: ResolvedToolUse(
            effects=frozenset(),
            targets=targets,
        ),
        execution_revision=f"{name}-v1",
        idempotent=True,
        concurrency_safe=concurrency_safe,
        cancellation_mode=CancellationMode.COOPERATIVE,
        interrupt_behavior=InterruptBehavior.CANCEL,
        timeout_seconds=1.0,
        max_model_output_bytes=4096,
    )
