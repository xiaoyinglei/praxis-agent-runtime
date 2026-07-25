from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping
from typing import Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
)

from rag.agent.tools.tool import (
    CancellationMode,
    InterruptBehavior,
    JsonValue,
    NormalizedToolOutput,
    ResolvedToolUse,
    Tool,
    ToolDefinition,
    json_schema_output,
    pydantic_input,
)

type PlanUpdater = Callable[
    [Mapping[str, JsonValue]],
    object | Awaitable[object],
]


class PlanStepInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    step_id: str | None = Field(
        default=None,
        min_length=1,
        max_length=80,
        description=(
            "Stable step identifier from the visible plan. Omit it for a new "
            "step; the runtime will assign one."
        ),
    )
    step: str = Field(
        min_length=1,
        max_length=180,
        description="A concise, verifiable implementation step.",
    )
    status: Literal["pending", "in_progress", "completed"] = Field(
        description="Current state of this exact step."
    )


class UpdatePlanInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    plan: list[PlanStepInput] = Field(
        min_length=1,
        max_length=20,
        description=(
            "The complete ordered strategy for the immutable goal."
        ),
    )
    explanation: str | None = Field(
        default=None,
        max_length=800,
        description="A short reason for the plan transition.",
    )

class UpdatePlanOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    accepted: bool
    revision: int = Field(ge=0)
    message: str = Field(min_length=1, max_length=800)
    authority: Literal["advisory"] = "advisory"


_UPDATE_INPUT_SCHEMA, _validate_update_input = pydantic_input(UpdatePlanInput)
_UPDATE_OUTPUT_SCHEMA, _unused_update_output_validator = pydantic_input(
    UpdatePlanOutput
)


def create_update_plan_tool(plan_updater: PlanUpdater) -> Tool:
    if not callable(plan_updater):
        raise TypeError("plan_updater must be callable")
    return Tool(
        definition=ToolDefinition(
            name="update_plan",
            description=(
                "Replace the visible advisory strategy for the current task. "
                'Every plan item requires only "step" and "status". The plan '
                "explains intended work; it never grants or restricts tool "
                "permission and never proves that the task is complete."
            ),
            input_schema=_UPDATE_INPUT_SCHEMA,
        ),
        validate_input=_validate_update_input,
        run=plan_updater,
        normalize_output=_normalize_update_output,
        output_schema=_UPDATE_OUTPUT_SCHEMA,
        static_effects=frozenset(),
        resolve_use=lambda _arguments: ResolvedToolUse(
            effects=frozenset(),
            targets=(),
        ),
        execution_revision="builtin-update-plan-v1",
        idempotent=True,
        concurrency_safe=False,
        cancellation_mode=CancellationMode.COOPERATIVE,
        interrupt_behavior=InterruptBehavior.FINISH_CURRENT,
        timeout_seconds=5.0,
        max_model_output_bytes=50_000,
    )


def _normalize_update_output(raw: object) -> NormalizedToolOutput:
    validated = UpdatePlanOutput.model_validate(raw)
    structured = json_schema_output(
        _UPDATE_OUTPUT_SCHEMA,
        validated.model_dump(mode="json"),
    )
    return NormalizedToolOutput(structured_content=structured)


__all__ = [
    "PlanStepInput",
    "PlanUpdater",
    "UpdatePlanInput",
    "UpdatePlanOutput",
    "create_update_plan_tool",
]
