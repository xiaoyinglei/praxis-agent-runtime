from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal, Self
from uuid import uuid4

from pydantic import BaseModel, Field, model_validator

PlanStepStatus = Literal["pending", "in_progress", "completed", "blocked", "skipped"]
PlanStatus = Literal["active", "complete", "blocked", "needs_replan"]
PlanEventType = Literal[
    "initialized",
    "llm_update",
    "decision_progress",
    "observation_progress",
    "needs_replan",
    "completed",
    "blocked",
]

MAX_PLAN_STEPS = 12
MAX_PLAN_EVENTS = 30
MAX_STEP_TITLE_CHARS = 180
MAX_PLAN_SUMMARY_CHARS = 800
MAX_OBJECTIVE_CHARS = 500
MAX_STEP_REFS = 16


class PlanStep(BaseModel):
    """A bounded todo item for the current agent run.

    Plan steps are planning state only. They do not authorize tools or bypass
    executor policy.
    """

    step_id: str
    title: str
    status: PlanStepStatus = "pending"

    @property
    def key(self) -> str:
        return self.step_id

    @model_validator(mode="after")
    def bound_fields(self) -> Self:
        self.step_id = _safe_identifier(self.step_id, prefix="step")
        self.title = _bounded_text(self.title, MAX_STEP_TITLE_CHARS) or "Untitled step"
        return self



class AgentPlan(BaseModel):
    objective: str
    status: PlanStatus = "active"
    revision: int = Field(default=0, ge=0)
    active_step_id: str | None = None
    steps: list[PlanStep] = Field(default_factory=list)
    summary: str | None = None

    @model_validator(mode="after")
    def bound_fields(self) -> Self:
        self.objective = _bounded_text(self.objective, MAX_OBJECTIVE_CHARS) or "Current task"
        self.steps = self.steps[:MAX_PLAN_STEPS]
        if self.summary is not None:
            self.summary = _bounded_text(self.summary, MAX_PLAN_SUMMARY_CHARS) or None
        self.active_step_id = _valid_active_step_id(self.active_step_id, self.steps)
        return self



class PlanEvent(BaseModel):
    event_id: str
    event_type: PlanEventType
    plan_revision: int = Field(ge=0)
    message: str
    related_step_id: str | None = None
    tool_call_ids: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)

    @property
    def key(self) -> str:
        return self.event_id

    @model_validator(mode="after")
    def bound_fields(self) -> Self:
        self.event_id = _safe_identifier(self.event_id, prefix="plan_event")
        self.message = _bounded_text(self.message, MAX_PLAN_SUMMARY_CHARS) or self.event_type
        if self.related_step_id is not None:
            self.related_step_id = _safe_identifier(self.related_step_id, prefix="step")
        self.tool_call_ids = _dedupe_texts(self.tool_call_ids, limit=MAX_STEP_REFS)
        self.warnings = _dedupe_texts(self.warnings, limit=MAX_STEP_REFS)
        return self


@dataclass(slots=True)
class PlanTracker:
    max_steps: int = MAX_PLAN_STEPS

    def initialize_task(
        self,
        *,
        task: str,
    ) -> tuple[AgentPlan, list[PlanEvent]]:
        """Create an advisory task plan without deriving completion gaps."""

        step = PlanStep(
            step_id="step_task",
            title="Work on the current task.",
        )
        plan = AgentPlan(
            objective=task,
            status="active",
            active_step_id=step.step_id,
            steps=[step],
        )
        return plan, [
            self._event(
                "initialized",
                plan,
                message="Initialized advisory task plan.",
            )
        ]


    def replace_from_tool(
        self,
        plan: AgentPlan,
        *,
        steps: Sequence[PlanStep],
        summary: str | None,
    ) -> tuple[AgentPlan, list[PlanEvent]]:
        """Persist the complete visible plan submitted through update_plan."""

        warnings: list[str] = []
        if len(steps) > self.max_steps:
            warnings.append("steps_truncated")
        bounded_steps: list[PlanStep] = []
        for step in steps[: self.max_steps]:
            if step.status == "completed":
                warnings.append("unverified_completion_ignored")
                bounded_steps.append(
                    step.model_copy(update={"status": "pending"})
                )
                continue
            bounded_steps.append(step)
        active_step_id = next(
            (
                step.step_id
                for status in ("in_progress", "pending")
                for step in bounded_steps
                if step.status == status
            ),
            None,
        )
        updated = AgentPlan(
            objective=plan.objective,
            status=(
                "complete"
                if bounded_steps
                and all(step.status == "completed" for step in bounded_steps)
                else "active"
            ),
            revision=plan.revision + 1,
            active_step_id=active_step_id,
            steps=bounded_steps,
            summary=summary,
        )
        return updated, [
            self._event(
                "llm_update",
                updated,
                message="Applied update_plan tool update.",
                related_step_id=updated.active_step_id,
                warnings=warnings,
            )
        ]


    def record_completion(
        self,
        plan: AgentPlan | None,
        *,
        blocked: bool = False,
    ) -> tuple[AgentPlan | None, list[PlanEvent]]:
        if plan is None:
            return None, []
        target_status: PlanStatus = "blocked" if blocked else "complete"
        if plan.status == target_status:
            return plan, []
        step_status: PlanStepStatus = "blocked" if blocked else "completed"
        steps: list[PlanStep] = []
        for step in plan.steps:
            should_update = step.status in {"pending", "in_progress"}
            steps.append(
                step.model_copy(update={"status": step_status})
                if should_update
                else step
            )
        updated = plan.model_copy(
            update={
                "status": target_status,
                "revision": plan.revision + 1,
                "steps": steps,
            }
        )
        event_type: PlanEventType = "blocked" if blocked else "completed"
        return updated, [
            self._event(
                event_type,
                updated,
                message="Plan blocked." if blocked else "Plan completed.",
                related_step_id=updated.active_step_id,
            )
        ]


    @staticmethod
    def _event(
        event_type: PlanEventType,
        plan: AgentPlan,
        *,
        message: str,
        related_step_id: str | None = None,
        tool_call_ids: Sequence[str] = (),
        warnings: Sequence[str] = (),
    ) -> PlanEvent:
        return PlanEvent(
            event_id=f"plan_event_{uuid4().hex[:12]}",
            event_type=event_type,
            plan_revision=plan.revision,
            message=message,
            related_step_id=related_step_id,
            tool_call_ids=list(tool_call_ids),
            warnings=list(warnings),
        )

def _valid_active_step_id(active_step_id: str | None, steps: Sequence[PlanStep]) -> str | None:
    step_ids = {step.step_id for step in steps}
    if active_step_id is not None:
        safe = _safe_identifier(active_step_id, prefix="step")
        if safe in step_ids:
            return safe
    for status in ("in_progress", "pending"):
        for step in steps:
            if step.status == status:
                return step.step_id
    return steps[0].step_id if steps else None
def _safe_identifier(value: str, *, prefix: str) -> str:
    cleaned = "".join(char if char.isalnum() or char in {"_", "-"} else "_" for char in value.strip())
    cleaned = "_".join(part for part in cleaned.split("_") if part)
    if not cleaned:
        return f"{prefix}_unknown"
    if not cleaned.startswith(f"{prefix}_"):
        cleaned = f"{prefix}_{cleaned}"
    return cleaned[:80]


def _bounded_text(value: str, limit: int) -> str:
    normalized = " ".join(str(value).split())
    if len(normalized) <= limit:
        return normalized
    return normalized[:limit].rstrip()


def _dedupe_texts(values: Sequence[str], *, limit: int) -> list[str]:
    deduped: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = _bounded_text(str(value), MAX_PLAN_SUMMARY_CHARS)
        if not text or text in seen:
            continue
        seen.add(text)
        deduped.append(text)
        if len(deduped) >= limit:
            break
    return deduped


__all__ = [
    "AgentPlan",
    "MAX_PLAN_EVENTS",
    "MAX_PLAN_STEPS",
    "PlanTracker",
    "PlanEvent",
    "PlanStep",
    "PlanStatus",
    "PlanStepStatus",
]
