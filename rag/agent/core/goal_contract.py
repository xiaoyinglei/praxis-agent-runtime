from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from rag.agent.core.observations import (
    ComputationResult,
    ContextBinding,
    EvidenceRef,
)

DeliverableKind = Literal["answer", "evidence", "computation"]
AcceptanceRule = Literal[
    "non_empty_answer",
    "traceable_evidence",
    "reproducible_computation",
]


class GoalConstraint(BaseModel):
    model_config = ConfigDict(frozen=True)

    constraint_id: str = Field(min_length=1, max_length=200)
    constraint_type: str = Field(min_length=1, max_length=100)
    expected_value: object
    required: bool = True
    metadata: dict[str, object] = Field(default_factory=dict)


class GoalDeliverable(BaseModel):
    """One explicitly requested output and its deterministic acceptance rule."""

    model_config = ConfigDict(frozen=True)

    deliverable_id: str = Field(min_length=1, max_length=200)
    kind: DeliverableKind
    acceptance_rule: AcceptanceRule
    required: bool = True
    metadata: dict[str, object] = Field(default_factory=dict)

    @property
    def key(self) -> str:
        return self.deliverable_id

    @model_validator(mode="after")
    def validate_acceptance_rule(self) -> Self:
        expected_rules: dict[DeliverableKind, AcceptanceRule] = {
            "answer": "non_empty_answer",
            "evidence": "traceable_evidence",
            "computation": "reproducible_computation",
        }
        expected = expected_rules[self.kind]
        if self.acceptance_rule != expected:
            raise ValueError(
                f"acceptance_rule {self.acceptance_rule!r} is invalid for "
                f"{self.kind!r}; expected {expected!r}"
            )
        return self


class GoalSpec(BaseModel):
    """Immutable user-goal contract owned by runtime and completion checks."""

    model_config = ConfigDict(frozen=True)

    original_query: str
    deliverables: list[GoalDeliverable] = Field(default_factory=list)
    constraints: list[GoalConstraint] = Field(default_factory=list)
    success_criteria: list[str] = Field(default_factory=list)
    # Accepted only to preserve callers that persisted the former public schema.
    required_outputs: list[str] = Field(default_factory=lambda: ["answer"])
    required_evidence: list[str] = Field(default_factory=list)
    required_operations: list[str] = Field(default_factory=list)

    @property
    def fingerprint(self) -> str:
        canonical = json.dumps(
            self.model_dump(mode="json"),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    @model_validator(mode="after")
    def normalize_deliverables(self) -> Self:
        deliverables = list(self.deliverables)
        if not deliverables:
            if self.required_outputs:
                deliverables.append(_deliverable("answer"))
            if self.required_evidence:
                deliverables.append(_deliverable("evidence"))
            if self.required_operations:
                deliverables.append(_deliverable("computation"))
            object.__setattr__(self, "deliverables", deliverables)
        ids = [item.deliverable_id for item in deliverables]
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate deliverable_id values are not allowed")
        constraint_ids = [
            constraint.constraint_id for constraint in self.constraints
        ]
        if len(constraint_ids) != len(set(constraint_ids)):
            raise ValueError("duplicate constraint_id values are not allowed")
        return self


class GoalCompatibilityConfig(BaseModel):
    """Persisted goal contract kept outside the canonical LoopState schema."""

    model_config = ConfigDict(frozen=True)

    goal_spec: GoalSpec | None = None


class GoalContractIssue(BaseModel):
    model_config = ConfigDict(frozen=True)

    issue_id: str
    description: str
    kind: Literal["deliverable", "constraint"]


class GoalContractEvaluation(BaseModel):
    model_config = ConfigDict(frozen=True)

    issues: tuple[GoalContractIssue, ...] = ()

    @property
    def satisfied(self) -> bool:
        return not self.issues

    @property
    def issue_ids(self) -> list[str]:
        return [issue.issue_id for issue in self.issues]


class GoalContractEvaluator:
    """Evaluate an explicit contract without participating in loop routing."""

    def evaluate(
        self,
        *,
        goal_spec: GoalSpec,
        candidate: str,
        evidence_refs: Sequence[EvidenceRef],
        computation_results: Sequence[ComputationResult],
        context_bindings: Sequence[ContextBinding],
    ) -> GoalContractEvaluation:
        issues: list[GoalContractIssue] = []
        for constraint in goal_spec.constraints:
            if not constraint.required:
                continue
            if any(
                binding.constraint_id == constraint.constraint_id
                and binding.status == "satisfied"
                for binding in context_bindings
            ):
                continue
            issues.append(
                GoalContractIssue(
                    issue_id=f"constraint:{constraint.constraint_id}",
                    description=_constraint_issue_description(constraint),
                    kind="constraint",
                )
            )

        for deliverable in goal_spec.deliverables:
            if not deliverable.required or _deliverable_is_satisfied(
                deliverable,
                candidate=candidate,
                evidence_refs=evidence_refs,
                computation_results=computation_results,
            ):
                continue
            issues.append(
                GoalContractIssue(
                    issue_id=deliverable.deliverable_id,
                    description=_issue_description(deliverable.kind),
                    kind="deliverable",
                )
            )
        return GoalContractEvaluation(issues=tuple(issues))


def _deliverable(
    kind: DeliverableKind,
) -> GoalDeliverable:
    rules: dict[DeliverableKind, AcceptanceRule] = {
        "answer": "non_empty_answer",
        "evidence": "traceable_evidence",
        "computation": "reproducible_computation",
    }
    return GoalDeliverable(
        deliverable_id=kind,
        kind=kind,
        acceptance_rule=rules[kind],
    )


def _deliverable_is_satisfied(
    deliverable: GoalDeliverable,
    *,
    candidate: str,
    evidence_refs: Sequence[EvidenceRef],
    computation_results: Sequence[ComputationResult],
) -> bool:
    if deliverable.acceptance_rule == "non_empty_answer":
        return bool(candidate.strip())
    if deliverable.acceptance_rule == "traceable_evidence":
        return any(_is_traceable(ref) for ref in evidence_refs)
    return any(
        bool(result.expression and result.expression.strip())
        and any(_is_traceable(ref) for ref in result.evidence_refs)
        for result in computation_results
    )


def _is_traceable(ref: EvidenceRef) -> bool:
    if ref.citation_id and ref.citation_id.strip():
        return True
    if ref.citation_anchor and ref.citation_anchor.strip():
        return True
    return bool(
        ref.source == "asset"
        and ref.evidence_id
        and ref.evidence_id.startswith("asset:")
        and ref.evidence_id.removeprefix("asset:").isdigit()
    )


def _issue_description(kind: DeliverableKind) -> str:
    return {
        "answer": "Produce a non-empty answer.",
        "evidence": "Provide traceable evidence.",
        "computation": (
            "Provide a reproducible computation with traceable evidence."
        ),
    }[kind]


def _constraint_issue_description(constraint: GoalConstraint) -> str:
    if (
        constraint.constraint_type == "workspace_change"
        and constraint.expected_value is True
    ):
        return (
            "Make a real workspace change with a write tool before finishing; "
            "a prose-only answer does not complete this task."
        )
    if (
        constraint.constraint_type == "verification_after_change"
        and constraint.expected_value is True
    ):
        return (
            "Run a recognized test, lint, type-check, or build command after the "
            "latest workspace change and keep every such verification green."
        )
    return (
        "Bind context satisfying the required constraint "
        f"{constraint.constraint_id!r}."
    )


__all__ = [
    "AcceptanceRule",
    "DeliverableKind",
    "GoalCompatibilityConfig",
    "GoalConstraint",
    "GoalContractEvaluation",
    "GoalContractEvaluator",
    "GoalContractIssue",
    "GoalDeliverable",
    "GoalSpec",
]
