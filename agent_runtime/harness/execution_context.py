"""Immutable Turn and Step request views shared by execution components."""

from __future__ import annotations

import copy
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

from agent_runtime.harness.control import StepControl
from agent_runtime.harness.protocol import HarnessModelRequest, ModelMessage
from agent_runtime.tools.tool import Tool


@dataclass(frozen=True, slots=True)
class TurnContext:
    """Identity and initial settings; each new Step captures its own binding."""

    thread_id: str
    turn_id: str
    binding_manifest: Mapping[str, Any]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "binding_manifest",
            MappingProxyType(dict(self.binding_manifest)),
        )


@dataclass(frozen=True, slots=True)
class StepContext:
    """One immutable model-request view captured inside a Turn."""

    turn: TurnContext
    step: int
    binding_manifest: Mapping[str, Any]
    messages: tuple[ModelMessage, ...]
    tools: tuple[Tool, ...]
    model_token_budget_remaining: int | None
    input_revision: int
    budget_pressure: bool = False
    purpose: str = "agent_step"
    request_id: str | None = None
    input_token_limit: int | None = None
    output_token_limit: int | None = None
    compaction_plan: Mapping[str, Any] | None = None
    continuation_summary_role: str | None = None
    control: StepControl = field(default_factory=StepControl)
    frozen_request: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "binding_manifest", MappingProxyType(copy.deepcopy(dict(self.binding_manifest))))

    def model_request(self) -> HarnessModelRequest:
        return HarnessModelRequest(
            thread_id=self.turn.thread_id,
            turn_id=self.turn.turn_id,
            messages=self.messages,
            binding_manifest=self.binding_manifest,
            tools=self.tools,
            step=self.step,
            model_token_budget_remaining=self.model_token_budget_remaining,
            budget_pressure=self.budget_pressure,
            purpose=self.purpose,
            request_id=self.request_id,
            input_token_limit=self.input_token_limit,
            output_token_limit=self.output_token_limit,
            continuation_summary_role=self.continuation_summary_role,
            control=self.control,
        )
