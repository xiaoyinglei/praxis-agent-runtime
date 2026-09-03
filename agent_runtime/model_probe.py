"""Ephemeral provider capability probes using the production model path."""

from __future__ import annotations

import asyncio
import os
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from enum import StrEnum
from typing import Literal
from uuid import uuid4

from pydantic import BaseModel

from agent_runtime.core.llm_registry import ModelRegistry, ResolvedModel
from agent_runtime.core.model_request import (
    ModelRequest,
    ModelSettings,
    ToolChoice,
    build_model_request,
    build_stable_context,
)
from agent_runtime.model_definition import ModelExecutionDefinition
from agent_runtime.modeling.contracts import LLMCallStage
from agent_runtime.modeling.gateway import LLMGateway, ProviderDelta, ProviderDeltaChannel
from agent_runtime.tools.tool import (
    CancellationMode,
    InterruptBehavior,
    JsonValue,
    NormalizedToolOutput,
    ResolvedToolUse,
    Tool,
    ToolDefinition,
    json_schema_input,
)


class ProbeLevel(StrEnum):
    CONNECTIVITY = "connectivity"
    STREAM = "stream"
    FULL = "full"


@dataclass(frozen=True, slots=True)
class ModelProbeEvidence:
    level: ProbeLevel
    connectivity_ok: bool
    text_delta_count: int
    completion_ok: bool
    tool_call_ok: bool | None
    structured_output_ok: bool | None


class ModelProbeError(RuntimeError):
    def __init__(self, *, phase: str, detail: str) -> None:
        self.phase = phase
        super().__init__(f"Model probe {phase} failed: {detail}")


class _StructuredProbe(BaseModel):
    ok: Literal[True]


class ModelProbe:
    """Probe one frozen model definition without persisting evidence or running tools."""

    def __init__(self, registry: ModelRegistry) -> None:
        self._registry = registry

    async def run(
        self,
        definition: ModelExecutionDefinition,
        *,
        level: ProbeLevel,
    ) -> ModelProbeEvidence:
        if not isinstance(level, ProbeLevel):
            raise TypeError("level must be a ProbeLevel")
        credential_name = definition.api_key_env
        if credential_name is not None:
            credential = os.environ.get(credential_name)
            if not isinstance(credential, str) or not credential.strip():
                raise ModelProbeError(
                    phase="authentication",
                    detail=f"credential environment {credential_name} is not set",
                )
        resolved = await _safe_phase(
            "resolve",
            lambda: asyncio.to_thread(self._registry.resolve_definition, definition),
            definition=definition,
        )
        await self._check_connectivity(resolved, definition)
        if level is ProbeLevel.CONNECTIVITY:
            return ModelProbeEvidence(level, True, 0, False, None, None)

        text_delta_count = await self._check_stream(resolved, definition)
        if level is ProbeLevel.STREAM:
            return ModelProbeEvidence(level, True, text_delta_count, True, None, None)

        tool_call_ok: bool | None = None
        if definition.supports_tools:
            await self._check_tool_call(resolved, definition)
            tool_call_ok = True
        structured_output_ok: bool | None = None
        if definition.supports_structured_output:
            await self._check_structured_output(resolved, definition)
            structured_output_ok = True
        return ModelProbeEvidence(
            level,
            True,
            text_delta_count,
            True,
            tool_call_ok,
            structured_output_ok,
        )

    async def _check_connectivity(
        self,
        resolved: ResolvedModel,
        definition: ModelExecutionDefinition,
    ) -> None:
        list_models = getattr(resolved.generator, "list_models", None)
        if not callable(list_models):
            raise ModelProbeError(
                phase="connectivity",
                detail="provider adapter does not expose model discovery",
            )
        model_ids = await _safe_phase(
            "connectivity",
            lambda: asyncio.to_thread(list_models),
            definition=definition,
        )
        if definition.model not in model_ids:
            raise ModelProbeError(
                phase="model_identity",
                detail=f"configured model {definition.model!r} was not advertised",
            )

    async def _check_stream(
        self,
        resolved: ResolvedModel,
        definition: ModelExecutionDefinition,
    ) -> int:
        delta_count = 0

        def record_delta(delta: ProviderDelta) -> None:
            nonlocal delta_count
            if delta.channel is ProviderDeltaChannel.TEXT and delta.content:
                delta_count += 1

        request = _probe_request(definition=definition, tools=(), tool_choice=ToolChoice.none())
        gateway = _gateway(resolved)
        response = await _safe_phase(
            "stream",
            lambda: gateway.agenerate_model_request(
                stage=LLMCallStage.AGENT_STEP,
                request=request,
                provider=resolved.provider,
                supports_native_tools=resolved.supports_native_tools,
                stream=True,
                delta_sink=record_delta,
            ),
            definition=definition,
        )
        if not response.turn.text or delta_count == 0:
            raise ModelProbeError(
                phase="stream",
                detail="provider returned no text delta",
            )
        return delta_count

    async def _check_tool_call(
        self,
        resolved: ResolvedModel,
        definition: ModelExecutionDefinition,
    ) -> None:
        tool = _probe_tool()
        request = _probe_request(
            definition=definition,
            tools=(tool,),
            tool_choice=ToolChoice.named(tool.definition.name),
        )
        gateway = _gateway(resolved)
        response = await _safe_phase(
            "tool_call",
            lambda: gateway.agenerate_model_request(
                stage=LLMCallStage.AGENT_STEP,
                request=request,
                provider=resolved.provider,
                supports_native_tools=resolved.supports_native_tools,
                stream=True,
            ),
            definition=definition,
        )
        calls = response.turn.tool_calls
        if len(calls) != 1 or calls[0].name != tool.definition.name:
            raise ModelProbeError(
                phase="tool_call",
                detail="provider did not return the forced probe tool call",
            )
        await _safe_phase(
            "tool_call",
            lambda: asyncio.to_thread(tool.validate_input, calls[0].input),
            definition=definition,
        )

    async def _check_structured_output(
        self,
        resolved: ResolvedModel,
        definition: ModelExecutionDefinition,
    ) -> None:
        gateway = _gateway(resolved)
        result = await _safe_phase(
            "structured_output",
            lambda: gateway.agenerate_structured(
                stage=LLMCallStage.AGENT_STEP,
                prompt="Return an object whose ok field is true.",
                schema=_StructuredProbe,
                kwargs=resolved.kwargs,
            ),
            definition=definition,
        )
        if result.value.ok is not True:
            raise ModelProbeError(
                phase="structured_output",
                detail="provider returned an invalid structured probe result",
            )


async def _safe_phase[T](
    phase: str,
    operation: Callable[[], Awaitable[T]],
    *,
    definition: ModelExecutionDefinition,
) -> T:
    failure: ModelProbeError | None = None
    try:
        return await operation()
    except asyncio.CancelledError:
        raise
    except Exception as error:
        failure = ModelProbeError(
            phase=phase,
            detail=(
                f"provider error {type(error).__name__}; check endpoint, model, "
                f"and credential environment {definition.api_key_env or '<none>'}"
            ),
        )
    assert failure is not None
    raise failure


def _probe_request(
    *,
    definition: ModelExecutionDefinition,
    tools: tuple[Tool, ...],
    tool_choice: ToolChoice,
) -> ModelRequest:
    defaults = definition.defaults
    provider_options = (
        defaults.provider_options.model_dump(mode="python", exclude_none=True)
        if defaults.provider_options is not None
        else {}
    )
    return build_model_request(
        request_id=f"model-probe-{uuid4().hex}",
        context=build_stable_context(
            instructions=("You are a provider capability probe.",),
            initial_user_task="Reply with the text probe.",
        ),
        selected_tools=tools,
        settings=ModelSettings(
            model=definition.model,
            max_output_tokens=min(definition.max_output_tokens, 32),
            temperature=defaults.temperature or 0.0,
            top_p=defaults.top_p if defaults.top_p is not None else 1.0,
            parallel_tool_calls=(
                defaults.parallel_tool_calls
                if defaults.parallel_tool_calls is not None
                else False
            ),
            seed=defaults.seed,
            provider_options=provider_options,
        ),
        tool_choice=tool_choice,
    )


def _probe_tool() -> Tool:
    schema: dict[str, JsonValue] = {
        "type": "object",
        "properties": {"ok": {"const": True}},
        "required": ("ok",),
        "additionalProperties": False,
    }

    def must_not_execute(_arguments: object) -> object:
        raise AssertionError("model capability probe tool must never execute")

    return Tool(
        definition=ToolDefinition(
            name="probe_capability",
            description="Return ok=true to prove native tool-call support.",
            input_schema=schema,
        ),
        validate_input=json_schema_input(schema),
        run=must_not_execute,
        normalize_output=lambda _raw: NormalizedToolOutput(),
        output_schema=None,
        static_effects=frozenset(),
        resolve_use=lambda _arguments: ResolvedToolUse(effects=frozenset(), targets=()),
        execution_revision="model-probe-v1",
        idempotent=True,
        concurrency_safe=True,
        cancellation_mode=CancellationMode.COOPERATIVE,
        interrupt_behavior=InterruptBehavior.CANCEL,
        timeout_seconds=1.0,
        max_model_output_bytes=1_024,
    )


def _gateway(resolved: ResolvedModel) -> LLMGateway:
    gateway = resolved.gateway
    if not isinstance(gateway, LLMGateway):
        raise ModelProbeError(
            phase="resolve",
            detail="resolved model has no production gateway",
        )
    return gateway


__all__ = [
    "ModelProbe",
    "ModelProbeError",
    "ModelProbeEvidence",
    "ProbeLevel",
]
