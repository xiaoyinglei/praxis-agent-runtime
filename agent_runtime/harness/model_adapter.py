"""Adapter from retained Praxis model assets to the Harness model protocol."""

from __future__ import annotations

import asyncio
import hashlib
import inspect
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, cast

from agent_runtime.budget import ResourceUsage
from agent_runtime.core.llm_registry import ResolvedModel
from agent_runtime.core.messages import ModelMessage, StopReason
from agent_runtime.core.messages import ToolCall as ModelToolCall
from agent_runtime.core.model_request import (
    ContextBlock,
    ModelRequest,
    ModelSettings,
    StableModelContext,
    build_model_request,
    build_stable_context,
    canonical_model_request_json,
)
from agent_runtime.harness.protocol import (
    HarnessMessage,
    HarnessModelDelta,
    HarnessModelDeltaSink,
    HarnessModelRequest,
    HarnessModelResponse,
    HarnessToolCall,
    ModelDispatchCancelledError,
    ModelDispatchOutcomeUnknownError,
    ModelDispatchPreflightError,
    PreparedModelCall,
)
from agent_runtime.modeling.budget import LLMBudgetLedger
from agent_runtime.modeling.contracts import LLMCallStage
from agent_runtime.modeling.gateway import (
    LLMBudgetExceededError,
    LLMContextOverflowError,
    ProviderDelta,
    model_request_input_text,
)
from agent_runtime.modeling.local_agent_wire import render_local_agent_request
from agent_runtime.modeling.openai_wire import serialize_openai_request
from agent_runtime.models import ModelControlPlane
from agent_runtime.tools.tool import JsonValue


@dataclass(frozen=True, slots=True)
class _GatewayDispatch:
    request: ModelRequest
    wire_hash: str
    resolved: ResolvedModel
    model_token_budget_remaining: int | None


class GatewayHarnessModel:
    """Prepare canonical bytes synchronously; perform provider I/O only in dispatch."""

    def __init__(
        self,
        *,
        model_alias: str,
        resolved: ResolvedModel,
        instructions: tuple[str, ...],
    ) -> None:
        if not model_alias:
            raise ValueError("model_alias must be non-empty")
        if not instructions:
            raise ValueError("instructions must be non-empty")
        if resolved.gateway is None:
            raise ValueError("resolved model must provide an LLMGateway")
        self._model_alias = model_alias
        self._resolved = resolved
        self._instructions = instructions

    def prepare(self, request: HarnessModelRequest) -> PreparedModelCall:
        remaining = request.model_token_budget_remaining
        if remaining is not None and (isinstance(remaining, bool) or not isinstance(remaining, int) or remaining < 1):
            raise ValueError("remaining model token budget must be positive")
        messages = _model_messages(request.messages)
        first_user_index = next(
            (index for index, message in enumerate(messages) if message.role == "user"),
            None,
        )
        if first_user_index is None:
            raise ValueError("Harness model context must contain a user message")
        leading = messages[:first_user_index]
        if any(message.role != "context" for message in leading):
            raise ValueError("Harness model context may only place context fragments before the first user message")
        first = messages[first_user_index]
        context = build_stable_context(
            instructions=self._instructions,
            frozen_run_context=tuple(
                ContextBlock(name=f"harness_prefix_{index}", content=message.content)
                for index, message in enumerate(leading, start=1)
            ),
            initial_user_task=first.content,
            transcript=messages[first_user_index + 1 :],
        )
        settings = _model_settings(self._resolved)
        context, canonical_request, context_projection = _budgeted_request(
            request_id=f"{request.turn_id}:step:{request.step}",
            context=context,
            selected_tools=request.tools,
            settings=settings,
            resolved=self._resolved,
        )
        request_hash = hashlib.sha256(canonical_model_request_json(canonical_request).encode()).hexdigest()
        context_hash = hashlib.sha256(context.context_revision.encode()).hexdigest()
        tool_hash = hashlib.sha256(canonical_request.toolset_revision.encode()).hexdigest()
        if (self._resolved.provider in {"mlx", "ollama"} and not self._resolved.capabilities.supports_native_tools):
            local_wire = render_local_agent_request(
                canonical_request,
                provider=self._resolved.provider,
            )
            wire_hash = local_wire.provider_wire_hash
            serializer_revision = local_wire.serializer_revision
        else:
            openai_wire = serialize_openai_request(canonical_request)
            wire_hash = openai_wire.provider_wire_hash
            serializer_revision = openai_wire.serializer_revision
        return PreparedModelCall(
            request_hash=request_hash,
            context_hash=context_hash,
            tool_hash=tool_hash,
            wire_hash=wire_hash,
            request_ref={
                "model_alias": self._model_alias,
                "request_id": canonical_request.request_id,
                "prompt_revision": canonical_request.prompt_revision,
                "toolset_revision": canonical_request.toolset_revision,
                "serializer_revision": serializer_revision,
                "message_count": len(canonical_request.messages),
                "exposed_tool_names": canonical_request.exposed_tool_names,
                "model_token_budget_remaining": remaining,
                **({"context_projection": context_projection} if context_projection is not None else {}),
            },
            resource_request=_model_resource_request(
                request=canonical_request,
                settings=settings,
                resolved=self._resolved,
            ),
            dispatch_payload=_GatewayDispatch(
                request=canonical_request,
                wire_hash=wire_hash,
                resolved=self._resolved,
                model_token_budget_remaining=remaining,
            ),
        )

    async def dispatch(
        self,
        prepared: PreparedModelCall,
        *,
        delta_sink: HarnessModelDeltaSink | None = None,
    ) -> HarnessModelResponse:
        payload = prepared.dispatch_payload
        if not isinstance(payload, _GatewayDispatch):
            raise TypeError("prepared call does not belong to GatewayHarnessModel")
        resolved = payload.resolved
        gateway = resolved.gateway
        if gateway is None:
            raise RuntimeError("resolved model gateway disappeared before dispatch")
        ledger = (
            None
            if payload.model_token_budget_remaining is None
            else LLMBudgetLedger(total=payload.model_token_budget_remaining)
        )
        streamed_content: dict[str, list[str]] = {
            "text": [],
            "reasoning": [],
            "plan": [],
        }

        async def forward_delta(delta: ProviderDelta) -> None:
            streamed_content[delta.channel.value].append(delta.content)
            if delta_sink is None:
                return
            emitted = delta_sink(
                HarnessModelDelta(
                    channel=delta.channel.value,
                    content=delta.content,
                )
            )
            if inspect.isawaitable(emitted):
                await emitted

        try:
            response = await gateway.agenerate_model_request(
                stage=LLMCallStage.AGENT_STEP,
                request=payload.request,
                provider=resolved.provider,
                supports_native_tools=resolved.capabilities.supports_native_tools,
                stream=delta_sink is not None,
                delta_sink=forward_delta if delta_sink is not None else None,
                ledger=ledger,
                lease_id=f"{payload.request.request_id}:provider",
            )
        except LLMBudgetExceededError as exc:
            raise ModelDispatchPreflightError("Provider call exceeds the Turn's remaining model token budget.") from exc
        except LLMContextOverflowError as exc:
            raise ModelDispatchPreflightError(
                f"Model context exceeds the effective stage input budget: {exc.input_tokens} > {exc.max_input_tokens}."
            ) from exc
        except asyncio.CancelledError as exc:
            current = asyncio.current_task()
            if current is not None and current.cancelling():
                raise
            raise ModelDispatchCancelledError(
                "provider acknowledged model cancellation"
            ) from exc
        except BaseException as exc:
            if _is_uncertain_transport_failure(exc):
                raise ModelDispatchOutcomeUnknownError(str(exc).strip() or type(exc).__name__) from exc
            raise
        if response.provider_wire_hash != payload.wire_hash:
            raise RuntimeError("provider wire changed between prepare and dispatch")
        if response.turn.stop_reason is StopReason.MAX_TOKENS:
            return HarnessModelResponse(
                text=response.turn.text,
                provider_response_id=None,
                usage=response.usage.model_dump(mode="json"),
                tool_calls=tuple(
                    HarnessToolCall(
                        id=call.id,
                        name=call.name,
                        arguments=call.input,
                    )
                    for call in response.turn.tool_calls
                ),
                status="incomplete",
                incomplete_reason="max_output_tokens",
                reasoning_content=(
                    response.turn.reasoning_content
                    or "".join(streamed_content["reasoning"])
                    or None
                ),
                plan_content="".join(streamed_content["plan"]) or None,
            )
        if response.turn.stop_reason is StopReason.TOOL_USE and not response.turn.tool_calls:
            raise RuntimeError("tool-use stop reason did not include tool calls")
        if response.turn.stop_reason is StopReason.END_TURN and response.turn.tool_calls:
            raise RuntimeError("end-turn response unexpectedly included tool calls")
        if not response.turn.text.strip() and not response.turn.tool_calls:
            raise RuntimeError("model returned neither text nor tool calls")
        return HarnessModelResponse(
            text=response.turn.text,
            provider_response_id=None,
            usage=response.usage.model_dump(mode="json"),
            tool_calls=tuple(
                HarnessToolCall(
                    id=call.id,
                    name=call.name,
                    arguments=call.input,
                )
                for call in response.turn.tool_calls
            ),
            reasoning_content=(
                response.turn.reasoning_content
                or "".join(streamed_content["reasoning"])
                or None
            ),
            plan_content="".join(streamed_content["plan"]) or None,
        )


class ControlPlaneHarnessModel:
    """One trusted owner for per-Turn model binding and provider dispatch."""

    def __init__(
        self,
        *,
        control_plane: ModelControlPlane,
        instructions: tuple[str, ...],
    ) -> None:
        if not instructions:
            raise ValueError("instructions must be non-empty")
        self._control_plane = control_plane
        self._instructions = instructions

    def snapshot(self, *, thread_id: str, turn_id: str) -> dict[str, JsonValue]:
        return self._control_plane.freeze_model_binding(
            thread_id=thread_id,
            turn_id=turn_id,
        )

    def ensure_available(
        self,
        binding: Mapping[str, Any],
        *,
        thread_id: str,
        turn_id: str,
    ) -> None:
        model_binding = _authenticated_model_binding(binding)
        self._resolve_frozen(
            model_binding,
            thread_id=thread_id,
            turn_id=turn_id,
        )

    def prepare(self, request: HarnessModelRequest) -> PreparedModelCall:
        binding = _authenticated_model_binding(request.binding_manifest)
        resolved = self._resolve_frozen(
            binding,
            thread_id=request.thread_id,
            turn_id=request.turn_id,
        )
        envelope = binding["binding"]
        if not isinstance(envelope, Mapping):
            raise RuntimeError("validated model binding envelope changed type")
        alias = envelope["alias"]
        if not isinstance(alias, str):
            raise RuntimeError("validated model alias changed type")
        return GatewayHarnessModel(
            model_alias=alias,
            resolved=resolved,
            instructions=self._instructions,
        ).prepare(request)

    def _resolve_frozen(
        self,
        binding: Mapping[str, JsonValue],
        *,
        thread_id: str,
        turn_id: str,
    ) -> ResolvedModel:
        return self._control_plane.resolve_frozen_binding(
            binding,
            thread_id=thread_id,
            turn_id=turn_id,
        )

    async def dispatch(
        self,
        prepared: PreparedModelCall,
        *,
        delta_sink: HarnessModelDeltaSink | None = None,
    ) -> HarnessModelResponse:
        payload = prepared.dispatch_payload
        if not isinstance(payload, _GatewayDispatch):
            raise TypeError("prepared call does not belong to ControlPlaneHarnessModel")
        return await GatewayHarnessModel(
            model_alias=str(prepared.request_ref["model_alias"]),
            resolved=payload.resolved,
            instructions=self._instructions,
        ).dispatch(prepared, delta_sink=delta_sink)


_AUTHENTICATED_MODEL_BINDING_FIELDS = frozenset(
    {
        "authentication_schema_version",
        "trust_domain_id",
        "signing_key_id",
        "thread_id",
        "turn_id",
        "selection_requester",
        "binding",
        "signature",
    }
)


def _authenticated_model_binding(
    manifest: Mapping[str, Any],
) -> dict[str, JsonValue]:
    if "authentication_schema_version" not in manifest:
        raise RuntimeError(
            "legacy model binding is incomplete and cannot be resumed safely"
        )
    missing = _AUTHENTICATED_MODEL_BINDING_FIELDS.difference(manifest)
    if missing:
        raise ValueError(
            "Turn binding is missing authenticated model fields: " + ", ".join(sorted(missing))
        )
    return cast(
        dict[str, JsonValue],
        {key: manifest[key] for key in _AUTHENTICATED_MODEL_BINDING_FIELDS},
    )


def _is_uncertain_transport_failure(error: BaseException) -> bool:
    if isinstance(error, (ConnectionError, TimeoutError)):
        return True
    error_type = type(error)
    return (error_type.__module__, error_type.__name__) in {
        ("openai", "APIConnectionError"),
        ("openai", "APITimeoutError"),
        ("httpx", "ConnectError"),
        ("httpx", "ConnectTimeout"),
        ("httpx", "ReadError"),
        ("httpx", "ReadTimeout"),
        ("httpcore", "ConnectError"),
        ("httpcore", "ConnectTimeout"),
        ("httpcore", "ReadError"),
        ("httpcore", "ReadTimeout"),
    }


def _model_messages(messages: tuple[HarnessMessage, ...]) -> tuple[ModelMessage, ...]:
    if not messages:
        raise ValueError("Harness model request requires messages")
    converted: list[ModelMessage] = []
    for message in messages:
        if message.role not in {"user", "assistant", "tool", "context"}:
            raise ValueError(f"unsupported Harness message role: {message.role}")
        converted.append(
            ModelMessage(
                role=message.role,  # type: ignore[arg-type]
                content=message.content,
                tool_calls=tuple(
                    ModelToolCall(
                        id=call.id,
                        name=call.name,
                        input=dict(call.arguments),
                    )
                    for call in message.tool_calls
                ),
                tool_call_id=message.tool_call_id,
            )
        )
    return tuple(converted)


def _model_settings(resolved: ResolvedModel) -> ModelSettings:
    defaults = resolved.request_defaults

    provider_options = (
        defaults.provider_options.model_dump(
            mode="python",
            exclude_none=True,
        )
        if defaults.provider_options is not None
        else {}
    )

    return ModelSettings(
        model=resolved.model,
        max_output_tokens=resolved.capabilities.max_output_tokens,
        temperature=(
            defaults.temperature
            if defaults.temperature is not None
            else 0.0
        ),
        top_p=(
            defaults.top_p
            if defaults.top_p is not None
            else 1.0
        ),
        parallel_tool_calls=(
            defaults.parallel_tool_calls
            if defaults.parallel_tool_calls is not None
            else False
        ),
        seed=defaults.seed,
        provider_options=provider_options,
    )



def _model_resource_request(
    *,
    request: ModelRequest,
    settings: ModelSettings,
    resolved: ResolvedModel,
) -> ResourceUsage:
    """Compute the amount that must be reserved before provider dispatch."""

    input_tokens = 0
    count = getattr(resolved.token_accounting, "count", None)
    if callable(count):
        measured = count(
            model_request_input_text(
                request,
                provider=resolved.provider,
                supports_native_tools=resolved.capabilities.supports_native_tools,
            )
        )
        if isinstance(measured, bool) or not isinstance(measured, int) or measured < 0:
            raise RuntimeError("token accounting returned an invalid input count")
        input_tokens = measured

    max_output_tokens = settings.max_output_tokens or 0
    if (
        isinstance(max_output_tokens, bool)
        or not isinstance(max_output_tokens, int)
        or max_output_tokens < 0
    ):
        raise RuntimeError("model max_output_tokens is invalid")

    return ResourceUsage(
        input_tokens=input_tokens,
        output_tokens=max_output_tokens,
        model_calls=1,
    )

def _budgeted_request(
    *,
    request_id: str,
    context: StableModelContext,
    selected_tools: tuple[object, ...],
    settings: ModelSettings,
    resolved: ResolvedModel,
) -> tuple[StableModelContext, ModelRequest, Mapping[str, Any] | None]:
    def build(candidate: StableModelContext) -> ModelRequest:
        return build_model_request(
            request_id=request_id,
            context=candidate,
            selected_tools=selected_tools,  # type: ignore[arg-type]
            settings=settings,
        )

    canonical = build(context)
    gateway = resolved.gateway
    accounting = resolved.token_accounting
    effective_budget = getattr(gateway, "effective_stage_budget", None)
    count = getattr(accounting, "count", None)
    if not callable(effective_budget) or not callable(count):
        return context, canonical, None
    budget = effective_budget(
        LLMCallStage.AGENT_STEP,
        kwargs={"max_tokens": settings.max_output_tokens},
    )
    max_input_tokens = getattr(budget, "max_input_tokens", None)
    if isinstance(max_input_tokens, bool) or not isinstance(max_input_tokens, int):
        raise RuntimeError("gateway returned an invalid effective input budget")

    def measured(candidate: ModelRequest) -> int:
        return int(
            count(
                model_request_input_text(
                    candidate,
                    provider=resolved.provider,
                    supports_native_tools=(resolved.capabilities.supports_native_tools),
                )
            )
        )

    input_tokens = measured(canonical)
    if input_tokens <= max_input_tokens:
        return (
            context,
            canonical,
            {
                "compacted": False,
                "input_tokens": input_tokens,
                "max_input_tokens": max_input_tokens,
            },
        )

    transcript_count = len(context.transcript)
    candidates: list[tuple[int, int]] = []
    for summary_chars in (4_000, 2_000, 1_000):
        for tail_count in (12, 8, 4, 2, 0):
            tail_start = max(0, transcript_count - min(tail_count, transcript_count))
            pair = (tail_start, summary_chars)
            if pair not in candidates:
                candidates.append(pair)
    most_compact = (context, canonical, input_tokens)
    for tail_start, summary_chars in candidates:
        projected = context.project_compaction(
            tail_start=tail_start,
            max_summary_chars=summary_chars,
            project_tool_results=True,
        )
        if projected is context:
            continue
        candidate = build(projected)
        candidate_tokens = measured(candidate)
        most_compact = (projected, candidate, candidate_tokens)
        if candidate_tokens <= max_input_tokens:
            return (
                projected,
                candidate,
                {
                    "compacted": True,
                    "input_tokens": candidate_tokens,
                    "max_input_tokens": max_input_tokens,
                    "parent_context_revision": context.context_revision,
                    "projected_context_revision": projected.context_revision,
                    "retained_tail_count": len(projected.transcript) - 1,
                    "summary_max_chars": summary_chars,
                },
            )
    projected, candidate, candidate_tokens = most_compact
    return (
        projected,
        candidate,
        {
            "compacted": projected is not context,
            "input_tokens": candidate_tokens,
            "max_input_tokens": max_input_tokens,
            "parent_context_revision": context.context_revision,
            "projected_context_revision": projected.context_revision,
            "retained_tail_count": max(0, len(projected.transcript) - 1),
        },
    )
