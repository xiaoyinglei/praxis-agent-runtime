from __future__ import annotations

import asyncio
import inspect
import json
import queue as thread_queue
import threading
from collections.abc import AsyncGenerator, Awaitable, Callable, Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, replace
from enum import StrEnum
from typing import Any, Protocol, cast
from uuid import uuid4

from pydantic import BaseModel

from agent_runtime.core.messages import StopReason, ToolCall, ToolUseResult
from agent_runtime.core.model_request import ModelRequest
from agent_runtime.modeling.contracts import (
    DEFAULT_LLM_STAGE_BUDGETS,
    LLMCallResult,
    LLMCallStage,
    LLMProviderResult,
    LLMStageBudget,
    LLMUsage,
    normalize_llm_usage,
)
from agent_runtime.modeling.local_agent_wire import (
    parse_local_agent_response,
    render_local_agent_request,
)
from agent_runtime.modeling.quota import (
    ProviderQuotaGate,
    acquire_provider_quota,
)
from agent_runtime.modeling.openai_wire import (
    parse_openai_response,
    parse_openai_usage,
    serialize_openai_request,
)


@dataclass
class StreamChunk:
    """LLM 流式输出的一个 chunk。"""

    type: str  # "text_delta" | "thinking_delta" | "tool_use_start" |
    # "tool_input_delta" | "content_block_stop" | "message_stop"
    content: str = ""
    tool_name: str = ""
    tool_id: str = ""
    stop_reason: str = ""  # "end_turn" | "tool_use" | "max_tokens"
    usage: LLMUsage | None = None


class ProviderDeltaChannel(StrEnum):
    """Provider-visible model output channels."""

    TEXT = "text"
    REASONING = "reasoning"
    PLAN = "plan"


@dataclass(frozen=True, slots=True)
class ProviderDelta:
    """One honest provider delta, before it becomes a Turn Item delta."""

    channel: ProviderDeltaChannel
    content: str


ProviderDeltaSink = Callable[[ProviderDelta], None | Awaitable[None]]

_PROVIDER_BRIDGE_MAXSIZE = 64
_PROVIDER_BRIDGE_POLL_SECONDS = 0.005
_PROVIDER_BRIDGE_PUT_TIMEOUT_SECONDS = 0.05


@dataclass(frozen=True, slots=True)
class AgentModelResponse:
    turn: ToolUseResult
    usage: LLMUsage
    provider_wire_hash: str
    serializer_revision: str
    wire_kind: str


class TokenAccounting(Protocol):
    def count(self, text: str) -> int: ...

    def clip(
        self,
        text: str,
        token_budget: int,
        *,
        add_ellipsis: bool = False,
    ) -> str: ...


def model_request_input_text(
    request: ModelRequest,
    *,
    provider: str,
    supports_native_tools: bool,
) -> str:
    """Return the exact text accounted by the gateway before provider I/O."""

    if provider in {"mlx", "ollama"} and not supports_native_tools:
        return render_local_agent_request(
            request,
            provider=provider,
        ).prompt
    return serialize_openai_request(request).serialized_json


class AsyncBudgetLedger(Protocol):
    async def reserve(self, lease_id: str, amount: int) -> bool: ...
    async def commit(self, lease_id: str, actual: int) -> int: ...
    async def refund(self, lease_id: str) -> int: ...


_ACTIVE_LLM_BUDGET_LEDGER: ContextVar[AsyncBudgetLedger | None] = ContextVar(
    "active_llm_budget_ledger",
    default=None,
)


def current_llm_budget_ledger() -> AsyncBudgetLedger | None:
    return _ACTIVE_LLM_BUDGET_LEDGER.get()


@contextmanager
def llm_budget_scope(
    ledger: AsyncBudgetLedger | None,
) -> Iterator[None]:
    token = _ACTIVE_LLM_BUDGET_LEDGER.set(ledger)
    try:
        yield
    finally:
        _ACTIVE_LLM_BUDGET_LEDGER.reset(token)


def _resolve_budget_ledger(
    explicit: AsyncBudgetLedger | None,
    *,
    inherit_ambient: bool,
) -> AsyncBudgetLedger | None:
    if type(inherit_ambient) is not bool:
        raise TypeError("inherit_ambient must be a bool")
    if explicit is not None:
        return explicit
    return current_llm_budget_ledger() if inherit_ambient else None


class LLMGatewayError(RuntimeError):
    pass


class LLMToolCallValidationError(LLMGatewayError):
    """A provider rejected its own generated tool call against the schema."""

    def __init__(
        self,
        *,
        validation_error: str,
        failed_generation: str = "",
    ) -> None:
        self.validation_error = validation_error[:2_000]
        self.failed_generation = failed_generation[:4_000]
        super().__init__(f"Provider rejected a generated tool call: {self.validation_error}")


class LLMContextOverflowError(LLMGatewayError):
    def __init__(
        self,
        *,
        stage: LLMCallStage,
        input_tokens: int,
        max_input_tokens: int,
    ) -> None:
        super().__init__(f"{stage.value} input uses {input_tokens} tokens; maximum is {max_input_tokens}")
        self.stage = stage
        self.input_tokens = input_tokens
        self.max_input_tokens = max_input_tokens


class LLMBudgetExceededError(LLMGatewayError):
    def __init__(self, *, stage: LLMCallStage, required_tokens: int) -> None:
        super().__init__(f"Insufficient LLM token budget for {stage.value}; required reservation is {required_tokens}")
        self.stage = stage
        self.required_tokens = required_tokens


class LLMGateway:
    def __init__(
        self,
        *,
        generator: object,
        token_accounting: TokenAccounting,
        model_context_tokens: int,
        stage_budgets: Mapping[LLMCallStage, LLMStageBudget] | None = None,
        provider_quota_gate: ProviderQuotaGate | None = None,
    ) -> None:
        if model_context_tokens <= 0:
            raise ValueError("model_context_tokens must be positive")
        self._generator = generator
        self._token_accounting = token_accounting
        self._model_context_tokens = model_context_tokens
        self._stage_budgets = dict(stage_budgets or DEFAULT_LLM_STAGE_BUDGETS)
        self._provider_quota_gate = provider_quota_gate

    @property
    def token_accounting(self) -> TokenAccounting:
        return self._token_accounting

    def stage_budget(self, stage: LLMCallStage) -> LLMStageBudget:
        return self._stage_budget(stage).model_copy()

    def effective_stage_budget(
        self,
        stage: LLMCallStage,
        *,
        kwargs: Mapping[str, Any] | None = None,
    ) -> LLMStageBudget:
        budget = self._stage_budget(stage)
        max_output_tokens = self._effective_max_output_tokens(
            budget,
            kwargs=kwargs,
        )
        max_input_tokens = min(
            budget.max_input_tokens,
            max(
                self._model_context_tokens - max_output_tokens - budget.safety_margin_tokens,
                0,
            ),
        )
        if max_input_tokens <= 0:
            raise ValueError(f"No input budget remains for {stage.value} after reserving output and safety margin")
        return budget.model_copy(
            update={
                "max_input_tokens": max_input_tokens,
                "max_output_tokens": max_output_tokens,
            }
        )

    async def agenerate_model_request(
        self,
        *,
        stage: LLMCallStage,
        request: ModelRequest,
        provider: str,
        supports_native_tools: bool,
        stream: bool = False,
        delta_sink: ProviderDeltaSink | None = None,
        ledger: AsyncBudgetLedger | None = None,
        lease_id: str | None = None,
        inherit_budget_ledger: bool = True,
    ) -> AgentModelResponse:
        """Serialize and execute one already-selected canonical agent request."""

        if not isinstance(request, ModelRequest):
            raise TypeError("request must be a canonical ModelRequest")
        if not isinstance(provider, str) or not provider:
            raise ValueError("provider must be a non-empty string")
        if type(supports_native_tools) is not bool:
            raise TypeError("supports_native_tools must be a bool")
        if type(stream) is not bool:
            raise TypeError("stream must be a bool")

        if provider in {"mlx", "ollama"} and not supports_native_tools:
            local_wire = render_local_agent_request(request, provider=provider)
            budget, stage_kwargs, input_tokens, reservation = self._prepare_call(
                stage=stage,
                prompt=local_wire.prompt,
                kwargs=local_wire.generation_options,
            )
            await acquire_provider_quota(
                self._provider_quota_gate,
                provider=provider,
                model=request.settings.model,
                requested_tokens=reservation,
            )
            del budget
            effective_ledger = _resolve_budget_ledger(
                ledger,
                inherit_ambient=inherit_budget_ledger,
            )
            effective_lease_id = lease_id or f"{stage.value}:canonical:{request.request_id}"
            if effective_ledger is not None:
                if not await effective_ledger.reserve(effective_lease_id, reservation):
                    raise LLMBudgetExceededError(stage=stage, required_tokens=reservation)
            try:
                provider_result = await asyncio.to_thread(
                    self._invoke_text,
                    local_wire.prompt,
                    stage_kwargs,
                )
            except BaseException:
                if effective_ledger is not None:
                    await effective_ledger.refund(effective_lease_id)
                raise
            turn = parse_local_agent_response(provider_result.value)
            usage = provider_result.usage or normalize_llm_usage(
                input_tokens=input_tokens,
                output_tokens=self._token_accounting.count(provider_result.value),
                input_tokens_include_cache=True,
                usage_source="tokenizer_estimate",
            )
            if effective_ledger is not None:
                await effective_ledger.commit(effective_lease_id, usage.total_tokens)
            if stream and turn.text and delta_sink is not None:
                await _emit_provider_delta(
                    delta_sink,
                    ProviderDelta(ProviderDeltaChannel.TEXT, turn.text),
                )
            return AgentModelResponse(
                turn=turn,
                usage=usage,
                provider_wire_hash=local_wire.provider_wire_hash,
                serializer_revision=local_wire.serializer_revision,
                wire_kind=provider,
            )

        wire = serialize_openai_request(request)
        payload = _plain_wire_mapping(wire.payload)
        messages = _plain_wire_sequence(payload.get("messages", []), field_name="messages")
        tools = _plain_wire_sequence(payload.get("tools", []), field_name="tools")
        provider_kwargs = {
            key: value
            for key, value in payload.items()
            if key not in {"model", "messages", "tools", "max_completion_tokens"}
        }
        if request.settings.provider_options:
            provider_kwargs["extra_body"] = {key: provider_kwargs.pop(key) for key in request.settings.provider_options}
        max_completion_tokens = payload.get("max_completion_tokens")
        if isinstance(max_completion_tokens, int):
            # Route the canonical request limit through the gateway so context
            # projection, reservation, and the provider call use one budget.
            provider_kwargs["max_tokens"] = max_completion_tokens
        if stream:
            chunks = self.astream_with_tools(
                stage=stage,
                messages=messages,
                tools=tools,
                ledger=ledger,
                lease_id=lease_id or f"{stage.value}:canonical-stream:{request.request_id}",
                inherit_budget_ledger=inherit_budget_ledger,
                quota_provider=provider,
                quota_model=request.settings.model,
                kwargs=provider_kwargs,
            )
            text = ""
            reasoning_content = ""
            calls: list[ToolCall] = []
            active_tool_id = ""
            tool_names: dict[str, str] = {}
            tool_arguments: dict[str, str] = {}
            completed_tool_ids: set[str] = set()
            stop_reason = "end_turn"
            final_usage: LLMUsage | None = None
            async for chunk in chunks:
                if chunk.type == "text_delta":
                    text += chunk.content
                    if delta_sink is not None:
                        await _emit_provider_delta(
                            delta_sink,
                            ProviderDelta(
                                ProviderDeltaChannel.TEXT,
                                chunk.content,
                            ),
                        )
                elif chunk.type == "thinking_delta":
                    reasoning_content += chunk.content
                    if delta_sink is not None:
                        await _emit_provider_delta(
                            delta_sink,
                            ProviderDelta(
                                ProviderDeltaChannel.REASONING,
                                chunk.content,
                            ),
                        )
                elif chunk.type == "plan_delta":
                    if delta_sink is not None:
                        await _emit_provider_delta(
                            delta_sink,
                            ProviderDelta(
                                ProviderDeltaChannel.PLAN,
                                chunk.content,
                            ),
                        )
                elif chunk.type == "tool_use_start":
                    active_tool_id = chunk.tool_id or f"call_{len(tool_names) + 1}"
                    tool_names[active_tool_id] = chunk.tool_name
                    tool_arguments[active_tool_id] = ""
                elif chunk.type == "tool_input_delta":
                    tool_id = chunk.tool_id or active_tool_id
                    if tool_id in tool_arguments:
                        tool_arguments[tool_id] += chunk.content
                elif chunk.type == "content_block_stop":
                    tool_id = chunk.tool_id or active_tool_id
                    tool_name = tool_names.get(tool_id, "")
                    if not tool_name or tool_id in completed_tool_ids:
                        continue
                    calls.append(
                        ToolCall(
                            id=tool_id,
                            name=tool_name,
                            input=_tool_arguments(tool_arguments.get(tool_id, "")),
                        )
                    )
                    completed_tool_ids.add(tool_id)
                    if active_tool_id == tool_id:
                        active_tool_id = ""
                elif chunk.type == "message_stop":
                    stop_reason = chunk.stop_reason or stop_reason
                    final_usage = chunk.usage
            if final_usage is None:
                raise RuntimeError("canonical streaming call ended without usage")
            normalized_stop = (
                StopReason.TOOL_USE
                if calls or stop_reason in {"tool_use", "tool_calls"}
                else StopReason.MAX_TOKENS
                if stop_reason == "max_tokens"
                else StopReason.END_TURN
            )
            turn = ToolUseResult(
                tool_calls=calls,
                text=text,
                reasoning_content=reasoning_content or None,
                stop_reason=normalized_stop,
                raw_stop_reason=stop_reason,
            )
            return AgentModelResponse(
                turn=turn,
                usage=final_usage,
                provider_wire_hash=wire.provider_wire_hash,
                serializer_revision=wire.serializer_revision,
                wire_kind="openai",
            )

        accounted_prompt = wire.serialized_json
        budget, stage_kwargs, input_tokens, reservation = self._prepare_call(
            stage=stage,
            prompt=accounted_prompt,
            kwargs=provider_kwargs,
        )
        await acquire_provider_quota(
            self._provider_quota_gate,
            provider=provider,
            model=request.settings.model,
            requested_tokens=reservation,
        )
        del budget
        effective_ledger = _resolve_budget_ledger(
            ledger,
            inherit_ambient=inherit_budget_ledger,
        )
        effective_lease_id = lease_id or f"{stage.value}:canonical:{request.request_id}"
        if effective_ledger is not None:
            if not await effective_ledger.reserve(effective_lease_id, reservation):
                raise LLMBudgetExceededError(stage=stage, required_tokens=reservation)
        try:
            provider_result = await asyncio.to_thread(
                self._invoke_with_tools,
                messages,
                tools,
                stage_kwargs,
            )
        except BaseException:
            if effective_ledger is not None:
                await effective_ledger.refund(effective_lease_id)
            raise
        turn = parse_openai_response(provider_result.value)
        usage = (
            provider_result.usage
            or parse_openai_usage(provider_result.value)
            or normalize_llm_usage(
                input_tokens=input_tokens,
                output_tokens=self._token_accounting.count(turn.text),
                input_tokens_include_cache=True,
                usage_source="tokenizer_estimate",
            )
        )
        if effective_ledger is not None:
            await effective_ledger.commit(effective_lease_id, usage.total_tokens)
        return AgentModelResponse(
            turn=turn,
            usage=usage,
            provider_wire_hash=wire.provider_wire_hash,
            serializer_revision=wire.serializer_revision,
            wire_kind="openai",
        )

    async def agenerate_text(
        self,
        *,
        stage: LLMCallStage,
        prompt: str,
        ledger: AsyncBudgetLedger | None = None,
        lease_id: str | None = None,
        kwargs: Mapping[str, Any] | None = None,
    ) -> LLMCallResult[str]:
        effective_ledger = ledger or current_llm_budget_ledger()
        _, call_kwargs, input_tokens, reservation = self._prepare_call(
            stage=stage,
            prompt=prompt,
            kwargs=kwargs,
        )
        effective_lease_id = lease_id or f"{stage.value}:{id(prompt)}"
        if effective_ledger is not None:
            reserved = await effective_ledger.reserve(effective_lease_id, reservation)
            if not reserved:
                raise LLMBudgetExceededError(
                    stage=stage,
                    required_tokens=reservation,
                )

        try:
            provider_result = await asyncio.to_thread(
                self._invoke_text,
                prompt,
                call_kwargs,
            )
        except asyncio.CancelledError:
            if effective_ledger is not None:
                await effective_ledger.refund(effective_lease_id)
            raise
        except Exception:
            if effective_ledger is not None:
                await effective_ledger.refund(effective_lease_id)
            raise

        usage = provider_result.usage or LLMUsage(
            input_tokens=input_tokens,
            output_tokens=self._token_accounting.count(provider_result.value),
            source="tokenizer_estimate",
        )
        if effective_ledger is not None:
            await effective_ledger.commit(effective_lease_id, usage.total_tokens)
        return LLMCallResult(
            value=provider_result.value,
            usage=usage,
            stage=stage,
        )

    async def agenerate_structured[T: BaseModel](
        self,
        *,
        stage: LLMCallStage,
        prompt: str,
        schema: type[T],
        ledger: AsyncBudgetLedger | None = None,
        lease_id: str | None = None,
        kwargs: Mapping[str, Any] | None = None,
    ) -> LLMCallResult[T]:
        effective_ledger = ledger or current_llm_budget_ledger()
        accounted_prompt = structured_accounted_prompt(prompt, schema)
        budget, call_kwargs, input_tokens, reservation = self._prepare_call(
            stage=stage,
            prompt=accounted_prompt,
            kwargs=kwargs,
        )
        effective_lease_id = lease_id or f"{stage.value}:{id(prompt)}"
        if effective_ledger is not None:
            reserved = await effective_ledger.reserve(effective_lease_id, reservation)
            if not reserved:
                raise LLMBudgetExceededError(
                    stage=stage,
                    required_tokens=reservation,
                )

        try:
            provider_result = await asyncio.to_thread(
                self._invoke_structured,
                prompt,
                schema,
                call_kwargs,
            )
        except asyncio.CancelledError:
            if effective_ledger is not None:
                await effective_ledger.refund(effective_lease_id)
            raise
        except Exception:
            if effective_ledger is not None:
                await effective_ledger.refund(effective_lease_id)
            raise

        usage = provider_result.usage or LLMUsage(
            input_tokens=input_tokens,
            output_tokens=self._token_accounting.count(provider_result.value.model_dump_json()),
            source="tokenizer_estimate",
        )
        if effective_ledger is not None:
            await effective_ledger.commit(effective_lease_id, usage.total_tokens)
        return LLMCallResult(
            value=provider_result.value,
            usage=usage,
            stage=stage,
        )

    async def agenerate_with_tools(
        self,
        *,
        stage: LLMCallStage,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        ledger: AsyncBudgetLedger | None = None,
        lease_id: str | None = None,
        kwargs: Mapping[str, Any] | None = None,
    ) -> LLMCallResult[Any]:
        """Native tool calling path with budget accounting.

        ``messages`` and ``tools`` are in OpenAI wire format.  Returns the
        raw provider response (caller parses via ``OpenAIAdapter``).
        Falls back to ``generate_text`` when the generator lacks
        ``generate_with_tools``.
        """
        effective_ledger = ledger or current_llm_budget_ledger()
        accounted_prompt = _account_messages(messages, tools)
        budget, call_kwargs, input_tokens, reservation = self._prepare_call(
            stage=stage,
            prompt=accounted_prompt,
            kwargs=kwargs,
        )
        effective_lease_id = lease_id or f"{stage.value}:tools:{id(messages)}"
        if effective_ledger is not None:
            reserved = await effective_ledger.reserve(effective_lease_id, reservation)
            if not reserved:
                raise LLMBudgetExceededError(
                    stage=stage,
                    required_tokens=reservation,
                )

        try:
            provider_result = await asyncio.to_thread(
                self._invoke_with_tools,
                messages,
                tools,
                call_kwargs,
            )
        except asyncio.CancelledError:
            if effective_ledger is not None:
                await effective_ledger.refund(effective_lease_id)
            raise
        except Exception:
            if effective_ledger is not None:
                await effective_ledger.refund(effective_lease_id)
            raise

        usage = provider_result.usage or LLMUsage(
            input_tokens=input_tokens,
            output_tokens=0,  # unknown for raw responses
            source="tokenizer_estimate",
        )
        if effective_ledger is not None:
            await effective_ledger.commit(effective_lease_id, usage.total_tokens)
        return LLMCallResult(
            value=provider_result.value,
            usage=usage,
            stage=stage,
        )

    async def astream_with_tools(
        self,
        *,
        stage: LLMCallStage,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        ledger: AsyncBudgetLedger | None = None,
        lease_id: str | None = None,
        inherit_budget_ledger: bool = True,
        quota_provider: str | None = None,
        quota_model: str | None = None,
        kwargs: Mapping[str, Any] | None = None,
    ) -> AsyncGenerator[StreamChunk, None]:
        """流式调用 LLM（带工具）。

        自动检测 generator 是否支持 stream_with_tools：
        - 支持：真流式，逐 chunk yield
        - 不支持：先拿完整结果，再诚实地发一个整体 delta

        producer 在独立 daemon 线程中运行，通过有界线程队列与
        async consumer 桥接。

        Yields:
            StreamChunk: text_delta / tool_use_start / tool_input_delta /
                         content_block_stop / message_stop
        """
        # Legacy ledger accounting is optional. Harness canonical calls
        # explicitly disable ambient inheritance because durable Rollout
        # reservations are their authoritative cumulative budget.
        effective_ledger = _resolve_budget_ledger(
            ledger,
            inherit_ambient=inherit_budget_ledger,
        )
        accounted_prompt = _account_messages(messages, tools)
        budget, call_kwargs, input_tokens, reservation = self._prepare_call(
            stage=stage,
            prompt=accounted_prompt,
            kwargs=kwargs,
        )
        effective_lease_id = lease_id or f"{stage.value}:stream:{id(messages)}"
        if quota_provider is not None or quota_model is not None:
            if not quota_provider or not quota_model:
                raise ValueError("quota_provider and quota_model must be supplied together")
            await acquire_provider_quota(
                self._provider_quota_gate,
                provider=quota_provider,
                model=quota_model,
                requested_tokens=reservation,
            )
        if effective_ledger is not None:
            reserved = await effective_ledger.reserve(effective_lease_id, reservation)
            if not reserved:
                raise LLMBudgetExceededError(
                    stage=stage,
                    required_tokens=reservation,
                )

        bridge: thread_queue.Queue[StreamChunk] = thread_queue.Queue(
            maxsize=_PROVIDER_BRIDGE_MAXSIZE
        )
        stop_producer = threading.Event()
        producer_done = threading.Event()
        producer_error: list[BaseException] = []

        def _put(chunk: StreamChunk) -> bool:
            while not stop_producer.is_set():
                try:
                    bridge.put(
                        chunk,
                        timeout=_PROVIDER_BRIDGE_PUT_TIMEOUT_SECONDS,
                    )
                    return True
                except thread_queue.Full:
                    continue
            return False

        def _producer() -> None:
            try:
                self._invoke_streaming_with_tools(
                    messages,
                    tools,
                    call_kwargs,
                    put=_put,
                    stop_event=stop_producer,
                )
            except BaseException as exc:
                normalized = _tool_call_validation_error(exc) if isinstance(exc, Exception) else None
                producer_error.append(normalized or exc)
            finally:
                producer_done.set()

        producer_thread = threading.Thread(
            target=_producer,
            name=f"llm-stream-{id(messages)}",
            daemon=True,
        )
        producer_thread.start()

        # 并发 yield chunks
        total_output_tokens = 0
        final_stop: StreamChunk | None = None
        reported_usage: LLMUsage | None = None
        try:
            while True:
                try:
                    chunk = bridge.get_nowait()
                except thread_queue.Empty:
                    if producer_done.is_set():
                        break
                    await asyncio.sleep(_PROVIDER_BRIDGE_POLL_SECONDS)
                    continue
                if chunk.type in {"text_delta", "thinking_delta"}:
                    total_output_tokens += self._token_accounting.count(
                        chunk.content
                    )
                if chunk.type == "message_stop":
                    final_stop = chunk
                    reported_usage = chunk.usage
                    continue
                yield chunk
        except asyncio.CancelledError:
            stop_producer.set()
            _request_provider_stream_cancel(self._generator)
            if effective_ledger is not None:
                await effective_ledger.refund(effective_lease_id)
            raise
        except GeneratorExit:
            stop_producer.set()
            _request_provider_stream_cancel(self._generator)
            if effective_ledger is not None:
                await effective_ledger.refund(effective_lease_id)
            raise

        # 检查 producer 是否有异常
        if producer_error:
            if effective_ledger is not None:
                await effective_ledger.refund(effective_lease_id)
            raise producer_error[0]

        # 提交预算
        usage = reported_usage or normalize_llm_usage(
            input_tokens=input_tokens,
            output_tokens=total_output_tokens,
            input_tokens_include_cache=True,
            usage_source="tokenizer_estimate",
        )
        if effective_ledger is not None:
            await effective_ledger.commit(effective_lease_id, usage.total_tokens)
        yield replace(
            final_stop or StreamChunk(type="message_stop", stop_reason="end_turn"),
            usage=usage,
        )

    def generate_text(
        self,
        *,
        stage: LLMCallStage,
        prompt: str,
        kwargs: Mapping[str, Any] | None = None,
    ) -> LLMCallResult[str]:
        _, call_kwargs, input_tokens, _ = self._prepare_call(
            stage=stage,
            prompt=prompt,
            kwargs=kwargs,
        )
        provider_result = self._invoke_text(prompt, call_kwargs)
        usage = provider_result.usage or LLMUsage(
            input_tokens=input_tokens,
            output_tokens=self._token_accounting.count(provider_result.value),
            source="tokenizer_estimate",
        )
        return LLMCallResult(
            value=provider_result.value,
            usage=usage,
            stage=stage,
        )

    def generate_structured[T: BaseModel](
        self,
        *,
        stage: LLMCallStage,
        prompt: str,
        schema: type[T],
        kwargs: Mapping[str, Any] | None = None,
    ) -> LLMCallResult[T]:
        accounted_prompt = structured_accounted_prompt(prompt, schema)
        _, call_kwargs, input_tokens, _ = self._prepare_call(
            stage=stage,
            prompt=accounted_prompt,
            kwargs=kwargs,
        )
        provider_result = self._invoke_structured(
            prompt,
            schema,
            call_kwargs,
        )
        usage = provider_result.usage or LLMUsage(
            input_tokens=input_tokens,
            output_tokens=self._token_accounting.count(provider_result.value.model_dump_json()),
            source="tokenizer_estimate",
        )
        return LLMCallResult(
            value=provider_result.value,
            usage=usage,
            stage=stage,
        )

    def _stage_budget(self, stage: LLMCallStage) -> LLMStageBudget:
        try:
            return self._stage_budgets[stage]
        except KeyError as exc:
            raise ValueError(f"No LLM stage budget configured for {stage.value}") from exc

    def _prepare_call(
        self,
        *,
        stage: LLMCallStage,
        prompt: str,
        kwargs: Mapping[str, Any] | None,
    ) -> tuple[LLMStageBudget, dict[str, Any], int, int]:
        budget = self.effective_stage_budget(stage, kwargs=kwargs)
        call_kwargs = dict(kwargs or {})
        max_output_tokens = budget.max_output_tokens
        call_kwargs["max_tokens"] = max_output_tokens

        max_input_tokens = budget.max_input_tokens
        input_tokens = self._token_accounting.count(prompt)
        if input_tokens > max_input_tokens:
            raise LLMContextOverflowError(
                stage=stage,
                input_tokens=input_tokens,
                max_input_tokens=max_input_tokens,
            )
        return (
            budget,
            call_kwargs,
            input_tokens,
            input_tokens + max_output_tokens,
        )

    @staticmethod
    def _effective_max_output_tokens(
        budget: LLMStageBudget,
        *,
        kwargs: Mapping[str, Any] | None,
    ) -> int:
        requested_output = (kwargs or {}).get("max_tokens")
        if isinstance(requested_output, int) and requested_output > 0:
            return min(budget.max_output_tokens, requested_output)
        return budget.max_output_tokens

    def _invoke_text(
        self,
        prompt: str,
        kwargs: dict[str, Any],
    ) -> LLMProviderResult[str]:
        with_usage = getattr(self._generator, "generate_text_with_usage", None)
        if callable(with_usage):
            result = with_usage(prompt=prompt, **kwargs)
            if isinstance(result, LLMProviderResult):
                return result
            raise TypeError("generate_text_with_usage must return LLMProviderResult")

        generate_text = getattr(self._generator, "generate_text", None)
        if callable(generate_text):
            return LLMProviderResult(value=str(generate_text(prompt=prompt, **kwargs)))

        chat = getattr(self._generator, "chat", None)
        if callable(chat):
            return LLMProviderResult(value=str(chat(prompt, **kwargs)))

        raise RuntimeError("Configured generator cannot generate text")

    def _invoke_structured[T: BaseModel](
        self,
        prompt: str,
        schema: type[T],
        kwargs: dict[str, Any],
    ) -> LLMProviderResult[T]:
        with_usage = getattr(self._generator, "generate_structured_with_usage", None)
        if callable(with_usage):
            result = with_usage(prompt=prompt, schema=schema, **kwargs)
            if isinstance(result, LLMProviderResult):
                return LLMProviderResult(
                    value=schema.model_validate(result.value),
                    usage=result.usage,
                )
            raise TypeError("generate_structured_with_usage must return LLMProviderResult")

        generate_structured = getattr(self._generator, "generate_structured", None)
        if callable(generate_structured):
            return LLMProviderResult(
                value=schema.model_validate(generate_structured(prompt=prompt, schema=schema, **kwargs))
            )
        raise RuntimeError("Configured generator cannot generate structured output")

    def _invoke_with_tools(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        kwargs: dict[str, Any],
    ) -> LLMProviderResult[Any]:
        generate_with_tools = getattr(self._generator, "generate_with_tools", None)
        if callable(generate_with_tools):
            try:
                result = generate_with_tools(messages=messages, tools=tools, **kwargs)
            except Exception as exc:
                normalized = _tool_call_validation_error(exc)
                if normalized is not None:
                    raise normalized from exc
                raise
            if isinstance(result, LLMProviderResult):
                return result
            raise TypeError("generate_with_tools must return LLMProviderResult")

        # Fallback: render messages as prompt, call generate_text
        prompt = _render_messages_as_prompt(messages)
        return self._invoke_text(prompt, kwargs)

    def _invoke_streaming_with_tools(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        kwargs: dict[str, Any],
        *,
        put: Callable[[StreamChunk], bool],
        stop_event: threading.Event,
    ) -> None:
        """在 daemon 线程中调用同步 generator 的流式接口。"""

        stream_with_tools = getattr(self._generator, "stream_with_tools", None)

        if callable(stream_with_tools):
            # 真流式路径
            for chunk in stream_with_tools(
                messages=messages, tools=tools, **kwargs
            ):
                if stop_event.is_set():
                    return
                if isinstance(chunk, StreamChunk):
                    if not put(chunk):
                        return
                elif isinstance(chunk, dict):
                    if not put(StreamChunk(**chunk)):
                        return
        else:
            # 非流式 provider：先拿完整结果，不伪造 token 时序。
            result = self._invoke_with_tools(messages, tools, kwargs)
            raw = result.value
            text = ""
            reasoning_content = ""
            tool_calls_list: list[dict[str, Any]] = []
            stop_reason = "end_turn"

            if (isinstance(raw, Mapping) and "choices" in raw) or hasattr(raw, "choices"):
                turn = parse_openai_response(raw)
                text = turn.text
                reasoning_content = turn.reasoning_content or ""
                stop_reason = turn.stop_reason.value
                tool_calls_list = [
                    {
                        "id": call.id,
                        "function": {
                            "name": call.name,
                            "arguments": json.dumps(
                                call.input,
                                ensure_ascii=False,
                                sort_keys=True,
                                separators=(",", ":"),
                            ),
                        },
                    }
                    for call in turn.tool_calls
                ]
            elif isinstance(raw, dict):
                choices = raw.get("choices", [])
                if choices:
                    message = choices[0].get("message", {})
                    text = message.get("content", "") or ""
                    tool_calls_list = message.get("tool_calls", []) or []
            elif hasattr(raw, "content"):
                text = str(getattr(raw, "content", ""))
                tool_calls_list = list(getattr(raw, "tool_calls", []))
            else:
                text = str(raw) if raw else ""

            if reasoning_content:
                if not put(
                    StreamChunk(
                        type="thinking_delta",
                        content=reasoning_content,
                    )
                ):
                    return

            if text:
                if not put(StreamChunk(
                    type="text_delta",
                    content=text,
                )):
                    return

            # yield 工具调用
            for tc in tool_calls_list:
                tc_id = tc.get("id", f"tc_{uuid4().hex[:12]}")
                tc_name = tc.get("function", {}).get("name", "") or tc.get("name", "")
                tc_args = tc.get("function", {}).get("arguments", "{}") or tc.get("input", "{}")
                if isinstance(tc_args, dict):
                    tc_args = json.dumps(tc_args)

                if not put(StreamChunk(
                    type="tool_use_start",
                    tool_name=tc_name,
                    tool_id=tc_id,
                )):
                    return
                if not put(StreamChunk(type="tool_input_delta", content=tc_args)):
                    return
                if not put(StreamChunk(type="content_block_stop")):
                    return

            if tool_calls_list:
                stop_reason = "tool_use"
            put(StreamChunk(
                type="message_stop",
                stop_reason=stop_reason,
                usage=result.usage,
            ))


def _render_messages_as_prompt(messages: list[dict[str, Any]]) -> str:
    """Render OpenAI messages as a flat prompt for fallback path."""
    parts: list[str] = []
    for msg in messages:
        role = msg.get("role", "unknown")
        content = msg.get("content", "")
        if role == "system":
            parts.append(f"[System]\n{content}")
        elif role == "user":
            parts.append(f"[User]\n{content}")
        elif role == "assistant":
            parts.append(f"[Assistant]\n{content}")
        elif role == "tool":
            parts.append(f"[Tool Result: {msg.get('tool_call_id', '')}]\n{content}")
    return "\n\n".join(parts)


async def _emit_provider_delta(
    sink: ProviderDeltaSink,
    delta: ProviderDelta,
) -> None:
    emitted = sink(delta)
    if inspect.isawaitable(emitted):
        await emitted


def _request_provider_stream_cancel(generator: object) -> None:
    cancel_stream = getattr(generator, "cancel_stream", None)
    if not callable(cancel_stream):
        return

    def _cancel() -> None:
        try:
            cancel_stream()
        except Exception:
            pass

    threading.Thread(
        target=_cancel,
        name=f"llm-stream-cancel-{id(generator)}",
        daemon=True,
    ).start()


def _plain_wire_mapping(value: Mapping[str, object]) -> dict[str, Any]:
    return {key: _plain_wire_value(item) for key, item in value.items()}


def _plain_wire_sequence(value: object, *, field_name: str) -> list[dict[str, Any]]:
    if not isinstance(value, (list, tuple)):
        raise TypeError(f"canonical wire {field_name} must be a sequence")
    items: list[dict[str, Any]] = []
    for item in value:
        if not isinstance(item, Mapping):
            raise TypeError(f"canonical wire {field_name} must contain mappings")
        items.append(_plain_wire_mapping(item))
    return items


def _plain_wire_value(value: object) -> Any:
    if isinstance(value, Mapping):
        return _plain_wire_mapping(value)
    if isinstance(value, tuple):
        return [_plain_wire_value(item) for item in value]
    return value


def _tool_arguments(value: str) -> dict[str, Any]:
    if not value:
        return {}
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return {"_raw": value[:20_000]}
    return dict(parsed) if isinstance(parsed, Mapping) else {"_raw": parsed}


def _account_messages(
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
) -> str:
    """Approximate token-countable text from messages + tools."""
    prompt = _render_messages_as_prompt(messages)
    if tools:
        tool_desc = "\n".join(t.get("function", {}).get("name", "") for t in tools)
        prompt += f"\n\n[Tools]\n{tool_desc}"
    return prompt


def structured_accounted_prompt(
    prompt: str,
    schema: type[BaseModel],
) -> str:
    schema_json = json.dumps(schema.model_json_schema(), ensure_ascii=False)
    return f"{prompt}\n\nJSON schema:\n{schema_json}"


def _tool_call_validation_error(
    exc: Exception,
) -> LLMToolCallValidationError | None:
    if isinstance(exc, LLMToolCallValidationError):
        return exc
    body = getattr(exc, "body", None)
    payload: Mapping[str, object] | None = body if isinstance(body, Mapping) else None
    if payload is not None and isinstance(payload.get("error"), Mapping):
        payload = cast(Mapping[str, object], payload["error"])

    message = ""
    code = ""
    failed_generation = ""
    if payload is not None:
        raw_message = payload.get("message")
        raw_code = payload.get("code")
        raw_generation = payload.get("failed_generation")
        if isinstance(raw_message, str):
            message = raw_message
        if isinstance(raw_code, str):
            code = raw_code
        if isinstance(raw_generation, str):
            failed_generation = raw_generation
        elif raw_generation is not None:
            try:
                failed_generation = json.dumps(
                    raw_generation,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
            except (TypeError, ValueError):
                failed_generation = ""

    fallback = str(exc)
    searchable = f"{code} {message} {fallback}".casefold()
    if (
        code.casefold() != "tool_use_failed"
        and "tool call validation failed" not in searchable
        and "tool_call_validation" not in searchable
    ):
        return None
    return LLMToolCallValidationError(
        validation_error=message or fallback,
        failed_generation=failed_generation,
    )


__all__ = [
    "AgentModelResponse",
    "LLMBudgetExceededError",
    "LLMContextOverflowError",
    "LLMGateway",
    "LLMGatewayError",
    "LLMToolCallValidationError",
    "ProviderDelta",
    "ProviderDeltaChannel",
    "ProviderDeltaSink",
    "StreamChunk",
    "current_llm_budget_ledger",
    "llm_budget_scope",
    "model_request_input_text",
    "structured_accounted_prompt",
]
