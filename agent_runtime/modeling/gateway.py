from __future__ import annotations

import asyncio
import json
import queue as thread_queue
import threading
from collections.abc import AsyncGenerator, Callable, Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, replace
from typing import Any, Protocol, cast
from uuid import uuid4

from pydantic import BaseModel

from agent_runtime.core.messages import (
    StopReason,
    ToolUseResult,
    decode_tool_arguments,
    outputs_from_parts,
    outputs_tool_calls,
)
from agent_runtime.core.model_request import ModelRequest
from agent_runtime.modeling.contracts import (
    DEFAULT_LLM_STAGE_BUDGETS,
    ContextCapacityPolicy,
    LLMCallResult,
    LLMCallStage,
    LLMProviderResult,
    LLMStageBudget,
    LLMUsage,
    normalize_llm_usage,
)
from agent_runtime.modeling.events import (
    ModelEventSink,
    ModelOutput,
    ModelOutputCompleted,
    ModelOutputDelta,
    ModelOutputKind,
    ModelOutputStarted,
    ModelResponseCompleted,
    ModelResponseStarted,
    ModelStreamState,
    emit_model_event,
    validate_model_outputs,
)
from agent_runtime.modeling.local_agent_wire import (
    parse_local_agent_response,
    render_local_agent_request,
)
from agent_runtime.modeling.openai_wire import (
    parse_openai_response,
    parse_openai_usage,
    serialize_openai_request,
)
from agent_runtime.modeling.quota import (
    ProviderQuotaGate,
    acquire_provider_quota,
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
    output_id: str = ""
    output_kind: str = ""
    block_index: int | None = None
    provider_response_id: str | None = None
    provider_data: Mapping[str, Any] | None = None


_PROVIDER_BRIDGE_MAXSIZE = 64
_PROVIDER_BRIDGE_POLL_SECONDS = 0.005
_PROVIDER_BRIDGE_PUT_TIMEOUT_SECONDS = 0.05
_DEFAULT_CONTEXT_POLICY = ContextCapacityPolicy()


@dataclass(frozen=True, slots=True)
class AgentModelResponse:
    outputs: tuple[ModelOutput, ...]
    stop_reason: StopReason
    raw_stop_reason: str
    usage: LLMUsage
    provider_wire_hash: str
    serializer_revision: str
    wire_kind: str
    provider_response_id: str | None = None

    def __post_init__(self) -> None:
        validate_model_outputs(self.outputs)
        if not isinstance(self.stop_reason, StopReason):
            raise TypeError("model stop_reason must be a StopReason")

    @property
    def turn(self) -> ToolUseResult:
        return ToolUseResult(
            text="".join(output.content for output in self.outputs if output.kind in {"text", "refusal"}),
            reasoning_content="".join(output.content for output in self.outputs if output.kind == "reasoning") or None,
            tool_calls=list(outputs_tool_calls(self.outputs)) if self.stop_reason is StopReason.TOOL_USE else [],
            stop_reason=self.stop_reason,
            raw_stop_reason=self.raw_stop_reason,
        )

    @classmethod
    def from_turn(
        cls,
        *,
        turn: ToolUseResult,
        usage: LLMUsage,
        provider_wire_hash: str,
        serializer_revision: str,
        wire_kind: str,
        provider_response_id: str | None = None,
    ) -> AgentModelResponse:
        """Adapt flat provider-parser fields into the canonical output snapshot."""
        return cls(
            outputs=outputs_from_parts(
                text=turn.text,
                reasoning_content=turn.reasoning_content,
                tool_calls=tuple(turn.tool_calls),
            ),
            stop_reason=turn.stop_reason,
            raw_stop_reason=turn.raw_stop_reason,
            usage=usage,
            provider_wire_hash=provider_wire_hash,
            serializer_revision=serializer_revision,
            wire_kind=wire_kind,
            provider_response_id=provider_response_id,
        )


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
        context_policy: ContextCapacityPolicy | None = _DEFAULT_CONTEXT_POLICY,
        agent_output_policy: str | None = None,
        model_max_output_tokens: int | None = None,
    ) -> None:
        if model_context_tokens <= 0:
            raise ValueError("model_context_tokens must be positive")
        self._generator = generator
        self._token_accounting = token_accounting
        self._model_context_tokens = model_context_tokens
        self._stage_budgets = dict(stage_budgets or DEFAULT_LLM_STAGE_BUDGETS)
        self._provider_quota_gate = provider_quota_gate
        self.context_policy = context_policy
        self.agent_output_policy = agent_output_policy
        self._model_max_output_tokens = model_max_output_tokens

    @property
    def model_context_tokens(self) -> int:
        return self._model_context_tokens

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
        if stage == LLMCallStage.AGENT_STEP and self.agent_output_policy == "model-aware-v1":
            requested = (kwargs or {}).get("max_tokens")
            if type(requested) is int and requested > 0:
                max_output_tokens = requested
            if self._model_max_output_tokens is not None:
                max_output_tokens = min(max_output_tokens, self._model_max_output_tokens)
            max_output_tokens = min(max_output_tokens, self._model_context_tokens - budget.safety_margin_tokens - 1)
        hard_input = self._model_context_tokens - max_output_tokens - budget.safety_margin_tokens
        max_input_tokens = hard_input if budget.max_input_tokens is None else min(budget.max_input_tokens, hard_input)
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
        event_sink: ModelEventSink | None = None,
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
            response = AgentModelResponse.from_turn(
                turn=turn,
                usage=usage,
                provider_wire_hash=local_wire.provider_wire_hash,
                serializer_revision=local_wire.serializer_revision,
                wire_kind=provider,
            )
            if stream:
                await _emit_complete_model_response(event_sink, response)
            return response

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
            assembled = _ModelChunkAssembler(event_sink)
            async for chunk in chunks:
                await assembled.accept(chunk)
            if assembled.usage is None or not assembled.state.completed:
                raise RuntimeError("canonical streaming call ended without a terminal response and usage")
            return AgentModelResponse(
                stop_reason=_normalized_stop(assembled.stop_reason),
                raw_stop_reason=assembled.stop_reason,
                usage=assembled.usage,
                provider_wire_hash=wire.provider_wire_hash,
                serializer_revision=wire.serializer_revision,
                wire_kind="openai",
                outputs=assembled.state.outputs,
                provider_response_id=assembled.state.provider_response_id,
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
                output_tokens=sum(
                    self._token_accounting.count(part) for part in _openai_output_parts(provider_result.value)
                ),
                input_tokens_include_cache=True,
                usage_source="tokenizer_estimate",
            )
        )
        if effective_ledger is not None:
            await effective_ledger.commit(effective_lease_id, usage.total_tokens)
        return AgentModelResponse(
            stop_reason=turn.stop_reason,
            raw_stop_reason=turn.raw_stop_reason,
            usage=usage,
            provider_wire_hash=wire.provider_wire_hash,
            serializer_revision=wire.serializer_revision,
            wire_kind="openai",
            outputs=_outputs_from_chat_response(provider_result.value, turn),
            provider_response_id=_provider_field(provider_result.value, "id"),
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
        effective_lease_id = lease_id or f"{stage.value}:{uuid4().hex}"
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
        effective_lease_id = lease_id or f"{stage.value}:{uuid4().hex}"
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
        effective_lease_id = lease_id or f"{stage.value}:stream:{uuid4().hex}"
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
        output_parts: dict[tuple[str, str], list[str]] = {}
        active_tool = ""
        anonymous_tools = 0
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
                if chunk.type in {"text_delta", "thinking_delta", "plan_delta"}:
                    output_parts.setdefault((chunk.type, ""), []).append(chunk.content)
                elif chunk.type == "tool_use_start":
                    anonymous_tools += 1
                    active_tool = chunk.tool_id or f"anonymous:{anonymous_tools}"
                    output_parts.setdefault(("tool_name", active_tool), []).append(chunk.tool_name)
                elif chunk.type == "tool_input_delta":
                    output_parts.setdefault(("tool_input", chunk.tool_id or active_tool), []).append(chunk.content)
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

        if final_stop is None:
            if effective_ledger is not None:
                await effective_ledger.refund(effective_lease_id)
            raise ConnectionError("provider stream ended without a terminal stop")

        # 提交预算
        usage = reported_usage or normalize_llm_usage(
            input_tokens=input_tokens,
            output_tokens=sum(self._token_accounting.count("".join(parts)) for parts in output_parts.values()),
            input_tokens_include_cache=True,
            usage_source="tokenizer_estimate",
        )
        if effective_ledger is not None:
            await effective_ledger.commit(effective_lease_id, usage.total_tokens)
        yield replace(
            final_stop,
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
        if stage == LLMCallStage.CONTEXT_COMPACTION and stage not in self._stage_budgets:
            # Compaction consumes agent history. The generic summarize input
            # limit describes a different task and can cause needless fan-out.
            # New Turns independently use model capacity; legacy definitions
            # retain the agent cap and the configured summary output allowance.
            agent = self._stage_budget(LLMCallStage.AGENT_STEP)
            summary = self._stage_budget(LLMCallStage.LLM_SUMMARIZE)
            return summary.model_copy(update={
                "max_input_tokens": agent.max_input_tokens if self.context_policy is None else None,
                "safety_margin_tokens": max(agent.safety_margin_tokens, summary.safety_margin_tokens),
            })
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
        assert max_input_tokens is not None  # effective_stage_budget resolves optional caps
        count = getattr(self._token_accounting, "count_for_budget", None) or self._token_accounting.count
        input_tokens = count(prompt)
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
            turn = parse_openai_response(raw)
            outputs = _outputs_from_chat_response(raw, turn)
            if not put(StreamChunk(
                type="response_start", provider_response_id=_provider_field(raw, "id"),
            )):
                return
            delta_types = {
                "text": "text_delta", "reasoning": "thinking_delta", "refusal": "refusal_delta",
            }
            for output in outputs:
                if not put(StreamChunk(
                    type="tool_use_start" if output.kind == "tool_call" else "content_block_start",
                    output_id=output.output_id,
                    output_kind=output.kind,
                    tool_id=output.tool_call_id or "",
                    tool_name=output.tool_name or "",
                    provider_data=output.provider_data,
                )):
                    return
                if output.content and not put(StreamChunk(
                    type="tool_input_delta" if output.kind == "tool_call" else delta_types[output.kind],
                    output_id=output.output_id,
                    content=output.content,
                )):
                    return
                if not put(StreamChunk(type="content_block_stop", output_id=output.output_id)):
                    return
            put(StreamChunk(
                type="message_stop",
                stop_reason=turn.raw_stop_reason,
                usage=result.usage,
            ))


def _openai_output_parts(response: object) -> tuple[str, ...]:
    """Visible generated channels, including tool fragments from incomplete steps."""
    def field(value: object, name: str, default: object = None) -> Any:
        return value.get(name, default) if isinstance(value, Mapping) else getattr(value, name, default)

    choices = field(response, "choices", ())
    if not choices:
        return ()
    message = field(choices[0], "message")
    parts = [field(message, "content", ""),
             field(message, "reasoning_content") or field(message, "reasoning", "")]
    for call in field(message, "tool_calls", ()) or ():
        function = field(call, "function")
        parts.extend((field(function, "name", ""), field(function, "arguments", "")))
    return tuple(part if isinstance(part, str) else json.dumps(part, ensure_ascii=False) for part in parts if part)


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


def _normalized_stop(stop_reason: str) -> StopReason:
    return (
        StopReason.MAX_TOKENS if stop_reason in {"max_tokens", "length"}
        else StopReason.TOOL_USE if stop_reason in {"tool_use", "tool_calls"}
        else StopReason.END_TURN if stop_reason in {"end_turn", "stop"}
        else StopReason.INCOMPLETE
    )


def _provider_field(value: object, name: str, default: Any = None) -> Any:
    return value.get(name, default) if isinstance(value, Mapping) else getattr(value, name, default)


def _outputs_from_chat_response(raw: object, turn: ToolUseResult) -> tuple[ModelOutput, ...]:
    """Preserve available Chat output objects, including unexecuted tool input."""
    choices = _provider_field(raw, "choices", ())
    message = _provider_field(choices[0], "message")
    outputs = []
    if turn.reasoning_content:
        opaque = _provider_field(message, "reasoning_details")
        outputs.append(ModelOutput("reasoning", "reasoning", turn.reasoning_content,
                                   provider_data={"reasoning_details": opaque} if opaque is not None else {}))
    if turn.text:
        outputs.append(ModelOutput("text", "text", turn.text))
    refusal = _provider_field(message, "refusal")
    if isinstance(refusal, str) and refusal:
        outputs.append(ModelOutput("refusal", "refusal", refusal))
    for index, raw_call in enumerate(_provider_field(message, "tool_calls", ()) or (), start=1):
        function = _provider_field(raw_call, "function")
        call_id = _provider_field(raw_call, "id") or f"call_{index}"
        name = _provider_field(function, "name", "")
        raw_arguments = _provider_field(function, "arguments", "{}")
        arguments, error = decode_tool_arguments(raw_arguments)
        content = raw_arguments if isinstance(raw_arguments, str) else json.dumps(raw_arguments, ensure_ascii=False)
        outputs.append(
            ModelOutput(
                f"tool:{call_id}",
                "tool_call",
                content,
                tool_call_id=call_id,
                tool_name=name,
                arguments=arguments,
                argument_error=error,
            )
        )
    return tuple(outputs)


def _response_terminal(response: AgentModelResponse) -> ModelResponseCompleted:
    incomplete = response.stop_reason in {StopReason.MAX_TOKENS, StopReason.INCOMPLETE}
    return ModelResponseCompleted(
        provider_response_id=response.provider_response_id,
        usage=response.usage.model_dump(mode="json"),
        stop_reason=response.raw_stop_reason or response.stop_reason.value,
        status="incomplete" if incomplete else "completed",
        incomplete_reason=(
            "max_output_tokens"
            if response.stop_reason is StopReason.MAX_TOKENS
            else response.raw_stop_reason or "unknown"
        )
        if incomplete
        else None,
    )


async def _emit_complete_model_response(sink: ModelEventSink | None, response: AgentModelResponse) -> None:
    await emit_model_event(sink, ModelResponseStarted(response.provider_response_id))
    for output in response.outputs:
        await emit_model_event(sink, ModelOutputStarted(replace(output, content="", arguments=None)))
        if output.content:
            await emit_model_event(sink, ModelOutputDelta(output.output_id, output.delta_kind, output.content))
        await emit_model_event(sink, ModelOutputCompleted(output))
    await emit_model_event(sink, _response_terminal(response))


class _ModelChunkAssembler:
    """Translate provider chunks into one validated, ordered model stream."""

    def __init__(self, sink: ModelEventSink | None) -> None:
        self.sink = sink
        self.state = ModelStreamState()
        self.usage: LLMUsage | None = None
        self.stop_reason = "unknown"
        self._blocks: dict[int, str] = {}
        self._tool_outputs: dict[str, str] = {}
        self._active: str | None = None
        self._active_tool: str | None = None
        self._closed: set[str] = set()

    async def _emit(self, event: Any) -> None:
        self.state.accept(event)
        await emit_model_event(self.sink, event)

    def _get(self, output_id: str) -> ModelOutput:
        return next(output for output in self.state.outputs if output.output_id == output_id)

    def _identity(self, chunk: StreamChunk, kind: str) -> str:
        references = [chunk.output_id] if chunk.output_id else []
        if chunk.block_index is not None and chunk.block_index in self._blocks:
            references.append(self._blocks[chunk.block_index])
        if chunk.tool_id in self._tool_outputs:
            references.append(self._tool_outputs[chunk.tool_id])
        if len(set(references)) > 1:
            raise ValueError("provider output identity references disagree")
        if references:
            return references[0]
        if chunk.block_index is not None:
            return f"block:{chunk.block_index}"
        if chunk.tool_id:
            return f"tool:{chunk.tool_id}"
        if kind == "tool_call" and self._active_tool is not None:
            return self._active_tool
        candidate = kind
        existing = {output.output_id for output in self.state.outputs}
        index = 1
        while candidate in self._closed:
            index += 1
            candidate = f"{kind}:{index}"
        return candidate if candidate in existing or kind != "tool_call" else f"tool:{len(existing) + 1}"

    async def _start(self, chunk: StreamChunk, kind: str) -> str:
        output_id = self._identity(chunk, kind)
        existing = {output.output_id for output in self.state.outputs}
        if output_id not in existing:
            await self._emit(
                ModelOutputStarted(
                    ModelOutput(
                        output_id,
                        cast(ModelOutputKind, kind),
                        tool_call_id=(chunk.tool_id or output_id) if kind == "tool_call" else None,
                        tool_name=chunk.tool_name if kind == "tool_call" else None,
                        provider_data=chunk.provider_data or {},
                    )
                )
            )
        elif output_id in self._closed:
            raise ValueError("provider reused a completed output identity")
        elif self._get(output_id).kind != kind:
            raise ValueError("provider output kind changed")
        if chunk.block_index is not None:
            self._blocks[chunk.block_index] = output_id
        self._active = output_id
        if kind == "tool_call":
            call_id = self._get(output_id).tool_call_id
            if call_id is not None:
                self._tool_outputs[call_id] = output_id
            self._active_tool = output_id
        return output_id

    async def _finish(self, output_id: str, *, complete_tool: bool) -> None:
        output = self._get(output_id)
        if output.kind == "tool_call" and complete_tool:
            arguments, error = decode_tool_arguments(output.content)
            output = replace(output, arguments=arguments, argument_error=error)
        await self._emit(ModelOutputCompleted(output))
        self._closed.add(output_id)
        if self._active_tool == output_id:
            self._active_tool = None

    async def accept(self, chunk: StreamChunk) -> None:
        if not self.state.started:
            await self._emit(ModelResponseStarted(chunk.provider_response_id))
        if chunk.provider_response_id:
            if self.state.provider_response_id and self.state.provider_response_id != chunk.provider_response_id:
                raise ValueError("provider changed response identity")
            self.state.provider_response_id = chunk.provider_response_id
        kinds = {
            "text_delta": "text",
            "thinking_delta": "reasoning",
            "reasoning_summary_delta": "reasoning_summary",
            "plan_delta": "plan",
            "refusal_delta": "refusal",
        }
        if chunk.type in kinds:
            kind = kinds[chunk.type]
            output_id = await self._start(chunk, kind)
            await self._emit(ModelOutputDelta(output_id, self._get(output_id).delta_kind, chunk.content))
        elif chunk.type in {"content_block_start", "output_start", "tool_use_start"}:
            kind = "tool_call" if chunk.type == "tool_use_start" or chunk.tool_name else chunk.output_kind or "text"
            output_id = await self._start(chunk, kind)
            if chunk.content:
                await self._emit(ModelOutputDelta(output_id, self._get(output_id).delta_kind, chunk.content))
        elif chunk.type == "tool_input_delta":
            output_id = self._identity(chunk, "tool_call")
            await self._emit(ModelOutputDelta(output_id, "tool_arguments", chunk.content))
        elif chunk.type == "signature_delta":
            signature_output_id = chunk.output_id or (
                self._blocks.get(chunk.block_index) if chunk.block_index is not None else self._active
            )
            if signature_output_id is None:
                raise ValueError("signature delta has no active output")
            await self._emit(ModelOutputDelta(signature_output_id, "signature", chunk.content))
        elif chunk.type in {"content_block_stop", "output_stop"}:
            stopping_output_id = (
                self._identity(chunk, "tool_call")
                if chunk.output_id or chunk.block_index is not None or chunk.tool_id
                else self._active_tool or self._active
            )
            if stopping_output_id is None:
                raise ValueError("provider output stop has no active output")
            await self._finish(stopping_output_id, complete_tool=True)
        elif chunk.type == "message_stop":
            self.stop_reason = chunk.stop_reason or "unknown"
            self.usage = chunk.usage
            if self.usage is None:
                raise RuntimeError("canonical streaming call ended without usage")
            for output in self.state.outputs:
                if output.output_id not in self._closed:
                    await self._finish(output.output_id, complete_tool=False)
            incomplete = _normalized_stop(self.stop_reason) in {StopReason.MAX_TOKENS, StopReason.INCOMPLETE}
            await self._emit(
                ModelResponseCompleted(
                    provider_response_id=self.state.provider_response_id,
                    usage=self.usage.model_dump(mode="json"),
                    stop_reason=self.stop_reason,
                    status="incomplete" if incomplete else "completed",
                    incomplete_reason=(
                        "max_output_tokens"
                        if _normalized_stop(self.stop_reason) is StopReason.MAX_TOKENS
                        else self.stop_reason
                    )
                    if incomplete
                    else None,
                )
            )
        elif chunk.type not in {"message_start", "response_start"}:
            raise ValueError(f"unsupported provider chunk type: {chunk.type}")


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


def _account_messages(
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
) -> str:
    """Approximate token-countable text from messages + tools."""
    prompt = _render_messages_as_prompt(messages)
    if tools:
        tool_desc = json.dumps(tools, ensure_ascii=False, separators=(",", ":"))
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
    "StreamChunk",
    "current_llm_budget_ledger",
    "llm_budget_scope",
    "model_request_input_text",
    "structured_accounted_prompt",
]
