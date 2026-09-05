from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from agent_runtime.budget import BudgetLimitExceededError, ResourceUsage
from agent_runtime.budget.pricing import (
    PricingUnavailableError,
    actual_model_cost_micros,
    estimated_model_cost_micros,
    pricing_micros_per_1m,
    pricing_revision,
)
from agent_runtime.core.llm_registry import ResolvedModel
from agent_runtime.core.messages import StopReason, ToolUseResult
from agent_runtime.harness.model_adapter import GatewayHarnessModel
from agent_runtime.harness.protocol import HarnessMessage, HarnessModelRequest
from agent_runtime.harness.rollout import RolloutStore
from agent_runtime.model_definition import ModelCapabilities, RequestDefaultsDefinition
from agent_runtime.modeling.config import GenerationConfig
from agent_runtime.modeling.contracts import (
    LLMCallStage,
    LLMStageBudget,
    normalize_llm_usage,
)
from agent_runtime.modeling.gateway import AgentModelResponse, LLMGateway
from agent_runtime.modeling.openai_wire import serialize_openai_request
from agent_runtime.modeling.quota import ProviderQuotaPreflightError


class CharacterAccounting:
    def count(self, text: str) -> int:
        return len(text)

    def clip(
        self, text: str, token_budget: int, *, add_ellipsis: bool = False
    ) -> str:
        del add_ellipsis
        return text[:token_budget]


class CapturingGateway:
    def __init__(self) -> None:
        self.requests: list[object] = []

    def effective_stage_budget(
        self, stage: LLMCallStage, *, kwargs: object = None
    ) -> LLMStageBudget:
        del stage, kwargs
        return LLMStageBudget(
            max_input_tokens=4096,
            max_output_tokens=100,
            safety_margin_tokens=0,
        )

    async def agenerate_model_request(self, **kwargs: object) -> AgentModelResponse:
        request = kwargs["request"]
        self.requests.append(request)
        return AgentModelResponse(
            turn=ToolUseResult(
                text="done",
                stop_reason=StopReason.END_TURN,
                raw_stop_reason="stop",
            ),
            usage=normalize_llm_usage(
                input_tokens=1000,
                output_tokens=500,
                input_tokens_include_cache=True,
                usage_source="provider",
            ),
            provider_wire_hash=serialize_openai_request(request).provider_wire_hash,
            serializer_revision="openai-wire-v1",
            wire_kind="openai-compatible",
        )


class RejectingQuotaGate:
    def __init__(self) -> None:
        self.calls = 0

    async def acquire(
        self, *, provider: str, model: str, requested_tokens: int
    ) -> None:
        del provider, model, requested_tokens
        self.calls += 1
        raise ProviderQuotaPreflightError("quota exhausted")


class NeverCalledProvider:
    def __init__(self) -> None:
        self.calls = 0

    def generate_with_tools(self, **_kwargs: object) -> object:
        self.calls += 1
        raise AssertionError("provider I/O must not start")


def _resolved(
    gateway: object, *, pricing: dict[str, int | None]
) -> ResolvedModel:
    revision = pricing_revision(
        provider="openai-compatible",
        model="provider-model",
        pricing=pricing,
    )
    return ResolvedModel(
        generator=object(),
        gateway=gateway,  # type: ignore[arg-type]
        model="provider-model",
        provider="openai-compatible",
        capabilities=ModelCapabilities(
            context_window_tokens=8192,
            max_context_window_tokens=8192,
            max_output_tokens=100,
            supports_native_tools=True,
            supports_structured_output=True,
        ),
        token_accounting=CharacterAccounting(),  # type: ignore[arg-type]
        request_defaults=RequestDefaultsDefinition(),
        generation_config=GenerationConfig(),
        pricing_micros_per_1m=pricing,
        pricing_revision=revision,
    )


def test_integer_pricing_and_revision_are_deterministic() -> None:
    pricing = pricing_micros_per_1m(
        input_cost_per_1m=2.0,
        output_cost_per_1m=8.0,
        cache_read_cost_per_1m=None,
        cache_write_cost_per_1m=None,
    )
    assert estimated_model_cost_micros(
        input_tokens=1000,
        max_output_tokens=500,
        pricing=pricing,
    ) == 6000
    assert pricing_revision(
        provider="p", model="m", pricing=pricing
    ) == pricing_revision(
        provider="p", model="m", pricing=dict(reversed(tuple(pricing.items())))
    )


def test_actual_cost_uses_input_rate_for_unconfigured_cache_price() -> None:
    pricing = {
        "input": 2_000_000,
        "output": 8_000_000,
        "cache_read": None,
        "cache_write": None,
    }
    assert actual_model_cost_micros(
        {
            "input_tokens": 1000,
            "logical_input_tokens": 1000,
            "uncached_input_tokens": 700,
            "cache_read_input_tokens": 300,
            "output_tokens": 500,
        },
        pricing=pricing,
    ) == 6000


def test_cost_limit_is_enforced_by_durable_budget_state(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    with RolloutStore(tmp_path / "rollout.sqlite3") as store:
        thread = store.create_thread(workspace=workspace)
        turn = store.start_turn(
            thread_id=thread.thread_id,
            user_message="bounded",
            binding_manifest={
                "model_token_budget_total": 10000,
                "model_cost_budget_total_micros": 5000,
            },
        )
        operation = store.prepare_model_operation(
            turn_id=turn.turn_id,
            request_hash="r",
            context_hash="c",
            tool_hash="t",
            wire_hash="w",
            request_ref={"request_id": "req"},
        )
        with pytest.raises(BudgetLimitExceededError) as raised:
            store.dispatch_model_attempt(
                operation.operation_id,
                resource_request=ResourceUsage(
                    input_tokens=10,
                    cost_micros=5001,
                    model_calls=1,
                ),
                allow_protected_budget=True,
            )
        assert raised.value.resource == "cost_micros"


def test_adapter_records_reserved_and_actual_versioned_cost() -> None:
    pricing = {
        "input": 2_000_000,
        "output": 8_000_000,
        "cache_read": None,
        "cache_write": None,
    }
    gateway = CapturingGateway()
    model = GatewayHarnessModel(
        model_alias="priced",
        resolved=_resolved(gateway, pricing=pricing),
        instructions=("Answer.",),
    )
    prepared = model.prepare(
        HarnessModelRequest(
            thread_id="thread",
            turn_id="turn",
            messages=(HarnessMessage(role="user", content="hello"),),
            binding_manifest={"model_cost_budget_total_micros": 1_000_000},
        )
    )
    assert prepared.resource_request.cost_micros > 0
    assert prepared.request_ref["pricing_known"] is True
    response = asyncio.run(model.dispatch(prepared))
    assert response.usage["cost_micros"] == 6000
    assert response.usage["pricing_revision"] == prepared.request_ref["pricing_revision"]


def test_cost_limited_turn_rejects_unknown_pricing_before_provider_io() -> None:
    gateway = CapturingGateway()
    model = GatewayHarnessModel(
        model_alias="unpriced",
        resolved=_resolved(gateway, pricing={}),
        instructions=("Answer.",),
    )
    with pytest.raises(PricingUnavailableError):
        model.prepare(
            HarnessModelRequest(
                thread_id="thread",
                turn_id="turn",
                messages=(HarnessMessage(role="user", content="hello"),),
                binding_manifest={"model_cost_budget_total_micros": 1000},
            )
        )
    assert gateway.requests == []


def test_provider_quota_gate_rejects_before_provider_io() -> None:
    provider = NeverCalledProvider()
    gate = RejectingQuotaGate()
    gateway = LLMGateway(
        generator=provider,
        token_accounting=CharacterAccounting(),
        model_context_tokens=8192,
        stage_budgets={
            LLMCallStage.AGENT_STEP: LLMStageBudget(
                max_input_tokens=4096,
                max_output_tokens=100,
                safety_margin_tokens=0,
            )
        },
        provider_quota_gate=gate,
    )
    from agent_runtime.core.model_request import (
        ModelSettings,
        build_model_request,
        build_stable_context,
    )

    request = build_model_request(
        request_id="quota-test",
        context=build_stable_context(
            instructions=("Answer.",),
            initial_user_task="hello",
        ),
        selected_tools=(),
        settings=ModelSettings(model="provider-model", max_output_tokens=100),
    )
    with pytest.raises(ProviderQuotaPreflightError):
        asyncio.run(
            gateway.agenerate_model_request(
                stage=LLMCallStage.AGENT_STEP,
                request=request,
                provider="openai-compatible",
                supports_native_tools=True,
            )
        )
    assert gate.calls == 1
    assert provider.calls == 0
