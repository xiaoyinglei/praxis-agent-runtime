from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from agent_runtime.core.llm_registry import ResolvedModel
from agent_runtime.core.messages import StopReason, ToolUseResult
from agent_runtime.harness import (
    CompletionDecision,
    CompletionProposal,
    ControlPlaneHarnessModel,
    GatewayHarnessModel,
    HarnessAgent,
    HarnessMessage,
    HarnessModelRequest,
    ModelDispatchPreflightError,
    RolloutContextManager,
    RolloutStore,
    RuntimeComposition,
)
from agent_runtime.modeling.contracts import (
    LLMCallStage,
    LLMStageBudget,
    normalize_llm_usage,
)
from agent_runtime.modeling.gateway import (
    AgentModelResponse,
    LLMBudgetExceededError,
    model_request_input_text,
)
from agent_runtime.modeling.openai_wire import serialize_openai_request
from agent_runtime.models import (
    ModelCatalog,
    ModelControlPlane,
    ModelSessionState,
    ModelSpec,
)


class CapturingGateway:
    def __init__(self) -> None:
        self.requests: list[object] = []
        self.stages: list[LLMCallStage] = []

    async def agenerate_model_request(self, **kwargs: object) -> AgentModelResponse:
        request = kwargs["request"]
        self.requests.append(request)
        self.stages.append(kwargs["stage"])  # type: ignore[arg-type]
        return AgentModelResponse(
            turn=ToolUseResult(
                text="real gateway answer",
                stop_reason=StopReason.END_TURN,
                raw_stop_reason="stop",
            ),
            usage=normalize_llm_usage(
                input_tokens=3,
                output_tokens=2,
                input_tokens_include_cache=True,
                usage_source="provider",
            ),
            provider_wire_hash=serialize_openai_request(request).provider_wire_hash,
            serializer_revision="openai-wire-v1",
            wire_kind="openai-compatible",
        )


class MaxTokensGateway(CapturingGateway):
    async def agenerate_model_request(self, **kwargs: object) -> AgentModelResponse:
        request = kwargs["request"]
        self.requests.append(request)
        self.stages.append(kwargs["stage"])  # type: ignore[arg-type]
        return AgentModelResponse(
            turn=ToolUseResult(
                text="partial answer",
                stop_reason=StopReason.MAX_TOKENS,
                raw_stop_reason="length",
            ),
            usage=normalize_llm_usage(
                input_tokens=7,
                output_tokens=256,
                input_tokens_include_cache=True,
                usage_source="provider",
            ),
            provider_wire_hash=serialize_openai_request(request).provider_wire_hash,
            serializer_revision="openai-wire-v1",
            wire_kind="openai-compatible",
        )


class BudgetRejectingGateway:
    def __init__(self) -> None:
        self.provider_calls = 0

    async def agenerate_model_request(self, **kwargs: object) -> AgentModelResponse:
        ledger = kwargs["ledger"]
        lease_id = kwargs["lease_id"]
        assert hasattr(ledger, "reserve")
        assert isinstance(lease_id, str)
        if not await ledger.reserve(lease_id, 11):
            raise LLMBudgetExceededError(
                stage=LLMCallStage.FINAL_SYNTHESIS,
                required_tokens=11,
            )
        self.provider_calls += 1
        raise AssertionError("provider I/O must not begin")


class CharacterAccounting:
    def count(self, text: str) -> int:
        return len(text)

    def clip(
        self,
        text: str,
        token_budget: int,
        *,
        add_ellipsis: bool = False,
    ) -> str:
        suffix = "..." if add_ellipsis and len(text) > token_budget else ""
        return text[: max(0, token_budget - len(suffix))] + suffix


class BudgetAwareCapturingGateway(CapturingGateway):
    def __init__(self, *, max_input_tokens: int) -> None:
        super().__init__()
        self.token_accounting = CharacterAccounting()
        self.max_input_tokens = max_input_tokens

    def effective_stage_budget(
        self,
        stage: LLMCallStage,
        *,
        kwargs: object = None,
    ) -> LLMStageBudget:
        del stage, kwargs
        return LLMStageBudget(
            max_input_tokens=self.max_input_tokens,
            max_output_tokens=256,
            safety_margin_tokens=0,
        )

    async def agenerate_model_request(self, **kwargs: object) -> AgentModelResponse:
        request = kwargs["request"]
        provider = kwargs["provider"]
        supports_native_tools = kwargs["supports_native_tools"]
        assert self.token_accounting.count(
            model_request_input_text(
                request,
                provider=provider,
                supports_native_tools=supports_native_tools,
            )
        ) <= self.max_input_tokens
        return await super().agenerate_model_request(**kwargs)


def test_gateway_adapter_prepares_canonical_wire_before_provider_io() -> None:
    gateway = CapturingGateway()
    resolved = ResolvedModel(
        generator=object(),
        kwargs={"max_tokens": 256, "temperature": 0.0},
        context_window_tokens=8_192,
        gateway=gateway,
        provider="openai-compatible",
        model="provider-model",
        supports_native_tools=True,
    )
    model = GatewayHarnessModel(
        model_alias="test-model",
        resolved=resolved,
        instructions=("Answer the user directly.",),
    )
    request = HarnessModelRequest(
        thread_id="thread-1",
        turn_id="turn-1",
        messages=(HarnessMessage(role="user", content="hello"),),
        binding_manifest={"model_alias": "test-model"},
    )

    prepared = model.prepare(request)

    assert gateway.requests == []
    assert len(prepared.request_hash) == 64
    assert len(prepared.context_hash) == 64
    assert len(prepared.tool_hash) == 64
    assert prepared.wire_hash.startswith("wire_")
    assert prepared.request_ref["model_alias"] == "test-model"

    response = asyncio.run(model.dispatch(prepared))

    assert len(gateway.requests) == 1
    assert gateway.stages == [LLMCallStage.AGENT_STEP]
    assert response.text == "real gateway answer"
    assert response.provider_response_id is None
    assert response.usage["input_tokens"] == 3
    assert response.usage["output_tokens"] == 2


def test_gateway_adapter_returns_known_incomplete_response_on_max_tokens() -> None:
    gateway = MaxTokensGateway()
    resolved = ResolvedModel(
        generator=object(),
        kwargs={"max_tokens": 256, "temperature": 0.0},
        context_window_tokens=8_192,
        gateway=gateway,
        provider="openai-compatible",
        model="provider-model",
        supports_native_tools=True,
    )
    model = GatewayHarnessModel(
        model_alias="test-model",
        resolved=resolved,
        instructions=("Answer the user directly.",),
    )
    prepared = model.prepare(
        HarnessModelRequest(
            thread_id="thread-1",
            turn_id="turn-1",
            messages=(HarnessMessage(role="user", content="hello"),),
            binding_manifest={"model_alias": "test-model"},
        )
    )

    response = asyncio.run(model.dispatch(prepared))

    assert gateway.stages == [LLMCallStage.AGENT_STEP]
    assert response.status == "incomplete"
    assert response.incomplete_reason == "max_output_tokens"
    assert response.text == "partial answer"
    assert response.usage["output_tokens"] == 256


def test_gateway_adapter_compacts_transcript_before_durable_provider_dispatch() -> None:
    gateway = BudgetAwareCapturingGateway(max_input_tokens=9_000)
    resolved = ResolvedModel(
        generator=object(),
        kwargs={"max_tokens": 256, "temperature": 0.0},
        context_window_tokens=32_768,
        gateway=gateway,
        token_accounting=gateway.token_accounting,
        provider="openai-compatible",
        model="provider-model",
        supports_native_tools=True,
    )
    model = GatewayHarnessModel(
        model_alias="test-model",
        resolved=resolved,
        instructions=("Answer the user directly.",),
    )
    messages = (
        HarnessMessage(role="user", content="fix the package"),
        *tuple(
            HarnessMessage(
                role="assistant",
                content=f"exploration-{index}:" + ("x" * 3_000),
            )
            for index in range(8)
        ),
        HarnessMessage(role="assistant", content="latest finding must survive"),
    )

    prepared = model.prepare(
        HarnessModelRequest(
            thread_id="thread-1",
            turn_id="turn-1",
            messages=messages,
            binding_manifest={"model_alias": "test-model"},
        )
    )

    projection = prepared.request_ref["context_projection"]
    assert projection["compacted"] is True
    assert projection["input_tokens"] <= projection["max_input_tokens"]
    response = asyncio.run(model.dispatch(prepared))
    assert response.text == "real gateway answer"
    wire = serialize_openai_request(gateway.requests[0]).serialized_json
    assert "latest finding must survive" in wire
    assert "context_compaction" in wire
    assert len(wire) <= gateway.max_input_tokens


def test_unchanged_stable_prefix_keeps_identical_provider_wire_bytes() -> None:
    gateway = CapturingGateway()
    model = GatewayHarnessModel(
        model_alias="test-model",
        resolved=ResolvedModel(
            generator=object(),
            kwargs={"max_tokens": 256, "temperature": 0.0},
            context_window_tokens=8_192,
            gateway=gateway,
            provider="openai-compatible",
            model="provider-model",
            supports_native_tools=True,
        ),
        instructions=("Stable system instruction.",),
    )
    first = model.prepare(
        HarnessModelRequest(
            thread_id="thread-1",
            turn_id="turn-1",
            messages=(HarnessMessage(role="user", content="stable initial task"),),
            binding_manifest={"model_alias": "test-model"},
        )
    )
    asyncio.run(model.dispatch(first))
    second = model.prepare(
        HarnessModelRequest(
            thread_id="thread-1",
            turn_id="turn-1",
            messages=(
                HarnessMessage(role="user", content="stable initial task"),
                HarnessMessage(role="assistant", content="dynamic transcript tail"),
            ),
            binding_manifest={"model_alias": "test-model"},
        )
    )
    asyncio.run(model.dispatch(second))

    first_wire = serialize_openai_request(gateway.requests[0])
    second_wire = serialize_openai_request(gateway.requests[1])
    first_messages = first_wire.payload["messages"]
    second_messages = second_wire.payload["messages"]
    assert isinstance(first_messages, tuple)
    assert isinstance(second_messages, tuple)
    assert second_messages[: len(first_messages)] == first_messages
    assert first.context_hash == second.context_hash
    assert first.request_ref["prompt_revision"] == second.request_ref["prompt_revision"]
    assert first.wire_hash != second.wire_hash


def test_gateway_adapter_enforces_remaining_budget_before_provider_io() -> None:
    gateway = BudgetRejectingGateway()
    model = GatewayHarnessModel(
        model_alias="test-model",
        resolved=ResolvedModel(
            generator=object(),
            kwargs={"max_tokens": 256},
            context_window_tokens=8_192,
            gateway=gateway,
            provider="openai-compatible",
            model="provider-model",
            supports_native_tools=True,
        ),
        instructions=("Answer the user directly.",),
    )
    prepared = model.prepare(
        HarnessModelRequest(
            thread_id="thread-1",
            turn_id="turn-1",
            messages=(HarnessMessage(role="user", content="hello"),),
            binding_manifest={"model_alias": "test-model"},
            model_token_budget_remaining=10,
        )
    )

    with pytest.raises(ModelDispatchPreflightError, match="remaining model token budget"):
        asyncio.run(model.dispatch(prepared))

    assert prepared.request_ref["model_token_budget_remaining"] == 10
    assert gateway.provider_calls == 0


def test_compaction_changes_the_actual_provider_wire_and_preserves_critical_facts(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    gateway = CapturingGateway()
    model = GatewayHarnessModel(
        model_alias="test-model",
        resolved=ResolvedModel(
            generator=object(),
            kwargs={"max_tokens": 256},
            context_window_tokens=8_192,
            gateway=gateway,
            provider="openai-compatible",
            model="provider-model",
            supports_native_tools=True,
        ),
        instructions=("Answer the user directly.",),
    )
    with RolloutStore(tmp_path / "rollout.sqlite3") as store:
        thread = store.create_thread(workspace=workspace)
        first = store.start_turn(
            thread_id=thread.thread_id,
            user_message="obsolete provider-visible question",
            binding_manifest={"model_alias": "test-model"},
        )
        store.complete_turn(turn_id=first.turn_id, answer="obsolete answer")
        second = store.start_turn(
            thread_id=thread.thread_id,
            user_message="current provider-visible question",
            binding_manifest={"model_alias": "test-model"},
        )
        manager = RolloutContextManager(store)
        before_messages = manager.build(second.turn_id)
        before = model.prepare(
            HarnessModelRequest(
                thread_id=thread.thread_id,
                turn_id=second.turn_id,
                messages=before_messages,
                binding_manifest={"model_alias": "test-model"},
            )
        )
        asyncio.run(model.dispatch(before))

        covered = tuple(item.item_id for item in store.list_context_items(second.turn_id)[:2])
        manager.compact(
            turn_id=second.turn_id,
            covered_item_ids=covered,
            summary="Compacted provider-visible summary.",
            preserved_facts={
                "architecture_and_safety_constraints": ["workspace only"],
                "file_changes": ["src/app.py changed"],
                "verification_results": ["test passed"],
                "unresolved_work": ["full suite pending"],
                "uncertain_side_effects": [],
            },
            context_version=2,
        )
        after_messages = manager.build(second.turn_id)
        after = model.prepare(
            HarnessModelRequest(
                thread_id=thread.thread_id,
                turn_id=second.turn_id,
                messages=after_messages,
                binding_manifest={"model_alias": "test-model"},
            )
        )
        asyncio.run(model.dispatch(after))

    before_wire = serialize_openai_request(gateway.requests[0]).serialized_json
    after_wire = serialize_openai_request(gateway.requests[1]).serialized_json
    assert "obsolete provider-visible question" in before_wire
    assert "obsolete provider-visible question" not in after_wire
    assert "obsolete answer" not in after_wire
    assert "Compacted provider-visible summary" in after_wire
    assert "workspace only" in after_wire
    assert "src/app.py changed" in after_wire
    assert "test passed" in after_wire
    assert "full suite pending" in after_wire
    assert "current provider-visible question" in after_wire
    assert before.context_hash != after.context_hash
    assert before.wire_hash != after.wire_hash


class ResolvedRegistry:
    fallback_model = None

    def __init__(self, models: dict[str, ResolvedModel]) -> None:
        self._models = models

    def resolve(self, alias: str) -> ResolvedModel:
        return self._models[alias]


def _spec(model_id: str) -> ModelSpec:
    return ModelSpec(
        id=model_id,
        provider="openai-compatible",
        provider_model=f"provider-{model_id}",
        context_window=8_192,
        supports_tools=True,
        supports_structured_output=False,
        location="cloud",
    )


def test_control_plane_adapter_dispatches_the_turns_frozen_model_binding() -> None:
    first_gateway = CapturingGateway()
    second_gateway = CapturingGateway()
    resolved = {
        "model-a": ResolvedModel(
            generator=object(),
            kwargs={"max_tokens": 256},
            gateway=first_gateway,
            provider="openai-compatible",
            model="provider-model-a",
        ),
        "model-b": ResolvedModel(
            generator=object(),
            kwargs={"max_tokens": 256},
            gateway=second_gateway,
            provider="openai-compatible",
            model="provider-model-b",
        ),
    }
    control_plane = ModelControlPlane(
        catalog=ModelCatalog(
            specs={model_id: _spec(model_id) for model_id in resolved},
            default_model_id="model-a",
        ),
        state=ModelSessionState(current_model_id="model-a"),
        registry=ResolvedRegistry(resolved),
    )
    model = ControlPlaneHarnessModel(
        control_plane=control_plane,
        instructions=("Answer the user directly.",),
    )
    frozen = model.snapshot()
    control_plane.switch_model("model-b", requested_by="user", persist=False)

    model.ensure_available(frozen)

    prepared = model.prepare(
        HarnessModelRequest(
            thread_id="thread-1",
            turn_id="turn-1",
            messages=(HarnessMessage(role="user", content="hello"),),
            binding_manifest=frozen,
        )
    )
    asyncio.run(model.dispatch(prepared))

    assert frozen["model_alias"] == "model-a"
    assert isinstance(frozen["model_catalog_revision"], str)
    assert len(first_gateway.requests) == 1
    assert second_gateway.requests == []


class AcceptAnswer:
    def evaluate(self, proposal: CompletionProposal) -> CompletionDecision:
        assert proposal.answer == "real gateway answer"
        return CompletionDecision(action="accept", reason="integration answer accepted")


def test_candidate_sdk_crosses_control_plane_gateway_and_rollout_store(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    gateway = CapturingGateway()
    resolved = ResolvedModel(
        generator=object(),
        kwargs={"max_tokens": 256},
        gateway=gateway,
        provider="openai-compatible",
        model="provider-model-a",
    )
    control_plane = ModelControlPlane(
        catalog=ModelCatalog(
            specs={"model-a": _spec("model-a")},
            default_model_id="model-a",
        ),
        state=ModelSessionState(current_model_id="model-a"),
        registry=ResolvedRegistry({"model-a": resolved}),
    )
    model = ControlPlaneHarnessModel(
        control_plane=control_plane,
        instructions=("Answer the user directly.",),
    )

    with RuntimeComposition.open(
        database=tmp_path / "rollout.sqlite3",
        workspace=workspace,
        model=model,
        completion_gate=AcceptAnswer(),
    ) as runtime:
        result = HarnessAgent(runtime.thread_manager).run("hello through candidate SDK")

        assert result.answer == "real gateway answer"
        assert len(gateway.requests) == 1
        turn = runtime.store.read_turn(result.turn_id)
        assert turn.binding_manifest["model_alias"] == "model-a"
        assert runtime.store.verify().valid is True
