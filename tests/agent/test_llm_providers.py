from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType, SimpleNamespace

import pytest

from rag.agent.core.context import AgentRunConfig
from rag.agent.core.definition import AgentRuntimePolicy
from rag.agent.core.llm_providers import LLMLoopModelTurnProvider
from rag.agent.core.messages import ModelMessage, StopReason, ToolUseResult
from rag.agent.loop.runtime import ModelTurnEnvelope
from rag.agent.loop.state import create_loop_state
from rag.agent.tools.tool import (
    CancellationMode,
    InterruptBehavior,
    JsonValue,
    NormalizedToolOutput,
    ResolvedToolUse,
    Tool,
    ToolDefinition,
    json_schema_input,
)
from rag.assembly.tokenizer import TokenAccountingService, TokenizerContract
from rag.schema.llm import LLMCallStage, LLMStageBudget, LLMUsage


def _tool(name: str) -> Tool:
    schema: Mapping[str, JsonValue] = {
        "type": "object",
        "properties": {},
        "additionalProperties": False,
    }
    return Tool(
        definition=ToolDefinition(
            name=name,
            description=f"Use {name}.",
            input_schema=schema,
        ),
        validate_input=json_schema_input(schema),
        run=lambda _arguments: {},
        normalize_output=lambda _raw: NormalizedToolOutput(),
        output_schema=None,
        static_effects=frozenset(),
        resolve_use=lambda _arguments: ResolvedToolUse(effects=frozenset(), targets=()),
        execution_revision=f"{name}-v1",
        idempotent=True,
        concurrency_safe=True,
        cancellation_mode=CancellationMode.COOPERATIVE,
        interrupt_behavior=InterruptBehavior.CANCEL,
        timeout_seconds=1.0,
        max_model_output_bytes=4096,
    )


class _CanonicalGateway:
    def __init__(self, turn: ToolUseResult | None = None) -> None:
        self.calls: list[dict[str, object]] = []
        self.turn = turn or ToolUseResult(
            text="canonical answer",
            tool_calls=[],
            stop_reason=StopReason.END_TURN,
            raw_stop_reason="stop",
        )
        self.token_accounting = TokenAccountingService(
            TokenizerContract(
                embedding_model_name="canonical-gateway",
                tokenizer_model_name="canonical-gateway",
                chunking_tokenizer_model_name="canonical-gateway",
                tokenizer_backend="simple",
                max_context_tokens=32_768,
                prompt_reserved_tokens=256,
                local_files_only=True,
            )
        )

    def effective_stage_budget(
        self,
        stage: LLMCallStage,
        *,
        kwargs: Mapping[str, object] | None = None,
    ) -> LLMStageBudget:
        del stage, kwargs
        return LLMStageBudget(
            max_input_tokens=32_000,
            max_output_tokens=4_096,
        )

    async def agenerate_model_request(self, **kwargs: object) -> object:
        self.calls.append(dict(kwargs))
        return SimpleNamespace(
            turn=self.turn,
            usage=LLMUsage(
                input_tokens=12,
                output_tokens=2,
                cached_input_tokens=4,
                source="provider",
                logical_input_tokens=12,
                uncached_input_tokens=8,
                cache_read_input_tokens=4,
                cache_write_input_tokens=0,
                usage_source="provider",
                raw_provider_usage={"prompt_tokens": 12},
            ),
            provider_wire_hash="wire_provider_parity",
            serializer_revision="provider-wire-v1",
            wire_kind=str(kwargs["provider"]),
        )


def _state():
    return create_loop_state(
        current_message="Inspect README.md.",
        run_config=AgentRunConfig(
            turn_id="provider-parity",
        ),
    )


def _definition() -> AgentRuntimePolicy:
    return AgentRuntimePolicy.test_factory(
        system_prompt="Use canonical tools.",
        allowed_tools=["list_files", "read_file"],
    )


class TestLLMLoopModelTurnProviderBasic:
    """Basic sanity checks for LLMLoopModelTurnProvider."""

    def test_module_exports_expected_symbols(self) -> None:
        """Verify the module exports the expected public API after cleanup."""
        import rag.agent.core.llm_providers as m

        assert hasattr(m, "LLMLoopModelTurnProvider")
        assert hasattr(m, "LoopModelDecision")
        assert hasattr(m, "parse_loop_model_turn")
        assert hasattr(m, "create_loop_model_turn_provider")

    def test_retrieval_hint_code_is_removed(self) -> None:
        """Verify LLMRetrievalHintProvider no longer exists in the module."""
        import rag.agent.core.llm_providers as m

        assert not hasattr(m, "LLMRetrievalHintProvider")
        assert not hasattr(m, "create_default_providers")
        assert not hasattr(m, "RetrievalHintDecision")
        assert not hasattr(m, "_extract_quoted_terms")
        assert not hasattr(m, "_validate_retrieval_signals")


@pytest.mark.anyio
async def test_all_supported_providers_receive_the_same_canonical_request() -> None:
    snapshot = MappingProxyType(
        {
            "list_files": _tool("list_files"),
            "read_file": _tool("read_file"),
        }
    )
    captured = []
    envelopes = []

    for provider_name, supports_native_tools in (
        ("openai-compatible", True),
        ("mlx", False),
        ("ollama", False),
    ):
        gateway = _CanonicalGateway()
        provider = LLMLoopModelTurnProvider(
            gateway,
            model="test-model",
            provider=provider_name,
            supports_native_tools=supports_native_tools,
            registry_snapshot=snapshot,
            resident_tool_names=("list_files", "read_file"),
        )
        envelope = await provider.next_turn(
            _state(),
            definition=_definition(),
            budget_remaining=10_000,
        )
        assert isinstance(envelope, ModelTurnEnvelope)
        captured.append(gateway.calls[0]["request"])
        envelopes.append(envelope)

    assert [request.exposed_tool_names for request in captured] == [
        ("list_files", "read_file"),
        ("list_files", "read_file"),
        ("list_files", "read_file"),
    ]
    assert len({request.request_id for request in captured}) == 1
    assert len({request.prompt_revision for request in captured}) == 1
    assert len({request.toolset_revision for request in captured}) == 1
    assert all(envelope.model_call_record.request_id == captured[0].request_id for envelope in envelopes)
    assert all(envelope.draft.final_answer == "canonical answer" for envelope in envelopes)


@pytest.mark.anyio
async def test_model_defaults_and_reasoning_continuation_reach_canonical_messages() -> None:
    gateway = _CanonicalGateway(
        ToolUseResult(
            text="",
            reasoning_content="I should inspect README.md first.",
            tool_calls=[],
            stop_reason=StopReason.END_TURN,
            raw_stop_reason="stop",
        )
    )
    provider = LLMLoopModelTurnProvider(
        gateway,
        model="kimi-k2.6",
        provider="kimi",
        supports_native_tools=True,
        registry_snapshot=MappingProxyType({}),
        resident_tool_names=(),
        kwargs={
            "max_tokens": 32_768,
            "temperature": 1.0,
            "top_p": 0.95,
            "provider_options": {"thinking": {"type": "enabled"}},
        },
        context_window_tokens=65_536,
    )

    envelope = await provider.next_turn(
        _state(),
        definition=_definition(),
        budget_remaining=100_000,
    )

    request = gateway.calls[0]["request"]
    assert request.settings.max_output_tokens == 32_768
    assert request.settings.temperature == 1.0
    assert request.settings.top_p == 0.95
    assert request.settings.provider_options == {
        "thinking": {"type": "enabled"}
    }
    assert envelope.assistant_message == ModelMessage(
        role="assistant",
        content="",
        reasoning_content="I should inspect README.md first.",
    )
