from __future__ import annotations

from collections.abc import Mapping

import pytest
from langgraph.checkpoint.memory import MemorySaver

from agent_runtime.core.checkpointing import LangGraphCheckpointStore, agent_checkpoint_serde
from agent_runtime.core.context import AgentRunConfig
from agent_runtime.core.definition import AgentRuntimePolicy
from agent_runtime.core.model_request import ModelCallRecord
from agent_runtime.loop.runtime import ModelTurnEnvelope
from agent_runtime.loop.state import LoopState, ModelTurnDraft
from agent_runtime.result import AgentResult
from agent_runtime.service import AgentRunRequest, AgentService
from agent_runtime.tools.registry import ToolRegistry
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
from rag.schema.llm import LLMUsage


def _tool() -> Tool:
    schema: Mapping[str, JsonValue] = {
        "type": "object",
        "properties": {},
        "additionalProperties": False,
    }
    return Tool(
        definition=ToolDefinition(
            name="read_file",
            description="Read one file.",
            input_schema=schema,
        ),
        validate_input=json_schema_input(schema),
        run=lambda _arguments: {},
        normalize_output=lambda _raw: NormalizedToolOutput(),
        output_schema=None,
        static_effects=frozenset(),
        resolve_use=lambda _arguments: ResolvedToolUse(effects=frozenset(), targets=()),
        execution_revision="read-file-v1",
        idempotent=True,
        concurrency_safe=True,
        cancellation_mode=CancellationMode.COOPERATIVE,
        interrupt_behavior=InterruptBehavior.CANCEL,
        timeout_seconds=1.0,
        max_model_output_bytes=4096,
    )


def _registry() -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(_tool())
    return registry


def _definition() -> AgentRuntimePolicy:
    return AgentRuntimePolicy.test_factory(
        system_prompt="Answer directly.",
        allowed_tools=["read_file"],
    )


class _UsageProvider:
    def __init__(self, usage: LLMUsage) -> None:
        self._usage = usage

    async def next_turn(
        self,
        state: LoopState,
        *,
        definition: AgentRuntimePolicy,
        budget_remaining: int,
    ) -> ModelTurnEnvelope:
        del state, definition, budget_remaining
        return ModelTurnEnvelope(
            draft=ModelTurnDraft(action="finish", final_answer="usage answer"),
            model_call_record=ModelCallRecord(
                request_id="request-usage",
                prompt_revision="prompt-usage",
                toolset_revision="tools-usage",
                provider_wire_hash="wire-usage",
                usage=self._usage,
            ),
            provider_serializer_revision="provider-wire-v1",
        )


def _run_config(run_id: str) -> AgentRunConfig:
    return AgentRunConfig(
        turn_id=run_id,
    )


@pytest.mark.anyio
async def test_provider_usage_reaches_checkpoint_internal_and_public_results() -> None:
    run_id = "usage-provider"
    usage = LLMUsage(
        input_tokens=30,
        output_tokens=7,
        cached_input_tokens=11,
        source="provider",
        logical_input_tokens=30,
        uncached_input_tokens=19,
        cache_read_input_tokens=11,
        cache_write_input_tokens=0,
        usage_source="provider",
        raw_provider_usage={"prompt_tokens": 30, "cached_tokens": 11},
    )
    checkpointer = MemorySaver(serde=agent_checkpoint_serde())
    service = AgentService(
        definition=_definition(),
        tool_registry=_registry(),
        model_turn_provider=_UsageProvider(usage),
        checkpointer=checkpointer,
    )

    internal = await service.run(
        AgentRunRequest(message="Answer.", turn_id=run_id)
    )
    public = AgentResult._from_internal(internal)
    loaded = await LangGraphCheckpointStore(
        checkpointer,
        run_config=_run_config(run_id),
    ).load_latest()

    assert internal.model_call_records == [
        ModelCallRecord(
            request_id="request-usage",
            prompt_revision="prompt-usage",
            toolset_revision="tools-usage",
            provider_wire_hash="wire-usage",
            usage=usage,
        )
    ]
    assert loaded is not None
    assert loaded["model_call_records"] == internal.model_call_records
    assert public.usage.input_tokens == 30
    assert public.usage.output_tokens == 7
    assert public.usage.total_tokens == 37
    assert public.usage.logical_input_tokens == 30
    assert public.usage.uncached_input_tokens == 19
    assert public.usage.cache_read_input_tokens == 11
    assert public.usage.cache_write_input_tokens == 0
    assert public.usage.usage_source == "provider"


@pytest.mark.anyio
async def test_estimated_usage_keeps_missing_cache_fields_unknown() -> None:
    usage = LLMUsage(
        input_tokens=8,
        output_tokens=2,
        source="tokenizer_estimate",
        logical_input_tokens=8,
        uncached_input_tokens=None,
        cache_read_input_tokens=None,
        cache_write_input_tokens=None,
        usage_source="tokenizer_estimate",
    )
    service = AgentService(
        definition=_definition(),
        tool_registry=_registry(),
        model_turn_provider=_UsageProvider(usage),
    )

    public = AgentResult._from_internal(
        await service.run(
            AgentRunRequest(
                message="Answer.",
                turn_id="usage-estimated",
            )
        )
    )

    assert public.usage.cache_read_input_tokens is None
    assert public.usage.cache_write_input_tokens is None
    assert public.usage.usage_source == "tokenizer_estimate"
