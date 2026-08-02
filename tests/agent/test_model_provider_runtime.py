from __future__ import annotations

from collections.abc import Mapping

import pytest

from rag.agent.core import model_provider_runtime
from rag.agent.core.context import AgentRunConfig
from rag.agent.core.definition import AgentRuntimePolicy
from rag.agent.core.goal_contract import GoalSpec
from rag.agent.core.model_provider_runtime import (
    ModelProviderResolver,
    ResultDrivenModelTurnProvider,
)
from rag.agent.loop.state import ModelTurnDraft, create_loop_state
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
        resolve_use=lambda _arguments: ResolvedToolUse(
            effects=frozenset(),
            targets=(),
        ),
        execution_revision="read-v1",
        idempotent=True,
        concurrency_safe=True,
        cancellation_mode=CancellationMode.COOPERATIVE,
        interrupt_behavior=InterruptBehavior.CANCEL,
        timeout_seconds=1.0,
        max_model_output_bytes=4096,
    )


def _policy() -> AgentRuntimePolicy:
    return AgentRuntimePolicy.test_factory(
        system_prompt="Answer.",
        allowed_tools=["read_file"],
    )


def _state():
    state = create_loop_state(
        current_message="Answer.",
        run_config=AgentRunConfig(
            turn_id="provider-runtime",
        ),
    )
    state["resident_tool_names"] = ["read_file"]
    return state


class _Provider:
    async def next_turn(self, *args: object, **kwargs: object) -> ModelTurnDraft:
        del args, kwargs
        return ModelTurnDraft(action="finish", final_answer="direct")


def test_resolver_reuses_injected_provider() -> None:
    provider = _Provider()
    resolver = ModelProviderResolver(
        model_turn_provider=provider,
        model_registry=None,
        policy=_policy(),
        registry_snapshot={"read_file": _tool()},
    )

    assert resolver.resolve(_state()) is provider


def test_resolver_passes_runtime_goal_to_canonical_provider(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}
    provider = _Provider()
    goal = GoalSpec(original_query="Implement the requested change.")

    def create_provider(*args: object, **kwargs: object) -> _Provider:
        captured["args"] = args
        captured.update(kwargs)
        return provider

    monkeypatch.setattr(
        model_provider_runtime,
        "create_loop_model_turn_provider",
        create_provider,
    )

    resolved = ModelProviderResolver(
        model_turn_provider=None,
        model_registry=object(),  # type: ignore[arg-type]
        policy=_policy(),
        registry_snapshot={"read_file": _tool()},
        goal_spec=goal,
    ).resolve(_state())

    assert resolved is provider
    assert captured["goal_spec"] == goal


def test_strict_registry_failure_is_not_hidden() -> None:
    class _BrokenRegistry:
        def resolve_for_node(self, **kwargs: object) -> object:
            del kwargs
            raise RuntimeError("model provider broken")

    resolver = ModelProviderResolver(
        model_turn_provider=None,
        model_registry=_BrokenRegistry(),  # type: ignore[arg-type]
        policy=_policy(),
        registry_snapshot={"read_file": _tool()},
    )

    with pytest.raises(RuntimeError, match="model provider broken"):
        resolver.resolve(_state())


def test_non_strict_failure_records_diagnostic_and_falls_back() -> None:
    class _BrokenRegistry:
        def resolve_for_node(self, **kwargs: object) -> object:
            del kwargs
            raise RuntimeError("model provider broken")

    state = _state()
    provider = ModelProviderResolver(
        model_turn_provider=None,
        model_registry=_BrokenRegistry(),  # type: ignore[arg-type]
        policy=_policy(),
        registry_snapshot={"read_file": _tool()},
        strict_model_provider=False,
    ).resolve(state)

    assert isinstance(provider, ResultDrivenModelTurnProvider)
    assert state["runtime_diagnostics"][-1].code == ("default_providers_initialization_failed")
