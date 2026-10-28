"""Regression coverage for declaration parsing and frozen model compatibility."""

from pathlib import Path

import pytest

from agent_runtime.core.llm_config import AgentModelsConfig, ModelProvider, ModelSpec
from agent_runtime.core.llm_registry import ModelRegistry
from agent_runtime.model_definition import ModelExecutionDefinition, canonical_definition_json
from agent_runtime.modeling.contracts import LLMCallStage


def test_catalog_rejects_duplicate_yaml_at_nested_mapping(tmp_path: Path) -> None:
    path = tmp_path / "models.yaml"
    path.write_text("""models:
  arbitrary:
    capability: chat
    provider: mlx
    context_window_tokens: 8192
    context_window_tokens: 16384
defaults:
  primary_model: arbitrary
""")
    with pytest.raises(ValueError, match="duplicate mapping key.*context_window_tokens.*line 6"):
        ModelRegistry._load_yaml_file(path)


def test_new_config_does_not_require_unused_stage_budgets() -> None:
    config = AgentModelsConfig(
        models={"arbitrary": ModelSpec(provider=ModelProvider.MLX, context_window_tokens=8192)},
        default_model="arbitrary",
    )
    obsolete = {
        LLMCallStage.GOAL_CONTRACT,
        LLMCallStage.RETRIEVAL_HINT,
        LLMCallStage.TOOL_DECISION,
        LLMCallStage.LLM_COMPARE,
        LLMCallStage.FINAL_SYNTHESIS,
    }
    assert not obsolete.intersection(config.llm_stage_budgets)
    AgentModelsConfig.model_validate(config.model_dump())


def test_old_definition_retains_canonical_bytes_and_revision() -> None:
    config = AgentModelsConfig(
        models={"arbitrary": ModelSpec(provider=ModelProvider.MLX, context_window_tokens=8192)},
        default_model="arbitrary",
    )
    definition = ModelRegistry(config).get_model_definition("arbitrary")
    payload = definition.model_dump(mode="json")
    payload.pop("continuation_summary_role", None)
    payload.pop("agent_output_policy", None)
    old = ModelExecutionDefinition.model_validate(payload)
    canonical = canonical_definition_json(old)
    assert b"continuation_summary_role" not in canonical
    restored = ModelExecutionDefinition.model_validate_json(canonical)
    assert canonical_definition_json(restored) == canonical
    assert restored.definition_revision == old.definition_revision
    assert old.definition_revision == "sha256:d3058f762fecc81539db52249ca5db6a164b49a26d75bc19b27e74f1332e6694"


@pytest.mark.parametrize("provider,model_id", [("deepseek", "deepseek-flash"), ("other", "another")])
def test_new_context_role_depends_on_provenance_and_configuration(provider: str, model_id: str) -> None:
    from dataclasses import replace

    from agent_runtime.harness.model_adapter import GatewayHarnessModel
    from agent_runtime.harness.protocol import HarnessMessage, HarnessModelRequest, restore_harness_message
    from agent_runtime.modeling.openai_wire import serialize_openai_request
    from tests.agent.harness.test_model_adapter import BudgetAwareCapturingGateway, _resolved_model

    gateway = BudgetAwareCapturingGateway(max_input_tokens=9000)
    resolved = _resolved_model(
        gateway=gateway,
        token_accounting=gateway.token_accounting,
        provider=provider,
        model_id=model_id,
    )
    request = HarnessModelRequest(
        thread_id="t",
        turn_id="u",
        binding_manifest={},
        messages=(
            HarnessMessage(role="user", content="Continue work."),
            HarnessMessage(role="context", content="Any presentation", context_category="continuation_summary"),
            restore_harness_message({
                "role": "context",
                "content": "Context compaction:\nThis is an ordinary constraint.",
            }),
        ),
    )
    for summary_role, expected in (("context", "user"), ("assistant", "assistant")):
        model = GatewayHarnessModel(
            model_id=model_id,
            instructions=("Continue task.",),
            resolved=replace(resolved, continuation_summary_role=summary_role),
        )
        wire = serialize_openai_request(model.prepare(request).dispatch_payload.request).payload
        assert [message["role"] for message in wire["messages"]][-3:] == ["user", expected, "user"]


def test_summary_setting_freezes_at_model_definition_boundary(monkeypatch: pytest.MonkeyPatch) -> None:
    from agent_runtime.core import llm_registry

    spec = ModelSpec(
        provider=ModelProvider.OPENAI_COMPATIBLE,
        provider_name="deepseek",
        context_window_tokens=8192,
        continuation_summary_role="assistant",
    )
    config = AgentModelsConfig(models={"deepseek-flash": spec}, default_model="deepseek-flash")
    registry = ModelRegistry(config)
    frozen = registry.get_model_definition("deepseek-flash")
    archive = ModelExecutionDefinition.model_validate_json(canonical_definition_json(frozen))
    config.models["deepseek-flash"] = spec.model_copy(update={"continuation_summary_role": "context"})
    monkeypatch.setattr(llm_registry, "_build_chat_generator", lambda **kwargs: object())
    assert registry.resolve_definition(archive).continuation_summary_role == "assistant"
    assert ModelRegistry(config).resolve("deepseek-flash").continuation_summary_role == "context"
    assert (
        archive.definition_revision != ModelRegistry(config).get_model_definition("deepseek-flash").definition_revision
    )


def test_legacy_runtime_fields_are_readable_without_tightening_frozen_validation() -> None:
    from agent_runtime.core.llm_config import ModelRuntimeConfig
    from agent_runtime.core.llm_registry import _model_spec_from_definition
    from agent_runtime.model_registry import ModelRuntimeDeclaration

    config = AgentModelsConfig(
        models={
            "arbitrary": ModelSpec(
                provider=ModelProvider.MLX,
                context_window_tokens=8192,
                runtime=ModelRuntimeConfig(
                    expected_model_contains="", startup_timeout_seconds=17.0, poll_interval_seconds=0.25
                ),
            )
        },
        default_model="arbitrary",
    )
    definition = ModelRegistry(config).get_model_definition("arbitrary")
    canonical = canonical_definition_json(definition)
    restored = ModelExecutionDefinition.model_validate_json(canonical)
    assert canonical_definition_json(restored) == canonical
    runtime = _model_spec_from_definition(restored).runtime
    assert runtime is not None
    assert runtime.expected_model_contains == ""
    assert runtime.startup_timeout_seconds == 17.0
    assert runtime.poll_interval_seconds == 0.25
    with pytest.raises(ValueError, match="trimmed|blank|empty"):
        ModelRuntimeDeclaration(expected_model_contains="")
