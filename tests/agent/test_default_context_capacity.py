import pytest
import yaml

from agent_runtime.core import llm_registry as registry_module
from agent_runtime.core.llm_registry import ModelRegistry
from agent_runtime.modeling.contracts import LLMCallStage


@pytest.fixture
def project_registry(tmp_path, monkeypatch):
    monkeypatch.delenv("RAG_AGENT_MODELS_PATH", raising=False)
    monkeypatch.delenv("RAG_AGENT_MODELS", raising=False)
    monkeypatch.setenv("PRAXIS_DISABLE_DOTENV", "1")
    monkeypatch.setenv("PRAXIS_MODEL_REGISTRY_PATH", str(tmp_path / "model-registry.yaml"))
    # Only provider construction is replaced; actual catalog loading,
    # definition freezing, tokenizer and Gateway resolution remain in use.
    monkeypatch.setattr(registry_module, "_build_chat_generator", lambda **kwargs: object())
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    return ModelRegistry.from_env(workspace=workspace, worktree=workspace)


@pytest.mark.parametrize(
    "model_id,output",
    [
        ("deepseek-flash", 32768),
        ("openai/gpt-oss-120b", 8192),
        ("mlx-community/Qwen3.5-9B-4bit", 8192),
    ],
)
def test_project_catalog_uses_model_capacity_without_default_input_cap(project_registry, model_id, output):
    definition = project_registry.get_model_definition(model_id)
    assert definition.llm_stage_budgets["agent_step"].max_input_tokens is None
    resolved = project_registry.resolve(model_id)
    budget = resolved.gateway.effective_stage_budget(LLMCallStage.AGENT_STEP, kwargs={"max_tokens": output})
    assert budget.max_input_tokens == definition.capabilities.context_window_tokens - output - 512
    assert resolved.gateway.context_policy.revision == "context-capacity-v2"


def test_project_catalog_admits_complete_request_above_64k(project_registry):
    from agent_runtime.harness import GatewayHarnessModel, HarnessMessage, HarnessModelRequest

    model = GatewayHarnessModel(
        model_id="deepseek-flash",
        resolved=project_registry.resolve("deepseek-flash"),
        instructions=("Use the supplied evidence.",),
    )
    prepared = model.prepare(
        HarnessModelRequest(
            thread_id="thread",
            turn_id="turn",
            binding_manifest={},
            messages=(HarnessMessage(role="user", content="x " * 70000),),
        )
    )
    projection = prepared.request_ref["context_projection"]
    assert prepared.resource_request.input_tokens > 64000
    assert projection["explicit_input_cap"] is None
    assert prepared.resource_request.output_tokens == 65536
    assert projection["effective_input_limit"] == 1048576 - 65536 - 512
    assert projection["auto_compact_limit"] == 943718
    assert projection["post_compact_target"] == 754974


def test_explicit_user_yaml_input_cap_still_combines_with_model_capacity(tmp_path, monkeypatch):
    from agent_runtime.harness import (
        ContextCompactionRequiredError,
        GatewayHarnessModel,
        HarnessMessage,
        HarnessModelRequest,
    )

    data = yaml.safe_load(ModelRegistry._BUNDLED_CONFIG_PATH.read_text())
    data["llm_budgets"]["agent_step"]["max_input_tokens"] = 64000
    path = tmp_path / "custom-models.yaml"
    path.write_text(yaml.safe_dump(data))
    registry = ModelRegistry(ModelRegistry._load_yaml_file(path))
    monkeypatch.setattr(registry_module, "_build_chat_generator", lambda **kwargs: object())
    resolved = registry.resolve("deepseek-flash")
    budget = resolved.gateway.effective_stage_budget(LLMCallStage.AGENT_STEP, kwargs={"max_tokens": 32768})
    assert budget.max_input_tokens == 64000
    assert registry.get_model_definition("deepseek-flash").llm_stage_budgets["agent_step"].max_input_tokens == 64000
    summary = resolved.gateway.effective_stage_budget(LLMCallStage.CONTEXT_COMPACTION)
    assert summary.max_input_tokens == 1048576 - summary.max_output_tokens - summary.safety_margin_tokens
    model = GatewayHarnessModel(
        model_id="deepseek-flash", resolved=resolved, instructions=("Use the supplied evidence.",)
    )
    with pytest.raises(ContextCompactionRequiredError) as rejected:
        model.prepare(
            HarnessModelRequest(
                thread_id="thread",
                turn_id="turn",
                binding_manifest={},
                messages=(HarnessMessage(role="user", content="x " * 70000),),
            )
        )
    assert rejected.value.max_input_tokens == 64000
