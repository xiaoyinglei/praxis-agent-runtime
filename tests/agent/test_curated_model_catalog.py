"""The curated snapshot must reach the public CLI and wheel without drift."""

import subprocess
import sys
from pathlib import Path
from shutil import copyfile

import pytest
import yaml

from agent_runtime.core import llm_registry
from agent_runtime.core.llm_registry import ModelRegistry
from agent_runtime.harness import GatewayHarnessModel, HarnessMessage, HarnessModelRequest
from agent_runtime.modeling.contracts import LLMCallStage
from scripts.sync_model_catalog import rendered_configs

ROOT = Path(__file__).resolve().parents[2]


def test_public_catalog_is_current_and_reproducible():
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts/sync_model_catalog.py"), "--check"],
        cwd=ROOT, text=True, capture_output=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("change", ["demote", "delete"])
def test_generation_removes_withdrawn_managed_models_but_preserves_research(tmp_path, change):
    (tmp_path / "configs").mkdir()
    for name in ("model-catalog.yaml", "models.yaml", "agent-models.yaml"):
        copyfile(ROOT / "configs" / name, tmp_path / "configs" / name)
    snapshot_path = tmp_path / "configs/model-catalog.yaml"
    snapshot = yaml.safe_load(snapshot_path.read_text())
    if change == "demote":
        snapshot["models"]["kimi-k3"]["integration"] = "catalog_only"
        snapshot["models"]["kimi-k3"]["blocker"] = "Transport compatibility withdrawn."
    else:
        del snapshot["models"]["kimi-k3"]
    snapshot_path.write_text(yaml.safe_dump(snapshot))
    original_research = yaml.safe_load((tmp_path / "configs/models.yaml").read_text())
    previous_public = yaml.safe_load((tmp_path / "configs/agent-models.yaml").read_text())
    outputs = rendered_configs(tmp_path)
    for path, content in outputs.items():
        assert "kimi-k3" not in yaml.safe_load(content)["models"]
        path.write_text(content)
    research = yaml.safe_load(outputs[tmp_path / "configs/models.yaml"])
    for model_id, definition in original_research["models"].items():
        if model_id not in previous_public["models"]:
            assert research["models"][model_id] == definition
    assert rendered_configs(tmp_path) == outputs


def test_snapshot_has_sources_and_keeps_unsupported_transports_out_of_runtime():
    snapshot = yaml.safe_load((ROOT / "configs/model-catalog.yaml").read_text())
    public = yaml.safe_load((ROOT / "configs/agent-models.yaml").read_text())
    assert snapshot["as_of"] == "2026-10-22"
    assert {"openai", "anthropic", "google", "deepseek", "kimi", "qwen", "zhipu", "minimax"} <= {
        item["provider"] for item in snapshot["models"].values()
    }
    for model_id, item in snapshot["models"].items():
        assert item["sources"] and all(url.startswith("https://") for url in item["sources"])
        if item["integration"] == "catalog_only":
            assert item["blocker"]
            assert model_id not in public["models"]
        else:
            assert item["integration"] == "chat_completions"
            assert public["models"][model_id] == item["config"]
    assert public["defaults"]["primary_model"] == "deepseek-flash"


@pytest.mark.parametrize("model_id", [
    "deepseek-flash", "kimi-k3", "qwen3.8-max", "MiniMax-M3.1-Flash", "openai/gpt-oss-120b",
])
def test_registered_model_uses_its_own_capacity_and_wire_settings(model_id, monkeypatch):
    monkeypatch.setattr(llm_registry, "_build_chat_generator", lambda **kwargs: object())
    config = ModelRegistry._load_yaml_file(ROOT / "configs/agent-models.yaml")
    registry = ModelRegistry(config)
    definition = registry.get_model_definition(model_id)
    model = GatewayHarnessModel(
        model_id=model_id, resolved=registry.resolve(model_id), instructions=("Use supplied evidence.",),
    )
    prepared = model.prepare(HarnessModelRequest(
        thread_id="t", turn_id="u", binding_manifest={},
        messages=(HarnessMessage(role="user", content="Return the result."),),
    ))
    projection = prepared.request_ref["context_projection"]
    assert projection["model_context_tokens"] == definition.context_window_tokens
    assert projection["explicit_input_cap"] is None
    assert projection["effective_input_limit"] == (
        definition.context_window_tokens - prepared.resource_request.output_tokens - 512
    )
    assert 0 < prepared.resource_request.output_tokens <= definition.max_output_tokens
    wire = prepared.dispatch_payload.request
    assert wire.settings.model == model_id
    assert wire.settings.temperature == definition.defaults.temperature
    assert wire.settings.top_p == definition.defaults.top_p


@pytest.mark.parametrize("model_id", ["qwen3.8-max", "qwen3.8-flash"])
def test_qwen_runtime_respects_separate_input_limit_with_small_output(model_id, monkeypatch):
    monkeypatch.setattr(llm_registry, "_build_chat_generator", lambda **kwargs: object())
    registry = ModelRegistry(ModelRegistry._load_yaml_file(ROOT / "configs/agent-models.yaml"))
    gateway = registry.resolve(model_id).gateway
    stages = set(registry.get_model_definition(model_id).llm_stage_budgets)
    stages.add(LLMCallStage.CONTEXT_COMPACTION.value)
    for stage in stages:
        budget = gateway.effective_stage_budget(LLMCallStage(stage), kwargs={"max_tokens": 8192})
        assert budget.max_input_tokens <= 983616
