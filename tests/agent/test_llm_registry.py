from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest
import yaml

from agent_runtime.core import llm_registry as llm_registry_module
from agent_runtime.core.llm_config import AgentModelsConfig, ModelProvider, ModelSpec
from agent_runtime.core.llm_registry import (
    ModelNotAvailableError,
    ModelRegistry,
    UnknownModelAliasError,
    _chat_provider_config,
)
from agent_runtime.model_definition import canonical_definition_json
from agent_runtime.modeling.config import GenerationTaskConfig
from agent_runtime.modeling.contracts import LLMCallStage, LLMStageBudget


def _ollama_spec(model: str = "test-model") -> ModelSpec:
    return ModelSpec(
        provider=ModelProvider.OLLAMA,
        model=model,
        base_url="http://localhost:11434",
        context_window_tokens=32768,
    )


def _make_config(
    *,
    default_model: str = "main",
    fallback_model: str | None = None,
) -> AgentModelsConfig:
    models: dict[str, ModelSpec] = {"main": _ollama_spec("main-model")}
    if fallback_model:
        models["fast"] = _ollama_spec("fast-model")
    return AgentModelsConfig(
        models=models,
        default_model=default_model,
        fallback_model=fallback_model,
    )


def test_model_execution_definition_has_fixed_canonical_digest() -> None:
    registry = ModelRegistry(_make_config())

    definition = registry.get_model_definition("main")
    payload = canonical_definition_json(definition)

    assert b'"api_key_env":null' in payload
    assert b'"request_context_tokens":null' in payload
    assert definition.definition_revision == (
        "sha256:0d78906982c607629387196daf265f42dfc585b5da3b53fb780a86feef0fe786"
    )


def test_model_definition_digest_changes_only_for_selected_request_definition() -> None:
    baseline = _make_config()
    changed = baseline.model_copy(deep=True)
    changed.models["main"].max_tokens = baseline.models["main"].max_tokens + 1
    unrelated = baseline.model_copy(
        update={
            "models": {
                **baseline.models,
                "unrelated": _ollama_spec("other-model"),
            }
        },
        deep=True,
    )

    base_digest = ModelRegistry(baseline).get_model_definition("main").definition_revision

    assert ModelRegistry(changed).get_model_definition("main").definition_revision != base_digest
    assert ModelRegistry(unrelated).get_model_definition("main").definition_revision == base_digest


def test_model_definition_digest_covers_generation_defaults_and_stage_budgets() -> None:
    baseline = _make_config()
    baseline_digest = ModelRegistry(baseline).get_model_definition("main").definition_revision

    generation_changed = baseline.model_copy(deep=True)
    generation_changed.generation = replace(
        generation_changed.generation,
        answer=GenerationTaskConfig(max_tokens=777, temperature=0.25),
    )
    budget_changed = baseline.model_copy(deep=True)
    budget_changed.llm_stage_budgets[LLMCallStage.AGENT_STEP] = LLMStageBudget(
        max_input_tokens=63_999,
        max_output_tokens=32_768,
        safety_margin_tokens=512,
    )
    defaults_changed = baseline.model_copy(deep=True)
    defaults_changed.models["main"].defaults = {"temperature": 0.25}

    assert (
        ModelRegistry(generation_changed).get_model_definition("main").definition_revision
        != baseline_digest
    )
    assert (
        ModelRegistry(budget_changed).get_model_definition("main").definition_revision
        != baseline_digest
    )
    assert (
        ModelRegistry(defaults_changed).get_model_definition("main").definition_revision
        != baseline_digest
    )


def test_model_definition_snapshot_is_not_mutated_by_callers() -> None:
    registry = ModelRegistry(_make_config())
    first = registry.get_model_definition("main")
    revision = first.definition_revision

    first.llm_stage_budgets["agent_step"] = first.llm_stage_budgets[
        "agent_step"
    ].model_copy(update={"max_input_tokens": 1})

    assert registry.get_model_definition("main").definition_revision == revision


def test_model_definition_rejects_non_finite_request_values() -> None:
    config = _make_config()
    config.models["main"].defaults = {"temperature": float("nan")}

    with pytest.raises(ValueError, match="finite|JSON"):
        ModelRegistry(config)


def test_model_definition_rejects_non_json_request_values() -> None:
    config = _make_config()
    config.models["main"].defaults = {"provider_options": {"modes": {"a", "b"}}}

    with pytest.raises(ValueError, match="JSON|json|provider_options|extra"):
        ModelRegistry(config)


@pytest.mark.parametrize(
    "unsafe_defaults",
    [
        {"api_key": "plaintext-secret"},
        {"headers": {"Authorization": "Bearer plaintext-secret"}},
        {"provider_options": {"thinking": {"type": "enabled"}, "token": "secret"}},
    ],
)
def test_whole_catalog_override_rejects_transport_or_secret_defaults(
    unsafe_defaults: dict[str, object],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("RAG_AGENT_MODELS_PATH", raising=False)
    monkeypatch.setenv(
        "RAG_AGENT_MODELS",
        json.dumps(
            {
                "version": 1,
                "models": {
                    "main": {
                        "provider": "ollama",
                        "model": "main-model",
                        "defaults": unsafe_defaults,
                    }
                },
                "default_model": "main",
            }
        ),
    )

    with pytest.raises(ValueError, match="extra|forbidden|provider_options"):
        ModelRegistry.from_env(env_path=str(tmp_path / "missing.env"))


@pytest.mark.parametrize(
    "unsafe_fields",
    [
        {"base_url": "https://example.com/v1?api_key=plaintext-secret"},
        {"api_key_env": "sk-plaintext-secret"},
        {
            "runtime": {
                "health_url": "http://127.0.0.1/health?token=plaintext-secret"
            }
        },
    ],
)
def test_whole_catalog_override_rejects_secrets_in_endpoint_fields(
    unsafe_fields: dict[str, object],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = {"provider": "ollama", "model": "main-model", **unsafe_fields}
    monkeypatch.delenv("RAG_AGENT_MODELS_PATH", raising=False)
    monkeypatch.setenv(
        "RAG_AGENT_MODELS",
        json.dumps(
            {
                "version": 1,
                "models": {"main": model},
                "default_model": "main",
            }
        ),
    )

    with pytest.raises(ValueError, match="api_key_env|credentials|query|fragment"):
        ModelRegistry.from_env(env_path=str(tmp_path / "missing.env"))


@pytest.mark.parametrize(
    "invalid_fields",
    [
        {"max_tokens": -7},
        {"timeout_seconds": -1},
        {"input_cost_per_1m": -3},
        {"location": "local", "base_url": "https://api.example.com/v1"},
        {
            "provider": "ollama",
            "location": "cloud",
            "base_url": "https://api.example.com/v1",
        },
        {"api_key": "plaintext-secret"},
        {"headers": {"Authorization": "Bearer plaintext-secret"}},
        {"unknown_request_knob": True},
    ],
)
def test_whole_catalog_json_override_rejects_invalid_or_unknown_model_fields(
    invalid_fields: dict[str, object],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model: dict[str, object] = {
        "provider": "openai_compatible",
        "model": "main-model",
        "base_url": "https://api.example.com/v1",
        "location": "cloud",
    }
    model.update(invalid_fields)
    monkeypatch.delenv("RAG_AGENT_MODELS_PATH", raising=False)
    monkeypatch.setenv(
        "RAG_AGENT_MODELS",
        json.dumps(
            {
                "version": 1,
                "models": {"main": model},
                "default_model": "main",
            }
        ),
    )

    with pytest.raises(ValueError):
        ModelRegistry.from_env(env_path=str(tmp_path / "missing.env"))


def test_whole_catalog_yaml_override_rejects_unknown_chat_fields(tmp_path: Path) -> None:
    config_path = tmp_path / "models.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "models": {
                    "main": {
                        "capability": "chat",
                        "provider": "ollama",
                        "model": "main-model",
                        "api_key": "plaintext-secret",
                    }
                },
                "defaults": {"primary_model": "main"},
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="unknown fields.*api_key"):
        ModelRegistry._load_yaml_file(config_path)


@pytest.mark.parametrize("source", ["json", "yaml"])
def test_whole_catalog_override_rejects_explicit_null_defaults(
    source: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = {
        "version": 1,
        "models": {
            "main": {
                "provider": "ollama",
                "model": "main-model",
                "defaults": {"temperature": None},
            }
        },
        "default_model": "main",
    }
    if source == "json":
        monkeypatch.delenv("RAG_AGENT_MODELS_PATH", raising=False)
        monkeypatch.setenv("RAG_AGENT_MODELS", json.dumps(payload))
        with pytest.raises(ValueError, match="defaults do not accept null"):
            ModelRegistry.from_env(env_path=str(tmp_path / "missing.env"))
    else:
        config_path = tmp_path / "models.yaml"
        config_path.write_text(
            yaml.safe_dump(
                {
                    "version": 1,
                    "models": {
                        "main": {
                            "capability": "chat",
                            **payload["models"]["main"],
                        }
                    },
                    "defaults": {"primary_model": "main"},
                }
            ),
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match="defaults do not accept null"):
            ModelRegistry._load_yaml_file(config_path)


def test_execution_definition_freezes_the_same_effective_endpoint_as_runtime() -> None:
    config = AgentModelsConfig(
        models={
            "main": ModelSpec(
                provider=ModelProvider.OLLAMA,
                model="main-model",
            )
        },
        default_model="main",
    )
    registry = ModelRegistry(config)

    definition = registry.get_model_definition("main")
    provider = _chat_provider_config(registry.get_model_spec("main"))

    assert definition.base_url == "http://localhost:11434"
    assert definition.location == "local"
    assert definition.tokenizer_model == "main-model"
    assert provider.base_url == definition.base_url


def test_resolved_kwargs_cannot_mutate_cached_definition(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _make_config()
    config.models["main"].defaults = {
        "provider_options": {"thinking": {"type": "enabled"}}
    }
    registry = ModelRegistry(config)
    original_revision = registry.get_model_definition("main").definition_revision
    monkeypatch.setattr(
        "agent_runtime.core.llm_registry._build_chat_generator",
        lambda spec: object(),
    )

    resolved = registry.resolve("main")
    provider_options = resolved.kwargs["provider_options"]
    assert isinstance(provider_options, dict)
    thinking = provider_options["thinking"]
    assert isinstance(thinking, dict)
    thinking["type"] = "disabled"

    assert registry.get_model_definition("main").definition_revision == original_revision


def test_load_configs_models_maps_openai_compatible_protocol(tmp_path: Path) -> None:
    config_path = tmp_path / "models.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "models": {
                    "qwen3_8b_mlx_4bit": {
                        "capability": "chat",
                        "provider": "qwen",
                        "protocol": "openai_compatible",
                        "model": "Qwen/Qwen3-8B-MLX-4bit",
                        "max_tokens": 16384,
                        "base_url": "http://127.0.0.1:8080/v1",
                        "context_window_tokens": 32768,
                        "defaults": {
                            "temperature": 1.0,
                            "top_p": 0.95,
                            "provider_options": {
                                "thinking": {"type": "enabled"},
                            },
                        },
                    }
                },
                "defaults": {"primary_model": "qwen3_8b_mlx_4bit"},
                "llm_budgets": {
                    "tool_decision": {
                        "max_input_tokens": 12000,
                        "max_output_tokens": 2048,
                        "safety_margin_tokens": 512,
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    config = ModelRegistry._load_yaml_file(config_path)

    assert config.default_model == "qwen3_8b_mlx_4bit"
    assert config.models["qwen3_8b_mlx_4bit"].provider is ModelProvider.OPENAI_COMPATIBLE
    assert config.models["qwen3_8b_mlx_4bit"].base_url == "http://127.0.0.1:8080/v1"
    assert config.models["qwen3_8b_mlx_4bit"].context_window_tokens == 32768
    assert config.models["qwen3_8b_mlx_4bit"].max_tokens == 16384
    assert config.models["qwen3_8b_mlx_4bit"].defaults == {
        "temperature": 1.0,
        "top_p": 0.95,
        "provider_options": {"thinking": {"type": "enabled"}},
    }
    assert config.llm_stage_budgets[LLMCallStage.TOOL_DECISION].max_input_tokens == 12000


def test_load_configs_models_preserves_api_key_env_for_cloud_models(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = tmp_path / "models.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "models": {
                    "mimo_cloud": {
                        "capability": "chat",
                        "provider": "mimo",
                        "protocol": "openai_compatible",
                        "model": "mimo-v2-flash",
                        "base_url": "https://api.xiaomimimo.com/v1",
                        "api_key_env": "MIMO_API_KEY",
                    }
                },
                "defaults": {"primary_model": "mimo_cloud"},
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("MIMO_API_KEY", "sk-test")

    config = ModelRegistry._load_yaml_file(config_path)
    provider_config = _chat_provider_config(config.models["mimo_cloud"])

    assert config.models["mimo_cloud"].api_key_env == "MIMO_API_KEY"
    assert provider_config.api_key == "sk-test"


def test_load_configs_models_supports_provider_section_schema(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = tmp_path / "models.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "providers": {
                    "local_mlx_chat_8080": {
                        "protocol": "openai_compatible",
                        "location": "local",
                        "base_url": "http://127.0.0.1:8080/v1",
                        "runtime": {
                            "health_url": "http://127.0.0.1:8080/v1/models",
                            "launch_command_template": [
                                "uv",
                                "run",
                                "python",
                                "-m",
                                "mlx_lm.server",
                                "--model",
                                "{model}",
                            ],
                        },
                    },
                    "groq": {
                        "protocol": "openai_compatible",
                        "location": "cloud",
                        "base_url": "https://api.groq.com/openai/v1",
                        "api_key_env": "GROQ_API_KEY",
                    },
                },
                "models": {
                    "qwen3_8b_mlx_4bit": {
                        "capability": "chat",
                        "provider": "local_mlx_chat_8080",
                        "model": "mlx-community/Qwen3-8B-4bit",
                        "context_window_tokens": 32768,
                        "runtime": {
                            "expected_model_contains": "Qwen3-8B-4bit",
                        },
                    },
                    "groq_gpt_oss_120b": {
                        "capability": "chat",
                        "provider": "groq",
                        "model": "openai/gpt-oss-120b",
                        "tokenizer_model": "gpt-oss-120b",
                        "context_window_tokens": 131072,
                        "request_context_tokens": 8000,
                    },
                },
                "defaults": {"primary_model": "groq_gpt_oss_120b"},
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("GROQ_API_KEY", "sk-test")

    config = ModelRegistry._load_yaml_file(config_path)
    groq = config.models["groq_gpt_oss_120b"]
    local = config.models["qwen3_8b_mlx_4bit"]
    provider_config = _chat_provider_config(groq)

    assert config.default_model == "groq_gpt_oss_120b"
    assert groq.provider is ModelProvider.OPENAI_COMPATIBLE
    assert groq.provider_name == "groq"
    assert groq.base_url == "https://api.groq.com/openai/v1"
    assert groq.api_key_env == "GROQ_API_KEY"
    assert groq.location == "cloud"
    assert groq.context_window_tokens == 131072
    assert groq.request_context_tokens == 8000
    assert groq.tokenizer_model == "gpt-oss-120b"
    assert provider_config.api_key == "sk-test"
    assert provider_config.base_url == "https://api.groq.com/openai/v1"
    assert local.provider is ModelProvider.OPENAI_COMPATIBLE
    assert local.provider_name == "local_mlx_chat_8080"
    assert local.base_url == "http://127.0.0.1:8080/v1"
    assert local.location == "local"
    assert local.runtime is not None
    assert local.runtime.health_url == "http://127.0.0.1:8080/v1/models"
    assert local.runtime.expected_model_contains == "Qwen3-8B-4bit"
    assert local.runtime.launch_command == (
        "uv",
        "run",
        "python",
        "-m",
        "mlx_lm.server",
        "--model",
        "mlx-community/Qwen3-8B-4bit",
    )


def test_repository_catalog_declares_local_qwen35_9b() -> None:
    config = ModelRegistry._load_yaml_file(Path("configs/models.yaml"))

    spec = config.models["qwen3_5_9b_mlx_4bit"]
    assert spec.provider is ModelProvider.OPENAI_COMPATIBLE
    assert spec.provider_name == "local_mlx_chat_8080"
    assert spec.model == "mlx-community/Qwen3.5-9B-4bit"
    assert spec.context_window_tokens == 262_144
    assert spec.location == "local"
    assert spec.runtime is not None
    assert spec.runtime.health_url == "http://127.0.0.1:8080/v1/models"
    assert spec.runtime.expected_model_contains == "Qwen3.5-9B-4bit"


def test_load_configs_models_preserves_generation_config(tmp_path: Path) -> None:
    config_path = tmp_path / "models.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "models": {
                    "main": {
                        "capability": "chat",
                        "provider": "qwen",
                        "protocol": "openai_compatible",
                        "model": "main-model",
                        "base_url": "http://127.0.0.1:8080/v1",
                    },
                    "mimo_cloud": {
                        "capability": "chat",
                        "provider": "mimo",
                        "protocol": "openai_compatible",
                        "model": "mimo-v2-flash",
                        "base_url": "https://api.xiaomimimo.com/v1",
                    },
                },
                "defaults": {"primary_model": "main"},
                "generation": {
                    "factcheck": {
                        "model": "mimo_cloud",
                        "max_tokens": 2048,
                        "temperature": 0.3,
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    config = ModelRegistry._load_yaml_file(config_path)

    assert config.generation.factcheck.model == "mimo_cloud"
    assert config.generation.factcheck.max_tokens == 2048
    assert config.generation.factcheck.temperature == 0.3


def test_from_env_loads_dotenv_before_resolving_model_config(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = tmp_path / "models.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "models": {
                    "mimo_cloud": {
                        "capability": "chat",
                        "provider": "mimo",
                        "protocol": "openai_compatible",
                        "model": "mimo-v2-flash",
                        "base_url": "https://api.xiaomimimo.com/v1",
                        "api_key_env": "MIMO_API_KEY",
                    }
                },
                "defaults": {"primary_model": "mimo_cloud"},
            }
        ),
        encoding="utf-8",
    )
    env_path = tmp_path / ".env"
    env_path.write_text(
        f"RAG_AGENT_MODELS_PATH={config_path}\nMIMO_API_KEY=sk-dotenv\n",
        encoding="utf-8",
    )
    monkeypatch.delenv("RAG_AGENT_MODELS_PATH", raising=False)
    monkeypatch.delenv("MIMO_API_KEY", raising=False)

    registry = ModelRegistry.from_env(env_path=str(env_path))
    spec = registry._config.models["mimo_cloud"]
    provider_config = _chat_provider_config(spec)

    assert registry.default_model == "mimo_cloud"
    assert provider_config.api_key == "sk-dotenv"
    # _load_env_file writes directly to os.environ, so release monkeypatch's
    # original snapshots before removing those dynamically-created values.
    monkeypatch.undo()
    import os

    os.environ.pop("RAG_AGENT_MODELS_PATH", None)
    os.environ.pop("MIMO_API_KEY", None)


class TestModelRegistryProperties:
    def test_default_model(self) -> None:
        reg = ModelRegistry(_make_config(default_model="main"))
        assert reg.default_model == "main"

    def test_fallback_model_none(self) -> None:
        reg = ModelRegistry(_make_config())
        assert reg.fallback_model is None

    def test_fallback_model_set(self) -> None:
        reg = ModelRegistry(_make_config(fallback_model="fast"))
        assert reg.fallback_model == "fast"


class TestModelRegistryResolve:
    def test_unknown_alias_raises(self) -> None:
        reg = ModelRegistry(_make_config())
        with pytest.raises(UnknownModelAliasError):
            reg.resolve("nonexistent")

    def test_resolve_definition_uses_complete_frozen_values_without_alias_lookup(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        source_config = _make_config()
        source_config.models["main"] = ModelSpec(
            provider=ModelProvider.OPENAI_COMPATIBLE,
            provider_name="frozen-provider",
            protocol="openai_compatible",
            model="provider/frozen-v1",
            tokenizer_model="frozen-tokenizer",
            max_tokens=777,
            timeout_seconds=45.5,
            base_url="http://127.0.0.1:9090/v1",
            defaults={
                "temperature": 0.25,
                "top_p": 0.8,
                "parallel_tool_calls": False,
                "seed": 9,
            },
            context_window_tokens=65_536,
            request_context_tokens=12_345,
            supports_tools=False,
            supports_structured_output=False,
            location="local",
            runtime={
                "health_url": "http://127.0.0.1:9090/health",
                "launch_command": ["frozen-server", "--port", "9090"],
                "expected_model_contains": "frozen-v1",
                "startup_timeout_seconds": 17.0,
                "poll_interval_seconds": 0.25,
            },
        )
        source_config.generation = replace(
            source_config.generation,
            answer=GenerationTaskConfig(
                model="main",
                max_tokens=701,
                temperature=0.15,
            ),
        )
        source_config.llm_stage_budgets = {
            stage: LLMStageBudget(
                max_input_tokens=10_000 + index,
                max_output_tokens=1_000 + index,
                safety_margin_tokens=100 + index,
            )
            for index, stage in enumerate(LLMCallStage)
        }
        definition = ModelRegistry(source_config).get_model_definition("main")
        target = ModelRegistry(_make_config())
        observed: dict[str, ModelSpec] = {}
        generator = object()

        def build_frozen(spec: ModelSpec) -> object:
            observed["spec"] = spec.model_copy(deep=True)
            return generator

        monkeypatch.setattr(llm_registry_module, "_build_chat_generator", build_frozen)

        resolved = target.resolve_definition(definition)

        frozen_spec = observed["spec"]
        assert frozen_spec.model == "provider/frozen-v1"
        assert frozen_spec.tokenizer_model == "frozen-tokenizer"
        assert frozen_spec.timeout_seconds == 45.5
        assert frozen_spec.runtime is not None
        assert frozen_spec.runtime.launch_command == ("frozen-server", "--port", "9090")
        assert resolved.generator is generator
        assert resolved.kwargs == {
            "max_tokens": 777,
            "temperature": 0.25,
            "top_p": 0.8,
            "parallel_tool_calls": False,
            "seed": 9,
        }
        assert resolved.context_window_tokens == 12_345
        assert resolved.model == "provider/frozen-v1"
        assert resolved.provider == "frozen-provider"
        assert resolved.supports_native_tools is False
        assert resolved.definition_revision == definition.definition_revision
        assert resolved.generation_config is not None
        assert resolved.generation_config.answer.max_tokens == 701
        assert resolved.gateway is not None
        for stage in LLMCallStage:
            expected = definition.llm_stage_budgets[stage.value]
            actual = resolved.gateway.stage_budget(stage)
            assert actual.model_dump() == expected.model_dump()

    def test_resolve_definition_reloads_current_credential_for_each_dispatch(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        config = AgentModelsConfig(
            models={
                "cloud": ModelSpec(
                    provider=ModelProvider.OPENAI_COMPATIBLE,
                    model="cloud-model",
                    base_url="https://api.example.com/v1",
                    api_key_env="FROZEN_TEST_API_KEY",
                    location="cloud",
                )
            },
            default_model="cloud",
        )
        registry = ModelRegistry(config)
        definition = registry.get_model_definition("cloud")
        observed_keys: list[str | None] = []

        def capture_credential(spec: ModelSpec) -> object:
            observed_keys.append(_chat_provider_config(spec).api_key)
            return object()

        monkeypatch.setattr(llm_registry_module, "_build_chat_generator", capture_credential)
        monkeypatch.setenv("FROZEN_TEST_API_KEY", "first-secret")
        first = registry.resolve_definition(definition)
        monkeypatch.setenv("FROZEN_TEST_API_KEY", "rotated-secret")
        second = registry.resolve_definition(definition)

        assert first.generator is not second.generator
        assert observed_keys == ["first-secret", "rotated-secret"]

    def test_resolve_definition_does_not_echo_provider_secret_errors(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        registry = ModelRegistry(_make_config())
        definition = registry.get_model_definition("main")

        def fail_with_secret(_spec: ModelSpec) -> object:
            raise RuntimeError("provider rejected secret-value")

        monkeypatch.setattr(llm_registry_module, "_build_chat_generator", fail_with_secret)
        with pytest.raises(ModelNotAvailableError) as captured:
            registry.resolve_definition(definition)

        assert "secret-value" not in str(captured.value)

    def test_resolve_returns_generator(self) -> None:
        reg = ModelRegistry(_make_config())
        resolved = reg.resolve("main")
        assert resolved.generator is not None
        assert resolved.kwargs["max_tokens"] == 2048
        assert resolved.context_window_tokens == 32768
        assert resolved.gateway is not None
        assert resolved.token_accounting is resolved.gateway.token_accounting

    def test_caches_same_alias(self) -> None:
        reg = ModelRegistry(_make_config())
        r1 = reg.resolve("main")
        r2 = reg.resolve("main")
        assert r1.generator is r2.generator

    def test_kwargs_include_model_defaults(self) -> None:
        spec = ModelSpec(
            provider=ModelProvider.OLLAMA,
            model="x",
            max_tokens=512,
            defaults={"temperature": 0.3, "top_p": 0.8},
        )
        config = AgentModelsConfig(
            models={"test": spec},
            default_model="test",
        )
        reg = ModelRegistry(config)
        resolved = reg.resolve("test")
        assert resolved.kwargs["max_tokens"] == 512
        assert resolved.kwargs["temperature"] == 0.3
        assert resolved.kwargs["top_p"] == 0.8

    def test_request_context_limit_caps_runtime_without_rewriting_model_window(
        self,
    ) -> None:
        spec = ModelSpec(
            provider=ModelProvider.OLLAMA,
            model="request-capped",
            context_window_tokens=131_072,
            request_context_tokens=8_000,
        )
        registry = ModelRegistry(
            AgentModelsConfig(
                models={"capped": spec},
                default_model="capped",
            )
        )

        resolved = registry.resolve("capped")

        assert registry.get_model_spec("capped").context_window_tokens == 131_072
        assert resolved.context_window_tokens == 8_000
        assert (
            resolved.gateway.effective_stage_budget(
                LLMCallStage.TOOL_DECISION,
                kwargs={"max_tokens": 2_048},
            ).max_input_tokens
            == 5_440
        )

    def test_explicit_tokenizer_model_avoids_provider_id_fallback_counting(
        self,
    ) -> None:
        spec = ModelSpec(
            provider=ModelProvider.OLLAMA,
            model="provider/gpt-oss-120b",
            tokenizer_model="gpt-oss-120b",
        )
        registry = ModelRegistry(
            AgentModelsConfig(
                models={"tokenized": spec},
                default_model="tokenized",
            )
        )

        resolved = registry.resolve("tokenized")

        assert resolved.token_accounting is not None
        assert resolved.token_accounting.backend_descriptor() == (
            "tiktoken",
            "gpt-oss-120b",
        )
        assert resolved.token_accounting.count('{"tools":[{"name":"read_file"}]}') > 4

    def test_explicit_model_output_limit_expands_tool_decision_stage_budget(
        self,
    ) -> None:
        spec = ModelSpec(
            provider=ModelProvider.OLLAMA,
            model="long-thinking",
            max_tokens=32_768,
            context_window_tokens=131_072,
            request_context_tokens=65_536,
        )
        registry = ModelRegistry(
            AgentModelsConfig(
                models={"long": spec},
                default_model="long",
            )
        )

        resolved = registry.resolve("long")
        budget = resolved.gateway.effective_stage_budget(
            LLMCallStage.TOOL_DECISION,
            kwargs={"max_tokens": resolved.kwargs["max_tokens"]},
        )

        assert budget.max_output_tokens == 32_768
        assert budget.max_input_tokens > 30_000


class TestModelRegistryResolveOrFallback:
    def test_falls_back_when_alias_unknown(self) -> None:
        reg = ModelRegistry(_make_config(fallback_model="fast"))
        resolved = reg.resolve_or_fallback("missing")
        # 应该降级到 fallback_model="fast"
        assert resolved.generator is not None

    def test_does_not_infinite_loop_when_fallback_also_missing(self) -> None:
        reg = ModelRegistry(_make_config())
        with pytest.raises(UnknownModelAliasError):
            reg.resolve_or_fallback("missing")


class TestModelRegistryResolveForNode:
    def test_uses_explicit_node_model(self) -> None:
        config = _make_config(fallback_model="fast")
        reg = ModelRegistry(config)
        resolved = reg.resolve_for_node(node_model="fast", node_name="retrieval_hint")
        assert resolved.generator is not None

    def test_uses_default_when_node_model_is_none(self) -> None:
        reg = ModelRegistry(_make_config())
        resolved = reg.resolve_for_node(node_model=None, node_name="retrieval_hint")
        assert resolved.generator is not None
