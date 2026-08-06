from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from agent_runtime.core.llm_config import AgentModelsConfig, ModelProvider, ModelSpec
from agent_runtime.core.llm_registry import (
    ModelRegistry,
    UnknownModelAliasError,
    _chat_provider_config,
)
from agent_runtime.modeling.contracts import LLMCallStage


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
