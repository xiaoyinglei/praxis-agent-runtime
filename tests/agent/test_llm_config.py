from __future__ import annotations

import pytest

from agent_runtime.core.llm_config import (
    AgentModelsConfig,
    ModelProvider,
    ModelSpec,
    normalize_model_endpoint,
)


class TestModelSpec:
    def test_minimal_mlx_spec_requires_explicit_context_window(self) -> None:
        spec = ModelSpec(
            provider=ModelProvider.MLX,
            context_window_tokens=32_768,
        )

        assert spec.provider == ModelProvider.MLX
        assert spec.context_window_tokens == 32_768
        assert spec.max_output_tokens is None
        assert spec.timeout_seconds == 120.0
        assert spec.base_url is None
        assert spec.defaults == {}

    def test_context_window_is_required(self) -> None:
        with pytest.raises(
            ValueError,
            match="context_window_tokens",
        ):
            ModelSpec.model_validate(
                {
                    "provider": "mlx",
                }
            )

    def test_explicit_output_capability(self) -> None:
        spec = ModelSpec(
            provider=ModelProvider.MLX,
            context_window_tokens=32_768,
            max_output_tokens=16_384,
        )

        assert spec.context_window_tokens == 32_768
        assert spec.max_output_tokens == 16_384

    def test_ollama_spec_with_output_limit(self) -> None:
        spec = ModelSpec(
            provider=ModelProvider.OLLAMA,
            base_url="http://localhost:11434",
            context_window_tokens=32_768,
            max_output_tokens=1_024,
        )

        assert spec.provider == ModelProvider.OLLAMA
        assert spec.base_url == "http://localhost:11434"
        assert spec.max_output_tokens == 1_024

    def test_defaults_stores_temperature(self) -> None:
        spec = ModelSpec(
            provider=ModelProvider.MLX,
            context_window_tokens=32_768,
            defaults={
                "temperature": 0.0,
                "top_p": 0.9,
            },
        )

        assert spec.defaults["temperature"] == 0.0
        assert spec.defaults["top_p"] == 0.9

    def test_rejects_removed_max_context_window_field(self) -> None:
        with pytest.raises(
            ValueError,
            match="extra",
        ):
            ModelSpec.model_validate(
                {
                    "provider": "mlx",
                    "context_window_tokens": 65_536,
                    "max_context_window_tokens": 32_768,
                }
            )

    def test_rejects_output_limit_above_effective_maximum(self) -> None:
        with pytest.raises(
            ValueError,
            match="max_output_tokens",
        ):
            ModelSpec(
                provider=ModelProvider.MLX,
                context_window_tokens=32_768,
                max_output_tokens=32_769,
            )

    @pytest.mark.parametrize(
        "host",
        [
            "127.0.0.2",
            "127.1",
            "2130706433",
            "0x7f000001",
        ],
    )
    def test_legacy_loopback_spellings_have_one_location(
        self,
        host: str,
    ) -> None:
        endpoint = normalize_model_endpoint(
            provider=ModelProvider.OPENAI_COMPATIBLE,
            base_url=f"http://{host}:8080/v1",
            location=None,
        )

        assert endpoint.location == "local"

        with pytest.raises(
            ValueError,
            match="location",
        ):
            normalize_model_endpoint(
                provider=ModelProvider.OPENAI_COMPATIBLE,
                base_url=f"http://{host}:8080/v1",
                location="cloud",
            )

    def test_model_spec_rejects_unknown_fields_and_invalid_capabilities(
        self,
    ) -> None:
        with pytest.raises(
            ValueError,
            match="extra",
        ):
            ModelSpec.model_validate(
                {
                    "provider": "ollama",
                    "context_window_tokens": 32_768,
                    "headers": {"x": "secret"},
                }
            )

        with pytest.raises(
            ValueError,
            match="greater than 0",
        ):
            ModelSpec(
                provider=ModelProvider.OLLAMA,
                context_window_tokens=32_768,
                max_output_tokens=-1,
            )


class TestAgentModelsConfig:
    def test_minimal_config(self) -> None:
        spec = ModelSpec(
            provider=ModelProvider.MLX,
            context_window_tokens=32_768,
        )

        config = AgentModelsConfig(
            models={"main": spec},
            default_model="main",
        )

        assert config.default_model == "main"
        assert config.fallback_model is None
        assert config.version == 1
        assert len(config.models) == 1

    def test_config_with_fallback(self) -> None:
        main = ModelSpec(
            provider=ModelProvider.MLX,
            context_window_tokens=32_768,
        )
        fast = ModelSpec(
            provider=ModelProvider.MLX,
            context_window_tokens=16_384,
        )

        config = AgentModelsConfig(
            models={
                "main": main,
                "fast": fast,
            },
            default_model="main",
            fallback_model="fast",
        )

        assert config.fallback_model == "fast"

    def test_rejects_empty_models(self) -> None:
        with pytest.raises(
            ValueError,
            match="models must not be empty",
        ):
            AgentModelsConfig(
                default_model="missing",
            )

    def test_rejects_missing_default_model(self) -> None:
        spec = ModelSpec(
            provider=ModelProvider.MLX,
            context_window_tokens=32_768,
        )

        with pytest.raises(
            ValueError,
            match="default_model not found",
        ):
            AgentModelsConfig(
                models={"real": spec},
                default_model="missing",
            )

    def test_rejects_missing_fallback_model(self) -> None:
        spec = ModelSpec(
            provider=ModelProvider.MLX,
            context_window_tokens=32_768,
        )

        with pytest.raises(
            ValueError,
            match="fallback_model not found",
        ):
            AgentModelsConfig(
                models={"real": spec},
                default_model="real",
                fallback_model="missing",
            )

    def test_default_and_fallback_same_model_is_valid(self) -> None:
        spec = ModelSpec(
            provider=ModelProvider.MLX,
            context_window_tokens=32_768,
        )

        config = AgentModelsConfig(
            models={"shared": spec},
            default_model="shared",
            fallback_model="shared",
        )

        assert (
            config.default_model
            == config.fallback_model
            == "shared"
        )

    def test_parse_from_yaml_string(self) -> None:
        import yaml

        yaml_text = """
version: 1
models:
  Qwen3-14B-MLX-4bit:
    provider: mlx
    context_window_tokens: 32768
    max_output_tokens: 4096
  Qwen3-8B-MLX-4bit:
    provider: mlx
    context_window_tokens: 16384
default_model: Qwen3-14B-MLX-4bit
fallback_model: Qwen3-8B-MLX-4bit
"""

        data = yaml.safe_load(yaml_text)
        config = AgentModelsConfig.model_validate(data)

        assert config.default_model == "Qwen3-14B-MLX-4bit"
        assert config.fallback_model == "Qwen3-8B-MLX-4bit"

        main = config.models["Qwen3-14B-MLX-4bit"]

        assert main.provider == ModelProvider.MLX
        assert main.context_window_tokens == 32_768
        assert main.max_output_tokens == 4_096
