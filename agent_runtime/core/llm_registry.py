from __future__ import annotations

import json
import os
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
from importlib.resources import as_file, files
from pathlib import Path
from typing import Any, Literal, Protocol

from agent_runtime.core.llm_config import AgentModelsConfig, ModelProvider, ModelSpec
from agent_runtime.model_config_io import discover_git_worktree
from agent_runtime.model_definition import (
    ModelExecutionDefinition,
    build_model_execution_definition,
    effective_stage_budgets,
)
from agent_runtime.model_registry import UserModelDefinition, UserModelRegistryStore
from agent_runtime.modeling.config import GenerationConfig, GenerationTaskConfig
from agent_runtime.modeling.contracts import parse_llm_stage_budgets


class UnknownModelAliasError(KeyError):
    """别名在 models 中不存在。"""


class ModelNotAvailableError(RuntimeError):
    """模型构造失败（加载出错等）。"""


@dataclass(slots=True)
class ResolvedModel:
    generator: object
    kwargs: dict[str, Any]
    context_window_tokens: int = 32_768
    gateway: Any | None = None
    token_accounting: Any | None = None
    provider: str = "openai-compatible"
    model: str = "agent-model"
    supports_native_tools: bool = True


@dataclass(frozen=True, slots=True)
class ChatProviderConfig:
    base_url: str
    api_key: str | None


class ModelResolver(Protocol):
    @property
    def default_model(self) -> str: ...

    @property
    def fallback_model(self) -> str | None: ...

    @property
    def generation_config(self) -> GenerationConfig: ...

    def resolve(self, alias: str) -> ResolvedModel: ...

    def resolve_or_fallback(self, alias: str) -> ResolvedModel: ...

    def resolve_for_node(
        self,
        *,
        node_model: str | None,
        node_name: str,
    ) -> ResolvedModel: ...


class ModelRegistry:
    """按 alias 解析并缓存 Generator 实例。

    加载顺序：RAG_AGENT_MODELS_PATH(YAML) > RAG_AGENT_MODELS(JSON) > models.yaml 内置默认
    """

    _BUNDLED_CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "models.yaml"
    _BUNDLED_CONFIG_PACKAGE = "agent_runtime"
    _BUNDLED_CONFIG_RESOURCE = ("_data", "models.yaml")

    def __init__(
        self,
        config: AgentModelsConfig,
        *,
        origins: Mapping[str, Literal["builtin", "user", "override"]] | None = None,
    ) -> None:
        self._config = config.model_copy(deep=True)
        if origins is not None and set(origins) != set(self._config.models):
            raise ValueError("model origins must cover exactly the configured models")
        default_origin: Literal["builtin", "user", "override"] = "override"
        self._origins = {
            alias: origins[alias] if origins is not None else default_origin
            for alias in self._config.models
        }
        self._definitions = {
            alias: build_model_execution_definition(spec=spec, config=self._config)
            for alias, spec in self._config.models.items()
        }
        for definition in self._definitions.values():
            # Validate canonical JSON at the catalog boundary so unsupported
            # values fail before a model can be selected or dispatched.
            _ = definition.definition_revision
        self._cache: dict[str, ResolvedModel] = {}

    @property
    def default_model(self) -> str:
        return self._config.default_model

    @property
    def fallback_model(self) -> str | None:
        return self._config.fallback_model

    @property
    def generation_config(self) -> GenerationConfig:
        return self._config.generation

    @property
    def model_ids(self) -> tuple[str, ...]:
        return tuple(self._config.models)

    def get_model_spec(self, alias: str) -> ModelSpec:
        spec = self._config.models.get(alias)
        if spec is None:
            raise UnknownModelAliasError(f"Model alias {alias!r} not found in config")
        return spec.model_copy(deep=True)

    def origin(self, alias: str) -> Literal["builtin", "user", "override"]:
        try:
            return self._origins[alias]
        except KeyError as exc:
            raise UnknownModelAliasError(f"Model alias {alias!r} not found in config") from exc

    def get_model_definition(self, alias: str) -> ModelExecutionDefinition:
        try:
            return self._definitions[alias].model_copy(deep=True)
        except KeyError as exc:
            raise UnknownModelAliasError(f"Model alias {alias!r} not found in config") from exc

    @classmethod
    def from_env(cls, env_path: str = ".env", *, default_model: str | None = None) -> ModelRegistry:
        _load_env_file(Path(env_path))
        config, origins = cls._load_effective_config()
        if default_model is not None:
            if default_model not in config.models:
                raise UnknownModelAliasError(f"Model alias {default_model!r} not found in config")
            config = config.model_copy(
                update={
                    "default_model": default_model,
                    "fallback_model": default_model,
                }
            )
        return cls(config, origins=origins)

    @classmethod
    def _load_effective_config(
        cls,
    ) -> tuple[AgentModelsConfig, dict[str, Literal["builtin", "user", "override"]]]:
        if os.environ.get("RAG_AGENT_MODELS_PATH") or os.environ.get("RAG_AGENT_MODELS"):
            override = cls._load_config()
            return override, {alias: "override" for alias in override.models}

        built_in = cls._load_config()
        workspace = Path.cwd().resolve()
        registry_path = _user_registry_path()
        store = UserModelRegistryStore(
            path=registry_path,
            workspace=workspace,
            worktree=discover_git_worktree(workspace),
            built_in_aliases=built_in.models,
            whole_catalog_override_active=False,
        )
        user_snapshot = store.read()
        collisions = sorted(set(built_in.models).intersection(user_snapshot.document.models))
        if collisions:
            raise ValueError(
                "User model registry collides with built-in aliases: " + ", ".join(collisions)
            )
        models = dict(built_in.models)
        models.update(
            {
                alias: _user_definition_to_model_spec(definition)
                for alias, definition in user_snapshot.document.models.items()
            }
        )
        effective = built_in.model_copy(update={"models": models}, deep=True)
        origins: dict[str, Literal["builtin", "user", "override"]] = {
            alias: "builtin" for alias in built_in.models
        }
        origins.update({alias: "user" for alias in user_snapshot.document.models})
        return effective, origins

    @classmethod
    def _load_config(cls) -> AgentModelsConfig:
        # 1. RAG_AGENT_MODELS_PATH → YAML 文件
        yaml_path = os.environ.get("RAG_AGENT_MODELS_PATH")
        if yaml_path:
            return cls._load_yaml_file(Path(yaml_path))

        # 2. RAG_AGENT_MODELS → JSON 字符串
        json_text = os.environ.get("RAG_AGENT_MODELS")
        if json_text:
            return AgentModelsConfig.model_validate(json.loads(json_text))

        # 3. wheel 中 force-included 的 models.yaml
        resource = files(cls._BUNDLED_CONFIG_PACKAGE)
        for part in cls._BUNDLED_CONFIG_RESOURCE:
            resource = resource.joinpath(part)
        if resource.is_file():
            with as_file(resource) as resource_path:
                return cls._load_yaml_file(resource_path)

        # 4. 源码仓库中的 models.yaml（不依赖进程 cwd）
        if cls._BUNDLED_CONFIG_PATH.is_file():
            return cls._load_yaml_file(cls._BUNDLED_CONFIG_PATH)

        raise FileNotFoundError(
            "No agent model config found. Set RAG_AGENT_MODELS_PATH, "
            "RAG_AGENT_MODELS, or install a distribution with bundled models.yaml."
        )

    @staticmethod
    def _load_yaml_file(path: Path) -> AgentModelsConfig:
        import yaml

        with open(path, encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}

        # Support configs/models.yaml: models keyed by alias plus defaults.
        raw_models = data.get("models", {})
        defaults = data.get("defaults", {})
        raw_providers = data.get("providers", {})
        providers = raw_providers if isinstance(raw_providers, dict) else {}

        agent_models: dict[str, dict[str, object]] = {}
        for alias, entry in raw_models.items():
            if not isinstance(entry, dict):
                continue
            if entry.get("capability") != "chat":
                continue
            merged = _merge_provider_model_entry(
                alias=str(alias),
                entry=entry,
                providers=providers,
            )
            cost = entry.get("cost")
            if not isinstance(cost, dict):
                cost = {}
            agent_models[alias] = {
                "provider": _agent_provider_kind(merged),
                "provider_name": entry.get("provider"),
                "protocol": merged.get("protocol"),
                "model": entry["model"],
                "tokenizer_model": entry.get("tokenizer_model"),
                "max_tokens": entry.get("max_tokens", 2048),
                "defaults": entry.get("defaults", {}),
                "base_url": merged.get("base_url"),
                "api_key_env": merged.get("api_key_env"),
                "context_window_tokens": entry.get("context_window_tokens", 32_768),
                "request_context_tokens": entry.get("request_context_tokens"),
                "supports_tools": entry.get("tools", entry.get("supports_tools", True)),
                "supports_structured_output": entry.get(
                    "structured_output",
                    entry.get("supports_structured_output", True),
                ),
                "location": merged.get("location"),
                "input_cost_per_1m": cost.get("input_per_1m"),
                "output_cost_per_1m": cost.get("output_per_1m"),
                "cache_read_cost_per_1m": cost.get("cache_read_per_1m"),
                "cache_write_cost_per_1m": cost.get("cache_write_per_1m"),
                "runtime": merged.get("runtime"),
            }

        default_model = defaults.get("primary_model", "")
        if not default_model and agent_models:
            default_model = next(iter(agent_models))

        return AgentModelsConfig.model_validate(
            {
                "version": data.get("version", 1),
                "models": agent_models,
                "default_model": default_model,
                "fallback_model": data.get("fallback_model", default_model),
                "generation": _parse_generation_config(data.get("generation")),
                "llm_stage_budgets": parse_llm_stage_budgets(data.get("llm_budgets")),
            }
        )

    def resolve(self, alias: str) -> ResolvedModel:
        """别名 → (Generator, kwargs)。按 alias 缓存，同 alias 多次调用返回同一 Generator。"""
        from agent_runtime.modeling.gateway import LLMGateway
        from agent_runtime.modeling.tokenization import TokenAccountingService, TokenizerContract

        if alias in self._cache:
            return self._cache[alias]

        spec = self._config.models.get(alias)
        if spec is None:
            raise UnknownModelAliasError(f"Model alias {alias!r} not found in config")

        try:
            generator = _build_chat_generator(spec)
        except Exception as exc:
            raise ModelNotAvailableError(f"Failed to build provider for {alias!r}: {exc}") from exc

        kwargs: dict[str, Any] = {
            "max_tokens": spec.max_tokens,
            **deepcopy(spec.defaults),
        }
        runtime_context_tokens = min(
            spec.context_window_tokens,
            spec.request_context_tokens or spec.context_window_tokens,
        )
        token_accounting = TokenAccountingService(
            TokenizerContract(
                embedding_model_name=spec.model,
                tokenizer_model_name=spec.tokenizer_model or spec.model,
                chunking_tokenizer_model_name=(spec.tokenizer_model or spec.model),
                tokenizer_backend="auto",
                max_context_tokens=runtime_context_tokens,
                prompt_reserved_tokens=512,
                local_files_only=True,
            )
        )
        stage_budgets = effective_stage_budgets(config=self._config, spec=spec)
        resolved = ResolvedModel(
            generator=generator,
            kwargs=kwargs,
            context_window_tokens=runtime_context_tokens,
            gateway=LLMGateway(
                generator=generator,
                token_accounting=token_accounting,
                model_context_tokens=runtime_context_tokens,
                stage_budgets=stage_budgets,
            ),
            token_accounting=token_accounting,
            provider=spec.provider_name or spec.provider.value,
            model=spec.model,
            supports_native_tools=spec.supports_tools,
        )
        self._cache[alias] = resolved
        return resolved

    def resolve_or_fallback(self, alias: str) -> ResolvedModel:
        """尝试解析 alias，失败时降级到 fallback_model。"""
        try:
            return self.resolve(alias)
        except (UnknownModelAliasError, ModelNotAvailableError):
            if self._config.fallback_model and alias != self._config.fallback_model:
                return self.resolve(self._config.fallback_model)
            raise

    def resolve_for_node(
        self,
        *,
        node_model: str | None,
        node_name: str,
    ) -> ResolvedModel:
        """根据节点指定的 model alias（可为 None）解析 Generator。

        node_model 非空 → 直接用该 alias（失败降级到 fallback）
        node_model 为空 → 用 default_model（失败降级到 fallback）
        """
        alias = node_model or self._config.default_model
        return self.resolve_or_fallback(alias)


def _build_chat_generator(spec: ModelSpec) -> object:
    """Construct only the chat capability required by AgentRuntime.

    Embedding and reranking construction remains owned by RAG assembly.
    Imports are intentionally local so importing the runtime never imports
    optional HTTP clients or model backends.
    """

    config = _chat_provider_config(spec)
    if spec.provider in {ModelProvider.MLX, ModelProvider.OPENAI_COMPATIBLE}:
        from agent_runtime.modeling.chat import OpenAICompatibleChatGenerator

        return OpenAICompatibleChatGenerator(
            model=spec.model,
            base_url=config.base_url,
            api_key=config.api_key,
            supports_tools=spec.supports_tools,
        )
    if spec.provider is ModelProvider.OLLAMA:
        from agent_runtime.modeling.providers.ollama.generator import OllamaGenerator

        return OllamaGenerator(
            base_url=config.base_url,
            default_model=spec.model,
            timeout_seconds=spec.timeout_seconds,
        )
    raise ValueError(f"Unsupported provider: {spec.provider}")


def _chat_provider_config(spec: ModelSpec) -> ChatProviderConfig:
    if spec.provider is ModelProvider.OLLAMA:
        return ChatProviderConfig(base_url=spec.base_url or "http://localhost:11434", api_key=None)
    if spec.provider in {ModelProvider.MLX, ModelProvider.OPENAI_COMPATIBLE}:
        return ChatProviderConfig(
            base_url=spec.base_url or "http://127.0.0.1:8080/v1",
            api_key=_api_key_from_env(spec.api_key_env),
        )
    raise ValueError(f"Unsupported provider: {spec.provider}")


def _merge_provider_model_entry(
    *,
    alias: str,
    entry: dict[str, object],
    providers: dict[object, object],
) -> dict[str, object]:
    """Merge connection-level provider fields into a model declaration.

    Newer configs split ``providers`` (how to connect) from ``models`` (what to
    run). Older configs keep those fields directly on the model; those continue
    to work because model fields override provider fields.
    """
    provider_ref = entry.get("provider")
    provider_entry = providers.get(provider_ref)
    provider = provider_entry if isinstance(provider_entry, dict) else {}
    merged: dict[str, object] = {
        "provider": provider_ref,
        "protocol": _first_present(entry, provider, "protocol"),
        "base_url": _first_present(entry, provider, "base_url"),
        "api_key_env": _first_present(entry, provider, "api_key_env"),
        "location": _first_present(entry, provider, "location"),
    }
    runtime = _merge_runtime_config(
        alias=alias,
        model=str(entry.get("model", "")),
        provider_runtime=provider.get("runtime"),
        model_runtime=entry.get("runtime"),
    )
    if runtime:
        merged["runtime"] = runtime
    return merged


def _first_present(
    primary: dict[str, object],
    fallback: dict[object, object],
    key: str,
) -> object | None:
    value = primary.get(key)
    if value is not None:
        return value
    fallback_value = fallback.get(key)
    return fallback_value if fallback_value is not None else None


def _merge_runtime_config(
    *,
    alias: str,
    model: str,
    provider_runtime: object,
    model_runtime: object,
) -> dict[str, object]:
    provider = provider_runtime if isinstance(provider_runtime, dict) else {}
    model_specific = model_runtime if isinstance(model_runtime, dict) else {}
    runtime: dict[str, object] = {}
    for key in (
        "health_url",
        "expected_model_contains",
        "startup_timeout_seconds",
        "poll_interval_seconds",
    ):
        value = _first_present(model_specific, provider, key)
        if value is not None:
            runtime[key] = value

    launch_command = _first_present(model_specific, provider, "launch_command")
    if launch_command is None:
        template = _first_present(model_specific, provider, "launch_command_template")
        if isinstance(template, list | tuple):
            launch_command = _expand_launch_command_template(
                template,
                alias=alias,
                model=model,
            )
    if launch_command is not None:
        runtime["launch_command"] = launch_command
    return runtime


def _expand_launch_command_template(
    template: list[object] | tuple[object, ...],
    *,
    alias: str,
    model: str,
) -> list[str]:
    return [str(part).replace("{model}", model).replace("{alias}", alias) for part in template]


def _agent_provider_kind(entry: dict[str, object]) -> str:
    protocol = _normalized_provider_value(entry.get("protocol"))
    provider = _normalized_provider_value(entry.get("provider"))
    if protocol == "openai_compatible":
        return ModelProvider.OPENAI_COMPATIBLE.value
    if provider in {"openai_compatible", "qwen", "deepseek", "groq", "mimo"}:
        return ModelProvider.OPENAI_COMPATIBLE.value
    if provider == "ollama":
        return ModelProvider.OLLAMA.value
    if provider == "mlx":
        return ModelProvider.MLX.value
    return provider


def _normalized_provider_value(value: object) -> str:
    return str(value or "").strip().lower().replace("-", "_")


def _parse_generation_config(raw: object) -> GenerationConfig:
    if not isinstance(raw, dict):
        return GenerationConfig()

    def parse_task(name: str) -> GenerationTaskConfig:
        entry = raw.get(name)
        if not isinstance(entry, dict):
            return GenerationTaskConfig()
        return GenerationTaskConfig(
            model=entry.get("model"),
            max_tokens=int(entry["max_tokens"]) if "max_tokens" in entry else None,
            temperature=float(entry["temperature"]) if "temperature" in entry else None,
        )

    return GenerationConfig(
        summary=parse_task("summary"),
        answer=parse_task("answer"),
        planner=parse_task("planner"),
        synthesize=parse_task("synthesize"),
        factcheck=parse_task("factcheck"),
    )


def _api_key_from_env(name: str | None) -> str | None:
    if not name:
        return None
    value = os.environ.get(name)
    return value.strip() if isinstance(value, str) and value.strip() else None


def _user_registry_path() -> Path:
    configured = os.environ.get("PRAXIS_MODEL_REGISTRY_PATH")
    if configured:
        return Path(configured)
    config_home = os.environ.get("XDG_CONFIG_HOME")
    root = Path(config_home) if config_home else Path.home() / ".config"
    return root / "praxis" / "models.yaml"


def _user_definition_to_model_spec(definition: UserModelDefinition) -> ModelSpec:
    runtime = (
        definition.runtime.model_dump(mode="json", exclude_none=True)
        if definition.runtime is not None
        else None
    )
    return ModelSpec.model_validate(
        {
            "provider": definition.provider,
            "model": definition.model,
            "tokenizer_model": definition.tokenizer_model,
            "provider_name": definition.provider_name,
            "protocol": definition.protocol,
            "max_tokens": definition.max_tokens,
            "timeout_seconds": definition.timeout_seconds,
            "base_url": definition.base_url,
            "api_key_env": definition.api_key_env,
            "defaults": definition.defaults.model_dump(mode="json", exclude_none=True),
            "context_window_tokens": definition.context_window_tokens,
            "request_context_tokens": definition.request_context_tokens,
            "supports_tools": definition.supports_tools,
            "supports_structured_output": definition.supports_structured_output,
            "location": definition.location,
            "input_cost_per_1m": definition.input_cost_per_1m,
            "output_cost_per_1m": definition.output_cost_per_1m,
            "cache_read_cost_per_1m": definition.cache_read_cost_per_1m,
            "cache_write_cost_per_1m": definition.cache_write_cost_per_1m,
            "runtime": runtime,
        }
    )


def _load_env_file(path: Path) -> None:
    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, raw_value = stripped.split("=", maxsplit=1)
        key = key.strip()
        if not key or key in os.environ:
            continue
        os.environ[key] = _parse_env_value(raw_value.strip())


def _parse_env_value(value: str) -> str:
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]
    return value
