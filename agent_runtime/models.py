from __future__ import annotations

import json
import os
from collections.abc import Iterable, Mapping
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Literal, Protocol, cast

from agent_runtime.core.llm_config import ModelSpec as InternalModelSpec
from agent_runtime.core.llm_config import normalize_model_endpoint
from agent_runtime.core.llm_registry import (
    ModelNotAvailableError,
    ModelRegistry,
    ModelResolver,
    ResolvedModel,
    UnknownModelAliasError,
)
from agent_runtime.model_config_io import (
    ConfigVersionConflict,
    FileVersion,
    atomic_replace_bytes,
    exclusive_config_lock,
    file_fingerprint,
)
from agent_runtime.model_definition import ModelExecutionDefinition
from agent_runtime.modeling.config import GenerationConfig

ModelLocation = Literal["local", "cloud"]
ModelSwitchRequester = Literal["user", "agent", "system"]
ModelOrigin = Literal["builtin", "user", "override"]
_MISSING_SESSION_FINGERPRINT = "missing"


class LocalRuntimeReadyManager(Protocol):
    def ensure_ready(self, spec: ModelSpec) -> None: ...


class ModelPolicyError(ValueError):
    """A model switch request was rejected by policy."""


@dataclass(frozen=True, slots=True)
class ModelRuntimeSpec:
    health_url: str | None = None
    launch_command: tuple[str, ...] = ()
    expected_model_contains: str | None = None
    startup_timeout_seconds: float = 60.0
    poll_interval_seconds: float = 1.0


@dataclass(frozen=True, slots=True)
class ModelSpec:
    id: str
    provider: str
    provider_model: str
    context_window: int
    supports_tools: bool
    supports_structured_output: bool
    location: ModelLocation
    protocol: str | None = None
    base_url: str | None = None
    api_key_env: str | None = None
    max_output_tokens: int = 2048
    input_cost_per_1m: float | None = None
    output_cost_per_1m: float | None = None
    runtime: ModelRuntimeSpec | None = None


class ModelCatalog:
    """Runtime-facing model catalog built from model declarations."""

    def __init__(
        self,
        *,
        specs: Mapping[str, ModelSpec],
        default_model_id: str,
        origins: Mapping[str, ModelOrigin] | None = None,
        definitions: Mapping[str, ModelExecutionDefinition] | None = None,
    ) -> None:
        if not specs:
            raise ValueError("model catalog must not be empty")
        if default_model_id not in specs:
            raise UnknownModelAliasError(f"Default model {default_model_id!r} not found in catalog")
        if origins is not None and set(origins) != set(specs):
            raise ValueError("model catalog origins must cover exactly the model specs")
        if definitions is not None and set(definitions) != set(specs):
            raise ValueError("model catalog definitions must cover exactly the model specs")
        self._specs = MappingProxyType(dict(specs))
        self._origins = MappingProxyType(
            dict(origins) if origins is not None else {model_id: "override" for model_id in specs}
        )
        self._definitions = MappingProxyType(deepcopy(dict(definitions or {})))
        self.default_model_id = default_model_id

    @classmethod
    def from_config_file(cls, path: Path) -> ModelCatalog:
        return cls.from_registry(ModelRegistry(ModelRegistry._load_yaml_file(path)))

    @classmethod
    def from_env(cls, env_path: str = ".env") -> ModelCatalog:
        return cls.from_registry(ModelRegistry.from_env(env_path=env_path))

    @classmethod
    def from_registry(cls, registry: ModelRegistry) -> ModelCatalog:
        specs = {
            model_id: _to_public_spec(model_id, registry.get_model_spec(model_id)) for model_id in registry.model_ids
        }
        return cls(
            specs=specs,
            default_model_id=registry.default_model,
            origins={model_id: registry.origin(model_id) for model_id in registry.model_ids},
            definitions={
                model_id: registry.get_model_definition(model_id) for model_id in registry.model_ids
            },
        )

    def list_models(self) -> list[ModelSpec]:
        return [self._specs[model_id] for model_id in self._specs]

    def get(self, model_id: str) -> ModelSpec:
        try:
            return self._specs[model_id]
        except KeyError as exc:
            raise UnknownModelAliasError(f"Model alias {model_id!r} not found in catalog") from exc

    def has(self, model_id: str) -> bool:
        return model_id in self._specs

    def origin(self, model_id: str) -> ModelOrigin:
        self.get(model_id)
        return cast(ModelOrigin, self._origins[model_id])

    def definition(self, model_id: str) -> ModelExecutionDefinition:
        self.get(model_id)
        try:
            return self._definitions[model_id].model_copy(deep=True)
        except KeyError as exc:
            raise RuntimeError(
                f"Model catalog entry {model_id!r} has no execution definition"
            ) from exc


@dataclass(slots=True)
class ModelSessionState:
    """Mutable model choice for one runtime session."""

    current_model_id: str
    selection_requester: ModelSwitchRequester = "system"
    file_revision: int = 0
    fingerprint: str = _MISSING_SESSION_FINGERPRINT

    @property
    def file_version(self) -> FileVersion:
        return FileVersion(
            revision=self.file_revision,
            fingerprint=self.fingerprint,
        )

    @classmethod
    def load(cls, path: Path, *, default_model_id: str) -> ModelSessionState:
        return ModelSessionStore(path).read(default_model_id=default_model_id)

    def save(self, path: Path) -> None:
        stored = ModelSessionStore(path).select(
            self.current_model_id,
            expected=self.file_version,
        )
        self.file_revision = stored.file_revision
        self.fingerprint = stored.fingerprint


class ModelSessionStore:
    """Crash-safe, compare-and-swap storage for one selected model alias."""

    def __init__(self, path: Path) -> None:
        self.path = path

    def read(self, *, default_model_id: str) -> ModelSessionState:
        try:
            payload = self.path.read_bytes()
        except FileNotFoundError:
            return ModelSessionState(current_model_id=default_model_id)
        document, legacy = _parse_session_document(payload)
        return ModelSessionState(
            current_model_id=cast(str, document["current_model_id"]),
            selection_requester="user",
            file_revision=0 if legacy else cast(int, document["revision"]),
            fingerprint=file_fingerprint(payload),
        )

    def select(
        self,
        model_id: str,
        *,
        expected: FileVersion,
    ) -> ModelSessionState:
        if not isinstance(model_id, str) or not model_id.strip() or model_id != model_id.strip():
            raise ValueError("model session current_model_id must be a non-empty trimmed string")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with exclusive_config_lock(self.path):
            observed = self.read(default_model_id=model_id)
            if observed.file_version != expected:
                raise ConfigVersionConflict(
                    "model session changed since it was read "
                    f"(expected revision {expected.revision}, observed {observed.file_revision})"
                )
            revision = observed.file_revision + 1
            encoded = _encode_session_document(
                revision=revision,
                current_model_id=model_id,
            )
            fingerprint = file_fingerprint(encoded)
            atomic_replace_bytes(
                self.path,
                encoded,
                intended_fingerprint=fingerprint,
            )
        return ModelSessionState(
            current_model_id=model_id,
            selection_requester="user",
            file_revision=revision,
            fingerprint=fingerprint,
        )


def _parse_session_document(payload: bytes) -> tuple[dict[str, object], bool]:
    def reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
        document: dict[str, object] = {}
        for key, value in pairs:
            if key in document:
                raise ValueError(f"model session contains duplicate key {key!r}")
            document[key] = value
        return document

    try:
        value = json.loads(payload, object_pairs_hook=reject_duplicate_keys)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("model session is not valid UTF-8 JSON") from error
    if not isinstance(value, dict):
        raise ValueError("model session must be a JSON object")
    keys = set(value)
    if keys == {"current_model_id"}:
        legacy = True
    elif keys == {"version", "revision", "current_model_id"}:
        legacy = False
        version = value["version"]
        revision = value["revision"]
        if type(version) is not int or version != 1:
            raise ValueError("model session version must be 1")
        if type(revision) is not int or revision < 0:
            raise ValueError("model session revision must be a non-negative integer")
    else:
        raise ValueError("model session has unexpected or missing fields")
    current = value["current_model_id"]
    if not isinstance(current, str) or not current.strip() or current != current.strip():
        raise ValueError("model session current_model_id must be a non-empty trimmed string")
    return cast(dict[str, object], value), legacy


def _encode_session_document(*, revision: int, current_model_id: str) -> bytes:
    return (
        json.dumps(
            {
                "version": 1,
                "revision": revision,
                "current_model_id": current_model_id,
            },
            ensure_ascii=False,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")


@dataclass(frozen=True, slots=True)
class ModelPolicy:
    """Policy gate for model switches. This is not a router."""

    allowed_user_model_ids: frozenset[str] | None = None
    allowed_agent_model_ids: frozenset[str] | None = None
    allowed_system_model_ids: frozenset[str] | None = None

    def review_switch(
        self,
        *,
        catalog: ModelCatalog,
        target_model_id: str,
        requested_by: ModelSwitchRequester,
    ) -> ModelSpec:
        spec = catalog.get(target_model_id)
        allowed = self._allowed_ids_for(requested_by)
        if allowed is not None and target_model_id not in allowed:
            raise ModelPolicyError(f"Model {target_model_id!r} is not allowed for {requested_by} requests")
        return spec

    def _allowed_ids_for(
        self,
        requested_by: ModelSwitchRequester,
    ) -> frozenset[str] | None:
        if requested_by == "agent":
            return self.allowed_agent_model_ids
        if requested_by == "system":
            return self.allowed_system_model_ids
        return self.allowed_user_model_ids


class ModelControlPlane:
    """Shared facade for model catalog, session choice, policy, and resolution."""

    def __init__(
        self,
        *,
        catalog: ModelCatalog,
        state: ModelSessionState,
        policy: ModelPolicy | None = None,
        registry: ModelResolver | None = None,
        session_path: Path | None = None,
        local_runtime_manager: LocalRuntimeReadyManager | None = None,
        session_diagnostics: tuple[str, ...] = (),
    ) -> None:
        if not catalog.has(state.current_model_id):
            raise UnknownModelAliasError(f"Model alias {state.current_model_id!r} not found in catalog")
        self.catalog = catalog
        self.state = state
        self.policy = policy or ModelPolicy()
        self._registry = registry
        self._session_path = session_path
        self._session_store = ModelSessionStore(session_path) if session_path is not None else None
        self._local_runtime_manager = local_runtime_manager
        self.session_diagnostics = session_diagnostics

    @classmethod
    def from_config_file(
        cls,
        path: Path,
        *,
        initial_model_id: str | None = None,
        initial_selection_requester: ModelSwitchRequester = "system",
        session_path: Path | None = None,
        policy: ModelPolicy | None = None,
        local_runtime_manager: LocalRuntimeReadyManager | None = None,
    ) -> ModelControlPlane:
        registry = ModelRegistry(ModelRegistry._load_yaml_file(path))
        return cls.from_registry(
            registry,
            initial_model_id=initial_model_id,
            initial_selection_requester=initial_selection_requester,
            session_path=session_path,
            policy=policy,
            local_runtime_manager=local_runtime_manager,
        )

    @classmethod
    def from_env(
        cls,
        env_path: str = ".env",
        *,
        initial_model_id: str | None = None,
        initial_selection_requester: ModelSwitchRequester = "system",
        session_path: Path | None = None,
        policy: ModelPolicy | None = None,
        local_runtime_manager: LocalRuntimeReadyManager | None = None,
    ) -> ModelControlPlane:
        registry = ModelRegistry.from_env(env_path=env_path)
        return cls.from_registry(
            registry,
            initial_model_id=initial_model_id,
            initial_selection_requester=initial_selection_requester,
            session_path=session_path,
            policy=policy,
            local_runtime_manager=local_runtime_manager,
        )

    @classmethod
    def from_registry(
        cls,
        registry: ModelRegistry,
        *,
        initial_model_id: str | None = None,
        initial_selection_requester: ModelSwitchRequester = "system",
        session_path: Path | None = None,
        policy: ModelPolicy | None = None,
        local_runtime_manager: LocalRuntimeReadyManager | None = None,
    ) -> ModelControlPlane:
        catalog = ModelCatalog.from_registry(registry)
        state, diagnostics = _load_session_state(
            catalog=catalog,
            initial_model_id=initial_model_id,
            initial_selection_requester=initial_selection_requester,
            session_path=session_path,
        )
        return cls(
            catalog=catalog,
            state=state,
            policy=policy,
            registry=registry,
            session_path=session_path,
            local_runtime_manager=local_runtime_manager,
            session_diagnostics=diagnostics,
        )

    @property
    def default_model(self) -> str:
        return self.state.current_model_id

    @property
    def fallback_model(self) -> str | None:
        if self._registry is None:
            return None
        return self._registry.fallback_model

    @property
    def generation_config(self) -> GenerationConfig:
        if self._registry is None:
            raise RuntimeError("Model resolver is not configured")
        return self._registry.generation_config

    def list_models(self) -> list[ModelSpec]:
        return self.catalog.list_models()

    def current_model(self) -> ModelSpec:
        return self.catalog.get(self.state.current_model_id)

    def switch_model(
        self,
        model_id: str,
        *,
        requested_by: ModelSwitchRequester,
        persist: bool = True,
    ) -> ModelSpec:
        spec = self.policy.review_switch(
            catalog=self.catalog,
            target_model_id=model_id,
            requested_by=requested_by,
        )
        persisted = None
        if persist and self._session_store is not None:
            persisted = self._session_store.select(
                spec.id,
                expected=self.state.file_version,
            )
        self.state.current_model_id = spec.id
        self.state.selection_requester = requested_by
        if persisted is not None:
            self.state.file_revision = persisted.file_revision
            self.state.fingerprint = persisted.fingerprint
        return spec

    def request_model_switch(self, model_id: str) -> ModelSpec:
        return self.switch_model(model_id, requested_by="agent")

    def resolve(self, alias: str) -> ResolvedModel:
        if self._registry is None:
            raise RuntimeError("Model resolver is not configured")
        spec = self.catalog.get(alias)
        self._ensure_model_ready(spec)
        return self._registry.resolve(alias)

    def resolve_or_fallback(self, alias: str) -> ResolvedModel:
        return self.resolve(alias)

    def resolve_for_node(
        self,
        *,
        node_model: str | None,
        node_name: str,
    ) -> ResolvedModel:
        del node_name
        model_id = node_model or self.state.current_model_id
        return self.resolve(model_id)

    def _ensure_model_ready(self, spec: ModelSpec) -> None:
        if spec.location == "local":
            manager = self._local_runtime_manager
            if manager is None:
                from agent_runtime.local_runtime import LocalRuntimeManager

                manager = cast(LocalRuntimeReadyManager, LocalRuntimeManager())
                self._local_runtime_manager = manager
            manager.ensure_ready(spec)
            return
        if spec.api_key_env:
            value = os.environ.get(spec.api_key_env)
            if not isinstance(value, str) or not value.strip():
                raise ModelNotAvailableError(
                    f"Model {spec.id!r} is unavailable because environment variable "
                    f"{spec.api_key_env} is not set. Export it or add it to .env; "
                    "use AGENT_ENV_FILE to select a different env file"
                )

    def close(self) -> None:
        manager = self._local_runtime_manager
        close = getattr(manager, "close", None)
        if callable(close):
            close()


def _load_session_state(
    *,
    catalog: ModelCatalog,
    initial_model_id: str | None,
    initial_selection_requester: ModelSwitchRequester,
    session_path: Path | None,
) -> tuple[ModelSessionState, tuple[str, ...]]:
    store = ModelSessionStore(session_path) if session_path is not None else None
    if initial_model_id is not None:
        persisted = (
            store.read(default_model_id=catalog.default_model_id)
            if store is not None
            else ModelSessionState(current_model_id=catalog.default_model_id)
        )
        state = ModelSessionState(
            current_model_id=initial_model_id,
            selection_requester=initial_selection_requester,
            file_revision=persisted.file_revision,
            fingerprint=persisted.fingerprint,
        )
    elif store is not None:
        state = store.read(default_model_id=catalog.default_model_id)
    else:
        state = ModelSessionState(current_model_id=catalog.default_model_id)
    if catalog.has(state.current_model_id):
        return state, ()
    if initial_model_id is not None or store is None:
        raise UnknownModelAliasError(f"Model alias {state.current_model_id!r} not found in catalog")
    stale_model_id = state.current_model_id
    try:
        repaired = store.select(catalog.default_model_id, expected=state.file_version)
    except ConfigVersionConflict:
        newer = store.read(default_model_id=catalog.default_model_id)
        if catalog.has(newer.current_model_id):
            return newer, (
                f"Persisted model {stale_model_id!r} became stale; preserved newer selection "
                f"{newer.current_model_id!r}.",
            )
        repaired = store.select(catalog.default_model_id, expected=newer.file_version)
    repaired.selection_requester = "user"
    return repaired, (
        f"Persisted model {stale_model_id!r} is unavailable; repaired selection to "
        f"effective default {catalog.default_model_id!r}.",
    )


def _to_public_spec(
    model_id: str,
    spec: InternalModelSpec,
) -> ModelSpec:
    provider = str(spec.provider_name or spec.provider)
    provider_model = str(spec.model)
    endpoint = normalize_model_endpoint(
        provider=spec.provider,
        base_url=spec.base_url,
        location=spec.location,
    )
    return ModelSpec(
        id=model_id,
        provider=provider,
        provider_model=provider_model,
        context_window=int(spec.context_window_tokens),
        supports_tools=bool(spec.supports_tools),
        supports_structured_output=bool(spec.supports_structured_output),
        location=endpoint.location,
        protocol=spec.protocol,
        base_url=endpoint.base_url,
        api_key_env=spec.api_key_env,
        max_output_tokens=int(spec.max_tokens),
        input_cost_per_1m=spec.input_cost_per_1m,
        output_cost_per_1m=spec.output_cost_per_1m,
        runtime=_to_public_runtime_spec(spec.runtime),
    )


def _to_public_runtime_spec(runtime: object | None) -> ModelRuntimeSpec | None:
    if runtime is None:
        return None
    launch_command = getattr(runtime, "launch_command", ()) or ()
    return ModelRuntimeSpec(
        health_url=getattr(runtime, "health_url", None),
        launch_command=tuple(str(part) for part in launch_command),
        expected_model_contains=getattr(runtime, "expected_model_contains", None),
        startup_timeout_seconds=float(getattr(runtime, "startup_timeout_seconds", 60.0)),
        poll_interval_seconds=float(getattr(runtime, "poll_interval_seconds", 1.0)),
    )


def format_model_rows(
    specs: Iterable[ModelSpec],
    *,
    current_model_id: str,
) -> list[str]:
    lines = []
    for spec in specs:
        marker = "*" if spec.id == current_model_id else " "
        caps = []
        if spec.supports_tools:
            caps.append("tools")
        if spec.supports_structured_output:
            caps.append("structured")
        cap_text = ",".join(caps) if caps else "-"
        cost = "-"
        if spec.input_cost_per_1m is not None or spec.output_cost_per_1m is not None:
            cost = f"{spec.input_cost_per_1m or 0:g}/{spec.output_cost_per_1m or 0:g}"
        lines.append(
            f"{marker} {spec.id}  provider={spec.provider}  "
            f"model={spec.provider_model}  ctx={spec.context_window}  "
            f"{spec.location}  caps={cap_text}  cost={cost}"
        )
    return lines


__all__ = [
    "ModelCatalog",
    "ModelControlPlane",
    "ModelLocation",
    "ModelNotAvailableError",
    "ModelOrigin",
    "ModelPolicy",
    "ModelPolicyError",
    "ModelRuntimeSpec",
    "ModelSessionState",
    "ModelSessionStore",
    "ModelSpec",
    "ModelSwitchRequester",
    "format_model_rows",
]
