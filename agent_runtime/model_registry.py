"""Strict user model declarations and a crash-safe CAS registry editor."""

from __future__ import annotations

import ipaddress
import os
import re
from collections.abc import Callable, Collection, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Self, cast
from urllib.parse import urlsplit

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from agent_runtime.core.llm_config import ModelGenerationDefaults, ModelProvider
from agent_runtime.model_config_io import (
    CommitOutcomeUnknown,
    ConfigVersionConflict,
    FileVersion,
    atomic_replace_bytes,
    exclusive_config_lock,
    file_fingerprint,
    validate_user_config_path,
)

_ALIAS_PATTERN = re.compile(r"^[a-z0-9](?:[a-z0-9._-]{0,62}[a-z0-9])?$")
_ENVIRONMENT_NAME_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_RESERVED_ALIASES = frozenset(
    {
        "list",
        "current",
        "switch",
        "use",
        "add",
        "update",
        "probe",
        "remove",
        "show",
        "trust",
        "default",
    }
)
_UNSET_PATHS = frozenset(
    {
        "tokenizer_model",
        "provider_name",
        "base_url",
        "api_key_env",
        "request_context_tokens",
        "input_cost_per_1m",
        "output_cost_per_1m",
        "cache_read_cost_per_1m",
        "cache_write_cost_per_1m",
        "runtime.health_url",
        "runtime.expected_model_contains",
        "defaults.temperature",
        "defaults.top_p",
        "defaults.parallel_tool_calls",
        "defaults.seed",
    }
)


class RegistryCollisionError(ValueError):
    """An alias is already owned by the user or built-in catalog."""


class RegistryEntryNotFound(KeyError):  # noqa: N818
    """A registry mutation referred to an absent user alias."""


class InvalidUnsetPath(ValueError):  # noqa: N818
    """A patch attempted to unset a non-nullable version-1 field."""


class RegistryOverrideActiveError(RuntimeError):
    """The runtime is using a whole-catalog override that ignores this registry."""


class RegistryConfigValidationError(ValueError):
    """The registry YAML is syntactically invalid or structurally ambiguous."""


class _StrictSafeLoader(yaml.SafeLoader):  # type: ignore[misc]
    """Safe YAML loader that rejects duplicate keys at every mapping depth."""

    def construct_mapping(self, node: Any, deep: bool = False) -> dict[Any, Any]:
        if not isinstance(node, yaml.MappingNode):
            return cast(dict[Any, Any], super().construct_mapping(node, deep=deep))
        self.flatten_mapping(node)
        mapping: dict[Any, Any] = {}
        for key_node, value_node in node.value:
            key = self.construct_object(key_node, deep=deep)
            try:
                duplicate = key in mapping
            except TypeError as error:
                raise RegistryConfigValidationError(
                    f"Registry YAML mapping key at line {key_node.start_mark.line + 1} is not scalar"
                ) from error
            if duplicate:
                raise RegistryConfigValidationError(
                    f"Registry YAML contains duplicate mapping key {key!r} "
                    f"at line {key_node.start_mark.line + 1}"
                )
            mapping[key] = self.construct_object(value_node, deep=deep)
        return mapping


class ModelRuntimeDeclaration(BaseModel):
    """User-owned local process readiness and launch declaration."""

    model_config = ConfigDict(extra="forbid")

    health_url: str | None = None
    launch_command: tuple[str, ...] = ()
    expected_model_contains: str | None = Field(default=None, min_length=1)
    startup_timeout_seconds: float = Field(default=60.0, gt=0, allow_inf_nan=False)
    poll_interval_seconds: float = Field(default=1.0, gt=0, allow_inf_nan=False)

    @field_validator("health_url")
    @classmethod
    def validate_health_url(cls, value: str | None) -> str | None:
        return _validate_http_url(value, field_name="runtime.health_url")

    @field_validator("launch_command", mode="before")
    @classmethod
    def validate_launch_command_input(cls, value: object) -> object:
        if isinstance(value, str):
            raise ValueError("runtime.launch_command must be an argv array, not a shell command")
        return value

    @field_validator("launch_command")
    @classmethod
    def validate_launch_argv(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if any(not part or "\x00" in part for part in value):
            raise ValueError("runtime.launch_command argv entries must be non-empty and contain no NUL")
        return value

    @field_validator("expected_model_contains")
    @classmethod
    def reject_blank_or_padded_expected_model(cls, value: str | None) -> str | None:
        return _reject_blank_or_padded_text(value)


class UserModelDefinition(BaseModel):
    """A flat, secret-free model declaration owned by the user registry."""

    model_config = ConfigDict(extra="forbid")

    provider: ModelProvider
    model: str = Field(min_length=1)
    tokenizer_model: str | None = Field(default=None, min_length=1)
    provider_name: str | None = Field(default=None, min_length=1)
    protocol: str | None = Field(default=None, min_length=1)
    max_tokens: int = Field(default=2048, gt=0, strict=True)
    timeout_seconds: float = Field(default=120.0, gt=0, allow_inf_nan=False)
    base_url: str | None = None
    api_key_env: str | None = None
    defaults: ModelGenerationDefaults = Field(default_factory=ModelGenerationDefaults)
    context_window_tokens: int = Field(default=32_768, gt=0, strict=True)
    request_context_tokens: int | None = Field(default=None, gt=0, strict=True)
    supports_tools: bool = Field(default=True, strict=True)
    supports_structured_output: bool = Field(default=True, strict=True)
    location: Literal["local", "cloud"] | None = None
    input_cost_per_1m: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    output_cost_per_1m: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    cache_read_cost_per_1m: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    cache_write_cost_per_1m: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    runtime: ModelRuntimeDeclaration | None = None

    @field_validator("model", "tokenizer_model", "provider_name", "protocol")
    @classmethod
    def reject_blank_or_padded_text(cls, value: str | None) -> str | None:
        return _reject_blank_or_padded_text(value)

    @field_validator("base_url")
    @classmethod
    def validate_base_url(cls, value: str | None) -> str | None:
        return _validate_http_url(value, field_name="base_url")

    @field_validator("api_key_env")
    @classmethod
    def validate_api_key_environment_name(cls, value: str | None) -> str | None:
        if value is not None and _ENVIRONMENT_NAME_PATTERN.fullmatch(value) is None:
            raise ValueError("api_key_env must be an environment-variable name")
        return value

    @model_validator(mode="after")
    def validate_budget_and_endpoint_consistency(self) -> Self:
        request_limit = self.request_context_tokens or self.context_window_tokens
        if request_limit > self.context_window_tokens:
            raise ValueError("request_context_tokens must not exceed context_window_tokens")
        if self.max_tokens > request_limit:
            raise ValueError("max_tokens must not exceed the effective request context limit")

        endpoint_location = _endpoint_location(self.base_url)
        if self.location == "cloud" and self.base_url is None:
            raise ValueError("cloud model requires base_url")
        if self.location is not None and endpoint_location is not None and self.location != endpoint_location:
            raise ValueError(f"location={self.location!r} conflicts with the base_url endpoint location")
        effective_location = self.location or endpoint_location
        if self.provider in {ModelProvider.MLX, ModelProvider.OLLAMA} and effective_location == "cloud":
            raise ValueError(f"provider {self.provider.value!r} is local and cannot declare location='cloud'")
        return self

    def to_persisted_mapping(self) -> dict[str, object]:
        """Return the stable declaration shape, excluding absent nullable fields."""

        return self.model_dump(mode="json", exclude_none=True)


class UserModelRegistryDocument(BaseModel):
    """The complete versioned user registry persisted in one YAML file."""

    model_config = ConfigDict(extra="forbid")

    version: Literal[1] = 1
    revision: int = Field(ge=0, strict=True)
    models: dict[str, UserModelDefinition] = Field(default_factory=dict)

    @field_validator("models")
    @classmethod
    def validate_aliases(cls, value: dict[str, UserModelDefinition]) -> dict[str, UserModelDefinition]:
        for alias in value:
            _validate_alias(alias)
        return value

    def to_persisted_mapping(self) -> dict[str, object]:
        return {
            "version": self.version,
            "revision": self.revision,
            "models": {
                alias: definition.to_persisted_mapping()
                for alias, definition in sorted(self.models.items())
            },
        }


@dataclass(frozen=True, slots=True)
class RegistrySnapshot:
    document: UserModelRegistryDocument
    fingerprint: str

    @property
    def version(self) -> FileVersion:
        return FileVersion(revision=self.document.revision, fingerprint=self.fingerprint)


class ModelDefinitionPatch(BaseModel):
    """Either a typed replacement or a deep field patch with explicit unsets."""

    model_config = ConfigDict(extra="forbid")

    changes: dict[str, Any] = Field(default_factory=dict)
    unset_paths: tuple[str, ...] = ()
    replacement: UserModelDefinition | None = None

    @field_validator("changes")
    @classmethod
    def reject_none_deletions(cls, value: dict[str, Any]) -> dict[str, Any]:
        _reject_none_changes(value)
        return value

    @model_validator(mode="after")
    def validate_mutation_shape(self) -> Self:
        if self.replacement is not None and (self.changes or self.unset_paths):
            raise ValueError("complete replacement cannot be combined with changes or unset_paths")
        return self


@dataclass(frozen=True, slots=True)
class MutationReceipt:
    base_version: FileVersion
    intended_revision: int
    intended_fingerprint: str


@dataclass(frozen=True, slots=True)
class RegistryMutationResult:
    snapshot: RegistrySnapshot
    receipt: MutationReceipt
    changed: bool


class RegistryCommitOutcomeUnknown(CommitOutcomeUnknown):  # noqa: N818
    """A registry visibility commit happened but durability is not acknowledged."""

    def __init__(self, message: str, *, receipt: MutationReceipt) -> None:
        super().__init__(message)
        self.receipt = receipt


class UserModelRegistryStore:
    """The sole writer for one validated user registry path."""

    def __init__(
        self,
        *,
        path: Path,
        workspace: Path,
        worktree: Path,
        built_in_aliases: Collection[str],
        whole_catalog_override_active: bool | None = None,
    ) -> None:
        self.path = validate_user_config_path(path, workspace=workspace, worktree=worktree)
        self._built_in_aliases = frozenset(built_in_aliases)
        self._whole_catalog_override_active = (
            bool(os.environ.get("RAG_AGENT_MODELS_PATH") or os.environ.get("RAG_AGENT_MODELS"))
            if whole_catalog_override_active is None
            else whole_catalog_override_active
        )

    def read(self) -> RegistrySnapshot:
        return self._read_unlocked()

    def add(
        self,
        alias: str,
        definition: UserModelDefinition,
        *,
        expected: FileVersion,
    ) -> RegistryMutationResult:
        _validate_alias(alias)

        def apply(models: dict[str, UserModelDefinition]) -> dict[str, UserModelDefinition]:
            if alias in self._built_in_aliases:
                raise RegistryCollisionError(f"Model alias {alias!r} is owned by the built-in catalog")
            if alias in models:
                raise RegistryCollisionError(f"User model alias {alias!r} already exists")
            models[alias] = definition
            return models

        return self._mutate(expected=expected, apply=apply)

    def update(
        self,
        alias: str,
        mutation: ModelDefinitionPatch,
        *,
        expected: FileVersion,
    ) -> RegistryMutationResult:
        _validate_alias(alias)

        def apply(models: dict[str, UserModelDefinition]) -> dict[str, UserModelDefinition]:
            validated_mutation = _revalidate_patch(mutation)
            current = models.get(alias)
            if current is None:
                raise RegistryEntryNotFound(f"User model alias {alias!r} does not exist")
            models[alias] = _apply_patch(current, validated_mutation)
            return models

        return self._mutate(expected=expected, apply=apply)

    def remove(self, alias: str, *, expected: FileVersion) -> RegistryMutationResult:
        _validate_alias(alias)

        def apply(models: dict[str, UserModelDefinition]) -> dict[str, UserModelDefinition]:
            if alias not in models:
                raise RegistryEntryNotFound(f"User model alias {alias!r} does not exist")
            del models[alias]
            return models

        return self._mutate(expected=expected, apply=apply)

    def reconcile(self, receipt: MutationReceipt) -> RegistrySnapshot:
        with exclusive_config_lock(self.path):
            snapshot = self._read_unlocked()
            if (
                snapshot.document.revision != receipt.intended_revision
                or snapshot.fingerprint != receipt.intended_fingerprint
            ):
                raise ConfigVersionConflict(
                    "Registry does not match the mutation receipt's exact intended post-state"
                )
            return snapshot

    def _mutate(
        self,
        *,
        expected: FileVersion,
        apply: Callable[[dict[str, UserModelDefinition]], dict[str, UserModelDefinition]],
    ) -> RegistryMutationResult:
        self._reject_override_mode()
        self.path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        with exclusive_config_lock(self.path):
            current = self._read_unlocked()
            if current.version != expected:
                raise ConfigVersionConflict(
                    f"Stale registry version: expected {expected}, observed {current.version}"
                )
            models = apply(dict(current.document.models))
            validated = _revalidate_document(
                revision=current.document.revision,
                models=models,
            )
            models = validated.models
            self._validate_effective_aliases(models)
            unchanged = _normalized_models(models) == _normalized_models(current.document.models)
            if unchanged:
                receipt = MutationReceipt(
                    base_version=current.version,
                    intended_revision=current.document.revision,
                    intended_fingerprint=current.fingerprint,
                )
                return RegistryMutationResult(snapshot=current, receipt=receipt, changed=False)

            document = UserModelRegistryDocument(
                revision=current.document.revision + 1,
                models=models,
            )
            payload = _serialize_document(document)
            fingerprint = file_fingerprint(payload)
            receipt = MutationReceipt(
                base_version=current.version,
                intended_revision=document.revision,
                intended_fingerprint=fingerprint,
            )
            try:
                atomic_replace_bytes(self.path, payload, intended_fingerprint=fingerprint)
            except CommitOutcomeUnknown as error:
                raise RegistryCommitOutcomeUnknown(str(error), receipt=receipt) from error
            snapshot = RegistrySnapshot(document=document, fingerprint=fingerprint)
            return RegistryMutationResult(snapshot=snapshot, receipt=receipt, changed=True)

    def _read_unlocked(self) -> RegistrySnapshot:
        try:
            payload = self.path.read_bytes()
        except FileNotFoundError:
            return RegistrySnapshot(
                document=UserModelRegistryDocument(revision=0),
                fingerprint=file_fingerprint(b""),
            )
        parsed = _load_registry_yaml(payload)
        document = UserModelRegistryDocument.model_validate(parsed)
        return RegistrySnapshot(document=document, fingerprint=file_fingerprint(payload))

    def _validate_effective_aliases(self, models: Mapping[str, UserModelDefinition]) -> None:
        collisions = sorted(self._built_in_aliases.intersection(models))
        if collisions:
            raise RegistryCollisionError(
                f"User registry collides with built-in aliases: {', '.join(collisions)}"
            )

    def _reject_override_mode(self) -> None:
        if self._whole_catalog_override_active:
            raise RegistryOverrideActiveError(
                "User registry mutation is disabled while RAG_AGENT_MODELS_PATH or "
                "RAG_AGENT_MODELS supplies a whole-catalog override"
            )


def _validate_alias(alias: str) -> None:
    if _ALIAS_PATTERN.fullmatch(alias) is None:
        raise ValueError(
            "Model alias must be 1-64 lowercase letters, digits, dots, underscores, or hyphens, "
            "and must start and end with a letter or digit"
        )
    if alias in _RESERVED_ALIASES:
        raise ValueError(f"Model alias {alias!r} is reserved")


def _reject_blank_or_padded_text(value: str | None) -> str | None:
    if value is not None and (not value.strip() or value != value.strip()):
        raise ValueError("text values must be non-blank and have no surrounding whitespace")
    return value


def _validate_http_url(value: str | None, *, field_name: str) -> str | None:
    if value is None:
        return None
    if any(character.isspace() for character in value):
        raise ValueError(f"{field_name} must not contain whitespace")
    if any(ord(character) < 0x20 or ord(character) == 0x7F for character in value):
        raise ValueError(f"{field_name} must not contain ASCII control characters")
    parts = urlsplit(value)
    if parts.scheme not in {"http", "https"} or not parts.netloc or parts.hostname is None:
        raise ValueError(f"{field_name} must be an absolute HTTP(S) URL")
    try:
        _ = parts.port
    except ValueError as error:
        raise ValueError(f"{field_name} contains an invalid port") from error
    if parts.username is not None or parts.password is not None or parts.query or parts.fragment:
        raise ValueError(f"{field_name} must not contain credentials, query parameters, or fragments")
    return value


def _endpoint_location(base_url: str | None) -> Literal["local", "cloud"] | None:
    if base_url is None:
        return None
    hostname = urlsplit(base_url).hostname
    if hostname == "localhost":
        return "local"
    if hostname is not None:
        try:
            if ipaddress.ip_address(hostname).is_loopback:
                return "local"
        except ValueError:
            pass
    return "cloud"


def _apply_patch(current: UserModelDefinition, mutation: ModelDefinitionPatch) -> UserModelDefinition:
    if mutation.replacement is not None:
        return mutation.replacement
    _reject_none_changes(mutation.changes)
    payload: dict[str, Any] = current.to_persisted_mapping()
    _deep_merge(payload, mutation.changes)
    for path in mutation.unset_paths:
        if path not in _UNSET_PATHS:
            raise InvalidUnsetPath(f"Field {path!r} cannot be unset in registry schema version 1")
        _unset_path(payload, path)
    return UserModelDefinition.model_validate(payload)


def _deep_merge(target: dict[str, Any], changes: Mapping[str, Any]) -> None:
    for key, value in changes.items():
        existing = target.get(key)
        if isinstance(existing, dict) and isinstance(value, Mapping):
            _deep_merge(existing, value)
        else:
            target[key] = value


def _unset_path(payload: dict[str, Any], path: str) -> None:
    parts = path.split(".")
    owner = payload
    for part in parts[:-1]:
        nested = owner.get(part)
        if not isinstance(nested, dict):
            return
        owner = nested
    owner.pop(parts[-1], None)


def _normalized_models(models: Mapping[str, UserModelDefinition]) -> dict[str, dict[str, object]]:
    return {alias: definition.to_persisted_mapping() for alias, definition in sorted(models.items())}


def _revalidate_document(
    *,
    revision: int,
    models: Mapping[str, UserModelDefinition],
) -> UserModelRegistryDocument:
    """Cross the mutation trust boundary using plain data, never model identity."""

    plain_models = {
        alias: definition.model_dump(mode="python", exclude_none=True, warnings=False)
        for alias, definition in models.items()
    }
    return UserModelRegistryDocument.model_validate(
        {
            "version": 1,
            "revision": revision,
            "models": plain_models,
        }
    )


def _revalidate_patch(mutation: ModelDefinitionPatch) -> ModelDefinitionPatch:
    """Rebuild a caller-supplied patch from its explicitly set plain fields."""

    plain_patch = mutation.model_dump(
        mode="json",
        exclude_unset=True,
        warnings=False,
    )
    return ModelDefinitionPatch.model_validate(plain_patch)


def _reject_none_changes(value: object, *, path: str = "changes") -> None:
    if value is None:
        raise ValueError(f"{path} cannot be null; nullable deletions must use unset_paths")
    if isinstance(value, Mapping):
        for key, child in value.items():
            _reject_none_changes(child, path=f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _reject_none_changes(child, path=f"{path}[{index}]")


def _serialize_document(document: UserModelRegistryDocument) -> bytes:
    text = str(
        yaml.safe_dump(
            document.to_persisted_mapping(),
            allow_unicode=True,
            sort_keys=True,
        )
    )
    return text.encode("utf-8")


def _load_registry_yaml(payload: bytes) -> object:
    try:
        parsed: object = yaml.load(payload, Loader=_StrictSafeLoader)
    except RegistryConfigValidationError:
        raise
    except yaml.YAMLError as error:
        raise RegistryConfigValidationError(f"Invalid registry YAML: {error}") from error
    return parsed


__all__ = [
    "InvalidUnsetPath",
    "ModelDefinitionPatch",
    "ModelGenerationDefaults",
    "ModelRuntimeDeclaration",
    "MutationReceipt",
    "RegistryCollisionError",
    "RegistryCommitOutcomeUnknown",
    "RegistryConfigValidationError",
    "RegistryEntryNotFound",
    "RegistryMutationResult",
    "RegistryOverrideActiveError",
    "RegistrySnapshot",
    "UserModelDefinition",
    "UserModelRegistryDocument",
    "UserModelRegistryStore",
]
