"""Transactional application service for user model administration."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from agent_runtime.core.llm_config import ModelProvider
from agent_runtime.core.llm_registry import ModelRegistry, user_model_registry_path
from agent_runtime.model_config_io import discover_git_worktree
from agent_runtime.model_definition import ModelExecutionDefinition
from agent_runtime.model_probe import ModelProbe, ModelProbeEvidence, ProbeLevel
from agent_runtime.model_registry import (
    ModelDefinitionPatch,
    RegistryEntryNotFound,
    RegistryMutationResult,
    UserModelDefinition,
    UserModelRegistryStore,
    load_user_model_definition,
)
from agent_runtime.model_trust import ModelBindingTrustDomain, TrustDomainStatus
from agent_runtime.models import ModelCatalog, ModelControlPlane, ModelOrigin, ModelSpec


class CurrentModelRemovalError(ValueError):
    """The addressed session still selects the alias being removed."""


@dataclass(frozen=True, slots=True)
class ModelDefinitionArguments:
    provider: ModelProvider | None = None
    model: str | None = None
    tokenizer_model: str | None = None
    provider_name: str | None = None
    protocol: str | None = None
    max_tokens: int | None = None
    timeout_seconds: float | None = None
    base_url: str | None = None
    api_key_env: str | None = None
    context_window_tokens: int | None = None
    supports_tools: bool | None = None
    supports_structured_output: bool | None = None
    location: Literal["local", "cloud"] | None = None

    def for_add(self) -> UserModelDefinition:
        if self.provider is None or self.model is None:
            raise ValueError("--provider and --provider-model are required without --from")
        values = self._changes()
        return UserModelDefinition.model_validate(values)

    def for_update(self, *, unset_paths: tuple[str, ...]) -> ModelDefinitionPatch:
        return ModelDefinitionPatch(changes=self._changes(), unset_paths=unset_paths)

    def is_empty(self) -> bool:
        return not self._changes()

    def _changes(self) -> dict[str, object]:
        values = {
            "provider": self.provider,
            "model": self.model,
            "tokenizer_model": self.tokenizer_model,
            "provider_name": self.provider_name,
            "protocol": self.protocol,
            "max_tokens": self.max_tokens,
            "timeout_seconds": self.timeout_seconds,
            "base_url": self.base_url,
            "api_key_env": self.api_key_env,
            "context_window_tokens": self.context_window_tokens,
            "supports_tools": self.supports_tools,
            "supports_structured_output": self.supports_structured_output,
            "location": self.location,
        }
        return {key: value for key, value in values.items() if value is not None}


@dataclass(frozen=True, slots=True)
class ModelAdminEntry:
    alias: str
    origin: ModelOrigin
    spec: ModelSpec
    definition: ModelExecutionDefinition
    definition_revision: str


@dataclass(frozen=True, slots=True)
class ModelMutationOutcome:
    alias: str
    definition_revision: str
    registry_revision: int
    registry_fingerprint: str
    changed: bool
    probe_evidence: ModelProbeEvidence | None
    unverified: bool


@dataclass(frozen=True, slots=True)
class ModelSessionSelection:
    spec: ModelSpec
    diagnostics: tuple[str, ...]


class ModelAdminService:
    """Coordinate validation, probing, and CAS commits without CLI concerns."""

    def __init__(
        self,
        *,
        workspace: Path,
        worktree: Path,
        session_path: Path,
    ) -> None:
        self.workspace = workspace.expanduser().resolve()
        self.worktree = worktree.expanduser().resolve()
        self.session_path = session_path
        self.registry = ModelRegistry.from_env(
            workspace=self.workspace,
            worktree=self.worktree,
        )
        self.catalog = ModelCatalog.from_registry(self.registry)
        bundled = ModelRegistry._load_config()
        registry_path = user_model_registry_path()
        self.store = UserModelRegistryStore(
            path=registry_path,
            workspace=self.workspace,
            worktree=self.worktree,
            built_in_aliases=bundled.models,
            whole_catalog_override_active=bool(
                os.environ.get("RAG_AGENT_MODELS_PATH")
                or os.environ.get("RAG_AGENT_MODELS")
            ),
        )
        self.probe_service = ModelProbe(self.registry)
        self.trust_domain = ModelBindingTrustDomain(
            registry_path.parent / "binding-trust.json",
            workspace=self.workspace,
            worktree=self.worktree,
        )

    @classmethod
    def from_env(
        cls,
        *,
        workspace: Path | None = None,
        session_path: Path,
    ) -> ModelAdminService:
        resolved_workspace = (workspace or Path.cwd()).expanduser().resolve()
        return cls(
            workspace=resolved_workspace,
            worktree=discover_git_worktree(resolved_workspace),
            session_path=session_path,
        )

    def list_models(self) -> tuple[ModelAdminEntry, ...]:
        return tuple(self._entry(alias) for alias in self.registry.model_ids)

    def show(self, alias: str) -> ModelAdminEntry:
        return self._entry(alias)

    def current(self) -> ModelSpec:
        return self.current_selection().spec

    def current_selection(self) -> ModelSessionSelection:
        control_plane = self._control_plane()
        return ModelSessionSelection(
            spec=control_plane.current_model(),
            diagnostics=control_plane.session_diagnostics,
        )

    def switch(self, alias: str) -> ModelSpec:
        return self._control_plane().switch_model(alias, requested_by="user")

    def trust_init(self) -> TrustDomainStatus:
        return self.trust_domain.initialize()

    def trust_status(self) -> TrustDomainStatus:
        return self.trust_domain.status()

    async def probe(self, alias: str, *, level: ProbeLevel) -> ModelProbeEvidence:
        return await self.probe_service.run(
            self.registry.get_model_definition(alias),
            level=level,
        )

    async def add(
        self,
        alias: str,
        *,
        arguments: ModelDefinitionArguments,
        from_path: Path | None,
        skip_probe: bool,
    ) -> ModelMutationOutcome:
        if from_path is not None and not arguments.is_empty():
            raise ValueError("--from cannot be combined with model definition flags")
        definition = (
            load_user_model_definition(from_path)
            if from_path is not None
            else arguments.for_add()
        )
        snapshot = self.store.read()
        candidate = self.store.preview_add(alias, definition, snapshot=snapshot)
        execution = self.registry.execution_definition_for_user_model(candidate)
        evidence = None if skip_probe else await self.probe_service.run(execution, level=ProbeLevel.FULL)
        result = self.store.add(alias, candidate, expected=snapshot.version)
        return _mutation_outcome(
            alias=alias,
            execution=execution,
            result=result,
            evidence=evidence,
            unverified=skip_probe,
        )

    async def update(
        self,
        alias: str,
        *,
        arguments: ModelDefinitionArguments,
        from_path: Path | None,
        unset_paths: tuple[str, ...],
        skip_probe: bool,
    ) -> ModelMutationOutcome:
        if from_path is not None and (not arguments.is_empty() or unset_paths):
            raise ValueError("--from cannot be combined with patch flags or --unset")
        mutation = (
            ModelDefinitionPatch(replacement=load_user_model_definition(from_path))
            if from_path is not None
            else arguments.for_update(unset_paths=unset_paths)
        )
        snapshot = self.store.read()
        candidate = self.store.preview_update(alias, mutation, snapshot=snapshot)
        current = snapshot.document.models[alias]
        execution = self.registry.execution_definition_for_user_model(candidate)
        if candidate.to_persisted_mapping() == current.to_persisted_mapping():
            return _mutation_outcome(
                alias=alias,
                execution=execution,
                result=self.store.update(alias, mutation, expected=snapshot.version),
                evidence=None,
                unverified=False,
            )
        evidence = None if skip_probe else await self.probe_service.run(execution, level=ProbeLevel.FULL)
        result = self.store.update(alias, mutation, expected=snapshot.version)
        return _mutation_outcome(
            alias=alias,
            execution=execution,
            result=result,
            evidence=evidence,
            unverified=skip_probe,
        )

    def remove(self, alias: str) -> RegistryMutationResult:
        entry = self.show(alias)
        if entry.origin != "user":
            raise RegistryEntryNotFound(f"Model alias {alias!r} is not user-owned and cannot be removed")
        if self.current().id == alias:
            raise CurrentModelRemovalError(
                f"Model alias {alias!r} is selected; switch this session before removing it"
            )
        snapshot = self.store.read()
        return self.store.remove(alias, expected=snapshot.version)

    def _entry(self, alias: str) -> ModelAdminEntry:
        spec = self.catalog.get(alias)
        definition = self.registry.get_model_definition(alias)
        return ModelAdminEntry(
            alias=alias,
            origin=self.registry.origin(alias),
            spec=spec,
            definition=definition,
            definition_revision=definition.definition_revision,
        )

    def _control_plane(self) -> ModelControlPlane:
        return ModelControlPlane.from_registry(
            self.registry,
            session_path=self.session_path,
        )


def _mutation_outcome(
    *,
    alias: str,
    execution: ModelExecutionDefinition,
    result: RegistryMutationResult,
    evidence: ModelProbeEvidence | None,
    unverified: bool,
) -> ModelMutationOutcome:
    return ModelMutationOutcome(
        alias=alias,
        definition_revision=execution.definition_revision,
        registry_revision=result.snapshot.document.revision,
        registry_fingerprint=result.snapshot.fingerprint,
        changed=result.changed,
        probe_evidence=evidence,
        unverified=unverified,
    )


__all__ = [
    "CurrentModelRemovalError",
    "ModelAdminEntry",
    "ModelAdminService",
    "ModelDefinitionArguments",
    "ModelMutationOutcome",
    "ModelSessionSelection",
]
