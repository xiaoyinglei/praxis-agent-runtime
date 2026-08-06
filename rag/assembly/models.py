from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Literal

from agent_runtime.modeling.tokenization import TokenAccountingService, TokenizerContract

AssemblyStatus = Literal["valid", "degraded", "invalid"]
IssueSeverity = Literal["info", "warning", "error"]
DecisionSource = Literal["explicit", "config", "compat_env", "default"]


@dataclass(frozen=True, slots=True)
class ProviderConfig:
    provider_kind: str
    location: str = "runtime"
    profile_id: str | None = None
    label: str | None = None
    api_key: str | None = None
    base_url: str | None = None
    chat_model: str | None = None
    chat_model_path: str | None = None
    chat_backend: str | None = None
    embedding_model: str | None = None
    embedding_space: str | None = None
    rerank_model: str | None = None
    embedding_model_path: str | None = None
    rerank_model_path: str | None = None
    embedding_batch_size: int | None = None
    rerank_batch_size: int | None = None
    device: str | None = None
    enabled: bool = True


@dataclass(frozen=True, slots=True)
class TokenizerConfig:
    embedding_model_name: str | None = None
    tokenizer_model_name: str | None = None
    chunking_tokenizer_model_name: str | None = None
    tokenizer_backend: str | None = None
    chunk_token_size: int | None = None
    chunk_overlap_tokens: int | None = None
    max_context_tokens: int | None = None
    prompt_reserved_tokens: int | None = None
    local_files_only: bool | None = None


@dataclass(frozen=True, slots=True)
class AssemblyConfig:
    profiles: tuple[ProviderConfig, ...] = ()
    chat: ProviderConfig | None = None
    embedding: ProviderConfig | None = None
    rerank: ProviderConfig | None = None
    tokenizer: TokenizerConfig | None = None


@dataclass(frozen=True, slots=True)
class AssemblyOverrides:
    chat: ProviderConfig | None = None
    embedding: ProviderConfig | None = None
    rerank: ProviderConfig | None = None
    tokenizer: TokenizerConfig | None = None
    # Pre-built runtime providers (checked before static ProviderConfig resolution).
    # Set by _runtime() when RAG_EMBEDDING_SERVICE_URL / RAG_RERANK_SERVICE_URL is present.
    embedding_provider: object | None = None
    rerank_provider: object | None = None


@dataclass(frozen=True, slots=True)
class CapabilityRequirements:
    require_embedding: bool = True
    require_chat: bool = False
    require_rerank: bool = False
    allow_degraded: bool = True
    default_context_tokens: int = 2048
    default_chunk_token_size: int = 480
    default_chunk_overlap_tokens: int = 64
    default_prompt_reserved_tokens: int = 256


@dataclass(frozen=True, slots=True)
class AssemblyRequest:
    requirements: CapabilityRequirements = field(default_factory=CapabilityRequirements)
    config: AssemblyConfig | None = None
    overrides: AssemblyOverrides | None = None


@dataclass(frozen=True, slots=True)
class AssemblyIssue:
    severity: IssueSeverity
    code: str
    message: str


@dataclass(frozen=True, slots=True)
class AssemblyDecision:
    capability: str
    source: DecisionSource
    provider_kind: str
    provider_name: str | None
    model_name: str | None
    location: str | None
    reason: str
    selected: bool = True


@dataclass(frozen=True, slots=True)
class CapabilityProfile:
    profile_id: str
    label: str
    provider_kind: str
    location: str
    chat_model: str | None
    embedding_model: str | None
    rerank_model: str | None
    supports_chat: bool
    supports_embedding: bool
    supports_rerank: bool
    provider_config: ProviderConfig = field(repr=False)
    factory: Callable[[], object] = field(repr=False)

    def create_provider(self) -> object:
        return self.factory()


@dataclass(frozen=True, slots=True)
class CapabilityCatalog:
    profiles: tuple[CapabilityProfile, ...] = ()
    diagnostics: tuple[AssemblyIssue, ...] = ()
    compatibility_inputs: dict[str, str | int | bool | None] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class AssemblyContracts:
    token_contract: TokenizerContract
    token_accounting: TokenAccountingService
    runtime_contract_payload: dict[str, str | int | bool]


@dataclass(frozen=True, slots=True)
class AssemblyDiagnostics:
    status: AssemblyStatus
    issues: tuple[AssemblyIssue, ...] = ()
    decisions: tuple[AssemblyDecision, ...] = ()
    compatibility_inputs: dict[str, str | int | bool | None] = field(default_factory=dict)

    @property
    def warnings(self) -> tuple[AssemblyIssue, ...]:
        return tuple(issue for issue in self.issues if issue.severity == "warning")

    @property
    def errors(self) -> tuple[AssemblyIssue, ...]:
        return tuple(issue for issue in self.issues if issue.severity == "error")

    def raise_for_invalid(self) -> None:
        if self.status != "invalid":
            return
        if self.errors:
            detail = "; ".join(issue.message for issue in self.errors)
        else:
            detail = "assembly produced an invalid result"
        raise RuntimeError(detail)


@dataclass(frozen=True, slots=True)
class RuntimeContractGovernance:
    status: AssemblyStatus
    should_persist: bool
    mismatches: dict[str, tuple[Any | None, Any | None]] = field(default_factory=dict)
    issues: tuple[AssemblyIssue, ...] = ()

    def raise_for_invalid(self) -> None:
        if self.status != "invalid":
            return
        detail = "; ".join(issue.message for issue in self.issues) or "runtime contract governance failed"
        raise RuntimeError(detail)


@dataclass(frozen=True, slots=True)
class CapabilityBundle:
    request: AssemblyRequest
    effective_request: AssemblyRequest
    diagnostics: AssemblyDiagnostics
    contracts: AssemblyContracts
    embedding_bindings: tuple[Any, ...]
    chat_bindings: tuple[Any, ...]
    rerank_bindings: tuple[Any, ...]
    profiles: tuple[CapabilityProfile, ...] = ()

    @property
    def status(self) -> AssemblyStatus:
        return self.diagnostics.status

    @property
    def token_contract(self) -> TokenizerContract:
        return self.contracts.token_contract

    @property
    def token_accounting(self) -> TokenAccountingService:
        return self.contracts.token_accounting

    @property
    def runtime_contract_payload(self) -> dict[str, str | int | bool]:
        return self.contracts.runtime_contract_payload


__all__ = [
    "AssemblyConfig",
    "AssemblyContracts",
    "AssemblyDecision",
    "AssemblyDiagnostics",
    "AssemblyIssue",
    "AssemblyOverrides",
    "AssemblyRequest",
    "AssemblyStatus",
    "CapabilityBundle",
    "CapabilityCatalog",
    "CapabilityProfile",
    "CapabilityRequirements",
    "DecisionSource",
    "IssueSeverity",
    "ProviderConfig",
    "RuntimeContractGovernance",
    "TokenizerConfig",
]
