from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, cast

from agent_runtime.modeling.tokenization import (
    DEFAULT_TOKENIZER_FALLBACK_MODEL,
    TokenAccountingService,
    TokenizerContract,
)
from agent_runtime.text import load_env_file
from rag.assembly.bindings import (
    CapabilityBinding,
    ChatCapabilityBinding,
    EmbeddingCapabilityBinding,
    RerankCapabilityBinding,
    _provider_model,
    _provider_name,
    _supports_capability,
)
from rag.assembly.models import (
    AssemblyConfig,
    AssemblyContracts,
    AssemblyDecision,
    AssemblyDiagnostics,
    AssemblyIssue,
    AssemblyOverrides,
    AssemblyRequest,
    AssemblyStatus,
    CapabilityBundle,
    CapabilityCatalog,
    CapabilityProfile,
    DecisionSource,
    ProviderConfig,
    RuntimeContractGovernance,
)
from rag.assembly.support import (
    FallbackEmbeddingRepo,
    build_provider,
    compatibility_config_from_environment,
    first_bool,
    first_non_blank,
    first_non_negative_int,
    first_positive_int,
)


@dataclass(frozen=True, slots=True)
class _CandidateSource:
    source: DecisionSource
    provider_config: ProviderConfig
    cache_key: str
    reason: str


class CapabilityAssemblyService:
    def __init__(self, *, env_path: str = ".env") -> None:
        self._env_path = env_path

    def catalog_from_environment(self, *, config: AssemblyConfig | None = None) -> CapabilityCatalog:
        self._load_env()
        compatibility_config, compatibility_inputs = self._compatibility_config_from_environment()
        profiles = self._catalog_profiles(config=config, compatibility_config=compatibility_config)
        return CapabilityCatalog(
            profiles=profiles,
            diagnostics=(),
            compatibility_inputs=compatibility_inputs,
        )

    def evaluate_request(self, request: AssemblyRequest) -> CapabilityBundle:
        self._load_env()
        compatibility_config, compatibility_inputs = self._compatibility_config_from_environment()
        effective_request = request
        profiles = self._catalog_profiles(config=effective_request.config, compatibility_config=compatibility_config)
        provider_cache: dict[str, object] = {}
        issues: list[AssemblyIssue] = []
        decisions: list[AssemblyDecision] = []

        embedding_bindings: list[EmbeddingCapabilityBinding] = []
        chat_bindings: list[ChatCapabilityBinding] = []
        rerank_bindings: list[RerankCapabilityBinding] = []

        embedding_binding = cast(
            EmbeddingCapabilityBinding | None,
            self._resolve_binding(
                capability="embedding",
                request=effective_request,
                compatibility_config=compatibility_config,
                compatibility_inputs=compatibility_inputs,
                provider_cache=provider_cache,
                issues=issues,
                decisions=decisions,
                default_space="default",
            ),
        )
        if embedding_binding is not None:
            embedding_bindings.append(embedding_binding)

        chat_binding = cast(
            ChatCapabilityBinding | None,
            self._resolve_binding(
                capability="chat",
                request=effective_request,
                compatibility_config=compatibility_config,
                provider_cache=provider_cache,
                issues=issues,
                decisions=decisions,
            ),
        )
        if chat_binding is not None:
            chat_bindings.append(chat_binding)

        rerank_binding = cast(
            RerankCapabilityBinding | None,
            self._resolve_binding(
                capability="rerank",
                request=effective_request,
                compatibility_config=compatibility_config,
                provider_cache=provider_cache,
                issues=issues,
                decisions=decisions,
            ),
        )
        if rerank_binding is not None:
            rerank_bindings.append(rerank_binding)

        if not embedding_bindings:
            fallback = FallbackEmbeddingRepo()
            fallback_binding = EmbeddingCapabilityBinding(backend=fallback, space="default", location="local")
            embedding_bindings.append(fallback_binding)
            decisions.append(
                AssemblyDecision(
                    capability="embedding",
                    source="default",
                    provider_kind="fallback",
                    provider_name=fallback_binding.provider_name,
                    model_name=fallback_binding.model_name,
                    location=fallback_binding.location,
                    reason="No configured embedding provider was available; using fallback embedding backend.",
                )
            )
            issues.append(
                AssemblyIssue(
                    severity="warning",
                    code="fallback_embedding_selected",
                    message="No configured embedding provider was available; using fallback embedding backend.",
                )
            )

        token_contract = self._build_token_contract(
            request=effective_request,
            compatibility_config=compatibility_config,
            compatibility_inputs=compatibility_inputs,
            issues=issues,
            decisions=decisions,
            embedding_binding=embedding_bindings[0],
        )
        token_accounting = TokenAccountingService(token_contract)
        contracts = AssemblyContracts(
            token_contract=token_contract,
            token_accounting=token_accounting,
            runtime_contract_payload=self._runtime_contract_payload(
                token_contract,
                token_accounting,
                embedding_binding=embedding_bindings[0],
            ),
        )
        diagnostics = AssemblyDiagnostics(
            status=self._diagnostics_status(issues),
            issues=tuple(issues),
            decisions=tuple(decisions),
            compatibility_inputs=compatibility_inputs,
        )
        return CapabilityBundle(
            request=request,
            effective_request=effective_request,
            diagnostics=diagnostics,
            contracts=contracts,
            embedding_bindings=tuple(embedding_bindings),
            chat_bindings=tuple(chat_bindings),
            rerank_bindings=tuple(rerank_bindings),
            profiles=profiles,
        )

    def assemble_request(self, request: AssemblyRequest) -> CapabilityBundle:
        bundle = self.evaluate_request(request)
        bundle.diagnostics.raise_for_invalid()
        return bundle

    def govern_runtime_contract(
        self,
        *,
        bundle: CapabilityBundle,
        stored_payload: dict[str, Any] | None,
    ) -> RuntimeContractGovernance:
        current_payload = bundle.runtime_contract_payload
        if stored_payload is None:
            return RuntimeContractGovernance(status="valid", should_persist=True)
        mismatches: dict[str, tuple[Any | None, Any | None]] = {}
        for key in ("tokenizer_backend", "chunk_token_size", "chunk_overlap_tokens"):
            if stored_payload.get(key) != current_payload.get(key):
                mismatches[key] = (current_payload.get(key), stored_payload.get(key))
        if not mismatches:
            return RuntimeContractGovernance(status="valid", should_persist=False)
        details = ", ".join(
            f"{field}: current={current!r} stored={stored!r}" for field, (current, stored) in mismatches.items()
        )
        issue = AssemblyIssue(
            severity="error",
            code="runtime_contract_mismatch",
            message=(
                "RAG runtime contract does not match the existing index. "
                f"Mismatched fields: {details}. Use the same embedding/tokenizer contract or rebuild the index."
            ),
        )
        return RuntimeContractGovernance(
            status="invalid",
            should_persist=False,
            mismatches=mismatches,
            issues=(issue,),
        )

    def _load_env(self) -> None:
        load_env_file(self._env_path)

    def _compatibility_config_from_environment(self) -> tuple[AssemblyConfig, dict[str, str | int | bool | None]]:
        return compatibility_config_from_environment()

    def _catalog_profiles(
        self,
        *,
        config: AssemblyConfig | None,
        compatibility_config: AssemblyConfig,
    ) -> tuple[CapabilityProfile, ...]:
        profiles: list[CapabilityProfile] = []
        for provider_config in [*(config.profiles if config is not None else ()), *compatibility_config.profiles]:
            profile_id = provider_config.profile_id
            if not profile_id or not provider_config.enabled:
                continue
            profiles.append(self._profile_from_provider_config(provider_config))
        return tuple(profiles)

    def _resolve_binding(
        self,
        *,
        capability: str,
        request: AssemblyRequest,
        compatibility_config: AssemblyConfig,
        compatibility_inputs: dict[str, str | int | bool | None] | None = None,
        provider_cache: dict[str, object],
        issues: list[AssemblyIssue],
        decisions: list[AssemblyDecision],
        default_space: str = "default",
    ) -> CapabilityBinding | None:
        # ── pre-built runtime provider (highest priority; used by HTTP service env) ──
        overrides = request.overrides or AssemblyOverrides()
        attr_name = f"{capability}_provider"
        runtime_provider = getattr(overrides, attr_name, None)
        if runtime_provider is not None and capability in ("embedding", "rerank"):
            if capability == "embedding":
                binding: CapabilityBinding = EmbeddingCapabilityBinding(
                    backend=runtime_provider,
                    space=self._embedding_space_from_provider(runtime_provider, default_space),
                    location="runtime",
                )
            else:
                binding = RerankCapabilityBinding(
                    backend=runtime_provider,
                    location="runtime",
                )
            decisions.append(
                AssemblyDecision(
                    capability=capability,
                    source="explicit",
                    provider_kind="runtime-http",
                    provider_name=getattr(binding, "provider_name", None),
                    model_name=getattr(binding, "model_name", None),
                    location="runtime",
                    reason=f"Using pre-built runtime {capability} provider via service URL env.",
                )
            )
            return binding

        candidates = self._capability_candidates(
            capability=capability,
            request=request,
            compatibility_config=compatibility_config,
        )
        return self._choose_capability_binding(
            capability=capability,
            candidates=candidates,
            provider_cache=provider_cache,
            issues=issues,
            decisions=decisions,
            required=getattr(request.requirements, f"require_{capability}"),
            allow_degraded=request.requirements.allow_degraded,
            compatibility_inputs=compatibility_inputs,
            default_space=default_space,
        )

    def _capability_candidates(
        self,
        *,
        capability: str,
        request: AssemblyRequest,
        compatibility_config: AssemblyConfig,
    ) -> list[_CandidateSource]:
        candidates: list[_CandidateSource] = []
        overrides = request.overrides or AssemblyOverrides()
        config = request.config or AssemblyConfig()

        explicit_spec = getattr(overrides, capability)
        if explicit_spec is not None and explicit_spec.enabled:
            candidates.append(
                _CandidateSource(
                    source="explicit",
                    provider_config=explicit_spec,
                    cache_key=f"explicit:{capability}",
                    reason=f"Using explicit {capability} override.",
                )
            )

        config_spec = getattr(config, capability)
        if config_spec is not None and config_spec.enabled:
            candidates.append(
                _CandidateSource(
                    source="config",
                    provider_config=config_spec,
                    cache_key=f"config:{capability}",
                    reason=f"Using structured config for {capability}.",
                )
            )

        compat_profile = next(
            (
                provider_config
                for provider_config in compatibility_config.profiles
                if provider_config.enabled and self._provider_supports(provider_config, capability)
            ),
            None,
        )
        if compat_profile is not None:
            candidates.append(
                _CandidateSource(
                    source="compat_env",
                    provider_config=compat_profile,
                    cache_key=f"compat:{capability}:{compat_profile.profile_id or compat_profile.provider_kind}",
                    reason=f"Using compatibility environment config for {capability}.",
                )
            )
        return candidates

    def _choose_capability_binding(
        self,
        *,
        capability: str,
        candidates: Sequence[_CandidateSource],
        provider_cache: dict[str, object],
        issues: list[AssemblyIssue],
        decisions: list[AssemblyDecision],
        required: bool,
        allow_degraded: bool,
        compatibility_inputs: dict[str, str | int | bool | None] | None = None,
        default_space: str = "default",
    ) -> CapabilityBinding | None:
        del compatibility_inputs
        for candidate in candidates:
            provider = self._provider_from_cache(candidate, provider_cache)
            if not _supports_capability(provider, capability):
                issues.append(
                    AssemblyIssue(
                        severity="warning",
                        code=f"{capability}_candidate_unusable",
                        message=(
                            f"{candidate.reason} Provider {candidate.provider_config.provider_kind!r} does not "
                            f"provide a usable {capability} capability."
                        ),
                    )
                )
                decisions.append(
                    AssemblyDecision(
                        capability=capability,
                        source=candidate.source,
                        provider_kind=candidate.provider_config.provider_kind,
                        provider_name=_provider_name(provider),
                        model_name=_provider_model(provider, capability),
                        location=candidate.provider_config.location,
                        reason=f"{candidate.reason} Candidate was rejected because the capability is unavailable.",
                        selected=False,
                    )
                )
                continue
            if capability == "embedding":
                binding: CapabilityBinding = EmbeddingCapabilityBinding(
                    backend=provider,
                    space=(
                        candidate.provider_config.embedding_space
                        or self._embedding_space_from_provider(provider, default_space)
                    ),
                    location=candidate.provider_config.location,
                )
            elif capability == "chat":
                binding = ChatCapabilityBinding(
                    backend=provider,
                    location=candidate.provider_config.location,
                )
            else:
                binding = RerankCapabilityBinding(
                    backend=provider,
                    location=candidate.provider_config.location,
                )
            decisions.append(
                AssemblyDecision(
                    capability=capability,
                    source=candidate.source,
                    provider_kind=candidate.provider_config.provider_kind,
                    provider_name=getattr(binding, "provider_name", None),
                    model_name=getattr(binding, "model_name", None),
                    location=candidate.provider_config.location,
                    reason=candidate.reason,
                )
            )
            return binding

        if required:
            issues.append(
                AssemblyIssue(
                    severity="warning" if allow_degraded else "error",
                    code=f"missing_required_{capability}",
                    message=f"No usable {capability} capability could be assembled.",
                )
            )
        return None

    def _build_token_contract(
        self,
        *,
        request: AssemblyRequest,
        embedding_binding: EmbeddingCapabilityBinding,
        compatibility_config: AssemblyConfig,
        compatibility_inputs: dict[str, str | int | bool | None],
        issues: list[AssemblyIssue],
        decisions: list[AssemblyDecision],
    ) -> TokenizerContract:
        explicit_tokenizer = request.overrides.tokenizer if request.overrides is not None else None
        config_tokenizer = request.config.tokenizer if request.config is not None else None
        compat_tokenizer = compatibility_config.tokenizer

        embedding_model = embedding_binding.model_name or DEFAULT_TOKENIZER_FALLBACK_MODEL
        locked_embedding_model = first_non_blank(
            explicit_tokenizer.embedding_model_name if explicit_tokenizer is not None else None,
            config_tokenizer.embedding_model_name if config_tokenizer is not None else None,
            compat_tokenizer.embedding_model_name if compat_tokenizer is not None else None,
        )
        if locked_embedding_model is not None and locked_embedding_model != embedding_model:
            issues.append(
                AssemblyIssue(
                    severity="error",
                    code="embedding_contract_mismatch",
                    message=(
                        "Configured embedding model does not match the selected embedding capability: "
                        f"{locked_embedding_model!r} != {embedding_model!r}."
                    ),
                )
            )

        tokenizer_model = first_non_blank(
            explicit_tokenizer.tokenizer_model_name if explicit_tokenizer is not None else None,
            config_tokenizer.tokenizer_model_name if config_tokenizer is not None else None,
            compat_tokenizer.tokenizer_model_name if compat_tokenizer is not None else None,
            embedding_model,
            DEFAULT_TOKENIZER_FALLBACK_MODEL,
        )
        chunking_tokenizer_model = first_non_blank(
            explicit_tokenizer.chunking_tokenizer_model_name if explicit_tokenizer is not None else None,
            config_tokenizer.chunking_tokenizer_model_name if config_tokenizer is not None else None,
            compat_tokenizer.chunking_tokenizer_model_name if compat_tokenizer is not None else None,
            tokenizer_model,
            DEFAULT_TOKENIZER_FALLBACK_MODEL,
        )
        tokenizer_backend = (
            first_non_blank(
                explicit_tokenizer.tokenizer_backend if explicit_tokenizer is not None else None,
                config_tokenizer.tokenizer_backend if config_tokenizer is not None else None,
                compat_tokenizer.tokenizer_backend if compat_tokenizer is not None else None,
            )
            or "auto"
        )
        chunk_token_size = max(
            32,
            first_positive_int(
                explicit_tokenizer.chunk_token_size if explicit_tokenizer is not None else None,
                config_tokenizer.chunk_token_size if config_tokenizer is not None else None,
                compat_tokenizer.chunk_token_size if compat_tokenizer is not None else None,
                request.requirements.default_chunk_token_size,
            ),
        )
        chunk_overlap_tokens = max(
            0,
            first_non_negative_int(
                explicit_tokenizer.chunk_overlap_tokens if explicit_tokenizer is not None else None,
                config_tokenizer.chunk_overlap_tokens if config_tokenizer is not None else None,
                compat_tokenizer.chunk_overlap_tokens if compat_tokenizer is not None else None,
                request.requirements.default_chunk_overlap_tokens,
            ),
        )
        max_context_tokens = max(
            64,
            first_positive_int(
                explicit_tokenizer.max_context_tokens if explicit_tokenizer is not None else None,
                config_tokenizer.max_context_tokens if config_tokenizer is not None else None,
                compat_tokenizer.max_context_tokens if compat_tokenizer is not None else None,
                request.requirements.default_context_tokens,
            ),
        )
        prompt_reserved_tokens = max(
            32,
            first_positive_int(
                explicit_tokenizer.prompt_reserved_tokens if explicit_tokenizer is not None else None,
                config_tokenizer.prompt_reserved_tokens if config_tokenizer is not None else None,
                compat_tokenizer.prompt_reserved_tokens if compat_tokenizer is not None else None,
                request.requirements.default_prompt_reserved_tokens,
            ),
        )
        local_files_only = first_bool(
            explicit_tokenizer.local_files_only if explicit_tokenizer is not None else None,
            config_tokenizer.local_files_only if config_tokenizer is not None else None,
            compat_tokenizer.local_files_only if compat_tokenizer is not None else None,
            True,
        )
        source: DecisionSource = "default"
        if explicit_tokenizer is not None:
            source = "explicit"
        elif config_tokenizer is not None:
            source = "config"
        elif compat_tokenizer is not None:
            source = "compat_env"
        decisions.append(
            AssemblyDecision(
                capability="tokenizer_contract",
                source=source,
                provider_kind="tokenizer-contract",
                provider_name="tokenizer-contract",
                model_name=tokenizer_model,
                location="assembly",
                reason="Tokenizer contract was assembled through the unified contract governance chain.",
            )
        )
        compatibility_inputs["resolved_embedding_model_name"] = embedding_model
        return TokenizerContract(
            embedding_model_name=embedding_model,
            tokenizer_model_name=tokenizer_model or DEFAULT_TOKENIZER_FALLBACK_MODEL,
            chunking_tokenizer_model_name=(
                chunking_tokenizer_model or tokenizer_model or DEFAULT_TOKENIZER_FALLBACK_MODEL
            ),
            tokenizer_backend=tokenizer_backend,
            chunk_token_size=chunk_token_size,
            chunk_overlap_tokens=chunk_overlap_tokens,
            max_context_tokens=max_context_tokens,
            prompt_reserved_tokens=prompt_reserved_tokens,
            local_files_only=local_files_only,
        )

    @staticmethod
    def _runtime_contract_payload(
        token_contract: TokenizerContract,
        token_accounting: TokenAccountingService,
        *,
        embedding_binding: EmbeddingCapabilityBinding,
    ) -> dict[str, str | int | bool]:
        tokenizer_backend, _tokenizer_model = token_accounting.backend_descriptor()
        return {
            "embedding_model_name": token_contract.embedding_model_name,
            "embedding_space": embedding_binding.space,
            "tokenizer_model_name": token_contract.tokenizer_model_name,
            "chunking_tokenizer_model_name": token_contract.chunking_tokenizer_model_name,
            "tokenizer_backend": tokenizer_backend,
            "chunk_token_size": token_contract.chunk_token_size,
            "chunk_overlap_tokens": token_contract.normalized_chunk_overlap_tokens(),
        }

    @staticmethod
    def _embedding_space_from_provider(provider: object, default_space: str) -> str:
        value = getattr(provider, "embedding_space", None)
        return value if isinstance(value, str) and value.strip() else default_space

    @staticmethod
    def _diagnostics_status(issues: Sequence[AssemblyIssue]) -> AssemblyStatus:
        if any(issue.severity == "error" for issue in issues):
            return "invalid"
        if any(issue.severity == "warning" for issue in issues):
            return "degraded"
        return "valid"

    def _profile_from_provider_config(self, provider_config: ProviderConfig) -> CapabilityProfile:
        profile_id = provider_config.profile_id or provider_config.provider_kind
        model_label = (
            provider_config.chat_model or provider_config.embedding_model or provider_config.rerank_model or "default"
        )
        label = provider_config.label or f"{provider_config.provider_kind} / {model_label}"

        def factory() -> object:
            return self._build_provider(provider_config)

        provider_preview = self._build_provider(provider_config)
        return CapabilityProfile(
            profile_id=profile_id,
            label=label,
            provider_kind=provider_config.provider_kind,
            location=provider_config.location,
            chat_model=_provider_model(provider_preview, "chat"),
            embedding_model=_provider_model(provider_preview, "embedding"),
            rerank_model=_provider_model(provider_preview, "rerank"),
            supports_chat=_supports_capability(provider_preview, "chat"),
            supports_embedding=_supports_capability(provider_preview, "embedding"),
            supports_rerank=_supports_capability(provider_preview, "rerank"),
            provider_config=provider_config,
            factory=factory,
        )

    def _provider_supports(self, provider_config: ProviderConfig, capability: str) -> bool:
        try:
            provider = self._build_provider(provider_config)
        except RuntimeError:
            return False
        return _supports_capability(provider, capability)

    def _provider_from_cache(self, candidate: _CandidateSource, provider_cache: dict[str, object]) -> object:
        provider = provider_cache.get(candidate.cache_key)
        if provider is None:
            provider = self._build_provider(candidate.provider_config)
            provider_cache[candidate.cache_key] = provider
        return provider

    def _build_provider(self, provider_config: ProviderConfig) -> object:
        return build_provider(provider_config)


__all__ = ["CapabilityAssemblyService"]
