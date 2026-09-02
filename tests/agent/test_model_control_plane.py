from __future__ import annotations

import json
import os
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml
from typer.testing import CliRunner

from agent_runtime import model_config_io
from agent_runtime.cli import agent_app
from agent_runtime.core.llm_registry import (
    ModelNotAvailableError,
    ModelRegistry,
    UnknownModelAliasError,
)
from agent_runtime.local_runtime import (
    EndpointConflictError,
    LocalProviderProbe,
    LocalRuntimeError,
)
from agent_runtime.model_config_io import (
    CommitOutcomeUnknown,
    ConfigVersionConflict,
    file_fingerprint,
)
from agent_runtime.model_trust import (
    BindingAuthenticationError,
    ModelBindingTrustDomain,
    TrustDomainNotInitializedError,
    TrustedDefinitionNotFoundError,
    TrustedModelDefinitionArchive,
)
from agent_runtime.modeling.contracts import DEFAULT_LLM_STAGE_BUDGETS, LLMCallStage
from agent_runtime.models import (
    ModelCatalog,
    ModelControlPlane,
    ModelPolicy,
    ModelPolicyError,
    ModelRuntimeSpec,
    ModelSessionState,
    ModelSessionStore,
    ModelSpec,
    ModelSwitchRequester,
    SessionCommitOutcomeUnknown,
)
from agent_runtime.text import load_env_file


def _write_models_config(path: Path) -> None:
    path.write_text(
        yaml.safe_dump(
            {
                "models": {
                    "local_qwen": {
                        "capability": "chat",
                        "provider": "qwen",
                        "protocol": "openai_compatible",
                        "model": "mlx-community/Qwen3-14B-4bit",
                        "base_url": "http://127.0.0.1:8080/v1",
                        "context_window_tokens": 32768,
                        "tools": True,
                        "structured_output": True,
                        "location": "local",
                        "runtime": {
                            "health_url": "http://127.0.0.1:8080/v1/models",
                            "launch_command": ["uv", "run", "python", "-m", "mlx_lm.server"],
                            "expected_model_contains": "Qwen3-14B",
                            "startup_timeout_seconds": 5,
                        },
                    },
                    "mimo_cloud": {
                        "capability": "chat",
                        "provider": "mimo",
                        "protocol": "openai_compatible",
                        "model": "mimo-v2.5-pro",
                        "base_url": "https://token-plan-cn.xiaomimimo.com/v1",
                        "api_key_env": "MIMO_API_KEY",
                        "context_window_tokens": 256000,
                        "tools": True,
                        "structured_output": True,
                        "location": "cloud",
                        "cost": {
                            "input_per_1m": 0.5,
                            "output_per_1m": 2.0,
                        },
                    },
                    "embed": {
                        "capability": "embedding",
                        "provider": "qwen",
                        "model": "embedding-model",
                    },
                },
                "defaults": {"primary_model": "local_qwen"},
            }
        ),
        encoding="utf-8",
    )


def test_model_catalog_loads_runtime_specs_without_embedding_models(tmp_path: Path) -> None:
    config_path = tmp_path / "models.yaml"
    _write_models_config(config_path)

    catalog = ModelCatalog.from_config_file(config_path)

    assert [spec.id for spec in catalog.list_models()] == ["local_qwen", "mimo_cloud"]
    spec = catalog.get("mimo_cloud")
    assert spec.provider == "mimo"
    assert spec.provider_model == "mimo-v2.5-pro"
    assert spec.context_window == 256000
    assert spec.supports_tools is True
    assert spec.supports_structured_output is True
    assert spec.location == "cloud"
    assert spec.runtime is None
    assert spec.input_cost_per_1m == 0.5
    assert spec.output_cost_per_1m == 2.0
    assert catalog.default_model_id == "local_qwen"
    local = catalog.get("local_qwen")
    assert local.runtime is not None
    assert local.runtime.health_url == "http://127.0.0.1:8080/v1/models"
    assert local.runtime.expected_model_contains == "Qwen3-14B"


def test_effective_catalog_layers_user_registry_with_provenance(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry_path = (tmp_path / "user-config" / "models.yaml").resolve()
    registry_path.parent.mkdir()
    registry_path.write_text(
        yaml.safe_dump(
            {
                "version": 1,
                "revision": 1,
                "models": {
                    "my_qwen": {
                        "provider": "openai_compatible",
                        "provider_name": "local-test",
                        "model": "Qwen/Qwen3.5-9B",
                        "base_url": "http://127.0.0.1:8080/v1",
                        "location": "local",
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("PRAXIS_MODEL_REGISTRY_PATH", str(registry_path))
    monkeypatch.delenv("RAG_AGENT_MODELS_PATH", raising=False)
    monkeypatch.delenv("RAG_AGENT_MODELS", raising=False)

    first = ModelCatalog.from_env(env_path=str(tmp_path / "missing.env"))
    assert first.origin("groq_gpt_oss_120b") == "builtin"
    assert first.origin("my_qwen") == "user"
    assert first.get("my_qwen").provider_model == "Qwen/Qwen3.5-9B"
    assert first.default_model_id == "groq_gpt_oss_120b"
    first_definition_revision = first.definition("my_qwen").definition_revision
    assert first.definition("my_qwen").provider == "openai_compatible"
    assert first.definition("my_qwen").generation.answer.max_tokens == 4096

    registry_path.write_text(
        registry_path.read_text(encoding="utf-8").replace("Qwen3.5-9B", "Qwen3.5-14B"),
        encoding="utf-8",
    )
    assert first.get("my_qwen").provider_model == "Qwen/Qwen3.5-9B"
    refreshed = ModelCatalog.from_env(env_path=str(tmp_path / "missing.env"))
    assert refreshed.get("my_qwen").provider_model == "Qwen/Qwen3.5-14B"
    assert refreshed.definition("my_qwen").definition_revision != first_definition_revision


def test_effective_catalog_rejects_user_shadowing_builtin(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry_path = (tmp_path / "user-config" / "models.yaml").resolve()
    registry_path.parent.mkdir()
    registry_path.write_text(
        yaml.safe_dump(
            {
                "version": 1,
                "revision": 1,
                "models": {
                    "groq_gpt_oss_120b": {
                        "provider": "openai_compatible",
                        "model": "attacker/model",
                        "base_url": "https://example.com/v1",
                        "location": "cloud",
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("PRAXIS_MODEL_REGISTRY_PATH", str(registry_path))
    monkeypatch.delenv("RAG_AGENT_MODELS_PATH", raising=False)
    monkeypatch.delenv("RAG_AGENT_MODELS", raising=False)

    with pytest.raises(ValueError, match="collid|built-in"):
        ModelCatalog.from_env(env_path=str(tmp_path / "missing.env"))


def test_effective_catalog_fails_loudly_for_malformed_user_registry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry_path = (tmp_path / "user-config" / "models.yaml").resolve()
    registry_path.parent.mkdir()
    registry_path.write_text("version: 2\nrevision: 1\nmodels: {}\n", encoding="utf-8")
    monkeypatch.setenv("PRAXIS_MODEL_REGISTRY_PATH", str(registry_path))
    monkeypatch.delenv("RAG_AGENT_MODELS_PATH", raising=False)
    monkeypatch.delenv("RAG_AGENT_MODELS", raising=False)

    with pytest.raises(ValueError, match="version"):
        ModelCatalog.from_env(env_path=str(tmp_path / "missing.env"))


def test_whole_catalog_override_replaces_layers_and_marks_origin(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    override_path = tmp_path / "override.yaml"
    _write_models_config(override_path)
    registry_path = (tmp_path / "user-config" / "models.yaml").resolve()
    registry_path.parent.mkdir()
    registry_path.write_text(
        "version: 1\nrevision: 0\nmodels: {}\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("RAG_AGENT_MODELS_PATH", str(override_path))
    monkeypatch.setenv("PRAXIS_MODEL_REGISTRY_PATH", str(registry_path))

    catalog = ModelCatalog.from_env(env_path=str(tmp_path / "missing.env"))

    assert [item.id for item in catalog.list_models()] == ["local_qwen", "mimo_cloud"]
    assert catalog.origin("local_qwen") == "override"
    assert not catalog.has("groq_gpt_oss_120b")


def test_model_catalog_deep_copies_supplied_definitions() -> None:
    source = ModelCatalog.from_config_file(Path("configs/models.yaml"))
    definition = source.definition("kimi_cloud")
    catalog = ModelCatalog(
        specs={"kimi_cloud": source.get("kimi_cloud")},
        default_model_id="kimi_cloud",
        origins={"kimi_cloud": "builtin"},
        definitions={"kimi_cloud": definition},
    )
    original_revision = catalog.definition("kimi_cloud").definition_revision

    provider_options = definition.defaults.provider_options
    assert provider_options is not None
    thinking = provider_options.thinking
    assert thinking is not None
    object.__setattr__(thinking, "type", "disabled")

    assert catalog.definition("kimi_cloud").definition_revision == original_revision


def test_bundled_default_chat_model_is_groq_control() -> None:
    catalog = ModelCatalog.from_config_file(Path("configs/models.yaml"))

    spec = catalog.get(catalog.default_model_id)

    assert catalog.default_model_id == "groq_gpt_oss_120b"
    assert spec.provider == "groq"
    assert spec.provider_model == "openai/gpt-oss-120b"
    assert spec.location == "cloud"
    assert spec.api_key_env == "GROQ_API_KEY"


def test_bundled_kimi_k26_cloud_model_is_available_for_diagnostics() -> None:
    catalog = ModelCatalog.from_config_file(Path("configs/models.yaml"))

    spec = catalog.get("kimi_cloud")

    assert spec.provider == "kimi"
    assert spec.provider_model == "kimi-k2.6"
    assert spec.location == "cloud"
    assert spec.api_key_env == "MOONSHOT_API_KEY"
    assert spec.context_window == 262_144


def test_bundled_local_qwen8_runtime_is_available_for_local_testing() -> None:
    catalog = ModelCatalog.from_config_file(Path("configs/models.yaml"))

    spec = catalog.get("qwen3_8b_mlx_4bit")

    assert spec.provider == "local_mlx_chat_8080"
    assert spec.provider_model == "mlx-community/Qwen3-8B-4bit"
    assert spec.location == "local"
    assert spec.runtime is not None
    assert spec.runtime.health_url == "http://127.0.0.1:8080/v1/models"
    assert spec.runtime.expected_model_contains == "Qwen3-8B-4bit"
    assert "{model}" not in spec.runtime.launch_command
    assert "mlx-community/Qwen3-8B-4bit" in spec.runtime.launch_command


def test_bundled_tool_decision_budget_supports_coding_turns() -> None:
    payload = yaml.safe_load(Path("configs/models.yaml").read_text(encoding="utf-8"))

    budget = payload["llm_budgets"]["tool_decision"]
    assert budget["max_input_tokens"] == 32_000
    assert budget["max_output_tokens"] == 4_096
    default = DEFAULT_LLM_STAGE_BUDGETS[LLMCallStage.TOOL_DECISION]
    assert default.max_input_tokens == 32_000
    assert default.max_output_tokens == 4_096


def test_env_loader_uses_shared_env_for_linked_worktree(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    primary = tmp_path / "repository"
    common_git = primary / ".git"
    worktree_git = common_git / "worktrees" / "feature"
    worktree = tmp_path / "feature-worktree"
    worktree_git.mkdir(parents=True)
    worktree.mkdir()
    (worktree / ".git").write_text(
        f"gitdir: {worktree_git}\n",
        encoding="utf-8",
    )
    (worktree_git / "commondir").write_text("../..\n", encoding="utf-8")
    shared_env = primary / ".env"
    shared_env.write_text("WORKTREE_SHARED_KEY=available\n", encoding="utf-8")
    monkeypatch.delenv("AGENT_ENV_FILE", raising=False)
    monkeypatch.delenv("WORKTREE_SHARED_KEY", raising=False)

    loaded = load_env_file(worktree / ".env")

    assert loaded == shared_env.resolve()
    assert os.environ["WORKTREE_SHARED_KEY"] == "available"


def test_model_policy_reviews_agent_model_switch_requests(tmp_path: Path) -> None:
    config_path = tmp_path / "models.yaml"
    _write_models_config(config_path)
    catalog = ModelCatalog.from_config_file(config_path)
    state = ModelSessionState(current_model_id="local_qwen")
    policy = ModelPolicy(allowed_agent_model_ids=frozenset({"local_qwen"}))
    control = ModelControlPlane(catalog=catalog, state=state, policy=policy)

    with pytest.raises(ModelPolicyError, match="not allowed"):
        control.switch_model("mimo_cloud", requested_by="agent")

    assert state.current_model_id == "local_qwen"
    control.switch_model("mimo_cloud", requested_by="user")
    assert state.current_model_id == "mimo_cloud"
    assert state.selection_requester == "user"


@pytest.mark.parametrize("requested_by", ["user", "agent", "system"])
def test_model_policy_reviews_frozen_definition_without_catalog(
    tmp_path: Path,
    requested_by: ModelSwitchRequester,
) -> None:
    config_path = tmp_path / "models.yaml"
    _write_models_config(config_path)
    catalog = ModelCatalog.from_config_file(config_path)
    definition = catalog.definition("mimo_cloud")
    policy = ModelPolicy(
        allowed_user_model_ids=frozenset({"mimo_cloud"}),
        allowed_agent_model_ids=frozenset({"mimo_cloud"}),
        allowed_system_model_ids=frozenset({"mimo_cloud"}),
        allowed_provider_kinds=frozenset({definition.provider.value}),
        allowed_remote_hosts=frozenset({"token-plan-cn.xiaomimimo.com"}),
    )

    reviewed = policy.review_binding(
        alias="mimo_cloud",
        definition=definition,
        requested_by=requested_by,
    )

    assert reviewed == definition


def test_model_policy_rejects_frozen_provider_host_and_local_launch(
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "models.yaml"
    _write_models_config(config_path)
    catalog = ModelCatalog.from_config_file(config_path)
    cloud = catalog.definition("mimo_cloud")
    local = catalog.definition("local_qwen")

    with pytest.raises(ModelPolicyError, match="provider"):
        ModelPolicy(allowed_provider_kinds=frozenset({"ollama"})).review_binding(
            alias="mimo_cloud",
            definition=cloud,
            requested_by="user",
        )
    with pytest.raises(ModelPolicyError, match="host"):
        ModelPolicy(allowed_remote_hosts=frozenset({"api.example.com"})).review_binding(
            alias="mimo_cloud",
            definition=cloud,
            requested_by="user",
        )
    with pytest.raises(ModelPolicyError, match="launch"):
        ModelPolicy(allow_local_launch=False).review_binding(
            alias="local_qwen",
            definition=local,
            requested_by="user",
        )
    assert local.runtime is not None
    unsafe_health = local.model_copy(
        update={
            "runtime": local.runtime.model_copy(
                update={"health_url": "https://metadata.evil.example/status"}
            )
        }
    )
    with pytest.raises(ModelPolicyError, match="health"):
        ModelPolicy().review_binding(
            alias="local_qwen",
            definition=unsafe_health,
            requested_by="user",
        )


def test_model_policy_revalidates_frozen_endpoint_and_credential_reference(
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "models.yaml"
    _write_models_config(config_path)
    definition = ModelCatalog.from_config_file(config_path).definition("mimo_cloud")
    wrong_location = definition.model_copy(update={"location": "local"})
    unsafe_credential = definition.model_copy(update={"api_key_env": "TOKEN=value"})

    with pytest.raises(ModelPolicyError, match="invalid"):
        ModelPolicy().review_binding(
            alias="mimo_cloud",
            definition=wrong_location,
            requested_by="user",
        )
    with pytest.raises(ModelPolicyError, match="invalid"):
        ModelPolicy().review_binding(
            alias="mimo_cloud",
            definition=unsafe_credential,
            requested_by="user",
        )


def test_freeze_and_resolve_authenticated_binding_without_alias_resolution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = tmp_path / "models.yaml"
    workspace = tmp_path / "workspace"
    trusted_root = tmp_path / "trusted"
    workspace.mkdir(mode=0o700)
    trusted_root.mkdir(mode=0o700)
    _write_models_config(config_path)
    trust = ModelBindingTrustDomain(
        trusted_root / "binding-trust.json",
        workspace=workspace,
        worktree=workspace,
    )
    archive = TrustedModelDefinitionArchive(
        trusted_root / "model-definitions",
        workspace=workspace,
        worktree=workspace,
    )
    trust.initialize()
    ready: list[str] = []

    probe = SimpleNamespace(
        ensure_ready=lambda spec: ready.append(spec.id)
    )

    control = ModelControlPlane.from_config_file(
        config_path,
        initial_model_id="local_qwen",
        initial_selection_requester="user",
        trust_domain=trust,
        definition_archive=archive,
        local_provider_probe=probe,
    )
    frozen_definition = control.catalog.definition("local_qwen")
    resolved_sentinel = SimpleNamespace(model="resolved-frozen")
    resolved_definitions: list[object] = []

    def resolve_definition(_registry: object, definition: object) -> object:
        resolved_definitions.append(definition)
        return resolved_sentinel

    monkeypatch.setattr(ModelRegistry, "resolve_definition", resolve_definition)

    binding = control.freeze_model_binding(thread_id="thread-1", turn_id="turn-1")
    resolved = control.resolve_frozen_binding(
        binding,
        thread_id="thread-1",
        turn_id="turn-1",
    )

    assert resolved is resolved_sentinel
    assert ready == ["local_qwen"]
    assert resolved_definitions == [frozen_definition]
    assert binding["selection_requester"] == "user"
    assert binding["thread_id"] == "thread-1"
    envelope = binding["binding"]
    assert isinstance(envelope, dict)
    assert envelope["alias"] == "local_qwen"
    assert envelope["origin"] == "override"
    assert archive.load(frozen_definition.definition_revision) == frozen_definition

    tampered = {**binding, "turn_id": "turn-2"}
    with pytest.raises(BindingAuthenticationError):
        control.resolve_frozen_binding(
            tampered,
            thread_id="thread-1",
            turn_id="turn-2",
        )
    assert ready == ["local_qwen"]
    assert resolved_definitions == [frozen_definition]

    archived_path = archive.path / f"{frozen_definition.definition_revision}.json"
    archived_path.unlink()
    with pytest.raises(TrustedDefinitionNotFoundError):
        control.resolve_frozen_binding(
            binding,
            thread_id="thread-1",
            turn_id="turn-1",
        )
    assert ready == ["local_qwen"]
    assert resolved_definitions == [frozen_definition]


def test_freeze_requires_explicit_trust_before_archive_mutation(tmp_path: Path) -> None:
    config_path = tmp_path / "models.yaml"
    workspace = tmp_path / "workspace"
    trusted_root = tmp_path / "trusted"
    workspace.mkdir(mode=0o700)
    trusted_root.mkdir(mode=0o700)
    _write_models_config(config_path)
    trust_path = trusted_root / "binding-trust.json"
    archive_path = trusted_root / "model-definitions"
    control = ModelControlPlane.from_config_file(
        config_path,
        initial_model_id="local_qwen",
        trust_domain=ModelBindingTrustDomain(
            trust_path,
            workspace=workspace,
            worktree=workspace,
        ),
        definition_archive=TrustedModelDefinitionArchive(
            archive_path,
            workspace=workspace,
            worktree=workspace,
        ),
    )

    with pytest.raises(TrustDomainNotInitializedError, match="agent model trust init"):
        control.freeze_model_binding(thread_id="thread-1", turn_id="turn-1")

    assert not trust_path.exists()
    assert not archive_path.exists()


def test_frozen_binding_checks_current_credential_before_provider_construction(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = tmp_path / "models.yaml"
    workspace = tmp_path / "workspace"
    trusted_root = tmp_path / "trusted"
    workspace.mkdir(mode=0o700)
    trusted_root.mkdir(mode=0o700)
    _write_models_config(config_path)
    trust = ModelBindingTrustDomain(
        trusted_root / "binding-trust.json",
        workspace=workspace,
        worktree=workspace,
    )
    archive = TrustedModelDefinitionArchive(
        trusted_root / "model-definitions",
        workspace=workspace,
        worktree=workspace,
    )
    trust.initialize()
    control = ModelControlPlane.from_config_file(
        config_path,
        initial_model_id="mimo_cloud",
        initial_selection_requester="user",
        trust_domain=trust,
        definition_archive=archive,
    )
    binding = control.freeze_model_binding(thread_id="thread-1", turn_id="turn-1")
    provider_calls: list[object] = []

    def resolve_definition(_registry: object, definition: object) -> object:
        provider_calls.append(definition)
        return object()

    monkeypatch.setattr(ModelRegistry, "resolve_definition", resolve_definition)
    monkeypatch.delenv("MIMO_API_KEY", raising=False)

    with pytest.raises(ModelNotAvailableError, match="MIMO_API_KEY"):
        control.resolve_frozen_binding(
            binding,
            thread_id="thread-1",
            turn_id="turn-1",
        )

    assert provider_calls == []


def test_local_frozen_binding_checks_current_credential_before_readiness(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = tmp_path / "models.yaml"
    workspace = tmp_path / "workspace"
    trusted_root = tmp_path / "trusted"
    workspace.mkdir(mode=0o700)
    trusted_root.mkdir(mode=0o700)
    _write_models_config(config_path)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config["models"]["local_qwen"]["api_key_env"] = "LOCAL_AUTH_TOKEN"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    trust = ModelBindingTrustDomain(
        trusted_root / "binding-trust.json",
        workspace=workspace,
        worktree=workspace,
    )
    archive = TrustedModelDefinitionArchive(
        trusted_root / "model-definitions",
        workspace=workspace,
        worktree=workspace,
    )
    trust.initialize()
    readiness_calls: list[str] = []
    provider_calls: list[object] = []
    control = ModelControlPlane.from_config_file(
        config_path,
        initial_model_id="local_qwen",
        initial_selection_requester="user",
        trust_domain=trust,
        definition_archive=archive,
        local_provider_probe=SimpleNamespace(
            ensure_ready=lambda spec: readiness_calls.append(spec.id)
        ),
    )
    monkeypatch.setenv("LOCAL_AUTH_TOKEN", "present-while-freezing")
    binding = control.freeze_model_binding(thread_id="thread-1", turn_id="turn-1")
    monkeypatch.delenv("LOCAL_AUTH_TOKEN")
    monkeypatch.setattr(
        ModelRegistry,
        "resolve_definition",
        lambda _registry, definition: provider_calls.append(definition),
    )

    with pytest.raises(ModelNotAvailableError, match="LOCAL_AUTH_TOKEN"):
        control.resolve_frozen_binding(
            binding,
            thread_id="thread-1",
            turn_id="turn-1",
        )

    assert readiness_calls == []
    assert provider_calls == []


def test_frozen_binding_requires_resolver_before_local_readiness(tmp_path: Path) -> None:
    config_path = tmp_path / "models.yaml"
    workspace = tmp_path / "workspace"
    trusted_root = tmp_path / "trusted"
    workspace.mkdir(mode=0o700)
    trusted_root.mkdir(mode=0o700)
    _write_models_config(config_path)
    trust = ModelBindingTrustDomain(
        trusted_root / "binding-trust.json",
        workspace=workspace,
        worktree=workspace,
    )
    trust.initialize()
    readiness_calls: list[str] = []
    control = ModelControlPlane(
        catalog=ModelCatalog.from_config_file(config_path),
        state=ModelSessionState(
            current_model_id="local_qwen"
        ),
        registry=None,
        trust_domain=trust,
        definition_archive=TrustedModelDefinitionArchive(
            trusted_root / "model-definitions",
            workspace=workspace,
            worktree=workspace,
        ),
        local_provider_probe=SimpleNamespace(
            ensure_ready=lambda spec: (
                readiness_calls.append(spec.id)
            )
        ),
    )
    binding = control.freeze_model_binding(thread_id="thread-1", turn_id="turn-1")

    with pytest.raises(RuntimeError, match="cannot resolve frozen definitions"):
        control.resolve_frozen_binding(
            binding,
            thread_id="thread-1",
            turn_id="turn-1",
        )

    assert readiness_calls == []


def test_frozen_binding_uses_the_same_snapshot_after_hmac_verification(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = tmp_path / "models.yaml"
    workspace = tmp_path / "workspace"
    trusted_root = tmp_path / "trusted"
    workspace.mkdir(mode=0o700)
    trusted_root.mkdir(mode=0o700)
    _write_models_config(config_path)
    trust = ModelBindingTrustDomain(
        trusted_root / "binding-trust.json",
        workspace=workspace,
        worktree=workspace,
    )
    archive = TrustedModelDefinitionArchive(
        trusted_root / "model-definitions",
        workspace=workspace,
        worktree=workspace,
    )
    trust.initialize()
    control = ModelControlPlane.from_config_file(
        config_path,
        initial_model_id="local_qwen",
        initial_selection_requester="user",
        trust_domain=trust,
        definition_archive=archive,
        local_provider_probe=SimpleNamespace(
            ensure_ready=lambda _spec: None
        ),
    )
    binding = control.freeze_model_binding(thread_id="thread-1", turn_id="turn-1")
    envelope = binding["binding"]
    assert isinstance(envelope, dict)
    original_load = archive.load

    def mutate_original_after_verification(revision: str) -> object:
        envelope["alias"] = "mimo_cloud"
        return original_load(revision)

    monkeypatch.setattr(archive, "load", mutate_original_after_verification)
    monkeypatch.setattr(
        ModelRegistry,
        "resolve_definition",
        lambda _registry, _definition: object(),
    )
    control.policy = ModelPolicy(allowed_user_model_ids=frozenset({"mimo_cloud"}))

    with pytest.raises(ModelPolicyError, match="local_qwen"):
        control.resolve_frozen_binding(
            binding,
            thread_id="thread-1",
            turn_id="turn-1",
        )


def test_agent_selection_requester_paths_are_explicit(tmp_path: Path) -> None:
    config_path = tmp_path / "models.yaml"
    _write_models_config(config_path)
    control = ModelControlPlane.from_config_file(
        config_path,
        initial_model_id="local_qwen",
        initial_selection_requester="system",
        session_path=None,
    )

    assert control.state.selection_requester == "system"

    control.switch_model("mimo_cloud", requested_by="user", persist=False)
    assert control.state.selection_requester == "user"

    control.request_model_switch("local_qwen")
    assert control.state.selection_requester == "agent"


def test_selection_requester_rejects_unknown_policy_domains(tmp_path: Path) -> None:
    config_path = tmp_path / "models.yaml"
    _write_models_config(config_path)

    with pytest.raises(ValueError, match="requester"):
        ModelSessionState(
            current_model_id="local_qwen",
            selection_requester="root",  # type: ignore[arg-type]
        )
    with pytest.raises(ValueError, match="requester"):
        ModelControlPlane.from_config_file(
            config_path,
            initial_model_id="local_qwen",
            initial_selection_requester="root",  # type: ignore[arg-type]
        )


def test_selection_requester_rejects_objects_that_compare_equal_to_system() -> None:
    class ForgedRequester:
        def __hash__(self) -> int:
            return hash("system")

        def __eq__(self, other: object) -> bool:
            return other == "system"

    with pytest.raises(ValueError, match="requester"):
        ModelSessionState(
            current_model_id="local_qwen",
            selection_requester=ForgedRequester(),  # type: ignore[arg-type]
        )


def test_initial_and_restored_selections_are_reviewed_in_requester_domain(
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "models.yaml"
    session_path = tmp_path / "model-session.json"
    _write_models_config(config_path)
    user_policy = ModelPolicy(allowed_user_model_ids=frozenset({"local_qwen"}))

    with pytest.raises(ModelPolicyError, match="not allowed"):
        ModelControlPlane.from_config_file(
            config_path,
            initial_model_id="mimo_cloud",
            initial_selection_requester="user",
            policy=user_policy,
        )

    session_path.write_text(
        json.dumps(
            {
                "version": 1,
                "revision": 3,
                "current_model_id": "mimo_cloud",
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ModelPolicyError, match="not allowed"):
        ModelControlPlane.from_config_file(
            config_path,
            session_path=session_path,
            policy=user_policy,
        )


def test_control_plane_resolves_provider_from_session_current_model(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = tmp_path / "models.yaml"
    _write_models_config(config_path)
    monkeypatch.setenv("RAG_AGENT_MODELS_PATH", str(config_path))
    monkeypatch.setenv("MIMO_API_KEY", "sk-test")
    resolved_aliases: list[str] = []

    def fake_resolve(self: ModelRegistry, alias: str):  # type: ignore[no-untyped-def]
        resolved_aliases.append(alias)
        return object()

    monkeypatch.setattr(ModelRegistry, "resolve", fake_resolve)

    control = ModelControlPlane.from_env(initial_model_id="mimo_cloud")
    resolved = control.resolve_for_node(node_model=None, node_name="tool_decision")

    assert resolved is not None
    assert resolved_aliases == ["mimo_cloud"]


def test_control_plane_does_not_fallback_from_explicit_model(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = tmp_path / "models.yaml"
    _write_models_config(config_path)
    monkeypatch.setenv(
        "RAG_AGENT_MODELS_PATH",
        str(config_path),
    )

    resolved_aliases: list[str] = []

    class FakeLocalProviderProbe:
        def ensure_ready(
            self,
            spec: ModelSpec,
        ) -> None:
            del spec

    def fail_resolve(
        self: ModelRegistry,
        alias: str,
    ):  # type: ignore[no-untyped-def]
        del self
        resolved_aliases.append(alias)
        raise ModelNotAvailableError(
            f"{alias} failed"
        )

    monkeypatch.setattr(
        ModelRegistry,
        "resolve",
        fail_resolve,
    )

    control = ModelControlPlane.from_env(
        initial_model_id="local_qwen",
        local_provider_probe=(
            FakeLocalProviderProbe()
        ),
    )

    with pytest.raises(
        ModelNotAvailableError,
        match="local_qwen failed",
    ):
        control.resolve_for_node(
            node_model=None,
            node_name="tool_decision",
        )

    assert resolved_aliases == [
        "local_qwen"
    ]

def test_control_plane_probes_local_provider_before_resolving(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = tmp_path / "models.yaml"
    _write_models_config(config_path)

    monkeypatch.setenv(
        "RAG_AGENT_MODELS_PATH",
        str(config_path),
    )

    ensured: list[str] = []
    resolved_aliases: list[str] = []

    class FakeLocalProviderProbe:
        def ensure_ready(
            self,
            spec: ModelSpec,
        ) -> None:
            ensured.append(spec.id)

    def fake_resolve(
        self: ModelRegistry,
        alias: str,
    ):  # type: ignore[no-untyped-def]
        del self
        resolved_aliases.append(alias)
        return object()

    monkeypatch.setattr(
        ModelRegistry,
        "resolve",
        fake_resolve,
    )

    control = ModelControlPlane.from_env(
        initial_model_id="local_qwen",
        local_provider_probe=(
            FakeLocalProviderProbe()
        ),
    )

    result = control.resolve_for_node(
        node_model=None,
        node_name="tool_decision",
    )

    assert result is not None
    assert ensured == ["local_qwen"]
    assert resolved_aliases == [
        "local_qwen"
    ]

def test_control_plane_rejects_cloud_model_without_api_key(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = tmp_path / "models.yaml"
    _write_models_config(config_path)
    monkeypatch.delenv("MIMO_API_KEY", raising=False)

    control = ModelControlPlane.from_config_file(config_path, initial_model_id="mimo_cloud")

    with pytest.raises(
        ModelNotAvailableError,
        match="MIMO_API_KEY is not set",
    ):
        control.resolve_for_node(node_model=None, node_name="tool_decision")


def test_model_session_state_persists_without_rewriting_yaml(tmp_path: Path) -> None:
    config_path = tmp_path / "models.yaml"
    session_path = tmp_path / "model-session.json"
    _write_models_config(config_path)
    before = config_path.read_text(encoding="utf-8")

    control = ModelControlPlane.from_config_file(
        config_path,
        session_path=session_path,
    )
    control.switch_model("mimo_cloud", requested_by="user")

    restored = ModelControlPlane.from_config_file(
        config_path,
        session_path=session_path,
    )
    assert restored.current_model().id == "mimo_cloud"
    assert config_path.read_text(encoding="utf-8") == before


def test_model_session_legacy_record_loads_as_user_and_upgrades_on_switch(
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "models.yaml"
    session_path = tmp_path / "model-session.json"
    _write_models_config(config_path)
    legacy_bytes = b'{"current_model_id":"local_qwen"}\n'
    session_path.write_bytes(legacy_bytes)

    control = ModelControlPlane.from_config_file(config_path, session_path=session_path)

    assert control.state.selection_requester == "user"
    assert control.state.file_revision == 0
    assert control.state.fingerprint == file_fingerprint(legacy_bytes)
    assert session_path.read_bytes() == legacy_bytes

    control.switch_model("mimo_cloud", requested_by="user")

    assert json.loads(session_path.read_text(encoding="utf-8")) == {
        "version": 1,
        "revision": 1,
        "current_model_id": "mimo_cloud",
    }


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"version": 1, "revision": 0},
        {"version": 1, "revision": True, "current_model_id": "local_qwen"},
        {"version": 2, "revision": 0, "current_model_id": "local_qwen"},
        {
            "version": 1,
            "revision": 0,
            "current_model_id": "local_qwen",
            "selection_requester": "system",
        },
    ],
)
def test_model_session_store_rejects_malformed_or_privileged_records(
    tmp_path: Path,
    payload: dict[str, object],
) -> None:
    session_path = tmp_path / "model-session.json"
    session_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="session"):
        ModelSessionStore(session_path).read(default_model_id="local_qwen")


def test_model_session_switch_uses_revision_and_exact_fingerprint_cas(
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "models.yaml"
    session_path = tmp_path / "model-session.json"
    _write_models_config(config_path)
    first = ModelControlPlane.from_config_file(config_path, session_path=session_path)
    second = ModelControlPlane.from_config_file(config_path, session_path=session_path)

    first.switch_model("mimo_cloud", requested_by="user")

    with pytest.raises(ConfigVersionConflict, match="session"):
        second.switch_model("mimo_cloud", requested_by="user")

    assert second.current_model().id == "local_qwen"
    assert json.loads(session_path.read_text(encoding="utf-8"))["current_model_id"] == "mimo_cloud"


def test_model_session_fingerprint_detects_same_revision_rewrite(tmp_path: Path) -> None:
    config_path = tmp_path / "models.yaml"
    session_path = tmp_path / "model-session.json"
    _write_models_config(config_path)
    session_path.write_text(
        json.dumps({"version": 1, "revision": 7, "current_model_id": "local_qwen"}),
        encoding="utf-8",
    )
    control = ModelControlPlane.from_config_file(config_path, session_path=session_path)
    session_path.write_text(
        json.dumps({"version": 1, "revision": 7, "current_model_id": "mimo_cloud"}),
        encoding="utf-8",
    )

    with pytest.raises(ConfigVersionConflict, match="session"):
        control.switch_model("mimo_cloud", requested_by="user")

    assert control.current_model().id == "local_qwen"


def test_model_switch_write_failure_retains_selection_and_requester(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = tmp_path / "models.yaml"
    session_path = tmp_path / "model-session.json"
    _write_models_config(config_path)
    control = ModelControlPlane.from_config_file(
        config_path,
        initial_model_id="local_qwen",
        initial_selection_requester="system",
        session_path=session_path,
    )

    def fail_write(*_args: object, **_kwargs: object) -> None:
        raise OSError("disk unavailable")

    monkeypatch.setattr("agent_runtime.models.atomic_replace_bytes", fail_write)

    with pytest.raises(OSError, match="disk unavailable"):
        control.switch_model("mimo_cloud", requested_by="user")

    assert control.current_model().id == "local_qwen"
    assert control.state.selection_requester == "system"


def test_model_switch_unknown_commit_outcome_retains_in_memory_selection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = tmp_path / "models.yaml"
    session_path = tmp_path / "model-session.json"
    _write_models_config(config_path)
    control = ModelControlPlane.from_config_file(
        config_path,
        initial_model_id="local_qwen",
        initial_selection_requester="system",
        session_path=session_path,
    )

    def unknown_outcome(*_args: object, **_kwargs: object) -> None:
        raise CommitOutcomeUnknown("cannot confirm session durability")

    monkeypatch.setattr("agent_runtime.models.atomic_replace_bytes", unknown_outcome)

    with pytest.raises(CommitOutcomeUnknown, match="cannot confirm"):
        control.switch_model("mimo_cloud", requested_by="user")

    assert control.current_model().id == "local_qwen"
    assert control.state.selection_requester == "system"


def test_model_switch_unknown_post_replace_outcome_can_reconcile_exact_intent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = tmp_path / "models.yaml"
    session_path = tmp_path / "model-session.json"
    _write_models_config(config_path)
    control = ModelControlPlane.from_config_file(
        config_path,
        initial_model_id="local_qwen",
        initial_selection_requester="system",
        session_path=session_path,
    )

    real_directory_fsync = model_config_io._fsync_directory

    def fail_directory_fsync(_path: Path) -> None:
        raise OSError("directory fsync unavailable")

    monkeypatch.setattr(
        "agent_runtime.model_config_io._fsync_directory",
        fail_directory_fsync,
    )

    with pytest.raises(SessionCommitOutcomeUnknown) as captured:
        control.switch_model("mimo_cloud", requested_by="user")

    assert control.current_model().id == "local_qwen"
    assert control.state.selection_requester == "system"
    assert json.loads(session_path.read_text(encoding="utf-8")) == {
        "version": 1,
        "revision": 1,
        "current_model_id": "mimo_cloud",
    }
    with pytest.raises(ConfigVersionConflict):
        control.switch_model("mimo_cloud", requested_by="user")

    with pytest.raises(SessionCommitOutcomeUnknown) as still_unknown:
        control.reconcile_model_switch(captured.value.receipt)

    assert still_unknown.value.receipt == captured.value.receipt
    assert control.current_model().id == "local_qwen"
    assert control.state.selection_requester == "system"

    monkeypatch.setattr(
        "agent_runtime.model_config_io._fsync_directory",
        real_directory_fsync,
    )
    recovered = control.reconcile_model_switch(captured.value.receipt)

    assert recovered.id == "mimo_cloud"
    assert control.current_model().id == "mimo_cloud"
    assert control.state.selection_requester == "user"
    assert control.state.file_revision == 1


def test_model_switch_reconcile_rejects_reconstructed_privileged_receipt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = tmp_path / "models.yaml"
    session_path = tmp_path / "model-session.json"
    _write_models_config(config_path)
    control = ModelControlPlane.from_config_file(
        config_path,
        initial_model_id="local_qwen",
        initial_selection_requester="system",
        session_path=session_path,
        policy=ModelPolicy(
            allowed_user_model_ids=frozenset({"local_qwen", "mimo_cloud"}),
            allowed_system_model_ids=frozenset({"local_qwen", "mimo_cloud"}),
        ),
    )
    real_directory_fsync = model_config_io._fsync_directory

    def fail_directory_fsync(_path: Path) -> None:
        raise OSError("directory fsync unavailable")

    monkeypatch.setattr(model_config_io, "_fsync_directory", fail_directory_fsync)
    with pytest.raises(SessionCommitOutcomeUnknown) as captured:
        control.switch_model("mimo_cloud", requested_by="user")
    monkeypatch.setattr(model_config_io, "_fsync_directory", real_directory_fsync)

    forged = replace(captured.value.receipt, selection_requester="system")
    with pytest.raises(ValueError, match="issued by this control plane"):
        control.reconcile_model_switch(forged)

    assert control.current_model().id == "local_qwen"
    assert control.state.selection_requester == "system"
    recovered = control.reconcile_model_switch(captured.value.receipt)
    assert recovered.id == "mimo_cloud"
    assert control.state.selection_requester == "user"


def test_stale_persisted_alias_is_atomically_repaired_to_catalog_default(
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "models.yaml"
    session_path = tmp_path / "model-session.json"
    _write_models_config(config_path)
    session_path.write_text(
        json.dumps({"version": 1, "revision": 4, "current_model_id": "removed-model"}),
        encoding="utf-8",
    )

    control = ModelControlPlane.from_config_file(config_path, session_path=session_path)

    assert control.current_model().id == "local_qwen"
    assert control.state.selection_requester == "user"
    assert control.state.file_revision == 5
    assert any("removed-model" in diagnostic for diagnostic in control.session_diagnostics)
    assert json.loads(session_path.read_text(encoding="utf-8")) == {
        "version": 1,
        "revision": 5,
        "current_model_id": "local_qwen",
    }


def test_stale_repair_reviews_default_before_writing_session(tmp_path: Path) -> None:
    config_path = tmp_path / "models.yaml"
    session_path = tmp_path / "model-session.json"
    _write_models_config(config_path)
    original = json.dumps(
        {
            "version": 1,
            "revision": 7,
            "current_model_id": "removed-model",
        }
    ).encode()
    session_path.write_bytes(original)
    policy = ModelPolicy(allowed_user_model_ids=frozenset({"mimo_cloud"}))

    with pytest.raises(ModelPolicyError, match="local_qwen.*not allowed"):
        ModelControlPlane.from_config_file(
            config_path,
            session_path=session_path,
            policy=policy,
        )

    assert session_path.read_bytes() == original


def test_stale_repair_conflict_preserves_newer_valid_selection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = tmp_path / "models.yaml"
    session_path = tmp_path / "model-session.json"
    _write_models_config(config_path)
    session_path.write_text(
        json.dumps({"version": 1, "revision": 2, "current_model_id": "removed-model"}),
        encoding="utf-8",
    )
    original_select = ModelSessionStore.select
    calls = 0

    def race_once(
        self: ModelSessionStore,
        model_id: str,
        *,
        expected: object,
    ) -> ModelSessionState:
        nonlocal calls
        calls += 1
        if calls == 1:
            current = self.read(default_model_id="local_qwen")
            original_select(self, "mimo_cloud", expected=current.file_version)
            raise ConfigVersionConflict("simulated session race")
        return original_select(self, model_id, expected=expected)  # type: ignore[arg-type]

    monkeypatch.setattr(ModelSessionStore, "select", race_once)

    control = ModelControlPlane.from_config_file(config_path, session_path=session_path)

    assert calls == 1
    assert control.current_model().id == "mimo_cloud"
    assert control.state.file_revision == 3


def test_stale_repair_retries_newer_invalid_selection_only_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = tmp_path / "models.yaml"
    session_path = tmp_path / "model-session.json"
    _write_models_config(config_path)
    session_path.write_text(
        json.dumps({"version": 1, "revision": 2, "current_model_id": "removed-model"}),
        encoding="utf-8",
    )
    original_select = ModelSessionStore.select
    calls = 0

    def always_race(
        self: ModelSessionStore,
        model_id: str,
        *,
        expected: object,
    ) -> ModelSessionState:
        nonlocal calls
        calls += 1
        current = self.read(default_model_id="local_qwen")
        original_select(self, f"still-invalid-{calls}", expected=current.file_version)
        raise ConfigVersionConflict("simulated session race")

    monkeypatch.setattr(ModelSessionStore, "select", always_race)

    with pytest.raises(ConfigVersionConflict, match="simulated session race"):
        ModelControlPlane.from_config_file(config_path, session_path=session_path)

    assert calls == 2


def test_invalid_user_switch_keeps_state_and_never_resolves_a_provider(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = tmp_path / "models.yaml"
    session_path = tmp_path / "model-session.json"
    _write_models_config(config_path)
    monkeypatch.setenv("RAG_AGENT_MODELS_PATH", str(config_path))
    resolved_aliases: list[str] = []

    def resolve_model(self: ModelRegistry, alias: str) -> object:
        del self
        resolved_aliases.append(alias)
        return object()

    monkeypatch.setattr(ModelRegistry, "resolve", resolve_model)
    control = ModelControlPlane.from_env(
        initial_model_id="local_qwen",
        session_path=session_path,
    )

    with pytest.raises(UnknownModelAliasError, match="missing"):
        control.switch_model("missing", requested_by="user")

    assert control.current_model().id == "local_qwen"
    assert resolved_aliases == []
    assert not session_path.exists()


def test_agent_model_cli_uses_session_state_not_yaml(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = tmp_path / "models.yaml"
    session_path = tmp_path / "model-session.json"
    _write_models_config(config_path)
    before = config_path.read_text(encoding="utf-8")
    monkeypatch.setenv("RAG_AGENT_MODELS_PATH", str(config_path))
    runner = CliRunner()

    listed = runner.invoke(
        agent_app,
        ["model", "list", "--session-path", str(session_path)],
        env={"COLUMNS": "240"},
    )
    current = runner.invoke(
        agent_app,
        ["model", "current", "--session-path", str(session_path)],
        env={"COLUMNS": "240"},
    )
    switched = runner.invoke(
        agent_app,
        ["model", "switch", "mimo_cloud", "--session-path", str(session_path)],
        env={"COLUMNS": "240"},
    )
    after = runner.invoke(
        agent_app,
        ["model", "current", "--session-path", str(session_path)],
        env={"COLUMNS": "240"},
    )

    assert listed.exit_code == 0, listed.output
    assert "local_qwen" in listed.output
    assert "mimo_cloud" in listed.output
    assert current.exit_code == 0, current.output
    assert "local_qwen" in current.output
    assert switched.exit_code == 0, switched.output
    assert "mimo_cloud" in switched.output
    assert after.exit_code == 0, after.output
    assert "mimo_cloud" in after.output
    assert config_path.read_text(encoding="utf-8") == before

def test_local_provider_probe_rejects_endpoint_conflict() -> None:
    probe = LocalProviderProbe(
        request_json=lambda *_: {
            "data": [
                {"id": "other-model"}
            ]
        },
    )

    with pytest.raises(
        EndpointConflictError,
        match="endpoint conflict",
    ):
        probe.ensure_ready(
            ModelSpec(
                id="local_qwen",
                provider="qwen",
                provider_model=(
                    "models--mlx-community--"
                    "Qwen3-14B-4bit"
                ),
                context_window=32768,
                supports_tools=True,
                supports_structured_output=True,
                location="local",
                runtime=ModelRuntimeSpec(
                    health_url=(
                        "http://127.0.0.1:"
                        "8080/v1/models"
                    ),
                    launch_command=(
                        "uv",
                        "run",
                        "python",
                        "-m",
                        "mlx_lm.server",
                    ),
                    expected_model_contains=(
                        "Qwen3-14B"
                    ),
                ),
            )
        )

def test_bundled_qwen14_runtime_accepts_mlx_canonical_model_id() -> None:
    spec = ModelCatalog.from_config_file(
        Path("configs/models.yaml")
    ).get("qwen3_14b_mlx_4bit")

    assert spec.runtime is not None

    probe = LocalProviderProbe(
        request_json=lambda *_: {
            "data": [
                {
                    "id": (
                        "mlx-community/"
                        "Qwen3-14B-4bit"
                    )
                }
            ]
        },
    )

    probe.ensure_ready(spec)

