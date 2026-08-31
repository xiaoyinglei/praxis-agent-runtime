from __future__ import annotations

import multiprocessing
from collections.abc import Callable
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from agent_runtime.model_config_io import ConfigVersionConflict, FileVersion, file_fingerprint
from agent_runtime.model_registry import (
    InvalidUnsetPath,
    ModelDefinitionPatch,
    ModelGenerationDefaults,
    ModelRuntimeDeclaration,
    RegistryCollisionError,
    RegistryCommitOutcomeUnknown,
    RegistryConfigValidationError,
    RegistryEntryNotFound,
    RegistryOverrideActiveError,
    UserModelDefinition,
    UserModelRegistryDocument,
    UserModelRegistryStore,
)


def _definition(**updates: object) -> UserModelDefinition:
    values: dict[str, object] = {
        "provider": "openai_compatible",
        "model": "Qwen/Qwen3.5-9B",
        "location": "local",
        "base_url": "http://127.0.0.1:8080/v1",
        "context_window_tokens": 32_768,
        "max_tokens": 2_048,
    }
    values.update(updates)
    return UserModelDefinition.model_validate(values)


def _store(tmp_path: Path, **updates: object) -> UserModelRegistryStore:
    workspace = tmp_path / "workspace"
    config = tmp_path / "config"
    workspace.mkdir(exist_ok=True)
    config.mkdir(exist_ok=True)
    values: dict[str, object] = {
        "path": config / "models.yaml",
        "workspace": workspace,
        "worktree": workspace,
        "built_in_aliases": {"builtin"},
        "whole_catalog_override_active": False,
    }
    values.update(updates)
    return UserModelRegistryStore(**values)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "alias",
    [
        "A",
        "two words",
        "-leading",
        "trailing_",
        "a" * 65,
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
    ],
)
def test_schema_rejects_invalid_or_reserved_alias(alias: str) -> None:
    with pytest.raises(ValidationError):
        UserModelRegistryDocument(revision=0, models={alias: _definition()})


@pytest.mark.parametrize("alias", ["a", "a.b-c_d9", "a" * 64])
def test_schema_accepts_exact_alias_grammar(alias: str) -> None:
    document = UserModelRegistryDocument(revision=0, models={alias: _definition()})
    assert tuple(document.models) == (alias,)


@pytest.mark.parametrize("provider", ["openai", "anthropic", "custom"])
def test_schema_rejects_unsupported_provider(provider: str) -> None:
    with pytest.raises(ValidationError):
        _definition(provider=provider)


@pytest.mark.parametrize(
    "base_url",
    [
        "localhost:8080",
        "/v1",
        "ftp://host/v1",
        "http://",
        "http://host:invalid/v1",
        "http://key@host/v1",
        "https://host/v1?api_key=x",
        " http://host/v1",
        "http://host/v1 ",
        "http://local host/v1",
        "http://host/v 1",
        "http://local\u00a0host/v1",
        "http://host/v\u20031",
        "http://host/v1\nmodels",
        "http://host\t.example/v1",
    ],
)
def test_schema_rejects_non_absolute_or_secret_bearing_url(base_url: str) -> None:
    with pytest.raises(ValidationError):
        _definition(base_url=base_url)


@pytest.mark.parametrize(
    "base_url",
    [
        "http://local host/v1",
        "http://host/v 1",
        "http://local\u00a0host/v1",
        "http://host/v\u20031",
    ],
)
def test_schema_rejects_any_whitespace_in_base_url(base_url: str) -> None:
    with pytest.raises(ValidationError, match="whitespace"):
        _definition(base_url=base_url)


def test_schema_requires_cloud_endpoint_and_matching_location() -> None:
    with pytest.raises(ValidationError, match="cloud model requires base_url"):
        _definition(location="cloud", base_url=None)
    with pytest.raises(ValidationError, match="location"):
        _definition(location="cloud", base_url="http://127.0.0.1:8080/v1")
    with pytest.raises(ValidationError, match="location"):
        _definition(location="local", base_url="https://api.example.com/v1")
    with pytest.raises(ValidationError, match="local"):
        _definition(provider="ollama", location=None, base_url="https://api.example.com/v1")


@pytest.mark.parametrize(
    "health_url",
    [
        " http://127.0.0.1/health",
        "http://local host/health",
        "http://localhost/health\u00a0check",
        "http://localhost/health\u2003check",
        "http://127.0.0.1/health\nnext",
        "http://local\thost/health",
        "http://127.0.0.1/health\x7f",
    ],
)
def test_schema_rejects_whitespace_or_control_characters_in_health_url(health_url: str) -> None:
    with pytest.raises(ValidationError, match="whitespace|control"):
        ModelRuntimeDeclaration(health_url=health_url)


@pytest.mark.parametrize("api_key_env", ["sk-secret", "API KEY", "1TOKEN", "${TOKEN}", "TOKEN=value"])
def test_schema_accepts_only_credential_environment_names(api_key_env: str) -> None:
    with pytest.raises(ValidationError):
        _definition(api_key_env=api_key_env)


def test_schema_launch_command_is_shell_free_argv() -> None:
    runtime = ModelRuntimeDeclaration(launch_command=["uv", "run", "server", "--port", "8080"])
    assert runtime.launch_command == ("uv", "run", "server", "--port", "8080")
    with pytest.raises(ValidationError):
        ModelRuntimeDeclaration.model_validate({"launch_command": "uv run server"})
    with pytest.raises(ValidationError):
        ModelRuntimeDeclaration(launch_command=["uv", ""])
    with pytest.raises(ValidationError):
        ModelRuntimeDeclaration(launch_command=["uv\x00run"])


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("temperature", -0.1),
        ("temperature", 2.1),
        ("temperature", float("nan")),
        ("top_p", 0.0),
        ("top_p", 1.1),
        ("top_p", float("inf")),
        ("parallel_tool_calls", 1),
        ("seed", -(2**63) - 1),
        ("seed", 2**63),
    ],
)
def test_schema_typed_generation_defaults(field: str, value: object) -> None:
    with pytest.raises(ValidationError):
        ModelGenerationDefaults.model_validate({field: value})


@pytest.mark.parametrize("field", ["temperature", "top_p", "parallel_tool_calls", "seed"])
def test_schema_rejects_explicit_null_default(field: str) -> None:
    with pytest.raises(ValidationError, match="null"):
        ModelGenerationDefaults.model_validate({field: None})


def test_schema_normalized_mapping_omits_absent_values() -> None:
    definition = _definition(defaults={"temperature": 0.0})
    payload = definition.to_persisted_mapping()
    assert payload["defaults"] == {"temperature": 0.0}
    assert "tokenizer_model" not in payload
    assert "api_key_env" not in payload
    assert "runtime" not in payload


@pytest.mark.parametrize(
    "payload",
    [
        {"surprise": True},
        {"runtime": {"surprise": True}},
        {"defaults": {"surprise": True}},
    ],
)
def test_schema_rejects_unknown_fields(payload: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        _definition(**payload)


@pytest.mark.parametrize(
    "updates",
    [
        {"max_tokens": 0},
        {"timeout_seconds": 0},
        {"context_window_tokens": 0},
        {"request_context_tokens": 0},
        {"max_tokens": 4097, "context_window_tokens": 4096},
        {"max_tokens": 2048, "request_context_tokens": 1024},
        {"request_context_tokens": 4097, "context_window_tokens": 4096},
        {"input_cost_per_1m": -1},
    ],
)
def test_schema_rejects_invalid_budgets(updates: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        _definition(**updates)


def test_read_rejects_explicit_yaml_null_in_defaults(tmp_path: Path) -> None:
    store = _store(tmp_path)
    store.path.write_text(
        "version: 1\nrevision: 0\nmodels:\n"
        "  mine:\n    provider: mlx\n    model: m\n"
        "    defaults:\n      temperature: null\n",
        encoding="utf-8",
    )
    with pytest.raises(ValidationError, match="null"):
        store.read()


@pytest.mark.parametrize(
    ("yaml_text", "duplicate"),
    [
        (
            "version: 1\nrevision: 0\nmodels:\n"
            "  mine: {provider: mlx, model: first}\n"
            "  mine: {provider: mlx, model: second}\n",
            "mine",
        ),
        (
            "version: 1\nrevision: 0\nmodels:\n  mine:\n"
            "    provider: mlx\n    provider: ollama\n    model: m\n",
            "provider",
        ),
        (
            "version: 1\nrevision: 0\nmodels:\n  mine:\n"
            "    provider: mlx\n    model: m\n    defaults:\n"
            "      temperature: 0.1\n      temperature: 0.2\n",
            "temperature",
        ),
    ],
)
def test_read_rejects_duplicate_yaml_mapping_keys(
    tmp_path: Path,
    yaml_text: str,
    duplicate: str,
) -> None:
    store = _store(tmp_path)
    store.path.write_text(yaml_text, encoding="utf-8")

    with pytest.raises(RegistryConfigValidationError, match=f"duplicate.*{duplicate}"):
        store.read()


def test_add_rejects_user_and_builtin_collisions(tmp_path: Path) -> None:
    store = _store(tmp_path)
    initial = store.read().version
    committed = store.add("mine", _definition(), expected=initial)
    with pytest.raises(RegistryCollisionError, match="already exists"):
        store.add("mine", _definition(model="other"), expected=committed.snapshot.version)
    with pytest.raises(RegistryCollisionError, match="built-in"):
        store.add("builtin", _definition(), expected=committed.snapshot.version)


def test_add_revalidates_adversarial_constructed_definition_under_lock(tmp_path: Path) -> None:
    store = _store(tmp_path)
    bypass = UserModelDefinition.model_construct(
        provider="anthropic",
        model="",
        max_tokens=0,
        context_window_tokens=0,
    )

    with pytest.raises(ValidationError):
        store.add("mine", bypass, expected=store.read().version)

    assert not store.path.exists()


def test_patch_update_and_complete_replacement(tmp_path: Path) -> None:
    store = _store(tmp_path)
    added = store.add("mine", _definition(tokenizer_model="old"), expected=store.read().version)
    patched = store.update(
        "mine",
        ModelDefinitionPatch(
            changes={"max_tokens": 1024, "defaults": {"temperature": 0.5}},
        ),
        expected=added.snapshot.version,
    )
    updated = patched.snapshot.document.models["mine"]
    assert updated.model == "Qwen/Qwen3.5-9B"
    assert updated.tokenizer_model == "old"
    assert updated.max_tokens == 1024
    assert updated.defaults.temperature == 0.5

    replacement = _definition(provider="ollama", model="qwen:latest", base_url=None)
    replaced = store.update(
        "mine",
        ModelDefinitionPatch(replacement=replacement),
        expected=patched.snapshot.version,
    )
    assert replaced.snapshot.document.models["mine"] == replacement


def test_replacement_revalidates_adversarial_constructed_definition_under_lock(tmp_path: Path) -> None:
    store = _store(tmp_path)
    added = store.add("mine", _definition(), expected=store.read().version)
    before = store.path.read_bytes()
    bypass = UserModelDefinition.model_construct(
        provider="anthropic",
        model="",
        max_tokens=0,
        context_window_tokens=0,
    )

    with pytest.raises(ValidationError):
        store.update(
            "mine",
            ModelDefinitionPatch(replacement=bypass),
            expected=added.snapshot.version,
        )

    assert store.path.read_bytes() == before


@pytest.mark.parametrize(
    "changes",
    [
        {"protocol": None},
        {"location": None},
        {"runtime": None},
        {"api_key_env": None},
        {"runtime": {"health_url": None}},
        {"defaults": {"temperature": None}},
    ],
)
def test_patch_rejects_none_deletions_at_every_nesting_level(changes: dict[str, object]) -> None:
    with pytest.raises(ValidationError, match="unset_paths"):
        ModelDefinitionPatch(changes=changes)


def test_store_rejects_constructed_patch_with_none_deletion(tmp_path: Path) -> None:
    store = _store(tmp_path)
    added = store.add(
        "mine",
        _definition(protocol="openai_compatible"),
        expected=store.read().version,
    )
    before = store.path.read_bytes()
    bypass = ModelDefinitionPatch.model_construct(changes={"protocol": None})

    with pytest.raises(ValueError, match="unset_paths"):
        store.update("mine", bypass, expected=added.snapshot.version)

    assert store.path.read_bytes() == before


@pytest.mark.parametrize(
    ("changes", "unset_paths"),
    [
        ({"protocol": None}, ("tokenizer_model",)),
        ({}, ("runtime.launch_command",)),
        ({"max_tokens": 1024}, ("tokenizer_model",)),
    ],
)
def test_store_revalidates_constructed_replacement_patch_shape_under_lock(
    tmp_path: Path,
    changes: dict[str, object],
    unset_paths: tuple[str, ...],
) -> None:
    store = _store(tmp_path)
    added = store.add("mine", _definition(), expected=store.read().version)
    before = store.path.read_bytes()
    bypass = ModelDefinitionPatch.model_construct(
        replacement=_definition(model="replacement"),
        changes=changes,
        unset_paths=unset_paths,
    )

    with pytest.raises(ValidationError):
        store.update("mine", bypass, expected=added.snapshot.version)

    assert store.path.read_bytes() == before


_UNSET_CASES: tuple[tuple[str, Callable[[UserModelDefinition], object]], ...] = (
    ("tokenizer_model", lambda value: value.tokenizer_model),
    ("provider_name", lambda value: value.provider_name),
    ("base_url", lambda value: value.base_url),
    ("api_key_env", lambda value: value.api_key_env),
    ("request_context_tokens", lambda value: value.request_context_tokens),
    ("input_cost_per_1m", lambda value: value.input_cost_per_1m),
    ("output_cost_per_1m", lambda value: value.output_cost_per_1m),
    ("cache_read_cost_per_1m", lambda value: value.cache_read_cost_per_1m),
    ("cache_write_cost_per_1m", lambda value: value.cache_write_cost_per_1m),
    ("runtime.health_url", lambda value: value.runtime and value.runtime.health_url),
    ("runtime.expected_model_contains", lambda value: value.runtime and value.runtime.expected_model_contains),
    ("defaults.temperature", lambda value: value.defaults.temperature),
    ("defaults.top_p", lambda value: value.defaults.top_p),
    ("defaults.parallel_tool_calls", lambda value: value.defaults.parallel_tool_calls),
    ("defaults.seed", lambda value: value.defaults.seed),
)


@pytest.mark.parametrize(("path", "read_value"), _UNSET_CASES)
def test_update_supports_every_allowed_unset_path(
    tmp_path: Path,
    path: str,
    read_value: Callable[[UserModelDefinition], object],
) -> None:
    store = _store(tmp_path)
    complete = _definition(
        tokenizer_model="tokenizer",
        provider_name="provider",
        api_key_env="MODEL_API_KEY",
        request_context_tokens=8192,
        input_cost_per_1m=1,
        output_cost_per_1m=2,
        cache_read_cost_per_1m=0.1,
        cache_write_cost_per_1m=0.2,
        defaults={"temperature": 0.4, "top_p": 0.9, "parallel_tool_calls": True, "seed": 7},
        runtime={
            "health_url": "http://127.0.0.1:8080/health",
            "expected_model_contains": "Qwen",
            "launch_command": ["uv", "run", "server"],
        },
    )
    added = store.add("mine", complete, expected=store.read().version)
    result = store.update(
        "mine",
        ModelDefinitionPatch(unset_paths=(path,)),
        expected=added.snapshot.version,
    )
    assert read_value(result.snapshot.document.models["mine"]) is None


@pytest.mark.parametrize("path", ["model", "runtime.launch_command", "defaults", "arbitrary.path"])
def test_update_rejects_invalid_unset_path(tmp_path: Path, path: str) -> None:
    store = _store(tmp_path)
    added = store.add("mine", _definition(), expected=store.read().version)
    with pytest.raises(InvalidUnsetPath):
        store.update(
            "mine",
            ModelDefinitionPatch(unset_paths=(path,)),
            expected=added.snapshot.version,
        )


def test_explicit_empty_launch_command_clears_collection(tmp_path: Path) -> None:
    store = _store(tmp_path)
    added = store.add(
        "mine",
        _definition(runtime={"launch_command": ["uv", "run", "server"]}),
        expected=store.read().version,
    )
    updated = store.update(
        "mine",
        ModelDefinitionPatch(changes={"runtime": {"launch_command": []}}),
        expected=added.snapshot.version,
    )
    assert updated.snapshot.document.models["mine"].runtime is not None
    assert updated.snapshot.document.models["mine"].runtime.launch_command == ()


def test_identical_normalized_update_is_noop(tmp_path: Path) -> None:
    store = _store(tmp_path)
    added = store.add("mine", _definition(), expected=store.read().version)
    before = store.path.read_bytes()
    result = store.update(
        "mine",
        ModelDefinitionPatch(changes={"max_tokens": 2048}),
        expected=added.snapshot.version,
    )
    assert result.changed is False
    assert result.snapshot.document.revision == 1
    assert result.snapshot.fingerprint == added.snapshot.fingerprint
    assert store.path.read_bytes() == before


def test_remove_retains_empty_versioned_document_and_monotonic_revision(tmp_path: Path) -> None:
    store = _store(tmp_path)
    added = store.add("mine", _definition(), expected=store.read().version)
    removed = store.remove("mine", expected=added.snapshot.version)
    assert removed.changed is True
    assert removed.snapshot.document == UserModelRegistryDocument(revision=2, models={})
    assert store.path.exists()
    assert yaml.safe_load(store.path.read_text(encoding="utf-8")) == {
        "version": 1,
        "revision": 2,
        "models": {},
    }
    with pytest.raises(RegistryEntryNotFound):
        store.remove("mine", expected=removed.snapshot.version)


@pytest.mark.parametrize("stale_part", ["revision", "fingerprint"])
def test_mutation_rejects_stale_revision_or_fingerprint(tmp_path: Path, stale_part: str) -> None:
    store = _store(tmp_path)
    current = store.read().version
    expected = FileVersion(
        revision=current.revision + (1 if stale_part == "revision" else 0),
        fingerprint=("0" * 64 if stale_part == "fingerprint" else current.fingerprint),
    )
    with pytest.raises(ConfigVersionConflict):
        store.add("mine", _definition(), expected=expected)


def _concurrent_add(
    path: str,
    workspace: str,
    expected: FileVersion,
    alias: str,
    start: multiprocessing.synchronize.Event,
    output: multiprocessing.queues.Queue,
) -> None:
    store = UserModelRegistryStore(
        path=Path(path),
        workspace=Path(workspace),
        worktree=Path(workspace),
        built_in_aliases=(),
        whole_catalog_override_active=False,
    )
    start.wait(timeout=10)
    try:
        store.add(alias, _definition(), expected=expected)
    except ConfigVersionConflict:
        output.put("conflict")
    else:
        output.put("committed")


def test_two_processes_cannot_lose_an_update(tmp_path: Path) -> None:
    store = _store(tmp_path, built_in_aliases=())
    expected = store.read().version
    context = multiprocessing.get_context("spawn")
    start = context.Event()
    output = context.Queue()
    processes = [
        context.Process(
            target=_concurrent_add,
            args=(str(store.path), str(tmp_path / "workspace"), expected, alias, start, output),
        )
        for alias in ("one", "two")
    ]
    for process in processes:
        process.start()
    start.set()
    results = [output.get(timeout=15) for _ in processes]
    for process in processes:
        process.join(timeout=15)
        assert process.exitcode == 0
    assert sorted(results) == ["committed", "conflict"]
    assert len(store.read().document.models) == 1


def test_whole_catalog_override_rejects_all_mutations(tmp_path: Path) -> None:
    store = _store(tmp_path, whole_catalog_override_active=True)
    expected = store.read().version
    with pytest.raises(RegistryOverrideActiveError, match="RAG_AGENT_MODELS"):
        store.add("mine", _definition(), expected=expected)
    assert not store.path.exists()


def test_unknown_commit_outcome_carries_receipt_and_reconciles_exact_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = _store(tmp_path)
    expected = store.read().version

    def fail_directory_fsync(_path: Path) -> None:
        raise OSError("directory fsync failed")

    monkeypatch.setattr("agent_runtime.model_config_io._fsync_directory", fail_directory_fsync)
    with pytest.raises(RegistryCommitOutcomeUnknown) as captured:
        store.add("mine", _definition(), expected=expected)
    receipt = captured.value.receipt
    assert receipt.base_version == expected
    assert receipt.intended_revision == 1
    assert receipt.intended_fingerprint == file_fingerprint(store.path.read_bytes())
    reconciled = store.reconcile(receipt)
    assert reconciled.document.revision == 1
    assert tuple(reconciled.document.models) == ("mine",)


def test_reconcile_rejects_any_state_other_than_exact_receipt(tmp_path: Path) -> None:
    store = _store(tmp_path)
    result = store.add("mine", _definition(), expected=store.read().version)
    store.path.write_text("version: 1\nrevision: 1\nmodels: {}\n", encoding="utf-8")
    with pytest.raises(ConfigVersionConflict, match="intended post-state"):
        store.reconcile(result.receipt)
