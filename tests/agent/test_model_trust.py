from __future__ import annotations

import base64
import json
import multiprocessing
import os
import stat
from pathlib import Path

import pytest

from agent_runtime import model_config_io
from agent_runtime.core.llm_config import AgentModelsConfig, ModelProvider, ModelSpec
from agent_runtime.core.llm_registry import ModelRegistry
from agent_runtime.model_config_io import CommitOutcomeUnknown, UntrustedConfigPathError
from agent_runtime.model_definition import canonical_definition_json
from agent_runtime.model_trust import (
    BindingAuthenticationError,
    ModelBindingTrustDomain,
    TrustDomainNotInitializedError,
    TrustDomainValidationError,
    TrustedDefinitionNotFoundError,
    TrustedDefinitionValidationError,
    TrustedModelDefinitionArchive,
)


def _definition(model: str = "main-model"):
    config = AgentModelsConfig(
        models={
            "main": ModelSpec(
                provider=ModelProvider.OLLAMA,
                model=model,
                base_url="http://localhost:11434",
                context_window_tokens=32_768,
            )
        },
        default_model="main",
    )
    return ModelRegistry(config).get_model_definition("main")


def _paths(tmp_path: Path) -> tuple[Path, Path, Path]:
    workspace = tmp_path / "workspace"
    config_root = tmp_path / "config"
    workspace.mkdir(mode=0o700)
    config_root.mkdir(mode=0o700)
    return workspace, config_root / "binding-trust.json", config_root / "model-definitions"


def _trust_worker(
    path: str,
    workspace: str,
    start: multiprocessing.synchronize.Event,
    outcomes: multiprocessing.Queue[tuple[str, str]],
) -> None:
    if not start.wait(timeout=10):
        raise RuntimeError("trust initializer did not receive start signal")
    domain = ModelBindingTrustDomain(
        Path(path),
        workspace=Path(workspace),
        worktree=Path(workspace),
    )
    status = domain.initialize()
    outcomes.put((status.trust_domain_id, status.signing_key_id))


def _archive_worker(
    path: str,
    workspace: str,
    start: multiprocessing.synchronize.Event,
    outcomes: multiprocessing.Queue[str],
) -> None:
    if not start.wait(timeout=10):
        raise RuntimeError("archive installer did not receive start signal")
    archive = TrustedModelDefinitionArchive(
        Path(path),
        workspace=Path(workspace),
        worktree=Path(workspace),
    )
    outcomes.put(archive.ensure(_definition()))


def test_trust_status_and_sign_never_initialize_implicitly(tmp_path: Path) -> None:
    workspace, trust_path, _ = _paths(tmp_path)
    domain = ModelBindingTrustDomain(
        trust_path,
        workspace=workspace,
        worktree=workspace,
    )

    with pytest.raises(TrustDomainNotInitializedError, match="agent model trust init"):
        domain.status()
    with pytest.raises(TrustDomainNotInitializedError, match="agent model trust init"):
        domain.sign({"thread_id": "thread-1", "turn_id": "turn-1"})

    assert not trust_path.exists()


def test_trust_initialize_is_idempotent_private_and_redacted(tmp_path: Path) -> None:
    workspace, trust_path, _ = _paths(tmp_path)
    domain = ModelBindingTrustDomain(trust_path, workspace=workspace, worktree=workspace)

    first = domain.initialize()
    second = domain.initialize()

    assert first == second == domain.status()
    assert stat.S_IMODE(trust_path.parent.stat().st_mode) == 0o700
    assert stat.S_IMODE(trust_path.stat().st_mode) == 0o600
    assert first.trust_domain_id
    assert first.signing_key_id.startswith("sha256:")
    assert "key_base64" not in repr(first)
    assert not hasattr(first, "hmac_key_base64")


def test_concurrent_trust_initializers_converge_without_overwrite(tmp_path: Path) -> None:
    workspace, trust_path, _ = _paths(tmp_path)
    start = multiprocessing.Event()
    outcomes: multiprocessing.Queue[tuple[str, str]] = multiprocessing.Queue()
    processes = [
        multiprocessing.Process(
            target=_trust_worker,
            args=(str(trust_path), str(workspace), start, outcomes),
        )
        for _ in range(2)
    ]
    for process in processes:
        process.start()
    start.set()
    for process in processes:
        process.join(timeout=10)
        assert process.exitcode == 0

    assert outcomes.get(timeout=2) == outcomes.get(timeout=2)


def test_trust_initialization_crash_before_install_leaves_no_final_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace, trust_path, _ = _paths(tmp_path)
    domain = ModelBindingTrustDomain(trust_path, workspace=workspace, worktree=workspace)

    def fail_before_install(*_args: object, **_kwargs: object) -> str:
        raise OSError("install interrupted")

    monkeypatch.setattr("agent_runtime.model_trust.atomic_install_bytes", fail_before_install)
    with pytest.raises(OSError, match="install interrupted"):
        domain.initialize()

    assert not trust_path.exists()


def test_trust_post_install_unknown_requires_durability_before_success(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace, trust_path, _ = _paths(tmp_path)
    domain = ModelBindingTrustDomain(trust_path, workspace=workspace, worktree=workspace)
    real_directory_fsync = model_config_io._fsync_directory

    def fail_directory_fsync(_path: Path) -> None:
        raise OSError("directory fsync unavailable")

    monkeypatch.setattr(model_config_io, "_fsync_directory", fail_directory_fsync)
    with pytest.raises(CommitOutcomeUnknown):
        domain.initialize()
    assert trust_path.is_file()

    with pytest.raises(CommitOutcomeUnknown):
        domain.initialize()
    monkeypatch.setattr(model_config_io, "_fsync_directory", real_directory_fsync)
    assert domain.initialize() == domain.status()


@pytest.mark.parametrize(
    "mutate",
    [
        lambda payload: {**payload, "unexpected": True},
        lambda payload: {**payload, "version": True},
        lambda payload: {**payload, "trust_domain_id": "not-a-uuid"},
        lambda payload: {**payload, "hmac_key_base64": "%%%"},
        lambda payload: {**payload, "signing_key_id": "sha256:" + "0" * 64},
    ],
)
def test_trust_status_rejects_strict_schema_or_key_id_mismatch(
    tmp_path: Path,
    mutate,
) -> None:
    workspace, trust_path, _ = _paths(tmp_path)
    key = b"k" * 32
    payload = {
        "version": 1,
        "trust_domain_id": "d8ec2a14-31fd-4d7b-96e5-452296c361b0",
        "signing_key_id": "sha256:95c2cc7d82b5c9040c35f4f350e2187e3390c9e7230366f7c0f216c1d0a6f010",
        "hmac_key_base64": base64.b64encode(key).decode("ascii"),
    }
    trust_path.write_text(json.dumps(mutate(payload)), encoding="utf-8")
    os.chmod(trust_path, 0o600)
    domain = ModelBindingTrustDomain(trust_path, workspace=workspace, worktree=workspace)

    with pytest.raises(TrustDomainValidationError):
        domain.status()


def test_trust_rejects_symlink_unsafe_file_mode_and_parent_mode(tmp_path: Path) -> None:
    workspace, trust_path, _ = _paths(tmp_path)
    target = trust_path.with_name("real-trust.json")
    target.write_text("{}", encoding="utf-8")
    os.chmod(target, 0o600)
    trust_path.symlink_to(target)
    with pytest.raises((UntrustedConfigPathError, TrustDomainValidationError)):
        ModelBindingTrustDomain(trust_path, workspace=workspace, worktree=workspace).status()

    trust_path.unlink()
    domain = ModelBindingTrustDomain(trust_path, workspace=workspace, worktree=workspace)
    domain.initialize()
    os.chmod(trust_path, 0o644)
    with pytest.raises(TrustDomainValidationError, match="0600"):
        domain.status()
    os.chmod(trust_path, 0o600)
    os.chmod(trust_path.parent, 0o755)
    with pytest.raises(TrustDomainValidationError, match="0700"):
        domain.status()


def test_trust_rejects_duplicate_json_keys(tmp_path: Path) -> None:
    workspace, trust_path, _ = _paths(tmp_path)
    trust_path.write_bytes(
        b'{"version":1,"version":1,"trust_domain_id":"x",'
        b'"signing_key_id":"x","hmac_key_base64":"eA=="}'
    )
    os.chmod(trust_path, 0o600)

    with pytest.raises(TrustDomainValidationError, match="duplicate"):
        ModelBindingTrustDomain(
            trust_path,
            workspace=workspace,
            worktree=workspace,
        ).status()


def test_binding_signature_covers_complete_association(tmp_path: Path) -> None:
    workspace, trust_path, _ = _paths(tmp_path)
    domain = ModelBindingTrustDomain(trust_path, workspace=workspace, worktree=workspace)
    domain.initialize()
    association = {
        "authentication_schema_version": 1,
        "thread_id": "thread-1",
        "turn_id": "turn-1",
        "alias": "main",
        "selection_requester": "user",
        "binding": {"definition_revision": _definition().definition_revision},
    }

    signature = domain.sign(association)
    domain.verify(association, signature)

    for changed in (
        {**association, "thread_id": "thread-2"},
        {**association, "turn_id": "turn-2"},
        {**association, "selection_requester": "system"},
        {**association, "binding": {"definition_revision": _definition("other").definition_revision}},
    ):
        with pytest.raises(BindingAuthenticationError):
            domain.verify(changed, signature)


def test_archive_ensure_and_load_use_canonical_digest_and_private_mode(tmp_path: Path) -> None:
    workspace, _, archive_path = _paths(tmp_path)
    archive = TrustedModelDefinitionArchive(
        archive_path,
        workspace=workspace,
        worktree=workspace,
    )
    definition = _definition()

    revision = archive.ensure(definition)
    stored_path = archive_path / f"{revision}.json"

    assert revision == definition.definition_revision
    assert stored_path.read_bytes() == canonical_definition_json(definition)
    assert stat.S_IMODE(stored_path.stat().st_mode) == 0o600
    assert archive.load(revision) == definition


def test_concurrent_identical_archive_installs_converge(tmp_path: Path) -> None:
    workspace, _, archive_path = _paths(tmp_path)
    start = multiprocessing.Event()
    outcomes: multiprocessing.Queue[str] = multiprocessing.Queue()
    processes = [
        multiprocessing.Process(
            target=_archive_worker,
            args=(str(archive_path), str(workspace), start, outcomes),
        )
        for _ in range(2)
    ]
    for process in processes:
        process.start()
    start.set()
    for process in processes:
        process.join(timeout=10)
        assert process.exitcode == 0

    assert outcomes.get(timeout=2) == outcomes.get(timeout=2) == _definition().definition_revision


@pytest.mark.parametrize("existing", [b"{}", b"not-json", b'{"pretty": true}\n'])
def test_archive_conflicting_digest_path_fails_without_replacement(
    tmp_path: Path,
    existing: bytes,
) -> None:
    workspace, _, archive_path = _paths(tmp_path)
    archive_path.mkdir(mode=0o700)
    definition = _definition()
    target = archive_path / f"{definition.definition_revision}.json"
    target.write_bytes(existing)
    os.chmod(target, 0o600)
    archive = TrustedModelDefinitionArchive(
        archive_path,
        workspace=workspace,
        worktree=workspace,
    )

    with pytest.raises(TrustedDefinitionValidationError):
        archive.ensure(definition)

    assert target.read_bytes() == existing


def test_archive_rejects_noncanonical_valid_definition_and_wrong_file_mode(
    tmp_path: Path,
) -> None:
    workspace, _, archive_path = _paths(tmp_path)
    archive = TrustedModelDefinitionArchive(archive_path, workspace=workspace, worktree=workspace)
    definition = _definition()
    revision = archive.ensure(definition)
    target = archive_path / f"{revision}.json"
    parsed = json.loads(target.read_bytes())
    target.write_text(json.dumps(parsed, indent=2), encoding="utf-8")
    with pytest.raises(TrustedDefinitionValidationError, match="canonical"):
        archive.load(revision)
    target.write_bytes(canonical_definition_json(definition))
    os.chmod(target, 0o644)
    with pytest.raises(TrustedDefinitionValidationError, match="0600"):
        archive.load(revision)


def test_archive_rejects_canonical_definition_under_wrong_digest(tmp_path: Path) -> None:
    workspace, _, archive_path = _paths(tmp_path)
    archive = TrustedModelDefinitionArchive(archive_path, workspace=workspace, worktree=workspace)
    original = _definition()
    replacement = _definition("replacement-model")
    revision = archive.ensure(original)
    target = archive_path / f"{revision}.json"
    target.write_bytes(canonical_definition_json(replacement))
    os.chmod(target, 0o600)

    with pytest.raises(TrustedDefinitionValidationError, match="digest"):
        archive.load(revision)


def test_archive_rejects_workspace_and_symlinked_roots(tmp_path: Path) -> None:
    workspace, _, archive_path = _paths(tmp_path)
    with pytest.raises(UntrustedConfigPathError):
        TrustedModelDefinitionArchive(
            workspace / "model-definitions",
            workspace=workspace,
            worktree=workspace,
        )

    real_archive = archive_path.with_name("real-archive")
    real_archive.mkdir(mode=0o700)
    archive_path.symlink_to(real_archive, target_is_directory=True)
    with pytest.raises((UntrustedConfigPathError, TrustedDefinitionValidationError)):
        TrustedModelDefinitionArchive(
            archive_path,
            workspace=workspace,
            worktree=workspace,
        )


def test_archive_missing_or_malformed_revision_fails_closed(tmp_path: Path) -> None:
    workspace, _, archive_path = _paths(tmp_path)
    archive = TrustedModelDefinitionArchive(archive_path, workspace=workspace, worktree=workspace)

    with pytest.raises(TrustedDefinitionNotFoundError):
        archive.load("sha256:" + "0" * 64)
    with pytest.raises(TrustedDefinitionValidationError):
        archive.load("../../binding-trust")
    assert not archive_path.exists()


def test_archive_post_install_unknown_requires_durability_before_success(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace, _, archive_path = _paths(tmp_path)
    archive = TrustedModelDefinitionArchive(archive_path, workspace=workspace, worktree=workspace)
    definition = _definition()
    real_directory_fsync = model_config_io._fsync_directory

    def fail_directory_fsync(_path: Path) -> None:
        raise OSError("directory fsync unavailable")

    monkeypatch.setattr(model_config_io, "_fsync_directory", fail_directory_fsync)
    with pytest.raises(CommitOutcomeUnknown):
        archive.ensure(definition)
    with pytest.raises(CommitOutcomeUnknown):
        archive.ensure(definition)

    monkeypatch.setattr(model_config_io, "_fsync_directory", real_directory_fsync)
    assert archive.ensure(definition) == definition.definition_revision
