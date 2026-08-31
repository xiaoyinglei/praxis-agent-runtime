"""Behavior tests for crash-safe model configuration file I/O."""

from __future__ import annotations

import hashlib
import multiprocessing
import stat
import subprocess
import time
from pathlib import Path

import pytest

from agent_runtime import model_config_io
from agent_runtime.model_config_io import (
    CommitOutcomeUnknown,
    FileVersion,
    UntrustedConfigPathError,
    atomic_install_bytes,
    atomic_replace_bytes,
    discover_git_worktree,
    exclusive_config_lock,
    file_fingerprint,
    validate_user_config_path,
)


def _contending_lock_worker(path: str, events: multiprocessing.Queue[tuple[str, float]]) -> None:
    """Record lock acquisition time while holding the adjacent file lock."""

    target = Path(path)
    with exclusive_config_lock(target):
        events.put(("acquired", time.monotonic()))
        time.sleep(0.25)


def _install_config_worker(
    path: str,
    payload: bytes,
    outcomes: multiprocessing.Queue[str],
) -> None:
    """Race a no-replace install from a separate Python process."""

    outcomes.put(atomic_install_bytes(Path(path), payload))


def _make_git_worktree(tmp_path: Path) -> Path:
    worktree = tmp_path / "project"
    worktree.mkdir()
    subprocess.run(
        ["git", "init", "-q", str(worktree)],
        check=True,
        capture_output=True,
        text=True,
    )
    return worktree


class TestFileVersion:
    def test_is_immutable_value(self) -> None:
        version = FileVersion(revision=4, fingerprint="abc")

        assert version.revision == 4
        assert version.fingerprint == "abc"
        with pytest.raises(AttributeError):
            version.revision = 5  # type: ignore[misc]


class TestDiscoverGitWorktree:
    def test_returns_resolved_workspace_when_not_in_git_repository(
        self, tmp_path: Path
    ) -> None:
        workspace = tmp_path / "ordinary"
        workspace.mkdir()

        assert discover_git_worktree(workspace) == workspace.resolve()

    def test_returns_git_top_level_for_nested_workspace(self, tmp_path: Path) -> None:
        worktree = _make_git_worktree(tmp_path)
        workspace = worktree / "apps" / "praxis"
        workspace.mkdir(parents=True)

        assert discover_git_worktree(workspace) == worktree.resolve()


class TestValidateUserConfigPath:
    def test_rejects_absolute_external_path(self, tmp_path: Path) -> None:
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        external = tmp_path / "external" / "models.json"

        with pytest.raises(UntrustedConfigPathError):
            validate_user_config_path(
                external,
                workspace=workspace,
                worktree=workspace,
            )

    def test_resolves_relative_path_from_workspace(self, tmp_path: Path) -> None:
        workspace = tmp_path / "workspace"
        workspace.mkdir()

        result = validate_user_config_path(
            Path(".praxis/models.json"),
            workspace=workspace,
            worktree=workspace,
        )

        assert result == (workspace / ".praxis" / "models.json").resolve()

    def test_accepts_direct_workspace_path(self, tmp_path: Path) -> None:
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        direct = workspace / ".praxis" / "models.json"

        assert (
            validate_user_config_path(
                direct,
                workspace=workspace,
                worktree=workspace,
            )
            == direct.resolve()
        )

    def test_accepts_path_in_worktree_outside_nested_workspace(
        self, tmp_path: Path
    ) -> None:
        worktree = _make_git_worktree(tmp_path)
        workspace = worktree / "apps" / "praxis"
        workspace.mkdir(parents=True)
        shared = worktree / ".praxis" / "models.json"

        assert (
            validate_user_config_path(
                shared,
                workspace=workspace,
                worktree=worktree,
            )
            == shared.resolve()
        )

    def test_accepts_symlink_resolving_into_workspace(self, tmp_path: Path) -> None:
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        target = workspace / ".praxis" / "models.json"
        target.parent.mkdir()
        target.write_bytes(b"{}")
        alias = tmp_path / "workspace-link.json"
        alias.symlink_to(target)

        assert (
            validate_user_config_path(
                alias,
                workspace=workspace,
                worktree=workspace,
            )
            == target.resolve()
        )

    def test_accepts_symlink_resolving_elsewhere_in_worktree(self, tmp_path: Path) -> None:
        worktree = _make_git_worktree(tmp_path)
        workspace = worktree / "apps" / "praxis"
        workspace.mkdir(parents=True)
        target = worktree / ".praxis" / "models.json"
        target.parent.mkdir()
        target.write_bytes(b"{}")
        alias = workspace / "models-link.json"
        alias.symlink_to(target)

        assert (
            validate_user_config_path(
                alias,
                workspace=workspace,
                worktree=worktree,
            )
            == target.resolve()
        )

    def test_rejects_symlink_resolving_outside_worktree(self, tmp_path: Path) -> None:
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        external = tmp_path / "external" / "models.json"
        external.parent.mkdir()
        external.write_bytes(b"{}")
        alias = workspace / "models-link.json"
        alias.symlink_to(external)

        with pytest.raises(UntrustedConfigPathError):
            validate_user_config_path(
                alias,
                workspace=workspace,
                worktree=workspace,
            )


class TestAtomicConfigWrites:
    def test_fingerprint_is_sha256_of_exact_payload(self) -> None:
        payload = b'{"model":"local"}\n'

        assert file_fingerprint(payload) == hashlib.sha256(payload).hexdigest()

    def test_replaces_target_only_after_temp_file_is_durable(self, tmp_path: Path) -> None:
        target = tmp_path / "models.json"
        target.write_bytes(b"old")
        payload = b"new"

        atomic_replace_bytes(
            target,
            payload,
            intended_fingerprint=file_fingerprint(payload),
        )

        assert target.read_bytes() == payload

    def test_newly_written_config_is_user_only(self, tmp_path: Path) -> None:
        target = tmp_path / "models.json"
        payload = b"private"

        atomic_replace_bytes(
            target,
            payload,
            intended_fingerprint=file_fingerprint(payload),
        )

        assert stat.S_IMODE(target.stat().st_mode) == 0o600

    def test_rejects_mismatched_intended_fingerprint(self, tmp_path: Path) -> None:
        target = tmp_path / "models.json"

        with pytest.raises(ValueError, match="fingerprint"):
            atomic_replace_bytes(
                target,
                b"new",
                intended_fingerprint=file_fingerprint(b"other"),
            )

    def test_failure_before_replace_leaves_old_bytes(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        target = tmp_path / "models.json"
        target.write_bytes(b"old")
        payload = b"new"

        def fail_replace(*_args: object) -> None:
            raise OSError("replace failed before visibility commit")

        monkeypatch.setattr(model_config_io.os, "replace", fail_replace)

        with pytest.raises(OSError, match="before visibility"):
            atomic_replace_bytes(
                target,
                payload,
                intended_fingerprint=file_fingerprint(payload),
            )

        assert target.read_bytes() == b"old"

    def test_post_replace_directory_fsync_failure_reconciles_intended_bytes(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        target = tmp_path / "models.json"
        target.write_bytes(b"old")
        payload = b"new"

        def fail_directory_fsync(path: Path) -> None:
            raise OSError("directory fsync failed after replace")

        monkeypatch.setattr(model_config_io, "_fsync_directory", fail_directory_fsync)

        atomic_replace_bytes(
            target,
            payload,
            intended_fingerprint=file_fingerprint(payload),
        )

        assert target.read_bytes() == payload

    def test_post_replace_unreconciled_bytes_raise_outcome_unknown(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        target = tmp_path / "models.json"
        target.write_bytes(b"old")
        payload = b"new"

        def corrupt_then_fail_directory_fsync(path: Path) -> None:
            target.write_bytes(b"interloper")
            raise OSError("directory fsync failed after replace")

        monkeypatch.setattr(
            model_config_io,
            "_fsync_directory",
            corrupt_then_fail_directory_fsync,
        )

        with pytest.raises(CommitOutcomeUnknown, match="cannot confirm"):
            atomic_replace_bytes(
                target,
                payload,
                intended_fingerprint=file_fingerprint(payload),
            )

        assert target.read_bytes() == b"interloper"

    def test_install_creates_only_when_target_does_not_exist(self, tmp_path: Path) -> None:
        target = tmp_path / "models.json"

        assert atomic_install_bytes(target, b"first") == "created"
        assert target.read_bytes() == b"first"

    def test_install_never_overwrites_existing_different_bytes(self, tmp_path: Path) -> None:
        target = tmp_path / "models.json"
        target.write_bytes(b"existing")

        assert atomic_install_bytes(target, b"new") == "exists"
        assert target.read_bytes() == b"existing"

    def test_concurrent_identical_installs_converge_without_overwrite(
        self, tmp_path: Path
    ) -> None:
        target = tmp_path / "models.json"
        payload = b"same config"
        context = multiprocessing.get_context("spawn")
        outcomes: multiprocessing.Queue[str] = context.Queue()
        first = context.Process(
            target=_install_config_worker,
            args=(str(target), payload, outcomes),
        )
        second = context.Process(
            target=_install_config_worker,
            args=(str(target), payload, outcomes),
        )

        first.start()
        second.start()
        first.join(timeout=10)
        second.join(timeout=10)

        assert first.exitcode == 0
        assert second.exitcode == 0
        assert sorted(outcomes.get(timeout=2) for _ in range(2)) == ["created", "exists"]
        assert target.read_bytes() == payload

    def test_two_processes_serialize_on_adjacent_config_lock(self, tmp_path: Path) -> None:
        target = tmp_path / "models.json"
        context = multiprocessing.get_context("spawn")
        events: multiprocessing.Queue[tuple[str, float]] = context.Queue()
        first = context.Process(target=_contending_lock_worker, args=(str(target), events))
        second = context.Process(target=_contending_lock_worker, args=(str(target), events))

        first.start()
        second.start()
        first.join(timeout=10)
        second.join(timeout=10)

        assert first.exitcode == 0
        assert second.exitcode == 0
        acquired = sorted(events.get(timeout=2)[1] for _ in range(2))
        assert acquired[1] - acquired[0] >= 0.18
