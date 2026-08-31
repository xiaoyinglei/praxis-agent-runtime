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
    start: multiprocessing.synchronize.Event,
    outcomes: multiprocessing.Queue[tuple[bytes, str]],
) -> None:
    """Race a no-replace install from a separate Python process."""

    if not start.wait(timeout=10):
        raise RuntimeError("concurrent install did not receive start signal")
    outcomes.put((payload, atomic_install_bytes(Path(path), payload)))


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

    def test_raises_for_git_failure_other_than_not_a_repository(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        workspace = tmp_path / "ordinary"
        workspace.mkdir()

        def inaccessible_git(*_args: object, **_kwargs: object) -> subprocess.CompletedProcess[str]:
            return subprocess.CompletedProcess(
                args=["git"],
                returncode=128,
                stdout="",
                stderr="fatal: cannot access current directory: Permission denied",
            )

        monkeypatch.setattr(model_config_io.subprocess, "run", inaccessible_git)

        with pytest.raises(RuntimeError, match="cannot access current directory"):
            discover_git_worktree(workspace)


class TestValidateUserConfigPath:
    def test_allows_absolute_external_path(self, tmp_path: Path) -> None:
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        external = tmp_path / "external" / "models.json"

        assert (
            validate_user_config_path(
                external,
                workspace=workspace,
                worktree=workspace,
            )
            == external.resolve()
        )

    def test_rejects_relative_path(self, tmp_path: Path) -> None:
        workspace = tmp_path / "workspace"
        workspace.mkdir()

        with pytest.raises(UntrustedConfigPathError, match="absolute"):
            validate_user_config_path(
                Path(".praxis/models.json"),
                workspace=workspace,
                worktree=workspace,
            )

    def test_rejects_tilde_path_before_expansion(self, tmp_path: Path) -> None:
        workspace = tmp_path / "workspace"
        workspace.mkdir()

        with pytest.raises(UntrustedConfigPathError, match="absolute"):
            validate_user_config_path(
                Path("~/models.json"),
                workspace=workspace,
                worktree=workspace,
            )

    def test_rejects_direct_workspace_path(self, tmp_path: Path) -> None:
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        direct = workspace / ".praxis" / "models.json"

        with pytest.raises(UntrustedConfigPathError, match="workspace"):
            validate_user_config_path(
                direct,
                workspace=workspace,
                worktree=workspace,
            )

    def test_rejects_path_in_worktree_outside_nested_workspace(
        self, tmp_path: Path
    ) -> None:
        worktree = _make_git_worktree(tmp_path)
        workspace = worktree / "apps" / "praxis"
        workspace.mkdir(parents=True)
        shared = worktree / ".praxis" / "models.json"

        with pytest.raises(UntrustedConfigPathError, match="worktree"):
            validate_user_config_path(
                shared,
                workspace=workspace,
                worktree=worktree,
            )

    def test_rejects_symlink_resolving_into_workspace(self, tmp_path: Path) -> None:
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        target = workspace / ".praxis" / "models.json"
        target.parent.mkdir()
        target.write_bytes(b"{}")
        alias = tmp_path / "workspace-link.json"
        alias.symlink_to(target)

        with pytest.raises(UntrustedConfigPathError, match="workspace"):
            validate_user_config_path(
                alias,
                workspace=workspace,
                worktree=workspace,
            )

    def test_rejects_symlink_resolving_elsewhere_in_worktree(self, tmp_path: Path) -> None:
        worktree = _make_git_worktree(tmp_path)
        workspace = worktree / "apps" / "praxis"
        workspace.mkdir(parents=True)
        target = worktree / ".praxis" / "models.json"
        target.parent.mkdir()
        target.write_bytes(b"{}")
        alias = workspace / "models-link.json"
        alias.symlink_to(target)

        with pytest.raises(UntrustedConfigPathError, match="worktree"):
            validate_user_config_path(
                alias,
                workspace=workspace,
                worktree=worktree,
            )

    def test_allows_symlink_resolving_outside_worktree(self, tmp_path: Path) -> None:
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        external = tmp_path / "external" / "models.json"
        external.parent.mkdir()
        external.write_bytes(b"{}")
        alias = workspace / "models-link.json"
        alias.symlink_to(external)

        assert (
            validate_user_config_path(
                alias,
                workspace=workspace,
                worktree=workspace,
            )
            == external.resolve()
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
        start = context.Event()
        outcomes: multiprocessing.Queue[tuple[bytes, str]] = context.Queue()
        first = context.Process(
            target=_install_config_worker,
            args=(str(target), payload, start, outcomes),
        )
        second = context.Process(
            target=_install_config_worker,
            args=(str(target), payload, start, outcomes),
        )

        first.start()
        second.start()
        start.set()
        first.join(timeout=10)
        second.join(timeout=10)

        assert first.exitcode == 0
        assert second.exitcode == 0
        assert sorted(outcome for _, outcome in (outcomes.get(timeout=2) for _ in range(2))) == [
            "created",
            "exists",
        ]
        assert target.read_bytes() == payload

    def test_concurrent_different_installs_choose_one_complete_payload(
        self, tmp_path: Path
    ) -> None:
        target = tmp_path / "models.json"
        first_payload = b'{"model":"first"}'
        second_payload = b'{"model":"second"}'
        context = multiprocessing.get_context("spawn")
        start = context.Event()
        outcomes: multiprocessing.Queue[tuple[bytes, str]] = context.Queue()
        first = context.Process(
            target=_install_config_worker,
            args=(str(target), first_payload, start, outcomes),
        )
        second = context.Process(
            target=_install_config_worker,
            args=(str(target), second_payload, start, outcomes),
        )

        first.start()
        second.start()
        start.set()
        first.join(timeout=10)
        second.join(timeout=10)

        assert first.exitcode == 0
        assert second.exitcode == 0
        results = [outcomes.get(timeout=2) for _ in range(2)]
        assert sorted(outcome for _, outcome in results) == ["created", "exists"]
        winner = next(payload for payload, outcome in results if outcome == "created")
        assert winner in {first_payload, second_payload}
        assert target.read_bytes() == winner

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
