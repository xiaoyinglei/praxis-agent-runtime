"""Tests for WorkspaceRuntime, factory functions, and file import."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from agent_runtime.workspace import (
    WorkspacePathError,
    WorkspaceRuntime,
    create_temp_workspace,
    import_files,
    open_workspace,
    workspace_tree_sha256,
)

# ---------------------------------------------------------------------------
# WorkspacePathError
# ---------------------------------------------------------------------------


class TestWorkspacePathError:
    def test_is_value_error_subclass(self) -> None:
        assert issubclass(WorkspacePathError, ValueError)

    def test_can_be_raised_and_caught(self) -> None:
        with pytest.raises(WorkspacePathError):
            raise WorkspacePathError("bad path")


# ---------------------------------------------------------------------------
# WorkspaceRuntime.initialize
# ---------------------------------------------------------------------------


class TestWorkspaceInitialize:
    def test_creates_all_subdirs(self, tmp_path: Path) -> None:
        ws = WorkspaceRuntime(root=tmp_path, is_temporary=False)
        ws.initialize()
        for name in ("input_files", "scratch", "artifacts", "reports", "logs"):
            assert (ws.runtime_root / name).is_dir()

    def test_is_idempotent(self, tmp_path: Path) -> None:
        ws = WorkspaceRuntime(root=tmp_path, is_temporary=False)
        ws.initialize()
        # Call again — should not raise
        ws.initialize()
        assert ws.scratch.is_dir()


# ---------------------------------------------------------------------------
# WorkspaceRuntime properties
# ---------------------------------------------------------------------------


class TestWorkspaceProperties:
    def test_input_files(self, tmp_path: Path) -> None:
        ws = WorkspaceRuntime(root=tmp_path, is_temporary=False)
        assert ws.input_files == tmp_path / ".rag" / "agent_runtime" / "input_files"

    def test_scratch(self, tmp_path: Path) -> None:
        ws = WorkspaceRuntime(root=tmp_path, is_temporary=False)
        assert ws.scratch == tmp_path / ".rag" / "agent_runtime" / "scratch"

    def test_artifacts(self, tmp_path: Path) -> None:
        ws = WorkspaceRuntime(root=tmp_path, is_temporary=False)
        assert ws.artifacts == tmp_path / ".rag" / "agent_runtime" / "artifacts"

    def test_reports(self, tmp_path: Path) -> None:
        ws = WorkspaceRuntime(root=tmp_path, is_temporary=False)
        assert ws.reports == tmp_path / ".rag" / "agent_runtime" / "reports"

    def test_logs(self, tmp_path: Path) -> None:
        ws = WorkspaceRuntime(root=tmp_path, is_temporary=False)
        assert ws.logs == tmp_path / ".rag" / "agent_runtime" / "logs"


# ---------------------------------------------------------------------------
# resolve_path
# ---------------------------------------------------------------------------


class TestResolvePath:
    def test_returns_absolute(self, tmp_path: Path) -> None:
        ws = WorkspaceRuntime(root=tmp_path, is_temporary=False)
        result = ws.resolve_path("input_files/test.txt")
        assert result.is_absolute()
        assert result == (tmp_path / "input_files" / "test.txt").resolve()

    def test_accepts_path_object(self, tmp_path: Path) -> None:
        ws = WorkspaceRuntime(root=tmp_path, is_temporary=False)
        result = ws.resolve_path(Path("scratch/data.csv"))
        assert result.is_absolute()


# ---------------------------------------------------------------------------
# ensure_within_workspace
# ---------------------------------------------------------------------------


class TestEnsureWithinWorkspace:
    def test_rejects_dotdot_escape(self, tmp_path: Path) -> None:
        ws = WorkspaceRuntime(root=tmp_path, is_temporary=False)
        bad = tmp_path / "scratch" / ".." / ".." / "secret.txt"
        with pytest.raises(WorkspacePathError, match="escapes workspace boundary"):
            ws.ensure_within_workspace(bad)

    def test_rejects_absolute_outside(self, tmp_path: Path) -> None:
        ws = WorkspaceRuntime(root=tmp_path, is_temporary=False)
        outside = Path("/tmp/definitely_not_in_workspace.txt")
        with pytest.raises(WorkspacePathError, match="escapes workspace boundary"):
            ws.ensure_within_workspace(outside)

    def test_accepts_valid_path(self, tmp_path: Path) -> None:
        ws = WorkspaceRuntime(root=tmp_path, is_temporary=False)
        valid = tmp_path / "scratch" / "file.txt"
        result = ws.ensure_within_workspace(valid)
        assert result == valid.resolve()

    def test_accepts_root_itself(self, tmp_path: Path) -> None:
        ws = WorkspaceRuntime(root=tmp_path, is_temporary=False)
        result = ws.ensure_within_workspace(tmp_path)
        assert result == tmp_path.resolve()


# ---------------------------------------------------------------------------
# ensure_within_scratch
# ---------------------------------------------------------------------------


class TestEnsureWithinScratch:
    def test_rejects_non_scratch_path(self, tmp_path: Path) -> None:
        ws = WorkspaceRuntime(root=tmp_path, is_temporary=False)
        ws.initialize()
        artifacts_file = tmp_path / "artifacts" / "output.txt"
        with pytest.raises(WorkspacePathError, match="not within scratch"):
            ws.ensure_within_scratch(artifacts_file)

    def test_accepts_scratch_path(self, tmp_path: Path) -> None:
        ws = WorkspaceRuntime(root=tmp_path, is_temporary=False)
        ws.initialize()
        scratch_file = ws.scratch / "working.txt"
        result = ws.ensure_within_scratch(scratch_file)
        assert result == scratch_file.resolve()


# ---------------------------------------------------------------------------
# relative_to_root
# ---------------------------------------------------------------------------


class TestRelativeToRoot:
    def test_returns_relative_path(self, tmp_path: Path) -> None:
        ws = WorkspaceRuntime(root=tmp_path, is_temporary=False)
        ws.initialize()
        full = tmp_path / "artifacts" / "result.json"
        rel = ws.relative_to_root(full)
        assert rel == Path("artifacts/result.json")

    def test_rejects_escape(self, tmp_path: Path) -> None:
        ws = WorkspaceRuntime(root=tmp_path, is_temporary=False)
        bad = tmp_path / ".." / "outside.txt"
        with pytest.raises(WorkspacePathError):
            ws.relative_to_root(bad)


# ---------------------------------------------------------------------------
# create_temp_workspace
# ---------------------------------------------------------------------------


class TestCreateTempWorkspace:
    def test_root_is_canonical(self) -> None:
        ws = create_temp_workspace()
        try:
            assert ws.root == ws.root.resolve()
        finally:
            import shutil

            shutil.rmtree(ws.root, ignore_errors=True)

    def test_creates_temp_workspace_with_subdirs(self) -> None:
        ws = create_temp_workspace()
        try:
            assert ws.is_temporary is True
            assert ws.root.is_dir()
            for name in ("input_files", "scratch", "artifacts", "reports", "logs"):
                assert (ws.root / name).is_dir()
        finally:
            # Cleanup
            import shutil

            shutil.rmtree(ws.root, ignore_errors=True)

    def test_prefix_applied(self) -> None:
        ws = create_temp_workspace(prefix="test_ws_")
        try:
            assert ws.root.name.startswith("test_ws_")
        finally:
            import shutil

            shutil.rmtree(ws.root, ignore_errors=True)


# ---------------------------------------------------------------------------
# open_workspace
# ---------------------------------------------------------------------------


class TestOpenWorkspace:
    def test_opens_existing(self, tmp_path: Path) -> None:
        ws = open_workspace(tmp_path)
        assert ws.root == tmp_path.resolve()
        assert ws.is_temporary is False
        assert ws.scratch.is_dir()

    def test_creates_when_create_true(self, tmp_path: Path) -> None:
        target = tmp_path / "new_workspace"
        ws = open_workspace(target, create=True)
        assert ws.root.is_dir()
        assert ws.artifacts.is_dir()

    def test_raises_when_path_missing_and_create_false(self, tmp_path: Path) -> None:
        target = tmp_path / "nonexistent"
        with pytest.raises(FileNotFoundError):
            open_workspace(target, create=False)

    def test_accepts_string_path(self, tmp_path: Path) -> None:
        ws = open_workspace(str(tmp_path))
        assert ws.root == tmp_path.resolve()


# ---------------------------------------------------------------------------
# workspace_tree_sha256
# ---------------------------------------------------------------------------


class TestWorkspaceTreeSha256:
    def test_temporary_runtime_writes_do_not_count_as_user_changes(self) -> None:
        workspace = create_temp_workspace()
        try:
            baseline = workspace_tree_sha256(workspace.root)

            (workspace.scratch / "skill.py").write_text(
                "runtime helper\n",
                encoding="utf-8",
            )
            (workspace.logs / "runtime.log").write_text(
                "runtime\n",
                encoding="utf-8",
            )

            assert workspace_tree_sha256(workspace.root) == baseline
        finally:
            shutil.rmtree(workspace.root, ignore_errors=True)

    def test_non_git_snapshot_counts_source_but_not_runtime_state(
        self,
        tmp_path: Path,
    ) -> None:
        workspace = open_workspace(tmp_path)
        baseline = workspace_tree_sha256(workspace.root)

        (workspace.logs / "runtime.log").write_text("runtime\n", encoding="utf-8")
        assert workspace_tree_sha256(workspace.root) == baseline

        (workspace.root / "source.py").write_text("value = 1\n", encoding="utf-8")
        assert workspace_tree_sha256(workspace.root) != baseline

    def test_git_snapshot_tracks_dirty_and_untracked_source_state(
        self,
        tmp_path: Path,
    ) -> None:
        git = shutil.which("git")
        if git is None:
            pytest.skip("git is unavailable")
        repository = tmp_path / "repository"
        repository.mkdir()
        subprocess.run(
            (git, "init", "--quiet"),
            cwd=repository,
            check=True,
        )
        (repository / ".gitignore").write_text(".cache/\n", encoding="utf-8")
        source = repository / "source.py"
        source.write_text("value = 1\n", encoding="utf-8")
        subprocess.run(
            (git, "add", ".gitignore", "source.py"),
            cwd=repository,
            check=True,
        )
        subprocess.run(
            (
                git,
                "-c",
                "user.name=Test",
                "-c",
                "user.email=test@example.invalid",
                "commit",
                "--quiet",
                "-m",
                "initial",
            ),
            cwd=repository,
            check=True,
        )
        baseline = workspace_tree_sha256(repository)

        source.write_text("value = 2\n", encoding="utf-8")
        assert workspace_tree_sha256(repository) != baseline
        source.write_text("value = 1\n", encoding="utf-8")
        assert workspace_tree_sha256(repository) == baseline

        untracked = repository / "new_source.py"
        untracked.write_text("new = True\n", encoding="utf-8")
        assert workspace_tree_sha256(repository) != baseline
        untracked.unlink()

        ignored = repository / ".cache" / "runtime.bin"
        ignored.parent.mkdir()
        ignored.write_bytes(b"runtime")
        assert workspace_tree_sha256(repository) == baseline

        runtime_log = repository / ".rag" / "agent_runtime" / "logs" / "run.log"
        runtime_log.parent.mkdir(parents=True)
        runtime_log.write_text("runtime\n", encoding="utf-8")
        assert workspace_tree_sha256(repository) == baseline

    def test_snapshot_ignores_git_index_and_head_only_changes(
        self,
        tmp_path: Path,
    ) -> None:
        git = shutil.which("git")
        if git is None:
            pytest.skip("git is unavailable")
        repository = tmp_path / "repository"
        repository.mkdir()
        subprocess.run((git, "init", "--quiet"), cwd=repository, check=True)
        source = repository / "source.py"
        source.write_text("value = 1\n", encoding="utf-8")
        subprocess.run((git, "add", "source.py"), cwd=repository, check=True)
        subprocess.run(
            (
                git,
                "-c",
                "user.name=Test",
                "-c",
                "user.email=test@example.invalid",
                "commit",
                "--quiet",
                "-m",
                "initial",
            ),
            cwd=repository,
            check=True,
        )

        source.write_text("value = 2\n", encoding="utf-8")
        changed_filesystem = workspace_tree_sha256(repository)
        subprocess.run((git, "add", "source.py"), cwd=repository, check=True)
        assert workspace_tree_sha256(repository) == changed_filesystem

        subprocess.run(
            (
                git,
                "-c",
                "user.name=Test",
                "-c",
                "user.email=test@example.invalid",
                "commit",
                "--quiet",
                "-m",
                "metadata only",
            ),
            cwd=repository,
            check=True,
        )
        assert workspace_tree_sha256(repository) == changed_filesystem


# ---------------------------------------------------------------------------
# import_files
# ---------------------------------------------------------------------------


class TestImportFiles:
    def test_reuses_file_already_in_workspace(self, tmp_path: Path) -> None:
        ws = open_workspace(tmp_path)
        src = tmp_path / "upload.txt"
        src.write_text("hello")
        results = import_files(ws, [src])
        assert len(results) == 1
        assert results == [src.resolve()]

    def test_stages_external_file_in_namespaced_runtime_storage(
        self,
        tmp_path: Path,
    ) -> None:
        workspace_root = tmp_path / "workspace"
        source_root = tmp_path / "source"
        source_root.mkdir()
        ws = open_workspace(workspace_root, create=True)
        src = source_root / "data.csv"
        src.write_text("a,b")
        first = import_files(ws, [src], namespace="turn-1")
        second = import_files(ws, [src], namespace="turn-1")
        assert first[0].parent == ws.input_files / "turn-1"
        assert first[0].name == "data.csv"
        assert second[0].name == "data__1.csv"
        assert first[0] != second[0]

    def test_rejects_directory(self, tmp_path: Path) -> None:
        ws = open_workspace(tmp_path)
        with pytest.raises(ValueError, match="Directory import not supported"):
            import_files(ws, [tmp_path])

    def test_rejects_missing_file(self, tmp_path: Path) -> None:
        ws = open_workspace(tmp_path)
        with pytest.raises(FileNotFoundError, match="Source file not found"):
            import_files(ws, [tmp_path / "nope.txt"])

    def test_returns_multiple_paths(self, tmp_path: Path) -> None:
        ws = open_workspace(tmp_path)
        a = tmp_path / "a.txt"
        b = tmp_path / "b.txt"
        a.write_text("A")
        b.write_text("B")
        results = import_files(ws, [a, b])
        assert len(results) == 2
        assert {r.name for r in results} == {"a.txt", "b.txt"}

    def test_preserves_file_content(self, tmp_path: Path) -> None:
        ws = open_workspace(tmp_path)
        src = tmp_path / "payload.bin"
        content = bytes(range(256))
        src.write_bytes(content)
        results = import_files(ws, [src])
        assert results[0].read_bytes() == content


# ---------------------------------------------------------------------------
# _unique_dest helper
# ---------------------------------------------------------------------------


class TestUniqueDest:
    def test_no_collision(self, tmp_path: Path) -> None:
        from agent_runtime.workspace import _unique_dest

        dest = _unique_dest(tmp_path, "file.txt")
        assert dest == tmp_path / "file.txt"

    def test_collision_appends_counter(self, tmp_path: Path) -> None:
        from agent_runtime.workspace import _unique_dest

        (tmp_path / "file.txt").write_text("existing")
        dest = _unique_dest(tmp_path, "file.txt")
        assert dest.name == "file__1.txt"

    def test_multiple_collisions(self, tmp_path: Path) -> None:
        from agent_runtime.workspace import _unique_dest

        (tmp_path / "file.txt").write_text("1")
        (tmp_path / "file__1.txt").write_text("2")
        dest = _unique_dest(tmp_path, "file.txt")
        assert dest.name == "file__2.txt"
