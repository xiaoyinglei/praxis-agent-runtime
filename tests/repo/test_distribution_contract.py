from __future__ import annotations

import configparser
import re
import subprocess
import tomllib
import zipfile
from pathlib import Path

import yaml

ROOT = Path(__file__).parents[2]
CI_WORKFLOW = ROOT / ".github" / "workflows" / "ci.yml"
PYPROJECT = ROOT / "pyproject.toml"


def _build_wheel(output_dir: Path) -> Path:
    subprocess.run(
        ["uv", "build", "--wheel", "--out-dir", str(output_dir)],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    wheels = sorted(output_dir.glob("*.whl"))
    assert len(wheels) == 1, f"expected exactly one wheel, found: {wheels}"
    return wheels[0]


def _wheel_members(wheel: Path) -> set[str]:
    with zipfile.ZipFile(wheel) as archive:
        return set(archive.namelist())


def _entry_point(wheel: Path, command: str) -> str:
    with zipfile.ZipFile(wheel) as archive:
        metadata_files = [
            name for name in archive.namelist() if name.endswith(".dist-info/entry_points.txt")
        ]
        assert len(metadata_files) == 1, f"expected one entry_points.txt, found: {metadata_files}"
        parser = configparser.ConfigParser(interpolation=None)
        parser.read_string(archive.read(metadata_files[0]).decode("utf-8"))
    return parser["console_scripts"][command]


def test_wheel_contains_runtime_and_excludes_legacy_agent(tmp_path: Path) -> None:
    wheel = _build_wheel(tmp_path)
    names = _wheel_members(wheel)
    legacy_agent_prefix = "rag/" + "agent/"

    assert "agent_runtime/cli.py" in names
    assert not any(name.startswith(legacy_agent_prefix) for name in names)
    assert _entry_point(wheel, "agent") == "agent_runtime.cli:agent_app"


def test_ci_keeps_native_linux_and_macos_checks_required() -> None:
    config = yaml.safe_load(CI_WORKFLOW.read_text(encoding="utf-8"))
    jobs = config["jobs"]

    assert {"linux", "macos"} <= jobs.keys(), "CI must exercise both supported OS backends"
    assert jobs["linux"]["runs-on"] == "ubuntu-24.04"
    assert jobs["macos"]["runs-on"] == "macos-14"
    for name in ("linux", "macos"):
        job = jobs[name]
        assert not job.get("continue-on-error", False)
        assert "if" not in job, f"{name} must run for every PR, including shared protocol changes"
        assert all(not step.get("continue-on-error", False) for step in job["steps"])


def test_macos_ci_runs_real_sandbox_command_file_and_terminal_regressions() -> None:
    config = yaml.safe_load(CI_WORKFLOW.read_text(encoding="utf-8"))
    job = config["jobs"].get("macos")
    assert job is not None, "macOS must have a native regression job"
    workflow = "\n".join(step.get("run", "") for step in job["steps"])
    commands = re.sub(r"\s+", " ", workflow.replace("\\\n", " "))

    assert "test -x /usr/bin/sandbox-exec" in commands
    assert "/usr/bin/sandbox-exec -p" in commands
    assert "uv sync --locked" in commands
    assert "--all-extras" not in commands, "native regressions must not duplicate the heavy RAG/MLX install"
    assert "uv run pytest -q" in commands
    for path in (
        "tests/agent/test_run_command_safety.py",
        "tests/agent/test_command_sessions.py",
        "tests/agent/test_command_recovery.py",
        "tests/agent/harness/test_macos_command_sessions.py",
        "tests/agent/harness/test_real_sandbox_path.py",
        "tests/agent/test_permission_sandbox.py",
        "tests/agent/test_permission_integration.py",
        "tests/agent/test_file_manifest.py",
        "tests/agent/test_local_files.py",
        "tests/agent/test_cli_chat_commands.py",
        "tests/agent/harness/test_public_agent_cutover.py",
        "tests/agent/test_terminal_input.py",
        "tests/agent/test_terminal_render.py",
        "tests/agent/test_terminal_ui.py",
        "tests/agent/test_terminal_ui_modes.py",
        "tests/agent/test_terminal_approval.py",
        "tests/agent/test_terminal_steering.py",
        "tests/agent/test_terminal_connections.py",
    ):
        assert path in commands, f"missing native macOS regression: {path}"


def test_linux_ci_runs_full_gates_then_smokes_the_installed_wheel() -> None:
    config = yaml.safe_load(CI_WORKFLOW.read_text(encoding="utf-8"))
    # Inspect executable step bodies rather than YAML formatting. Shell line
    # continuations and indentation do not change the installation contract.
    workflow = "\n".join(step.get("run", "") for step in config["jobs"]["linux"]["steps"])
    commands = re.sub(r"\s+", " ", workflow.replace("\\\n", " "))
    ordered_commands = [
        "uv sync --locked --all-extras",
        "uv run --all-extras ruff check .",
        "uv run --all-extras mypy",
        "uv run --all-extras pytest -q",
        "uv run --all-extras python scripts/agent_web_quality.py",
        "uv run --all-extras playwright install --with-deps --only-shell chromium",
        "PRAXIS_TEST_BROWSER=1 uv run --all-extras pytest -q tests/agent/test_web_browser.py "
        "tests/agent/test_connection_manager_browser.py",
        "sudo /usr/bin/python3 -I scripts/ubuntu/smoke-model-service.py --wheel-dir",
        "uv run --all-extras lint-imports",
        "uv build",
        "uv run --all-extras python scripts/agent_cli_smoke.py",
        "uv run --all-extras python scripts/agent_delivery_smoke.py --fake-model --verbose",
        "uv run --all-extras python scripts/agent_public_path_benchmark.py validate",
        "uv run --all-extras python scripts/agent_model_quality_gate.py validate",
        "uv export --locked --all-extras --no-dev --no-hashes --no-emit-project",
        'uv venv --no-project --python 3.12 "$smoke_venv"',
        'uv pip install --python "$smoke_venv/bin/python" --constraint "$smoke_root/constraints.txt" "$wheel"',
        'cd "$smoke_root"',
        '"$smoke_venv/bin/agent" --help',
        '"$smoke_venv/bin/agent" run --help',
        '"$smoke_venv/bin/python" "$GITHUB_WORKSPACE/scripts/agent_delivery_smoke.py" --fake-model',
        'uv pip install --python "$smoke_venv/bin/python" --constraint "$smoke_root/constraints.txt" "${wheel}[rag]"',
        '"$smoke_venv/bin/rag" --help',
    ]

    cursor = -1
    for command in ordered_commands:
        cursor = commands.find(command, cursor + 1)
        assert cursor >= 0, f"missing or out-of-order CI command: {command}"

    assert "sudo apt-get install -y bubblewrap apparmor libseccomp2" in commands
    assert "/usr/bin/bwrap --unshare-user --unshare-pid --ro-bind / / /bin/true" in commands
    assert "python3 scripts/ubuntu/install-codex-patch.py --destination" in commands
    assert 'mktemp -d "$RUNNER_TEMP/praxis-wheel-smoke.XXXXXX"' in workflow
    assert "mapfile -d '' -t wheels" in workflow
    assert '"${#wheels[@]}" -ne 1' in workflow


def test_mypy_allows_the_platform_only_mlx_dependency_to_be_absent() -> None:
    config = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))
    ignored_modules = {
        module
        for override in config["tool"]["mypy"]["overrides"]
        if override.get("ignore_missing_imports") is True
        for module in override["module"]
    }

    assert {"mlx", "mlx.*"} <= ignored_modules


def test_distribution_has_no_langgraph_runtime_dependency() -> None:
    config = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))
    dependencies = tuple(config["project"]["dependencies"])
    lock = (ROOT / "uv.lock").read_text(encoding="utf-8")

    assert all("langgraph" not in dependency.lower() for dependency in dependencies)
    assert 'name = "langgraph"' not in lock
    assert 'name = "langgraph-checkpoint"' not in lock
