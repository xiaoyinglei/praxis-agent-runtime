from __future__ import annotations

import configparser
import subprocess
import tomllib
import zipfile
from pathlib import Path

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


def test_ci_runs_full_gates_then_smokes_the_installed_wheel() -> None:
    workflow = CI_WORKFLOW.read_text(encoding="utf-8")
    ordered_commands = [
        "uv sync --locked",
        "uv run ruff check .",
        "uv run mypy",
        "uv run pytest -q",
        "uv run lint-imports",
        "uv build",
        "uv run python scripts/agent_cli_smoke.py",
        "uv run python scripts/agent_delivery_smoke.py --fake-model --verbose",
        (
            "uv run python scripts/agent_harness_acceptance.py validate --schema "
            "evals/harness/acceptance_v1.json "
            "--contract docs/design/praxis_harness_architecture.md"
        ),
        (
            "uv run python scripts/agent_code_benchmark.py validate "
            "evals/code_agent/benchmark_v1.json --repository ."
        ),
        'uv venv --no-project --python 3.12 "$smoke_venv"',
        'uv pip install --python "$smoke_venv/bin/python" "$wheel"',
        '"$smoke_venv/bin/agent" --help',
        '"$smoke_venv/bin/rag" --help',
    ]

    cursor = -1
    for command in ordered_commands:
        cursor = workflow.find(command, cursor + 1)
        assert cursor >= 0, f"missing or out-of-order CI command: {command}"

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
