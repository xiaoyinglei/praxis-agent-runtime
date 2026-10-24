from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
import zipfile
from pathlib import Path

import yaml

from agent_runtime.core.llm_registry import ModelRegistry


def test_built_wheel_loads_complete_public_catalog_outside_repo(
    tmp_path: Path,
) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    dist_dir = tmp_path / "dist"
    dist_dir.mkdir()
    subprocess.run(
        ["uv", "build", "--wheel", "--out-dir", str(dist_dir)],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )
    wheel, = sorted(dist_dir.glob("*.whl"))
    run_dir = tmp_path / "outside-repo"
    run_dir.mkdir()
    public_config = yaml.safe_load((repo_root / "configs/agent-models.yaml").read_text(encoding="utf-8"))
    snapshot = yaml.safe_load((repo_root / "configs/model-catalog.yaml").read_text(encoding="utf-8"))
    expected_runtime_config = ModelRegistry._load_yaml_file(
        repo_root / "configs/agent-models.yaml"
    ).model_dump(mode="json")
    decoy_yaml = textwrap.dedent(
        """
        models:
          decoy/model:
            capability: chat
            provider: decoy
            protocol: openai_compatible
        defaults:
          primary_model: decoy/model
        """
    )
    decoy_dir = run_dir / "configs"
    decoy_dir.mkdir()
    for name in ("models.yaml", "agent-models.yaml"):
        (decoy_dir / name).write_text(decoy_yaml, encoding="utf-8")
    env = os.environ.copy()
    env.pop("RAG_AGENT_MODELS", None)
    env.pop("RAG_AGENT_MODELS_PATH", None)
    env["PRAXIS_MODEL_REGISTRY_PATH"] = str(
        (tmp_path / "isolated-user-config" / "models.yaml").resolve()
    )
    env["PYTHONPATH"] = str(wheel)
    script = textwrap.dedent(
        """
        import asyncio
        import json
        from importlib.resources import files
        from pathlib import Path

        import yaml

        import agent_runtime
        import agent_runtime.core.llm_registry as registry_module
        from agent_runtime import Agent
        from agent_runtime.tools.builtins.data import create_inspect_data_file_tool
        from agent_runtime.workspace import open_workspace

        Path("sample.csv").write_text("name,value\\nA,42\\n")
        inspection = asyncio.run(create_inspect_data_file_tool(open_workspace(Path.cwd())).run({
            "path": "sample.csv",
        }))
        assert inspection.valid and inspection.tables[0].rows == [["name", "value"], ["A", "42"]]

        resources = files("agent_runtime").joinpath("_data")
        public = yaml.safe_load(resources.joinpath("models.yaml").read_text())
        snapshot = yaml.safe_load(resources.joinpath("model-catalog.yaml").read_text())
        supported = {
            model_id for model_id, entry in snapshot["models"].items()
            if entry["integration"] == "chat_completions"
        }
        assert set(public["models"]) == supported
        agent = Agent(model_session_path=None)
        assert {spec.id for spec in agent.models()} == supported
        selected = {
            model_id: Agent(model=model_id, model_session_path=None).current_model().id
            for model_id in sorted(supported)
        }
        rejected = []
        for model_id, entry in snapshot["models"].items():
            if entry["integration"] != "catalog_only":
                continue
            try:
                Agent(model=model_id, model_session_path=None).current_model()
            except registry_module.UnknownModelIdError:
                rejected.append(model_id)
            else:
                raise AssertionError(f"Catalog-only model became runtime-supported: {model_id}")
        print(json.dumps({
            "agent_runtime_file": agent_runtime.__file__,
            "registry_file": registry_module.__file__,
            "public": public,
            "snapshot": snapshot,
            "runtime_config": registry_module.ModelRegistry._load_config().model_dump(mode="json"),
            "selected": selected,
            "default_model": agent.current_model().id,
            "rejected": sorted(rejected),
        }))
        """
    )

    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=run_dir,
        env=env,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stdout + completed.stderr
    payload = json.loads(completed.stdout.strip())
    assert ".whl" in payload["agent_runtime_file"]
    assert ".whl" in payload["registry_file"]
    expected_catalog = {
        "public": public_config,
        "snapshot": snapshot,
        "runtime_config": expected_runtime_config,
        "selected": {model_id: model_id for model_id in public_config["models"]},
        "default_model": public_config["defaults"]["primary_model"],
        "rejected": sorted(
            model_id for model_id, entry in snapshot["models"].items()
            if entry["integration"] == "catalog_only"
        ),
    }
    assert {key: payload[key] for key in expected_catalog} == expected_catalog

    unpacked = tmp_path / "site-packages"
    with zipfile.ZipFile(wheel) as archive:
        archive.extractall(unpacked)
    decoy_dir = unpacked / "configs"
    decoy_dir.mkdir()
    for name in ("models.yaml", "agent-models.yaml"):
        (decoy_dir / name).write_text(decoy_yaml, encoding="utf-8")
    env["PYTHONPATH"] = str(unpacked)

    installed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=run_dir,
        env=env,
        capture_output=True,
        text=True,
    )

    assert installed.returncode == 0, installed.stdout + installed.stderr
    installed_payload = json.loads(installed.stdout.strip())
    assert installed_payload["agent_runtime_file"].startswith(str(unpacked))
    assert installed_payload["registry_file"].startswith(str(unpacked))
    assert {key: installed_payload[key] for key in expected_catalog} == expected_catalog
