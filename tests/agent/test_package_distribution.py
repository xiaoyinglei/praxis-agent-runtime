from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
import zipfile
from pathlib import Path


def test_built_wheel_loads_bundled_qwen35_model_outside_repo(
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
    env = os.environ.copy()
    env.pop("RAG_AGENT_MODELS", None)
    env.pop("RAG_AGENT_MODELS_PATH", None)
    env["PRAXIS_MODEL_REGISTRY_PATH"] = str(
        (tmp_path / "isolated-user-config" / "models.yaml").resolve()
    )
    env["PYTHONPATH"] = str(wheel)
    script = textwrap.dedent(
        """
        import json

        import agent_runtime
        import agent_runtime.core.llm_registry as registry_module
        from agent_runtime import Agent

        spec = Agent(model="qwen3_5_9b_mlx_4bit").current_model()
        print(json.dumps({
            "agent_runtime_file": agent_runtime.__file__,
            "registry_file": registry_module.__file__,
            "model": spec.provider_model,
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
    assert payload["model"] == "mlx-community/Qwen3.5-9B-4bit"

    unpacked = tmp_path / "site-packages"
    with zipfile.ZipFile(wheel) as archive:
        archive.extractall(unpacked)
    decoy_config = unpacked / "configs" / "models.yaml"
    decoy_config.parent.mkdir()
    decoy_config.write_text(
        textwrap.dedent(
            """
            models:
              decoy:
                capability: chat
                provider: decoy
                protocol: openai_compatible
                model: decoy/model
            defaults:
              primary_model: decoy
            """
        ),
        encoding="utf-8",
    )
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
    assert installed_payload["model"] == "mlx-community/Qwen3.5-9B-4bit"
