from __future__ import annotations

import subprocess
import sys


def _run_import_probe(code: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-c", code],
        check=False,
        cwd=".",
        text=True,
        capture_output=True,
    )


def test_agent_runtime_import_does_not_load_rag_provider_or_embedding_stack() -> None:
    result = _run_import_probe(
        """
import sys
import agent_runtime
forbidden = [
    "agent_runtime.knowledge_providers.rag",
    "sentence_transformers",
    "sklearn",
]
loaded = [name for name in forbidden if name in sys.modules]
print("\\n".join(loaded))
raise SystemExit(1 if loaded else 0)
""".strip()
    )

    assert result.returncode == 0, result.stdout + result.stderr


def test_agent_model_current_uses_light_cli_path() -> None:
    result = _run_import_probe(
        """
import sys
from typer.testing import CliRunner
from agent_runtime.cli import agent_app

result = CliRunner().invoke(agent_app, ["model", "current"], env={"COLUMNS": "240"})
print(result.output)
if result.exit_code != 0:
    raise SystemExit(result.exit_code)

forbidden = [
    "agent_runtime.agent",
    "agent_runtime.knowledge_providers.rag",
    "agent_runtime.service",
    "agent_runtime.builtin_registry",
    "sentence_transformers",
]
loaded = [name for name in forbidden if name in sys.modules]
print("\\n".join(loaded))
raise SystemExit(1 if loaded else 0)
""".strip()
    )

    assert result.returncode == 0, result.stdout + result.stderr
