"""Shared fixtures for agent contract tests."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


@pytest.fixture(autouse=True)
def isolate_user_model_registry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keep every Agent test independent of the developer's real registry."""

    monkeypatch.setenv(
        "PRAXIS_MODEL_REGISTRY_PATH",
        str(
            (
                tmp_path.parent
                / f"{tmp_path.name}-isolated-user-config"
                / "models.yaml"
            ).resolve()
        ),
    )


@pytest.fixture
def fake_sandbox_exec(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> Path:
    """Run process-lifecycle tests without depending on macOS Seatbelt.

    The wrapper only emulates sandbox-exec's argv contract. Tests that assert
    actual filesystem or network isolation must continue to use real Seatbelt.
    """

    from agent_runtime.tools.builtins import shell as shell_module

    executable = tmp_path / "fake-sandbox-exec"
    executable.write_text(
        "#!/bin/sh\n"
        'if [ "$1" != "-p" ] || [ "$#" -lt 3 ]; then\n'
        "  exit 64\n"
        "fi\n"
        "shift 2\n"
        'exec "$@"\n',
        encoding="utf-8",
    )
    executable.chmod(0o755)
    monkeypatch.setattr(
        shell_module,
        "_SANDBOX_EXEC_PATH",
        str(executable),
    )
    create_subprocess_exec = asyncio.create_subprocess_exec

    async def execute_without_seatbelt(
        *argv: str,
        **kwargs: Any,
    ) -> asyncio.subprocess.Process:
        if len(argv) < 5 or argv[0] != str(executable) or argv[1] != "-p":
            raise AssertionError("unexpected sandbox-exec argv")
        return await create_subprocess_exec(*argv[3:], **kwargs)

    monkeypatch.setattr(
        shell_module.asyncio,
        "create_subprocess_exec",
        execute_without_seatbelt,
    )
    return executable
