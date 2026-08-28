from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from agent_runtime.runtime.mcp import (
    decide_mcp_config_trust,
    open_trusted_product_mcp_tools,
)


def test_workspace_mcp_config_requires_trust_before_server_start(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    config = workspace / "configs" / "mcp_servers.yaml"
    config.parent.mkdir(parents=True)
    config.write_text("servers: []\n", encoding="utf-8")

    with pytest.raises(PermissionError, match="workspace MCP config is not trusted"):
        decide_mcp_config_trust(
            config,
            workspace_root=workspace,
            trust_workspace=False,
        )


def test_mcp_config_drift_fails_before_stdio_start(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = tmp_path / "workspace"
    config = workspace / "configs" / "mcp_servers.yaml"
    config.parent.mkdir(parents=True)
    config.write_text("servers: []\n", encoding="utf-8")
    decision = decide_mcp_config_trust(
        config,
        workspace_root=workspace,
        trust_workspace=True,
    )
    config.write_text("servers:\n  - enabled: true\n", encoding="utf-8")
    starts: list[str] = []

    def forbidden_start(*_args: object, **_kwargs: object) -> object:
        starts.append("started")
        raise AssertionError("MCP stdio must not start after trust drift")

    monkeypatch.setattr("agent_runtime.runtime.mcp.stdio_client", forbidden_start)

    async def open_runtime() -> None:
        async with open_trusted_product_mcp_tools(
            config,
            trust=decision,
        ):
            raise AssertionError("drifted MCP config must not open")

    with pytest.raises(RuntimeError, match="changed after trust decision"):
        asyncio.run(open_runtime())
    assert starts == []
