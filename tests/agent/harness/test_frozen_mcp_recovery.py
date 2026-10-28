from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from pathlib import Path

import pytest
from typer.testing import CliRunner

from agent_runtime import Agent, cli
from agent_runtime.harness import RuntimeState
from agent_runtime.permission_policy import PermissionPolicy, ToolRule
from agent_runtime.runtime import mcp
from tests.agent.harness.test_public_agent_cutover import PatchThenAnswerModel


@pytest.mark.parametrize("entrypoint", ["sdk", "cli"])
def test_frozen_disabled_mcp_recovers_without_trusting_new_workspace_config(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    entrypoint: str,
) -> None:
    target = tmp_path / "value.txt"
    target.write_text("before", encoding="utf-8")
    database = tmp_path / "rollout.sqlite3"
    original = Agent(workspace_path=tmp_path, checkpoint_db=database, enable_workspace_mcp=False)
    monkeypatch.setattr(original, "_model_factory", PatchThenAnswerModel)
    paused = asyncio.run(original.run("change value.txt", require_workspace_change=False))
    assert paused.status == "paused"
    config = tmp_path / "untrusted-mcp.yaml"
    config.write_text("servers: []\n", encoding="utf-8")
    monkeypatch.setenv("AGENT_MCP_CONFIG", str(config))

    if entrypoint == "sdk":
        restarted = Agent(workspace_path=tmp_path, checkpoint_db=database)
        monkeypatch.setattr(restarted, "_model_factory", PatchThenAnswerModel)
        result = asyncio.run(restarted.resume(paused.turn_id, "allow_once"))
        assert result.status == "done"
    else:
        create_facade = cli._create_agent_facade

        def create_real_agent(**options: object) -> Agent:
            agent = create_facade(**options)
            monkeypatch.setattr(agent, "_model_factory", PatchThenAnswerModel)
            return agent

        monkeypatch.setattr(cli, "_create_agent_facade", create_real_agent)
        result = CliRunner().invoke(cli.agent_app, [
            "recover", paused.turn_id, "--rollout-db", str(database), "--action", "allow_once",
        ])
        assert result.exit_code == 0, result.output or repr(result.exception)

    assert target.read_text(encoding="utf-8") == "after"
    with RuntimeState.open_sync(database) as store:
        assert store.read_turn(paused.turn_id).status == "completed"
        assert store.verify().valid


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("original_enabled", "current_enabled", "original_denied", "current_denied", "expected_discoveries"),
    [
        (False, True, False, False, 0),
        (True, False, False, False, 0),
        (True, True, True, False, 0),
        (True, True, False, True, 0),
        (True, True, False, False, 1),
    ],
)
async def test_public_resume_keeps_frozen_mcp_discovery_ceiling(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    original_enabled: bool,
    current_enabled: bool,
    original_denied: bool,
    current_denied: bool,
    expected_discoveries: int,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    target = workspace / "value.txt"
    target.write_text("before", encoding="utf-8")
    database = tmp_path / "rollout.sqlite3"

    def policy(denied: bool) -> PermissionPolicy:
        return PermissionPolicy(tools=(ToolRule(pattern="mcp__*", decision="deny"),) if denied else ())

    original_config = workspace / "original-mcp.yaml"
    original_config.write_text("servers: []\n", encoding="utf-8")
    monkeypatch.setenv("AGENT_MCP_CONFIG", str(original_config))
    original = Agent(
        workspace_path=workspace, checkpoint_db=database,
        enable_workspace_mcp=original_enabled, permissions=policy(original_denied),
        mcp_config_trust=mcp.decide_mcp_config_trust(
            original_config, workspace_root=workspace, trust_workspace=True,
        ),
    )
    monkeypatch.setattr(original, "_model_factory", PatchThenAnswerModel)
    paused = await original.run("change value.txt", require_workspace_change=False)
    assert paused.status == "paused"
    assert target.read_text(encoding="utf-8") == "before"
    async with (await RuntimeState.open(database)) as store:
        frozen_policy = store.read_turn(paused.turn_id).binding_manifest["mcp_policy"]
    assert frozen_policy["workspace_discovery_enabled"] is original_enabled

    # A configuration appears after the Turn pauses. Recovery must decide
    # whether discovery is allowed before touching this new configuration.
    config = workspace / "mcp.yaml"
    config.write_text("servers: []\n", encoding="utf-8")
    monkeypatch.setenv("AGENT_MCP_CONFIG", str(config))
    discoveries: list[Path] = []
    opened: list[Path] = []
    resolve_config = mcp.resolve_product_mcp_config

    def discover(root: Path) -> Path | None:
        discoveries.append(root)
        return resolve_config(root)

    @asynccontextmanager
    async def open_tools(path: Path, **options: object):
        opened.append(path)
        yield ()

    monkeypatch.setattr(mcp, "resolve_product_mcp_config", discover)
    monkeypatch.setattr(mcp, "open_trusted_product_mcp_tools", open_tools)
    restarted = Agent(
        workspace_path=workspace, checkpoint_db=database,
        enable_workspace_mcp=current_enabled, permissions=policy(current_denied),
        mcp_config_trust=mcp.decide_mcp_config_trust(config, workspace_root=workspace, trust_workspace=True),
    )
    monkeypatch.setattr(restarted, "_model_factory", PatchThenAnswerModel)

    resumed = await restarted.resume(paused.turn_id, "allow_once")

    # Changed permission scope invalidates the old approval before execution.
    scope_changed = original_denied != current_denied
    assert resumed.status == ("paused" if scope_changed else "done")
    assert resumed.turn_id == paused.turn_id
    assert target.read_text(encoding="utf-8") == ("before" if scope_changed else "after")
    assert len(discoveries) == expected_discoveries
    assert len(opened) == expected_discoveries
    assert restarted.enable_workspace_mcp is current_enabled
    async with (await RuntimeState.open(database)) as store:
        assert store.read_turn(paused.turn_id).binding_manifest["mcp_policy"] == frozen_policy
        assert store.verify().valid


@pytest.mark.anyio
async def test_public_resume_rechecks_current_tool_denial_before_one_time_approval(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = tmp_path / "value.txt"
    target.write_text("before", encoding="utf-8")
    database = tmp_path / "rollout.sqlite3"
    original = Agent(workspace_path=tmp_path, checkpoint_db=database, enable_workspace_mcp=False)
    monkeypatch.setattr(original, "_model_factory", PatchThenAnswerModel)
    paused = await original.run("change value.txt", require_workspace_change=False)
    assert paused.status == "paused"

    restarted = Agent(
        workspace_path=tmp_path, checkpoint_db=database, enable_workspace_mcp=False,
        permissions=PermissionPolicy(tools=(ToolRule(pattern="apply_patch", decision="deny"),)),
    )
    monkeypatch.setattr(restarted, "_model_factory", PatchThenAnswerModel)
    resumed = await restarted.resume(paused.turn_id, "allow_once")

    assert resumed.status == "paused"
    assert target.read_text(encoding="utf-8") == "before"
    async with (await RuntimeState.open(database)) as store:
        [approval] = store.list_approvals(paused.turn_id)
        assert approval.status == "invalidated"
        assert store.verify().valid
