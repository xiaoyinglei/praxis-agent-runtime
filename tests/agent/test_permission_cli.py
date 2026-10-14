import json

from typer.main import get_command
from typer.testing import CliRunner

from agent_runtime.cli import agent_app


def test_permissions_inspection_and_denial_need_no_model(tmp_path):
    config = tmp_path / "permissions.json"
    config.write_text(json.dumps({"profile": "read-only"}))
    result = CliRunner().invoke(agent_app, ["permissions", "--permissions", str(config), "--workspace", str(tmp_path)])
    assert result.exit_code == 0, result.output
    assert json.loads(result.output)["effective"]["profile"] == "read-only"
    result = CliRunner().invoke(
        agent_app,
        [
            "permissions",
            "--permissions",
            str(config),
            "--workspace",
            str(tmp_path),
            "--path",
            str(tmp_path / "result"),
            "--write",
        ],
    )
    assert result.exit_code == 0, result.output
    assert json.loads(result.output)["decision"] == "deny"


def test_public_cli_exposes_policy_for_new_and_recovered_work():
    runner = CliRunner()
    group = get_command(agent_app)
    for command in ("run", "chat", "resume", "recover"):
        result = runner.invoke(agent_app, [command, "--help"])
        assert result.exit_code == 0
        # Rich can truncate option labels at narrow terminal widths. Inspect
        # the real Click option rather than the width-dependent projection.
        assert any("--permissions" in parameter.opts for parameter in group.commands[command].params)


def test_invalid_policy_fails_before_execution(tmp_path):
    config = tmp_path / "permissions.json"
    config.write_text('{"allow_everything_typo":true}')
    result = CliRunner().invoke(agent_app, ["permissions", "--permissions", str(config)])
    assert result.exit_code != 0


def test_policy_sources_cannot_be_modified(tmp_path):
    from agent_runtime import Agent

    config = tmp_path / "permissions.json"
    config.write_text('{"profile":"trusted-local"}')
    permissions = Agent(workspace_path=tmp_path, permissions=config).effective_permissions()
    assert permissions.path_access(config)
    assert not permissions.path_access(config, write=True)
