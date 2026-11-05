import json
from pathlib import Path

from typer.testing import CliRunner

from agent_runtime.cli import agent_app
from agent_runtime.harness import RuntimeStore, host_ops


def test_local_cli_config_binding_and_revocation(tmp_path: Path):
    database = tmp_path / "runtime.db"
    with RuntimeStore.open_sync(database) as store:
        initial = host_ops.receive_command(store, workspace=tmp_path, kind="start", payload={"text": "work"})
    config = tmp_path / "source.json"
    config.write_text(
        json.dumps({"source_id": "gh", "kind": "github", "repository": "o/r", "webhook_secret_env": "GH_SECRET"})
    )
    runner = CliRunner()
    options = ["--checkpoint-db", str(database), "--workspace", str(tmp_path)]
    assert runner.invoke(agent_app, ["channel", "configure", initial.thread_id, str(config), *options]).exit_code == 0
    assert (
        runner.invoke(
            agent_app, ["channel", "bind", "gh", "github:o/r:1", initial.thread_id, "github:o/r:1", *options]
        ).exit_code
        == 0
    )
    args = ["channel", "grant", "gh", "7", "github:o/r:1", initial.thread_id, "--operations", "status,notify", *options]
    assert runner.invoke(agent_app, args).exit_code == 0
    assert runner.invoke(agent_app, [*args, "--no-enabled"]).exit_code == 0
    result = runner.invoke(agent_app, ["channel", "status", *options])
    assert result.exit_code == 0, result.output
    assert json.loads(result.output)["grant"][0]["enabled"] is False
