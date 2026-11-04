import json

from typer.testing import CliRunner

from agent_runtime.background import BackgroundHandle
from agent_runtime.cli import agent_app


def test_run_background_uses_sdk_and_returns_durable_identity(tmp_path, monkeypatch):
    from agent_runtime.agent import Agent

    calls = []
    async def submit(self, text, **options):
        calls.append((text, options))
        return BackgroundHandle(
            database=tmp_path / "runtime.db", workspace=tmp_path,
            thread_id="thread-test", turn_id="turn-test", autostart=False,
        )
    monkeypatch.setattr(Agent, "start_background", submit)
    result = CliRunner().invoke(agent_app, [
        "run", "inspect repository", "--background", "--input-id", "request-1",
        "--checkpoint-db", str(tmp_path / "runtime.db"), "--max-steps", "7",
    ])
    assert result.exit_code == 0, result.output
    assert json.loads(result.output)["turn_id"] == "turn-test"
    assert calls[0][1]["input_id"] == "request-1"
    assert calls[0][1]["max_steps"] == 7


def test_background_status_does_not_start_a_model_or_host(tmp_path, monkeypatch):
    async def status(self):
        return {"thread_id": self.thread_id, "turn_id": self.turn_id, "status": "paused"}
    monkeypatch.setattr(BackgroundHandle, "status", status)
    result = CliRunner().invoke(agent_app, [
        "background", "status", "thread-test", "turn-test", "--checkpoint-db", str(tmp_path / "runtime.db")
    ])
    assert result.exit_code == 0, result.output
    assert json.loads(result.output)["status"] == "paused"


def test_background_result_reads_the_sdk_result(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from agent_runtime.result import AgentUsage

    async def result(self):
        return SimpleNamespace(
            thread_id=self.thread_id, turn_id=self.turn_id, status="done", answer="verified result",
            stop_reason="completed", usage=AgentUsage(), diagnostics=(),
            delivery={"status": "unverified", "next_actions": ["Record scope and evidence"]},
        )

    monkeypatch.setattr(BackgroundHandle, "result", result)
    result = CliRunner().invoke(agent_app, [
        "background", "result", "thread-test", "turn-test", "--checkpoint-db", str(tmp_path / "runtime.db")
    ])
    assert result.exit_code == 0, result.output
    assert json.loads(result.output)["answer"] == "verified result"
    assert json.loads(result.output)["delivery"]["status"] == "unverified"


def test_background_wait_preserves_timeout_semantics(tmp_path, monkeypatch):
    timeouts = []

    async def wait(self, *, timeout=None):
        timeouts.append(timeout)
        return {"status": "completed"}

    monkeypatch.setattr(BackgroundHandle, "wait", wait)
    result = CliRunner().invoke(agent_app, [
        "background", "wait", "thread-test", "turn-test", "--checkpoint-db", str(tmp_path / "runtime.db"),
        "--timeout", "2",
    ])
    assert result.exit_code == 0, result.output
    assert json.loads(result.output)["status"] == "completed"
    assert timeouts == [2]
