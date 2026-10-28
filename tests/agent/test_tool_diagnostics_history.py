from __future__ import annotations

import asyncio
import json
import sqlite3
from dataclasses import replace
from pathlib import Path

import pytest

from agent_runtime.core import runtime_diagnostics
from agent_runtime.harness import RuntimeState, ToolOrchestrator
from agent_runtime.tools.permissions import ToolExecutionContext
from tests.agent.harness.test_public_agent_cutover import PatchThenAnswerModel
from tests.agent.harness.test_public_result_projection import AcceptPublicAnswer
from tests.agent.harness.test_tool_orchestrator import _read_tool
from tests.agent.test_single_tool_executor import _call


def test_exception_diagnostic_is_bounded_and_omits_source_and_locals(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TEST_TOOL_SECRET", "tool-credential-0123456789")

    def recurse(depth: int) -> None:
        local_private_value = "locals-must-not-appear"
        if depth:
            recurse(depth - 1)
        else:
            raise RuntimeError("tool-credential-0123456789 " + "detail " * 500)
        assert local_private_value

    assert hasattr(runtime_diagnostics, "exception_diagnostic")
    try:
        recurse(25)
    except RuntimeError as error:
        diagnostic = runtime_diagnostics.exception_diagnostic(error, stage="runner_failed")
    encoded = json.dumps(diagnostic)
    assert "tool-credential-0123456789" not in encoded
    assert "locals-must-not-appear" not in encoded
    assert "raise RuntimeError" not in encoded
    assert diagnostic["exception_type"] == "RuntimeError"
    assert len(diagnostic["message"]) <= 500
    assert len(diagnostic["traceback"]) == 12
    assert diagnostic["traceback_truncated"] is True


def test_rollout_read_only_does_not_create_or_modify_database(tmp_path: Path) -> None:
    absent = tmp_path / "absent" / "rollout.sqlite3"
    with pytest.raises((FileNotFoundError, sqlite3.OperationalError)):
        RuntimeState.open_sync(absent, read_only=True)
    assert not absent.parent.exists()
    database = tmp_path / "rollout.sqlite3"
    with RuntimeState.open_sync(database) as store:
        thread = store.create_thread(workspace=tmp_path)
        turn = store.start_turn(thread_id=thread.thread_id, user_message="read", binding_manifest={})
        thread = store.read_thread(thread.thread_id)
    before = database.read_bytes()
    with RuntimeState.open_sync(database, read_only=True) as reader:
        assert reader.read_thread(thread.thread_id) == thread
        with pytest.raises(sqlite3.OperationalError, match="readonly"):
            reader.create_thread(workspace=tmp_path)
        with pytest.raises(sqlite3.OperationalError, match="readonly"):
            reader.commit_artifact(turn_id=turn.turn_id, content=b"private", media_type="text/plain", name="private")
        assert not reader.artifact_root.exists()
    assert database.read_bytes() == before


def test_diagnostic_survives_durable_tool_result_and_history_lookup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from agent_runtime import history

    monkeypatch.setenv("TEST_TOOL_TOKEN", "history-secret-0123456789")

    def fail(_arguments: object) -> object:
        raise LookupError("history-secret-0123456789 unavailable")

    database = tmp_path / "rollout.sqlite3"
    with RuntimeState.open_sync(database) as store:
        thread = store.create_thread(workspace=tmp_path)
        turn = store.start_turn(
            thread_id=thread.thread_id, user_message="read", binding_manifest={"model_id": "test"},
        )
        tool = replace(_read_tool(), run=fail)
        orchestrator = ToolOrchestrator(
            store=store, tools={"read_file": tool}, execution_context=ToolExecutionContext(workspace_root=tmp_path),
        )
        call = _call("read_file", arguments={"path": "private-argument-path"})
        result = asyncio.run(orchestrator.execute(turn_id=turn.turn_id, call=call))
        assert result.metadata["diagnostic"]["turn_id"] == turn.turn_id
        [operation] = store.list_tool_operations(turn.turn_id)
        item = store.read_item(operation.result_item_id)
        assert item.payload["metadata"]["diagnostic"]["exception_type"] == "LookupError"
        assert store.verify().valid
    before = database.read_bytes()
    rows = history.read_history(database, turn_id=turn.turn_id, tool_call_id=call.tool_call_id)
    assert len(rows) == 1
    row = rows[0]
    assert row["item_id"] == item.item_id
    assert row["operation_id"] == operation.operation_id
    assert row["diagnostic"]["turn_id"] == turn.turn_id
    assert "history-secret-0123456789" not in json.dumps(row)
    assert "private-argument-path" not in json.dumps(row)
    assert database.read_bytes() == before
    assert history.read_history(database, item_id=item.item_id) == rows


def test_commit_failure_keeps_diagnostic_without_claiming_durable_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from agent_runtime.harness.tool_orchestrator import ToolResultCommitError

    monkeypatch.setenv("TEST_COMMIT_SECRET", "commit-secret-0123456789")
    with RuntimeState.open_sync(tmp_path / "rollout.sqlite3") as store:
        thread = store.create_thread(workspace=tmp_path)
        turn = store.start_turn(thread_id=thread.thread_id, user_message="read", binding_manifest={})
        tool = _read_tool()
        orchestrator = ToolOrchestrator(
            store=store, tools={"read_file": tool}, execution_context=ToolExecutionContext(workspace_root=tmp_path),
        )

        def fail_commit(**_kwargs: object) -> bool:
            raise OSError("commit-secret-0123456789 storage unavailable")

        monkeypatch.setattr(store, "commit_tool_operation_result", fail_commit)
        call = _call("read_file", arguments={"path": "README.md"})
        with pytest.raises(ToolResultCommitError) as captured:
            asyncio.run(orchestrator.execute(turn_id=turn.turn_id, call=call))
        result = captured.value.result
        assert result.error_code == "tool_result_commit_failed"
        assert result.retryable is False
        diagnostic = result.metadata["diagnostic"]
        assert diagnostic["exception_type"] == "OSError"
        assert diagnostic["turn_id"] == turn.turn_id
        assert diagnostic["tool_call_id"] == call.tool_call_id
        assert "commit-secret-0123456789" not in diagnostic["message"]
        assert "commit-secret-0123456789" not in str(captured.value)
        assert captured.value.__suppress_context__ is True
        assert not any(item.kind == "tool_result" for item in store.list_items(turn.turn_id))
        [operation] = store.list_tool_operations(turn.turn_id)
        assert operation.status == "running"
        assert diagnostic["operation_id"] == operation.operation_id


@pytest.mark.anyio
async def test_public_session_failure_is_inspectable_without_model_replay(tmp_path: Path) -> None:
    from agent_runtime.harness import HarnessModelResponse, HarnessToolCall, PreparedModelCall, Session
    from agent_runtime.history import read_history

    class DiagnosticModel(PatchThenAnswerModel):
        async def dispatch(self, prepared: PreparedModelCall, *, event_sink: object = None) -> HarnessModelResponse:
            if prepared.request_ref["request_id"].endswith(":step:1"):
                return HarnessModelResponse(
                    text="",
                    tool_calls=(HarnessToolCall(id="public-failure", name="read_file", arguments={"path": "x"}),),
                    provider_response_id="failure-call", usage={"input_tokens": 1, "output_tokens": 1},
                )
            return HarnessModelResponse(
                text="read failed", provider_response_id="failure-answer",
                usage={"input_tokens": 1, "output_tokens": 1},
            )

    def fail(_arguments: object) -> object:
        raise ArithmeticError("ordinary tool bug")

    database = tmp_path / "rollout.sqlite3"
    async with await Session.open(
        database=database, workspace=tmp_path, model=DiagnosticModel(),
        tools={"read_file": replace(_read_tool(), run=fail)}, completion_gate=AcceptPublicAnswer(),
    ) as session:
        result = await session.submit("read the file")
        assert result.status == "done", (result.stop_reason, result.diagnostics)
        assert result.tool_calls[0].error_code == "runner_failed"
        assert session.store.verify().valid
    [row] = (await asyncio.to_thread(read_history, database, turn_id=result.turn_id, tool_call_id="public-failure"))
    assert row["diagnostic"]["exception_type"] == "ArithmeticError"
    assert row["diagnostic"]["turn_id"] == result.turn_id
    assert row["retryable"] is False


def test_public_cli_history_is_read_only_and_never_opens_model(tmp_path, monkeypatch):
    import json

    from typer.testing import CliRunner

    from agent_runtime import cli
    from tests.agent.test_agent_cli_resume import _persist_cli_turn

    database = tmp_path / 'state.sqlite'
    turn_id = _persist_cli_turn(database, tmp_path / 'workspace')
    before = database.read_bytes()
    monkeypatch.setattr(cli, '_create_agent_facade', lambda **kwargs: pytest.fail('History must not open an Agent'))
    result = CliRunner().invoke(cli.agent_app, ['history', '--checkpoint-db', str(database)])
    assert result.exit_code == 0, result.output
    assert json.loads(result.output)[0]['turn_id'] == turn_id
    assert database.read_bytes() == before
    missing = tmp_path / 'missing.sqlite'
    absent = CliRunner().invoke(cli.agent_app, ['history', '--checkpoint-db', str(missing)])
    assert absent.exit_code != 0 and not missing.exists()


def test_history_web_details_are_bounded_and_exclude_private_content(tmp_path: Path) -> None:
    from agent_runtime.history import read_history

    database = tmp_path / "history.sqlite"
    with RuntimeState.open_sync(database) as store:
        thread = store.create_thread(workspace=tmp_path)
        turn = store.start_turn(thread_id=thread.thread_id, user_message="read", binding_manifest={})
        item = store.record_tool_result(turn_id=turn.turn_id, operation_id=None, result={
            "tool_call_id": "web-call", "tool_name": "web_fetch", "is_error": True,
            "error_code": "web_browser_budget_exceeded", "error_message": "browser budget exhausted",
            "retryable": False,
            "structured_content": {
                "url": "https://user:private-password@example.com/path?query=private-query#private-fragment",
                "source_id": "source-1", "static_source_id": "source-0", "content": "private-body",
                "render_mode": "browser", "failure_stage": "browser_render",
                "request_budget": {"used": 3, "remaining": 4, "limit": 7, "unit": "network_tool_attempts",
                                   "last_reservation_item_id": "usage-1", "unknown_attempts_counted": True},
                "render_diagnostics": {
                    "requests": 2, "wire_bytes": 400, "raw_body": "private-resource-body",
                    "failed_resources": [{"url": "https://example.com/r?token=private-resource-token",
                                          "reason": "timeout", "method": "GET", "resource_type": "Script"}],
                },
                "comparison": {"status": "changed", "baseline": {"source_id": "source-0",
                                                                  "url": "https://example.com/path?private=query"}},
                "failure_origin": {"item_id": "failed-item", "tool_call_id": "failed-call", "body": "private-body"},
            },
            "metadata": {"diagnostic": {"traceback": [{"filename": "x.py", "lineno": 1, "function": "f",
                                                        "locals": "private-locals", "source": "private-source"}]}},
        })
    [row] = read_history(database, item_id=item.item_id)
    assert row["web"]["source_id"] == "source-1"
    assert row["web"]["url"] == "https://example.com/path"
    assert row["web"]["request_budget"]["remaining"] == 4
    assert row["web"]["request_budget"]["last_reservation_item_id"] == "usage-1"
    assert row["web"]["failure_origin"] == {"item_id": "failed-item", "tool_call_id": "failed-call"}
    assert row["web"]["render_diagnostics"]["requests"] == 2
    assert row["web"]["comparison"]["baseline"]["source_id"] == "source-0"
    assert "private" not in json.dumps(row)


def test_history_reads_committed_results_from_an_active_wal(tmp_path: Path) -> None:
    from agent_runtime.history import list_history_turns, read_history

    database = tmp_path / "live.sqlite"
    with RuntimeState.open_sync(database) as writer:
        thread = writer.create_thread(workspace=tmp_path)
        turn = writer.start_turn(thread_id=thread.thread_id, user_message="read", binding_manifest={})
        item = writer.record_tool_result(turn_id=turn.turn_id, operation_id=None, result={
            "tool_call_id": "call-live", "tool_name": "read_file", "is_error": False,
        })
        [row] = read_history(database, item_id=item.item_id)
        assert row["turn_id"] == turn.turn_id
        assert row["tool_call_id"] == "call-live"
        assert list_history_turns(database)[0]["tool_result_count"] == 1


def test_cli_tool_commit_error_reports_safe_diagnostic_without_success(monkeypatch):
    import click

    from agent_runtime.cli import _run_cli_async
    from agent_runtime.core.runtime_diagnostics import exception_diagnostic
    from agent_runtime.harness.tool_orchestrator import ToolResultCommitError
    from agent_runtime.tools.tool import ToolResult

    monkeypatch.setenv('TEST_COMMIT_SECRET', 'fake-commit-secret-0123456789')
    diagnostic = exception_diagnostic(OSError('fake-commit-secret-0123456789 disk unavailable'),
                                      stage='commit', turn_id='turn_test', tool_call_id='call_test',
                                      operation_id='op_test')
    async def fail():
        raise ToolResultCommitError(ToolResult('call_test', 'read_file', is_error=True,
            error_code='tool_result_commit_failed', error_message='Tool result was not accepted.',
            metadata={'diagnostic': diagnostic}))
    with pytest.raises(click.ClickException) as caught:
        _run_cli_async(fail())
    assert 'OSError' in str(caught.value)
    assert 'turn_test' in str(caught.value) and 'call_test' in str(caught.value)
    assert 'fake-commit-secret-0123456789' not in str(caught.value)
    assert '未确认持久化' in str(caught.value)
