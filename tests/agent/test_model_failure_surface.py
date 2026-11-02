from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import cast

import pytest

from agent_runtime.cli import _display_agent_result
from agent_runtime.harness import RuntimeStore, TurnResult, facts, model_ops, turn_ops
from agent_runtime.result import AgentResult
from agent_runtime.streaming.events import EventType, StreamEvent
from agent_runtime.terminal_render import TerminalToolEventDisplay
from agent_runtime.terminal_ui import Conversation, ConversationEventDisplay


def _failure_payload(**changes):
    return {
        "kind": "rate_limited",
        "phase": "response",
        "provider_outcome": "rejected",
        "billing_outcome": "released",
        "http_status": 429,
        "provider_code": "rate_limit_exceeded",
        "request_id": "req-safe-42",
        "retry_at": 2_000_000_001.0,
        "retryable": True,
        "output_started": False,
        "message": "Provider rate limit reached.",
        **changes,
    }


def _append_attempt_failure(store, turn, thread, *, unknown=False, failure=None, request="first"):
    operation = model_ops.prepare_model_operation(
        store,
        turn_id=turn.turn_id,
        request_hash="request-hash",
        context_hash="context-hash",
        tool_hash="tool-hash",
        wire_hash="wire-hash",
        request_ref={"input_revision": 1, "request_id": request},
    )
    attempt = model_ops.dispatch_model_attempt(store, operation.operation_id)

    def seed():
        with store.persistence.transaction():
            facts.append_fact(
                store,
                thread_id=thread.thread_id,
                turn_id=turn.turn_id,
                record_type="model_attempt_unknown" if unknown else "model_attempt_rejected",
                producer="runtime",
                payload={
                    "operation_id": operation.operation_id,
                    "attempt_id": attempt.attempt_id,
                    "generation": attempt.generation,
                    "reason": "provider request failed",
                    "error_type": "LegacyError",
                    "error_message": "legacy diagnostic",
                    **({} if failure is None else {"failure": failure}),
                },
            )

    store.run_sync(seed)
    return operation


def _project(store, turn, thread, status):
    return AgentResult._from_harness(
        TurnResult(thread_id=thread.thread_id, turn_id=turn.turn_id, answer=None, status=status),
        store=store,
    )


@pytest.mark.parametrize("status", ["failed", "paused", "completed"])
def test_public_result_restores_latest_structured_failure_for_this_turn(tmp_path: Path, status):
    with RuntimeStore.open_sync(tmp_path / "surface.sqlite") as store:
        thread = turn_ops.create_thread(store, workspace=tmp_path)
        turn = turn_ops.start_turn(store, thread_id=thread.thread_id, user_message="task", binding_manifest={})
        _append_attempt_failure(store, turn, thread, failure=_failure_payload(request_id="old"))
        latest = _failure_payload(request_id="req-latest")
        operation = _append_attempt_failure(
            store, turn, thread, unknown=status == "paused", failure=latest, request="second"
        )
        if status == "paused":
            turn_ops.pause_turn(store, turn_id=turn.turn_id, reason="outcome unknown")
        elif status == "failed":
            turn_ops.fail_turn(store, turn_id=turn.turn_id, reason_code="model_provider_failed", message="failed")
        else:
            turn_ops.complete_turn(store, turn_id=turn.turn_id, answer="recovered")
        result = _project(store, turn, thread, status)
        if status == "completed":
            assert result.model_error is None
            return
        from agent_runtime import ModelFailure

        assert isinstance(result.model_error, ModelFailure)
        assert result.model_error.to_payload() == latest
        if status == "paused":
            assert result.pause is not None
            assert result.pause.kind == "model_retry"
            assert result.pause.context["operation_id"] == operation.operation_id
            assert dict(result.pause.context["failure"]) == latest


def test_legacy_result_has_no_model_error_and_keeps_diagnostics(tmp_path):
    with RuntimeStore.open_sync(tmp_path / "legacy.sqlite") as store:
        thread = turn_ops.create_thread(store, workspace=tmp_path)
        turn = turn_ops.start_turn(store, thread_id=thread.thread_id, user_message="task", binding_manifest={})
        _append_attempt_failure(store, turn, thread)
        result = _project(store, turn, thread, "failed")
        assert result.model_error is None
        assert result.diagnostics[-1].message == "legacy diagnostic"


@pytest.mark.parametrize("chat", [False, True])
@pytest.mark.parametrize("status", ["failed", "paused"])
def test_cli_shows_actionable_model_failure_without_verbose(tmp_path, capsys, chat, status):
    with RuntimeStore.open_sync(tmp_path / "cli.sqlite") as store:
        thread = turn_ops.create_thread(store, workspace=tmp_path)
        turn = turn_ops.start_turn(store, thread_id=thread.thread_id, user_message="task", binding_manifest={})
        _append_attempt_failure(store, turn, thread, failure=_failure_payload())
        result = replace(_project(store, turn, thread, status), stop_reason="model_provider_failed")
    if chat:
        view = Conversation()
        display = ConversationEventDisplay(view)
        display.present_result(result)
        text = view.plain_text()
    else:
        _display_agent_result(result, verbose=False)
        text = capsys.readouterr().out
    assert "rate_limit" in text
    assert "HTTP 429" in text
    assert "rate_limit_exceeded" in text
    assert "req-safe-42" in text
    assert "Provider rate limit reached." in text


def test_model_failure_does_not_leak_from_previous_turn(tmp_path):
    with RuntimeStore.open_sync(tmp_path / "isolation.sqlite") as store:
        thread = turn_ops.create_thread(store, workspace=tmp_path)
        old = turn_ops.start_turn(store, thread_id=thread.thread_id, user_message="old", binding_manifest={})
        _append_attempt_failure(store, old, thread, failure=_failure_payload())
        turn_ops.fail_turn(store, turn_id=old.turn_id, reason_code="model_provider_failed", message="failed")
        current = turn_ops.start_turn(store, thread_id=thread.thread_id, user_message="new", binding_manifest={})
        assert _project(store, current, thread, "paused").model_error is None


def test_structured_failure_restoration_discards_untrusted_message(tmp_path, capsys):
    with RuntimeStore.open_sync(tmp_path / "safe.sqlite") as store:
        thread = turn_ops.create_thread(store, workspace=tmp_path)
        turn = turn_ops.start_turn(store, thread_id=thread.thread_id, user_message="task", binding_manifest={})
        _append_attempt_failure(store, turn, thread, failure=_failure_payload(message="RAW BODY sk-secret"))
        result = _project(store, turn, thread, "failed")
    _display_agent_result(result, verbose=True)
    assert result.model_error.message == "Provider rate limit reached."
    assert "sk-secret" not in capsys.readouterr().out


def _retry_event():
    return StreamEvent(
        type=cast(EventType, "model_retry_scheduled"),
        turn_id="turn",
        sequence=1,
        data={
            "operation_id": "op",
            "attempt_id": "attempt",
            "generation": 1,
            "retry_number": 1,
            "max_retries": 3,
            "retry_at": 2_000_000_001.0,
            "deadline_at": 2_000_000_030.0,
            "delay_ms": 1500.0,
            "failure": _failure_payload(),
        },
    )


@pytest.mark.anyio
@pytest.mark.parametrize("chat", [False, True])
async def test_retry_wait_is_runtime_notification_and_deduplicated(chat, capsys):
    view = Conversation()
    display = ConversationEventDisplay(view) if chat else TerminalToolEventDisplay(interactive=False)
    event = _retry_event()
    display.begin_turn()
    await display.emit(event)
    await display.emit(event)
    assert "重试" in display.activity_text()
    text = view.plain_text() if chat else capsys.readouterr().out
    assert "1/3" in text
    assert "1500" in text
    assert "rate_limit" in text
    assert text.count("1/3") == 1
    assert display.answer_streamed is False
    if chat:
        assert all(block.kind != "answer" for block in view.blocks)
    await display.emit(StreamEvent(type=EventType.TURN_CANCELLATION_REQUESTED, turn_id="turn", sequence=2))
    stopping = display.activity_text()
    assert "停止" in stopping
    await display.emit(event)
    assert "停止" in display.activity_text()
    assert "重试" not in display.activity_text()
    text = view.plain_text() if chat else capsys.readouterr().out
    assert text.count("1/3") == (1 if chat else 0)
    display.finish()
