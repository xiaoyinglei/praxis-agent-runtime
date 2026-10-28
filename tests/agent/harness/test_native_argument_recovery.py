"""A committed malformed call must remain recoverable without executing it."""

from __future__ import annotations

import asyncio
import json

import pytest

from agent_runtime.core.messages import StopReason, ToolCall, ToolUseResult
from agent_runtime.harness import RolloutContextManager, RuntimeState, TurnExecutor
from agent_runtime.modeling.contracts import normalize_llm_usage
from agent_runtime.modeling.gateway import AgentModelResponse, model_request_input_text
from agent_runtime.modeling.openai_wire import serialize_openai_request
from tests.agent.harness.test_affordable_tool_continuation import FileWorkflowGateway, file_runner
from tests.provider.test_native_argument_errors import native_response


class CorrectedArgumentsGateway(FileWorkflowGateway):
    async def agenerate_model_request(self, **kwargs):
        request = kwargs["request"]
        self.requests.append(request)
        count = len(self.requests)
        if count == 1:
            turn = native_response('{"file_path":"result.txt","new_string":"unfinished')
        elif count == 2:
            feedback = [json.loads(m.content) for m in kwargs["request"].messages if m.role == "tool"]
            assert feedback[-1]["error_code"] == "invalid_arguments"
            assert "JSON" in feedback[-1]["error_message"]
            assert "missing" not in feedback[-1]["error_message"]
            assert not (self.workspace / "result.txt").exists()
            turn = ToolUseResult(
                tool_calls=[
                    ToolCall(
                        id="corrected",
                        name="apply_patch",
                        input={
                            "file_path": "result.txt",
                            "new_string": self.body,
                        },
                    )
                ],
                stop_reason=StopReason.TOOL_USE,
                raw_stop_reason="tool_calls",
            )
        elif count == 3:
            turn = ToolUseResult(
                tool_calls=[ToolCall(id="read", name="read_file", input={"path": "result.txt"})],
                stop_reason=StopReason.TOOL_USE,
                raw_stop_reason="tool_calls",
            )
        else:
            observed = [
                json.loads(m.content) for m in request.messages if m.role == "tool" and m.tool_call_id == "read"
            ]
            assert observed[-1]["structured_content"]["content"] == self.body
            turn = ToolUseResult(
                text="Created and inspected the file.", stop_reason=StopReason.END_TURN, raw_stop_reason="stop"
            )
        measured = len(
            model_request_input_text(
                request, provider=kwargs["provider"], supports_native_tools=kwargs["supports_native_tools"]
            )
        )
        return AgentModelResponse(
            turn=turn,
            usage=normalize_llm_usage(
                input_tokens=measured,
                output_tokens=50,
                input_tokens_include_cache=True,
                usage_source="provider",
            ),
            provider_wire_hash=serialize_openai_request(request).provider_wire_hash,
            serializer_revision="openai-wire-v1",
            wire_kind="openai-compatible",
        )


@pytest.mark.parametrize("restart", [False, True])
def test_bad_call_feedback_then_corrected_file_action_survives_restart(tmp_path, monkeypatch, restart):
    class Crash(BaseException):
        pass

    gateway = CorrectedArgumentsGateway(path="result.txt", body="verified\n")
    gateway.workspace = tmp_path
    original_handle = TurnExecutor._handle_model_response

    async def stop_after_commit(self, **kwargs):
        raise Crash()

    database = tmp_path / "run.db"
    with RuntimeState.open_sync(database) as store:
        runner, turn, initial_gateway = file_runner(store, tmp_path)
        monkeypatch.setattr(initial_gateway, "agenerate_model_request", gateway.agenerate_model_request)
        if restart:
            monkeypatch.setattr(TurnExecutor, "_handle_model_response", stop_after_commit)
            with pytest.raises(Crash):
                asyncio.run(runner.run_turn(runner.restore_turn_context(turn.turn_id), start_step=1))
            assert not (tmp_path / "result.txt").exists()
            assert not store.list_tool_operations(turn.turn_id)
            monkeypatch.setattr(TurnExecutor, "_handle_model_response", original_handle)
        else:
            result = asyncio.run(runner.run_turn(runner.restore_turn_context(turn.turn_id), start_step=1))
            assert result.status == "completed"
    with RuntimeState.open_sync(database) as reopened:
        runner._store = reopened
        runner._context_manager = RolloutContextManager(reopened)
        runner._tool_orchestrator._store = reopened
        if restart:
            result = asyncio.run(runner.recover_committed_model_response(turn_id=turn.turn_id))
            assert result.status == "completed"
        assert (tmp_path / "result.txt").read_text() == "verified\n"
        results = [i.payload for i in reopened.list_items(turn.turn_id) if i.kind == "tool_result"]
        assert len(results) == 3
        assert results[0]["error_code"] == "invalid_arguments"
        assert "JSON" in results[0]["error_message"]
        assert len(gateway.requests) == 4
        assert len(reopened.list_tool_operations(turn.turn_id)) == 2
        messages = RolloutContextManager(reopened).build(turn.turn_id)
        call = next(c for m in messages for c in m.tool_calls if c.id == "patch")
        assert "JSON" in call.argument_error
        context = runner.restore_turn_context(turn.turn_id)
        for operation in reopened.list_model_operations(turn.turn_id):
            replay = asyncio.run(runner._prepare_step(runner.restore_step_context(context, operation)))
            assert (replay.request_hash, replay.context_hash, replay.tool_hash, replay.wire_hash) == (
                operation.request_hash, operation.context_hash, operation.tool_hash, operation.wire_hash,
            )
        assert reopened.verify().valid
