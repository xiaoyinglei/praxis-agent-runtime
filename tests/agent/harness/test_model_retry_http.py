"""Real SDK + wire codec + durable driver, with local HTTP transport evidence."""

import asyncio
import json
from dataclasses import replace

import httpx
import pytest

from agent_runtime.harness import RolloutContextManager, RuntimeStore, TurnRunner
from agent_runtime.harness.retry_policy import RetryPolicy
from agent_runtime.model_definition import RequestDefaultsDefinition
from tests.agent.harness.test_compaction_consistency import Accept
from tests.agent.harness.test_provider_failure_boundary import _generator, _model


def stream_response():
    chunks = [
        {
            "id": "chat",
            "choices": [{"index": 0, "delta": {"role": "assistant", "content": "answer"}, "finish_reason": None}],
        },
        {
            "id": "chat",
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12},
        },
    ]
    body = "".join("data: " + json.dumps(c) + "\n\n" for c in chunks) + "data: [DONE]\n\n"
    return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=body.encode())


@pytest.mark.parametrize("status", [429, 503])
@pytest.mark.parametrize("extensions", [False, True])
def test_one_wire_per_attempt_and_exact_frozen_request_replay(tmp_path, status, extensions):
    bodies = []

    def respond(request):
        bodies.append(request.content)
        if len(bodies) == 1:
            return httpx.Response(
                status,
                json={"error": {"code": "rate_limit_exceeded" if status == 429 else "server_error"}},
                headers={"Retry-After": "0"},
            )
        return stream_response()

    gen = _generator(respond)
    try:
        with RuntimeStore.open_sync(tmp_path / "wire.db") as store:
            from agent_runtime.harness import turn_ops

            thread = turn_ops.create_thread(store, workspace=tmp_path)
            model = _model(gen)
            if extensions:
                model._resolved = replace(
                    model._resolved,
                    request_defaults=RequestDefaultsDefinition(
                        provider_options={
                            "thinking": {"type": "enabled"},
                            "chat_template_kwargs": {"enable_thinking": True},
                        }
                    ),
                )
            runner = TurnRunner(
                thread_id=thread.thread_id,
                store=store,
                model=model,
                context_manager=RolloutContextManager(store),
                completion_gate=Accept(),
            )
            result = asyncio.run(
                runner.run(
                    turn_id="wire-turn",
                    user_message="answer",
                    binding_manifest={
                        "model_retry_policy": RetryPolicy(base_delay_ms=0, max_delay_ms=0).to_payload(),
                        "model_token_budget_total": 20000,
                        "model_step_budget": 4,
                    },
                )
            )
            assert result.status == "completed"
            assert len(bodies) == 2
            assert bodies[0] == bodies[1]
            [op] = store.list_model_operations(result.turn_id)
            assert len(store.list_model_attempts(op.operation_id)) == 2
            budget = store.read_budget_state(result.turn_id)
            assert budget.exposure.model_calls == 2
            assert budget.used.retries == 1
            assert (budget.uncertain.total_tokens > 0) == (status == 503)
            assert store.verify().valid
    finally:
        gen.close()


@pytest.mark.parametrize(
    "delta",
    [
        {"content": "partial"},
        {"reasoning_content": "partial reasoning"},
        {
            "tool_calls": [
                {
                    "index": 0,
                    "id": "call",
                    "type": "function",
                    "function": {"name": "read_file", "arguments": '{"path":'},
                }
            ]
        },
        {
            "tool_calls": [
                {
                    "index": 0,
                    "id": "call",
                    "type": "function",
                    "function": {"name": "read_file", "arguments": '{"path":"x"}'},
                }
            ]
        },
    ],
)
def test_any_partial_output_does_not_replay_or_execute_tools(tmp_path, delta):
    calls = []

    class Broken(httpx.SyncByteStream):
        def __iter__(self):
            yield (
                "data: "
                + json.dumps({"id": "partial", "choices": [{"index": 0, "delta": delta, "finish_reason": None}]})
                + "\n\n"
            ).encode()
            raise httpx.ReadError("connection lost")

    def respond(request):
        calls.append(request)
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, stream=Broken())

    gen = _generator(respond)
    try:
        with RuntimeStore.open_sync(tmp_path / "stream.db") as store:
            from agent_runtime.harness import turn_ops

            thread = turn_ops.create_thread(store, workspace=tmp_path)
            runner = TurnRunner(
                thread_id=thread.thread_id,
                store=store,
                model=_model(gen),
                context_manager=RolloutContextManager(store),
                completion_gate=Accept(),
            )
            result = asyncio.run(
                runner.run(
                    turn_id="stream-turn",
                    user_message="answer",
                    binding_manifest={"model_retry_policy": RetryPolicy(base_delay_ms=0, max_delay_ms=0).to_payload()},
                )
            )
            assert result.status == "paused"
            assert len(calls) == 1
            assert not store.list_tool_operations(result.turn_id)
            [op] = store.list_model_operations(result.turn_id)
            assert op.status == "unknown"
            assert len(store.list_model_attempts(op.operation_id)) == 1
            assert store.verify().valid
    finally:
        gen.close()
