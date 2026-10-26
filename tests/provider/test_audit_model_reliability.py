import asyncio
import json
import threading

import pytest

from agent_runtime.modeling.budget import LLMBudgetLedger
from agent_runtime.modeling.contracts import LLMCallStage, LLMProviderResult, LLMUsage
from tests.provider.test_llm_gateway import _canonical_request, _gateway


class CharacterAccounting:
    def count(self, text):
        return len(text)


@pytest.mark.anyio
@pytest.mark.parametrize("stream", [False, True])
async def test_missing_usage_includes_tool_names_arguments_and_reasoning(stream):
    args = json.dumps({"path": "long argument"})

    class Generator:
        def generate_with_tools(self, **kwargs):
            return LLMProviderResult(
                {
                    "choices": [
                        {
                            "finish_reason": "tool_calls",
                            "message": {
                                "content": "answer",
                                "reasoning_content": "reasoning",
                                "tool_calls": [{"id": "call", "function": {"name": "read_file", "arguments": args}}],
                            },
                        }
                    ]
                }
            )

        def stream_with_tools(self, **kwargs):
            yield {"type": "text_delta", "content": "answer"}
            yield {"type": "thinking_delta", "content": "reasoning"}
            yield {"type": "tool_use_start", "tool_id": "call", "tool_name": "read_file"}
            yield {"type": "tool_input_delta", "tool_id": "call", "content": args}
            yield {"type": "content_block_stop", "tool_id": "call"}
            yield {"type": "message_stop", "stop_reason": "tool_use"}

    gateway = _gateway(Generator(), max_input_tokens=4000, max_output_tokens=128, model_context_tokens=8192)
    gateway._token_accounting = CharacterAccounting()
    response = await gateway.agenerate_model_request(
        stage=LLMCallStage.TOOL_DECISION,
        request=_canonical_request(),
        provider="openai-compatible",
        supports_native_tools=True,
        stream=stream,
    )
    assert response.usage.output_tokens == len("answer") + len("reasoning") + len("read_file") + len(args)
    assert response.usage.source == "tokenizer_estimate"


@pytest.mark.anyio
async def test_stream_output_estimate_is_independent_of_delta_partition_and_includes_plan():
    class Generator:
        def __init__(self, split):
            self.split = split

        def stream_with_tools(self, **kwargs):
            for channel, text in [
                ("text_delta", "hello world"),
                ("thinking_delta", "think twice"),
                ("plan_delta", "inspect files"),
            ]:
                for fragment in list(text) if self.split else [text]:
                    yield {"type": channel, "content": fragment}
            yield {"type": "tool_use_start", "tool_id": "call", "tool_name": "read_file"}
            args = '{"path": "some file"}'
            for fragment in list(args) if self.split else [args]:
                yield {"type": "tool_input_delta", "tool_id": "call", "content": fragment}
            yield {"type": "content_block_stop", "tool_id": "call"}
            yield {"type": "message_stop", "stop_reason": "tool_use"}

    results = []
    for split in (False, True):
        response = await _gateway(
            Generator(split), max_input_tokens=2000, max_output_tokens=128, model_context_tokens=4000
        ).agenerate_model_request(
            stage=LLMCallStage.TOOL_DECISION,
            request=_canonical_request(),
            provider="openai-compatible",
            supports_native_tools=True,
            stream=True,
        )
        results.append(response.usage.output_tokens)
    assert results == [10, 10]  # 2 text + 2 reasoning + 2 plan + 1 tool name + 3 JSON words


@pytest.mark.anyio
async def test_provider_reported_usage_stays_authoritative():
    class Generator:
        def stream_with_tools(self, **kwargs):
            yield {"type": "text_delta", "content": "a long uncounted message"}
            yield {
                "type": "message_stop",
                "stop_reason": "end_turn",
                "usage": LLMUsage(input_tokens=2, output_tokens=73, source="provider"),
            }

    result = await _gateway(
        Generator(), max_input_tokens=2000, max_output_tokens=128, model_context_tokens=4000
    ).agenerate_model_request(
        stage=LLMCallStage.TOOL_DECISION,
        request=_canonical_request(),
        provider="openai-compatible",
        supports_native_tools=True,
        stream=True,
    )
    assert result.usage.output_tokens == 73
    assert result.usage.source == "provider"


@pytest.mark.anyio
async def test_concurrent_equal_prompt_defaults_have_distinct_reservations():
    class Generator:
        def __init__(self):
            self.started = 0
            self.release = threading.Event()

        def generate_text_with_usage(self, **kwargs):
            self.started += 1
            self.release.wait(3)
            return LLMProviderResult("done", LLMUsage(input_tokens=6, output_tokens=4, source="provider"))

    generator = Generator()
    gateway = _gateway(generator, max_input_tokens=20, max_output_tokens=4, model_context_tokens=32)
    ledger = LLMBudgetLedger(total=25)
    prompt = "one two three four five six"
    tasks = [
        asyncio.create_task(gateway.agenerate_text(stage=LLMCallStage.TOOL_DECISION, prompt=prompt, ledger=ledger))
        for _ in range(3)
    ]
    try:
        for _ in range(100):
            if generator.started >= 2 and any(task.done() for task in tasks):
                break
            await asyncio.sleep(0.005)
        assert await ledger.reserved() == 20
        assert generator.started == 2
    finally:
        generator.release.set()
        results = await asyncio.gather(*tasks, return_exceptions=True)
    assert sum(isinstance(result, Exception) for result in results) == 1
    assert await ledger.committed() == 20


@pytest.mark.anyio
async def test_ledger_rejects_duplicate_active_lease_without_replacing_it():
    ledger = LLMBudgetLedger(total=100)
    assert await ledger.reserve("same", 10)
    with pytest.raises(ValueError, match="already reserved"):
        await ledger.reserve("same", 20)
    assert await ledger.reserved() == 10
    await ledger.refund("same")
    assert await ledger.reserve("same", 20)


@pytest.mark.anyio
async def test_discarded_partial_tool_output_still_has_an_estimated_cost():
    class Generator:
        def generate_with_tools(self, **kwargs):
            return LLMProviderResult(
                {
                    "choices": [
                        {
                            "finish_reason": "length",
                            "message": {
                                "content": None,
                                "tool_calls": [
                                    {"id": "partial", "function": {"name": "read_file", "arguments": '{"path":'}}
                                ],
                            },
                        }
                    ]
                }
            )

    gateway = _gateway(Generator(), max_input_tokens=4000, max_output_tokens=128, model_context_tokens=8192)
    gateway._token_accounting = CharacterAccounting()
    result = await gateway.agenerate_model_request(
        stage=LLMCallStage.TOOL_DECISION,
        request=_canonical_request(),
        provider="openai-compatible",
        supports_native_tools=True,
    )
    assert result.turn.tool_calls == []
    assert result.usage.output_tokens == len("read_file") + len('{"path":')


@pytest.mark.anyio
async def test_public_session_settles_estimated_tool_and_reasoning_output_in_durable_budget(tmp_path, monkeypatch):
    from agent_runtime.harness import Session
    from agent_runtime.modeling.contracts import LLMStageBudget
    from agent_runtime.modeling.gateway import LLMGateway
    from tests.agent.harness.test_model_adapter import (
        AcceptAnswer,
        BoundGatewayHarnessModel,
        CharacterAccounting,
        _resolved_model,
    )
    from tests.provider.test_llm_gateway import _canonical_tool

    class Generator:
        def __init__(self):
            self.calls = 0

        def generate_with_tools(self, **kwargs):
            self.calls += 1
            if self.calls == 1:
                return LLMProviderResult(
                    {
                        "choices": [
                            {
                                "finish_reason": "tool_calls",
                                "message": {
                                    "content": None,
                                    "reasoning_content": "reasoned thought",
                                    "tool_calls": [
                                        {"id": "read", "function": {"name": "read_file", "arguments": "{}"}}
                                    ],
                                },
                            }
                        ]
                    }
                )
            return LLMProviderResult(
                {"choices": [{"finish_reason": "stop", "message": {"content": "real gateway answer"}}]}
            )

    monkeypatch.setenv("PRAXIS_CONFIG_DIR", str(tmp_path / "user-config"))
    accounting = CharacterAccounting()
    gateway = LLMGateway(
        generator=Generator(),
        token_accounting=accounting,
        model_context_tokens=100000,
        stage_budgets={LLMCallStage.AGENT_STEP: LLMStageBudget(max_input_tokens=80000, max_output_tokens=256)},
    )
    model = BoundGatewayHarnessModel(
        model_id="test",
        instructions=("Use the requested tool then answer.",),
        resolved=_resolved_model(gateway=gateway, context_window_tokens=100000, token_accounting=accounting),
    )
    async with await Session.open(
        database=tmp_path / "r.db",
        workspace=tmp_path,
        model=model,
        tools={"read_file": _canonical_tool("read_file")},
        completion_gate=AcceptAnswer(),
    ) as session:
        result = await session.submit("read the available file")
        assert result.status == "done"
        budget = session.store.read_budget_state(result.turn_id)
        assert budget.used.output_tokens == len("read_file") + len("{}") + len("reasoned thought") + len(
            "real gateway answer"
        )
        assert budget.used.model_calls == 2
        operations = session.store.list_model_operations(result.turn_id)
        assert len(operations) == 2
        assert all(
            attempt.usage["source"] == "tokenizer_estimate"
            for operation in operations
            for attempt in session.store.list_model_attempts(operation.operation_id)
        )
