"""protocol.models 流式事件与唯一装配点(设计 §5.9;D1/D2;B9/B24 前置)。

装配是流式/非流式共用的**唯一**实现:``complete`` 只是
``assemble_model_events(await collect_events(stream(req)))``。这里钉住畸形
provider 响应(分片 name、非 JSON/非 object arguments、缺 finish)的装配语义:
不抛穿 turn,坏调用丢弃并留下有界提示。
"""

from __future__ import annotations

import dataclasses

import pytest

from praxis.adapters.models import tokenization
from praxis.adapters.models.openai_compat import OpenAICompatAdapter
from praxis.adapters.models.tokenization import count_text_tokens, measure_request
from praxis.protocol.models import (
    CostMicros,
    ModelMessage,
    ModelReasoningDelta,
    ModelSettings,
    ModelSpec,
    ModelStreamFinished,
    ModelTextDelta,
    ModelToolCall,
    ModelToolCallDelta,
    Usage,
    assemble_model_events,
)
from praxis.protocol.tools import ToolDefinition


def test_assemble_accumulates_tool_name_fragments() -> None:
    events = [
        ModelToolCallDelta(index=0, call_id="c0", name_fragment="read"),
        ModelToolCallDelta(
            index=0, call_id=None, name_fragment="_file", arguments_fragment='{"pa'
        ),
        ModelToolCallDelta(index=0, call_id=None, arguments_fragment='th":"a"}'),
        ModelStreamFinished(Usage(3, 4), "tool_calls"),
    ]
    resp = assemble_model_events(events)
    assert [(c.call_id, c.name, dict(c.arguments)) for c in resp.tool_calls] == [
        ("c0", "read_file", {"path": "a"})
    ]
    assert (resp.usage.input_tokens, resp.usage.output_tokens, resp.usage.source) == (
        3,
        4,
        "provider",
    )


def test_assemble_invalid_arguments_drops_without_raising() -> None:
    resp = assemble_model_events(
        [
            ModelToolCallDelta(0, "c0", "bash", "not-json"),
            ModelStreamFinished(Usage(1, 1), "tool_calls"),
        ]
    )
    assert resp.tool_calls == () and "invalid tool call arguments" in resp.content


def test_assemble_without_finish_is_estimated_zero_usage() -> None:
    usage = assemble_model_events([ModelTextDelta("hi")]).usage
    assert usage.source == "estimated"
    assert (usage.input_tokens, usage.output_tokens) == (0, 0)


def test_spec_validation_and_wire_model_id() -> None:
    spec = ModelSpec(
        model_id="alias",
        remote_model_id="real/deepseek-v3",
        context_window_tokens=8192,
        max_output_tokens=4096,
        api_key_env="DEEPSEEK_API_KEY",
    )
    assert spec.wire_model_id == "real/deepseek-v3" and spec.cost.is_zero
    with pytest.raises(ValueError):
        ModelSpec(model_id="m", context_window_tokens=100, max_output_tokens=200)
    with pytest.raises(ValueError):
        ModelSpec(model_id="m", api_key_env="bad-name!")


def test_assemble_keeps_reasoning_separate_from_content() -> None:
    # 思考文本单独累积:content 只含正文,绝不混拼(会作为助手历史回灌)。
    resp = assemble_model_events(
        [
            ModelReasoningDelta("think "),
            ModelTextDelta("hi"),
            ModelReasoningDelta("more"),
            ModelStreamFinished(Usage(1, 2), "stop"),
        ]
    )
    assert resp.content == "hi"
    assert resp.reasoning == "think more"
    assert resp.tool_calls == () and resp.finish_reason == "stop"


def test_assemble_reasoning_only_leaves_content_empty() -> None:
    resp = assemble_model_events(
        [
            ModelReasoningDelta("only thinking"),
            ModelStreamFinished(Usage(1, 2), "stop"),
        ]
    )
    assert resp.content == "" and resp.reasoning == "only thinking"


def test_assemble_keeps_first_call_id_and_orders_calls_by_index() -> None:
    events = [
        ModelToolCallDelta(index=1, call_id="c1", name_fragment="b", arguments_fragment="{}"),
        ModelToolCallDelta(index=0, call_id=None, name_fragment="a", arguments_fragment="{}"),
        ModelToolCallDelta(index=0, call_id="c0"),
        ModelStreamFinished(Usage(1, 1), "tool_calls"),
    ]
    resp = assemble_model_events(events)
    assert [c.call_id for c in resp.tool_calls] == ["c0", "c1"]
    assert [c.name for c in resp.tool_calls] == ["a", "b"]


def test_assemble_non_object_arguments_are_dropped_with_bounded_hint() -> None:
    resp = assemble_model_events(
        [
            ModelToolCallDelta(0, "c0", "bash", "[1, 2]"),
            ModelStreamFinished(Usage(1, 1), "tool_calls"),
        ]
    )
    assert resp.tool_calls == ()
    assert "[invalid tool call arguments dropped: 1]" in resp.content


def test_model_events_are_frozen() -> None:
    with pytest.raises(dataclasses.FrozenInstanceError):
        ModelTextDelta("x").text = "y"  # type: ignore[misc]


def test_usage_total_tokens_and_cost_micros_validation() -> None:
    assert Usage(3, 4).total_tokens == 7
    assert Usage(3, 4).source == "provider"
    assert CostMicros().is_zero and not CostMicros(output_per_1m=1).is_zero
    with pytest.raises(ValueError):
        CostMicros(input_per_1m=-1)


# ---------------------------------------------------------------------------
# 测量口径(B24):只测 build_request 一次定型的出站文本,没有第二个口径
# ---------------------------------------------------------------------------


def test_measure_request_uses_serialized_json() -> None:
    adapter = OpenAICompatAdapter(ModelSpec(model_id="m"), api_key="k")
    call = ModelToolCall(call_id="c1", name="read_file", arguments={"path": "a.txt"})
    tool = ToolDefinition(name="read_file", description="read", input_schema={"type": "object"})
    req = adapter.build_request(
        [ModelMessage(role="assistant", content="", tool_calls=(call,))],
        [tool],
        ModelSettings(),
    )
    measurement = measure_request(req)
    assert measurement == count_text_tokens(
        req.serialized_json, tokenizer_model=req.tokenizer_model
    )
    assert measurement.count > 0
    assert '"tool_calls"' in req.serialized_json


def test_tiktoken_failure_falls_back_single_entry(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(tokenizer_model: str | None) -> object:
        raise RuntimeError("offline")

    monkeypatch.setattr(tokenization, "_get_encoder", boom)
    measurement = count_text_tokens("a" * 400)
    assert measurement.source == "heuristic"
    assert measurement.count == 400 // 4 + 1
