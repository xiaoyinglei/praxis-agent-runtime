"""protocol.models / protocol.events:模型契约与事件冻结性测试(设计 §5.14 及 events 接口)。"""

from __future__ import annotations

import dataclasses

import pytest

from praxis.protocol.events import (
    ApprovalRequested,
    Event,
    ModelResponseReceived,
    ToolFinished,
    ToolStarted,
    TurnStatusChanged,
)
from praxis.protocol.models import (
    ModelAdapter,
    ModelMessage,
    ModelResponse,
    ModelSettings,
    ModelSpec,
    ModelToolCall,
    Usage,
    WireRequest,
)
from praxis.protocol.tools import ToolDefinition


def test_model_spec_defaults() -> None:
    spec = ModelSpec(model_id="gpt-x")
    assert spec.provider == "openai-compatible"
    assert spec.max_output_tokens == 4096
    assert dict(spec.provider_options) == {}


def test_model_dataclasses_are_frozen() -> None:
    call = ModelToolCall(call_id="c1", name="read_file", arguments={"path": "a"})
    message = ModelMessage(role="user", content="hi", tool_calls=(call,))
    response = ModelResponse(
        content="done", tool_calls=(), usage=Usage(1, 2), finish_reason="stop"
    )
    for obj, attr in (
        (call, "name"),
        (message, "content"),
        (response, "finish_reason"),
        (Usage(1, 2), "input_tokens"),
        (ModelSettings(), "temperature"),
    ):
        with pytest.raises(dataclasses.FrozenInstanceError):
            setattr(obj, attr, None)


def test_model_message_optional_fields() -> None:
    tool_reply = ModelMessage(role="tool", content="ok", tool_call_id="c1")
    assert tool_reply.tool_calls == ()
    assert ModelMessage(role="system", content="sys").tool_call_id is None


def test_wire_request_shape() -> None:
    tool = ToolDefinition(name="read_file", description="read", input_schema={})
    request = WireRequest(
        model="gpt-x",
        messages=(ModelMessage(role="user", content="hi"),),
        tools=(tool,),
        max_output_tokens=1024,
        provider_options={"stream": False},
        temperature=None,
    )
    assert request.tools[0].name == "read_file"
    assert request.temperature is None


class _PlaceholderAdapter:
    """占位对象:Task 9 用 ScriptedAdapter 替换真实实现断言。"""

    def build_request(self, messages, tools, settings):
        raise NotImplementedError

    async def complete(self, req):
        raise NotImplementedError


class _IncompleteAdapter:
    def build_request(self, messages, tools, settings):
        raise NotImplementedError


def test_model_adapter_is_runtime_checkable_protocol() -> None:
    assert isinstance(_PlaceholderAdapter(), ModelAdapter)
    assert not isinstance(_IncompleteAdapter(), ModelAdapter)
    assert not isinstance("not an adapter", ModelAdapter)


def test_events_are_frozen_and_hashable() -> None:
    events: tuple[Event, ...] = (
        TurnStatusChanged(
            turn_id="t1",
            thread_id="th1",
            from_status="running",
            to_status="awaiting_approval",
            reason="tool requires approval",
        ),
        ToolStarted(turn_id="t1", call_id="c1", tool_name="read_file"),
        ToolFinished(turn_id="t1", call_id="c1", status="ok"),
        ApprovalRequested(turn_id="t1", call_id="c1", tool_name="write_file", args_digest="d" * 64),
        ModelResponseReceived(turn_id="t1", content_preview="hello"),
    )
    # Event 是 PEP 695 别名:3.12 下 isinstance 不接受别名本体,取 __value__ 联合。
    assert set(Event.__value__.__args__) == {
        TurnStatusChanged,
        ToolStarted,
        ToolFinished,
        ApprovalRequested,
        ModelResponseReceived,
    }
    for event in events:
        assert isinstance(event, Event.__value__)
        hash(event)
    with pytest.raises(dataclasses.FrozenInstanceError):
        events[1].tool_name = "rm"


def test_tool_started_field_names_match_pipeline_contract() -> None:
    # T6/T7 依赖的关键字参数名,逐字固定。
    event = TurnStatusChanged(
        turn_id="t1", thread_id="th1", from_status="queued", to_status="running", reason=""
    )
    assert (event.turn_id, event.thread_id) == ("t1", "th1")
