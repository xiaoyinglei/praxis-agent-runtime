"""protocol.tools 瘦身 Tool ACI 与 protocol.json freeze/thaw 回归测试(设计 §5.1)。"""

from __future__ import annotations

import math

import pytest
from hypothesis import given
from hypothesis import strategies as st

from praxis.protocol.json import freeze_json, thaw_json
from praxis.protocol.tools import (
    CancellationMode,
    InterruptBehavior,
    ResolvedUse,
    ResultStatus,
    Tool,
    ToolCall,
    ToolContentBlock,
    ToolDefinition,
    ToolEffect,
    ToolResult,
    ToolTarget,
    ToolValidationError,
    args_digest,
    json_schema_input,
)

# ---------------------------------------------------------------------------
# protocol.json:freeze/thaw


@given(
    value=st.recursive(
        st.none()
        | st.booleans()
        | st.integers(min_value=-(10**6), max_value=10**6)
        | st.floats(allow_nan=False, allow_infinity=False)
        | st.text(max_size=8),
        lambda children: st.lists(children, max_size=3)
        | st.dictionaries(st.text(max_size=5), children, max_size=3),
        max_leaves=15,
    )
)
def test_freeze_thaw_roundtrip(value: object) -> None:
    assert thaw_json(freeze_json(value, path="$")) == value


def test_freeze_rejects_non_finite_floats() -> None:
    for bad in (math.nan, math.inf, -math.inf):
        with pytest.raises(ValueError):
            freeze_json(bad, path="$")
    with pytest.raises(ValueError):
        freeze_json({"x": [math.inf]}, path="$")


def test_freeze_rejects_non_json_values() -> None:
    with pytest.raises(TypeError):
        freeze_json({"x": object()}, path="$")
    with pytest.raises(TypeError):
        freeze_json({1: "x"}, path="$")


def test_freeze_produces_immutable_containers() -> None:
    frozen = freeze_json({"a": [1, {"b": 2}]}, path="$")
    with pytest.raises(TypeError):
        frozen["a"] = 3  # type: ignore[index]


# ---------------------------------------------------------------------------
# args_digest


def test_args_digest_stable_under_key_order() -> None:
    assert args_digest({"a": 1, "b": {"c": 2}}) == args_digest({"b": {"c": 2}, "a": 1})


def test_args_digest_differs_for_different_values() -> None:
    assert args_digest({"a": 1}) != args_digest({"a": 2})


# ---------------------------------------------------------------------------
# ToolResult:status × is_error 一致性(移植旧错误字段检查 + 新 status 规则)


def test_tool_result_error_fields_require_is_error() -> None:
    with pytest.raises(ValueError):
        ToolResult(tool_call_id="c", tool_name="read", error_code="boom")


@pytest.mark.parametrize(
    "status",
    [
        ResultStatus.ERROR,
        ResultStatus.DENIED,
        ResultStatus.TIMEOUT,
        ResultStatus.CANCELLED,
    ],
)
def test_tool_result_error_like_status_requires_is_error(status: ResultStatus) -> None:
    with pytest.raises(ValueError):
        ToolResult(tool_call_id="c", tool_name="read", status=status)
    result = ToolResult(tool_call_id="c", tool_name="read", status=status, is_error=True, error_code="x")
    assert result.is_error


def test_tool_result_ok_status_forbids_error_flag_and_fields() -> None:
    with pytest.raises(ValueError):
        ToolResult(tool_call_id="c", tool_name="read", status=ResultStatus.OK, is_error=True, error_code="x")
    with pytest.raises(ValueError):
        ToolResult(tool_call_id="c", tool_name="read", status=ResultStatus.OK, error_code="x")


@pytest.mark.parametrize(
    "status",
    [ResultStatus.NOT_EXECUTED, ResultStatus.OUTCOME_UNKNOWN, ResultStatus.PAUSED],
)
def test_tool_result_neutral_status_has_no_is_error_constraint(status: ResultStatus) -> None:
    ToolResult(tool_call_id="c", tool_name="read", status=status)
    ToolResult(tool_call_id="c", tool_name="read", status=status, is_error=True, error_code="halted")


def test_tool_result_text_joins_text_blocks() -> None:
    result = ToolResult(
        tool_call_id="c",
        tool_name="read",
        content=(
            ToolContentBlock(type="text", data={"text": "a"}),
            ToolContentBlock(type="image", data={"media_type": "image/png"}),
            ToolContentBlock(type="text", data={"text": "b"}),
        ),
    )
    assert result.text == "a\nb"


def test_tool_result_content_must_be_blocks() -> None:
    with pytest.raises(TypeError):
        ToolResult(tool_call_id="c", tool_name="read", content=({"text": "x"},))  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# ToolCall / ToolTarget / ResolvedUse


def test_tool_call_freezes_arguments() -> None:
    call = ToolCall(tool_call_id="c", tool_name="read", arguments={"path": "x"})
    with pytest.raises(TypeError):
        call.arguments["path"] = "y"  # type: ignore[index]


def test_tool_target_requires_non_empty_strings() -> None:
    with pytest.raises(ValueError):
        ToolTarget(kind="", value="x")
    with pytest.raises(ValueError):
        ToolTarget(kind="path", value="")


def test_resolved_use_coerces_sequences() -> None:
    use = ResolvedUse(
        effects=[ToolEffect.READ_WORKSPACE],  # type: ignore[list-item]
        targets=[ToolTarget(kind="path", value="a")],  # type: ignore[list-item]
    )
    assert isinstance(use.effects, frozenset)
    assert isinstance(use.targets, tuple)


# ---------------------------------------------------------------------------
# json_schema_input(移植)


def test_json_schema_input_rejects_extra_and_reports_path() -> None:
    validate = json_schema_input(
        {
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
            "additionalProperties": False,
        }
    )
    assert validate({"path": "x"}) == {"path": "x"}
    with pytest.raises(ToolValidationError) as ei:
        validate({"path": "x", "evil": 1})
    assert "evil" in ei.value.path


def test_json_schema_input_reports_missing_required_path() -> None:
    validate = json_schema_input(
        {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}
    )
    with pytest.raises(ToolValidationError) as ei:
        validate({})
    assert "path" in ei.value.path


def test_json_schema_input_canonicalizes_nested_arguments() -> None:
    validate = json_schema_input(
        {
            "type": "object",
            "properties": {
                "items": {"type": "array", "items": {"type": "integer"}},
            },
        }
    )
    canonical = validate({"items": [1, 2]})
    assert canonical == {"items": (1, 2)}


def test_json_schema_input_rejects_non_finite_input() -> None:
    validate = json_schema_input({"type": "object"})
    with pytest.raises(ToolValidationError):
        validate({"x": math.nan})


# ---------------------------------------------------------------------------
# Tool:移植旧约束(本地副作用工具不得 NOT_CANCELLABLE)


def _make_tool(**overrides: object) -> Tool:
    definition = ToolDefinition(name="demo", description="demo tool", input_schema={"type": "object"})
    kwargs: dict[str, object] = dict(
        definition=definition,
        validate_input=lambda arguments: arguments,
        run=lambda arguments: ToolResult(tool_call_id="c", tool_name="demo"),
        static_effects=frozenset({ToolEffect.READ_WORKSPACE}),
        resolve_use=lambda arguments: ResolvedUse(effects=frozenset({ToolEffect.READ_WORKSPACE}), targets=()),
        idempotent=True,
        concurrency_safe=True,
        cancellation_mode=CancellationMode.COOPERATIVE,
        interrupt_behavior=InterruptBehavior.CANCEL,
        timeout_seconds=30.0,
        max_model_output_bytes=4096,
    )
    kwargs.update(overrides)
    return Tool(**kwargs)


def test_tool_rejects_not_cancellable_side_effects() -> None:
    for effect in (ToolEffect.WRITE_WORKSPACE, ToolEffect.EXECUTE_PROCESS, ToolEffect.DESTRUCTIVE):
        with pytest.raises(ValueError):
            _make_tool(
                static_effects=frozenset({effect}),
                cancellation_mode=CancellationMode.NOT_CANCELLABLE,
            )
    # 只读/网络工具不受该约束
    _make_tool(
        static_effects=frozenset({ToolEffect.NETWORK}),
        cancellation_mode=CancellationMode.NOT_CANCELLABLE,
    )


def test_tool_validates_basic_fields() -> None:
    with pytest.raises(TypeError):
        _make_tool(definition="not-a-definition")
    with pytest.raises(ValueError):
        _make_tool(timeout_seconds=0)
    with pytest.raises(ValueError):
        _make_tool(max_model_output_bytes=0)
    with pytest.raises(TypeError):
        _make_tool(idempotent=1)
