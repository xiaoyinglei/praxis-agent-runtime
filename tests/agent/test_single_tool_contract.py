from __future__ import annotations

from collections.abc import Callable
from dataclasses import FrozenInstanceError, replace

import pytest

from agent_runtime.tools.tool import (
    ArtifactReference,
    CancellationMode,
    InterruptBehavior,
    NormalizedToolOutput,
    ResolvedToolUse,
    Tool,
    ToolCall,
    ToolCallOrigin,
    ToolContentBlock,
    ToolDefinition,
    ToolEffect,
    ToolResult,
    ToolTarget,
)


def _normalize_output(output: object) -> NormalizedToolOutput:
    text = str(output)
    return NormalizedToolOutput(
        content=(ToolContentBlock(type="text", data={"text": text}),),
        structured_content={"output": text},
    )


def _origin() -> ToolCallOrigin:
    return ToolCallOrigin(
        request_id="request-1",
        toolset_revision="toolset-v3",
        exposed_tool_names=("read_text",),
    )


def _tool(**changes: object) -> Tool:
    tool = Tool(
        definition=ToolDefinition(
            name="read_text",
            description="Read text from the workspace.",
            input_schema={
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
            },
        ),
        validate_input=lambda arguments: arguments,
        run=lambda arguments: {"text": arguments["path"]},
        normalize_output=_normalize_output,
        output_schema={
            "type": "object",
            "properties": {"text": {"type": "string"}},
        },
        static_effects=frozenset({ToolEffect.READ_WORKSPACE}),
        resolve_use=lambda _arguments: ResolvedToolUse(effects=frozenset(), targets=()),
        execution_revision="read-text-v1",
        idempotent=True,
        concurrency_safe=True,
        cancellation_mode=CancellationMode.COOPERATIVE,
        interrupt_behavior=InterruptBehavior.CANCEL,
        timeout_seconds=10.0,
        max_model_output_bytes=4096,
    )
    return replace(tool, **changes)


def test_tool_projects_definition_without_runner() -> None:
    schema = {"type": "object", "required": ["path"]}
    tool = _tool(
        definition=ToolDefinition(
            name="read_text",
            description="Read text from the workspace.",
            input_schema=schema,
        )
    )

    schema["required"].append("mutated")

    assert tool.definition.name == "read_text"
    assert tool.definition.input_schema["required"] == ("path",)
    assert not hasattr(tool.definition, "run")
    with pytest.raises(FrozenInstanceError):
        tool.definition.name = "other"  # type: ignore[misc]


def test_tool_rejects_local_side_effect_with_non_cancellable_mode() -> None:
    with pytest.raises(ValueError, match="local side-effecting"):
        _tool(
            static_effects=frozenset({ToolEffect.WRITE_WORKSPACE}),
            cancellation_mode=CancellationMode.NOT_CANCELLABLE,
            interrupt_behavior=InterruptBehavior.FINISH_CURRENT,
        )


def test_tool_call_origin_preserves_checkpoint_evidence() -> None:
    origin = ToolCallOrigin(
        request_id="request-1",
        toolset_revision="toolset-v3",
        exposed_tool_names=("read_text", "list_files"),
    )
    call = ToolCall(
        tool_call_id="call-1",
        tool_name="read_text",
        arguments={"path": "notes.txt"},
        origin=origin,
    )

    assert origin.request_id == "request-1"
    assert origin.toolset_revision == "toolset-v3"
    assert origin.exposed_tool_names == ("read_text", "list_files")
    assert call.origin == origin
    with pytest.raises(FrozenInstanceError):
        origin.request_id = "other"  # type: ignore[misc]


@pytest.mark.parametrize("field_name", ["request_id", "toolset_revision"])
def test_tool_call_origin_rejects_mutable_scalar_evidence(field_name: str) -> None:
    values: dict[str, object] = {
        "request_id": "request-1",
        "toolset_revision": "toolset-v3",
        "exposed_tool_names": (),
    }
    values[field_name] = ["mutable"]

    with pytest.raises(TypeError, match=field_name):
        ToolCallOrigin(**values)  # type: ignore[arg-type]


def test_tool_call_origin_freezes_exposed_names_without_splitting_strings() -> None:
    exposed_names = ["read_text"]
    origin = ToolCallOrigin(
        request_id="request-1",
        toolset_revision="toolset-v3",
        exposed_tool_names=exposed_names,  # type: ignore[arg-type]
    )

    exposed_names.append("mutated")

    assert origin.exposed_tool_names == ("read_text",)
    with pytest.raises(TypeError, match="exposed_tool_names"):
        ToolCallOrigin(
            request_id="request-1",
            toolset_revision="toolset-v3",
            exposed_tool_names="read_text",  # type: ignore[arg-type]
        )


def test_tool_call_rejects_non_origin_record() -> None:
    with pytest.raises(TypeError, match="origin"):
        ToolCall(
            tool_call_id="call-1",
            tool_name="read_text",
            arguments={"path": "notes.txt"},
            origin={"request_id": "request-1"},  # type: ignore[arg-type]
        )


@pytest.mark.parametrize("field_name", ["artifact_id", "media_type", "name"])
def test_artifact_reference_rejects_mutable_string_fields(field_name: str) -> None:
    values: dict[str, object] = {
        "artifact_id": "artifact-1",
        "media_type": "text/plain",
        "name": "result.txt",
    }
    values[field_name] = ["mutable"]

    with pytest.raises(TypeError, match=field_name):
        ArtifactReference(**values)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "factory",
    [
        pytest.param(
            lambda: ToolDefinition(
                name=["read_text"],  # type: ignore[arg-type]
                description="Read text.",
                input_schema={},
            ),
            id="definition-name",
        ),
        pytest.param(
            lambda: ToolDefinition(
                name="read_text",
                description=["Read text."],  # type: ignore[arg-type]
                input_schema={},
            ),
            id="definition-description",
        ),
        pytest.param(
            lambda: ToolCall(
                tool_call_id=["call-1"],  # type: ignore[arg-type]
                tool_name="read_text",
                arguments={},
                origin=_origin(),
            ),
            id="call-id",
        ),
        pytest.param(
            lambda: ToolCall(
                tool_call_id="call-1",
                tool_name=["read_text"],  # type: ignore[arg-type]
                arguments={},
                origin=_origin(),
            ),
            id="call-name",
        ),
        pytest.param(
            lambda: ToolResult(
                tool_call_id=["call-1"],  # type: ignore[arg-type]
                tool_name="read_text",
            ),
            id="result-id",
        ),
        pytest.param(
            lambda: ToolResult(
                tool_call_id="call-1",
                tool_name=["read_text"],  # type: ignore[arg-type]
            ),
            id="result-name",
        ),
        pytest.param(
            lambda: _tool(execution_revision=["read-text-v1"]),
            id="execution-revision",
        ),
    ],
)
def test_required_contract_strings_reject_non_string_values(
    factory: Callable[[], object],
) -> None:
    with pytest.raises(TypeError):
        factory()


def test_tool_content_blocks_support_extensible_json_payloads() -> None:
    blocks = (
        ToolContentBlock(type="text", data={"text": "first"}),
        ToolContentBlock(type="image", data={"artifact_id": "image-1"}),
        ToolContentBlock(type="resource", data={"uri": "artifact://result-1"}),
    )

    assert [block.type for block in blocks] == ["text", "image", "resource"]
    assert blocks[1].data == {"artifact_id": "image-1"}
    with pytest.raises(TypeError, match="JSON-compatible"):
        ToolContentBlock(type="image", data={"value": object()})


def test_tool_result_metadata_is_not_model_content() -> None:
    result = ToolResult(
        tool_call_id="call-1",
        tool_name="read_text",
        content=(
            ToolContentBlock(type="text", data={"text": "first"}),
            ToolContentBlock(type="text", data={"text": "second"}),
        ),
        structured_content={"text": "first\nsecond"},
        is_error=False,
        error_code=None,
        error_message=None,
        truncated=False,
        metadata={"trace_id": "runtime-only"},
        attachments=(
            ArtifactReference(
                artifact_id="artifact-1",
                media_type="text/plain",
                name="result.txt",
            ),
        ),
    )

    assert [block.data["text"] for block in result.content] == ["first", "second"]
    assert result.metadata == {"trace_id": "runtime-only"}
    assert all("runtime-only" not in repr(block.data) for block in result.content)
    assert not hasattr(result, "model_content")
    assert result.attachments[0].artifact_id == "artifact-1"


def test_tool_rejects_string_cancellation_mode_before_effect_comparison() -> None:
    with pytest.raises(TypeError, match="cancellation_mode"):
        _tool(
            static_effects=frozenset({ToolEffect.WRITE_WORKSPACE}),
            cancellation_mode="not_cancellable",
        )


def test_tool_rejects_invalid_interrupt_behavior() -> None:
    with pytest.raises(TypeError, match="interrupt_behavior"):
        _tool(interrupt_behavior="cancel")


@pytest.mark.parametrize("field_name", ["idempotent", "concurrency_safe"])
def test_tool_requires_exact_boolean_flags(field_name: str) -> None:
    with pytest.raises(TypeError, match=field_name):
        _tool(**{field_name: 1})


@pytest.mark.parametrize(
    ("value", "error_type"),
    [
        pytest.param(True, TypeError, id="bool"),
        pytest.param("10", TypeError, id="string"),
        pytest.param(float("inf"), ValueError, id="infinite"),
        pytest.param(float("nan"), ValueError, id="nan"),
    ],
)
def test_tool_requires_positive_finite_real_timeout(
    value: object,
    error_type: type[Exception],
) -> None:
    with pytest.raises(error_type, match="timeout_seconds"):
        _tool(timeout_seconds=value)


@pytest.mark.parametrize("field_name", ["is_error", "retryable", "truncated"])
def test_tool_result_requires_exact_boolean_flags(field_name: str) -> None:
    values: dict[str, object] = {
        "tool_call_id": "call-1",
        "tool_name": "read_text",
        "is_error": False,
        "retryable": False,
        "truncated": False,
    }
    values[field_name] = 1
    if field_name in {"is_error", "retryable"}:
        values["is_error"] = 1 if field_name == "is_error" else True
        values["error_code"] = "tool_error"

    with pytest.raises(TypeError, match=field_name):
        ToolResult(**values)  # type: ignore[arg-type]


@pytest.mark.parametrize("field_name", ["error_code", "error_message"])
def test_tool_result_requires_string_error_fields(field_name: str) -> None:
    values: dict[str, object] = {
        "tool_call_id": "call-1",
        "tool_name": "read_text",
        "is_error": True,
        "error_code": "tool_error",
        "error_message": "failed",
    }
    values[field_name] = ["mutable"]

    with pytest.raises(TypeError, match=field_name):
        ToolResult(**values)  # type: ignore[arg-type]


@pytest.mark.parametrize("field_name", ["kind", "value"])
def test_tool_target_requires_non_empty_strings(field_name: str) -> None:
    values = {"kind": "workspace_path", "value": "notes.txt"}
    values[field_name] = ""

    with pytest.raises(ValueError, match=field_name):
        ToolTarget(**values)


def test_resolved_tool_use_freezes_effects_and_targets() -> None:
    effects = {ToolEffect.READ_WORKSPACE}
    targets = [ToolTarget(kind="workspace_path", value="notes.txt")]

    resolved = ResolvedToolUse(effects=effects, targets=targets)  # type: ignore[arg-type]
    effects.add(ToolEffect.NETWORK)
    targets.append(ToolTarget(kind="workspace_path", value="other.txt"))

    assert resolved.effects == frozenset({ToolEffect.READ_WORKSPACE})
    assert resolved.targets == (
        ToolTarget(kind="workspace_path", value="notes.txt"),
    )


def test_resolved_tool_use_validates_members() -> None:
    with pytest.raises(TypeError, match="effects"):
        ResolvedToolUse(effects=frozenset({"read_workspace"}), targets=())  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="targets"):
        ResolvedToolUse(effects=frozenset(), targets=("notes.txt",))  # type: ignore[arg-type]


def test_normalized_output_is_identity_free_and_reusable() -> None:
    normalized = _normalize_output("same output")

    first = ToolResult(
        tool_call_id="call-1",
        tool_name="read_text",
        content=normalized.content,
        structured_content=normalized.structured_content,
        is_error=normalized.is_error,
        error_code=normalized.error_code,
        error_message=normalized.error_message,
        retryable=normalized.retryable,
        metadata=normalized.metadata,
        attachments=normalized.attachments,
    )
    second = ToolResult(
        tool_call_id="call-2",
        tool_name="read_text",
        content=normalized.content,
        structured_content=normalized.structured_content,
        is_error=normalized.is_error,
        error_code=normalized.error_code,
        error_message=normalized.error_message,
        retryable=normalized.retryable,
        metadata=normalized.metadata,
        attachments=normalized.attachments,
    )

    assert not hasattr(normalized, "tool_call_id")
    assert not hasattr(normalized, "tool_name")
    assert first.tool_call_id == "call-1"
    assert second.tool_call_id == "call-2"
    assert first.content == second.content == normalized.content


def test_normalized_output_freezes_canonical_payload() -> None:
    content = [ToolContentBlock(type="text", data={"text": "result"})]
    structured_content = {"items": ["one"]}
    metadata = {"adapter": "fixture"}
    attachments = [ArtifactReference(artifact_id="artifact-1")]

    normalized = NormalizedToolOutput(
        content=content,  # type: ignore[arg-type]
        structured_content=structured_content,
        metadata=metadata,
        attachments=attachments,  # type: ignore[arg-type]
    )
    content.append(ToolContentBlock(type="text", data={"text": "mutated"}))
    structured_content["items"].append("mutated")
    metadata["adapter"] = "mutated"
    attachments.append(ArtifactReference(artifact_id="artifact-2"))

    assert len(normalized.content) == 1
    assert normalized.structured_content == {"items": ("one",)}
    assert normalized.metadata == {"adapter": "fixture"}
    assert len(normalized.attachments) == 1


def test_normalized_output_validates_members_and_error_semantics() -> None:
    with pytest.raises(TypeError, match="content"):
        NormalizedToolOutput(content=("text",))  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="attachments"):
        NormalizedToolOutput(attachments=(object(),))  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="error_code"):
        NormalizedToolOutput(is_error=True)
