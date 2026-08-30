from __future__ import annotations

import hashlib
import os
import re
import shutil
from collections.abc import Awaitable, Callable, Mapping
from pathlib import Path
from typing import Literal, cast

from pydantic import BaseModel, ConfigDict, Field

from agent_runtime.tools.tool import (
    CancellationMode,
    InterruptBehavior,
    JsonValue,
    NormalizedToolOutput,
    ResolvedToolUse,
    Tool,
    ToolDefinition,
    ToolEffect,
    ToolTarget,
    ToolValidationError,
    _thaw_json,
    json_schema_output,
    pydantic_input,
)
from agent_runtime.workspace import WorkspaceRuntime

MAX_SKILL_INSTRUCTIONS_CHARS = 40_000

type SkillInvoker = Callable[
    [Mapping[str, JsonValue]],
    object | Awaitable[object],
]
type ActiveSkillRoot = Callable[[str], Path | str | None]


class InvokeSkillInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(
        min_length=1,
        max_length=500,
        description="Exact skill id or unambiguous listed skill name.",
    )
    args: str | None = Field(
        default=None,
        max_length=4000,
        description="Optional value substituted for the skill's arguments placeholder.",
    )


class SkillActivationEvent(BaseModel):
    model_config = ConfigDict(extra="forbid")

    event_type: Literal["skill_activation"] = "skill_activation"
    success: bool
    name: str = Field(max_length=500)
    skill_id: str = Field(default="", max_length=500)
    source: str = Field(default="", max_length=100)
    fingerprint: str = Field(default="", max_length=128)
    instructions: str = Field(default="", max_length=MAX_SKILL_INSTRUCTIONS_CHARS)
    args: str | None = Field(default=None, max_length=4000)
    truncated: bool = False
    error_code: str | None = Field(default=None, max_length=200)
    error_message: str | None = Field(default=None, max_length=2000)


class MaterializeSkillAssetInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    skill_id: str = Field(
        min_length=1,
        max_length=500,
        description="Unique id of an already active skill.",
    )
    relative_path: str = Field(
        min_length=1,
        max_length=4096,
        description="Skill-local file under scripts/ or references/.",
    )


class MaterializeSkillAssetOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    workspace_path: str
    source_fingerprint: str
    size_bytes: int = Field(ge=0)


_INVOKE_INPUT_SCHEMA, _validate_invoke_input = pydantic_input(InvokeSkillInput)
_ACTIVATION_SCHEMA, _unused_activation_validator = pydantic_input(
    SkillActivationEvent
)
_ASSET_INPUT_SCHEMA, _validate_asset_model = pydantic_input(
    MaterializeSkillAssetInput
)
_ASSET_OUTPUT_SCHEMA, _unused_asset_output_validator = pydantic_input(
    MaterializeSkillAssetOutput
)


def create_invoke_skill_tool(
    invoke_skill: SkillInvoker,
    *,
    execution_revision: str = "catalog-v1",
    available_skill_ids: tuple[str, ...] = (),
) -> Tool:
    if not callable(invoke_skill):
        raise TypeError("invoke_skill must be callable")
    if not isinstance(execution_revision, str) or not execution_revision:
        raise ValueError("execution_revision must be non-empty")
    if any(not isinstance(skill_id, str) or not skill_id for skill_id in available_skill_ids):
        raise ValueError("available_skill_ids must contain non-empty strings")
    if len(set(available_skill_ids)) != len(available_skill_ids):
        raise ValueError("available_skill_ids must be unique")
    input_schema = _thaw_json(_INVOKE_INPUT_SCHEMA)
    if not isinstance(input_schema, dict):
        raise RuntimeError("invoke_skill input schema must be an object")
    if available_skill_ids:
        properties = input_schema.get("properties")
        if not isinstance(properties, dict):
            raise RuntimeError("invoke_skill schema properties must be an object")
        name_schema = properties.get("name")
        if not isinstance(name_schema, dict):
            raise RuntimeError("invoke_skill name schema must be an object")
        name_schema["enum"] = list(available_skill_ids)
    return Tool(
        definition=ToolDefinition(
            name="invoke_skill",
            description=(
                "Load one explicitly listed skill and emit a bounded canonical "
                "activation event containing its instructions. Invoke a matching skill "
                "before following that workflow; never guess an unlisted skill id."
            ),
            input_schema=cast(Mapping[str, JsonValue], input_schema),
        ),
        validate_input=_validate_invoke_input,
        run=invoke_skill,
        normalize_output=_normalize_skill_activation,
        output_schema=_ACTIVATION_SCHEMA,
        static_effects=frozenset(),
        resolve_use=lambda arguments: ResolvedToolUse(
            effects=frozenset(),
            targets=(ToolTarget(kind="skill", value=str(arguments["name"])),),
        ),
        execution_revision=f"integration-invoke-skill-v1:{execution_revision}",
        idempotent=True,
        concurrency_safe=False,
        cancellation_mode=CancellationMode.COOPERATIVE,
        interrupt_behavior=InterruptBehavior.CANCEL,
        timeout_seconds=5.0,
        max_model_output_bytes=300_000,
    )


def create_materialize_skill_asset_tool(
    workspace: WorkspaceRuntime,
    *,
    active_skill_root: ActiveSkillRoot,
) -> Tool:
    if not callable(active_skill_root):
        raise TypeError("active_skill_root must be callable")
    return Tool(
        definition=ToolDefinition(
            name="materialize_skill_asset",
            description=(
                "Copy one scripts/ or references/ file from an already activated skill "
                "into workspace scratch. Use the returned workspace path with file or "
                "command tools; arbitrary skill-root and parent traversal is rejected."
            ),
            input_schema=_ASSET_INPUT_SCHEMA,
        ),
        validate_input=_validate_asset_input,
        run=lambda arguments: _materialize_skill_asset(
            workspace,
            active_skill_root,
            MaterializeSkillAssetInput.model_validate(arguments),
        ),
        normalize_output=_normalize_asset_output,
        output_schema=_ASSET_OUTPUT_SCHEMA,
        static_effects=frozenset({ToolEffect.WRITE_WORKSPACE}),
        resolve_use=lambda arguments: ResolvedToolUse(
            effects=frozenset({ToolEffect.WRITE_WORKSPACE}),
            targets=(
                ToolTarget(
                    kind="active_skill",
                    value=str(arguments["skill_id"]),
                ),
                ToolTarget(
                    kind="workspace_path",
                    value=str(
                        _asset_destination(
                            workspace,
                            str(arguments["skill_id"]),
                            _validate_asset_path(str(arguments["relative_path"])),
                        ).resolve()
                    ),
                ),
            ),
        ),
        execution_revision="integration-materialize-skill-asset-v1",
        idempotent=True,
        concurrency_safe=False,
        cancellation_mode=CancellationMode.COOPERATIVE,
        interrupt_behavior=InterruptBehavior.FINISH_CURRENT,
        timeout_seconds=5.0,
        max_model_output_bytes=50_000,
    )


def create_skill_tools(
    workspace: WorkspaceRuntime,
    *,
    invoke_skill: SkillInvoker,
    active_skill_root: ActiveSkillRoot,
    invoke_execution_revision: str = "catalog-v1",
    available_skill_ids: tuple[str, ...] = (),
) -> tuple[Tool, ...]:
    return (
        create_invoke_skill_tool(
            invoke_skill,
            execution_revision=invoke_execution_revision,
            available_skill_ids=available_skill_ids,
        ),
        create_materialize_skill_asset_tool(
            workspace,
            active_skill_root=active_skill_root,
        ),
    )


def _normalize_skill_activation(raw: object) -> NormalizedToolOutput:
    value = _object_mapping(raw)
    instructions_value = value.get(
        "instructions",
        value.get("loaded_content", ""),
    )
    instructions = instructions_value if isinstance(instructions_value, str) else ""
    bounded_instructions = instructions[:MAX_SKILL_INSTRUCTIONS_CHARS]
    success = bool(value.get("success", True))
    error_code_value = value.get("error_code")
    error_code = (
        str(error_code_value)[:200]
        if error_code_value
        else (None if success else "skill_activation_failed")
    )
    error_message_value = value.get("error_message")
    error_message = (
        str(error_message_value)[:2000]
        if error_message_value
        else (bounded_instructions[:2000] if not success else None)
    )
    event = SkillActivationEvent(
        success=success,
        name=str(value.get("name", ""))[:500],
        skill_id=str(value.get("skill_id", ""))[:500],
        source=str(value.get("source", ""))[:100],
        fingerprint=str(value.get("fingerprint", ""))[:128],
        instructions=bounded_instructions,
        args=(str(value["args"])[:4000] if value.get("args") is not None else None),
        truncated=len(instructions) > len(bounded_instructions),
        error_code=error_code,
        error_message=error_message,
    )
    structured = json_schema_output(
        _ACTIVATION_SCHEMA,
        event.model_dump(mode="json"),
    )
    return NormalizedToolOutput(
        structured_content=structured,
        is_error=not success,
        error_code=error_code,
        error_message=error_message,
        retryable=False,
    )


def _validate_asset_input(
    arguments: Mapping[str, JsonValue],
) -> Mapping[str, JsonValue]:
    canonical = _validate_asset_model(arguments)
    _validate_asset_path(str(canonical["relative_path"]))
    return canonical


def _materialize_skill_asset(
    workspace: WorkspaceRuntime,
    active_skill_root: ActiveSkillRoot,
    request: MaterializeSkillAssetInput,
) -> MaterializeSkillAssetOutput:
    root_value = active_skill_root(request.skill_id)
    if root_value is None:
        raise ValueError(f"skill is not active: {request.skill_id}")
    root = Path(root_value).expanduser().resolve()
    relative = _validate_asset_path(request.relative_path)
    source = (root / relative).resolve()
    _ensure_within(root, source)
    if not source.is_file():
        raise FileNotFoundError(f"skill asset not found: {request.relative_path}")

    destination = _asset_destination(
        workspace,
        request.skill_id,
        relative,
    )
    workspace.ensure_within_scratch(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    data = destination.read_bytes()
    return MaterializeSkillAssetOutput(
        workspace_path=workspace.relative_to_root(destination).as_posix(),
        source_fingerprint=hashlib.sha256(data).hexdigest(),
        size_bytes=len(data),
    )


def _asset_destination(
    workspace: WorkspaceRuntime,
    skill_id: str,
    relative: Path,
) -> Path:
    return workspace.scratch / "skills" / _safe_skill_id(skill_id) / relative


def _validate_asset_path(value: str) -> Path:
    path = Path(value)
    if path.is_absolute() or value.strip() in {"", "."} or ".." in path.parts:
        raise ToolValidationError(
            path="$.relative_path",
            message="skill asset path must be relative without parent traversal",
        )
    if not path.parts or path.parts[0] not in {"scripts", "references"}:
        raise ToolValidationError(
            path="$.relative_path",
            message="skill asset must be under scripts/ or references/",
        )
    if len(path.parts) < 2:
        raise ToolValidationError(
            path="$.relative_path",
            message="skill asset path must include a file name",
        )
    return path


def _ensure_within(root: Path, child: Path) -> None:
    try:
        common = Path(os.path.commonpath((str(root), str(child))))
    except ValueError:
        common = Path()
    if common != root:
        raise ValueError("skill asset path escapes its active skill root")


def _safe_skill_id(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._") or "skill"


def _object_mapping(raw: object) -> Mapping[str, object]:
    if isinstance(raw, Mapping):
        return raw
    if isinstance(raw, BaseModel):
        return raw.model_dump(mode="json")
    raise TypeError("skill invocation must return a mapping or BaseModel")


def _normalize_asset_output(raw: object) -> NormalizedToolOutput:
    validated = MaterializeSkillAssetOutput.model_validate(raw)
    structured = json_schema_output(
        _ASSET_OUTPUT_SCHEMA,
        validated.model_dump(mode="json"),
    )
    return NormalizedToolOutput(structured_content=structured)


__all__ = [
    "ActiveSkillRoot",
    "InvokeSkillInput",
    "MAX_SKILL_INSTRUCTIONS_CHARS",
    "MaterializeSkillAssetInput",
    "MaterializeSkillAssetOutput",
    "SkillActivationEvent",
    "SkillInvoker",
    "create_invoke_skill_tool",
    "create_materialize_skill_asset_tool",
    "create_skill_tools",
]
