"""协议层 Tool ACI:瘦身后的工具契约(设计 §5.1)。

移植自旧 agent_runtime/tools/tool.py 的公开契约与校验逻辑,并刻意删除:
normalize_output/NormalizedToolOutput(run 直接返回 ToolResult)、output_schema、
execution_revision、approval_profile、stream/continuation/after_commit、
committed_file_changes,以及约 600 行的 pydantic_input 审计机器。
builtins 一律使用 json_schema_input。
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Awaitable, Callable, Iterable, Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from numbers import Real
from typing import Literal

from jsonschema import exceptions as jsonschema_exceptions  # type: ignore[import-untyped]
from jsonschema.protocols import (  # type: ignore[import-untyped]
    Validator as JsonSchemaValidator,
)
from jsonschema.validators import validator_for  # type: ignore[import-untyped]
from referencing import Registry
from referencing.exceptions import Unresolvable

from praxis.protocol.json import JsonValue, freeze_json, thaw_json

_MAX_VALIDATION_PATH_LENGTH = 256
_MAX_VALIDATION_MESSAGE_LENGTH = 512


def _bounded_text(value: str, *, limit: int) -> str:
    value = " ".join(value.split())
    if len(value) <= limit:
        return value
    return f"{value[: limit - 1]}…"


class ToolValidationError(ValueError):
    """Bounded, caller-safe details for one tool schema validation failure."""

    path: str
    message: str

    def __init__(self, *, path: str, message: str) -> None:
        self.path = _bounded_text(path, limit=_MAX_VALIDATION_PATH_LENGTH)
        self.message = _bounded_text(message, limit=_MAX_VALIDATION_MESSAGE_LENGTH)
        super().__init__(f"{self.path}: {self.message}")


def _require_non_empty_string(value: object, *, field_name: str) -> None:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string")
    if not value:
        raise ValueError(f"{field_name} must not be empty")


def _require_exact_bool(value: object, *, field_name: str) -> None:
    if type(value) is not bool:
        raise TypeError(f"{field_name} must be a bool")


def _freeze_mapping(value: Mapping[str, object], *, path: str) -> Mapping[str, JsonValue]:
    frozen = freeze_json(value, path=path)
    if not isinstance(frozen, Mapping):
        raise TypeError(f"{path} must be a mapping")
    return frozen


def _canonical_json(value: object, *, subject: str) -> JsonValue:
    try:
        return freeze_json(value, path=subject)
    except (TypeError, ValueError):
        raise ToolValidationError(
            path="$",
            message=f"{subject} must contain only finite JSON-compatible values",
        ) from None


def _canonical_mapping(value: Mapping[str, object], *, subject: str) -> Mapping[str, JsonValue]:
    canonical = _canonical_json(value, subject=subject)
    if not isinstance(canonical, Mapping):
        raise ToolValidationError(path="$", message=f"{subject} must be an object")
    return canonical


def _validate_error_fields(
    *,
    is_error: object,
    error_code: object,
    error_message: object,
    retryable: object,
) -> None:
    _require_exact_bool(is_error, field_name="is_error")
    _require_exact_bool(retryable, field_name="retryable")
    if error_code is not None:
        _require_non_empty_string(error_code, field_name="error_code")
    if error_message is not None and not isinstance(error_message, str):
        raise TypeError("error_message must be a string when provided")
    if is_error and error_code is None:
        raise ValueError("error_code is required when is_error=True")
    if not is_error and (error_code is not None or error_message is not None or retryable):
        raise ValueError("error fields require is_error=True")


class ToolEffect(StrEnum):
    READ_WORKSPACE = "read_workspace"
    WRITE_WORKSPACE = "write_workspace"
    EXECUTE_PROCESS = "execute_process"
    NETWORK = "network"
    DESTRUCTIVE = "destructive"


_LOCAL_SIDE_EFFECTS = frozenset(
    {
        ToolEffect.WRITE_WORKSPACE,
        ToolEffect.EXECUTE_PROCESS,
        ToolEffect.DESTRUCTIVE,
    }
)


class CancellationMode(StrEnum):
    COOPERATIVE = "cooperative"
    MANAGED_PROCESS = "managed_process"
    REMOTE_BEST_EFFORT = "remote_best_effort"
    NOT_CANCELLABLE = "not_cancellable"


class InterruptBehavior(StrEnum):
    CANCEL = "cancel"
    FINISH_CURRENT = "finish_current"


class ResultStatus(StrEnum):
    OK = "ok"
    ERROR = "error"
    DENIED = "denied"
    NOT_EXECUTED = "not_executed"
    OUTCOME_UNKNOWN = "outcome_unknown"
    PAUSED = "paused"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"


_ERROR_IMPLYING_STATUSES = frozenset(
    {
        ResultStatus.ERROR,
        ResultStatus.DENIED,
        ResultStatus.TIMEOUT,
        ResultStatus.CANCELLED,
    }
)


@dataclass(frozen=True, slots=True)
class ToolTarget:
    kind: str
    value: str

    def __post_init__(self) -> None:
        _require_non_empty_string(self.kind, field_name="target kind")
        _require_non_empty_string(self.value, field_name="target value")


@dataclass(frozen=True, slots=True)
class ResolvedUse:
    """一次调用实际触达的副作用与目标(权限判定的输入,移植旧 ResolvedToolUse)。"""

    effects: frozenset[ToolEffect]
    targets: tuple[ToolTarget, ...]

    def __post_init__(self) -> None:
        effects = frozenset(self.effects)
        if any(not isinstance(effect, ToolEffect) for effect in effects):
            raise TypeError("effects must contain ToolEffect values")
        targets = tuple(self.targets)
        if any(not isinstance(target, ToolTarget) for target in targets):
            raise TypeError("targets must contain ToolTarget values")
        object.__setattr__(self, "effects", effects)
        object.__setattr__(self, "targets", targets)


@dataclass(frozen=True, slots=True)
class ToolDefinition:
    """Immutable model-facing projection of a tool."""

    name: str
    description: str
    input_schema: Mapping[str, JsonValue]

    def __post_init__(self) -> None:
        _require_non_empty_string(self.name, field_name="tool name")
        _require_non_empty_string(self.description, field_name="tool description")
        object.__setattr__(
            self,
            "input_schema",
            _freeze_mapping(self.input_schema, path="input_schema"),
        )


@dataclass(frozen=True, slots=True)
class ToolCall:
    """一次工具调用请求;旧 origin/argument_error 随 fencing 一并删除。"""

    tool_call_id: str
    tool_name: str
    arguments: Mapping[str, JsonValue]

    def __post_init__(self) -> None:
        _require_non_empty_string(self.tool_call_id, field_name="tool_call_id")
        _require_non_empty_string(self.tool_name, field_name="tool_name")
        object.__setattr__(
            self,
            "arguments",
            _freeze_mapping(self.arguments, path="arguments"),
        )


@dataclass(frozen=True, slots=True)
class ToolContentBlock:
    """One model-visible block with an immutable JSON-compatible payload."""

    type: Literal["text", "image", "resource"]
    data: Mapping[str, JsonValue]

    def __post_init__(self) -> None:
        if self.type not in {"text", "image", "resource"}:
            raise ValueError("tool content block type must be text, image, or resource")
        object.__setattr__(
            self,
            "data",
            _freeze_mapping(self.data, path="content block data"),
        )


def _freeze_content(content: Iterable[ToolContentBlock]) -> tuple[ToolContentBlock, ...]:
    blocks = tuple(content)
    if any(not isinstance(block, ToolContentBlock) for block in blocks):
        raise TypeError("content must contain ToolContentBlock values")
    return blocks


@dataclass(frozen=True, slots=True)
class ToolResult:
    """工具结果:模型可见内容与运行时元数据分离;run 直接返回本类型。"""

    tool_call_id: str
    tool_name: str
    content: tuple[ToolContentBlock, ...] = ()
    status: ResultStatus = ResultStatus.OK
    is_error: bool = False
    error_code: str | None = None
    error_message: str | None = None
    retryable: bool = False
    truncated: bool = False
    metadata: Mapping[str, JsonValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require_non_empty_string(self.tool_call_id, field_name="tool_call_id")
        _require_non_empty_string(self.tool_name, field_name="tool_name")
        if not isinstance(self.status, ResultStatus):
            raise TypeError("status must be a ResultStatus")
        _validate_error_fields(
            is_error=self.is_error,
            error_code=self.error_code,
            error_message=self.error_message,
            retryable=self.retryable,
        )
        _require_exact_bool(self.truncated, field_name="truncated")
        if self.status in _ERROR_IMPLYING_STATUSES and not self.is_error:
            raise ValueError(f"status={self.status.value} requires is_error=True")
        if self.status is ResultStatus.OK and self.is_error:
            raise ValueError("status=ok requires is_error=False")
        object.__setattr__(self, "content", _freeze_content(self.content))
        object.__setattr__(
            self,
            "metadata",
            _freeze_mapping(self.metadata, path="metadata"),
        )

    @property
    def text(self) -> str:
        """拼接所有 text 块的 ``data["text"]``(非字符串值跳过)。"""
        parts: list[str] = []
        for block in self.content:
            if block.type == "text":
                value = block.data.get("text")
                if isinstance(value, str):
                    parts.append(value)
        return "\n".join(parts)


type ValidateInput = Callable[[Mapping[str, JsonValue]], Mapping[str, JsonValue]]
type ToolRunner = Callable[[Mapping[str, JsonValue]], ToolResult | Awaitable[ToolResult]]
type ResolveUse = Callable[[Mapping[str, JsonValue]], ResolvedUse]


def _json_path(parts: Iterable[str | int]) -> str:
    path = "$"
    for part in parts:
        path += f"[{part}]" if isinstance(part, int) else f".{part}"
    return _bounded_text(path, limit=_MAX_VALIDATION_PATH_LENGTH)


def _jsonschema_error_path(error: jsonschema_exceptions.ValidationError) -> str:
    parts = list(error.absolute_path)
    if error.validator == "required" and isinstance(error.instance, Mapping):
        for name in error.validator_value:
            if isinstance(name, str) and name not in error.instance:
                parts.append(name)
                break
    elif (
        error.validator == "additionalProperties"
        and isinstance(error.instance, Mapping)
        and not error.schema.get("patternProperties")
    ):
        properties = error.schema.get("properties", {})
        if isinstance(properties, Mapping):
            for name in error.instance:
                if name not in properties:
                    parts.append(name)
                    break
    return _json_path(parts)


def _tool_error_from_jsonschema_validation(
    error: jsonschema_exceptions.ValidationError,
) -> ToolValidationError:
    validator = error.validator if isinstance(error.validator, str) else "validation"
    return ToolValidationError(
        path=_jsonschema_error_path(error),
        message=f"{validator}: validation failed",
    )


def _tool_error_from_jsonschema_schema(
    error: jsonschema_exceptions.SchemaError,
) -> ToolValidationError:
    validator = error.validator if isinstance(error.validator, str) else "schema"
    return ToolValidationError(
        path=_json_path(error.absolute_path),
        message=f"{validator}: {error.message}",
    )


def _json_schema_validator(schema: Mapping[str, JsonValue]) -> JsonSchemaValidator:
    canonical_schema = _canonical_mapping(schema, subject="schema")
    schema_copy = thaw_json(canonical_schema)
    if not isinstance(schema_copy, dict):
        raise ToolValidationError(path="$", message="schema must be an object")

    if "$schema" in schema_copy:
        validator_type = validator_for(schema_copy, default=None)
        if validator_type is None:
            raise ToolValidationError(
                path="$.$schema",
                message="unsupported JSON Schema dialect",
            )
    else:
        validator_type = validator_for(schema_copy)
    try:
        validator_type.check_schema(schema_copy)
    except jsonschema_exceptions.SchemaError as error:
        raise _tool_error_from_jsonschema_schema(error) from None
    return validator_type(schema_copy, registry=Registry())


def _validate_with_json_schema(
    validator: JsonSchemaValidator,
    canonical_value: JsonValue,
) -> None:
    instance = thaw_json(canonical_value)
    try:
        error = next(validator.iter_errors(instance), None)
    except Unresolvable:
        raise ToolValidationError(
            path="$",
            message="schema reference could not be resolved locally",
        ) from None
    if error is not None:
        raise _tool_error_from_jsonschema_validation(error)


def json_schema_input(schema: Mapping[str, JsonValue]) -> ValidateInput:
    """Build an input validator directly from a complete JSON Schema document."""

    validator = _json_schema_validator(schema)

    def validate(arguments: Mapping[str, JsonValue]) -> Mapping[str, JsonValue]:
        canonical_arguments = _canonical_mapping(arguments, subject="input")
        _validate_with_json_schema(validator, canonical_arguments)
        return canonical_arguments

    return validate


def args_digest(arguments: Mapping[str, JsonValue]) -> str:
    """参数规范化摘要:键序无关,供审计与幂等判断使用。"""
    payload = json.dumps(thaw_json(arguments), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class Tool:
    """工具契约:定义 + 校验 + 运行 + 副作用声明 + 执行策略(瘦身移植)。"""

    definition: ToolDefinition
    validate_input: ValidateInput
    run: ToolRunner
    static_effects: frozenset[ToolEffect]
    resolve_use: ResolveUse
    idempotent: bool
    concurrency_safe: bool
    cancellation_mode: CancellationMode
    interrupt_behavior: InterruptBehavior
    timeout_seconds: float
    max_model_output_bytes: int

    def __post_init__(self) -> None:
        if not isinstance(self.definition, ToolDefinition):
            raise TypeError("definition must be a ToolDefinition")
        for name in ("validate_input", "run", "resolve_use"):
            if not callable(getattr(self, name)):
                raise TypeError(f"{name} must be callable")

        static_effects = frozenset(self.static_effects)
        if any(not isinstance(effect, ToolEffect) for effect in static_effects):
            raise TypeError("static_effects must contain ToolEffect values")
        object.__setattr__(self, "static_effects", static_effects)
        if not isinstance(self.cancellation_mode, CancellationMode):
            raise TypeError("cancellation_mode must be a CancellationMode")
        if not isinstance(self.interrupt_behavior, InterruptBehavior):
            raise TypeError("interrupt_behavior must be an InterruptBehavior")
        _require_exact_bool(self.idempotent, field_name="idempotent")
        _require_exact_bool(self.concurrency_safe, field_name="concurrency_safe")
        if (
            static_effects & _LOCAL_SIDE_EFFECTS
            and self.cancellation_mode is CancellationMode.NOT_CANCELLABLE
        ):
            raise ValueError(
                "local side-effecting tools cannot use cancellation mode not_cancellable"
            )

        if isinstance(self.timeout_seconds, bool) or not isinstance(
            self.timeout_seconds,
            Real,
        ):
            raise TypeError("timeout_seconds must be a real number")
        if not math.isfinite(self.timeout_seconds) or self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be a positive finite number")
        if not isinstance(self.max_model_output_bytes, int) or isinstance(
            self.max_model_output_bytes,
            bool,
        ):
            raise ValueError("max_model_output_bytes must be a positive integer")
        if self.max_model_output_bytes <= 0:
            raise ValueError("max_model_output_bytes must be a positive integer")
