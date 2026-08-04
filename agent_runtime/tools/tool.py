from __future__ import annotations

import math
from collections.abc import Awaitable, Callable, Iterable, Mapping
from dataclasses import dataclass, field, is_dataclass
from enum import Enum, StrEnum
from numbers import Real
from types import MappingProxyType, UnionType
from typing import (
    Annotated,
    Any,
    Literal,
    TypeAliasType,
    Union,
    get_args,
    get_origin,
    get_type_hints,
)

from jsonschema import exceptions as jsonschema_exceptions  # type: ignore[import-untyped]
from jsonschema.protocols import (  # type: ignore[import-untyped]
    Validator as JsonSchemaValidator,
)
from jsonschema.validators import validator_for  # type: ignore[import-untyped]
from pydantic import (
    AfterValidator,
    AliasChoices,
    AliasPath,
    BaseModel,
    BeforeValidator,
    PlainSerializer,
    PlainValidator,
    WrapSerializer,
    WrapValidator,
)
from pydantic import ValidationError as PydanticValidationError
from pydantic.fields import FieldInfo
from referencing import Registry
from referencing.exceptions import Unresolvable
from typing_extensions import is_typeddict

type JsonValue = (
    str
    | int
    | float
    | bool
    | None
    | tuple[JsonValue, ...]
    | Mapping[str, JsonValue]
)

type _MutableJsonValue = (
    str
    | int
    | float
    | bool
    | None
    | list[_MutableJsonValue]
    | dict[str, _MutableJsonValue]
)


_MAX_VALIDATION_PATH_LENGTH = 256
_MAX_VALIDATION_MESSAGE_LENGTH = 512
_PYDANTIC_VALIDATION_ONLY_METADATA_TYPES = (
    BeforeValidator,
    AfterValidator,
    PlainValidator,
    WrapValidator,
)
_PYDANTIC_SERIALIZATION_METADATA_TYPES = (PlainSerializer, WrapSerializer)
_VALIDATION_AFFECTING_JSON_SCHEMA_KEYS = frozenset(
    {
        "$anchor",
        "$defs",
        "$dynamicAnchor",
        "$dynamicRef",
        "$id",
        "$ref",
        "$schema",
        "additionalItems",
        "additionalProperties",
        "allOf",
        "anyOf",
        "const",
        "contains",
        "contentEncoding",
        "contentMediaType",
        "contentSchema",
        "definitions",
        "dependencies",
        "dependentRequired",
        "dependentSchemas",
        "discriminator",
        "else",
        "enum",
        "exclusiveMaximum",
        "exclusiveMinimum",
        "format",
        "if",
        "items",
        "maxContains",
        "maxItems",
        "maxLength",
        "maxProperties",
        "maximum",
        "minContains",
        "minItems",
        "minLength",
        "minProperties",
        "minimum",
        "multipleOf",
        "not",
        "oneOf",
        "pattern",
        "patternProperties",
        "prefixItems",
        "properties",
        "propertyNames",
        "required",
        "then",
        "type",
        "unevaluatedItems",
        "unevaluatedProperties",
        "uniqueItems",
    }
)


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
        self.message = _bounded_text(
            message,
            limit=_MAX_VALIDATION_MESSAGE_LENGTH,
        )
        super().__init__(f"{self.path}: {self.message}")


def _freeze_json(value: object, *, path: str) -> JsonValue:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{path} must contain only finite JSON numbers")
        return value
    if isinstance(value, Mapping):
        frozen: dict[str, JsonValue] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(f"{path} keys must be strings")
            frozen[key] = _freeze_json(item, path=f"{path}.{key}")
        return MappingProxyType(frozen)
    if isinstance(value, (list, tuple)):
        return tuple(
            _freeze_json(item, path=f"{path}[{index}]")
            for index, item in enumerate(value)
        )
    raise TypeError(f"{path} must contain only JSON-compatible values")


def _freeze_mapping(value: Mapping[str, object], *, path: str) -> Mapping[str, JsonValue]:
    frozen = _freeze_json(value, path=path)
    if not isinstance(frozen, Mapping):
        raise TypeError(f"{path} must be a mapping")
    return frozen


def _thaw_json(value: JsonValue) -> _MutableJsonValue:
    if isinstance(value, Mapping):
        return {key: _thaw_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw_json(item) for item in value]
    return value


def _canonical_json(value: object, *, subject: str) -> JsonValue:
    try:
        return _freeze_json(value, path=subject)
    except (TypeError, ValueError):
        raise ToolValidationError(
            path="$",
            message=f"{subject} must contain only finite JSON-compatible values",
        ) from None


def _canonical_mapping(
    value: Mapping[str, object],
    *,
    subject: str,
) -> Mapping[str, JsonValue]:
    canonical = _canonical_json(value, subject=subject)
    if not isinstance(canonical, Mapping):
        raise ToolValidationError(path="$", message=f"{subject} must be an object")
    return canonical


def _require_non_empty_string(value: object, *, field_name: str) -> None:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string")
    if not value:
        raise ValueError(f"{field_name} must not be empty")


def _require_exact_bool(value: object, *, field_name: str) -> None:
    if type(value) is not bool:
        raise TypeError(f"{field_name} must be a bool")


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
    if not is_error and (
        error_code is not None or error_message is not None or retryable
    ):
        raise ValueError("error fields require is_error=True")


class ToolEffect(StrEnum):
    READ_WORKSPACE = "read_workspace"
    WRITE_WORKSPACE = "write_workspace"
    EXECUTE_PROCESS = "execute_process"
    NETWORK = "network"
    DESTRUCTIVE = "destructive"


class ToolApprovalProfile(StrEnum):
    """Runtime-owned attestation used by explicit automatic-approval policies."""

    RESTRICTED_READ_ONLY_PROCESS = "restricted_read_only_process"


@dataclass(frozen=True, slots=True)
class ToolTarget:
    kind: str
    value: str

    def __post_init__(self) -> None:
        _require_non_empty_string(self.kind, field_name="target kind")
        _require_non_empty_string(self.value, field_name="target value")


@dataclass(frozen=True, slots=True)
class ResolvedToolUse:
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


class CancellationMode(StrEnum):
    COOPERATIVE = "cooperative"
    MANAGED_PROCESS = "managed_process"
    REMOTE_BEST_EFFORT = "remote_best_effort"
    NOT_CANCELLABLE = "not_cancellable"


class InterruptBehavior(StrEnum):
    CANCEL = "cancel"
    FINISH_CURRENT = "finish_current"


@dataclass(frozen=True, slots=True)
class ToolCallOrigin:
    request_id: str
    toolset_revision: str
    exposed_tool_names: tuple[str, ...]

    def __post_init__(self) -> None:
        _require_non_empty_string(self.request_id, field_name="origin request_id")
        _require_non_empty_string(
            self.toolset_revision,
            field_name="origin toolset_revision",
        )
        if not isinstance(self.exposed_tool_names, (list, tuple)):
            raise TypeError("origin exposed_tool_names must be a list or tuple")
        exposed_tool_names = tuple(self.exposed_tool_names)
        for name in exposed_tool_names:
            _require_non_empty_string(name, field_name="origin exposed tool name")
        object.__setattr__(self, "exposed_tool_names", exposed_tool_names)


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
    tool_call_id: str
    tool_name: str
    arguments: Mapping[str, JsonValue]
    origin: ToolCallOrigin

    def __post_init__(self) -> None:
        _require_non_empty_string(self.tool_call_id, field_name="tool_call_id")
        _require_non_empty_string(self.tool_name, field_name="tool_name")
        if not isinstance(self.origin, ToolCallOrigin):
            raise TypeError("origin must be a ToolCallOrigin")
        object.__setattr__(
            self,
            "arguments",
            _freeze_mapping(self.arguments, path="arguments"),
        )

    def __deepcopy__(self, memo: dict[int, object]) -> ToolCall:
        del memo
        return self


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


@dataclass(frozen=True, slots=True)
class ArtifactReference:
    """Stable reference to an artifact kept outside model content."""

    artifact_id: str
    media_type: str | None = None
    name: str | None = None

    def __post_init__(self) -> None:
        _require_non_empty_string(self.artifact_id, field_name="artifact_id")
        if self.media_type is not None and not isinstance(self.media_type, str):
            raise TypeError("media_type must be a string when provided")
        if self.name is not None and not isinstance(self.name, str):
            raise TypeError("name must be a string when provided")


def _freeze_content(
    content: tuple[ToolContentBlock, ...],
) -> tuple[ToolContentBlock, ...]:
    content = tuple(content)
    if any(not isinstance(block, ToolContentBlock) for block in content):
        raise TypeError("content must contain ToolContentBlock values")
    return content


def _freeze_attachments(
    attachments: tuple[ArtifactReference, ...],
) -> tuple[ArtifactReference, ...]:
    attachments = tuple(attachments)
    if any(not isinstance(item, ArtifactReference) for item in attachments):
        raise TypeError("attachments must contain ArtifactReference values")
    return attachments


@dataclass(frozen=True, slots=True)
class NormalizedToolOutput:
    """Identity-free canonical output produced by a tool adapter."""

    content: tuple[ToolContentBlock, ...] = ()
    structured_content: JsonValue | None = None
    is_error: bool = False
    error_code: str | None = None
    error_message: str | None = None
    retryable: bool = False
    metadata: Mapping[str, JsonValue] = field(default_factory=dict)
    attachments: tuple[ArtifactReference, ...] = ()

    def __post_init__(self) -> None:
        _validate_error_fields(
            is_error=self.is_error,
            error_code=self.error_code,
            error_message=self.error_message,
            retryable=self.retryable,
        )
        object.__setattr__(self, "content", _freeze_content(self.content))
        object.__setattr__(
            self,
            "attachments",
            _freeze_attachments(self.attachments),
        )
        if self.structured_content is not None:
            object.__setattr__(
                self,
                "structured_content",
                _freeze_json(self.structured_content, path="structured_content"),
            )
        object.__setattr__(
            self,
            "metadata",
            _freeze_mapping(self.metadata, path="metadata"),
        )


@dataclass(frozen=True, slots=True)
class ToolResult:
    """Tool outcome with model-visible content separated from runtime metadata."""

    tool_call_id: str
    tool_name: str
    content: tuple[ToolContentBlock, ...] = ()
    structured_content: JsonValue | None = None
    is_error: bool = False
    error_code: str | None = None
    error_message: str | None = None
    retryable: bool = False
    truncated: bool = False
    metadata: Mapping[str, JsonValue] = field(default_factory=dict)
    attachments: tuple[ArtifactReference, ...] = ()

    def __post_init__(self) -> None:
        _require_non_empty_string(self.tool_call_id, field_name="tool_call_id")
        _require_non_empty_string(self.tool_name, field_name="tool_name")
        _validate_error_fields(
            is_error=self.is_error,
            error_code=self.error_code,
            error_message=self.error_message,
            retryable=self.retryable,
        )
        _require_exact_bool(self.truncated, field_name="truncated")
        object.__setattr__(self, "content", _freeze_content(self.content))
        object.__setattr__(
            self,
            "attachments",
            _freeze_attachments(self.attachments),
        )
        if self.structured_content is not None:
            object.__setattr__(
                self,
                "structured_content",
                _freeze_json(self.structured_content, path="structured_content"),
            )
        object.__setattr__(
            self,
            "metadata",
            _freeze_mapping(self.metadata, path="metadata"),
        )

    def __deepcopy__(self, memo: dict[int, object]) -> ToolResult:
        del memo
        return self


type ValidateInput = Callable[[Mapping[str, JsonValue]], Mapping[str, JsonValue]]
type ToolRunner = Callable[[Mapping[str, JsonValue]], object | Awaitable[object]]
type NormalizeOutput = Callable[[object], NormalizedToolOutput]
type ResolveToolUse = Callable[[Mapping[str, JsonValue]], ResolvedToolUse]


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


def _close_pydantic_object_schemas(value: _MutableJsonValue) -> None:
    if isinstance(value, list):
        for item in value:
            _close_pydantic_object_schemas(item)
        return
    if not isinstance(value, dict):
        return
    if (
        value.get("type") == "object"
        and "properties" in value
        and "additionalProperties" not in value
    ):
        value["additionalProperties"] = False
    for item in value.values():
        _close_pydantic_object_schemas(item)


def _json_schema_validator(
    schema: Mapping[str, JsonValue],
) -> JsonSchemaValidator:
    canonical_schema = _canonical_mapping(schema, subject="schema")
    schema_copy = _thaw_json(canonical_schema)
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
    instance = _thaw_json(canonical_value)
    try:
        error = next(validator.iter_errors(instance), None)
    except Unresolvable:
        raise ToolValidationError(
            path="$",
            message="schema reference could not be resolved locally",
        ) from None
    if error is not None:
        raise _tool_error_from_jsonschema_validation(error)


def _tool_error_from_pydantic(error: PydanticValidationError) -> ToolValidationError:
    details = error.errors(
        include_url=False,
        include_context=False,
        include_input=False,
    )
    if not details:
        return ToolValidationError(path="$", message="input validation failed")
    first = details[0]
    error_type = first.get("type", "validation")
    return ToolValidationError(
        path=_json_path(first.get("loc", ())),
        message=f"{error_type}: validation failed",
    )


def _field_input_paths(
    model: type[BaseModel],
    field_name: str,
    field: FieldInfo,
) -> tuple[tuple[str | int, ...], ...]:
    paths: list[tuple[str | int, ...]] = []
    validation_alias = field.validation_alias
    has_alias = validation_alias is not None or field.alias is not None
    if model.model_config.get("validate_by_alias", True):
        if isinstance(validation_alias, str):
            paths.append((validation_alias,))
        elif field.alias is not None:
            paths.append((field.alias,))
    if not has_alias or model.model_config.get("validate_by_name", False):
        paths.append((field_name,))
    return tuple(paths)


def _value_at_input_path(
    value: object,
    path: tuple[str | int, ...],
) -> tuple[bool, object]:
    current = value
    for part in path:
        if isinstance(part, str) and isinstance(current, Mapping):
            if part not in current:
                return False, None
            current = current[part]
        elif isinstance(part, int) and isinstance(current, (list, tuple)):
            if part < 0 or part >= len(current):
                return False, None
            current = current[part]
        else:
            return False, None
    return True, current


def _reject_ignored_extras_in_value(
    validated: object,
    original: object,
    *,
    path: tuple[str | int, ...],
) -> None:
    if isinstance(validated, BaseModel) and isinstance(original, Mapping):
        _reject_ignored_model_extras(validated, original, path=path)
        return
    if isinstance(validated, (list, tuple)) and isinstance(original, (list, tuple)):
        if len(validated) != len(original):
            raise ToolValidationError(
                path=_json_path(path),
                message="sequence cardinality changed during validation",
            )
        for index, (validated_item, original_item) in enumerate(
            zip(validated, original, strict=False)
        ):
            _reject_ignored_extras_in_value(
                validated_item,
                original_item,
                path=(*path, index),
            )
        return
    if isinstance(validated, Mapping) and isinstance(original, Mapping):
        if len(validated) != len(original):
            raise ToolValidationError(
                path=_json_path(path),
                message=(
                    "mapping keys changed cardinality during validation; "
                    "normalized keys must remain unique"
                ),
            )
        for (original_key, original_item), validated_item in zip(
            original.items(),
            validated.values(),
            strict=False,
        ):
            _reject_ignored_extras_in_value(
                validated_item,
                original_item,
                path=(*path, original_key),
            )


def _reject_ignored_model_extras(
    validated: BaseModel,
    original: Mapping[str, object],
    *,
    path: tuple[str | int, ...],
) -> None:
    model = type(validated)
    if validated.model_extra:
        canonical_names = set(model.model_fields) | set(model.model_computed_fields)
        for key in original:
            if key in validated.model_extra and key in canonical_names:
                raise ToolValidationError(
                    path=_json_path((*path, key)),
                    message="extra key collides with canonical field name",
                )
    consumed_keys: set[str] = set()
    for field_name, model_field in model.model_fields.items():
        for input_path in _field_input_paths(model, field_name, model_field):
            found, original_value = _value_at_input_path(original, input_path)
            if not found:
                continue
            top_level_key = input_path[0]
            if isinstance(top_level_key, str):
                consumed_keys.add(top_level_key)
            _reject_ignored_extras_in_value(
                getattr(validated, field_name),
                original_value,
                path=(*path, *input_path),
            )
            break

    extra_policy = model.model_config.get("extra")
    if extra_policy in (None, "ignore"):
        for key in original:
            if key not in consumed_keys:
                raise ToolValidationError(
                    path=_json_path((*path, key)),
                    message="extra_forbidden: Extra inputs are not permitted",
                )
    elif extra_policy == "allow" and validated.model_extra:
        for key, extra_value in validated.model_extra.items():
            if key in original:
                _reject_ignored_extras_in_value(
                    extra_value,
                    original[key],
                    path=(*path, key),
                )


def _metadata_has_core_schema_capability(metadata: object) -> bool:
    return (
        callable(getattr(metadata, "__get_pydantic_core_schema__", None))
        or callable(getattr(metadata, "__get_validators__", None))
        or hasattr(metadata, "__pydantic_core_schema__")
    )


def _audit_pydantic_metadata(
    metadata_items: Iterable[object],
    *,
    path: tuple[str | int, ...],
    contains_model: bool,
) -> None:
    for metadata in metadata_items:
        if callable(getattr(metadata, "__get_pydantic_json_schema__", None)):
            raise ToolValidationError(
                path=_json_path(path),
                message="unsupported Pydantic JSON Schema customization: metadata hook",
            )
        has_core_schema = _metadata_has_core_schema_capability(metadata)
        if contains_model and has_core_schema:
            raise ToolValidationError(
                path=_json_path(path),
                message="unsupported Pydantic input shape: structural core-schema hook",
            )
        if isinstance(metadata, _PYDANTIC_SERIALIZATION_METADATA_TYPES) or (
            has_core_schema
            and not isinstance(
                metadata,
                _PYDANTIC_VALIDATION_ONLY_METADATA_TYPES,
            )
        ):
            raise ToolValidationError(
                path=_json_path(path),
                message="unsupported Pydantic serialization metadata",
            )


def _pydantic_hook_differs_from_base(
    model: type[BaseModel],
    hook_name: str,
) -> bool:
    resolved_hook = next(
        (
            (owner, owner.__dict__[hook_name])
            for owner in model.__mro__
            if hook_name in owner.__dict__
        ),
        None,
    )
    if resolved_hook is None:
        return False
    _, descriptor = resolved_hook
    base_descriptor = BaseModel.__dict__.get(hook_name)
    matches_base = descriptor is base_descriptor or (
        isinstance(descriptor, classmethod)
        and isinstance(base_descriptor, classmethod)
        and descriptor.__func__ is base_descriptor.__func__
    )
    return not matches_base and callable(getattr(model, hook_name, None))


def _pydantic_model_has_unverifiable_lifecycle(model: type[BaseModel]) -> bool:
    return any(
        _pydantic_hook_differs_from_base(model, hook_name)
        for hook_name in (
            "__get_pydantic_core_schema__",
            "__get_validators__",
            "model_validate",
        )
    )


def _pydantic_model_has_custom_lifecycle(model: type[BaseModel]) -> bool:
    return _pydantic_model_has_unverifiable_lifecycle(model) or bool(
        getattr(model, "__pydantic_custom_init__", False)
    ) or (
        getattr(model, "__pydantic_post_init__", None) is not None
    )


def _audit_json_schema_extra(
    extra: object,
    *,
    path: tuple[str | int, ...],
) -> None:
    if extra is None:
        return
    if callable(extra):
        raise ToolValidationError(
            path=_json_path(path),
            message="unsupported Pydantic JSON Schema customization: callable extra",
        )
    if not isinstance(extra, Mapping):
        raise ToolValidationError(
            path=_json_path(path),
            message="unsupported Pydantic JSON Schema customization",
        )
    keyword = next(
        (
            key
            for key in extra
            if key in _VALIDATION_AFFECTING_JSON_SCHEMA_KEYS
        ),
        None,
    )
    if keyword is not None:
        raise ToolValidationError(
            path=_json_path((*path, keyword)),
            message=(
                "unsupported Pydantic JSON Schema customization: "
                f"validation keyword {keyword}"
            ),
        )


def _audit_pydantic_annotation(
    annotation: object,
    *,
    path: tuple[str | int, ...],
    visited: set[type[BaseModel]],
    seen_aliases: set[TypeAliasType],
) -> bool:
    if isinstance(annotation, TypeAliasType):
        if annotation in seen_aliases:
            return False
        seen_aliases.add(annotation)
        return _audit_pydantic_annotation(
            annotation.__value__,
            path=path,
            visited=visited,
            seen_aliases=seen_aliases,
        )
    if is_dataclass(annotation):
        raise ToolValidationError(
            path=_json_path(path),
            message="unsupported Pydantic input shape: dataclass",
        )
    if is_typeddict(annotation):
        raise ToolValidationError(
            path=_json_path(path),
            message="unsupported Pydantic input shape: TypedDict",
        )
    if isinstance(annotation, type) and issubclass(annotation, BaseModel):
        _audit_pydantic_model(annotation, path=path, visited=visited)
        return True

    origin = get_origin(annotation)
    arguments = get_args(annotation)
    if origin is Annotated:
        contains_model = _audit_pydantic_annotation(
            arguments[0],
            path=path,
            visited=visited,
            seen_aliases=seen_aliases,
        )
        _audit_pydantic_metadata(
            arguments[1:],
            path=path,
            contains_model=contains_model,
        )
        return contains_model
    if origin is Literal:
        return False
    if origin in (Union, UnionType):
        contains_model = False
        for argument in arguments:
            branch_contains_model = _audit_pydantic_annotation(
                argument,
                path=path,
                visited=visited,
                seen_aliases=seen_aliases,
            )
            contains_model = contains_model or branch_contains_model
        return contains_model
    if origin in (list, tuple):
        contains_model = False
        for argument in arguments:
            if argument is Ellipsis:
                continue
            branch_contains_model = _audit_pydantic_annotation(
                argument,
                path=path,
                visited=visited,
                seen_aliases=seen_aliases,
            )
            contains_model = contains_model or branch_contains_model
        return contains_model
    if origin in (dict, Mapping):
        if not arguments:
            return False
        key_contains_model = _audit_pydantic_annotation(
            arguments[0],
            path=path,
            visited=visited,
            seen_aliases=seen_aliases,
        )
        if key_contains_model:
            raise ToolValidationError(
                path=_json_path(path),
                message="unsupported Pydantic input shape: model mapping key",
            )
        return _audit_pydantic_annotation(
            arguments[1],
            path=path,
            visited=visited,
            seen_aliases=seen_aliases,
        )
    if annotation in (Any, str, int, float, bool, type(None), list, tuple, dict, Mapping):
        return False
    if isinstance(annotation, type) and issubclass(annotation, Enum):
        if (
            _metadata_has_core_schema_capability(annotation)
            or callable(
                getattr(annotation, "__get_pydantic_json_schema__", None)
            )
            or hasattr(annotation, "__pydantic_serializer__")
        ):
            raise ToolValidationError(
                path=_json_path(path),
                message=(
                    "unsupported Pydantic Enum core/schema/serialization "
                    "customization"
                ),
            )
        return False
    raise ToolValidationError(
        path=_json_path(path),
        message="unsupported Pydantic input shape: composite annotation",
    )


def _audit_pydantic_model(
    model: type[BaseModel],
    *,
    path: tuple[str | int, ...] = (),
    visited: set[type[BaseModel]] | None = None,
) -> None:
    if visited is None:
        visited = set()
    if getattr(model, "__pydantic_root_model__", False):
        raise ToolValidationError(
            path=_json_path(path),
            message="unsupported Pydantic input shape: RootModel",
        )
    if model in visited:
        return
    visited.add(model)
    _audit_json_schema_extra(model.model_config.get("json_schema_extra"), path=path)
    if (
        model.model_config.get("json_schema_mode_override") is not None
        or model.model_config.get("schema_generator") is not None
        or _pydantic_hook_differs_from_base(
            model,
            "__get_pydantic_json_schema__",
        )
        or _pydantic_hook_differs_from_base(model, "model_json_schema")
    ):
        raise ToolValidationError(
            path=_json_path(path),
            message="unsupported Pydantic JSON Schema customization",
        )
    if (
        model.model_config.get("json_encoders")
        or _pydantic_hook_differs_from_base(model, "model_dump")
    ):
        raise ToolValidationError(
            path=_json_path(path),
            message="unsupported Pydantic serialization customization",
        )

    structural_fields: set[str] = set()
    if model.model_config.get("extra") == "allow":
        extra_annotation = get_type_hints(model, include_extras=True).get(
            "__pydantic_extra__"
        )
        if extra_annotation is not None:
            contains_model = _audit_pydantic_annotation(
                extra_annotation,
                path=(*path, "__pydantic_extra__"),
                visited=visited,
                seen_aliases=set(),
            )
            if contains_model:
                structural_fields.add("__pydantic_extra__")

    input_name_fields: dict[str, str] = {}
    for field_name, model_field in model.model_fields.items():
        validation_alias = model_field.validation_alias
        if (
            validation_alias is not None or model_field.alias is not None
        ) and model.model_config.get("validate_by_alias", True) is False:
            raise ToolValidationError(
                path=_json_path((*path, field_name)),
                message=(
                    "unsupported alias configuration: validate_by_alias=False "
                    "conflicts with generated validation schema"
                ),
            )
        if isinstance(validation_alias, (AliasPath, AliasChoices)):
            raise ToolValidationError(
                path=_json_path((*path, field_name)),
                message=(
                    "unsupported validation alias: complex alias paths and choices "
                    "cannot be represented faithfully in tool JSON Schema"
                ),
            )
        for input_path in _field_input_paths(model, field_name, model_field):
            input_name = input_path[0]
            if not isinstance(input_name, str):
                continue
            prior_field = input_name_fields.get(input_name)
            if prior_field is not None and prior_field != field_name:
                raise ToolValidationError(
                    path=_json_path((*path, field_name)),
                    message=(
                        "unsupported Pydantic input name collision: "
                        f"{input_name} maps to multiple fields"
                    ),
                )
            input_name_fields[input_name] = field_name

        provider_name = (
            validation_alias
            if isinstance(validation_alias, str)
            else model_field.alias or field_name
        )
        dump_name = (
            model_field.serialization_alias or model_field.alias or field_name
        )
        if dump_name != provider_name:
            raise ToolValidationError(
                path=_json_path((*path, field_name)),
                message=(
                    "unsupported Pydantic serialization customization: "
                    "serialization name differs from validation schema"
                ),
            )
        if model_field.exclude or model_field.exclude_if is not None:
            raise ToolValidationError(
                path=_json_path((*path, field_name)),
                message=(
                    "unsupported Pydantic serialization customization: "
                    "field exclusion"
                ),
            )
        _audit_json_schema_extra(
            model_field.json_schema_extra,
            path=(*path, field_name),
        )
        contains_model = _audit_pydantic_annotation(
            model_field.annotation,
            path=(*path, field_name),
            visited=visited,
            seen_aliases=set(),
        )
        _audit_pydantic_metadata(
            model_field.metadata,
            path=(*path, field_name),
            contains_model=contains_model,
        )
        if contains_model:
            structural_fields.add(field_name)

    first_structural_field = next(
        (
            field_name
            for field_name in model.model_fields
            if field_name in structural_fields
        ),
        None,
    )
    if (
        first_structural_field is None
        and "__pydantic_extra__" in structural_fields
    ):
        first_structural_field = "__pydantic_extra__"
    if (
        first_structural_field is not None
        and _pydantic_model_has_custom_lifecycle(model)
    ):
        raise ToolValidationError(
            path=_json_path((*path, first_structural_field)),
            message=(
                "unsupported Pydantic input shape: "
                "structural model lifecycle hook"
            ),
        )
    if (
        first_structural_field is None
        and _pydantic_model_has_unverifiable_lifecycle(model)
    ):
        raise ToolValidationError(
            path=_json_path(path),
            message="unsupported Pydantic input shape: model lifecycle hook",
        )

    decorators = model.__pydantic_decorators__
    if decorators.field_serializers:
        first_serializer = next(iter(decorators.field_serializers.values()))
        serialized_field = next(iter(first_serializer.info.fields), None)
        serializer_path = (
            (*path, serialized_field)
            if isinstance(serialized_field, str) and serialized_field != "*"
            else path
        )
        raise ToolValidationError(
            path=_json_path(serializer_path),
            message="unsupported Pydantic serialization customization: field serializer",
        )
    if decorators.model_serializers:
        raise ToolValidationError(
            path=_json_path(path),
            message="unsupported Pydantic serialization customization: model serializer",
        )
    if decorators.computed_fields:
        computed_field = next(iter(decorators.computed_fields))
        raise ToolValidationError(
            path=_json_path((*path, computed_field)),
            message="unsupported Pydantic serialization customization: computed field",
        )
    for decorator in decorators.field_validators.values():
        targeted_fields = set(decorator.info.fields)
        for field_name in model.model_fields:
            if field_name in structural_fields and (
                field_name in targeted_fields or "*" in targeted_fields
            ):
                raise ToolValidationError(
                    path=_json_path((*path, field_name)),
                    message="unsupported Pydantic input shape: structural validator",
                )
    if first_structural_field is not None and decorators.model_validators:
        raise ToolValidationError(
            path=_json_path((*path, first_structural_field)),
            message="unsupported Pydantic input shape: structural validator",
        )


def pydantic_input(
    model: type[BaseModel],
) -> tuple[dict[str, JsonValue], ValidateInput]:
    """Build a closed builtin schema and its canonical Pydantic validator."""

    if not isinstance(model, type) or not issubclass(model, BaseModel):
        raise TypeError("model must be a BaseModel subclass")
    _audit_pydantic_model(model)
    schema: _MutableJsonValue = model.model_json_schema(mode="validation")
    _close_pydantic_object_schemas(schema)
    if not isinstance(schema, dict):
        raise ToolValidationError(path="$", message="Pydantic schema must be an object")
    canonical_schema = _canonical_mapping(schema, subject="schema")
    schema_validator = _json_schema_validator(canonical_schema)

    def validate(arguments: Mapping[str, JsonValue]) -> Mapping[str, JsonValue]:
        try:
            validated = model.model_validate(arguments)
        except PydanticValidationError as error:
            raise _tool_error_from_pydantic(error) from None
        _reject_ignored_model_extras(validated, arguments, path=())
        provider_dumped = validated.model_dump(mode="json", by_alias=True)
        provider_arguments = _canonical_mapping(provider_dumped, subject="input")
        _validate_with_json_schema(schema_validator, provider_arguments)
        dumped = validated.model_dump(mode="json", by_alias=False)
        return _canonical_mapping(dumped, subject="input")

    return dict(canonical_schema), validate


def json_schema_input(schema: Mapping[str, JsonValue]) -> ValidateInput:
    """Build an input validator directly from a complete JSON Schema document."""

    validator = _json_schema_validator(schema)

    def validate(arguments: Mapping[str, JsonValue]) -> Mapping[str, JsonValue]:
        canonical_arguments = _canonical_mapping(arguments, subject="input")
        _validate_with_json_schema(validator, canonical_arguments)
        return canonical_arguments

    return validate


def json_schema_output(
    schema: Mapping[str, JsonValue] | None,
    value: JsonValue,
) -> JsonValue:
    """Canonicalize output and, when declared, validate its complete schema."""

    canonical_output = _canonical_json(value, subject="output")
    if schema is not None:
        _validate_with_json_schema(_json_schema_validator(schema), canonical_output)
    return canonical_output


_LOCAL_SIDE_EFFECTS = frozenset(
    {
        ToolEffect.WRITE_WORKSPACE,
        ToolEffect.EXECUTE_PROCESS,
        ToolEffect.DESTRUCTIVE,
    }
)


@dataclass(frozen=True, slots=True)
class Tool:
    definition: ToolDefinition
    validate_input: ValidateInput
    run: ToolRunner
    normalize_output: NormalizeOutput
    output_schema: Mapping[str, JsonValue] | None
    static_effects: frozenset[ToolEffect]
    resolve_use: ResolveToolUse
    execution_revision: str
    idempotent: bool
    concurrency_safe: bool
    cancellation_mode: CancellationMode
    interrupt_behavior: InterruptBehavior
    timeout_seconds: float
    max_model_output_bytes: int
    approval_profile: ToolApprovalProfile | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.definition, ToolDefinition):
            raise TypeError("definition must be a ToolDefinition")
        for field_name in ("validate_input", "run", "normalize_output", "resolve_use"):
            if not callable(getattr(self, field_name)):
                raise TypeError(f"{field_name} must be callable")

        static_effects = frozenset(self.static_effects)
        if any(not isinstance(effect, ToolEffect) for effect in static_effects):
            raise TypeError("static_effects must contain ToolEffect values")
        object.__setattr__(self, "static_effects", static_effects)
        if not isinstance(self.cancellation_mode, CancellationMode):
            raise TypeError("cancellation_mode must be a CancellationMode")
        if not isinstance(self.interrupt_behavior, InterruptBehavior):
            raise TypeError("interrupt_behavior must be an InterruptBehavior")
        if self.approval_profile is not None and not isinstance(
            self.approval_profile,
            ToolApprovalProfile,
        ):
            raise TypeError("approval_profile must be a ToolApprovalProfile or None")
        _require_exact_bool(self.idempotent, field_name="idempotent")
        _require_exact_bool(self.concurrency_safe, field_name="concurrency_safe")
        if (
            static_effects & _LOCAL_SIDE_EFFECTS
            and self.cancellation_mode is CancellationMode.NOT_CANCELLABLE
        ):
            raise ValueError(
                "local side-effecting tools cannot use cancellation mode not_cancellable"
            )

        _require_non_empty_string(
            self.execution_revision,
            field_name="execution_revision",
        )
        if isinstance(self.timeout_seconds, bool) or not isinstance(
            self.timeout_seconds,
            Real,
        ):
            raise TypeError("timeout_seconds must be a real number")
        if not math.isfinite(self.timeout_seconds) or self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be a positive finite number")
        if not isinstance(self.max_model_output_bytes, int) or isinstance(
            self.max_model_output_bytes, bool
        ):
            raise ValueError("max_model_output_bytes must be a positive integer")
        if self.max_model_output_bytes <= 0:
            raise ValueError("max_model_output_bytes must be a positive integer")
        if self.output_schema is not None:
            object.__setattr__(
                self,
                "output_schema",
                _freeze_mapping(self.output_schema, path="output_schema"),
            )
