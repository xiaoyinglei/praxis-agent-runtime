from __future__ import annotations

import warnings
from collections import deque
from collections.abc import Callable, Mapping
from collections.abc import Iterable as IterableABC
from collections.abc import Set as AbstractSet
from dataclasses import dataclass as stdlib_dataclass
from enum import StrEnum
from typing import Annotated, Any, Literal, TypeAliasType

import pytest
from pydantic import (
    AfterValidator,
    AliasChoices,
    AliasPath,
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    PlainSerializer,
    PlainValidator,
    RootModel,
    Strict,
    WrapSerializer,
    WrapValidator,
    computed_field,
    field_serializer,
    field_validator,
    model_serializer,
    model_validator,
)
from pydantic.dataclasses import dataclass as pydantic_dataclass
from pydantic_core import core_schema
from typing_extensions import TypedDict

from agent_runtime.tools.tool import (
    JsonValue,
    ToolValidationError,
    _reject_ignored_extras_in_value,
    json_schema_input,
    json_schema_output,
    pydantic_input,
)


class BuiltinArguments(BaseModel):
    count: int = Field(ge=5, le=10)
    mode: Literal["fast", "safe"]


class ExtensibleArguments(BaseModel):
    model_config = ConfigDict(extra="allow")

    query: str


class TypedExtraArguments(BaseModel):
    __pydantic_extra__: dict[str, int] = Field(init=False)
    model_config = ConfigDict(extra="allow")

    query: str


class StructuralExtraChild(BaseModel):
    label: str


class StructuralTypedExtraArguments(BaseModel):
    __pydantic_extra__: dict[str, StructuralExtraChild] = Field(init=False)
    model_config = ConfigDict(extra="allow")


class DroppingStructuralExtraValidatorArguments(BaseModel):
    __pydantic_extra__: dict[str, StructuralExtraChild] = Field(init=False)
    model_config = ConfigDict(extra="allow")

    @model_validator(mode="after")
    def drop_structural_extras(self) -> DroppingStructuralExtraValidatorArguments:
        self.__pydantic_extra__.clear()
        return self


class DroppingStructuralExtraLifecycleBase(BaseModel):
    @classmethod
    def model_validate(cls, obj: Any, **kwargs: Any) -> Any:
        validated = super().model_validate(obj, **kwargs)
        validated.__pydantic_extra__ = {}
        return validated


class DroppingStructuralExtraLifecycleArguments(
    DroppingStructuralExtraLifecycleBase
):
    __pydantic_extra__: dict[str, StructuralExtraChild] = Field(init=False)
    model_config = ConfigDict(extra="allow")


class AliasedArguments(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    internal_name: str = Field(alias="externalName")


class AliasDisabledArguments(BaseModel):
    model_config = ConfigDict(validate_by_alias=False, validate_by_name=True)

    internal_name: str = Field(alias="externalName")


class ParentWithAliasDisabledArguments(BaseModel):
    nested: AliasDisabledArguments


class SerializeByAliasArguments(BaseModel):
    model_config = ConfigDict(serialize_by_alias=True)

    internal_name: str = Field(alias="externalName")


class DuplicateAliasArguments(BaseModel):
    first: str = Field(alias="shared")
    second: str = Field(alias="shared")


class AliasCanonicalCollisionArguments(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    first: str = Field(alias="second")
    second: str


class ParentWithDuplicateAliasArguments(BaseModel):
    nested: DuplicateAliasArguments


class ParentWithAliasCanonicalCollisionArguments(BaseModel):
    nested: AliasCanonicalCollisionArguments


class NestedArguments(BaseModel):
    label: str


class ParentArguments(BaseModel):
    nested: NestedArguments


class ExtensibleNestedArguments(BaseModel):
    model_config = ConfigDict(extra="allow")

    label: str


class ParentWithExtensibleNestedArguments(BaseModel):
    nested: ExtensibleNestedArguments


class ExtensibleParentWithClosedNestedArguments(BaseModel):
    model_config = ConfigDict(extra="allow")

    nested: NestedArguments


class MappingArguments(BaseModel):
    children: dict[int, NestedArguments]


class AliasPathArguments(BaseModel):
    value: str = Field(validation_alias=AliasPath("payload", "value"))


class AliasChoicesWithPathArguments(BaseModel):
    value: str = Field(
        validation_alias=AliasChoices(
            "value",
            AliasPath("payload", "value"),
        )
    )


class InvalidGeneratedSchemaArguments(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={"type": "not-a-json-schema-type"}
    )

    value: str


class ExtensibleAliasedArguments(BaseModel):
    model_config = ConfigDict(extra="allow")

    internal: str = Field(alias="external")


class ParentWithExtensibleAliasedArguments(BaseModel):
    nested: ExtensibleAliasedArguments


@pydantic_dataclass
class DataclassArguments:
    label: str


class ParentWithDataclassArguments(BaseModel):
    nested: DataclassArguments


@stdlib_dataclass
class StandardDataclassArguments:
    label: str


class ParentWithStandardDataclassArguments(BaseModel):
    nested: StandardDataclassArguments


class FrozenNestedArguments(BaseModel):
    model_config = ConfigDict(frozen=True)

    label: str


class SetArguments(BaseModel):
    children: set[FrozenNestedArguments]


class FrozenSetArguments(BaseModel):
    children: frozenset[FrozenNestedArguments]


class ListRootArguments(RootModel[list[str]]):
    pass


class DictRootArguments(RootModel[dict[str, str]]):
    pass


class TypedDictArguments(TypedDict):
    label: str


class ParentWithTypedDictArguments(BaseModel):
    nested: TypedDictArguments


class SecretValidatorArguments(BaseModel):
    secret: str

    @field_validator("secret")
    @classmethod
    def reject_secret(cls, value: str) -> str:
        raise ValueError(f"rejected runtime value TOP_SECRET_123: {value}")


class DequeArguments(BaseModel):
    children: deque[NestedArguments]


class IterableArguments(BaseModel):
    children: IterableABC[NestedArguments]


class AbstractSetArguments(BaseModel):
    children: AbstractSet[FrozenNestedArguments]


AliasSetArgumentsType = TypeAliasType(  # noqa: UP040
    "AliasSetArgumentsType",
    set[FrozenNestedArguments],
)


class TypeAliasSetArguments(BaseModel):
    children: AliasSetArgumentsType


class DroppingListArguments(BaseModel):
    children: list[NestedArguments]

    @field_validator("children", mode="after")
    @classmethod
    def drop_last_child(
        cls,
        value: list[NestedArguments],
    ) -> list[NestedArguments]:
        return value[:-1]


class DroppingTupleArguments(BaseModel):
    children: tuple[NestedArguments, ...]

    @field_validator("children", mode="after")
    @classmethod
    def drop_last_child(
        cls,
        value: tuple[NestedArguments, ...],
    ) -> tuple[NestedArguments, ...]:
        return value[:-1]


class UnionWithDataclassArguments(BaseModel):
    nested: NestedArguments | DataclassArguments


class ListUnionWithDataclassArguments(BaseModel):
    nested: list[NestedArguments | DataclassArguments]


class TupleWithDataclassArguments(BaseModel):
    nested: tuple[NestedArguments, DataclassArguments]


class AlternateNestedArguments(BaseModel):
    alternate: str


class ReverseStructuralMetadata:
    def __get_pydantic_core_schema__(
        self,
        source_type: Any,
        handler: Any,
    ) -> Any:
        schema = handler(source_type)
        return core_schema.no_info_after_validator_function(
            lambda value: list(reversed(value)),
            schema,
        )


class CoreSchemaMetadataArguments(BaseModel):
    children: Annotated[
        list[NestedArguments | AlternateNestedArguments],
        ReverseStructuralMetadata(),
    ]


class FieldSerializerArguments(BaseModel):
    value: int

    @field_serializer("value")
    def serialize_value(self, value: int) -> str:
        return str(value)


class ModelSerializerArguments(BaseModel):
    value: int

    @model_serializer
    def serialize_model(self) -> dict[str, Any]:
        return {"renamed": self.value}


class ComputedFieldArguments(BaseModel):
    value: int

    @computed_field
    @property
    def doubled(self) -> int:
        return self.value * 2


class ExcludedFieldArguments(BaseModel):
    value: int = Field(exclude=True)


class ConditionallyExcludedFieldArguments(BaseModel):
    value: int = Field(exclude_if=lambda value: value == 0)


class InheritedSerializerBase(BaseModel):
    value: int

    @field_serializer("value")
    def serialize_value(self, value: int) -> str:
        return str(value)


class InheritedSerializerArguments(InheritedSerializerBase):
    label: str


class CustomModelDumpBase(BaseModel):
    def model_dump(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        dumped = super().model_dump(*args, **kwargs)
        dumped.pop("value", None)
        return dumped


class CustomModelDumpArguments(CustomModelDumpBase):
    value: int


with warnings.catch_warnings():
    warnings.simplefilter("ignore", DeprecationWarning)

    class JsonEncoderArguments(BaseModel):
        model_config = ConfigDict(json_encoders={int: str})

        value: int


class DivergentSerializationAliasArguments(BaseModel):
    value: int = Field(
        validation_alias="providerValue",
        serialization_alias="dumpedValue",
    )


class FieldSchemaOverrideArguments(BaseModel):
    value: int = Field(json_schema_extra={"type": "string"})


class ModelSchemaOverrideArguments(BaseModel):
    model_config = ConfigDict(json_schema_extra={"type": "string"})

    value: int


def _force_string_schema(schema: dict[str, Any]) -> None:
    schema["type"] = "string"


class CallableSchemaOverrideArguments(BaseModel):
    value: int = Field(json_schema_extra=_force_string_schema)


class JsonSchemaMetadata:
    def __get_pydantic_json_schema__(
        self,
        core: Any,
        handler: Any,
    ) -> Any:
        schema = handler(core)
        schema["type"] = "string"
        return schema


class MetadataSchemaOverrideArguments(BaseModel):
    value: Annotated[int, JsonSchemaMetadata()]


class InheritedJsonSchemaHookBase(BaseModel):
    @classmethod
    def __get_pydantic_json_schema__(
        cls,
        core: Any,
        handler: Any,
    ) -> Any:
        schema = handler(core)
        schema["type"] = "string"
        return schema


class InheritedJsonSchemaHookArguments(InheritedJsonSchemaHookBase):
    value: int


class ParentWithInheritedJsonSchemaHookArguments(BaseModel):
    nested: InheritedJsonSchemaHookArguments


class CustomModelJsonSchemaBase(BaseModel):
    @classmethod
    def model_json_schema(cls, *args: Any, **kwargs: Any) -> dict[str, Any]:
        schema = super().model_json_schema(*args, **kwargs)
        schema["type"] = "string"
        return schema


class CustomModelJsonSchemaArguments(CustomModelJsonSchemaBase):
    value: int


class SchemaModeOverrideArguments(BaseModel):
    model_config = ConfigDict(json_schema_mode_override="serialization")

    value: int


with warnings.catch_warnings():
    warnings.simplefilter("ignore", DeprecationWarning)

    class CustomSchemaGeneratorArguments(BaseModel):
        model_config = ConfigDict(schema_generator=object)

        value: int


class DocumentationSchemaMetadataArguments(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "title": "Documented arguments",
            "description": "Documentation only",
            "examples": [{"value": 1}],
            "default": {"value": 1},
            "x-contract-note": "safe",
        }
    )

    value: int = Field(
        default=1,
        title="Documented value",
        description="Documentation only",
        examples=[1],
        json_schema_extra={"x-field-note": "safe"},
    )


class UnvalidatedDefaultArguments(BaseModel):
    count: int = Field(default=0, ge=1)


class DefaultEncodedState(StrEnum):
    READY = "ready"


class DefaultEncodedEnumArguments(BaseModel):
    state: DefaultEncodedState


class CoreSchemaSerializedState(StrEnum):
    READY = "ready"

    @classmethod
    def __get_pydantic_core_schema__(
        cls,
        source_type: Any,
        handler: Any,
    ) -> Any:
        schema = handler(source_type)
        schema["serialization"] = core_schema.plain_serializer_function_ser_schema(
            lambda value, info: (
                value.value if info.by_alias else "outside-schema"
            ),
            info_arg=True,
            return_schema=core_schema.str_schema(),
        )
        return schema


class CoreSchemaSerializedEnumArguments(BaseModel):
    state: CoreSchemaSerializedState


def _serialize_extra_by_alias(value: int, info: Any) -> int | str:
    return value if info.by_alias else str(value)


class SerializedTypedExtraArguments(BaseModel):
    __pydantic_extra__: dict[
        str,
        Annotated[int, PlainSerializer(_serialize_extra_by_alias)],
    ] = Field(init=False)
    model_config = ConfigDict(extra="allow")


class ScalarPlainSerializerArguments(BaseModel):
    value: Annotated[
        int,
        PlainSerializer(lambda value: value + 1, return_type=int),
    ]


class ScalarWrapSerializerArguments(BaseModel):
    value: Annotated[
        int,
        WrapSerializer(
            lambda value, handler: handler(value) + 1,
            return_type=int,
        ),
    ]


class UnknownCoreSchemaMetadata:
    def __get_pydantic_core_schema__(
        self,
        source_type: Any,
        handler: Any,
    ) -> Any:
        schema = handler(source_type)
        schema["serialization"] = core_schema.plain_serializer_function_ser_schema(
            lambda value: value + 1,
            return_schema=core_schema.int_schema(),
        )
        return schema


class ScalarUnknownCoreSchemaArguments(BaseModel):
    value: Annotated[int, UnknownCoreSchemaMetadata()]


def _increment_before(value: Any) -> int:
    return int(value) + 1


def _increment_after(value: int) -> int:
    return value + 1


def _increment_plain(value: Any) -> int:
    return int(value) + 1


def _increment_wrap(value: Any, handler: Any) -> int:
    return handler(value) + 1


class ScalarBeforeValidatorArguments(BaseModel):
    value: Annotated[int, BeforeValidator(_increment_before)]


class ScalarAfterValidatorArguments(BaseModel):
    value: Annotated[int, AfterValidator(_increment_after)]


class ScalarPlainValidatorArguments(BaseModel):
    value: Annotated[int, PlainValidator(_increment_plain)]


class ScalarWrapValidatorArguments(BaseModel):
    value: Annotated[int, WrapValidator(_increment_wrap)]


def _reject_metadata_secret(value: str) -> str:
    raise ValueError(f"rejected metadata TOP_SECRET_456: {value}")


class SecretMetadataValidatorArguments(BaseModel):
    value: Annotated[str, AfterValidator(_reject_metadata_secret)]


class ScalarFieldAndStrictArguments(BaseModel):
    value: Annotated[int, Strict()] = Field(ge=1, le=3)


class LifecycleNestedA(BaseModel):
    kind: Literal["a"]
    payload: str


class LifecycleNestedB(BaseModel):
    kind: Literal["b"]
    payload: str
    discarded: bool = False


def _reverse_structural_children(model: Any) -> Any:
    model.children.reverse()
    return model


class CoreSchemaLifecycleArguments(BaseModel):
    children: list[LifecycleNestedA | LifecycleNestedB]

    @classmethod
    def __get_pydantic_core_schema__(
        cls,
        source_type: Any,
        handler: Any,
    ) -> Any:
        schema = handler(source_type)
        return core_schema.no_info_after_validator_function(
            _reverse_structural_children,
            schema,
        )


class CustomInitLifecycleArguments(BaseModel):
    children: list[LifecycleNestedA | LifecycleNestedB]

    def __init__(self, **data: Any) -> None:
        super().__init__(**data)
        self.children.reverse()


class PostInitLifecycleArguments(BaseModel):
    children: list[LifecycleNestedA | LifecycleNestedB]

    def model_post_init(self, context: Any) -> None:
        self.children.reverse()


class ModelValidateLifecycleArguments(BaseModel):
    children: list[LifecycleNestedA | LifecycleNestedB]

    @classmethod
    def model_validate(cls, obj: Any, **kwargs: Any) -> Any:
        validated = super().model_validate(obj, **kwargs)
        validated.children.reverse()
        return validated


class InheritedCoreSchemaLifecycleBase(BaseModel):
    @classmethod
    def __get_pydantic_core_schema__(
        cls,
        source_type: Any,
        handler: Any,
    ) -> Any:
        schema = handler(source_type)
        return core_schema.no_info_after_validator_function(
            _reverse_structural_children,
            schema,
        )


class InheritedCoreSchemaLifecycleArguments(InheritedCoreSchemaLifecycleBase):
    children: list[LifecycleNestedA | LifecycleNestedB]


class InheritedModelValidateLifecycleBase(BaseModel):
    @classmethod
    def model_validate(cls, obj: Any, **kwargs: Any) -> Any:
        validated = super().model_validate(obj, **kwargs)
        validated.children.reverse()
        return validated


class InheritedModelValidateLifecycleArguments(
    InheritedModelValidateLifecycleBase
):
    children: list[LifecycleNestedA | LifecycleNestedB]


class InheritedLegacyLifecycleBase(BaseModel):
    @classmethod
    def __get_validators__(cls) -> Any:
        yield _reverse_structural_children


class InheritedLegacyLifecycleArguments(InheritedLegacyLifecycleBase):
    children: list[LifecycleNestedA | LifecycleNestedB]


class DeclarativeInheritanceBase(BaseModel):
    inherited_value: str


class DeclarativeInheritedStructuralArguments(DeclarativeInheritanceBase):
    children: list[LifecycleNestedA | LifecycleNestedB]


def _reverse_scalar_value(model: Any) -> Any:
    model.value = model.value[::-1]
    return model


class ScalarCoreSchemaLifecycleArguments(BaseModel):
    value: str

    @classmethod
    def __get_pydantic_core_schema__(
        cls,
        source_type: Any,
        handler: Any,
    ) -> Any:
        schema = handler(source_type)
        return core_schema.no_info_after_validator_function(
            _reverse_scalar_value,
            schema,
        )


class DivergentScalarCoreSchemaLifecycleArguments(BaseModel):
    value: str

    @classmethod
    def __get_pydantic_core_schema__(
        cls,
        source_type: Any,
        handler: Any,
    ) -> Any:
        schema = handler(source_type)
        schema["serialization"] = core_schema.plain_serializer_function_ser_schema(
            lambda model, info: {
                "value": model.value if info.by_alias else 123,
            },
            info_arg=True,
        )
        return schema


class ScalarCustomInitLifecycleArguments(BaseModel):
    value: str

    def __init__(self, **data: Any) -> None:
        super().__init__(**data)
        self.value = self.value[::-1]


class ScalarPostInitLifecycleArguments(BaseModel):
    value: str

    def model_post_init(self, context: Any) -> None:
        self.value = self.value[::-1]


class ScalarModelValidateLifecycleArguments(BaseModel):
    value: str

    @classmethod
    def model_validate(cls, obj: Any, **kwargs: Any) -> Any:
        validated = super().model_validate(obj, **kwargs)
        validated.value = validated.value[::-1]
        return validated


class DivergentModelValidateResult(BaseModel):
    value: str

    def model_dump(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        if kwargs.get("by_alias"):
            return {"value": self.value}
        return {"value": 123}


class DivergentScalarModelValidateLifecycleArguments(BaseModel):
    value: str

    @classmethod
    def model_validate(cls, obj: Any, **kwargs: Any) -> Any:
        return DivergentModelValidateResult.model_validate(obj, **kwargs)


def _raw_input_validator() -> Callable[
    [Mapping[str, JsonValue]], Mapping[str, JsonValue]
]:
    return json_schema_input(
        {
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "type": "object",
            "properties": {
                "count": {"type": "integer", "minimum": 5, "maximum": 10},
                "mode": {"enum": ["fast", "safe"]},
            },
            "required": ["count", "mode"],
            "additionalProperties": False,
        }
    )


def test_pydantic_input_returns_closed_schema_and_canonical_arguments() -> None:
    schema, validate = pydantic_input(BuiltinArguments)
    original = {"count": 7, "mode": "fast"}

    arguments = validate(original)
    original["count"] = 8

    assert schema["additionalProperties"] is False
    assert schema["required"] == ("count", "mode")
    assert arguments == {"count": 7, "mode": "fast"}
    with pytest.raises(TypeError):
        arguments["count"] = 9  # type: ignore[index]


@pytest.mark.parametrize(
    ("arguments", "path"),
    [
        pytest.param({"mode": "fast"}, "$.count", id="required"),
        pytest.param({"count": 4, "mode": "fast"}, "$.count", id="minimum"),
        pytest.param({"count": 11, "mode": "fast"}, "$.count", id="maximum"),
        pytest.param({"count": 7, "mode": "turbo"}, "$.mode", id="enum"),
        pytest.param(
            {"count": 7, "mode": "fast", "discarded": True},
            "$.discarded",
            id="extra-is-not-silently-dropped",
        ),
    ],
)
def test_pydantic_input_rejects_invalid_arguments(
    arguments: dict[str, Any],
    path: str,
) -> None:
    _, validate = pydantic_input(BuiltinArguments)

    with pytest.raises(ToolValidationError) as error:
        validate(arguments)

    assert error.value.path == path
    assert len(error.value.message) <= 512


def test_pydantic_input_preserves_extras_only_when_model_explicitly_allows_them() -> None:
    schema, validate = pydantic_input(ExtensibleArguments)

    arguments = validate({"query": "status", "limit": 3})

    assert schema["additionalProperties"] is True
    assert arguments == {"query": "status", "limit": 3}


def test_pydantic_input_preserves_schema_safe_typed_extras() -> None:
    schema, validate = pydantic_input(TypedExtraArguments)

    arguments = validate({"query": "status", "limit": "3"})  # type: ignore[dict-item]

    assert schema["additionalProperties"] == {"type": "integer"}
    assert arguments == {"query": "status", "limit": 3}


def test_pydantic_input_preserves_schema_safe_structural_typed_extras() -> None:
    schema, validate = pydantic_input(StructuralTypedExtraArguments)

    arguments = validate({"bonus": {"label": "kept"}})

    assert schema["additionalProperties"] == {
        "$ref": "#/$defs/StructuralExtraChild"
    }
    assert arguments == {"bonus": {"label": "kept"}}


def test_pydantic_input_returns_coerced_canonical_arguments() -> None:
    _, validate = pydantic_input(BuiltinArguments)

    arguments = validate({"count": "7", "mode": "fast"})  # type: ignore[dict-item]

    assert arguments == {"count": 7, "mode": "fast"}


@pytest.mark.parametrize(
    "arguments",
    [
        pytest.param({"externalName": "status"}, id="schema-alias"),
        pytest.param({"internal_name": "status"}, id="python-field-name"),
    ],
)
def test_pydantic_input_accepts_aliases_and_dumps_canonical_field_names(
    arguments: dict[str, str],
) -> None:
    _, validate = pydantic_input(AliasedArguments)

    assert validate(arguments) == {"internal_name": "status"}


@pytest.mark.parametrize(
    ("model", "path"),
    [
        pytest.param(AliasDisabledArguments, "$.internal_name", id="top-level"),
        pytest.param(
            ParentWithAliasDisabledArguments,
            "$.nested.internal_name",
            id="nested",
        ),
    ],
)
def test_pydantic_input_rejects_aliases_when_alias_validation_is_disabled(
    model: type[BaseModel],
    path: str,
) -> None:
    properties = AliasDisabledArguments.model_json_schema(mode="validation")[
        "properties"
    ]
    assert "externalName" in properties

    with pytest.raises(ToolValidationError) as error:
        pydantic_input(model)

    assert error.value.path == path
    assert "validate_by_alias" in error.value.message
    assert len(error.value.message) <= 512


def test_pydantic_input_ignores_serialize_by_alias_for_canonical_dump() -> None:
    schema, validate = pydantic_input(SerializeByAliasArguments)

    properties = schema["properties"]
    assert "externalName" in properties  # type: ignore[operator]
    assert "internal_name" not in properties  # type: ignore[operator]
    assert validate({"externalName": "status"}) == {"internal_name": "status"}


@pytest.mark.parametrize(
    ("model", "path"),
    [
        pytest.param(DuplicateAliasArguments, "$.second", id="duplicate-top-level"),
        pytest.param(
            ParentWithDuplicateAliasArguments,
            "$.nested.second",
            id="duplicate-nested",
        ),
        pytest.param(
            AliasCanonicalCollisionArguments,
            "$.second",
            id="alias-canonical-top-level",
        ),
        pytest.param(
            ParentWithAliasCanonicalCollisionArguments,
            "$.nested.second",
            id="alias-canonical-nested",
        ),
    ],
)
def test_pydantic_input_rejects_accepted_input_name_collisions(
    model: type[BaseModel],
    path: str,
) -> None:
    with pytest.raises(ToolValidationError) as error:
        pydantic_input(model)

    assert error.value.path == path
    assert "input name collision" in error.value.message
    assert len(error.value.message) <= 512


@pytest.mark.parametrize(
    ("model", "path"),
    [
        pytest.param(FieldSerializerArguments, "$.value", id="field-serializer"),
        pytest.param(ModelSerializerArguments, "$", id="model-serializer"),
        pytest.param(ComputedFieldArguments, "$.doubled", id="computed-field"),
        pytest.param(ExcludedFieldArguments, "$.value", id="exclude"),
        pytest.param(
            ConditionallyExcludedFieldArguments,
            "$.value",
            id="exclude-if",
        ),
        pytest.param(
            InheritedSerializerArguments,
            "$.value",
            id="inherited-field-serializer",
        ),
        pytest.param(CustomModelDumpArguments, "$", id="inherited-model-dump"),
        pytest.param(JsonEncoderArguments, "$", id="json-encoders"),
        pytest.param(
            DivergentSerializationAliasArguments,
            "$.value",
            id="serialization-alias",
        ),
    ],
)
def test_pydantic_input_rejects_custom_serialization_surfaces(
    model: type[BaseModel],
    path: str,
) -> None:
    with pytest.raises(ToolValidationError) as error:
        pydantic_input(model)

    assert error.value.path == path
    assert "serialization" in error.value.message.lower()
    assert len(error.value.message) <= 512


@pytest.mark.parametrize(
    ("model", "path"),
    [
        pytest.param(
            FieldSchemaOverrideArguments,
            "$.value.type",
            id="field-extra",
        ),
        pytest.param(ModelSchemaOverrideArguments, "$.type", id="model-extra"),
        pytest.param(
            CallableSchemaOverrideArguments,
            "$.value",
            id="callable-field-extra",
        ),
        pytest.param(
            MetadataSchemaOverrideArguments,
            "$.value",
            id="metadata-hook",
        ),
        pytest.param(
            InheritedJsonSchemaHookArguments,
            "$",
            id="inherited-class-hook",
        ),
        pytest.param(
            ParentWithInheritedJsonSchemaHookArguments,
            "$.nested",
            id="nested-inherited-class-hook",
        ),
        pytest.param(
            CustomModelJsonSchemaArguments,
            "$",
            id="inherited-model-json-schema",
        ),
        pytest.param(SchemaModeOverrideArguments, "$", id="mode-override"),
        pytest.param(CustomSchemaGeneratorArguments, "$", id="schema-generator"),
    ],
)
def test_pydantic_input_rejects_validation_affecting_schema_customization(
    model: type[BaseModel],
    path: str,
) -> None:
    with pytest.raises(ToolValidationError) as error:
        pydantic_input(model)

    assert error.value.path == path
    assert "schema customization" in error.value.message.lower()
    assert len(error.value.message) <= 512


def test_pydantic_input_allows_documentation_only_schema_metadata() -> None:
    schema, validate = pydantic_input(DocumentationSchemaMetadataArguments)

    assert schema["x-contract-note"] == "safe"
    field_schema = schema["properties"]["value"]  # type: ignore[index]
    assert field_schema["x-field-note"] == "safe"  # type: ignore[index]
    assert validate({}) == {"value": 1}


def test_pydantic_input_validates_provider_dump_not_raw_coercion_input() -> None:
    _, validate = pydantic_input(UnvalidatedDefaultArguments)

    with pytest.raises(ToolValidationError) as error:
        validate({})

    assert error.value.path == "$.count"
    assert "minimum" in error.value.message


def test_pydantic_input_preserves_default_enum_json_encoding() -> None:
    _, validate = pydantic_input(DefaultEncodedEnumArguments)

    assert validate({"state": "ready"}) == {"state": "ready"}


def test_pydantic_input_rejects_enum_core_schema_serialization_hook() -> None:
    with pytest.raises(ToolValidationError) as error:
        pydantic_input(CoreSchemaSerializedEnumArguments)

    assert error.value.path == "$.state"
    assert "serialization" in error.value.message.lower()
    assert len(error.value.message) <= 512


def test_pydantic_input_rejects_typed_extra_serialization_metadata() -> None:
    with pytest.raises(ToolValidationError) as error:
        pydantic_input(SerializedTypedExtraArguments)

    assert error.value.path == "$.__pydantic_extra__"
    assert "serialization metadata" in error.value.message.lower()
    assert len(error.value.message) <= 512


@pytest.mark.parametrize(
    "model",
    [
        pytest.param(ScalarPlainSerializerArguments, id="plain-serializer"),
        pytest.param(ScalarWrapSerializerArguments, id="wrap-serializer"),
        pytest.param(
            ScalarUnknownCoreSchemaArguments,
            id="unknown-core-schema-metadata",
        ),
    ],
)
def test_pydantic_input_rejects_scalar_serialization_metadata(
    model: type[BaseModel],
) -> None:
    with pytest.raises(ToolValidationError) as error:
        pydantic_input(model)

    assert error.value.path == "$.value"
    assert "serialization metadata" in error.value.message
    assert len(error.value.message) <= 512


@pytest.mark.parametrize(
    ("model", "arguments"),
    [
        pytest.param(
            ScalarBeforeValidatorArguments,
            {"value": "1"},
            id="before-validator",
        ),
        pytest.param(
            ScalarAfterValidatorArguments,
            {"value": 1},
            id="after-validator",
        ),
        pytest.param(
            ScalarPlainValidatorArguments,
            {"value": "1"},
            id="plain-validator",
        ),
        pytest.param(
            ScalarWrapValidatorArguments,
            {"value": 1},
            id="wrap-validator",
        ),
    ],
)
def test_pydantic_input_allows_known_scalar_validation_metadata(
    model: type[BaseModel],
    arguments: dict[str, Any],
) -> None:
    _, validate = pydantic_input(model)

    assert validate(arguments) == {"value": 2}  # type: ignore[arg-type]


def test_pydantic_input_redacts_scalar_metadata_validator_errors() -> None:
    _, validate = pydantic_input(SecretMetadataValidatorArguments)

    with pytest.raises(ToolValidationError) as error:
        validate({"value": "TOP_SECRET_456"})

    assert "TOP_SECRET_456" not in error.value.message
    assert len(error.value.message) <= 512


def test_pydantic_input_allows_ordinary_field_and_strict_constraints() -> None:
    _, validate = pydantic_input(ScalarFieldAndStrictArguments)

    assert validate({"value": 2}) == {"value": 2}


def test_pydantic_input_rejects_nested_unknown_arguments() -> None:
    schema, validate = pydantic_input(ParentArguments)

    nested_schema = schema["$defs"]["NestedArguments"]  # type: ignore[index]
    assert nested_schema["additionalProperties"] is False  # type: ignore[index]
    with pytest.raises(ToolValidationError) as error:
        validate({"nested": {"label": "ready", "discarded": True}})  # type: ignore[dict-item]

    assert error.value.path == "$.nested.discarded"


def test_pydantic_input_preserves_explicitly_allowed_nested_extras() -> None:
    schema, validate = pydantic_input(ParentWithExtensibleNestedArguments)

    nested_schema = schema["$defs"]["ExtensibleNestedArguments"]  # type: ignore[index]
    assert nested_schema["additionalProperties"] is True  # type: ignore[index]
    assert validate({"nested": {"label": "ready", "detail": "kept"}}) == {  # type: ignore[dict-item]
        "nested": {"label": "ready", "detail": "kept"}
    }


def test_pydantic_input_rejects_ignored_nested_extras_under_extensible_parent() -> None:
    schema, validate = pydantic_input(ExtensibleParentWithClosedNestedArguments)

    nested_schema = schema["$defs"]["NestedArguments"]  # type: ignore[index]
    assert nested_schema["additionalProperties"] is False  # type: ignore[index]
    with pytest.raises(ToolValidationError) as error:
        validate(
            {
                "nested": {"label": "ready", "discarded": True},
                "root_detail": "kept",
            }  # type: ignore[dict-item]
        )

    assert error.value.path == "$.nested.discarded"


def test_pydantic_input_rejects_mapping_key_normalization_collisions() -> None:
    _, validate = pydantic_input(MappingArguments)

    with pytest.raises(ToolValidationError) as error:
        validate(
            {
                "children": {
                    "1": {"label": "first"},
                    "01": {"label": "second", "discarded": True},
                }
            }  # type: ignore[dict-item]
        )

    assert error.value.path == "$.children"
    assert "mapping keys" in error.value.message
    assert len(error.value.message) <= 512


@pytest.mark.parametrize(
    "model",
    [
        pytest.param(AliasPathArguments, id="alias-path"),
        pytest.param(AliasChoicesWithPathArguments, id="alias-choices-path"),
    ],
)
def test_pydantic_input_rejects_lossy_validation_aliases(
    model: type[BaseModel],
) -> None:
    with pytest.raises(ToolValidationError) as error:
        pydantic_input(model)

    assert error.value.path == "$.value"
    assert "unsupported validation alias" in error.value.message
    assert len(error.value.message) <= 512


def test_pydantic_input_rejects_invalid_generated_schema() -> None:
    with pytest.raises(ToolValidationError) as error:
        pydantic_input(InvalidGeneratedSchemaArguments)

    assert error.value.path == "$.type"
    assert len(error.value.message) <= 512


@pytest.mark.parametrize(
    ("model", "arguments", "path"),
    [
        pytest.param(
            ExtensibleAliasedArguments,
            {"external": "FIELD", "internal": "EXTRA"},
            "$.internal",
            id="top-level",
        ),
        pytest.param(
            ParentWithExtensibleAliasedArguments,
            {"nested": {"external": "FIELD", "internal": "EXTRA"}},
            "$.nested.internal",
            id="nested",
        ),
    ],
)
def test_pydantic_input_rejects_alias_extra_output_collisions(
    model: type[BaseModel],
    arguments: dict[str, Any],
    path: str,
) -> None:
    _, validate = pydantic_input(model)

    with pytest.raises(ToolValidationError) as error:
        validate(arguments)

    assert error.value.path == path
    assert error.value.message == "extra key collides with canonical field name"


@pytest.mark.parametrize(
    ("model", "path"),
    [
        pytest.param(ParentWithDataclassArguments, "$.nested", id="dataclass"),
        pytest.param(
            ParentWithStandardDataclassArguments,
            "$.nested",
            id="standard-dataclass",
        ),
        pytest.param(SetArguments, "$.children", id="set"),
        pytest.param(FrozenSetArguments, "$.children", id="frozenset"),
        pytest.param(ListRootArguments, "$", id="list-root-model"),
        pytest.param(DictRootArguments, "$", id="dict-root-model"),
        pytest.param(ParentWithTypedDictArguments, "$.nested", id="typed-dict"),
    ],
)
def test_pydantic_input_rejects_unsupported_lossy_shapes(
    model: type[BaseModel],
    path: str,
) -> None:
    with pytest.raises(ToolValidationError) as error:
        pydantic_input(model)

    assert error.value.path == path
    assert "unsupported Pydantic input shape" in error.value.message
    assert len(error.value.message) <= 512


def test_json_schema_input_redacts_runtime_values() -> None:
    validate = json_schema_input(
        {
            "type": "object",
            "properties": {"secret": {"type": "integer"}},
            "required": ["secret"],
        }
    )

    with pytest.raises(ToolValidationError) as error:
        validate({"secret": "TOP_SECRET_123"})

    assert error.value.path == "$.secret"
    assert error.value.message == "type: validation failed"
    assert "TOP_SECRET_123" not in str(error.value)


def test_json_schema_output_redacts_runtime_values() -> None:
    schema = {
        "type": "object",
        "properties": {"secret": {"type": "integer"}},
        "required": ["secret"],
    }

    with pytest.raises(ToolValidationError) as error:
        json_schema_output(schema, {"secret": "TOP_SECRET_123"})  # type: ignore[arg-type]

    assert error.value.path == "$.secret"
    assert error.value.message == "type: validation failed"
    assert "TOP_SECRET_123" not in str(error.value)


def test_pydantic_input_redacts_custom_validator_messages_and_values() -> None:
    _, validate = pydantic_input(SecretValidatorArguments)

    with pytest.raises(ToolValidationError) as error:
        validate({"secret": "TOP_SECRET_123"})

    assert error.value.path == "$.secret"
    assert error.value.message == "value_error: validation failed"
    assert "TOP_SECRET_123" not in str(error.value)


@pytest.mark.parametrize(
    "model",
    [
        pytest.param(DequeArguments, id="deque"),
        pytest.param(IterableArguments, id="iterable-abc"),
        pytest.param(AbstractSetArguments, id="set-abc"),
        pytest.param(TypeAliasSetArguments, id="type-alias-set"),
    ],
)
def test_pydantic_input_rejects_composites_outside_lossless_grammar(
    model: type[BaseModel],
) -> None:
    with pytest.raises(ToolValidationError) as error:
        pydantic_input(model)

    assert error.value.path == "$.children"
    assert "unsupported Pydantic input shape" in error.value.message


@pytest.mark.parametrize(
    "model",
    [
        pytest.param(DroppingListArguments, id="list"),
        pytest.param(DroppingTupleArguments, id="tuple"),
    ],
)
def test_pydantic_input_rejects_structural_field_validators(
    model: type[BaseModel],
) -> None:
    with pytest.raises(ToolValidationError) as error:
        pydantic_input(model)

    assert error.value.path == "$.children"
    assert "structural validator" in error.value.message


def test_pydantic_input_rejects_structural_typed_extra_model_validator() -> None:
    with pytest.raises(ToolValidationError) as error:
        pydantic_input(DroppingStructuralExtraValidatorArguments)

    assert error.value.path == "$.__pydantic_extra__"
    assert "structural validator" in error.value.message


def test_pydantic_input_rejects_structural_typed_extra_lifecycle_hook() -> None:
    assert "model_validate" not in DroppingStructuralExtraLifecycleArguments.__dict__
    assert "model_validate" in DroppingStructuralExtraLifecycleBase.__dict__

    with pytest.raises(ToolValidationError) as error:
        pydantic_input(DroppingStructuralExtraLifecycleArguments)

    assert error.value.path == "$.__pydantic_extra__"
    assert "structural model lifecycle hook" in error.value.message


@pytest.mark.parametrize(
    ("validated", "original"),
    [
        pytest.param(
            [NestedArguments(label="first")],
            [
                {"label": "first"},
                {"label": "second", "discarded": True},
            ],
            id="list",
        ),
        pytest.param(
            (NestedArguments(label="first"),),
            (
                {"label": "first"},
                {"label": "second", "discarded": True},
            ),
            id="tuple",
        ),
    ],
)
def test_pydantic_runtime_walker_rejects_sequence_cardinality_changes(
    validated: object,
    original: object,
) -> None:
    with pytest.raises(ToolValidationError) as error:
        _reject_ignored_extras_in_value(
            validated,
            original,
            path=("children",),
        )

    assert error.value.path == "$.children"
    assert "cardinality" in error.value.message


@pytest.mark.parametrize(
    "model",
    [
        pytest.param(UnionWithDataclassArguments, id="union"),
        pytest.param(ListUnionWithDataclassArguments, id="list-union"),
        pytest.param(TupleWithDataclassArguments, id="tuple"),
    ],
)
def test_pydantic_input_audits_every_composite_branch(
    model: type[BaseModel],
) -> None:
    with pytest.raises(ToolValidationError) as error:
        pydantic_input(model)

    assert error.value.path == "$.nested"
    assert "unsupported Pydantic input shape: dataclass" == error.value.message


def test_pydantic_input_rejects_structural_core_schema_metadata() -> None:
    metadata = CoreSchemaMetadataArguments.model_fields["children"].metadata
    assert any(
        callable(getattr(item, "__get_pydantic_core_schema__", None))
        for item in metadata
    )

    with pytest.raises(ToolValidationError) as error:
        pydantic_input(CoreSchemaMetadataArguments)

    assert error.value.path == "$.children"
    assert "structural core-schema hook" in error.value.message


@pytest.mark.parametrize(
    ("model", "lifecycle_marker"),
    [
        pytest.param(
            CoreSchemaLifecycleArguments,
            "__get_pydantic_core_schema__",
            id="core-schema",
        ),
        pytest.param(
            CustomInitLifecycleArguments,
            "__pydantic_custom_init__",
            id="custom-init",
        ),
        pytest.param(
            PostInitLifecycleArguments,
            "__pydantic_post_init__",
            id="post-init",
        ),
        pytest.param(
            ModelValidateLifecycleArguments,
            "model_validate",
            id="model-validate",
        ),
    ],
)
def test_pydantic_input_rejects_structural_model_lifecycle_hooks(
    model: type[BaseModel],
    lifecycle_marker: str,
) -> None:
    if lifecycle_marker in {"__get_pydantic_core_schema__", "model_validate"}:
        descriptor = model.__dict__.get(lifecycle_marker)
        assert descriptor is not None
        assert descriptor is not BaseModel.__dict__.get(lifecycle_marker)
    else:
        assert getattr(model, lifecycle_marker, None)

    with pytest.raises(ToolValidationError) as error:
        pydantic_input(model)

    assert error.value.path == "$.children"
    assert "structural model lifecycle hook" in error.value.message


@pytest.mark.parametrize(
    ("model", "owner", "hook_name"),
    [
        pytest.param(
            InheritedCoreSchemaLifecycleArguments,
            InheritedCoreSchemaLifecycleBase,
            "__get_pydantic_core_schema__",
            id="core-schema",
        ),
        pytest.param(
            InheritedModelValidateLifecycleArguments,
            InheritedModelValidateLifecycleBase,
            "model_validate",
            id="model-validate",
        ),
        pytest.param(
            InheritedLegacyLifecycleArguments,
            InheritedLegacyLifecycleBase,
            "__get_validators__",
            id="legacy-validators",
        ),
    ],
)
def test_pydantic_input_rejects_inherited_structural_model_lifecycle_hooks(
    model: type[BaseModel],
    owner: type[BaseModel],
    hook_name: str,
) -> None:
    assert hook_name not in model.__dict__
    assert hook_name in owner.__dict__

    with pytest.raises(ToolValidationError) as error:
        pydantic_input(model)

    assert error.value.path == "$.children"
    assert "structural model lifecycle hook" in error.value.message


def test_pydantic_input_allows_ordinary_declarative_model_inheritance() -> None:
    _, validate = pydantic_input(DeclarativeInheritedStructuralArguments)

    assert validate(
        {
            "inherited_value": "kept",
            "children": [{"kind": "a", "payload": "first"}],
        }
    ) == {
        "inherited_value": "kept",
        "children": ({"kind": "a", "payload": "first"},),
    }


@pytest.mark.parametrize(
    "model",
    [
        pytest.param(ScalarCustomInitLifecycleArguments, id="custom-init"),
        pytest.param(ScalarPostInitLifecycleArguments, id="post-init"),
    ],
)
def test_pydantic_input_allows_scalar_only_lifecycle_hooks(
    model: type[BaseModel],
) -> None:
    _, validate = pydantic_input(model)

    assert validate({"value": "abc"}) == {"value": "cba"}


@pytest.mark.parametrize(
    "model",
    [
        pytest.param(
            ScalarCoreSchemaLifecycleArguments,
            id="core-schema-validator",
        ),
        pytest.param(
            DivergentScalarCoreSchemaLifecycleArguments,
            id="core-schema-serializer",
        ),
        pytest.param(
            ScalarModelValidateLifecycleArguments,
            id="model-validate",
        ),
        pytest.param(
            DivergentScalarModelValidateLifecycleArguments,
            id="alternate-model-result",
        ),
    ],
)
def test_pydantic_input_rejects_unverifiable_scalar_model_hooks(
    model: type[BaseModel],
) -> None:
    with pytest.raises(ToolValidationError) as error:
        pydantic_input(model)

    assert error.value.path == "$"
    assert "model lifecycle hook" in error.value.message
    assert len(error.value.message) <= 512


@pytest.mark.parametrize(
    ("arguments", "path"),
    [
        pytest.param({"mode": "fast"}, "$.count", id="required"),
        pytest.param({"count": 4, "mode": "fast"}, "$.count", id="minimum"),
        pytest.param({"count": 11, "mode": "fast"}, "$.count", id="maximum"),
        pytest.param({"count": 7, "mode": "turbo"}, "$.mode", id="enum"),
        pytest.param(
            {"count": 7, "mode": "fast", "discarded": True},
            "$.discarded",
            id="additional-properties",
        ),
    ],
)
def test_raw_json_schema_rejects_complete_constraint_failures(
    arguments: dict[str, Any],
    path: str,
) -> None:
    validate = _raw_input_validator()

    with pytest.raises(ToolValidationError) as error:
        validate(arguments)

    assert error.value.path == path


def test_raw_json_schema_accepts_and_freezes_complete_valid_arguments() -> None:
    closed_schema = {
        "type": "object",
        "properties": {
            "items": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["items"],
        "additionalProperties": False,
    }
    validate = json_schema_input(closed_schema)
    source = {"items": ["one"]}
    arguments = validate(source)  # type: ignore[arg-type]
    source["items"].append("mutated")

    assert arguments == {"items": ("one",)}
    with pytest.raises(TypeError):
        arguments["items"] = ()  # type: ignore[index]


@pytest.mark.parametrize(
    ("keyword", "valid", "invalid"),
    [
        pytest.param(
            "oneOf",
            {"value": 7},
            {"value": True},
            id="oneOf",
        ),
        pytest.param(
            "anyOf",
            {"value": "ready"},
            {"value": False},
            id="anyOf",
        ),
    ],
)
def test_raw_json_schema_preserves_composition_keywords(
    keyword: str,
    valid: dict[str, Any],
    invalid: dict[str, Any],
) -> None:
    schema: dict[str, Any] = {
        "type": "object",
        "properties": {
            "value": {
                keyword: [
                    {"type": "integer"},
                    {"type": "string"},
                ]
            }
        },
        "required": ["value"],
    }
    validate = json_schema_input(schema)

    assert validate(valid) == valid
    with pytest.raises(ToolValidationError, match=keyword):
        validate(invalid)


def test_raw_json_schema_resolves_only_local_defs() -> None:
    schema = {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$defs": {
            "positive": {"type": "integer", "minimum": 1},
        },
        "type": "object",
        "properties": {"count": {"$ref": "#/$defs/positive"}},
        "required": ["count"],
    }
    validate = json_schema_input(schema)

    assert validate({"count": 1}) == {"count": 1}
    with pytest.raises(ToolValidationError) as error:
        validate({"count": 0})
    assert error.value.path == "$.count"


@pytest.mark.parametrize(
    "reference",
    [
        "https://example.invalid/external-schema.json",
        "file:///tmp/external-schema.json",
        "custom://schema/argument",
    ],
)
def test_raw_json_schema_never_retrieves_external_references(reference: str) -> None:
    validate = json_schema_input({"$ref": reference})

    with pytest.raises(ToolValidationError) as error:
        validate({})

    assert error.value.path == "$"
    assert error.value.message == "schema reference could not be resolved locally"
    assert reference not in str(error.value)


def test_raw_json_schema_respects_declared_dialect() -> None:
    validate = json_schema_input(
        {
            "$schema": "http://json-schema.org/draft-04/schema#",
            "type": "object",
            "properties": {
                "value": {
                    "type": "number",
                    "minimum": 5,
                    "exclusiveMinimum": True,
                }
            },
            "required": ["value"],
        }
    )

    with pytest.raises(ToolValidationError):
        validate({"value": 5})
    assert validate({"value": 6}) == {"value": 6}


def test_raw_json_schema_is_copied_at_factory_creation() -> None:
    schema: dict[str, Any] = {"type": "object", "required": ["count"]}
    validate = json_schema_input(schema)
    schema["required"].clear()

    with pytest.raises(ToolValidationError):
        validate({})


def test_invalid_json_schema_is_rejected_at_factory_creation() -> None:
    with pytest.raises(ToolValidationError) as error:
        json_schema_input({"type": "not-a-json-schema-type"})

    assert error.value.path == "$.type"
    assert len(error.value.message) <= 512


def test_json_schema_output_validates_and_freezes_output() -> None:
    schema = {
        "type": "object",
        "properties": {"status": {"enum": ["ready"]}},
        "required": ["status"],
        "additionalProperties": False,
    }
    source = {"status": "ready"}

    output = json_schema_output(schema, source)  # type: ignore[arg-type]
    source["status"] = "mutated"

    assert output == {"status": "ready"}
    with pytest.raises(ToolValidationError) as error:
        json_schema_output(schema, {"status": "failed"})  # type: ignore[arg-type]
    assert error.value.path == "$.status"


def test_json_schema_output_without_schema_still_freezes_json() -> None:
    source = {"items": ["one"]}

    output = json_schema_output(None, source)  # type: ignore[arg-type]
    source["items"].append("mutated")

    assert output == {"items": ("one",)}


def test_validation_errors_bound_path_and_message() -> None:
    enormous_name = "x" * 2_000
    validate = json_schema_input(
        {
            "type": "object",
            "properties": {enormous_name: {"enum": [7]}},
            "required": [enormous_name],
        }
    )

    with pytest.raises(ToolValidationError) as error:
        validate({enormous_name: 1})

    assert len(error.value.path) <= 256
    assert len(error.value.message) <= 512
    assert enormous_name not in str(error.value)


def test_regression_complete_schema_rejects_before_runner_could_execute() -> None:
    validate = json_schema_input(
        {
            "type": "object",
            "properties": {
                "count": {"type": "integer", "minimum": 5, "enum": [7]},
            },
            "required": ["count"],
        }
    )
    runner_called = False

    with pytest.raises(ToolValidationError):
        arguments = validate({"count": 1})
        runner_called = True
        assert arguments

    assert runner_called is False
