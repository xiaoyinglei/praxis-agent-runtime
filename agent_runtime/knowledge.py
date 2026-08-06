from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Annotated, Any, Literal, cast

from pydantic import BaseModel, ConfigDict, PlainSerializer

from agent_runtime.tools.tool import JsonValue


def _json_serializable(value: object) -> Any:
    if value is None or isinstance(value, (str, bool, int, float)):
        return cast(JsonValue, value)
    if isinstance(value, Mapping):
        return {str(key): _json_serializable(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [_json_serializable(item) for item in value]
    raise TypeError(f"knowledge dynamic value is not JSON-compatible: {type(value).__name__}")


type GroundingTargetJson = Annotated[
    Mapping[str, JsonValue],
    PlainSerializer(_json_serializable, return_type=Any, when_used="json"),
]


class RAGKnowledgeConfig(BaseModel):
    """Serializable, secret-free configuration for one RAG knowledge store."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    storage_root: Path = Path(".rag")
    embedding_model: str | None = None
    reranker_model: str | None = None
    vector_backend: Literal["milvus", "sqlite"] = "milvus"
    vector_namespace: str | None = None
    vector_collection_prefix: str | None = None


@dataclass(frozen=True, slots=True)
class AgentEvidence:
    evidence_id: str
    doc_id: int
    citation_anchor: str
    text: str
    score: float
    benchmark_doc_id: str | None = None
    source_id: int | None = None
    evidence_kind: str = "internal"
    record_type: str | None = None
    file_name: str | None = None
    section_path: tuple[str, ...] = ()
    page_start: int | None = None
    page_end: int | None = None
    source_type: str | None = None
    retrieval_channels: tuple[str, ...] = ()
    retrieval_family: str | None = None
    grounding_target: GroundingTargetJson | None = None

    def __post_init__(self) -> None:
        if self.grounding_target is not None:
            object.__setattr__(self, "grounding_target", _freeze_json_mapping(self.grounding_target))


@dataclass(frozen=True, slots=True)
class AgentCitation:
    citation_id: str
    evidence_id: str
    record_type: str
    file_name: str | None = None
    section_path: tuple[str, ...] = ()
    page_start: int | None = None
    page_end: int | None = None
    citation_anchor: str | None = None
    doc_id: int | None = None
    benchmark_doc_id: str | None = None
    source_id: int | None = None
    source_type: str | None = None


def agent_evidence_from_value(value: object) -> AgentEvidence:
    if isinstance(value, AgentEvidence):
        return value
    return AgentEvidence(
        evidence_id=str(_value(value, "evidence_id")),
        doc_id=int(_value(value, "doc_id")),
        citation_anchor=str(_value(value, "citation_anchor")),
        text=str(_value(value, "text")),
        score=float(_value(value, "score")),
        benchmark_doc_id=_optional_str(_value(value, "benchmark_doc_id")),
        source_id=_optional_int(_value(value, "source_id")),
        evidence_kind=str(_value(value, "evidence_kind", "internal")),
        record_type=_optional_str(_value(value, "record_type")),
        file_name=_optional_str(_value(value, "file_name")),
        section_path=tuple(str(item) for item in (_value(value, "section_path", ()) or ())),
        page_start=_optional_int(_value(value, "page_start")),
        page_end=_optional_int(_value(value, "page_end")),
        source_type=_optional_str(_value(value, "source_type")),
        retrieval_channels=tuple(str(item) for item in (_value(value, "retrieval_channels", ()) or ())),
        retrieval_family=_optional_str(_value(value, "retrieval_family")),
        grounding_target=_grounding_mapping(_value(value, "grounding_target")),
    )


def agent_citation_from_value(value: object) -> AgentCitation:
    if isinstance(value, AgentCitation):
        return value
    return AgentCitation(
        citation_id=str(_value(value, "citation_id")),
        evidence_id=str(_value(value, "evidence_id")),
        record_type=str(_value(value, "record_type")),
        file_name=_optional_str(_value(value, "file_name")),
        section_path=tuple(str(item) for item in (_value(value, "section_path", ()) or ())),
        page_start=_optional_int(_value(value, "page_start")),
        page_end=_optional_int(_value(value, "page_end")),
        citation_anchor=_optional_str(_value(value, "citation_anchor")),
        doc_id=_optional_int(_value(value, "doc_id")),
        benchmark_doc_id=_optional_str(_value(value, "benchmark_doc_id")),
        source_id=_optional_int(_value(value, "source_id")),
        source_type=_optional_str(_value(value, "source_type")),
    )


def _value(value: object, name: str, default: object = None) -> Any:
    return value.get(name, default) if isinstance(value, Mapping) else getattr(value, name, default)


def _optional_str(value: object) -> str | None:
    return value if isinstance(value, str) else None


def _optional_int(value: object) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _grounding_mapping(value: object) -> Mapping[str, JsonValue] | None:
    if value is None:
        return None
    if isinstance(value, Mapping):
        return cast(Mapping[str, JsonValue], dict(value))
    return {
        name: _freeze_json_value(getattr(value, name))
        for name in (
            "kind",
            "doc_id",
            "source_id",
            "section_id",
            "asset_id",
            "page_start",
            "page_end",
            "section_path",
            "raw_locator",
        )
    }


def _freeze_json_mapping(value: Mapping[str, object]) -> Mapping[str, JsonValue]:
    return MappingProxyType({str(key): _freeze_json_value(item) for key, item in value.items()})


def _freeze_json_value(value: object) -> JsonValue:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("knowledge JSON values must be finite")
        return value
    if isinstance(value, Mapping):
        return _freeze_json_mapping(cast(Mapping[str, object], value))
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return tuple(_freeze_json_value(item) for item in value)
    raise TypeError(f"knowledge dynamic value is not JSON-compatible: {type(value).__name__}")


__all__ = [
    "AgentCitation",
    "AgentEvidence",
    "RAGKnowledgeConfig",
    "agent_citation_from_value",
    "agent_evidence_from_value",
]
