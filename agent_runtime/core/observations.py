from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any, Literal, cast

from pydantic import BaseModel, Field

from agent_runtime.knowledge import (
    AgentCitation,
    AgentEvidence,
    agent_citation_from_value,
    agent_evidence_from_value,
)
from agent_runtime.memory.models import MemoryRef
from agent_runtime.tools.tool import ToolCall, ToolResult

_MAX_RUNTIME_WORKSPACE_FILE_CHANGES = 8


class EvidenceRef(BaseModel):
    evidence_id: str | None = None
    citation_id: str | None = None
    citation_anchor: str | None = None
    doc_id: int | None = None
    source: str | None = None

    @property
    def key(self) -> str:
        return "|".join(
            value
            for value in (
                self.evidence_id,
                self.citation_id,
                self.citation_anchor,
                None if self.doc_id is None else str(self.doc_id),
                self.source,
            )
            if value
        )


class ContextUnit(BaseModel):
    unit_id: str
    unit_type: str
    locator: dict[str, object] = Field(default_factory=dict)
    preview: str | dict[str, object] | None = None
    content_ref: str | None = None
    evidence_refs: list[EvidenceRef] = Field(default_factory=list)
    capabilities: list[str] = Field(default_factory=list)
    metadata: dict[str, object] = Field(default_factory=dict)

    @property
    def key(self) -> str:
        return self.unit_id


class AnswerCandidate(BaseModel):
    text: str
    source_tool_call_id: str | None = None
    source_tool_name: str | None = None
    evidence_refs: list[EvidenceRef] = Field(default_factory=list)


class ComputationResult(BaseModel):
    source_tool_call_id: str
    source_tool_name: str
    operation: str | None = None
    value_preview: str | None = None
    expression: str | None = None
    evidence_refs: list[EvidenceRef] = Field(default_factory=list)

    @property
    def key(self) -> str:
        return self.source_tool_call_id


class ContextBinding(BaseModel):
    binding_id: str
    constraint_id: str
    unit_id: str | None = None
    status: Literal["satisfied", "ambiguous", "violated"]
    evidence_refs: list[EvidenceRef] = Field(default_factory=list)
    rationale: str | None = None
    metadata: dict[str, object] = Field(default_factory=dict)

    @property
    def key(self) -> str:
        return self.binding_id


class StructuredObservation(BaseModel):
    tool_call_id: str
    tool_name: str
    status: Literal["ok", "error"]
    answer_candidate: AnswerCandidate | None = None
    evidence_refs: list[EvidenceRef] = Field(default_factory=list)
    context_units: list[ContextUnit] = Field(default_factory=list)
    locators: list[dict[str, object]] = Field(default_factory=list)
    asset_refs: list[int] = Field(default_factory=list)
    operation: str | None = None
    warnings: list[str] = Field(default_factory=list)
    error: str | None = None
    raw_result_ref: str
    raw_memory_ref: MemoryRef | None = None
    related_step_ids: list[str] = Field(default_factory=list)
    metadata: dict[str, object] = Field(default_factory=dict)


class ObservationError(BaseModel):
    tool_call_id: str
    tool_name: str
    code: str
    message: str
    retryable: bool
    detail: dict[str, object] = Field(default_factory=dict)


class ObservationBatch(BaseModel):
    structured_observations: list[StructuredObservation] = Field(default_factory=list)
    answer_candidates: list[AnswerCandidate] = Field(default_factory=list)
    evidence_refs: list[EvidenceRef] = Field(default_factory=list)
    computation_results: list[ComputationResult] = Field(default_factory=list)
    context_units: list[ContextUnit] = Field(default_factory=list)
    locators: list[dict[str, object]] = Field(default_factory=list)
    asset_refs: list[int] = Field(default_factory=list)
    evidence: list[AgentEvidence] = Field(default_factory=list)
    citations: list[AgentCitation] = Field(default_factory=list)
    errors: list[ObservationError] = Field(default_factory=list)

    def as_state_update(self) -> dict[str, object]:
        return {
            "structured_observations": list(self.structured_observations),
            "answer_candidates": list(self.answer_candidates),
            "evidence_refs": list(self.evidence_refs),
            "computation_results": list(self.computation_results),
            "context_units": list(self.context_units),
            "locators": list(self.locators),
            "asset_refs": list(self.asset_refs),
            "evidence": list(self.evidence),
            "citations": list(self.citations),
            "errors": [error.model_dump(mode="json") for error in self.errors],
        }


class ObservationBuilder:
    def from_tool_result(self, result: ToolResult) -> StructuredObservation:
        if result.is_error:
            error_message = result.error_message or "unknown tool error"
            return StructuredObservation(
                tool_call_id=result.tool_call_id,
                tool_name=result.tool_name,
                status="error",
                error=error_message,
                raw_result_ref=result.tool_call_id,
            )
        progress_error = tool_result_progress_error(result)
        if progress_error is not None:
            return StructuredObservation(
                tool_call_id=result.tool_call_id,
                tool_name=result.tool_name,
                status="error",
                error=progress_error,
                raw_result_ref=result.tool_call_id,
            )

        output = _result_output(result)
        observation_only = bool(getattr(output, "observation_only", False))
        if observation_only:
            evidence_refs: list[EvidenceRef] = []
        elif result.tool_name.startswith("agent_"):
            evidence_refs = _delegated_evidence_refs_from_output(output)
        else:
            evidence_refs = _dedupe_evidence_refs(
                [
                    *_evidence_refs_from_output(output),
                    *_search_evidence_refs_from_output(output),
                ]
            )
        answer_text = None if observation_only else _answer_text(result.tool_name, output)
        answer = (
            AnswerCandidate(
                text=answer_text,
                source_tool_call_id=result.tool_call_id,
                source_tool_name=result.tool_name,
                evidence_refs=evidence_refs,
            )
            if answer_text
            else None
        )
        locators = _locators_from_output(output, tool_name=result.tool_name)
        return StructuredObservation(
            tool_call_id=result.tool_call_id,
            tool_name=result.tool_name,
            status="ok",
            answer_candidate=answer,
            evidence_refs=evidence_refs,
            context_units=_context_units_from_output(
                result,
                evidence_refs=evidence_refs,
                locators=locators,
            ),
            locators=locators,
            asset_refs=_asset_refs_from_output(output),
            operation=_operation_from_output(output),
            raw_result_ref=result.tool_call_id,
        )


class ObservationExtractor:
    """Reduce tool outcomes into provenance channels without loop control decisions."""

    def __init__(self, observation_builder: ObservationBuilder | None = None) -> None:
        self._observation_builder = observation_builder or ObservationBuilder()

    def extract(
        self,
        tool_results: Sequence[ToolResult],
        *,
        seen_tool_call_ids: Sequence[str] = (),
    ) -> ObservationBatch:
        seen = set(seen_tool_call_ids)
        selected_results = [result for result in tool_results if result.tool_call_id not in seen]
        observations = [self._observation_builder.from_tool_result(result) for result in selected_results]
        if not observations:
            return ObservationBatch()

        results_by_id = {result.tool_call_id: result for result in selected_results}
        computation_results = [
            ComputationResult(
                source_tool_call_id=observation.tool_call_id,
                source_tool_name=observation.tool_name,
                operation=observation.operation,
                value_preview=(
                    None if observation.answer_candidate is None else observation.answer_candidate.text[:300]
                ),
                expression=_computation_expression(results_by_id.get(observation.tool_call_id)),
                evidence_refs=observation.evidence_refs,
            )
            for observation in observations
            if observation.operation is not None
        ]
        return ObservationBatch(
            structured_observations=observations,
            answer_candidates=[
                observation.answer_candidate for observation in observations if observation.answer_candidate is not None
            ],
            evidence_refs=[ref for observation in observations for ref in observation.evidence_refs],
            computation_results=computation_results,
            context_units=_dedupe_context_units(
                [unit for observation in observations for unit in observation.context_units]
            ),
            locators=[locator for observation in observations for locator in observation.locators],
            asset_refs=[asset_ref for observation in observations for asset_ref in observation.asset_refs],
            evidence=_evidence_from_outputs(observations, selected_results),
            citations=_citations_from_outputs(observations, selected_results),
            errors=_errors_from_results(selected_results),
        )

    def reduce_tool_results(self, state: dict[str, Any]) -> dict[str, Any]:
        seen = [
            observation.tool_call_id
            for observation in state.get("structured_observations", [])
            if isinstance(observation, StructuredObservation)
        ]
        batch = self.extract(
            list(state.get("tool_results", [])),
            seen_tool_call_ids=seen,
        )
        if not batch.structured_observations:
            return {}
        return cast(dict[str, Any], batch.as_state_update())


def _answer_text(tool_name: str, output: BaseModel | _OutputView | None) -> str | None:
    if output is None:
        return None
    if tool_name.startswith("agent_"):
        conclusion = getattr(output, "conclusion", None)
        if isinstance(conclusion, str) and conclusion.strip():
            return conclusion.strip()
        return None
    if tool_name in {
        "vector_search",
        "keyword_search",
        "grounding",
        "rerank",
        "graph_expand",
        "asset_list",
        "asset_inspect",
        "asset_read_slice",
        "list_files",
        "read_file",
    }:
        return None
    text = getattr(output, "text", None)
    if isinstance(text, str) and text.strip():
        return text.strip()
    answer_text = getattr(output, "answer_text", None)
    if isinstance(answer_text, str) and answer_text.strip():
        return answer_text.strip()
    markdown = getattr(output, "markdown", None)
    if isinstance(markdown, str) and markdown.strip():
        return markdown.strip()
    return output.model_dump_json(exclude_none=True)


def _evidence_refs_from_output(output: BaseModel | _OutputView | None) -> list[EvidenceRef]:
    if output is None:
        return []
    refs: list[EvidenceRef] = []
    for evidence_id in getattr(output, "evidence_ids", []) or []:
        refs.append(EvidenceRef(evidence_id=str(evidence_id), source="tool_output"))
    for citation_id in getattr(output, "citation_ids", []) or []:
        refs.append(EvidenceRef(citation_id=str(citation_id), source="tool_output"))
    for evidence in getattr(output, "evidence", []) or []:
        evidence_item = agent_evidence_from_value(evidence)
        refs.append(
            EvidenceRef(
                evidence_id=evidence_item.evidence_id,
                citation_anchor=evidence_item.citation_anchor,
                doc_id=evidence_item.doc_id,
                source="evidence",
            )
        )
    for citation in getattr(output, "citations", []) or []:
        if not isinstance(citation, (Mapping, AgentCitation)):
            continue
        citation_item = agent_citation_from_value(citation)
        refs.append(
            EvidenceRef(
                evidence_id=citation_item.evidence_id,
                citation_id=citation_item.citation_id,
                citation_anchor=citation_item.citation_anchor,
                doc_id=citation_item.doc_id,
                source="citation",
            )
        )
    locator = getattr(output, "locator", None)
    if locator is not None:
        locator_anchor = getattr(locator, "citation_anchor", None)
        refs.append(
            EvidenceRef(
                citation_anchor=str(locator_anchor) if locator_anchor else None,
                source="locator",
            )
        )
    asset_id = getattr(output, "asset_id", None)
    if isinstance(asset_id, int) and asset_id > 0:
        refs.append(EvidenceRef(evidence_id=f"asset:{asset_id}", source="asset"))
    return _dedupe_evidence_refs(refs)


def _delegated_evidence_refs_from_output(
    output: BaseModel | _OutputView | None,
) -> list[EvidenceRef]:
    if output is None:
        return []
    refs: list[EvidenceRef] = []
    for item in getattr(output, "evidence_refs", []) or []:
        evidence_id = getattr(item, "evidence_id", None)
        citation_id = getattr(item, "citation_id", None)
        citation_anchor = getattr(item, "citation_anchor", None)
        if not isinstance(evidence_id, str) or not evidence_id.strip():
            continue
        if not (isinstance(citation_id, str) and citation_id.strip()) and not (
            isinstance(citation_anchor, str) and citation_anchor.strip()
        ):
            continue
        doc_id = getattr(item, "doc_id", None)
        refs.append(
            EvidenceRef(
                evidence_id=evidence_id.strip(),
                citation_id=(citation_id.strip() if isinstance(citation_id, str) else None),
                citation_anchor=(citation_anchor.strip() if isinstance(citation_anchor, str) else None),
                doc_id=doc_id if isinstance(doc_id, int) else None,
                source="delegated_agent",
            )
        )
    for citation in getattr(output, "citations", []) or []:
        citation_item = agent_citation_from_value(citation)
        if any(ref.evidence_id == citation_item.evidence_id for ref in refs):
            continue
        refs.append(
            EvidenceRef(
                evidence_id=citation_item.evidence_id,
                citation_id=citation_item.citation_id,
                citation_anchor=citation_item.citation_anchor,
                doc_id=citation_item.doc_id,
                source="delegated_agent",
            )
        )
    return _dedupe_evidence_refs(refs)


def _search_evidence_refs_from_output(
    output: BaseModel | _OutputView | None,
) -> list[EvidenceRef]:
    if output is None:
        return []
    items = getattr(output, "items", None)
    if not isinstance(items, list):
        return []
    refs: list[EvidenceRef] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        ref = EvidenceRef(
            evidence_id=(str(item["evidence_id"]) if item.get("evidence_id") else None),
            citation_anchor=(str(item["citation_anchor"]) if item.get("citation_anchor") else None),
            doc_id=item["doc_id"] if isinstance(item.get("doc_id"), int) else None,
            source="retrieval",
        )
        if ref.key:
            refs.append(ref)
    return _dedupe_evidence_refs(refs)


def _context_units_from_output(
    result: ToolResult,
    *,
    evidence_refs: Sequence[EvidenceRef],
    locators: Sequence[dict[str, object]],
) -> list[ContextUnit]:
    output = _result_output(result)
    if output is None:
        return []
    if result.tool_name in {
        "vector_search",
        "keyword_search",
        "grounding",
        "rerank",
        "graph_expand",
    }:
        return _retrieval_context_units(result, evidence_refs=evidence_refs)
    if result.tool_name in {"asset_list", "asset_inspect"}:
        return [
            _asset_context_unit(result.tool_name, locator)
            for locator in locators
            if isinstance(locator.get("asset_id"), int)
        ]
    if result.tool_name == "asset_analyze":
        preview = _answer_text(result.tool_name, output)
        units = [
            ContextUnit(
                unit_id=f"computed:{result.tool_call_id}",
                unit_type="computed_result",
                preview=preview[:1000] if preview else None,
                content_ref=result.tool_call_id,
                evidence_refs=list(evidence_refs),
                metadata={"source_tool": result.tool_name},
            )
        ]
        if not bool(getattr(output, "observation_only", False)):
            units.extend(
                _asset_context_unit(result.tool_name, locator)
                for locator in locators
                if isinstance(locator.get("asset_id"), int) and isinstance(locator.get("asset_type"), str)
            )
        return units
    if result.tool_name == "list_files":
        return _list_files_context_units(output)
    if result.tool_name == "read_file":
        unit = _read_file_context_unit(output, tool_call_id=result.tool_call_id)
        return [] if unit is None else [unit]
    return []


def _retrieval_context_units(
    result: ToolResult,
    *,
    evidence_refs: Sequence[EvidenceRef],
) -> list[ContextUnit]:
    items = getattr(_result_output(result), "items", None)
    if not isinstance(items, list):
        return []
    units: list[ContextUnit] = []
    for index, item in enumerate(items):
        if not isinstance(item, dict):
            continue
        locator = _retrieval_locator(item)
        item_refs = _search_evidence_refs_from_output(_SingleSearchItemOutput(items=[item]))
        record_type = str(item.get("record_type", "") or "")
        unit_type = "document_section" if record_type == "section" else "retrieved_chunk"
        identifier = str(item["evidence_id"]) if item.get("evidence_id") else f"{result.tool_call_id}:{index}"
        text = str(item.get("text", "") or "")
        capabilities = ["text_extract", "text_synthesize", "quote"]
        if "ASSET_ANCHOR:" in text or "asset" in record_type:
            capabilities.append("asset_list")
        units.append(
            ContextUnit(
                unit_id=f"retrieval:{identifier}",
                unit_type=unit_type,
                locator=locator,
                preview=text[:1000] if text else None,
                content_ref=(str(item["evidence_id"]) if item.get("evidence_id") else result.tool_call_id),
                evidence_refs=item_refs or list(evidence_refs),
                capabilities=capabilities,
                metadata={"source_tool": result.tool_name},
            )
        )
    return units


class _SingleSearchItemOutput(BaseModel):
    items: list[dict[str, object]]


def _retrieval_locator(item: dict[str, object]) -> dict[str, object]:
    return {
        field: item[field]
        for field in (
            "doc_id",
            "source_id",
            "section_id",
            "page_start",
            "page_end",
            "record_type",
            "citation_anchor",
            "evidence_id",
            "score",
            "rerank_score",
            "retrieval_channels",
            "retrieval_family",
        )
        if item.get(field) not in (None, "", [])
    }


def _list_files_context_units(output: BaseModel | _OutputView) -> list[ContextUnit]:
    units: list[ContextUnit] = []
    entries = getattr(output, "entries", None)
    if not isinstance(entries, list):
        entries = getattr(output, "files", [])
    for file_info in entries or []:
        locator = _workspace_file_locator(file_info, source_tool="list_files")
        path = locator.get("path")
        if not isinstance(path, str) or not path.strip():
            continue
        is_dir = bool(locator.get("is_dir", False))
        units.append(
            ContextUnit(
                unit_id=(f"workspace_dir:{path}" if is_dir else f"workspace_file:{path}"),
                unit_type="workspace_dir" if is_dir else "workspace_file",
                locator=locator,
                preview=f"{path} ({locator.get('size_bytes', 0)} bytes)",
                content_ref=path,
                capabilities=_workspace_file_capabilities(
                    file_info,
                    is_dir=is_dir,
                ),
                metadata={"source_tool": "list_files"},
            )
        )
    return units


def _read_file_context_unit(
    output: BaseModel | _OutputView,
    *,
    tool_call_id: str,
) -> ContextUnit | None:
    path = getattr(output, "path", None)
    if not isinstance(path, str) or not path.strip():
        return None
    content = getattr(output, "content", None)
    return ContextUnit(
        unit_id=f"workspace_file:{path}",
        unit_type="workspace_file_content",
        locator=_read_file_locator(output),
        preview=(content[:1000] if isinstance(content, str) and content else None),
        content_ref=tool_call_id,
        capabilities=["read_file"],
        metadata={"source_tool": "read_file"},
    )


def _asset_context_unit(
    tool_name: str,
    locator: dict[str, object],
) -> ContextUnit:
    asset_id = locator.get("asset_id")
    if not isinstance(asset_id, int):
        raise ValueError("asset context unit requires integer asset_id")
    asset_type = str(locator.get("asset_type", "") or "")
    unit_type = {
        "table": "table_asset",
        "image": "image_asset",
    }.get(asset_type, "document_asset")
    advertised = locator.get("analysis_capabilities", [])
    analysis_capabilities = [str(capability) for capability in advertised] if isinstance(advertised, list) else []
    capabilities = list(
        dict.fromkeys(
            [
                *(["asset_inspect"] if tool_name == "asset_list" else []),
                *analysis_capabilities,
            ]
        )
    )
    preview_fields = {
        field: locator[field]
        for field in ("columns", "row_count", "column_count", "head_rows")
        if locator.get(field) not in (None, "", [])
    }
    return ContextUnit(
        unit_id=f"asset:{asset_id}",
        unit_type=unit_type,
        locator=dict(locator),
        preview=preview_fields or None,
        content_ref=f"asset:{asset_id}",
        capabilities=capabilities,
        metadata={
            "source_tool": tool_name,
            "inspection_status": {
                "asset_list": "listed",
                "asset_inspect": "inspected",
                "asset_analyze": "analyzed",
            }.get(tool_name, "observed"),
        },
    )


def _dedupe_context_units(
    units: Sequence[ContextUnit],
) -> list[ContextUnit]:
    return list({unit.unit_id: unit for unit in units}.values())


def _locators_from_output(
    output: BaseModel | _OutputView | None,
    *,
    tool_name: str | None = None,
) -> list[dict[str, object]]:
    if output is None:
        return []
    workspace_locators = _workspace_tool_locators_from_output(tool_name, output)
    if workspace_locators:
        return workspace_locators
    items = getattr(output, "items", None)
    if isinstance(items, list):
        return _search_asset_locators(items)
    locator = getattr(output, "locator", None)
    if locator is not None and hasattr(locator, "model_dump"):
        return [locator.model_dump(mode="json", exclude_none=True)]
    asset_id = getattr(output, "asset_id", None)
    if isinstance(asset_id, int) and asset_id > 0:
        values: dict[str, object] = {"asset_id": asset_id}
        for field in (
            "doc_id",
            "source_id",
            "section_id",
            "asset_type",
            "sheet_name",
            "page_no",
            "element_ref",
            "caption",
            "analysis_capabilities",
            "columns",
            "row_count",
            "column_count",
        ):
            value = getattr(output, field, None)
            if value not in (None, "", []):
                values[field] = value
        head_rows = getattr(output, "head_rows", None)
        if head_rows:
            values["head_rows"] = head_rows
        return [values]
    assets = getattr(output, "assets", None)
    if not isinstance(assets, list):
        return []
    return [_asset_locator_from_descriptor(asset) for asset in assets if hasattr(asset, "model_dump")]


def _workspace_tool_locators_from_output(
    tool_name: str | None,
    output: BaseModel | _OutputView,
) -> list[dict[str, object]]:
    if tool_name == "list_files":
        entries = getattr(output, "entries", None)
        if not isinstance(entries, list):
            entries = getattr(output, "files", [])
        return [_workspace_file_locator(file_info, source_tool="list_files") for file_info in entries or []]
    if tool_name == "read_file":
        return [_read_file_locator(output)]
    if tool_name == "search_text":
        matches = getattr(output, "matches", [])
        locators: list[dict[str, object]] = []
        for match in matches if isinstance(matches, list) else []:
            file_path = _workspace_file_field(match, "file_path")
            line_number = _workspace_file_field(match, "line_number")
            if not isinstance(file_path, str) or not isinstance(line_number, int):
                continue
            locators.append(
                {
                    "source_tool": "search_text",
                    "path": file_path,
                    "line_number": line_number,
                }
            )
        return locators
    return []


def _workspace_file_locator(
    file_info: object,
    *,
    source_tool: str,
) -> dict[str, object]:
    values: dict[str, object] = {"source_tool": source_tool}
    for field, output_field in (
        ("path", "path"),
        ("name", "name"),
        ("mime_type", "mime_type"),
    ):
        value = _workspace_file_field(file_info, field)
        if value not in (None, "", []):
            values[output_field] = value
    size_bytes = _workspace_file_field(file_info, "size_bytes")
    if not isinstance(size_bytes, int):
        size_bytes = _workspace_file_field(file_info, "size")
    if isinstance(size_bytes, int):
        values["size_bytes"] = size_bytes
    is_directory = _workspace_file_field(file_info, "is_directory")
    if not isinstance(is_directory, bool):
        is_directory = _workspace_file_field(file_info, "is_dir")
    if isinstance(is_directory, bool):
        values["is_dir"] = is_directory
    file_kind = _workspace_file_field(file_info, "file_kind")
    has_file_kind = isinstance(file_kind, str) and file_kind not in {"", "unknown"}
    if has_file_kind:
        values["file_kind"] = file_kind
        for field in ("is_binary", "readable_as_text"):
            value = _workspace_file_field(file_info, field)
            if isinstance(value, bool):
                values[field] = value
    return values


def _workspace_file_field(file_info: object, field: str) -> object:
    if isinstance(file_info, Mapping):
        return file_info.get(field)
    return getattr(file_info, field, None)


def _workspace_file_capabilities(
    file_info: object,
    *,
    is_dir: bool,
) -> list[str]:
    raw = _workspace_file_field(file_info, "capabilities")
    if isinstance(raw, list) and raw:
        return [str(item) for item in raw if str(item)]
    return ["list_files"] if is_dir else ["read_file"]


def _read_file_locator(output: BaseModel | _OutputView) -> dict[str, object]:
    values: dict[str, object] = {"source_tool": "read_file"}
    for output_field, locator_field in (
        ("path", "path"),
        ("size_bytes", "size_bytes"),
        ("truncated", "truncated"),
        ("is_binary", "is_binary"),
        ("encoding", "encoding"),
    ):
        value = getattr(output, output_field, None)
        if value not in (None, "", []):
            values[locator_field] = value
    return values


def _asset_locator_from_descriptor(asset: object) -> dict[str, object]:
    values: dict[str, object] = {}
    for field in (
        "asset_id",
        "doc_id",
        "source_id",
        "section_id",
        "asset_type",
        "page_no",
        "element_ref",
        "sheet_name",
        "caption",
        "row_count",
        "column_count",
        "columns",
        "analysis_capabilities",
    ):
        value = getattr(asset, field, None)
        if value not in (None, "", []):
            values[field] = value
    return values


def _asset_refs_from_output(output: BaseModel | _OutputView | None) -> list[int]:
    if output is None:
        return []
    assets = getattr(output, "assets", None)
    if isinstance(assets, list):
        return [
            asset_id
            for asset in assets
            if isinstance((asset_id := getattr(asset, "asset_id", None)), int) and asset_id > 0
        ]
    asset_id = getattr(output, "asset_id", None)
    return [asset_id] if isinstance(asset_id, int) and asset_id > 0 else []


def _search_asset_locators(
    items: Sequence[object],
) -> list[dict[str, object]]:
    locators: list[dict[str, object]] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        text = str(item.get("text", "") or "")
        record_type = str(item.get("record_type", "") or "")
        if "ASSET_ANCHOR:" not in text and "asset" not in record_type and "section" not in record_type:
            continue
        locator = _retrieval_locator(item)
        if locator.get("section_id") is not None:
            locators.append(locator)
    return locators


def _operation_from_output(output: BaseModel | _OutputView | None) -> str | None:
    if output is None:
        return None
    operation = getattr(output, "operation", None)
    return str(operation) if operation else None


def _computation_expression(result: ToolResult | None) -> str | None:
    if result is None or result.structured_content is None:
        return None
    query = getattr(_result_output(result), "query", None)
    if not isinstance(query, str) or not query.strip():
        return None
    return query.strip()[:1000]


def _dedupe_evidence_refs(
    refs: Sequence[EvidenceRef],
) -> list[EvidenceRef]:
    deduped: dict[str, EvidenceRef] = {}
    for ref in refs:
        if ref.key:
            deduped.setdefault(ref.key, ref)
    return list(deduped.values())


def _evidence_from_outputs(
    observations: Sequence[StructuredObservation],
    tool_results: Sequence[ToolResult],
) -> list[AgentEvidence]:
    observed_ids = {observation.tool_call_id for observation in observations}
    return [
        agent_evidence_from_value(item)
        for result in tool_results
        if result.tool_call_id in observed_ids and result.structured_content is not None
        for item in getattr(_result_output(result), "evidence", []) or []
    ]


def _citations_from_outputs(
    observations: Sequence[StructuredObservation],
    tool_results: Sequence[ToolResult],
) -> list[AgentCitation]:
    observed_ids = {observation.tool_call_id for observation in observations}
    return [
        agent_citation_from_value(item)
        for result in tool_results
        if result.tool_call_id in observed_ids and result.structured_content is not None
        for item in getattr(_result_output(result), "citations", []) or []
        if isinstance(item, (Mapping, AgentCitation))
    ]


def _errors_from_results(
    tool_results: Sequence[ToolResult],
) -> list[ObservationError]:
    return [
        ObservationError(
            tool_call_id=result.tool_call_id,
            tool_name=result.tool_name,
            code=result.error_code or "tool_error",
            message=result.error_message or "unknown tool error",
            retryable=result.retryable,
            detail=dict(result.metadata),
        )
        for result in tool_results
        if result.is_error
    ]


def grounded_workspace_paths(
    *,
    locators: Sequence[Mapping[str, object]] = (),
    input_paths: Sequence[str] = (),
    tool_results: Sequence[ToolResult] = (),
    tool_calls: Mapping[str, ToolCall] | None = None,
) -> tuple[str, ...]:
    """Reduce runtime-owned file evidence to normalized workspace paths.

    Model-authored plans are deliberately not an input. A path is grounded only
    when it came from the public file manifest, a successful tool observation,
    or a write result that the runtime recorded as changing the workspace.
    """

    grounded: dict[str, None] = {}

    def add(value: object) -> None:
        if not isinstance(value, str):
            return
        normalized = _normalize_grounded_workspace_path(value)
        if normalized is not None:
            grounded[normalized] = None

    for path in input_paths:
        add(path)
    for locator in locators:
        add(locator.get("path"))
    calls = tool_calls or {}
    for result in tool_results:
        if result.is_error:
            continue
        call = calls.get(result.tool_call_id)
        if call is not None and result.tool_name in {"list_files", "search_text", "read_file"}:
            requested_path = call.arguments.get("path")
            if requested_path is None and result.tool_name in {"list_files", "search_text"}:
                requested_path = "."
            add(requested_path)
        if isinstance(result.structured_content, Mapping):
            if result.tool_name == "read_file":
                add(result.structured_content.get("path"))
            elif result.tool_name == "list_files":
                entries = result.structured_content.get("entries")
                if not isinstance(entries, Sequence) or isinstance(
                    entries,
                    (str, bytes),
                ):
                    entries = result.structured_content.get("files")
                if isinstance(entries, Sequence) and not isinstance(
                    entries,
                    (str, bytes),
                ):
                    for entry in entries:
                        if isinstance(entry, Mapping):
                            add(entry.get("path"))
            elif result.tool_name == "search_text":
                add(result.structured_content.get("searched_file_path"))
                matches = result.structured_content.get("matches")
                if isinstance(matches, Sequence) and not isinstance(
                    matches,
                    (str, bytes),
                ):
                    for match in matches:
                        if isinstance(match, Mapping):
                            add(match.get("file_path"))
        for path, _before_sha256, _after_sha256 in runtime_workspace_file_changes(result):
            add(path)
        change = runtime_workspace_change(result)
        if change is not None and change[0] != ".":
            add(change[0])
    return tuple(grounded)


def runtime_workspace_change(
    result: ToolResult,
) -> tuple[str, str, str] | None:
    """Return runtime-owned workspace change evidence."""

    snapshot = runtime_workspace_snapshot(result)
    if snapshot is not None:
        before_sha256, after_sha256 = snapshot
        if before_sha256 != after_sha256:
            return ".", before_sha256, after_sha256
        return None

    # Checkpoints written before executor-owned tree snapshots stored this
    # apply_patch receipt. Current executions cannot emit these protected keys:
    # ToolExecutor strips tool-provided claims before returning a ToolResult.
    if result.is_error or result.tool_name != "apply_patch":
        return None
    if result.metadata.get("workspace_changed") is not True:
        return None
    file_path = result.metadata.get("file_path")
    legacy_before_sha256 = result.metadata.get("before_sha256")
    legacy_after_sha256 = result.metadata.get("after_sha256")
    normalized_path = _normalize_grounded_workspace_path(file_path) if isinstance(file_path, str) else None
    if (
        normalized_path is None
        or not isinstance(legacy_before_sha256, str)
        or not isinstance(legacy_after_sha256, str)
        or not _valid_sha256(legacy_before_sha256)
        or not _valid_sha256(legacy_after_sha256)
        or legacy_before_sha256 == legacy_after_sha256
    ):
        return None
    return normalized_path, legacy_before_sha256, legacy_after_sha256


def runtime_workspace_snapshot(
    result: ToolResult,
) -> tuple[str, str] | None:
    """Return a valid executor-owned before/after workspace snapshot."""

    if result.metadata.get("runtime_workspace_write") is not True:
        return None
    before_sha256 = result.metadata.get("workspace_tree_before_sha256")
    after_sha256 = result.metadata.get("workspace_tree_after_sha256")
    if (
        not isinstance(before_sha256, str)
        or not isinstance(after_sha256, str)
        or not _valid_sha256(before_sha256)
        or not _valid_sha256(after_sha256)
    ):
        return None
    return before_sha256, after_sha256


def runtime_workspace_file_changes(
    result: ToolResult,
) -> tuple[tuple[str, str, str], ...]:
    """Return executor-attested concrete file changes, failing closed."""

    if runtime_workspace_snapshot(result) is None:
        return ()
    raw_changes = result.metadata.get("runtime_workspace_file_changes")
    if not isinstance(raw_changes, Sequence) or isinstance(raw_changes, (str, bytes)):
        return ()
    if len(raw_changes) > _MAX_RUNTIME_WORKSPACE_FILE_CHANGES:
        return ()

    changes: list[tuple[str, str, str]] = []
    seen_paths: set[str] = set()
    for raw_change in raw_changes:
        if not isinstance(raw_change, Mapping):
            return ()
        path = raw_change.get("path")
        before_sha256 = raw_change.get("before_sha256")
        after_sha256 = raw_change.get("after_sha256")
        normalized_path = (
            _normalize_grounded_workspace_path(path)
            if isinstance(path, str)
            else None
        )
        if (
            normalized_path in {None, "."}
            or normalized_path in seen_paths
            or not _valid_sha256(before_sha256)
            or not _valid_sha256(after_sha256)
            or before_sha256 == after_sha256
        ):
            return ()
        assert normalized_path is not None
        assert isinstance(before_sha256, str)
        assert isinstance(after_sha256, str)
        seen_paths.add(normalized_path)
        changes.append((normalized_path, before_sha256, after_sha256))
    return tuple(changes)


def runtime_file_inspection_paths(
    result: ToolResult,
    *,
    call: ToolCall | None,
) -> tuple[str, ...]:
    """Return file paths whose current contents a successful tool observed."""

    if (
        result.is_error
        or call is None
        or call.tool_call_id != result.tool_call_id
        or call.tool_name != result.tool_name
        or not isinstance(result.structured_content, Mapping)
    ):
        return ()
    output = result.structured_content
    if result.tool_name == "read_file":
        requested_path = call.arguments.get("path")
        returned_path = output.get("path")
        normalized_requested = (
            _normalize_grounded_workspace_path(requested_path)
            if isinstance(requested_path, str)
            else None
        )
        normalized_returned = (
            _normalize_grounded_workspace_path(returned_path)
            if isinstance(returned_path, str)
            else None
        )
        if normalized_requested is None or normalized_requested != normalized_returned:
            return ()
        return (normalized_requested,)
    if result.tool_name != "search_text":
        return ()

    matches = output.get("matches")
    if not isinstance(matches, Sequence) or isinstance(matches, (str, bytes)):
        return ()
    matched_paths: dict[str, None] = {}
    for match in matches:
        if not isinstance(match, Mapping):
            return ()
        raw_path = match.get("file_path")
        normalized_path = (
            _normalize_grounded_workspace_path(raw_path)
            if isinstance(raw_path, str)
            else None
        )
        if normalized_path is None:
            return ()
        matched_paths.setdefault(normalized_path, None)
    if matched_paths:
        return tuple(matched_paths)

    total_matches = output.get("total_matches")
    if (
        not isinstance(total_matches, int)
        or isinstance(total_matches, bool)
        or total_matches != 0
        or output.get("truncated") is not False
    ):
        return ()
    requested_path = call.arguments.get("path")
    searched_path = output.get("searched_file_path")
    normalized_requested = (
        _normalize_grounded_workspace_path(requested_path)
        if isinstance(requested_path, str)
        else None
    )
    normalized_searched = (
        _normalize_grounded_workspace_path(searched_path)
        if isinstance(searched_path, str)
        else None
    )
    if normalized_requested is None or normalized_requested != normalized_searched:
        return ()
    return (normalized_requested,)


def _valid_sha256(value: object) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return True


def tool_result_progress_error(result: ToolResult) -> str | None:
    """Return why a technically completed tool call is not useful progress."""

    if result.is_error:
        return result.error_message or "tool execution failed"
    if result.tool_name == "apply_patch":
        if runtime_workspace_change(result) is None:
            return "write tool produced no workspace change"
        return None
    output = result.structured_content
    if result.tool_name == "search_text" and isinstance(output, Mapping):
        matches = output.get("matches")
        if (
            isinstance(matches, Sequence)
            and not isinstance(
                matches,
                (str, bytes),
            )
            and not matches
        ):
            return "search returned no matches"
        return None
    if result.tool_name == "list_files" and isinstance(output, Mapping):
        entries = output.get("entries")
        if (
            isinstance(entries, Sequence)
            and not isinstance(
                entries,
                (str, bytes),
            )
            and not entries
        ):
            return "directory listing returned no entries"
        return None
    if result.tool_name != "run_command":
        return None
    if not isinstance(output, Mapping):
        return "command result is missing an exit status"
    exit_code = output.get("exit_code")
    if isinstance(exit_code, bool) or not isinstance(exit_code, int):
        return "command result is missing an exit status"
    if exit_code != 0:
        return f"command exited with status {exit_code}"
    return None


def _normalize_grounded_workspace_path(value: str) -> str | None:
    normalized = value.strip().replace("\\", "/")
    while normalized.startswith("./"):
        normalized = normalized[2:]
    normalized = normalized.rstrip("/")
    if normalized == ".":
        return "."
    raw_parts = normalized.split("/")
    if (
        not normalized
        or normalized.startswith("/")
        or (len(normalized) >= 2 and normalized[1] == ":")
        or ".." in raw_parts
    ):
        return None
    parts = [part for part in raw_parts if part not in {"", "."}]
    return "/".join(parts) or None


class _OutputView:
    def __init__(self, value: Mapping[str, object]) -> None:
        self._value = value

    def __getattr__(self, name: str) -> object:
        try:
            return _plain_value(self._value[name])
        except KeyError as exc:
            raise AttributeError(name) from exc

    def model_dump_json(self, *, exclude_none: bool = False) -> str:
        value = _plain_value(self._value)
        if exclude_none and isinstance(value, dict):
            value = {key: item for key, item in value.items() if item is not None}
        return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _result_output(result: ToolResult) -> _OutputView | None:
    value = result.structured_content
    if not isinstance(value, Mapping):
        return None
    return _OutputView(value)


def _plain_value(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _plain_value(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [_plain_value(item) for item in value]
    return value


__all__ = [
    "AnswerCandidate",
    "ComputationResult",
    "ContextBinding",
    "ContextUnit",
    "EvidenceRef",
    "ObservationBatch",
    "ObservationBuilder",
    "ObservationError",
    "ObservationExtractor",
    "StructuredObservation",
    "grounded_workspace_paths",
    "runtime_file_inspection_paths",
    "runtime_workspace_change",
    "runtime_workspace_file_changes",
    "runtime_workspace_snapshot",
    "tool_result_progress_error",
]
