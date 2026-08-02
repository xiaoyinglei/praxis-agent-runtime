from __future__ import annotations

from rag.agent.core.observations import (
    ComputationResult,
    ContextUnit,
    EvidenceRef,
    ObservationExtractor,
    StructuredObservation,
    grounded_workspace_paths,
)
from rag.agent.tools.builtins.filesystem import (
    FileEntry,
    ListFilesOutput,
    ReadFileOutput,
)
from rag.agent.tools.builtins.search import SearchTextMatch, SearchTextOutput
from rag.agent.tools.tool import ToolCall, ToolCallOrigin, ToolResult
from rag.schema.query import AnswerCitation, EvidenceItem


def test_canonical_list_files_observation_preserves_workspace_locator() -> None:
    output = ListFilesOutput(
        entries=[
            FileEntry(
                path="input_files/data.csv",
                name="data.csv",
                size_bytes=12,
                is_directory=False,
                is_symlink=False,
            )
        ]
    )
    result = ToolResult(
        tool_call_id="tc-list",
        tool_name="list_files",
        structured_content=output.model_dump(mode="json"),
    )

    update = ObservationExtractor().reduce_tool_results({"tool_results": [result]})

    expected_locator = {
        "source_tool": "list_files",
        "path": "input_files/data.csv",
        "name": "data.csv",
        "size_bytes": 12,
        "is_dir": False,
    }
    [unit] = update["context_units"]
    assert unit == ContextUnit(
        unit_id="workspace_file:input_files/data.csv",
        unit_type="workspace_file",
        locator=expected_locator,
        preview="input_files/data.csv (12 bytes)",
        content_ref="input_files/data.csv",
        capabilities=["read_file"],
        metadata={"source_tool": "list_files"},
    )
    assert update["locators"] == [expected_locator]


def test_successful_inspection_grounds_its_requested_container() -> None:
    origin = ToolCallOrigin(
        request_id="ground-list-root",
        toolset_revision="ground-list-root-tools",
        exposed_tool_names=("list_files",),
    )
    call = ToolCall(
        tool_call_id="tc-list-root",
        tool_name="list_files",
        arguments={"path": ".", "limit": 50},
        origin=origin,
    )
    result = ToolResult(
        tool_call_id=call.tool_call_id,
        tool_name=call.tool_name,
        structured_content={
            "entries": [
                {
                    "path": "agent_runtime",
                    "name": "agent_runtime",
                    "is_directory": True,
                }
            ]
        },
    )

    paths = grounded_workspace_paths(
        tool_results=[result],
        tool_calls={call.tool_call_id: call},
    )

    assert paths == (".", "agent_runtime")


def test_grounded_paths_apply_the_published_search_default() -> None:
    origin = ToolCallOrigin(
        request_id="ground-default-search",
        toolset_revision="ground-default-search-tools",
        exposed_tool_names=("search_text",),
    )
    call = ToolCall(
        tool_call_id="tc-search-root",
        tool_name="search_text",
        arguments={"pattern": "apply_patch"},
        origin=origin,
    )
    result = ToolResult(
        tool_call_id=call.tool_call_id,
        tool_name=call.tool_name,
        structured_content={
            "matches": [],
            "total_matches": 0,
            "truncated": False,
        },
    )

    paths = grounded_workspace_paths(
        tool_results=[result],
        tool_calls={call.tool_call_id: call},
    )

    assert paths == (".",)


def test_canonical_read_file_observation_preserves_content_and_locator() -> None:
    output = ReadFileOutput(
        path="input_files/data.csv",
        content="name,value\nalpha,1\n",
        size_bytes=19,
        offset=0,
        truncated=False,
        is_binary=False,
        encoding="utf-8",
    )
    result = ToolResult(
        tool_call_id="tc-read",
        tool_name="read_file",
        structured_content=output.model_dump(mode="json"),
    )

    update = ObservationExtractor().reduce_tool_results({"tool_results": [result]})

    expected_locator = {
        "source_tool": "read_file",
        "path": "input_files/data.csv",
        "size_bytes": 19,
        "truncated": False,
        "is_binary": False,
        "encoding": "utf-8",
    }
    [unit] = update["context_units"]
    assert unit == ContextUnit(
        unit_id="workspace_file:input_files/data.csv",
        unit_type="workspace_file_content",
        locator=expected_locator,
        preview="name,value\nalpha,1\n",
        content_ref="tc-read",
        capabilities=["read_file"],
        metadata={"source_tool": "read_file"},
    )
    assert update["locators"] == [expected_locator]


def test_search_text_observation_preserves_patchable_source_locators() -> None:
    output = SearchTextOutput(
        matches=[
            SearchTextMatch(
                file_path="rag/agent/loop/runtime.py",
                line_number=718,
                line_content="                    _stream_tool_use_result(",
                match_start=20,
                match_end=43,
            ),
            SearchTextMatch(
                file_path="rag/agent/cli.py",
                line_number=111,
                line_content="    def _render_tool_result(self, event: StreamEvent) -> None:",
                match_start=8,
                match_end=27,
            ),
        ],
        total_matches=2,
    )
    result = ToolResult(
        tool_call_id="tc-search-code",
        tool_name="search_text",
        structured_content=output.model_dump(mode="json"),
    )

    update = ObservationExtractor().reduce_tool_results({"tool_results": [result]})

    assert update["locators"] == [
        {
            "source_tool": "search_text",
            "path": "rag/agent/loop/runtime.py",
            "line_number": 718,
        },
        {
            "source_tool": "search_text",
            "path": "rag/agent/cli.py",
            "line_number": 111,
        },
    ]


def test_empty_search_is_not_a_successful_plan_observation() -> None:
    result = ToolResult(
        tool_call_id="tc-search-empty",
        tool_name="search_text",
        structured_content={
            "matches": [],
            "total_matches": 0,
            "truncated": False,
        },
    )

    update = ObservationExtractor().reduce_tool_results(
        {"tool_results": [result]}
    )

    [observation] = update["structured_observations"]
    assert observation.status == "error"
    assert observation.error == "search returned no matches"


def test_neutral_rag_observation_preserves_evidence_and_citations() -> None:
    evidence = EvidenceItem(
        evidence_id="ev-policy",
        doc_id=7,
        citation_anchor="policy#3",
        text="Policy evidence",
        score=0.91,
        page_start=3,
        page_end=4,
        retrieval_channels=["vector", "rerank"],
    )
    citation = AnswerCitation(
        citation_id="cit-policy",
        evidence_id="ev-policy",
        record_type="section",
        citation_anchor="policy#3",
        doc_id=7,
        page_start=3,
        page_end=4,
    )
    result = ToolResult(
        tool_call_id="tc-rag",
        tool_name="rag_search_answer",
        structured_content={
            "text": "Grounded answer",
            "evidence": [evidence.model_dump(mode="json")],
            "citations": [citation.model_dump(mode="json")],
            "groundedness_flag": True,
        },
    )

    update = ObservationExtractor().reduce_tool_results({"tool_results": [result]})

    assert update["evidence"] == [evidence]
    assert update["citations"] == [citation]
    assert update["evidence_refs"] == [
        EvidenceRef(
            evidence_id="ev-policy",
            citation_anchor="policy#3",
            doc_id=7,
            source="evidence",
        ),
        EvidenceRef(
            evidence_id="ev-policy",
            citation_id="cit-policy",
            citation_anchor="policy#3",
            doc_id=7,
            source="citation",
        ),
    ]
    assert update["answer_candidates"][0].text == "Grounded answer"


def test_retrieval_observation_preserves_score_rerank_score_and_locator() -> None:
    result = ToolResult(
        tool_call_id="tc-search",
        tool_name="rerank",
        structured_content={
            "items": [
                {
                    "text": "Ranked evidence",
                    "doc_id": 8,
                    "section_id": 5,
                    "page_start": 2,
                    "page_end": 2,
                    "record_type": "section",
                    "citation_anchor": "doc#5",
                    "evidence_id": "ev-ranked",
                    "score": 0.73,
                    "rerank_score": 0.98,
                    "retrieval_channels": ["vector", "rerank"],
                }
            ]
        },
    )

    update = ObservationExtractor().reduce_tool_results({"tool_results": [result]})

    [unit] = update["context_units"]
    assert unit == ContextUnit(
        unit_id="retrieval:ev-ranked",
        unit_type="document_section",
        locator={
            "doc_id": 8,
            "section_id": 5,
            "page_start": 2,
            "page_end": 2,
            "record_type": "section",
            "citation_anchor": "doc#5",
            "evidence_id": "ev-ranked",
            "score": 0.73,
            "rerank_score": 0.98,
            "retrieval_channels": ["vector", "rerank"],
        },
        preview="Ranked evidence",
        content_ref="ev-ranked",
        evidence_refs=[
            EvidenceRef(
                evidence_id="ev-ranked",
                citation_anchor="doc#5",
                doc_id=8,
                source="retrieval",
            )
        ],
        capabilities=["text_extract", "text_synthesize", "quote"],
        metadata={"source_tool": "rerank"},
    )


def test_computation_observation_preserves_expression_and_asset_provenance() -> None:
    result = ToolResult(
        tool_call_id="tc-compute",
        tool_name="asset_analyze",
        structured_content={
            "asset_id": 14,
            "asset_type": "table",
            "sheet_name": "Sales",
            "operation": "dataframe_sql",
            "columns": ["total"],
            "rows": [["15.49"]],
            "raw_row_count": 1,
            "elapsed_ms": 1.0,
            "truncated": False,
            "query": "SELECT SUM(amount) AS total FROM sheet",
            "markdown": "| total |\n|---|\n| 15.49 |",
        },
    )

    update = ObservationExtractor().reduce_tool_results({"tool_results": [result]})

    assert update["computation_results"] == [
        ComputationResult(
            source_tool_call_id="tc-compute",
            source_tool_name="asset_analyze",
            operation="dataframe_sql",
            value_preview="| total |\n|---|\n| 15.49 |",
            expression="SELECT SUM(amount) AS total FROM sheet",
            evidence_refs=[EvidenceRef(evidence_id="asset:14", source="asset")],
        )
    ]
    assert update["locators"] == [
        {
            "asset_id": 14,
            "asset_type": "table",
            "sheet_name": "Sales",
            "columns": ["total"],
        }
    ]
    assert update["asset_refs"] == [14]


def test_structured_tool_error_is_visible_without_controller_fields() -> None:
    result = ToolResult(
        tool_call_id="tc-error",
        tool_name="vector_search",
        is_error=True,
        error_code="timeout",
        error_message="retrieval timed out",
        retryable=True,
        metadata={"provider": "vector"},
    )

    update = ObservationExtractor().reduce_tool_results({"tool_results": [result]})

    assert update["structured_observations"] == [
        StructuredObservation(
            tool_call_id="tc-error",
            tool_name="vector_search",
            status="error",
            error="retrieval timed out",
            raw_result_ref="tc-error",
        )
    ]
    assert update["errors"] == [
        {
            "tool_call_id": "tc-error",
            "tool_name": "vector_search",
            "code": "timeout",
            "message": "retrieval timed out",
            "retryable": True,
            "detail": {"provider": "vector"},
        }
    ]
    assert {
        "satisfied_requirements",
        "open_gaps",
        "no_progress_count",
        "iteration",
        "controller_next",
    }.isdisjoint(update)


def test_nonzero_command_is_not_a_successful_plan_observation() -> None:
    result = ToolResult(
        tool_call_id="tc-failed-check",
        tool_name="run_command",
        structured_content={
            "stdout": "",
            "stderr": "1 failed",
            "exit_code": 1,
            "timed_out": False,
            "sandbox_error": None,
        },
    )

    update = ObservationExtractor().reduce_tool_results(
        {"tool_results": [result]}
    )

    [observation] = update["structured_observations"]
    assert observation.status == "error"
    assert observation.error == "command exited with status 1"


def test_noop_patch_is_not_a_successful_plan_observation() -> None:
    result = ToolResult(
        tool_call_id="tc-noop-patch",
        tool_name="apply_patch",
        structured_content={
            "file_path": "rag/agent/loop/runtime.py",
            "replaced": False,
            "occurrences": 0,
            "message": "No change.",
        },
    )

    update = ObservationExtractor().reduce_tool_results(
        {"tool_results": [result]}
    )

    [observation] = update["structured_observations"]
    assert observation.status == "error"
    assert observation.error == "write tool produced no workspace change"


def test_neutral_reducer_skips_already_observed_tool_calls() -> None:
    result = ToolResult(
        tool_call_id="tc-text",
        tool_name="llm_summarize",
        structured_content={"text": "Summary"},
    )
    existing = StructuredObservation(
        tool_call_id="tc-text",
        tool_name="llm_summarize",
        status="ok",
        raw_result_ref="tc-text",
    )

    update = ObservationExtractor().reduce_tool_results(
        {
            "tool_results": [result],
            "structured_observations": [existing],
        }
    )

    assert update == {}
