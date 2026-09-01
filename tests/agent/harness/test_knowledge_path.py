from __future__ import annotations

import hashlib
from pathlib import Path

from agent_runtime.core.model_request import toolset_revision_for_tools
from agent_runtime.harness import (
    CompletionDecision,
    CompletionProposal,
    HarnessAgent,
    HarnessModelRequest,
    HarnessModelResponse,
    HarnessToolCall,
    PreparedModelCall,
    RuntimeComposition,
)
from agent_runtime.result import AgentResult
from agent_runtime.tools.integrations.knowledge import (
    KnowledgeSearchInput,
    KnowledgeSearchOutput,
)
from agent_runtime.tools.permissions import ToolExecutionContext


class KnowledgeModel:
    def snapshot(self, *, thread_id: str, turn_id: str) -> dict[str, str]:
        return {"model_alias": "knowledge-model", "model_revision": "model-v1"}

    def prepare(self, request: HarnessModelRequest) -> PreparedModelCall:
        digest = hashlib.sha256(f"step:{request.step}".encode()).hexdigest()
        revision = toolset_revision_for_tools(request.tools)
        return PreparedModelCall(
            request_hash=digest,
            context_hash=digest,
            tool_hash=revision,
            wire_hash=digest,
            request_ref={
                "request_id": f"{request.turn_id}:step:{request.step}",
                "toolset_revision": revision,
                "exposed_tool_names": [tool.definition.name for tool in request.tools],
            },
        )

    async def dispatch(self, prepared: PreparedModelCall) -> HarnessModelResponse:
        request_id = prepared.request_ref["request_id"]
        if isinstance(request_id, str) and request_id.endswith(":step:1"):
            return HarnessModelResponse(
                text="",
                provider_response_id="knowledge-call",
                usage={"input_tokens": 2, "output_tokens": 1},
                tool_calls=(
                    HarnessToolCall(
                        id="knowledge-call-1",
                        name="search_knowledge",
                        arguments={"query": "What is the retained fact?", "top_k": 3},
                    ),
                ),
            )
        return HarnessModelResponse(
            text="The retained fact is grounded.",
            provider_response_id="knowledge-answer",
            usage={"input_tokens": 8, "output_tokens": 4},
        )


class AcceptGroundedAnswer:
    def evaluate(self, proposal: CompletionProposal) -> CompletionDecision:
        return CompletionDecision(action="accept", reason="grounded result committed")


class DiscoveringKnowledgeModel(KnowledgeModel):
    def __init__(self) -> None:
        self.visible_tools: list[tuple[str, ...]] = []

    def prepare(self, request: HarnessModelRequest) -> PreparedModelCall:
        self.visible_tools.append(
            tuple(tool.definition.name for tool in request.tools)
        )
        return super().prepare(request)

    async def dispatch(self, prepared: PreparedModelCall) -> HarnessModelResponse:
        request_id = prepared.request_ref["request_id"]
        if isinstance(request_id, str) and request_id.endswith(":step:1"):
            return HarnessModelResponse(
                text="",
                provider_response_id="find-tools-call",
                usage={"input_tokens": 2, "output_tokens": 1},
                tool_calls=(
                    HarnessToolCall(
                        id="find-tools-1",
                        name="find_tools",
                        arguments={"query": "search knowledge documents", "limit": 1},
                    ),
                ),
            )
        if isinstance(request_id, str) and request_id.endswith(":step:2"):
            return await super().dispatch(
                PreparedModelCall(
                    request_hash=prepared.request_hash,
                    context_hash=prepared.context_hash,
                    tool_hash=prepared.tool_hash,
                    wire_hash=prepared.wire_hash,
                    request_ref={**prepared.request_ref, "request_id": "turn:step:1"},
                )
            )
        return HarnessModelResponse(
            text="The retained fact is grounded.",
            provider_response_id="knowledge-answer",
            usage={"input_tokens": 8, "output_tokens": 4},
        )


def test_configured_knowledge_uses_the_durable_harness_tool_path(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    calls: list[tuple[KnowledgeSearchInput, Path | None]] = []

    async def search(
        payload: KnowledgeSearchInput,
        execution_context: ToolExecutionContext,
    ) -> KnowledgeSearchOutput:
        calls.append((payload, execution_context.workspace_root))
        return KnowledgeSearchOutput.model_validate(
            {
                "results": [
                    {
                        "evidence_id": "evidence-1",
                        "doc_id": 1,
                        "citation_anchor": "doc-1#fact",
                        "text": "The retained fact",
                        "score": 0.99,
                        "source_type": "document",
                        "file_name": "facts.pdf",
                    }
                ],
                "answer_text": "The retained fact",
                "citations": ["doc-1#fact"],
                "groundedness_flag": True,
                "insufficient_evidence": False,
                "total_found": 1,
            }
        )

    with RuntimeComposition.open(
        database=tmp_path / "rollout.sqlite3",
        workspace=workspace,
        model=KnowledgeModel(),
        completion_gate=AcceptGroundedAnswer(),
        knowledge_runner=search,
        knowledge_revision="rag-corpus-v7",
        knowledge_config={"corpus": "fixture-v7"},
    ) as runtime:
        result = HarnessAgent(runtime.thread_manager).run("answer from knowledge")

        assert result.status == "completed"
        assert [(call.query, call.top_k, root) for call, root in calls] == [
            ("What is the retained fact?", 3, workspace.resolve())
        ]
        [operation] = runtime.store.list_tool_operations(result.turn_id)
        assert operation.tool_name == "search_knowledge"
        assert operation.status == "succeeded"
        [tool_result] = [
            item
            for item in runtime.store.list_items(result.turn_id)
            if item.kind == "tool_result"
        ]
        assert tool_result.payload["structured_content"]["citations"] == [
            "doc-1#fact"
        ]
        binding = runtime.store.read_turn(result.turn_id).binding_manifest
        assert binding["knowledge_revision"] == "rag-corpus-v7"
        assert binding["tool_execution_revisions"] == {
            "search_knowledge": "integration-search-knowledge-v1:rag-corpus-v7"
        }
        public = AgentResult._from_harness(result, store=runtime.store)
        [public_call] = public.tool_calls
        assert public_call.tool_name == "search_knowledge"
        assert public_call.arguments == {
            "query": "What is the retained fact?",
            "top_k": 3,
        }
        assert public_call.structured_output["citations"] == ("doc-1#fact",)
        assert public.usage.tool_calls == 1
        assert public.groundedness is True
        assert public.insufficient_evidence is False
        assert public.evidence[0].evidence_id == "evidence-1"
        assert public.citations[0].citation_anchor == "doc-1#fact"
        assert runtime.store.verify().valid is True


def test_discoverable_knowledge_is_hidden_until_find_tools_result_is_committed(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    async def search(
        _payload: KnowledgeSearchInput,
        execution_context: ToolExecutionContext,
    ) -> KnowledgeSearchOutput:
        assert execution_context.workspace_root == workspace.resolve()
        return KnowledgeSearchOutput(
            answer_text="The retained fact",
            citations=["doc-1#fact"],
            groundedness_flag=True,
        )

    model = DiscoveringKnowledgeModel()
    with RuntimeComposition.open(
        database=tmp_path / "rollout.sqlite3",
        workspace=workspace,
        model=model,
        completion_gate=AcceptGroundedAnswer(),
        knowledge_runner=search,
        knowledge_revision="rag-corpus-v7",
        knowledge_config={"corpus": "fixture-v7"},
        discoverable_tool_names=("search_knowledge",),
    ) as runtime:
        result = HarnessAgent(runtime.thread_manager).run("discover knowledge")

        assert result.status == "completed"
        assert model.visible_tools == [
            ("find_tools",),
            ("find_tools", "search_knowledge"),
            ("find_tools", "search_knowledge"),
        ]
        assert [
            (operation.tool_name, operation.status)
            for operation in runtime.store.list_tool_operations(result.turn_id)
        ] == [
            ("find_tools", "succeeded"),
            ("search_knowledge", "succeeded"),
        ]
        runtime.store.rebuild_projections()
        assert runtime.store.verify().valid is True
