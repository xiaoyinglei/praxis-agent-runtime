from __future__ import annotations

import hashlib
from pathlib import Path

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
from agent_runtime.tools.permissions import ToolExecutionContext


class ParentChildModel:
    def __init__(self) -> None:
        self.requests: list[HarnessModelRequest] = []

    def snapshot(self) -> dict[str, str]:
        return {"model_alias": "parent-child-model", "model_revision": "v1"}

    def prepare(self, request: HarnessModelRequest) -> PreparedModelCall:
        self.requests.append(request)
        digest = hashlib.sha256(
            repr(
                (
                    request.turn_id,
                    request.step,
                    request.messages,
                    tuple(tool.definition.name for tool in request.tools),
                )
            ).encode()
        ).hexdigest()
        return PreparedModelCall(
            request_hash=digest,
            context_hash=digest,
            tool_hash=digest,
            wire_hash=digest,
            request_ref={
                "request_id": f"{request.turn_id}:{request.step}",
                "toolset_revision": digest,
                "exposed_tool_names": [
                    tool.definition.name for tool in request.tools
                ],
            },
            dispatch_payload=request,
        )

    async def dispatch(self, prepared: PreparedModelCall) -> HarnessModelResponse:
        request = prepared.dispatch_payload
        assert isinstance(request, HarnessModelRequest)
        initial = request.messages[0].content
        if initial.startswith("child task"):
            return HarnessModelResponse(
                text="child conclusion",
                provider_response_id="child-response",
                usage={"input_tokens": 2, "output_tokens": 2},
            )
        tool_messages = [
            message for message in request.messages if message.role == "tool"
        ]
        if not tool_messages:
            return HarnessModelResponse(
                text="",
                provider_response_id="discover-response",
                usage={"input_tokens": 2, "output_tokens": 1},
                tool_calls=(
                    HarnessToolCall(
                        id="discover-task",
                        name="find_tools",
                        arguments={"query": "delegate isolated child task", "limit": 3},
                    ),
                ),
            )
        if not any("child_turn_id" in message.content for message in tool_messages):
            return HarnessModelResponse(
                text="",
                provider_response_id="delegate-response",
                usage={"input_tokens": 3, "output_tokens": 1},
                tool_calls=(
                    HarnessToolCall(
                        id="delegate-task",
                        name="task",
                        arguments={
                            "task": "child task: inspect only the supplied context",
                            "context_summary": "bounded parent fact",
                            "max_turns": 2,
                            "llm_budget_total": 123,
                        },
                    ),
                ),
            )
        return HarnessModelResponse(
            text="parent used child conclusion",
            provider_response_id="parent-response",
            usage={"input_tokens": 5, "output_tokens": 3},
        )


class AcceptBothAnswers:
    def evaluate(self, proposal: CompletionProposal) -> CompletionDecision:
        assert proposal.answer in {
            "child conclusion",
            "parent used child conclusion",
        }
        return CompletionDecision(action="accept", reason="bounded result accepted")


def test_subagent_tool_runs_an_independent_durable_thread(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    model = ParentChildModel()
    with RuntimeComposition.open(
        database=tmp_path / "rollout.sqlite3",
        workspace=workspace,
        model=model,
        completion_gate=AcceptBothAnswers(),
        tool_execution_context=ToolExecutionContext(
            workspace_root=workspace,
            cwd=workspace,
        ),
        enable_subagents=True,
    ) as runtime:
        agent = HarnessAgent(runtime.thread_manager)
        paused = agent.run("delegate this work")

        assert paused.status == "paused"
        assert paused.interaction_id is not None
        completed = agent.resume(paused.turn_id, "approve")

        assert completed.answer == "parent used child conclusion"
        threads = runtime.store.list_threads()
        assert len(threads) == 2
        child_thread = next(
            thread for thread in threads if thread.thread_id != completed.thread_id
        )
        child_turn_id = child_thread.head_turn_id
        assert child_turn_id is not None
        child_binding = runtime.store.read_turn(child_turn_id).binding_manifest
        assert child_binding["model_step_budget"] == 2
        assert child_binding["model_token_budget_total"] == 123
        assert child_binding["completion_policy"] == {
            "require_workspace_change": False
        }
        assert [
            item.payload.get("text")
            for item in runtime.store.list_context_items(child_turn_id)
            if item.kind == "user_message"
        ] == [
            "child task: inspect only the supplied context\n\n"
            "Context supplied by the parent agent:\n"
            "bounded parent fact"
        ]
        parent_results = [
            item
            for item in runtime.store.list_items(completed.turn_id)
            if item.kind == "tool_result"
            and item.payload.get("tool_name") == "task"
        ]
        assert parent_results[0].payload["metadata"]["child_turn_id"] == child_turn_id
        assert runtime.store.verify().valid is True
