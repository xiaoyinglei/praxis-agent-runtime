from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

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
from agent_runtime.tools.builtins.shell import create_run_command_tool
from agent_runtime.tools.permissions import ToolExecutionContext
from agent_runtime.tools.registry import ToolRegistry
from agent_runtime.workspace import open_workspace


class SandboxCommandModel:
    def snapshot(self) -> dict[str, str]:
        return {"model_alias": "sandbox-model", "model_revision": "sandbox-v1"}

    def prepare(self, request: HarnessModelRequest) -> PreparedModelCall:
        digest = hashlib.sha256(f"step:{request.step}".encode()).hexdigest()
        toolset_revision = toolset_revision_for_tools(request.tools)
        return PreparedModelCall(
            request_hash=digest,
            context_hash=digest,
            tool_hash=toolset_revision,
            wire_hash=digest,
            request_ref={
                "request_id": f"{request.turn_id}:step:{request.step}",
                "toolset_revision": toolset_revision,
                "exposed_tool_names": [tool.definition.name for tool in request.tools],
            },
        )

    async def dispatch(self, prepared: PreparedModelCall) -> HarnessModelResponse:
        request_id = prepared.request_ref["request_id"]
        if isinstance(request_id, str) and request_id.endswith(":step:1"):
            return HarnessModelResponse(
                text="",
                provider_response_id="sandbox-tool-call",
                usage={"input_tokens": 2, "output_tokens": 1},
                tool_calls=(
                    HarnessToolCall(
                        id="sandbox-call-1",
                        name="run_command",
                        arguments={
                            "command": "printf seatbelt-harness-ok",
                            "working_dir": ".",
                            "network": False,
                            "workspace_write": False,
                        },
                    ),
                ),
            )
        return HarnessModelResponse(
            text="sandbox command verified",
            provider_response_id="sandbox-final",
            usage={"input_tokens": 4, "output_tokens": 2},
        )


class AcceptSandboxAnswer:
    def evaluate(self, proposal: CompletionProposal) -> CompletionDecision:
        return CompletionDecision(action="accept", reason="sandbox result committed")


class SandboxWriteModel(SandboxCommandModel):
    async def dispatch(self, prepared: PreparedModelCall) -> HarnessModelResponse:
        request_id = prepared.request_ref["request_id"]
        if isinstance(request_id, str) and request_id.endswith(":step:1"):
            return HarnessModelResponse(
                text="",
                provider_response_id="sandbox-write-call",
                usage={"input_tokens": 2, "output_tokens": 1},
                tool_calls=(
                    HarnessToolCall(
                        id="sandbox-write-1",
                        name="run_command",
                        arguments={
                            "command": "printf approved-by-seatbelt > approved.txt",
                            "working_dir": ".",
                            "network": False,
                            "workspace_write": True,
                        },
                    ),
                ),
            )
        return HarnessModelResponse(
            text="sandbox write verified",
            provider_response_id="sandbox-write-final",
            usage={"input_tokens": 4, "output_tokens": 2},
        )


@pytest.mark.skipif(
    not Path("/usr/bin/sandbox-exec").is_file(),
    reason="real macOS Seatbelt is unavailable",
)
def test_candidate_sdk_runs_retained_command_tool_in_real_seatbelt(
    tmp_path: Path,
) -> None:
    workspace = open_workspace(tmp_path / "workspace", create=True)
    registry = ToolRegistry()
    registry.register(create_run_command_tool(workspace))

    with RuntimeComposition.open(
        database=tmp_path / "rollout.sqlite3",
        workspace=workspace.root,
        model=SandboxCommandModel(),
        completion_gate=AcceptSandboxAnswer(),
        tools=registry.freeze(),
        tool_execution_context=ToolExecutionContext(
            workspace_root=workspace.root,
            cwd=workspace.root,
            auto_approve_sandboxed=True,
        ),
    ) as runtime:
        result = HarnessAgent(runtime.thread_manager).run("verify Seatbelt")

        assert result.status == "completed"
        [tool_result] = [
            item
            for item in runtime.store.list_items(result.turn_id)
            if item.kind == "tool_result"
        ]
        output = tool_result.payload["structured_content"]
        assert output["stdout"] == "seatbelt-harness-ok"
        assert output["execution_mode"] == "restricted_sandbox"
        assert output["network_enabled"] is False
        [operation] = runtime.store.list_tool_operations(result.turn_id)
        assert operation.status == "succeeded"
        assert operation.claim_generation == 1
        assert runtime.store.verify().valid is True


@pytest.mark.skipif(
    not Path("/usr/bin/sandbox-exec").is_file(),
    reason="real macOS Seatbelt is unavailable",
)
def test_real_seatbelt_workspace_write_requires_durable_approval(
    tmp_path: Path,
) -> None:
    workspace = open_workspace(tmp_path / "workspace", create=True)
    registry = ToolRegistry()
    registry.register(create_run_command_tool(workspace))

    with RuntimeComposition.open(
        database=tmp_path / "rollout.sqlite3",
        workspace=workspace.root,
        model=SandboxWriteModel(),
        completion_gate=AcceptSandboxAnswer(),
        tools=registry.freeze(),
        tool_execution_context=ToolExecutionContext(
            workspace_root=workspace.root,
            cwd=workspace.root,
            auto_approve_sandboxed=True,
        ),
    ) as runtime:
        agent = HarnessAgent(runtime.thread_manager)
        paused = agent.run("write through Seatbelt")

        assert paused.status == "paused"
        assert not (workspace.root / "approved.txt").exists()

        resumed = agent.resume(paused.turn_id, "approve")

        assert resumed.turn_id == paused.turn_id
        assert resumed.status == "completed"
        assert (workspace.root / "approved.txt").read_text(encoding="utf-8") == (
            "approved-by-seatbelt"
        )
        [tool_result] = [
            item
            for item in runtime.store.list_items(resumed.turn_id)
            if item.kind == "tool_result"
        ]
        metadata = tool_result.payload["metadata"]
        assert metadata["runtime_workspace_write"] is True
        assert metadata["workspace_tree_changed"] is True
        [operation] = runtime.store.list_tool_operations(resumed.turn_id)
        assert operation.status == "succeeded"
        assert operation.claim_generation == 1
        assert runtime.store.verify().valid is True
