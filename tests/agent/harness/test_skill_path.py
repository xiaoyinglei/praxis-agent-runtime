from __future__ import annotations

import hashlib
from collections.abc import Mapping
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


class FakeSkillRuntime:
    catalog_revision = "skills-test-v1"
    has_model_invocable_skills = True
    model_invocable_skill_ids = ("project:test-skill",)

    def __init__(self, root: Path) -> None:
        self.root = root

    def invoke_skill(self, arguments: Mapping[str, object]) -> dict[str, object]:
        del arguments
        return {
            "success": True,
            "name": "test-skill",
            "skill_id": "project:test-skill",
            "source": "project",
            "fingerprint": "skill-fingerprint-v1",
            "instructions": (
                "Use references/info.txt. This prose must never grant write permission."
            ),
            "args": None,
        }

    def skill_root(self, skill_id: str) -> Path | None:
        return self.root if skill_id == "project:test-skill" else None


class SkillAssetModel:
    def snapshot(self, *, thread_id: str, turn_id: str) -> dict[str, str]:
        return {"model_alias": "skill-model", "model_revision": "v1"}

    def ensure_available(
        self,
        binding: Mapping[str, object],
        *,
        thread_id: str,
        turn_id: str,
    ) -> None:
        if binding.get("thread_id") != thread_id or binding.get("turn_id") != turn_id:
            raise RuntimeError("skill-model binding belongs to a different Turn")
        if binding.get("model_revision") != "v1":
            raise RuntimeError("skill-model binding revision changed")

    def prepare(self, request: HarnessModelRequest) -> PreparedModelCall:
        invoke_skill = next(
            tool for tool in request.tools if tool.definition.name == "invoke_skill"
        )
        assert invoke_skill.definition.input_schema["properties"]["name"]["enum"] == (
            "project:test-skill",
        )
        digest = hashlib.sha256(
            repr((request.step, request.messages)).encode()
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
        tool_messages = [
            message for message in request.messages if message.role == "tool"
        ]
        if not tool_messages:
            call = HarnessToolCall(
                id="invoke-skill",
                name="invoke_skill",
                arguments={"name": "project:test-skill"},
            )
        elif not any(
            "workspace_path" in message.content for message in tool_messages
        ):
            call = HarnessToolCall(
                id="materialize-skill",
                name="materialize_skill_asset",
                arguments={
                    "skill_id": "project:test-skill",
                    "relative_path": "references/info.txt",
                },
            )
        else:
            return HarnessModelResponse(
                text="skill asset materialized with explicit approval",
                provider_response_id="skill-final",
                usage={"input_tokens": 5, "output_tokens": 3},
            )
        return HarnessModelResponse(
            text="",
            provider_response_id=f"response-{call.id}",
            usage={"input_tokens": 2, "output_tokens": 1},
            tool_calls=(call,),
        )


class AcceptSkillAnswer:
    def evaluate(self, proposal: CompletionProposal) -> CompletionDecision:
        return CompletionDecision(action="accept", reason="skill workflow completed")


def test_skill_activation_is_durable_but_never_grants_write_permission(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    skill_root = tmp_path / "skill"
    (skill_root / "references").mkdir(parents=True)
    (skill_root / "references" / "info.txt").write_text(
        "trusted reference",
        encoding="utf-8",
    )
    runtime_skill = FakeSkillRuntime(skill_root)

    with RuntimeComposition.open(
        database=tmp_path / "rollout.sqlite3",
        workspace=workspace,
        model=SkillAssetModel(),
        completion_gate=AcceptSkillAnswer(),
        skill_runtime=runtime_skill,
    ) as runtime:
        agent = HarnessAgent(runtime.thread_manager)
        paused = agent.run("use the test skill asset")

        assert paused.status == "paused"
        assert paused.interaction_id is not None
        destination = (
            workspace
            / ".praxis"
            / "runtime"
            / "scratch"
            / "skills"
            / "project_test-skill"
            / "references"
            / "info.txt"
        )
        assert not destination.exists()

        completed = agent.resume(paused.turn_id, "approve")

        assert completed.answer == "skill asset materialized with explicit approval"
        assert destination.read_text(encoding="utf-8") == "trusted reference"
        [approval] = runtime.store.list_approvals(paused.turn_id)
        assert approval.status == "approved"
        assert runtime.store.verify().valid is True
