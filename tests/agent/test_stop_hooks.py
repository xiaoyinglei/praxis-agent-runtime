from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import pytest
from pydantic import BaseModel

from rag.agent.core.context import AgentRunConfig
from rag.agent.core.definition import AgentRuntimePolicy
from rag.agent.core.finalization import (
    FinishCandidateBuilder,
    FinishCandidateBuildError,
)
from rag.agent.core.goal_contract import GoalConstraint, GoalDeliverable, GoalSpec
from rag.agent.core.output_finalizer import OutputValidationExhaustedError
from rag.agent.loop.state import ModelTurnDraft, create_loop_state
from rag.agent.loop.stop_hooks import (
    GoalContractStopHook,
    StopHookBinding,
    StopHookRunner,
    StopVerdict,
    StructuredOutputStopHook,
    build_stop_hooks,
)
from rag.agent.tools.tool import ToolCall, ToolCallOrigin, ToolResult
from rag.agent.workspace import workspace_tree_sha256


class _StructuredAnswer(BaseModel):
    answer: str


def _config(run_id: str = "stop-hooks") -> AgentRunConfig:
    return AgentRunConfig(
        turn_id=run_id,
        llm_budget_total=100,
    )


def _state():
    return create_loop_state(current_message="Answer carefully", run_config=_config())


def _change_metadata(
    *,
    path: str = "src/example.py",
    before_sha256: str = "a" * 64,
    after_sha256: str = "b" * 64,
) -> dict[str, object]:
    return {
        "workspace_changed": True,
        "file_path": path,
        "before_sha256": before_sha256,
        "after_sha256": after_sha256,
    }


def _tree_change_metadata(
    *,
    before_sha256: str = "c" * 64,
    after_sha256: str = "d" * 64,
) -> dict[str, object]:
    return {
        "runtime_workspace_write": True,
        "workspace_tree_changed": before_sha256 != after_sha256,
        "workspace_tree_before_sha256": before_sha256,
        "workspace_tree_after_sha256": after_sha256,
    }


@dataclass
class _StaticHook:
    verdict: StopVerdict
    calls: list[str] | None = None

    async def evaluate(self, *, state: object, candidate: str) -> StopVerdict:
        del state
        if self.calls is not None:
            self.calls.append(candidate)
        return self.verdict


class _FailingHook:
    async def evaluate(self, *, state: object, candidate: str) -> StopVerdict:
        del state, candidate
        raise RuntimeError("hook unavailable")


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("action", "accepted", "halted"),
    [
        ("accept", True, False),
        ("warn", True, False),
        ("block", False, False),
        ("halt", False, True),
    ],
)
async def test_runner_supports_all_verdict_actions(
    action: str,
    accepted: bool,
    halted: bool,
) -> None:
    state = _state()
    runner = StopHookRunner(
        hooks=[
            StopHookBinding(
                name="static",
                hook=_StaticHook(
                    StopVerdict(
                        action=action,
                        code=f"static_{action}",
                        message=f"{action} message",
                    )
                ),
                critical=True,
            )
        ],
        max_blocks=3,
    )

    outcome = await runner.evaluate(state=state, candidate="candidate")

    assert outcome.accepted is accepted
    assert outcome.halted is halted
    if action == "warn":
        assert state["finish_state"].warnings[0].code == "static_warn"
    if action == "block":
        assert state["finish_state"].feedback[0].code == "static_block"


@pytest.mark.anyio
async def test_hooks_run_in_stable_order_and_warning_does_not_block() -> None:
    calls: list[str] = []
    state = _state()
    runner = StopHookRunner(
        hooks=[
            StopHookBinding(
                name="warning",
                hook=_StaticHook(
                    StopVerdict(
                        action="warn",
                        code="warning",
                        message="advisory warning",
                    ),
                    calls=calls,
                ),
                critical=False,
            ),
            StopHookBinding(
                name="accept",
                hook=_StaticHook(
                    StopVerdict(action="accept", code="accepted"),
                    calls=calls,
                ),
                critical=True,
            ),
        ],
        max_blocks=3,
    )

    outcome = await runner.evaluate(state=state, candidate="candidate")

    assert outcome.accepted is True
    assert calls == ["candidate", "candidate"]
    assert [item.code for item in state["finish_state"].warnings] == ["warning"]


@pytest.mark.anyio
async def test_repeated_equivalent_block_halts_at_configured_limit() -> None:
    state = _state()
    runner = StopHookRunner(
        hooks=[
            StopHookBinding(
                name="critic",
                hook=_StaticHook(
                    StopVerdict(
                        action="block",
                        code="missing_citation",
                        message="Add a citation.",
                    )
                ),
                critical=True,
            )
        ],
        max_blocks=2,
    )

    first = await runner.evaluate(state=state, candidate="draft one")
    second = await runner.evaluate(state=state, candidate="draft two")

    assert first.blocked is True
    assert second.halted is True
    assert second.code == "stop_hook_block_limit"
    assert state["finish_state"].feedback[0].occurrences == 2


@pytest.mark.anyio
async def test_advisory_failure_warns_but_critical_failure_halts() -> None:
    advisory_state = _state()
    advisory = StopHookRunner(
        hooks=[
            StopHookBinding(
                name="advisory",
                hook=_FailingHook(),
                critical=False,
            )
        ],
        max_blocks=2,
    )
    critical_state = _state()
    critical = StopHookRunner(
        hooks=[
            StopHookBinding(
                name="critical",
                hook=_FailingHook(),
                critical=True,
            )
        ],
        max_blocks=2,
    )

    advisory_outcome = await advisory.evaluate(
        state=advisory_state,
        candidate="candidate",
    )
    critical_outcome = await critical.evaluate(
        state=critical_state,
        candidate="candidate",
    )

    assert advisory_outcome.accepted is True
    assert advisory_state["finish_state"].warnings[0].code == "advisory_failed"
    assert critical_outcome.halted is True
    assert critical_outcome.code == "critical_failed"


class _StructuredFinalizer:
    async def finalize(
        self,
        *,
        definition: AgentRuntimePolicy,
        state: object,
        candidate_text: str,
    ) -> BaseModel:
        del definition, state
        return _StructuredAnswer(answer=candidate_text.upper())


class _ExhaustedFinalizer:
    async def finalize(
        self,
        *,
        definition: AgentRuntimePolicy,
        state: object,
        candidate_text: str,
    ) -> BaseModel:
        del definition, state, candidate_text
        raise OutputValidationExhaustedError(
            attempts=2,
            validation_errors=[
                {
                    "location": ["answer"],
                    "message": "field required",
                    "type": "missing",
                }
            ],
        )


@pytest.mark.anyio
async def test_structured_output_hook_is_critical_and_returns_validated_output() -> None:
    definition = AgentRuntimePolicy.test_factory(
        system_prompt="Return structured output.",
        allowed_tools=[],
        output_model=_StructuredAnswer,
    )
    state = _state()
    runner = StopHookRunner(
        hooks=[
            StopHookBinding(
                name="structured_output",
                hook=StructuredOutputStopHook(
                    definition=definition,
                    finalizer=_StructuredFinalizer(),
                ),
                critical=True,
            )
        ],
        max_blocks=definition.max_stop_hook_blocks,
    )

    outcome = await runner.evaluate(state=state, candidate="done")

    assert outcome.accepted is True
    assert outcome.final_output is not None
    assert outcome.final_output.data == {"answer": "DONE"}


@pytest.mark.anyio
async def test_structured_output_exhaustion_halts_with_validation_details() -> None:
    definition = AgentRuntimePolicy.test_factory(
        system_prompt="Return structured output.",
        allowed_tools=[],
        output_model=_StructuredAnswer,
    )
    runner = StopHookRunner(
        hooks=[
            StopHookBinding(
                name="structured_output",
                hook=StructuredOutputStopHook(
                    definition=definition,
                    finalizer=_ExhaustedFinalizer(),
                ),
                critical=True,
            )
        ],
        max_blocks=definition.max_stop_hook_blocks,
    )

    outcome = await runner.evaluate(state=_state(), candidate="invalid")

    assert outcome.halted is True
    assert outcome.code == "structured_output_invalid"
    assert outcome.detail["attempts"] == 2


@pytest.mark.anyio
async def test_explicit_goal_contract_blocks_missing_evidence_without_routing_fields() -> None:
    goal = GoalSpec(
        original_query="Answer with evidence",
        deliverables=[
            GoalDeliverable(
                deliverable_id="answer",
                kind="answer",
                acceptance_rule="non_empty_answer",
            ),
            GoalDeliverable(
                deliverable_id="evidence",
                kind="evidence",
                acceptance_rule="traceable_evidence",
            ),
        ],
    )
    state = _state()
    hook = GoalContractStopHook(goal_spec=goal)

    verdict = await hook.evaluate(state=state, candidate="Unsupported answer")

    assert verdict.action == "block"
    assert verdict.code == "goal_contract_unsatisfied"
    assert verdict.detail["unsatisfied_issue_ids"] == ["evidence"]
    assert "open_gap_ids" not in verdict.detail
    assert "open_gaps" not in state
    assert "satisfied_requirements" not in state


@pytest.mark.anyio
async def test_explicit_goal_contract_accepts_traceable_evidence() -> None:
    goal = GoalSpec(
        original_query="Answer with evidence",
        deliverables=[
            GoalDeliverable(
                deliverable_id="answer",
                kind="answer",
                acceptance_rule="non_empty_answer",
            ),
            GoalDeliverable(
                deliverable_id="evidence",
                kind="evidence",
                acceptance_rule="traceable_evidence",
            ),
        ],
    )
    state = _state()
    state["tool_results"] = [
        ToolResult(
            tool_call_id="tc-1",
            tool_name="test",
            structured_content={
                "evidence_refs": [
                    {
                        "evidence_id": "evidence-1",
                        "citation_id": "citation-1",
                    }
                ]
            },
        )
    ]

    verdict = await GoalContractStopHook(goal_spec=goal).evaluate(
        state=state,
        candidate="Supported answer",
    )

    assert verdict.action == "accept"
    assert verdict.code == "goal_contract_satisfied"


@pytest.mark.anyio
async def test_workspace_change_goal_rejects_prose_only_completion() -> None:
    goal = GoalSpec(
        original_query="Fix the implementation.",
        constraints=[
            GoalConstraint(
                constraint_id="workspace_change",
                constraint_type="workspace_change",
                expected_value=True,
            )
        ],
    )

    verdict = await GoalContractStopHook(goal_spec=goal).evaluate(
        state=_state(),
        candidate="Here is how you could fix it.",
    )

    assert verdict.action == "block"
    assert verdict.code == "goal_contract_unsatisfied"
    assert verdict.detail["unsatisfied_issue_ids"] == [
        "constraint:workspace_change"
    ]
    assert "real workspace change" in (verdict.message or "")


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("constraint_id", "constraint_type"),
    [
        ("workspace_change", "workspace_change"),
        ("verification_after_change", "verification_after_change"),
    ],
)
async def test_runtime_owned_constraint_rejects_tool_claimed_binding(
    constraint_id: str,
    constraint_type: str,
) -> None:
    goal = GoalSpec(
        original_query="Fix and verify the implementation.",
        constraints=[
            GoalConstraint(
                constraint_id=constraint_id,
                constraint_type=constraint_type,
                expected_value=True,
            )
        ],
    )
    state = _state()
    state["tool_results"] = [
        ToolResult(
            tool_call_id="untrusted-binding",
            tool_name="external_tool",
            structured_content={
                "context_bindings": [
                    {
                        "binding_id": "tool-claimed-completion",
                        "constraint_id": constraint_id,
                        "status": "satisfied",
                    }
                ]
            },
        )
    ]

    verdict = await GoalContractStopHook(goal_spec=goal).evaluate(
        state=state,
        candidate="The tool says this is complete.",
    )

    assert verdict.action == "block"
    assert verdict.detail["unsatisfied_issue_ids"] == [
        f"constraint:{constraint_id}"
    ]


@pytest.mark.anyio
async def test_workspace_change_rejects_untrusted_metadata_claim() -> None:
    goal = GoalSpec(
        original_query="Fix the implementation.",
        constraints=[
            GoalConstraint(
                constraint_id="workspace_change",
                constraint_type="workspace_change",
                expected_value=True,
            )
        ],
    )
    state = _state()
    state["tool_results"] = [
        ToolResult(
            tool_call_id="external-write-claim",
            tool_name="external_tool",
            metadata={"workspace_changed": True},
        )
    ]

    verdict = await GoalContractStopHook(goal_spec=goal).evaluate(
        state=state,
        candidate="The external tool says it changed the workspace.",
    )

    assert verdict.action == "block"
    assert verdict.detail["unsatisfied_issue_ids"] == [
        "constraint:workspace_change"
    ]


@pytest.mark.anyio
async def test_workspace_change_goal_accepts_runtime_change_evidence() -> None:
    goal = GoalSpec(
        original_query="Fix the implementation.",
        constraints=[
            GoalConstraint(
                constraint_id="workspace_change",
                constraint_type="workspace_change",
                expected_value=True,
            )
        ],
    )
    state = _state()
    state["tool_results"] = [
        ToolResult(
            tool_call_id="patch-1",
            tool_name="apply_patch",
            metadata=_change_metadata(),
        )
    ]

    verdict = await GoalContractStopHook(goal_spec=goal).evaluate(
        state=state,
        candidate="Implemented and verified.",
    )

    assert verdict.action == "accept"
    assert verdict.code == "goal_contract_satisfied"


@pytest.mark.anyio
async def test_workspace_change_goal_rechecks_final_file_content(
    tmp_path: Path,
) -> None:
    goal = GoalSpec(
        original_query="Fix the implementation.",
        constraints=[
            GoalConstraint(
                constraint_id="workspace_change",
                constraint_type="workspace_change",
                expected_value=True,
            )
        ],
    )
    target = tmp_path / "src" / "example.py"
    target.parent.mkdir()
    before = "before\n"
    after = "after\n"
    target.write_text(before, encoding="utf-8")
    state = _state()
    state["tool_results"] = [
        ToolResult(
            tool_call_id="patch-then-command-revert",
            tool_name="apply_patch",
            metadata=_change_metadata(
                before_sha256=hashlib.sha256(before.encode()).hexdigest(),
                after_sha256=hashlib.sha256(after.encode()).hexdigest(),
            ),
        )
    ]

    verdict = await GoalContractStopHook(
        goal_spec=goal,
        workspace_root=tmp_path,
    ).evaluate(
        state=state,
        candidate="The file was patched and then silently reverted.",
    )

    assert verdict.action == "block"
    assert verdict.detail["unsatisfied_issue_ids"] == [
        "constraint:workspace_change"
    ]

    target.write_text(after, encoding="utf-8")
    accepted = await GoalContractStopHook(
        goal_spec=goal,
        workspace_root=tmp_path,
    ).evaluate(
        state=state,
        candidate="The final file still matches the trusted patch receipt.",
    )

    assert accepted.action == "accept"
    assert accepted.code == "goal_contract_satisfied"


@pytest.mark.anyio
@pytest.mark.parametrize(
    "verification_command",
    [
        "pytest --color=yes",
        "pytest --cov=src",
        "cargo test --offline",
        "mypy --check-untyped-defs src",
        "go test -count=1 ./...",
        "dotnet test -c Release",
        "mvn -o test",
        "mvn test -DskipTests=false",
        "make -Ctests test",
        "make -fnotest.mk test",
    ],
)
async def test_verification_goal_accepts_test_after_latest_workspace_change(
    verification_command: str,
) -> None:
    goal = GoalSpec(
        original_query="Fix and verify the implementation.",
        constraints=[
            GoalConstraint(
                constraint_id="workspace_change",
                constraint_type="workspace_change",
                expected_value=True,
            ),
            GoalConstraint(
                constraint_id="verification_after_change",
                constraint_type="verification_after_change",
                expected_value=True,
            ),
        ],
    )
    state = _state()
    origin = ToolCallOrigin(
        request_id="verify-request",
        toolset_revision="verify-tools",
        exposed_tool_names=("apply_patch", "run_command"),
    )
    state["tool_results"] = [
        ToolResult(
            tool_call_id="patch-1",
            tool_name="apply_patch",
            metadata=_change_metadata(),
        ),
        ToolResult(
            tool_call_id="verify-1",
            tool_name="run_command",
            structured_content={
                "stdout": "1 passed",
                "stderr": "",
                "exit_code": 0,
                "timed_out": False,
                "truncated": False,
                "duration_ms": 10.0,
                "execution_mode": "restricted_sandbox",
                "network_enabled": False,
                "sandbox_error": None,
            },
        ),
    ]
    state["canonical_tool_calls"] = {
        "patch-1": ToolCall(
            tool_call_id="patch-1",
            tool_name="apply_patch",
            arguments={"patch": "*** Begin Patch"},
            origin=origin,
        ),
        "verify-1": ToolCall(
            tool_call_id="verify-1",
            tool_name="run_command",
            arguments={"command": verification_command},
            origin=origin,
        ),
    }

    verdict = await GoalContractStopHook(goal_spec=goal).evaluate(
        state=state,
        candidate="Implemented and verified.",
    )

    assert verdict.action == "accept"
    assert verdict.code == "goal_contract_satisfied"


@pytest.mark.anyio
async def test_write_enabled_command_is_a_workspace_change_with_later_verification() -> None:
    goal = GoalSpec(
        original_query="Fix and verify the implementation.",
        constraints=[
            GoalConstraint(
                constraint_id="workspace_change",
                constraint_type="workspace_change",
                expected_value=True,
            ),
            GoalConstraint(
                constraint_id="verification_after_change",
                constraint_type="verification_after_change",
                expected_value=True,
            ),
        ],
    )
    origin = ToolCallOrigin(
        request_id="command-write-request",
        toolset_revision="command-write-tools",
        exposed_tool_names=("run_command",),
    )
    state = _state()
    state["tool_results"] = [
        ToolResult(
            tool_call_id="write-command",
            tool_name="run_command",
            structured_content={
                "exit_code": 0,
                "timed_out": False,
                "sandbox_error": None,
            },
            metadata=_tree_change_metadata(),
        ),
        ToolResult(
            tool_call_id="verify-command",
            tool_name="run_command",
            structured_content={
                "exit_code": 0,
                "timed_out": False,
                "sandbox_error": None,
            },
        ),
    ]
    state["canonical_tool_calls"] = {
        "write-command": ToolCall(
            tool_call_id="write-command",
            tool_name="run_command",
            arguments={
                "command": "printf 'fixed\\n' > src/example.py",
                "workspace_write": True,
            },
            origin=origin,
        ),
        "verify-command": ToolCall(
            tool_call_id="verify-command",
            tool_name="run_command",
            arguments={
                "command": "uv run pytest -q",
                "workspace_write": False,
            },
            origin=origin,
        ),
    }

    verdict = await GoalContractStopHook(goal_spec=goal).evaluate(
        state=state,
        candidate="Implemented and verified.",
    )

    assert verdict.action == "accept"
    assert verdict.code == "goal_contract_satisfied"


@pytest.mark.anyio
@pytest.mark.parametrize(
    "late_write_metadata",
    [
        pytest.param(_tree_change_metadata(), id="current-attestation"),
        pytest.param(
            {"operation_id": "op_legacy_write"},
            id="legacy-executed-command",
        ),
    ],
)
async def test_workspace_write_after_verification_invalidates_stale_success(
    late_write_metadata: dict[str, object],
) -> None:
    goal = GoalSpec(
        original_query="Fix and verify the implementation.",
        constraints=[
            GoalConstraint(
                constraint_id="workspace_change",
                constraint_type="workspace_change",
                expected_value=True,
            ),
            GoalConstraint(
                constraint_id="verification_after_change",
                constraint_type="verification_after_change",
                expected_value=True,
            ),
        ],
    )
    origin = ToolCallOrigin(
        request_id="stale-verification-request",
        toolset_revision="stale-verification-tools",
        exposed_tool_names=("apply_patch", "run_command"),
    )
    state = _state()
    state["tool_results"] = [
        ToolResult(
            tool_call_id="patch-1",
            tool_name="apply_patch",
            metadata=_change_metadata(),
        ),
        ToolResult(
            tool_call_id="verify-before-write",
            tool_name="run_command",
            structured_content={
                "exit_code": 0,
                "timed_out": False,
                "sandbox_error": None,
            },
        ),
        ToolResult(
            tool_call_id="write-after-verification",
            tool_name="run_command",
            structured_content={
                "exit_code": 0,
                "timed_out": False,
                "sandbox_error": None,
            },
            metadata=late_write_metadata,
        ),
    ]
    state["canonical_tool_calls"] = {
        "verify-before-write": ToolCall(
            tool_call_id="verify-before-write",
            tool_name="run_command",
            arguments={"command": "uv run pytest -q"},
            origin=origin,
        ),
        "write-after-verification": ToolCall(
            tool_call_id="write-after-verification",
            tool_name="run_command",
            arguments={
                "command": "printf 'late change\\n' >> src/example.py",
                "workspace_write": True,
            },
            origin=origin,
        ),
    }

    verdict = await GoalContractStopHook(goal_spec=goal).evaluate(
        state=state,
        candidate="The earlier verification still passes.",
    )

    assert verdict.action == "block"
    assert verdict.detail["unsatisfied_issue_ids"] == [
        "constraint:verification_after_change"
    ]


@pytest.mark.anyio
async def test_noop_write_cannot_claim_an_external_later_change(
    tmp_path: Path,
) -> None:
    goal = GoalSpec(
        original_query="Fix the implementation.",
        constraints=[
            GoalConstraint(
                constraint_id="workspace_change",
                constraint_type="workspace_change",
                expected_value=True,
            )
        ],
    )
    baseline = workspace_tree_sha256(tmp_path)
    assert baseline is not None
    state = _state()
    state["tool_results"] = [
        ToolResult(
            tool_call_id="noop-write",
            tool_name="run_command",
            metadata=_tree_change_metadata(
                before_sha256=baseline,
                after_sha256=baseline,
            ),
        )
    ]
    (tmp_path / "external.py").write_text(
        "changed outside the tool\n",
        encoding="utf-8",
    )

    verdict = await GoalContractStopHook(
        goal_spec=goal,
        workspace_root=tmp_path,
    ).evaluate(
        state=state,
        candidate="A different process changed the workspace later.",
    )

    assert verdict.action == "block"
    assert verdict.detail["unsatisfied_issue_ids"] == [
        "constraint:workspace_change"
    ]


@pytest.mark.anyio
async def test_reverted_writes_cannot_claim_an_external_final_change(
    tmp_path: Path,
) -> None:
    goal = GoalSpec(
        original_query="Fix and verify the implementation.",
        constraints=[
            GoalConstraint(
                constraint_id="workspace_change",
                constraint_type="workspace_change",
                expected_value=True,
            ),
            GoalConstraint(
                constraint_id="verification_after_change",
                constraint_type="verification_after_change",
                expected_value=True,
            ),
        ],
    )
    baseline = workspace_tree_sha256(tmp_path)
    assert baseline is not None
    agent_file = tmp_path / "agent.py"
    agent_file.write_text("changed\n", encoding="utf-8")
    changed = workspace_tree_sha256(tmp_path)
    assert changed is not None and changed != baseline
    agent_file.unlink()
    reverted = workspace_tree_sha256(tmp_path)
    assert reverted == baseline
    (tmp_path / "external.py").write_text(
        "unrelated external change\n",
        encoding="utf-8",
    )

    origin = ToolCallOrigin(
        request_id="reverted-write-request",
        toolset_revision="reverted-write-tools",
        exposed_tool_names=("run_command",),
    )
    state = _state()
    state["tool_results"] = [
        ToolResult(
            tool_call_id="write",
            tool_name="run_command",
            metadata=_tree_change_metadata(
                before_sha256=baseline,
                after_sha256=changed,
            ),
        ),
        ToolResult(
            tool_call_id="revert",
            tool_name="run_command",
            metadata=_tree_change_metadata(
                before_sha256=changed,
                after_sha256=reverted,
            ),
        ),
        ToolResult(
            tool_call_id="verify",
            tool_name="run_command",
            structured_content={
                "exit_code": 0,
                "timed_out": False,
                "sandbox_error": None,
            },
        ),
    ]
    state["canonical_tool_calls"] = {
        "write": ToolCall(
            tool_call_id="write",
            tool_name="run_command",
            arguments={"command": "touch agent.py", "workspace_write": True},
            origin=origin,
        ),
        "revert": ToolCall(
            tool_call_id="revert",
            tool_name="run_command",
            arguments={"command": "rm agent.py", "workspace_write": True},
            origin=origin,
        ),
        "verify": ToolCall(
            tool_call_id="verify",
            tool_name="run_command",
            arguments={"command": "uv run pytest -q"},
            origin=origin,
        ),
    }

    verdict = await GoalContractStopHook(
        goal_spec=goal,
        workspace_root=tmp_path,
    ).evaluate(
        state=state,
        candidate="Implemented and verified.",
    )

    assert verdict.action == "block"
    assert verdict.detail["unsatisfied_issue_ids"] == [
        "constraint:workspace_change"
    ]


@pytest.mark.anyio
async def test_parallel_write_receipts_accept_the_attested_final_state(
    tmp_path: Path,
) -> None:
    goal = GoalSpec(
        original_query="Update both files.",
        constraints=[
            GoalConstraint(
                constraint_id="workspace_change",
                constraint_type="workspace_change",
                expected_value=True,
            )
        ],
    )
    baseline = workspace_tree_sha256(tmp_path)
    assert baseline is not None
    (tmp_path / "first.py").write_text("first\n", encoding="utf-8")
    first_finished = workspace_tree_sha256(tmp_path)
    assert first_finished is not None
    (tmp_path / "second.py").write_text("second\n", encoding="utf-8")
    final = workspace_tree_sha256(tmp_path)
    assert final is not None
    state = _state()
    state["tool_results"] = [
        ToolResult(
            tool_call_id="parallel-final",
            tool_name="apply_patch",
            metadata=_tree_change_metadata(
                before_sha256=baseline,
                after_sha256=final,
            ),
        ),
        ToolResult(
            tool_call_id="parallel-first",
            tool_name="apply_patch",
            metadata=_tree_change_metadata(
                before_sha256=baseline,
                after_sha256=first_finished,
            ),
        ),
    ]

    verdict = await GoalContractStopHook(
        goal_spec=goal,
        workspace_root=tmp_path,
    ).evaluate(
        state=state,
        candidate="Updated both files.",
    )

    assert verdict.action == "accept"


@pytest.mark.anyio
async def test_reverted_patch_sequence_is_not_a_net_workspace_change() -> None:
    goal = GoalSpec(
        original_query="Fix and verify the implementation.",
        constraints=[
            GoalConstraint(
                constraint_id="workspace_change",
                constraint_type="workspace_change",
                expected_value=True,
            ),
            GoalConstraint(
                constraint_id="verification_after_change",
                constraint_type="verification_after_change",
                expected_value=True,
            ),
        ],
    )
    original_hash = "a" * 64
    changed_hash = "b" * 64
    state = _state()
    state["tool_results"] = [
        ToolResult(
            tool_call_id="patch-forward",
            tool_name="apply_patch",
            metadata={
                "workspace_changed": True,
                "file_path": "src/example.py",
                "before_sha256": original_hash,
                "after_sha256": changed_hash,
            },
        ),
        ToolResult(
            tool_call_id="patch-revert",
            tool_name="apply_patch",
            metadata={
                "workspace_changed": True,
                "file_path": "src/example.py",
                "before_sha256": changed_hash,
                "after_sha256": original_hash,
            },
        ),
        ToolResult(
            tool_call_id="verify-revert",
            tool_name="run_command",
            structured_content={
                "exit_code": 0,
                "timed_out": False,
                "sandbox_error": None,
            },
        ),
    ]
    origin = ToolCallOrigin(
        request_id="reverted-patch-request",
        toolset_revision="reverted-patch-tools",
        exposed_tool_names=("apply_patch", "run_command"),
    )
    state["canonical_tool_calls"] = {
        "verify-revert": ToolCall(
            tool_call_id="verify-revert",
            tool_name="run_command",
            arguments={"command": "uv run pytest -q"},
            origin=origin,
        )
    }

    verdict = await GoalContractStopHook(goal_spec=goal).evaluate(
        state=state,
        candidate="Everything is back where it started.",
    )

    assert verdict.action == "block"
    assert verdict.detail["unsatisfied_issue_ids"] == [
        "constraint:workspace_change"
    ]


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("verification_command", "verification_before_change"),
    [
        ("printf 'looks good\\n'", False),
        ("uv run pytest -q", True),
        ("uv run pytest -q || true", False),
        ("uv run ruff check --fix .", False),
        ("uv run ruff check --fix . && uv run pytest -q", False),
        ("uv run pytest --help", False),
        ("touch verification-marker && uv run pytest -q", False),
        ("uv run pytest -q $(touch verification-marker)", False),
        ("./pytest", False),
        ("./mypy src/example.py", False),
        ("./python-evil -m pytest", False),
        ("./uv run pytest -q", False),
        ("./env pytest -q", False),
        ("env PATH=. pytest", False),
        ("CI=1 pytest -q", False),
        ("PYTEST_ADDOPTS=--collect-only pytest -q", False),
        ("env PYTEST_ADDOPTS=--collect-only pytest -q", False),
        ("pytest -o addopts=--collect-only", False),
        ("pytest --override-ini=addopts=--collect-only", False),
        ("pytest -c pytest-collect.ini", False),
        ("pytest --setup-only", False),
        ("go test -list . ./...", False),
        ("make -n test", False),
        ("make -q test", False),
        ("make --touch test", False),
        ("gradle -m test", False),
        ("gradle test --dry-run", False),
        ("gradle test -x test", False),
        ("nox -l", False),
        ("nox --list-sessions", False),
        ("tox -l", False),
        ("tox -a", False),
        ("jest --passWithNoTests", False),
        ("vitest --passWithNoTests", False),
        ("npm test --if-present", False),
        ("cargo test --no-run", False),
        ("mvn test -DskipTests", False),
        ("mvn test -Dmaven.test.skip=true", False),
        ("python -m pytest -q", False),
        ("uv run python -m pytest -q", False),
    ],
)
async def test_verification_goal_rejects_non_verification_or_stale_evidence(
    verification_command: str,
    verification_before_change: bool,
) -> None:
    goal = GoalSpec(
        original_query="Fix and verify the implementation.",
        constraints=[
            GoalConstraint(
                constraint_id="workspace_change",
                constraint_type="workspace_change",
                expected_value=True,
            ),
            GoalConstraint(
                constraint_id="verification_after_change",
                constraint_type="verification_after_change",
                expected_value=True,
            ),
        ],
    )
    state = _state()
    verification = ToolResult(
        tool_call_id="verify-1",
        tool_name="run_command",
        structured_content={
            "stdout": "ok",
            "stderr": "",
            "exit_code": 0,
            "timed_out": False,
            "truncated": False,
            "duration_ms": 10.0,
            "execution_mode": "restricted_sandbox",
            "network_enabled": False,
            "sandbox_error": None,
        },
    )
    change = ToolResult(
        tool_call_id="patch-1",
        tool_name="apply_patch",
        metadata=_change_metadata(),
    )
    state["tool_results"] = (
        [verification, change]
        if verification_before_change
        else [change, verification]
    )
    origin = ToolCallOrigin(
        request_id="verify-rejection-request",
        toolset_revision="verify-rejection-tools",
        exposed_tool_names=("run_command",),
    )
    state["canonical_tool_calls"] = {
        "verify-1": ToolCall(
            tool_call_id="verify-1",
            tool_name="run_command",
            arguments={"command": verification_command},
            origin=origin,
        )
    }

    verdict = await GoalContractStopHook(goal_spec=goal).evaluate(
        state=state,
        candidate="Implemented and verified.",
    )

    assert verdict.action == "block"
    assert verdict.detail["unsatisfied_issue_ids"] == [
        "constraint:verification_after_change"
    ]
    assert "after the latest workspace change" in (verdict.message or "")


@pytest.mark.anyio
async def test_workspace_change_evidence_binds_to_the_declared_constraint_id() -> None:
    goal = GoalSpec(
        original_query="Fix the implementation.",
        constraints=[
            GoalConstraint(
                constraint_id="implementation_changed",
                constraint_type="workspace_change",
                expected_value=True,
            )
        ],
    )
    state = _state()
    state["tool_results"] = [
        ToolResult(
            tool_call_id="patch-1",
            tool_name="apply_patch",
            metadata=_change_metadata(),
        )
    ]

    verdict = await GoalContractStopHook(goal_spec=goal).evaluate(
        state=state,
        candidate="Implemented and verified.",
    )

    assert verdict.action == "accept"
    assert verdict.code == "goal_contract_satisfied"


def test_stop_hook_factory_installs_goal_hook_only_when_explicitly_supplied() -> None:
    definition = AgentRuntimePolicy.test_factory(
        system_prompt="Answer.",
        allowed_tools=[],
    )
    goal = GoalSpec(original_query="Answer")

    ordinary = build_stop_hooks(definition=definition)
    explicit = build_stop_hooks(definition=definition, goal_spec=goal)

    assert all(binding.name != "goal_contract" for binding in ordinary)
    assert [binding.name for binding in explicit] == ["goal_contract"]


@pytest.mark.anyio
async def test_finish_candidate_builder_rejects_missing_final_answer() -> None:
    builder = FinishCandidateBuilder()

    with pytest.raises(
        FinishCandidateBuildError,
        match="model turn is incomplete",
    ):
        await builder.build(
            ModelTurnDraft(action="finish"),
            state=_state(),
        )
