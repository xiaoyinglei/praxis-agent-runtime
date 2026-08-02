from __future__ import annotations

from agent_runtime.planning import AgentPlan, PlanStep, PlanTracker


def test_plan_types_have_public_runtime_ownership() -> None:
    assert AgentPlan.__module__ == "agent_runtime.planning"
    assert PlanStep.__module__ == "agent_runtime.planning"


def test_initialize_plan_is_task_based_without_tool_routing() -> None:
    plan, events = PlanTracker().initialize_task(
        task="Summarize the workspace files.",
    )

    assert plan.objective == "Summarize the workspace files."
    assert plan.active_step_id == "step_task"
    assert plan.steps == [
        PlanStep(
            step_id="step_task",
            title="Work on the current task.",
        )
    ]
    assert events[0].event_type == "initialized"


def test_tool_plan_cannot_replace_runtime_owned_objective() -> None:
    tracker = PlanTracker()
    plan, _events = tracker.initialize_task(
        task="Implement the requested API change.",
    )

    updated, events = tracker.replace_from_tool(
        plan,
        steps=[
            PlanStep(
                step_id="step_other",
                title="Ignore the API change and write a status report.",
                status="in_progress",
            )
        ],
        summary="Try a different strategy.",
    )

    assert updated.objective == "Implement the requested API change."
    assert updated.steps[0].title == (
        "Ignore the API change and write a status report."
    )
    assert events[0].event_type == "llm_update"


def test_tool_plan_is_bounded_and_cannot_claim_completion() -> None:
    plan = AgentPlan(
        objective="Implement and verify.",
        active_step_id="step_task",
        steps=[PlanStep(step_id="step_task", title="Work")],
    )
    submitted = [
        PlanStep(
            step_id=f"step_{index}",
            title=f"Claimed complete step {index}",
            status="completed",
        )
        for index in range(3)
    ]

    updated, events = PlanTracker(max_steps=2).replace_from_tool(
        plan,
        steps=submitted,
        summary="Model-authored progress report.",
    )

    assert len(updated.steps) == 2
    assert [step.status for step in updated.steps] == [
        "pending",
        "pending",
    ]
    assert updated.active_step_id == "step_0"
    assert events[0].warnings == [
        "steps_truncated",
        "unverified_completion_ignored",
    ]


def test_runtime_completion_owns_final_plan_status() -> None:
    plan = AgentPlan(
        objective="Implement and verify.",
        active_step_id="step_implement",
        steps=[
            PlanStep(
                step_id="step_implement",
                title="Implement the change.",
                status="in_progress",
            ),
            PlanStep(
                step_id="step_verify",
                title="Verify the change.",
            ),
        ],
    )

    completed, complete_events = PlanTracker().record_completion(plan)
    blocked, blocked_events = PlanTracker().record_completion(
        plan,
        blocked=True,
    )

    assert completed is not None
    assert completed.status == "complete"
    assert [step.status for step in completed.steps] == [
        "completed",
        "completed",
    ]
    assert complete_events[0].event_type == "completed"
    assert blocked is not None
    assert blocked.status == "blocked"
    assert [step.status for step in blocked.steps] == [
        "blocked",
        "blocked",
    ]
    assert blocked_events[0].event_type == "blocked"
