from agent_runtime.harness.context_policy import ContextObservation, ContextPolicy


def test_hard_admission_and_guards_override_soft_target():
    policy = ContextPolicy()
    assert (
        policy.decide(ContextObservation(input_tokens=30, output_tokens=10, max_input_tokens=20)).reason
        == "model_context_window_exceeded"
    )
    assert (
        policy.decide(
            ContextObservation(input_tokens=5, output_tokens=1, max_input_tokens=20, guard_reason="byte_guard")
        ).reason
        == "byte_guard"
    )


def test_soft_pressure_is_optional_and_unknown_exposure_blocks_spending():
    policy = ContextPolicy()
    observation = ContextObservation(input_tokens=90, output_tokens=10, max_input_tokens=200, working_limit=80)
    decision = policy.decide(observation)
    assert decision.action == "compact_optional"
    assert decision.target_input_tokens == 64
    uncertain = policy.decide(
        ContextObservation(
            input_tokens=90, output_tokens=10, max_input_tokens=200, working_limit=80, uncertain_tokens=20
        )
    )
    assert uncertain.action == "admit"


def test_cumulative_pressure_separate_from_provider_capacity():
    decision = ContextPolicy().decide(
        ContextObservation(input_tokens=20, output_tokens=10, max_input_tokens=1000, normal_tokens_remaining=50)
    )
    assert decision.action == "pressure"
    assert decision.reason == "turn_budget_pressure"
