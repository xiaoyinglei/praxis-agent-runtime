"""Evaluation must separate runtime budget exhaustion from provider failure."""
import pytest

from scripts.agent_web_task_trials import classify_outcome


@pytest.mark.parametrize('sdk,provider,expected', [
    ({'stop_reason': 'model_budget_limit_exceeded', 'diagnostics': [
        {'code': 'model_budget_limit_exceeded', 'component': 'runtime'}]}, None, 'task_budget_exhausted'),
    ({'stop_reason': 'model_provider_failed'}, 'rate_limit', 'provider_or_network_failure'),
    ({'stop_reason': 'model_response_incomplete', 'diagnostics': [
        {'code': 'model_response_incomplete', 'component': 'model'}]}, None, 'model_response_failure'),
    ({'stop_reason': None}, None, 'task_failed'),
])
def test_eval_uses_actual_failure_contract(sdk, provider, expected):
    assert classify_outcome(sdk, provider=provider, timed_out=False, browser_unavailable=False,
                            passed=False) == expected
