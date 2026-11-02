"""Read-only budget projections from the host ledger, in explicit units."""

from collections.abc import Mapping
from typing import Any

from agent_runtime.budget import normal_token_remaining, protected_tokens_for_limit
from agent_runtime.harness.execution_context import TurnContext, step_limit_from_binding
from agent_runtime.harness.runtime_store import RuntimeStore


def agent_step_count(store: RuntimeStore, turn_id: str) -> int:
    operations = [
        op for op in store.list_model_operations(turn_id) if op.request_ref.get("purpose") != "context_summary"
    ]
    return max(
        (int(op.request_ref.get("step_snapshot", {}).get("step", index)) for index, op in enumerate(operations, 1)),
        default=0,
    )


def budget_status(store: RuntimeStore, turn: TurnContext) -> dict[str, Any]:
    state = store.read_budget_state(turn.turn_id)
    web = turn.binding_manifest.get("web_limits")
    web_remaining: int | str = "unset"
    if isinstance(web, Mapping) and isinstance(web.get("turn_requests"), int):
        used = store.read_resource_usage(turn.turn_id, "public_web_requests")["used"]
        web_remaining = max(0, web["turn_requests"] - used)
    limit = step_limit_from_binding(turn.binding_manifest)
    return {
        "agent_steps_remaining": "unset" if limit is None else max(0, limit - agent_step_count(store, turn.turn_id)),
        "model_requests_remaining": state.remaining("model_calls") if state.limits.model_calls is not None else "unset",
        "web_requests_remaining": web_remaining,
        "tokens_remaining": state.remaining("tokens") if state.limits.tokens is not None else "unset",
        "cost_remaining_micros": state.remaining("cost_micros") if state.limits.cost_micros is not None else "unset",
    }


def model_visible_budget(
    store: RuntimeStore,
    turn: TurnContext,
    *,
    step: int,
    previous: Mapping[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Facts only. Capacity pressure from compaction is not a configured budget.

    Remaining includes this not-yet-dispatched request. The ledger accounts for
    settled, reserved and uncertain usage; it never invents usage or prices.
    Notify initially, at 50/20/10 percent, and at the last two units. Each
    delivered snapshot identifies when it was observed. Exact admission uses
    the live ledger on every operation, independently of notification cadence.
    """
    status = budget_status(store, turn)
    facts = {key: value for key, value in status.items() if value != "unset" and key != "agent_steps_remaining"}
    step_limit = step_limit_from_binding(turn.binding_manifest)
    if step_limit is not None:
        facts["agent_requests_remaining_including_this"] = max(0, step_limit - step + 1)
    state = store.read_budget_state(turn.turn_id)
    if state.limits.tokens is not None:
        facts["tokens_available_for_ordinary_work"] = normal_token_remaining(state)
        facts["protected_tokens_for_recovery_and_response"] = protected_tokens_for_limit(state.limits.tokens)
    if not facts:
        return None
    web = turn.binding_manifest.get("web_limits", {})
    limits = {
        "tokens_remaining": state.limits.tokens,
        "cost_remaining_micros": state.limits.cost_micros,
        "model_requests_remaining": state.limits.model_calls,
        "web_requests_remaining": web.get("turn_requests") if isinstance(web, Mapping) else None,
        "agent_requests_remaining_including_this": step_limit,
    }

    def band(key: str, value: int, limit: int) -> int:
        boundaries = {limit // 2, limit // 5, limit // 10, 2, 1, 0}
        if key == "tokens_remaining":
            boundaries.add(2 * protected_tokens_for_limit(state.limits.tokens))
        return sum(value <= boundary for boundary in boundaries)

    if previous is not None and previous.get("budget_scope") == state.scope_id:
        changed = False
        for key, limit in limits.items():
            value, old = facts.get(key), previous.get(key)
            if isinstance(limit, int) and isinstance(value, int):
                if not isinstance(old, int) or value > old or band(key, value, limit) != band(key, old, limit):
                    changed = True
        if not changed:
            return dict(previous)
    return {
        "source": "runtime budget ledger before dispatch",
        "budget_scope": state.scope_id,
        "observed_before_agent_request": step,
        **facts,
    }
