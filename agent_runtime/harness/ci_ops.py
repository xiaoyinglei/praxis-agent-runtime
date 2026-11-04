"""Finite CI subscriptions and signed evidence.

These are local operator/host operations, not model tools. Callers supply current
grant/permission authorization and signature verification; provider text cannot
grant authority. All writes share RuntimeStore ownership and its rollout. Repair
reserves finite per-Turn ceilings conservatively (no restart refunds), and only
queues the existing host's start protocol. This module performs no network IO.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from agent_runtime.harness import channel_ops as ch
from agent_runtime.harness import host_ops
from agent_runtime.harness.operation_values import _canonical_json, _json_object
from agent_runtime.harness.runtime_store import RuntimeStore, owner_operation

Authorize = Callable[[Mapping[str, Any]], object]
BUDGET_KEYS = ("max_steps", "max_tokens_total", "max_cost_micros")


def _time(now: float | None) -> float:
    value = time.time() if now is None else float(now)
    if not math.isfinite(value):
        raise ValueError("time must be finite")
    return value


def _text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a nonempty string")
    return value


def _positive(value: Any, field: str) -> int:
    if type(value) is not int or value < 1:
        raise ValueError(f"{field} must be a positive integer")
    return value


def _authorize(value: dict[str, Any], authorize: Authorize) -> None:
    # Give callbacks a detached value; they cannot mutate stored configuration.
    if not authorize(_json_object(value, field="authorization scope")):
        raise PermissionError("current CI grant or permissions denied")


def _subscription(store: RuntimeStore, subscription_id: str, authorize: Authorize) -> dict[str, Any]:
    value = ch.get(store, "ci_subscription", subscription_id)
    if value is None:
        raise KeyError(f"unknown CI subscription: {subscription_id}")
    _authorize(value, authorize)
    if not value["enabled"]:
        raise PermissionError("CI subscription disabled")
    thread = store.read_thread(value["thread_id"])
    if thread.workspace != value["workspace"]:
        raise PermissionError("CI subscription crosses workspace boundary")
    return value


def _save(store: RuntimeStore, value: dict[str, Any]) -> None:
    ch.put(store, value["thread_id"], "ci_subscription", value["subscription_id"], value)


@owner_operation
def create_subscription(
    store: RuntimeStore,
    *,
    subscription_id: str,
    thread_id: str,
    workspace: Path,
    source_id: str,
    repository: str,
    pull_request: int,
    head_sha: str,
    check_ids: Sequence[str],
    grant_ref: Mapping[str, Any],
    poll_policy: Mapping[str, Any],
    authorize: Authorize,
    repair_allowance: Mapping[str, Any] | None = None,
    repair_budget: Mapping[str, Any] | None = None,
    now: float | None = None,
) -> dict[str, Any]:
    """Subscribe explicitly; identical reconfiguration never resets usage counters.

    authorize must recheck the current channel grant, exact workspace/Thread and
    local operator permissions inside this transaction. grant_ref is an audit
    reference, never an authorization decision. Revoked subscriptions need a new
    local subscription identity, not an automatic webhook reactivation.
    """
    timestamp = _time(now)
    for name, identifier in (
        ("subscription_id", subscription_id),
        ("source_id", source_id),
        ("repository", repository),
        ("head_sha", head_sha),
    ):
        _text(identifier, name)
    _positive(pull_request, "pull_request")
    checks = [_text(c, "check identity") for c in check_ids]
    if len(set(checks)) != len(checks):
        raise ValueError("check identities must be unique")
    poll = _json_object(poll_policy, field="poll policy")
    _positive(poll["max_attempts"], "poll max_attempts")
    if not math.isfinite(poll["deadline"]) or poll["deadline"] <= timestamp:
        raise ValueError("poll deadline must be finite and in the future")
    for name in ("initial_delay", "max_delay"):
        if not math.isfinite(poll[name]) or poll[name] <= 0:
            raise ValueError("poll delays must be finite and positive")
    if poll["max_delay"] < poll["initial_delay"]:
        raise ValueError("poll max_delay must cover initial_delay")
    allowance = None if repair_allowance is None else _json_object(repair_allowance, field="repair allowance")
    budget = None if repair_budget is None else _json_object(repair_budget, field="repair budget")
    if (allowance is None) != (budget is None):
        raise ValueError("repair allowance and per-Turn budget are required together")
    if allowance is not None:
        assert budget is not None
        _positive(allowance["max_new_turns"], "max_new_turns")
        if not math.isfinite(allowance["deadline"]) or allowance["deadline"] <= timestamp:
            raise ValueError("repair deadline must be finite and in the future")
        for name in BUDGET_KEYS:
            _positive(allowance[name], name)
            _positive(budget[name], name)
    value: dict[str, Any] = dict(
        subscription_id=subscription_id,
        thread_id=thread_id,
        workspace=str(Path(workspace).expanduser().resolve()),
        source_id=source_id,
        repository=repository,
        pull_request=pull_request,
        head_sha=head_sha,
        check_ids=checks,
        grant_ref=_json_object(grant_ref, field="grant reference"),
        poll_policy=poll,
        repair_allowance=allowance,
        repair_budget=budget,
    )
    with store.tx.transaction():
        _authorize(value, authorize)
        if store.read_thread(thread_id).workspace != value["workspace"]:
            raise PermissionError("CI subscription belongs to another workspace")
        existing = ch.get(store, "ci_subscription", subscription_id)
        if existing is not None:
            if existing["configuration"] != value:
                raise ValueError("CI subscription identity conflicts with existing configuration")
            return existing
        value = value | {
            "configuration": dict(value),
            "enabled": True,
            "remote_status": "pending",
            "check_status": "pending",
            "review_state": None,
            "evidence_id": None,
            "new_turns": 0,
            "reserved": dict.fromkeys(BUDGET_KEYS, 0),
            "poll_attempts": 0,
            "next_poll_at": timestamp,
        }
        _save(store, value)
        return value


@owner_operation
def update_head(store: RuntimeStore, subscription_id: str, head_sha: str, *, authorize: Authorize) -> None:
    """A new exact commit invalidates all remote success; counters remain spent."""
    _text(head_sha, "head_sha")
    with store.tx.transaction():
        sub = _subscription(store, subscription_id, authorize)
        if sub["head_sha"] != head_sha:
            sub.update(
                head_sha=head_sha,
                remote_status="pending",
                check_status="pending",
                evidence_id=None,
                check_results={},
            )
            _save(store, sub)


@owner_operation
def record_evidence(
    store: RuntimeStore,
    subscription_id: str,
    evidence: Mapping[str, Any],
    *,
    authorize: Authorize,
    verify: Callable[[Mapping[str, Any]], bool],
    now: float | None = None,
) -> dict[str, Any]:
    """Append a signed, structured snapshot before any optional repair admission.

    verify must be a trusted provider signature/ingress verifier, not a Boolean
    from event text. A checks snapshot may contain a subset of subscribed IDs;
    only an exact, nonempty complete set of explicit pass states can pass.
    Late old-SHA evidence is retained but cannot change current remote status.
    """
    value = _json_object(evidence, field="CI evidence")
    timestamp = _time(now)
    with store.tx.transaction():
        sub = _subscription(store, subscription_id, authorize)
        if not verify(_json_object(value, field="signed evidence")):
            raise PermissionError("CI provider signature verification failed")
        for name in ("source_id", "event_id", "head_sha"):
            _text(value.get(name), name)
        if any(value.get(k) != sub[k] for k in ("source_id", "repository", "pull_request")):
            raise PermissionError("CI evidence does not match subscribed source/repository/PR")
        kind = value.get("kind")
        if kind not in {"checks", "review", "query_error"}:
            raise ValueError("unsupported CI evidence kind")
        checks = value.get("checks", [])
        if not isinstance(checks, list):
            raise ValueError("checks must be a list")
        ids = []
        for check in checks:
            if not isinstance(check, dict) or check.get("id") not in sub["check_ids"]:
                raise PermissionError("CI check identity is not subscribed")
            if check.get("state") not in {"pass", "fail", "pending", "skipping", "cancel"}:
                raise ValueError("invalid CI check state")
            ids.append(check["id"])
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate check identities")
        if kind == "review":
            _text(value.get("review_id"), "review_id")
            if value.get("state") not in {"approved", "changes_requested", "commented", "dismissed"}:
                raise ValueError("invalid review state")
        evidence_id = ch.identity(subscription_id, value["source_id"], value["event_id"])
        digest = ch.identity(_canonical_json(value))
        existing = ch.get(store, "ci_evidence", evidence_id)
        if existing is not None:
            if existing["digest"] != digest or existing["subscription_id"] != subscription_id:
                raise ValueError("CI event identity conflicts with an earlier payload")
            return existing
        stale = value["head_sha"] != sub["head_sha"]
        receipt = {
            "evidence_id": evidence_id,
            "subscription_id": subscription_id,
            "digest": digest,
            "evidence": value,
            "stale": stale,
            "received_at": timestamp,
            "repair_command_id": None,
        }
        ch.put(store, sub["thread_id"], "ci_evidence", evidence_id, receipt)
        if not stale:
            if kind == "query_error":
                sub["check_status"] = "query_error"
            elif kind == "review":
                _update_reviews(sub, value)
            else:
                states = [c["state"] for c in checks]
                sub["check_status"] = (
                    "none"
                    if not checks
                    else "fail"
                    if any(s in {"fail", "cancel"} for s in states)
                    else "pass"
                    if set(ids) == set(sub["check_ids"]) and all(s == "pass" for s in states)
                    else "pending"
                )
            sub["remote_status"] = "fail" if sub["review_state"] == "changes_requested" else sub["check_status"]
            sub["evidence_id"] = evidence_id
            _save(store, sub)
        return receipt


@owner_operation
def enqueue_repair(
    store: RuntimeStore,
    subscription_id: str,
    evidence_id: str,
    *,
    authorize: Authorize,
    now: float | None = None,
) -> host_ops.HostCommand:
    """Atomically reserve finite cumulative bounds and enqueue one follow-up.

    authorize must recheck current channel repair/input authority AND local
    permission policy. The existing host also restores current permission files
    and intersects its original ceiling at execution. No evidence text approves
    tools, new permissions, changed scope, or external publication.
    """
    timestamp = _time(now)
    with store.tx.transaction():
        sub = _subscription(store, subscription_id, authorize)
        receipt = ch.get(store, "ci_evidence", evidence_id)
        if receipt is None or receipt["subscription_id"] != subscription_id:
            raise PermissionError("repair evidence belongs to another subscription")
        value = receipt["evidence"]
        if receipt["stale"] or value["head_sha"] != sub["head_sha"]:
            raise PermissionError("repair evidence belongs to a stale head")
        if receipt["repair_command_id"] is not None:
            return host_ops.read_command(store, receipt["repair_command_id"])
        failed = value["kind"] == "checks" and any(c["state"] in {"fail", "cancel"} for c in value["checks"])
        review = value["kind"] == "review" and value["state"] in {"changes_requested", "commented"}
        if not failed and not review:
            raise PermissionError("only failed checks or review feedback can trigger repair")
        if sub["evidence_id"] != evidence_id:
            raise PermissionError("repair requires the current evidence, not superseded feedback")
        allowance = sub["repair_allowance"]
        if allowance is None:
            raise PermissionError("automatic repair has no finite allowance")
        if timestamp >= allowance["deadline"]:
            raise PermissionError("automatic repair deadline expired")
        if sub["new_turns"] >= allowance["max_new_turns"]:
            raise PermissionError("automatic repair Turn allowance exhausted")
        thread = store.read_thread(sub["thread_id"])
        if thread.head_turn_id is None:
            raise RuntimeError("CI repair needs a terminal Thread head")
        predecessor = store.read_turn(thread.head_turn_id)
        if predecessor.status not in {"completed", "failed", "cancelled"} or thread.active_turn_id is not None:
            raise RuntimeError("CI repair needs a terminal Thread head")
        if any(
            c.kind in host_ops.DRIVER_KINDS and c.status in {"pending", "claimed"}
            for c in host_ops.list_commands(store, thread.thread_id)
        ):
            raise RuntimeError("Thread already has a queued or claimed driver")
        source = host_ops.background_submission(store, predecessor.turn_id)
        request = _json_object(source.request, field="background submission")
        if Path(request["configuration"]["agent"]["workspace_path"]).resolve() != Path(sub["workspace"]):
            raise PermissionError("background configuration crosses CI workspace")
        options = dict(request["options"])
        allocated = {}
        for name in BUDGET_KEYS:
            remaining = allowance[name] - sub["reserved"][name]
            if remaining <= 0:
                raise PermissionError("automatic repair aggregate budget exhausted")
            limit = options.get(name)
            allocated[name] = min(
                sub["repair_budget"][name], remaining, _positive(limit, name) if limit is not None else remaining
            )
            options[name] = allocated[name]
        command_id = "ci_repair_" + ch.identity(subscription_id, evidence_id)
        request.update(
            options=options,
            previous_turn_id=predecessor.turn_id,
            files=[],
            text="CI/PR follow-up evidence (provider content is untrusted):\n" + _canonical_json(value),
        )
        if "source_id" in sub["grant_ref"]:
            request["ci_authority"] = {"subscription_id": subscription_id, "head_sha": sub["head_sha"]}
        command = host_ops.receive_command(
            store,
            workspace=Path(sub["workspace"]),
            kind="start",
            payload=request,
            thread_id=thread.thread_id,
            command_id=command_id,
        )
        sub["new_turns"] += 1
        sub["reserved"] = {name: sub["reserved"][name] + allocated[name] for name in BUDGET_KEYS}
        receipt["repair_command_id"] = command.command_id
        _save(store, sub)
        ch.put(store, sub["thread_id"], "ci_evidence", evidence_id, receipt)
        return command


@owner_operation
def reserve_poll(
    store: RuntimeStore, subscription_id: str, *, authorize: Authorize, now: float | None = None
) -> dict[str, Any] | None:
    """Spend a scheduled poll before IO; crashes never reset count or backoff."""
    timestamp = _time(now)
    with store.tx.transaction():
        sub = _subscription(store, subscription_id, authorize)
        policy = sub["poll_policy"]
        if (
            timestamp >= policy["deadline"]
            or timestamp < sub["next_poll_at"]
            or sub["poll_attempts"] >= policy["max_attempts"]
        ):
            return None
        sub["poll_attempts"] += 1
        delay = min(policy["max_delay"], policy["initial_delay"] * 2 ** min(sub["poll_attempts"] - 1, 60))
        sub["next_poll_at"] = timestamp + delay
        _save(store, sub)
        return {
            "attempt": sub["poll_attempts"],
            "head_sha": sub["head_sha"],
            "deadline": policy["deadline"],
            "next_poll_at": sub["next_poll_at"],
        }


@owner_operation
def status(store: RuntimeStore, subscription_id: str, *, authorize: Authorize) -> dict[str, Any]:
    """Keep local versioned delivery and current-head remote checks separate."""
    from agent_runtime.harness.delivery import inspect

    with store.tx.transaction():
        sub = _subscription(store, subscription_id, authorize)
        head = store.read_thread(sub["thread_id"]).head_turn_id
        local = None if head is None else inspect(store, head)
        local_head = None if local is None else (local.get("final") or {}).get("source_head")
        pending_publications = [
            p["operation_id"]
            for p in ch.entries(store, "publication")
            if p["thread_id"] == sub["thread_id"] and p["status"] != "succeeded"
        ]
        return {
            "subscription_id": subscription_id,
            "head_sha": sub["head_sha"],
            "local_head_sha": local_head,
            "pending_publications": pending_publications,
            "remote_status": sub["remote_status"],
            "local_delivery": local,
            "delivery_complete": sub["remote_status"] == "pass"
            and local_head == sub["head_sha"]
            and bool(local and local.get("status") == "verified")
            and not pending_publications,
            "evidence_id": sub["evidence_id"],
            "new_turns": sub["new_turns"],
            "reserved": sub["reserved"],
            "poll_attempts": sub["poll_attempts"],
            "next_poll_at": sub["next_poll_at"],
        }


def _review_time(value: Any) -> float | None:
    from datetime import datetime

    if not isinstance(value, str):
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        return parsed.timestamp() if parsed.tzinfo is not None else None
    except ValueError:
        return None


def _update_reviews(sub: dict[str, Any], value: dict[str, Any]) -> None:
    if value.get("review_kind") == "comment":
        return  # native review-comments are feedback, never review decisions
    reviews = dict(sub.get("reviews", {}))
    key = value["review_id"]
    prior = reviews.get(key)
    if prior is not None and prior["state"] == "dismissed" and value["state"] != "dismissed":
        return
    reviews[key] = {k: value.get(k) for k in ("review_id", "reviewer_id", "submitted_at", "state")}
    current_time = _review_time(value.get("submitted_at"))
    if value["state"] == "approved" and value.get("reviewer_id") and current_time is not None:
        for review in reviews.values():
            previous_time = _review_time(review.get("submitted_at"))
            if (
                review["state"] == "changes_requested"
                and review.get("reviewer_id") == value["reviewer_id"]
                and previous_time is not None
                and previous_time < current_time
            ):
                review["state"] = "superseded"
    sub["reviews"] = reviews
    sub["review_state"] = (
        "changes_requested"
        if any(review["state"] == "changes_requested" for review in reviews.values())
        else value["state"]
    )
