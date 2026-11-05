"""Durable host input receipts and execution claims over the canonical rollout."""
from __future__ import annotations

import hashlib
import math
import time
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any
from uuid import uuid4

from agent_runtime.budget import ResourceUsage
from agent_runtime.harness import facts, turn_ops
from agent_runtime.harness.operation_values import _canonical_json, _json_object
from agent_runtime.harness.runtime_store import RuntimeStore, owner_operation, read_operation

CONTROL_KINDS = frozenset({"input", "cancel"})
DRIVER_KINDS = frozenset({"start", "resume"})


@dataclass(frozen=True, slots=True)
class HostCommand:
    command_id: str
    thread_id: str
    turn_id: str
    kind: str
    request_hash: str
    request: Mapping[str, Any]
    status: str
    generation: int
    owner: str | None
    lease_expires_at: float | None
    available_at: float
    result: Mapping[str, Any]
    applied_thread_sequence: int


def _snapshot(row: Mapping[str, Any]) -> HostCommand:
    return HostCommand(
        command_id=row["command_id"], thread_id=row["thread_id"], turn_id=row["turn_id"],
        kind=row["kind"], request_hash=row["request_hash"], request=MappingProxyType(dict(row["request"])),
        status=row["status"], generation=row["generation"], owner=row["owner"],
        lease_expires_at=row["lease_expires_at"], available_at=row["available_at"],
        result=MappingProxyType(dict(row["result"])), applied_thread_sequence=row["applied_thread_sequence"],
    )


@read_operation
def read_command(store: RuntimeStore, command_id: str) -> HostCommand:
    row = store.tx.first("host_commands", where={"command_id": command_id})
    if row is None:
        raise KeyError(f"unknown host command: {command_id}")
    return _snapshot(row)


@read_operation
def list_commands(store: RuntimeStore, thread_id: str | None = None) -> tuple[HostCommand, ...]:
    return tuple(_snapshot(row) for row in store.tx.rows(
        "host_commands", where={} if thread_id is None else {"thread_id": thread_id}, order_by="rowid"
    ))


def _matches_relocated_start(store: RuntimeStore, prior: Mapping[str, Any], identity: Mapping[str, Any]) -> bool:
    if prior["kind"] != "start" or identity["kind"] != "start":
        return False
    old_config = prior["request"].get("configuration")
    new_config = identity["payload"].get("configuration")
    if not isinstance(old_config, Mapping) or not isinstance(new_config, Mapping):
        return False
    old_agent = old_config.get("agent")
    new_agent = new_config.get("agent")
    if not isinstance(old_agent, Mapping) or not isinstance(new_agent, Mapping):
        return False
    old_database = old_agent.get("checkpoint_db")
    if (
        not isinstance(old_database, str)
        or Path(old_database) not in store.persistence.database_origins()
        or new_agent.get("checkpoint_db") != str(store.persistence.path.resolve())
    ):
        return False
    equivalent = {
        **identity,
        "payload": {
            **identity["payload"],
            "configuration": {**new_config, "agent": {**new_agent, "checkpoint_db": old_database}},
        },
    }
    request_hash = prior["request_hash"]
    equivalent_hash = hashlib.sha256(_canonical_json(equivalent).encode()).hexdigest()
    return isinstance(request_hash, str) and equivalent_hash == request_hash


@owner_operation
def receive_command(
    store: RuntimeStore, *, workspace: Path, kind: str, payload: Mapping[str, Any],
    command_id: str | None = None, thread_id: str | None = None, turn_id: str | None = None,
) -> HostCommand:
    if kind not in CONTROL_KINDS | DRIVER_KINDS:
        raise ValueError("unsupported host command")
    command_id = uuid4().hex if command_id is None else command_id
    if not isinstance(command_id, str) or not command_id.strip() or len(command_id) > 500:
        raise ValueError("command_id must be nonempty and at most 500 characters")
    workspace = workspace.expanduser().resolve()
    payload = _json_object(payload, field="host command")
    suffix = hashlib.sha256(command_id.encode()).hexdigest()
    thread_id = thread_id or ("thread_bg_" + suffix if kind == "start" else None)
    turn_id = turn_id or ("turn_bg_" + suffix if kind == "start" else None)
    if thread_id is None or turn_id is None:
        raise ValueError("control/resume requires explicit thread_id and turn_id")
    identity = {
        "workspace": str(workspace), "kind": kind, "thread_id": thread_id, "turn_id": turn_id, "payload": payload
    }
    digest = hashlib.sha256(_canonical_json(identity).encode()).hexdigest()
    with store.tx.transaction():
        prior = store.tx.first("host_commands", where={"command_id": command_id})
        if prior is not None:
            if prior["request_hash"] != digest and not _matches_relocated_start(store, prior, identity):
                raise ValueError("host command identity conflicts with an earlier request")
            return _snapshot(prior)
        try:
            thread = store.read_thread(thread_id)
        except KeyError:
            if kind != "start":
                raise PermissionError("command has no owned thread") from None
            thread = turn_ops.create_thread(store, workspace=workspace, thread_id=thread_id)
        if Path(thread.workspace).resolve() != workspace:
            raise PermissionError("host command belongs to another workspace")
        if kind == "start" and store.tx.first("turns", where={"turn_id": turn_id}) is not None:
            raise ValueError("start identity conflicts with an existing Turn")
        if kind != "start":
            try:
                target = store.read_turn(turn_id)
            except KeyError:
                target = None
            queued = store.tx.rows("host_commands", where={"thread_id": thread_id, "kind": "start"})
            if target is None and not any(c["turn_id"] == turn_id for c in queued):
                raise PermissionError("command has no owned target Turn")
            if target is not None and target.thread_id != thread_id:
                raise PermissionError("command targets another Thread's Turn")
        facts.append_fact(
            store, thread_id=thread_id, turn_id=None, record_type="host_command_received", producer="host",
            payload={"command_id": command_id, "turn_id": turn_id, "kind": kind,
                     "request_hash": digest, "request": payload},
        )
    return read_command(store, command_id)


def _clock(now: float | None, lease_seconds: float) -> float:
    value = time.time() if now is None else float(now)
    if not math.isfinite(value) or not math.isfinite(lease_seconds) or lease_seconds <= 0:
        raise ValueError("host time and lease must be finite and lease positive")
    return value


@owner_operation
def claim_next(
    store: RuntimeStore, *, owner: str, control: bool = False, now: float | None = None, lease_seconds: float = 30,
    exclude_commands: frozenset[str] = frozenset(),
) -> HostCommand | None:
    timestamp = _clock(now, lease_seconds)
    if not isinstance(owner, str) or not owner.strip():
        raise ValueError("host owner must be nonempty")
    with store.tx.transaction():
        commands = store.tx.rows("host_commands", where={"status": ("pending", "claimed")}, order_by="rowid")
        kinds = CONTROL_KINDS if control else DRIVER_KINDS
        for row in commands:
            if row["command_id"] in exclude_commands or row["kind"] not in kinds or row["available_at"] > timestamp:
                continue
            if row["status"] == "claimed" and row["lease_expires_at"] > timestamp:
                continue
            if not control and any(
                c["thread_id"] == row["thread_id"] and c["command_id"] != row["command_id"]
                and c["kind"] in DRIVER_KINDS and c["status"] == "claimed"
                for c in commands
            ):
                continue
            from agent_runtime.harness.channel_ops import authorize_command

            try:
                authorize_command(store, _snapshot(row))
            except PermissionError:
                facts.append_fact(
                    store, thread_id=row["thread_id"], turn_id=None,
                    record_type="host_command_settled", producer="host",
                    payload={"command_id": row["command_id"], "status": "rejected",
                             "result": {"reason": "channel_authority_revoked"}, "available_at": 0.0},
                )
                continue
            facts.append_fact(
                store, thread_id=row["thread_id"], turn_id=None, record_type="host_command_claimed", producer="host",
                payload={"command_id": row["command_id"], "owner": owner, "generation": row["generation"] + 1,
                         "lease_expires_at": timestamp + lease_seconds},
            )
            return read_command(store, str(row["command_id"]))
    return None


@read_operation
def claim_current(store: RuntimeStore, claim: HostCommand) -> bool:
    current = read_command(store, claim.command_id)
    return current.status == "claimed" and (current.owner, current.generation) == (claim.owner, claim.generation)


@owner_operation
def renew_claim(
    store: RuntimeStore, claim: HostCommand, *, now: float | None = None, lease_seconds: float = 30,
) -> bool:
    timestamp = _clock(now, lease_seconds)
    with store.tx.transaction():
        if not claim_current(store, claim):
            return False
        facts.append_fact(
            store, thread_id=claim.thread_id, turn_id=None, record_type="host_command_renewed", producer="host",
            payload={"command_id": claim.command_id, "owner": claim.owner, "generation": claim.generation,
                     "lease_expires_at": timestamp + lease_seconds},
        )
    return True


@owner_operation
def settle_command(
    store: RuntimeStore, claim: HostCommand, *, status: str, result: Mapping[str, Any], now: float | None = None,
    retry_after: float = 0,
) -> bool:
    if status not in {"applied", "rejected", "blocked", "pending"}:
        raise ValueError("invalid host receipt status")
    timestamp = _clock(now, 1)
    if not math.isfinite(retry_after) or retry_after < 0:
        raise ValueError("retry_after must be finite and nonnegative")
    value = _json_object(result, field="host receipt result")
    with store.tx.transaction():
        if not claim_current(store, claim):
            return False
        facts.append_fact(
            store, thread_id=claim.thread_id, turn_id=None,
            record_type="host_command_released" if status == "pending" else "host_command_settled", producer="host",
            payload={"command_id": claim.command_id, "status": status, "result": value,
                     "available_at": timestamp + retry_after if status == "pending" else 0.0},
        )
    return True


@owner_operation
def reject_queued_start(store: RuntimeStore, command_id: str) -> bool:
    """Cancel admission before a Turn exists, without minting execution authority."""
    with store.tx.transaction():
        command = read_command(store, command_id)
        if command.kind != "start" or command.status not in {"pending", "claimed"}:
            return False
        if store.tx.first("turns", where={"turn_id": command.turn_id}) is not None:
            raise RuntimeError("an existing Turn must use canonical cancellation")
        facts.append_fact(
            store, thread_id=command.thread_id, turn_id=None, record_type="host_command_settled", producer="host",
            payload={"command_id": command.command_id, "status": "rejected",
                     "result": {"reason": "cancelled_before_start"}, "available_at": 0.0},
        )
    return True


@owner_operation
def release_orphan_claims(store: RuntimeStore, *, owner: str) -> None:
    """Called only after acquiring the database's exclusive host OS lock."""
    with store.tx.transaction():
        for command in list_commands(store):
            if command.status == "claimed" and command.owner != owner:
                settle_command(
                    store, command, status="pending", result={"reason": "previous_host_stopped"},
                )


@read_operation
def background_submission(store: RuntimeStore, turn_id: str) -> HostCommand:
    """Find the configuration owner through canonical child allocations."""
    seen: set[str] = set()
    while turn_id not in seen:
        seen.add(turn_id)
        starts = store.tx.rows("host_commands", where={"kind": "start", "turn_id": turn_id}, order_by="rowid")
        if starts:
            return _snapshot(starts[0])
        allocation = store.tx.first("child_budget_allocations", where={"child_turn_id": turn_id})
        if allocation is None:
            break
        turn_id = str(allocation["parent_turn_id"])
    raise ValueError("Turn has no durable background submission")


@owner_operation
def enqueue_wakeups(store: RuntimeStore) -> None:
    """Reconstruct safe wakeups and child settlement without replaying effects."""
    from agent_runtime.harness import budget_ops
    from agent_runtime.harness.recovery import recovery_plan

    with store.tx.transaction():
        commands = list_commands(store)
        edges = store.tx.rows("child_budget_allocations", where={"status": "active"}, order_by="rowid")
        # Descendants are allocated after their parents; settle leaves first.
        for edge in reversed(edges):
            child_id = str(edge["child_turn_id"])
            try:
                background_submission(store, child_id)
            except ValueError:
                continue
            turn = store.read_turn(child_id)
            state = store.read_budget_state(child_id)
            if turn.status in {"completed", "failed", "cancelled"} and (
                state.reserved + state.uncertain + state.child_reserved == ResourceUsage()
            ):
                budget_ops.settle_child_budget(store, child_turn_id=child_id)
        for turn in store.list_turns():
            if turn.status not in {"running", "paused", "interrupted"}:
                continue
            try:
                start = background_submission(store, turn.turn_id)
            except ValueError:
                continue
            if any(
                c.kind in DRIVER_KINDS and c.status in {"pending", "claimed"}
                and (c.turn_id == turn.turn_id or c.turn_id == start.turn_id and c.status == "claimed")
                for c in commands
            ):
                continue
            decision = recovery_plan(store, turn.turn_id)
            if not decision.automatically_resumable:
                continue
            identity = f"{turn.turn_id}:{decision.kind}:{turn.applied_thread_sequence}"
            receive_command(
                store, workspace=Path(store.read_thread(turn.thread_id).workspace), kind="resume",
                payload={"action": "continue", "automatic": True}, thread_id=turn.thread_id,
                turn_id=turn.turn_id, command_id="wake_" + hashlib.sha256(identity.encode()).hexdigest(),
            )
