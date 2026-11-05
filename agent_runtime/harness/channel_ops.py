"""Channel authority, receipts and delivery facts in the canonical rollout.

Only local operators configure authority. Provider content never changes grants.
All channel commands borrow the RuntimeStore owner and SQLite write transaction.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from typing import Any

from agent_runtime.harness import facts
from agent_runtime.harness.operation_values import _canonical_json, _json_object
from agent_runtime.harness.runtime_store import RuntimeStore, owner_operation, read_operation


def identity(*parts: str) -> str:
    return hashlib.sha256(_canonical_json(parts).encode()).hexdigest()


@read_operation
def get(store: RuntimeStore, kind: str, key: str) -> dict[str, Any] | None:
    row = store.tx.first("channel_entries", where={"entry_id": identity(kind, key)})
    return None if row is None else dict(row["value"])


@read_operation
def entries(store: RuntimeStore, kind: str) -> tuple[dict[str, Any], ...]:
    return tuple(dict(row["value"]) for row in store.tx.rows("channel_entries", where={"kind": kind}, order_by="rowid"))


@owner_operation
def put(store: RuntimeStore, thread_id: str, kind: str, key: str, value: Mapping[str, Any]) -> None:
    data = _json_object(value, field="channel fact")
    entry_id = identity(kind, key)
    with store.tx.transaction():
        prior = store.tx.first("channel_entries", where={"entry_id": entry_id})
        if prior is not None and prior["thread_id"] != thread_id:
            raise PermissionError("channel identity belongs to another Thread")
        facts.append_fact(
            store,
            thread_id=thread_id,
            turn_id=None,
            record_type="channel_entry_updated",
            producer="channel",
            payload={"entry_id": entry_id, "kind": kind, "key": key, "value": data},
        )


@owner_operation
def configure_source(store: RuntimeStore, thread_id: str, config: Mapping[str, Any]) -> dict[str, Any]:
    source_id = str(config["source_id"])
    with store.tx.transaction():
        old = get(store, "source", source_id)
        value = {**config, "thread_id": thread_id, "revision": 1 if old is None else old["revision"] + 1}
        put(store, thread_id, "source", source_id, value)
        return value


@owner_operation
def bind(
    store: RuntimeStore, *, source_id: str, conversation_id: str, thread_id: str, destination: str, enabled: bool = True
) -> dict[str, Any]:
    """Local explicit audience authorization; never inferred from message links."""
    with store.tx.transaction():
        if get(store, "source", source_id) is None:
            raise ValueError("unknown channel source")
        key = identity(source_id, conversation_id)
        prior = get(store, "binding", key)
        value = dict(
            source_id=source_id,
            conversation_id=conversation_id,
            thread_id=thread_id,
            workspace=store.read_thread(thread_id).workspace,
            destination=destination,
            enabled=enabled,
            revision=1 if prior is None else prior["revision"] + 1,
        )
        put(store, thread_id, "binding", key, value)
        return value


@owner_operation
def set_grant(
    store: RuntimeStore,
    *,
    source_id: str,
    principal: str,
    conversation_id: str,
    thread_id: str,
    operations: list[str],
    role: str,
    enabled: bool = True,
    local_identity: str | None = None,
) -> dict[str, Any]:
    if (
        not principal
        or not role
        or set(operations)
        - {"input", "status", "respond", "cancel", "notify", "evidence", "push", "open_pr", "review_reply"}
    ):
        raise ValueError("invalid channel grant")
    with store.tx.transaction():
        key = identity(source_id, principal, conversation_id)
        prior = get(store, "grant", key)
        value = dict(
            grant_id=key,
            source_id=source_id,
            principal=principal,
            conversation_id=conversation_id,
            thread_id=thread_id,
            workspace=store.read_thread(thread_id).workspace,
            identity=local_identity or identity(source_id, principal),
            role=role,
            operations=operations,
            revision=1 if prior is None else prior["revision"] + 1,
            enabled=enabled,
        )
        put(store, thread_id, "grant", key, value)
        return value


@read_operation
def authorize(
    store: RuntimeStore,
    source_id: str,
    principal: str,
    conversation_id: str,
    operation: str,
    *,
    expected_revision: int | None = None,
) -> dict[str, Any]:
    source = get(store, "source", source_id)
    binding = get(store, "binding", identity(source_id, conversation_id))
    grant = get(store, "grant", identity(source_id, principal, conversation_id))
    if (
        not source
        or not source.get("enabled")
        or not binding
        or not binding["enabled"]
        or not grant
        or not grant["enabled"]
        or operation not in grant["operations"]
        or grant["thread_id"] != binding["thread_id"]
        or grant["workspace"] != binding["workspace"]
        or store.read_thread(binding["thread_id"]).workspace != grant["workspace"]
        or expected_revision is not None
        and grant["revision"] != expected_revision
    ):
        raise PermissionError("channel authority unavailable or revoked")
    return grant


@read_operation
def authorize_command(store: RuntimeStore, command: Any) -> None:
    ci_authority = command.request.get("ci_authority")
    if ci_authority is not None:
        from agent_runtime.channels.ci import authorize as authorize_ci

        sub = get(store, "ci_subscription", ci_authority["subscription_id"])
        if (
            sub is None
            or not sub["enabled"]
            or sub["head_sha"] != ci_authority["head_sha"]
            or sub["thread_id"] != command.thread_id
            or _time(None) >= sub["repair_allowance"]["deadline"]
        ):
            raise PermissionError("CI continuation no longer authorized for this head")
        authorize_ci(store, sub, repair=True)
    authority = command.request.get("channel_authority")
    if authority is None:
        return
    grant = authorize(
        store,
        authority["source_id"],
        authority["principal"],
        authority["conversation_id"],
        authority["operation"],
        expected_revision=authority["revision"],
    )
    binding = get(store, "binding", identity(authority["source_id"], authority["conversation_id"]))
    if (
        grant["thread_id"] != command.thread_id
        or binding is None
        or binding["revision"] != authority["binding_revision"]
    ):
        raise PermissionError("channel command binding changed")


@owner_operation
def receive(store: RuntimeStore, envelope: Any) -> dict[str, Any]:
    from pathlib import Path

    from agent_runtime.harness import host_ops

    with store.tx.transaction():
        grant = authorize(store, envelope.source_id, envelope.principal, envelope.conversation_id, envelope.operation)
        binding = get(store, "binding", identity(envelope.source_id, envelope.conversation_id))
        assert binding is not None
        if binding["destination"] != envelope.destination:
            raise PermissionError("channel audience differs from explicit binding")
        if envelope.principal_kind != "user" and envelope.operation != "evidence":
            raise PermissionError("provider evidence cannot control execution")
        key = identity(envelope.source_id, envelope.event_id)
        native_key = identity(envelope.source_id, getattr(envelope, "native_id", "") or envelope.digest)
        prior = get(store, "inbox", key) or get(store, "native_inbox", native_key)
        if prior is not None:
            if prior["digest"] != envelope.digest or prior["principal"] != envelope.principal:
                raise ValueError("channel event identity conflicts with earlier payload")
            return dict(prior["receipt"])
        thread_id = binding["thread_id"]
        thread = store.read_thread(thread_id)
        authority = {k: grant[k] for k in ("source_id", "principal", "conversation_id", "revision")}
        authority.update(operation=envelope.operation, binding_revision=binding["revision"])
        receipt: dict[str, Any] = {"status": "accepted", "thread_id": thread_id, "command_id": None, "input_id": key}
        if envelope.operation in {"status", "evidence"}:
            if envelope.operation == "status":
                from agent_runtime.harness.delivery import inspect as inspect_delivery

                current = None if thread.head_turn_id is None else store.read_turn(thread.head_turn_id)
                receipt["summary"] = {
                    "thread_id": thread_id,
                    "turn_id": thread.head_turn_id,
                    "status": "queued" if current is None else current.status,
                    "delivery": None if current is None else inspect_delivery(store, current.turn_id)["status"],
                }
                if "notify" in grant["operations"]:
                    enqueue_notification(
                        store,
                        grant["grant_id"],
                        thread.head_turn_id or "queued",
                        {**receipt["summary"], "request": envelope.event_id},
                    )
            if envelope.operation == "evidence":
                put(
                    store,
                    thread_id,
                    "evidence",
                    key,
                    {
                        "event_id": envelope.event_id,
                        "source_id": envelope.source_id,
                        "thread_id": thread_id,
                        "evidence": dict(envelope.evidence),
                        "text": envelope.text,
                    },
                )
        else:
            turn_id = thread.head_turn_id
            queued = [
                c
                for c in host_ops.list_commands(store, thread_id)
                if c.kind == "start"
                and c.status in {"pending", "claimed"}
                and store.tx.first("turns", where={"turn_id": c.turn_id}) is None
            ]
            if queued:
                turn_id = queued[-1].turn_id
            if turn_id is None:
                starts = [c for c in host_ops.list_commands(store, thread_id) if c.kind == "start"]
                if not starts:
                    raise ValueError("channel target has no configured submission")
                turn_id = starts[-1].turn_id
            try:
                turn = store.read_turn(turn_id)
            except KeyError:
                turn = None
            kind = {"input": "input", "respond": "resume", "cancel": "cancel"}[envelope.operation]
            payload: dict[str, Any] = {"channel_authority": authority}
            if envelope.operation == "input":
                if not envelope.text.strip() and not envelope.attachments:
                    raise ValueError("empty channel input")
                text = envelope.text
                if envelope.attachments:
                    text += "\nProvider attachment references (untrusted, not local paths): " + _canonical_json(
                        list(envelope.attachments)
                    )
                payload["text"] = text
                if turn is not None and turn.status in {"completed", "failed", "cancelled"}:
                    source = host_ops.background_submission(store, turn_id)
                    execution = {
                        key: source.request[key] for key in ("configuration", "options") if key in source.request
                    }
                    payload = {**execution, **payload, "previous_turn_id": turn_id, "files": []}
                    kind, turn_id = "start", "turn_channel_" + key
                elif turn is not None and turn.status != "running":
                    raise ValueError("paused target requires explicit response to its current interaction")
            elif envelope.operation == "respond":
                if not envelope.request_id or not envelope.action or turn is None:
                    raise ValueError("response requires a concrete current request")
                interaction = store.tx.first("interactions", where={"request_id": envelope.request_id})
                if interaction is None or interaction["turn_id"] != turn_id or interaction["status"] != "pending":
                    raise ValueError("response does not match a pending target interaction")
                payload.update(
                    request_id=envelope.request_id,
                    action=envelope.action,
                    text=envelope.text or None,
                    authorization_revision=turn.applied_thread_sequence,
                )
            command = host_ops.receive_command(
                store,
                workspace=Path(thread.workspace),
                kind=kind,
                payload=payload,
                command_id="channel_" + key,
                thread_id=thread_id,
                turn_id=turn_id,
            )
            receipt["command_id"] = command.command_id
            receipt["turn_id"] = command.turn_id
        value = {
            "digest": envelope.digest,
            "principal": envelope.principal,
            "receipt": receipt,
            "source_id": envelope.source_id,
            "thread_id": thread_id,
            "envelope": envelope.model_dump(mode="json"),
        }
        put(store, thread_id, "inbox", key, value)
        put(store, thread_id, "native_inbox", native_key, value)
        return receipt


def _time(now: float | None) -> float:
    import math
    import time

    value = time.time() if now is None else now
    if not math.isfinite(value):
        raise ValueError("time must be finite")
    return value


def _notification_authority(store: RuntimeStore, row: Mapping[str, Any]) -> dict[str, Any]:
    source = get(store, "source", row["source_id"])
    if source is None or source["revision"] != row.get("source_revision"):
        raise PermissionError("notification source configuration changed")
    grant = get(store, "grant", row["grant_id"])
    if grant is None:
        raise PermissionError("notification grant missing")
    grant = authorize(
        store,
        grant["source_id"],
        grant["principal"],
        grant["conversation_id"],
        "notify",
        expected_revision=row["grant_revision"],
    )
    binding = get(store, "binding", identity(grant["source_id"], grant["conversation_id"]))
    if binding is None or (binding["revision"], binding["destination"]) != (
        row["binding_revision"],
        row["destination"],
    ):
        raise PermissionError("notification audience changed")
    return grant


@owner_operation
def enqueue_notification(
    store: RuntimeStore,
    grant_id: str,
    turn_id: str,
    summary: Mapping[str, Any],
    *,
    now: float | None = None,
    max_attempts: int = 5,
    deadline_seconds: float = 3600,
    facts_snapshot: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    timestamp = _time(now)
    if type(max_attempts) is not int or not 1 <= max_attempts <= 20 or not 0 < deadline_seconds <= 86400:
        raise ValueError("notification retry bounds invalid")
    frozen = _json_object(summary, field="notification summary")
    if len(_canonical_json(frozen)) > 4000:
        raise ValueError("notification summary exceeds limit")
    with store.tx.transaction():
        grant = get(store, "grant", grant_id)
        if grant is None:
            raise PermissionError("unknown notification grant")
        authorize(store, grant["source_id"], grant["principal"], grant["conversation_id"], "notify")
        binding = get(store, "binding", identity(grant["source_id"], grant["conversation_id"]))
        assert binding is not None
        outbox_id = identity(grant["source_id"], binding["destination"], turn_id, _canonical_json(frozen))
        prior = get(store, "outbox", outbox_id)
        if prior is not None:
            return prior
        fact_source = frozen if facts_snapshot is None else facts_snapshot
        projection = None
        if "thread_id" in fact_source:
            projection = {
                key: fact_source[key]
                for key in ("thread_id", "turn_id", "status", "delivery", "request_id", "remote")
                if key in fact_source
            }
        if projection is not None and "request" not in frozen:
            for old in entries(store, "outbox"):
                if (
                    old["status"] == "pending"
                    and old["source_id"] == grant["source_id"]
                    and old["destination"] == binding["destination"]
                    and old["turn_id"] == turn_id
                    and old.get("fact_snapshot") is not None
                    and "request" not in old["summary"]
                ):
                    old.update(status="abandoned", reason="superseded_by_current_facts")
                    put(store, old["thread_id"], "outbox", old["outbox_id"], old)
        row = dict(
            fact_snapshot=projection,
            outbox_id=outbox_id,
            thread_id=grant["thread_id"],
            turn_id=turn_id,
            grant_id=grant_id,
            source_id=grant["source_id"],
            source_revision=_source_revision(store, grant["source_id"]),
            destination=binding["destination"],
            grant_revision=grant["revision"],
            binding_revision=binding["revision"],
            summary=frozen,
            text="Praxis " + _canonical_json(frozen),
            status="pending",
            attempt=0,
            generation=0,
            owner=None,
            lease_expires_at=0,
            available_at=timestamp,
            deadline=timestamp + deadline_seconds,
            max_attempts=max_attempts,
            remote_id=None,
            reason=None,
        )
        put(store, grant["thread_id"], "outbox", outbox_id, row)
        return row


@owner_operation
def claim_notification(
    store: RuntimeStore, *, owner: str, now: float | None = None, lease_seconds: float = 30, workspace: Any = None
) -> dict[str, Any] | None:
    timestamp = _time(now)
    if not owner or not 0 < lease_seconds <= 300:
        raise ValueError("invalid notification owner/lease")
    with store.tx.transaction():
        for row in entries(store, "outbox"):
            if workspace is not None and store.read_thread(row["thread_id"]).workspace != str(workspace):
                continue
            status = row["status"]
            if status in {"claimed", "sending"} and row["lease_expires_at"] <= timestamp:
                row.update(
                    status="uncertain" if status == "sending" else "pending",
                    owner=None,
                    reason="sender_stopped_after_dispatch" if status == "sending" else "claim_expired",
                )
                put(store, row["thread_id"], "outbox", row["outbox_id"], row)
            if row["status"] != "pending":
                continue
            try:
                _notification_authority(store, row)
                _notification_facts(store, row)
            except PermissionError:
                row.update(status="blocked", reason="authority_revoked")
                put(store, row["thread_id"], "outbox", row["outbox_id"], row)
                continue
            if row["deadline"] <= timestamp or row["attempt"] >= row["max_attempts"]:
                row.update(status="blocked", reason="retry_bound_exhausted")
                put(store, row["thread_id"], "outbox", row["outbox_id"], row)
                continue
            if row["available_at"] > timestamp:
                continue
            row.update(
                status="claimed",
                owner=owner,
                generation=row["generation"] + 1,
                lease_expires_at=timestamp + lease_seconds,
            )
            put(store, row["thread_id"], "outbox", row["outbox_id"], row)
            return row
    return None


def _current_notification(store: RuntimeStore, claim: Mapping[str, Any]) -> dict[str, Any]:
    row = get(store, "outbox", claim["outbox_id"])
    if (
        row is None
        or row["status"] not in {"claimed", "sending"}
        or (row["owner"], row["generation"]) != (claim["owner"], claim["generation"])
    ):
        raise ValueError("notification claim superseded")
    return row


@owner_operation
def begin_send(store: RuntimeStore, claim: Mapping[str, Any], *, now: float | None = None) -> dict[str, Any]:
    timestamp = _time(now)
    with store.tx.transaction():
        row = _current_notification(store, claim)
        if row["status"] != "claimed" or min(row["lease_expires_at"], row["deadline"]) <= timestamp:
            raise ValueError("notification dispatch expired or already started")
        _notification_authority(store, row)
        _notification_facts(store, row)
        row.update(status="sending", attempt=row["attempt"] + 1)
        put(store, row["thread_id"], "outbox", row["outbox_id"], row)
        return row


@owner_operation
def finish_send(
    store: RuntimeStore,
    claim: Mapping[str, Any],
    *,
    status: str,
    remote_id: str | None = None,
    retry_after: float | None = None,
    reason: str | None = None,
    now: float | None = None,
) -> None:
    import math
    import random

    timestamp = _time(now)
    if status not in {"acknowledged", "unsent", "uncertain", "blocked"}:
        raise ValueError("invalid provider delivery outcome")
    if retry_after is not None and (not math.isfinite(retry_after) or retry_after < 0):
        raise ValueError("invalid Retry-After")
    with store.tx.transaction():
        row = _current_notification(store, claim)
        if row["status"] != "sending":
            raise ValueError("notification has no dispatched attempt")
        row.update(status=status, remote_id=remote_id, reason=reason, owner=None)
        if status == "unsent":
            delay = max(retry_after or 0, min(300, 2 ** row["attempt"]) * (0.5 + random.random() * 0.5))
            row.update(status="pending", available_at=timestamp + delay)
            if row["attempt"] >= row["max_attempts"] or row["available_at"] >= row["deadline"]:
                row.update(status="blocked", reason="retry_bound_exhausted")
        put(store, row["thread_id"], "outbox", row["outbox_id"], row)


@owner_operation
def retry_notification(store: RuntimeStore, outbox_id: str, *, now: float | None = None) -> None:
    timestamp = _time(now)
    with store.tx.transaction():
        row = get(store, "outbox", outbox_id)
        if row is None or row["status"] not in {"blocked", "pending"}:
            raise ValueError("unknown delivery must be reconciled, never blindly retried")
        _notification_authority(store, row)
        _notification_facts(store, row)
        if row["deadline"] <= timestamp or row["attempt"] >= row["max_attempts"]:
            raise ValueError("notification retry authorization exhausted; create an explicit new notification")
        row.update(status="pending", available_at=timestamp, reason=None)
        put(store, row["thread_id"], "outbox", outbox_id, row)


@owner_operation
def abandon_notification(store: RuntimeStore, outbox_id: str) -> None:
    with store.tx.transaction():
        row = get(store, "outbox", outbox_id)
        if row is None or row["status"] in {"claimed", "sending", "acknowledged"}:
            raise ValueError("cannot abandon an active or acknowledged send")
        row.update(status="abandoned")
        put(store, row["thread_id"], "outbox", outbox_id, row)


@owner_operation
def reconcile_notification(store: RuntimeStore, outbox_id: str, remote_id: str | None) -> None:
    with store.tx.transaction():
        row = get(store, "outbox", outbox_id)
        if row is None or row["status"] != "uncertain":
            raise ValueError("notification does not require reconciliation")
        _notification_authority(store, row)
        if remote_id is not None:
            row.update(status="acknowledged", remote_id=remote_id, reason="provider_correlation_matched")
            put(store, row["thread_id"], "outbox", outbox_id, row)


@owner_operation
def finish_preflight(
    store: RuntimeStore,
    claim: Mapping[str, Any],
    *,
    status: str,
    retry_after: float | None = None,
    reason: str | None = None,
) -> None:
    with store.tx.transaction():
        row = _current_notification(store, claim)
        if row["status"] != "claimed" or status not in {"unsent", "blocked"}:
            raise ValueError("invalid unsent preflight outcome")
        # Preflight failures consume the same finite attempt/deadline budget.
        row.update(status="sending", attempt=row["attempt"] + 1)
        put(store, row["thread_id"], "outbox", row["outbox_id"], row)
        finish_send(store, claim, status=status, retry_after=retry_after, reason=reason)


def _source_revision(store: RuntimeStore, source_id: str) -> int:
    source = get(store, "source", source_id)
    if source is None:
        raise PermissionError("unknown source")
    return int(source["revision"])


@owner_operation
def reauthorize_notification(store: RuntimeStore, outbox_id: str, grant_id: str) -> None:
    with store.tx.transaction():
        row = get(store, "outbox", outbox_id)
        grant = get(store, "grant", grant_id)
        if row is None or grant is None or row["status"] not in {"blocked", "pending"}:
            raise ValueError("only a confirmed unsent notification can be reauthorized")
        authorize(store, grant["source_id"], grant["principal"], grant["conversation_id"], "notify")
        binding = get(store, "binding", identity(grant["source_id"], grant["conversation_id"]))
        if (
            binding is None
            or grant["thread_id"] != row["thread_id"]
            or grant["source_id"] != row["source_id"]
            or binding["destination"] != row["destination"]
        ):
            raise PermissionError("reauthorization may not change the original audience or workspace")
        row.update(
            grant_id=grant_id,
            grant_revision=grant["revision"],
            binding_revision=binding["revision"],
            source_revision=_source_revision(store, grant["source_id"]),
        )
        put(store, row["thread_id"], "outbox", outbox_id, row)
        retry_notification(store, outbox_id)


def _notification_facts(store: RuntimeStore, row: Mapping[str, Any]) -> None:
    expected = row.get("fact_snapshot")
    if expected is None:
        return
    from agent_runtime.harness.delivery import inspect as inspect_delivery

    thread = store.read_thread(row["thread_id"])
    turn_id = expected.get("turn_id")
    if thread.thread_id != expected["thread_id"] or thread.head_turn_id != turn_id:
        raise PermissionError("notification facts changed since queueing")
    if turn_id is not None:
        turn = store.read_turn(turn_id)
        if turn.status != expected["status"] or inspect_delivery(store, turn_id)["status"] != expected.get("delivery"):
            raise PermissionError("notification facts changed since queueing")
    if expected.get("request_id"):
        request = store.tx.first("interactions", where={"request_id": expected["request_id"]})
        if request is None or request["status"] != "pending":
            raise PermissionError("notification facts changed since queueing")
    for snapshot in expected.get("remote", []):
        sub = get(store, "ci_subscription", snapshot["subscription"])
        if (
            sub is None
            or not sub["enabled"]
            or sub["head_sha"] != snapshot["head_sha"]
            or sub["remote_status"] != snapshot["status"]
        ):
            raise PermissionError("notification facts changed since queueing")
