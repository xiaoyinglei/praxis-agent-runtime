"""CLI/SDK channel operations. HTTP only receives signed content, never grants."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any
from uuid import uuid4

from agent_runtime.channels.providers import ProviderClient, Source, parse_event
from agent_runtime.harness import RuntimeStore
from agent_runtime.harness import channel_ops as ch


class Channels:
    def __init__(self, *, database: Path, workspace: Path, autostart: bool = False) -> None:
        self.autostart = autostart
        self.database = database.expanduser().resolve()
        self.workspace = workspace.expanduser().resolve()

    def _thread(self, store: RuntimeStore, thread_id: str) -> None:
        if Path(store.read_thread(thread_id).workspace).resolve() != self.workspace:
            raise PermissionError("channel object belongs to another workspace")

    def _source(self, store: RuntimeStore, source_id: str) -> Source:
        row = ch.get(store, "source", source_id)
        if row is None:
            raise ValueError("unknown channel source")
        self._thread(store, row["thread_id"])
        return Source.model_validate({key: value for key, value in row.items() if key in Source.model_fields})

    async def configure(self, thread_id: str, source: Source) -> dict[str, Any]:
        async with await RuntimeStore.open(self.database) as store:

            def apply() -> dict[str, Any]:
                with store.tx.transaction():
                    self._thread(store, thread_id)
                    return ch.configure_source(store, thread_id, source.model_dump(mode="json"))

            return await store.run_async(apply)

    async def bind(self, **values: Any) -> dict[str, Any]:
        async with await RuntimeStore.open(self.database) as store:

            def apply() -> dict[str, Any]:
                with store.tx.transaction():
                    self._thread(store, values["thread_id"])
                    self._source(store, values["source_id"])
                    return ch.bind(store, **values)

            return await store.run_async(apply)

    async def grant(self, **values: Any) -> dict[str, Any]:
        async with await RuntimeStore.open(self.database) as store:

            def apply() -> dict[str, Any]:
                with store.tx.transaction():
                    self._thread(store, values["thread_id"])
                    self._source(store, values["source_id"])
                    return ch.set_grant(store, **values)

            return await store.run_async(apply)

    async def receive(self, source_id: str, headers: Mapping[str, str], raw: bytes) -> dict[str, Any]:
        # Parsing and authority share the same source revision and write transaction.
        started = time.monotonic()
        async with await RuntimeStore.open(self.database) as store:

            def apply() -> dict[str, Any]:
                with store.tx.transaction():
                    source = self._source(store, source_id)
                    deadline = started + source.timeout_seconds
                    if time.monotonic() >= deadline:
                        raise TimeoutError("ingress deadline exceeded before admission")
                    event = parse_event(source, headers, raw)
                    if event is None:
                        return {"status": "ignored"}
                    if isinstance(event, dict):
                        return event  # authenticated provider challenge only
                    if own_echo(store, source, event):
                        return {"status": "ignored_echo"}
                    receipt = ch.receive(store, event)
                    if event.operation == "evidence":
                        from agent_runtime.channels.ci import consume

                        inbox = ch.get(store, "inbox", receipt["input_id"])
                        assert inbox is not None
                        if not inbox.get("ci_processed"):
                            consume(store, event)
                            inbox["ci_processed"] = True
                            ch.put(store, receipt["thread_id"], "inbox", receipt["input_id"], inbox)
                    if time.monotonic() >= deadline:
                        raise TimeoutError("ingress deadline exceeded")
                    return receipt

            receipt = await store.run_async(apply)
        if self.autostart and receipt.get("status") == "accepted":
            from agent_runtime.background import ensure_host

            await asyncio.to_thread(ensure_host, self.database)
        return receipt

    async def subscribe_ci(self, *, principal: str, conversation_id: str, **values: Any) -> dict[str, Any]:
        from agent_runtime.channels.ci import authorize
        from agent_runtime.harness import ci_ops

        async with await RuntimeStore.open(self.database) as store:

            def apply() -> dict[str, Any]:
                with store.tx.transaction():
                    self._thread(store, values["thread_id"])
                    source = self._source(store, values["source_id"])
                    if source.kind != "github" or source.repository != values["repository"]:
                        raise PermissionError("CI source/repository mismatch")
                    grant = ch.authorize(store, values["source_id"], principal, conversation_id, "status")
                    binding = ch.get(store, "binding", ch.identity(values["source_id"], conversation_id))
                    if (
                        binding is None
                        or binding["destination"] != f"github:{source.repository}:{values['pull_request']}"
                    ):
                        raise PermissionError("CI PR differs from bound discussion")
                    ref = {k: grant[k] for k in ("source_id", "principal", "conversation_id", "revision")}
                    ref["binding_revision"] = binding["revision"]
                    ref["source_revision"] = ch._source_revision(store, values["source_id"])
                    return ci_ops.create_subscription(
                        store,
                        workspace=self.workspace,
                        grant_ref=ref,
                        authorize=lambda sub: authorize(store, sub, repair=values.get("repair_allowance") is not None),
                        **values,
                    )

            result = await store.run_async(apply)
        if self.autostart:
            from agent_runtime.background import ensure_host

            await asyncio.to_thread(ensure_host, self.database)
        return result

    async def ci_status(self, subscription_id: str) -> dict[str, Any]:
        from agent_runtime.channels.ci import authorize
        from agent_runtime.harness import ci_ops

        async with await RuntimeStore.open(self.database) as store:

            def read() -> dict[str, Any]:
                sub = ch.get(store, "ci_subscription", subscription_id)
                if sub is None:
                    raise KeyError("unknown CI subscription")
                self._thread(store, sub["thread_id"])
                return ci_ops.status(store, subscription_id, authorize=lambda value: authorize(store, value))

            return await store.run_async(read)

    async def ci_head(self, subscription_id: str, head_sha: str) -> None:
        from agent_runtime.channels.ci import authorize
        from agent_runtime.harness import ci_ops

        async with await RuntimeStore.open(self.database) as store:

            def apply() -> None:
                sub = ch.get(store, "ci_subscription", subscription_id)
                if sub is None:
                    raise KeyError("unknown CI subscription")
                self._thread(store, sub["thread_id"])
                ci_ops.update_head(store, subscription_id, head_sha, authorize=lambda value: authorize(store, value))

            await store.run_async(apply)

    async def disable_ci(self, subscription_id: str) -> None:
        async with await RuntimeStore.open(self.database) as store:

            def apply() -> None:
                with store.tx.transaction():
                    sub = ch.get(store, "ci_subscription", subscription_id)
                    if sub is None:
                        raise KeyError("unknown CI subscription")
                    self._thread(store, sub["thread_id"])
                    sub["enabled"] = False
                    ch.put(store, sub["thread_id"], "ci_subscription", subscription_id, sub)

            await store.run_async(apply)

    async def prepare_publication(self, **values: Any) -> dict[str, Any]:
        from agent_runtime.channels.publication import prepare

        return await prepare(self, **values)

    async def execute_publication(self, operation_id: str, **values: Any) -> dict[str, Any]:
        from agent_runtime.channels.publication import execute

        return await execute(self, operation_id, **values)

    async def inspect(self) -> dict[str, Any]:
        async with await RuntimeStore.open(self.database, read_only=True) as store:

            def read() -> dict[str, Any]:
                def owned(kind: str) -> list[dict[str, Any]]:
                    return [
                        r
                        for r in ch.entries(store, kind)
                        if store.read_thread(r["thread_id"]).workspace == str(self.workspace)
                    ]

                sources = owned("source")
                import os

                for source in sources:
                    refs = [v for k, v in source.items() if k.endswith("_env") and v]
                    source["credential_health"] = "configured" if all(os.environ.get(r) for r in refs) else "missing"
                    outbound_ref = (
                        source.get("outbound_token_env") if source["kind"] == "github" else source.get("app_secret_env")
                    )
                    source["outbound_health"] = (
                        "configured"
                        if outbound_ref
                        and os.environ.get(outbound_ref)
                        and (source["kind"] != "github" or source.get("outbound_principal"))
                        else "needs_configuration"
                    )
                return {
                    kind: owned(kind)
                    for kind in ("binding", "grant", "outbox", "ci_subscription", "publication", "channel_diagnostic")
                } | {"source": sources}

            return await store.run_async(store.read_snapshot, read)

    async def collect(self) -> None:
        async with await RuntimeStore.open(self.database) as store:
            await store.run_async(collect_notifications, store, self.workspace)

    async def deliver(self, *, client: ProviderClient | None = None) -> bool:
        client = client or ProviderClient()
        async with await RuntimeStore.open(self.database) as store:
            return await deliver_one(store, workspace=self.workspace, client=client)

    async def outbox_action(
        self, outbox_id: str, action: str, *, client: ProviderClient | None = None, grant_id: str | None = None
    ) -> None:
        async with await RuntimeStore.open(self.database) as store:

            def read() -> dict[str, Any]:
                row = ch.get(store, "outbox", outbox_id)
                if row is None:
                    raise KeyError("unknown outbox identity")
                self._thread(store, row["thread_id"])
                return row

            row = await store.run_async(read)
            if action == "reauthorize":
                if grant_id is None:
                    raise ValueError("reauthorize requires a concrete grant_id")
                await store.run_async(ch.reauthorize_notification, store, outbox_id, grant_id)
            elif action == "retry":
                await store.run_async(ch.retry_notification, store, outbox_id)
            elif action == "abandon":
                await store.run_async(ch.abandon_notification, store, outbox_id)
            elif action == "reconcile":
                source = await store.run_async(self._source, store, row["source_id"])

                async def authorize() -> None:
                    await store.run_async(ch._notification_authority, store, row)

                ack = await (client or ProviderClient()).reconcile(
                    source, row["destination"], outbox_id, authorize=authorize
                )
                await store.run_async(
                    ch.reconcile_notification, store, outbox_id, None if ack is None else ack.remote_id
                )
            else:
                raise ValueError("expected retry, abandon or reconcile")


def collect_notifications(store: RuntimeStore, workspace: Path | None = None) -> None:
    """Stable summaries only: never broadcast answers, raw logs, files or arguments."""
    from agent_runtime.harness.delivery import inspect as delivery
    from agent_runtime.harness.recovery import recovery_plan

    with store.tx.transaction():
        for grant in ch.entries(store, "grant"):
            if not grant["enabled"] or "notify" not in grant["operations"]:
                continue
            thread = store.read_thread(grant["thread_id"])
            if workspace is not None and thread.workspace != str(workspace):
                continue
            if thread.head_turn_id is None:
                continue
            turn = store.read_turn(thread.head_turn_id)
            if turn.status not in {"paused", "interrupted", "completed", "failed", "cancelled"}:
                continue
            try:
                ch.authorize(store, grant["source_id"], grant["principal"], grant["conversation_id"], "notify")
            except PermissionError:
                continue
            evidence = delivery(store, turn.turn_id)
            summary: dict[str, Any] = {
                "thread_id": thread.thread_id,
                "turn_id": turn.turn_id,
                "status": turn.status,
                "delivery": evidence["status"],
            }
            subscriptions = [
                sub
                for sub in ch.entries(store, "ci_subscription")
                if sub["enabled"] and sub["thread_id"] == thread.thread_id
            ]
            if subscriptions:
                summary["remote"] = [
                    {
                        "subscription": sub["subscription_id"],
                        "head_sha": sub["head_sha"],
                        "status": sub["remote_status"],
                    }
                    for sub in subscriptions[:8]
                ]
            unknown = [
                p["operation_id"]
                for p in ch.entries(store, "publication")
                if p["thread_id"] == thread.thread_id and p["status"] in {"sending", "unknown"}
            ]
            if unknown:
                summary["unknown_publications"] = unknown[:8]
            if turn.status in {"paused", "interrupted"}:
                recovery = recovery_plan(store, turn.turn_id)
                summary.update(
                    reason=recovery.reason_code, request_id=recovery.request_id, actions=list(recovery.actions)
                )
                if recovery.request_id:
                    request = store.tx.first("interactions", where={"request_id": recovery.request_id})
                    if request and request.get("operation_id"):
                        operation = store.read_tool_operation(str(request["operation_id"]))
                        summary["approval"] = {
                            "tool": operation.tool_name,
                            "resources": [dict(r) for r in operation.resources],
                            "effects": list(operation.effects),
                        }
            # Summary hash includes persisted delivery status and concrete request,
            # never transient token deltas or changing log cursors.
            bounded = bounded_summary(summary)
            ch.enqueue_notification(store, grant["grant_id"], turn.turn_id, bounded, facts_snapshot=summary)


async def deliver_one(
    store: RuntimeStore, *, workspace: Path | None = None, client: ProviderClient | None = None
) -> bool:
    # Host owns a database-wide sender. SDK may restrict to its workspace.
    owner = "sender_" + uuid4().hex
    claim = await store.run_async(ch.claim_notification, store, owner=owner, workspace=workspace, lease_seconds=90)
    if claim is None:
        return False
    source_row = await store.run_async(ch.get, store, "source", claim["source_id"])
    assert source_row is not None
    source = Source.model_validate({key: value for key, value in source_row.items() if key in Source.model_fields})
    remaining = claim["deadline"] - time.time()
    source = source.model_copy(update={"timeout_seconds": min(source.timeout_seconds, max(0.001, remaining))})
    dispatched = False

    async def authorize() -> None:
        nonlocal dispatched
        await store.run_async(ch.begin_send, store, claim)
        dispatched = True

    try:
        ack = await (client or ProviderClient()).send(
            source, claim["destination"], claim["text"], claim["outbox_id"], authorize=authorize
        )
    except asyncio.CancelledError:
        if dispatched:
            await asyncio.shield(
                store.run_async(
                    ch.finish_send, store, claim, status="uncertain", reason="sender_cancelled_after_dispatch"
                )
            )
        raise
    if not dispatched:
        # Token/auth failures happened before the provider's final write. Persist
        # the confirmed unsent outcome without claiming a dispatch took place.
        await store.run_async(
            ch.finish_preflight, store, claim, status=ack.status, retry_after=ack.retry_after, reason=ack.error
        )
    else:
        await store.run_async(
            ch.finish_send,
            store,
            claim,
            status=ack.status,
            remote_id=ack.remote_id,
            retry_after=ack.retry_after,
            reason=ack.error,
        )
    return True


def own_echo(store: RuntimeStore, source: Source, event: Any) -> bool:
    """Only an author-bound, exact persisted outbound payload is a self echo."""
    if source.kind != "github" or not source.outbound_principal or event.principal != source.outbound_principal:
        return False
    for row in ch.entries(store, "outbox"):
        if (
            row["source_id"] == source.source_id
            and row["destination"] == event.destination
            and row["status"] in {"sending", "uncertain", "acknowledged"}
            and event.text == row["text"] + "\n\n<!-- praxis-delivery:" + row["outbox_id"] + " -->"
        ):
            return True
    for row in ch.entries(store, "publication"):
        if (
            row["authority"]["source_id"] == source.source_id
            and row["status"] in {"sending", "unknown", "succeeded"}
            and row["kind"] == "open_pr"
            and event.destination.startswith(f"github:{row['repository']}:")
            and event.text
            == row["scope"]["title"]
            + "\n"
            + row["scope"]["body"]
            + "\n\n<!-- praxis-publication:"
            + row["correlation"]
            + " -->"
        ):
            return True
        if (
            row["authority"]["source_id"] == source.source_id
            and row["status"] in {"sending", "unknown", "succeeded"}
            and row["kind"] == "review_reply"
            and event.destination == f"github:{row['repository']}:{row['scope']['pull_request']}"
            and event.text == row["scope"]["body"] + "\n\n<!-- praxis-publication:" + row["correlation"] + " -->"
        ):
            return True
    return False


def bounded_summary(summary: dict[str, Any]) -> dict[str, Any]:
    from agent_runtime.harness.operation_values import _canonical_json

    if len(_canonical_json(summary)) <= 4000:
        return summary
    result = {
        key: summary[key]
        for key in ("thread_id", "turn_id", "status", "delivery", "request_id", "actions", "reason")
        if key in summary
    }
    result["details_omitted"] = True
    result["details"] = "Inspect agent background status with the displayed Thread/Turn and local database."
    if "approval" in summary:
        approval = summary["approval"]
        result["approval"] = {
            "tool": str(approval["tool"])[:128],
            "resource_count": len(approval["resources"]),
            "resources": [str(resource)[:160] for resource in approval["resources"][:4]],
            "effects": [str(effect)[:80] for effect in approval["effects"][:8]],
        }
    if "remote" in summary:
        result["remote_count"] = len(summary["remote"])
        result["remote"] = [
            {"subscription": str(s["subscription"])[:80], "status": s["status"]} for s in summary["remote"][:4]
        ]
    return result


def record_diagnostic(store: RuntimeStore, stage: str, reason: str) -> None:
    with store.tx.transaction():
        for source in ch.entries(store, "source"):
            value = {
                "thread_id": source["thread_id"],
                "source_id": source["source_id"],
                "stage": stage,
                "reason": reason,
                "status": "needs_action",
            }
            key = ch.identity(source["source_id"], stage, reason)
            if ch.get(store, "channel_diagnostic", key) != value:
                ch.put(store, source["thread_id"], "channel_diagnostic", key, value)
