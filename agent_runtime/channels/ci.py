"""Native GitHub evidence to finite canonical subscriptions and follow-up Turns."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from agent_runtime.channels.providers import Envelope
from agent_runtime.harness import RuntimeStore
from agent_runtime.harness import channel_ops as ch
from agent_runtime.harness import ci_ops as ci


def authorize(store: RuntimeStore, subscription: Mapping[str, Any], *, repair: bool = False) -> bool:
    ref = subscription["grant_ref"]
    source = ch.get(store, "source", ref["source_id"])
    if (
        source is None
        or source.get("kind") != "github"
        or source.get("repository") != subscription["repository"]
        or ("source_revision" in ref and source["revision"] != ref["source_revision"])
    ):
        raise PermissionError("CI source configuration changed")
    grant = ch.authorize(
        store,
        ref["source_id"],
        ref["principal"],
        ref["conversation_id"],
        "input" if repair else "status",
        expected_revision=ref["revision"],
    )
    binding = ch.get(store, "binding", ch.identity(ref["source_id"], ref["conversation_id"]))
    if (
        binding is None
        or binding["revision"] != ref["binding_revision"]
        or grant["thread_id"] != subscription["thread_id"]
    ):
        raise PermissionError("CI subscription authorization changed")
    return True


def consume(store: RuntimeStore, event: Envelope) -> None:
    """Caller has verified the native signature and current event-sender grant."""
    native = event.evidence
    if not native or native.get("provider") != "github":
        return
    for sub in ch.entries(store, "ci_subscription"):
        if (
            not sub["enabled"]
            or sub["source_id"] != event.source_id
            or sub["repository"] != native.get("repository")
            or sub["pull_request"] != native.get("pull_request")
        ):
            continue

        def auth(value: Mapping[str, Any]) -> bool:
            return authorize(store, value)

        try:
            auth(sub)
        except PermissionError:
            continue
        if native["event"] == "pull_request" and native["action"] == "synchronize":
            if native.get("before_sha") == sub["head_sha"]:
                ci.update_head(store, sub["subscription_id"], native["head_sha"], authorize=auth)
            elif native["head_sha"] != sub["head_sha"]:
                ch.put(
                    store,
                    sub["thread_id"],
                    "ci_head_observation",
                    event.event_id,
                    {
                        "thread_id": sub["thread_id"],
                        "subscription_id": sub["subscription_id"],
                        "before_sha": native.get("before_sha"),
                        "head_sha": native["head_sha"],
                        "reason": "noncontiguous_head_requires_explicit_selection",
                    },
                )
            continue
        evidence: dict[str, Any] = dict(
            source_id=event.source_id,
            event_id=event.event_id,
            repository=sub["repository"],
            pull_request=sub["pull_request"],
            head_sha=native.get("head_sha"),
        )
        if native["event"] == "check_run":
            check_id = native.get("check_identity")
            if check_id not in sub["check_ids"]:
                continue
            state = (
                "pending"
                if native.get("status") != "completed"
                else "pass"
                if native.get("conclusion") == "success"
                else "fail"
                if native.get("conclusion") in {"failure", "timed_out", "action_required", "startup_failure"}
                else "cancel"
                if native.get("conclusion") == "cancelled"
                else "skipping"
            )
            results = dict(sub.get("check_results", {})) if native["head_sha"] == sub["head_sha"] else {}
            old = results.get(check_id)
            run = int(native["check_id"])
            if native["action"] == "rerequested":
                state = "pending"
                results[check_id] = {"id": check_id, "state": state, "run": run, "invalidated": True}
            elif old is not None and old["run"] != run and state != "pending":
                # IDs establish identity, not temporal ordering. A competing
                # completed run cannot override the selected run's evidence.
                state = "pending"
                results[check_id] = {**old, "state": "pending"}
            elif old is not None and old["run"] == run and old.get("invalidated"):
                results[check_id] = old
            elif old is not None and old["run"] == run and old["state"] != "pending" and state == "pending":
                continue
            else:
                results[check_id] = {"id": check_id, "state": state, "run": run}
            evidence.update(kind="checks", checks=[{"id": v["id"], "state": v["state"]} for v in results.values()])
            if native["head_sha"] == sub["head_sha"]:
                sub["check_results"] = results
                ch.put(store, sub["thread_id"], "ci_subscription", sub["subscription_id"], sub)
        elif native["event"] in {"pull_request_review", "pull_request_review_comment"}:
            evidence.update(
                kind="review",
                review_id=native["review_id"],
                review_kind="comment" if native["event"] == "pull_request_review_comment" else "review",
                state=(native.get("state") or "commented").lower(),
                reviewer_id=native.get("reviewer_id"),
                submitted_at=native.get("submitted_at"),
                text=event.text,
                path=native.get("path"),
                line=native.get("line"),
                commit_id=native.get("commit_id"),
            )
            evidence["head_sha"] = native.get("commit_id") or native.get("head_sha")
        else:
            continue
        receipt = ci.record_evidence(store, sub["subscription_id"], evidence, authorize=auth, verify=lambda _: True)
        if sub["repair_allowance"] is not None and not receipt["stale"]:
            try:
                ci.enqueue_repair(
                    store,
                    sub["subscription_id"],
                    receipt["evidence_id"],
                    authorize=lambda value: authorize(store, value, repair=True),
                )
            except (PermissionError, RuntimeError) as error:
                # Structured evidence survives exhausted authority/budget/busy
                # admission. Host retries eligible evidence, never resets usage.
                ch.put(
                    store,
                    sub["thread_id"],
                    "ci_action",
                    receipt["evidence_id"],
                    {
                        "thread_id": sub["thread_id"],
                        "subscription_id": sub["subscription_id"],
                        "evidence_id": receipt["evidence_id"],
                        "reason": type(error).__name__,
                        "status": "waiting",
                    },
                )


def resume_eligible(store: RuntimeStore) -> None:
    """A busy Thread may finish later; retries preserve the original allowance."""
    for sub in ch.entries(store, "ci_subscription"):
        if not sub["enabled"] or sub["repair_allowance"] is None or sub["evidence_id"] is None:
            continue
        try:
            ci.enqueue_repair(
                store,
                sub["subscription_id"],
                sub["evidence_id"],
                authorize=lambda value: authorize(store, value, repair=True),
            )
        except (PermissionError, RuntimeError):
            pass


async def poll_one(store: RuntimeStore, client: Any) -> bool:
    import time

    from agent_runtime.channels.providers import ProviderError, Source

    def reserve() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]] | None:
        with store.tx.transaction():
            for sub in ch.entries(store, "ci_subscription"):
                if not sub["enabled"]:
                    continue
                try:
                    token = ci.reserve_poll(
                        store, sub["subscription_id"], authorize=lambda value: authorize(store, value)
                    )
                except PermissionError:
                    continue
                if token is not None:
                    source = ch.get(store, "source", sub["source_id"])
                    assert source is not None
                    return sub, token, source
        return None

    selected = await store.run_async(reserve)
    if selected is None:
        return False
    sub, token, source_data = selected
    source = Source.model_validate({k: v for k, v in source_data.items() if k in Source.model_fields})
    source = source.model_copy(
        update={"timeout_seconds": min(source.timeout_seconds, max(0.001, token["deadline"] - time.time()))}
    )

    async def auth() -> None:
        def check() -> None:
            current = ch.get(store, "ci_subscription", sub["subscription_id"])
            if current is None or not current["enabled"] or current["head_sha"] != token["head_sha"]:
                raise PermissionError("CI poll superseded")
            authorize(store, current)

        await store.run_async(check)

    try:
        snapshot = await client.poll_ci(source, sub["pull_request"], authorize=auth)
    except PermissionError:
        return True
    except (ProviderError, TimeoutError):
        snapshot = {"head_sha": token["head_sha"], "checks": [], "kind": "query_error"}

    def commit() -> None:
        with store.tx.transaction():
            current = ch.get(store, "ci_subscription", sub["subscription_id"])
            if current is None or current["head_sha"] != token["head_sha"]:
                return
            authorize(store, current)
            if current["evidence_id"] != sub["evidence_id"] or current.get("check_results") != sub.get("check_results"):
                ch.put(
                    store,
                    current["thread_id"],
                    "ci_poll_observation",
                    f"{sub['subscription_id']}:{token['attempt']}",
                    {
                        "thread_id": current["thread_id"],
                        "subscription_id": sub["subscription_id"],
                        "snapshot": snapshot,
                        "reason": "superseded_during_query",
                    },
                )
                return

            def auth_cb(value: Mapping[str, Any]) -> bool:
                return authorize(store, value)

            if snapshot["head_sha"] != token["head_sha"]:
                current["remote_status"] = "head_mismatch"
                current["observed_remote_head"] = snapshot["head_sha"]
                ch.put(store, current["thread_id"], "ci_subscription", current["subscription_id"], current)
            checks = [{"id": c["id"], "state": c["state"]} for c in snapshot["checks"] if c["id"] in sub["check_ids"]]
            if snapshot["head_sha"] == current["head_sha"] and snapshot.get("kind") != "query_error":
                current["check_results"] = {c["id"]: c for c in snapshot["checks"] if c["id"] in sub["check_ids"]}
                ch.put(store, current["thread_id"], "ci_subscription", current["subscription_id"], current)
            evidence = dict(
                source_id=sub["source_id"],
                event_id=f"poll:{sub['subscription_id']}:{token['attempt']}",
                repository=sub["repository"],
                pull_request=sub["pull_request"],
                head_sha=snapshot["head_sha"],
                checks=checks,
                kind=snapshot.get("kind", "checks"),
            )
            ci.record_evidence(store, sub["subscription_id"], evidence, authorize=auth_cb, verify=lambda _: True)
            resume_eligible(store)

    try:
        await store.run_async(commit)
    except PermissionError:
        pass
    return True
