"""CI evidence and finite follow-ups use canonical Thread facts and host commands."""

import pytest

from agent_runtime.harness import RuntimeStore, host_ops, turn_ops


def setup_subscription(store, workspace, **overrides):
    from agent_runtime.harness import ci_ops as ci

    submission = host_ops.receive_command(
        store,
        workspace=workspace,
        kind="start",
        command_id="initial",
        payload={
            "text": "fix",
            "configuration": {"agent": {"workspace_path": str(workspace)}},
            "options": {
                "max_steps": 8,
                "max_tokens_total": 800,
                "max_cost_micros": 80,
                "allow_write_tools": True,
                "allow_execute_tools": True,
                "allow_web_tools": False,
            },
            "files": [],
        },
    )
    turn = turn_ops.start_turn(
        store, thread_id=submission.thread_id, turn_id=submission.turn_id, user_message="fix", binding_manifest={}
    )
    turn_ops.complete_turn(store, turn_id=turn.turn_id, answer="done")
    claim = host_ops.claim_next(store, owner="host", now=1)
    host_ops.settle_command(store, claim, status="applied", result={}, now=1)
    values = dict(
        subscription_id="ci-1",
        thread_id=turn.thread_id,
        workspace=workspace,
        source_id="github",
        repository="owner/repo",
        pull_request=7,
        head_sha="sha-1",
        check_ids=("unit", "lint"),
        grant_ref={"grant_id": "grant", "revision": 1},
        repair_allowance={
            "deadline": 1000,
            "max_new_turns": 2,
            "max_steps": 10,
            "max_tokens_total": 1000,
            "max_cost_micros": 100,
        },
        repair_budget={"max_steps": 6, "max_tokens_total": 600, "max_cost_micros": 60},
        poll_policy={"deadline": 500, "max_attempts": 3, "initial_delay": 2, "max_delay": 20},
        authorize=lambda _: True,
        now=10,
    )
    values.update(overrides)
    return ci.create_subscription(store, **values), turn


def evidence(event_id="event-1", **overrides):
    value = dict(
        source_id="github",
        event_id=event_id,
        repository="owner/repo",
        pull_request=7,
        head_sha="sha-1",
        kind="checks",
        checks=[{"id": "unit", "state": "pass"}, {"id": "lint", "state": "pass"}],
    )
    value.update(overrides)
    return value


def record(store, value, **kwargs):
    from agent_runtime.harness import ci_ops as ci

    return ci.record_evidence(store, "ci-1", value, authorize=lambda _: True, verify=lambda _: True, now=20, **kwargs)


def test_evidence_dedup_conflict_exact_binding_and_rebuild(tmp_path):
    from agent_runtime.harness import ci_ops as ci

    database = tmp_path / "runtime.db"
    with RuntimeStore.open_sync(database) as store:
        setup_subscription(store, tmp_path)
        first = record(store, evidence())
        assert record(store, evidence()) == first
        with pytest.raises(ValueError, match="conflict"):
            record(store, evidence(checks=[]))
        for change in (
            {"repository": "other/repo"},
            {"pull_request": 8},
            {"checks": [{"id": "unsubscribed", "state": "pass"}]},
        ):
            with pytest.raises(PermissionError):
                record(store, evidence("wrong", **change))
        assert ci.status(store, "ci-1", authorize=lambda _: True)["remote_status"] == "pass"
        assert store.verify().valid
        store.rebuild_projections()
        assert ci.status(store, "ci-1", authorize=lambda _: True)["remote_status"] == "pass"
    with RuntimeStore.open_sync(database) as store:
        assert record(store, evidence()) == first


@pytest.mark.parametrize(
    "change,expected",
    [
        ({"checks": []}, "none"),
        ({"checks": [{"id": "unit", "state": "pass"}]}, "pending"),
        ({"checks": [{"id": "unit", "state": "pending"}]}, "pending"),
        ({"kind": "query_error", "checks": [], "error": "offline"}, "query_error"),
    ],
)
def test_non_success_evidence_never_passes(tmp_path, change, expected):
    from agent_runtime.harness import ci_ops as ci

    with RuntimeStore.open_sync(tmp_path / "runtime.db") as store:
        setup_subscription(store, tmp_path)
        record(store, evidence(**change))
        status = ci.status(store, "ci-1", authorize=lambda _: True)
        assert status["remote_status"] == expected
        assert status["delivery_complete"] is False
        assert "local_delivery" in status


def test_head_change_invalidates_pass_and_late_old_success(tmp_path):
    from agent_runtime.harness import ci_ops as ci

    with RuntimeStore.open_sync(tmp_path / "runtime.db") as store:
        setup_subscription(store, tmp_path)
        record(store, evidence())
        ci.update_head(store, "ci-1", "sha-2", authorize=lambda _: True)
        assert record(store, evidence("late"))["stale"] is True
        assert ci.status(store, "ci-1", authorize=lambda _: True)["remote_status"] == "pending"
        record(store, evidence("current", head_sha="sha-2"))
        assert ci.status(store, "ci-1", authorize=lambda _: True)["remote_status"] == "pass"


def test_authorization_and_signature_revalidated_before_cached_evidence(tmp_path):
    from agent_runtime.harness import ci_ops as ci

    with RuntimeStore.open_sync(tmp_path / "runtime.db") as store:
        setup_subscription(store, tmp_path)
        record(store, evidence())
        with pytest.raises(PermissionError):
            ci.record_evidence(store, "ci-1", evidence(), authorize=lambda _: False, verify=lambda _: True)
        with pytest.raises(PermissionError):
            ci.record_evidence(store, "ci-1", evidence(), authorize=lambda _: True, verify=lambda _: False)
        with pytest.raises(PermissionError):
            ci.status(store, "ci-1", authorize=lambda _: False)


def test_finite_repair_copies_submission_and_reserves_cumulative_budget(tmp_path):
    from agent_runtime.harness import ci_ops as ci

    database = tmp_path / "runtime.db"
    with RuntimeStore.open_sync(database) as store:
        _, turn = setup_subscription(store, tmp_path)
        receipt = record(store, evidence(checks=[{"id": "unit", "state": "fail"}]))
        command = ci.enqueue_repair(store, "ci-1", receipt["evidence_id"], authorize=lambda _: True, now=30)
        assert command.kind == "start"
        assert command.thread_id == turn.thread_id
        assert command.request["previous_turn_id"] == turn.turn_id
        assert (
            command.request["configuration"]
            == host_ops.background_submission(store, turn.turn_id).request["configuration"]
        )
        assert command.request["options"]["max_steps"] == 6
        assert command.request["options"]["allow_write_tools"] is True
        assert len(store.list_turns()) == 1  # the existing host, not CI, creates the Turn
        assert ci.enqueue_repair(store, "ci-1", receipt["evidence_id"], authorize=lambda _: True, now=30) == command
        with pytest.raises(PermissionError):
            ci.enqueue_repair(store, "ci-1", receipt["evidence_id"], authorize=lambda _: False, now=30)
    with RuntimeStore.open_sync(database) as store:
        current = ci.status(store, "ci-1", authorize=lambda _: True)
        assert current["reserved"]["max_steps"] == 6
        assert current["new_turns"] == 1
        # A different event cannot enqueue another start while the first is outstanding.
        another = record(store, evidence("event-2", checks=[{"id": "unit", "state": "fail"}]))
        with pytest.raises(RuntimeError, match="queued"):
            ci.enqueue_repair(store, "ci-1", another["evidence_id"], authorize=lambda _: True, now=31)
        assert ci.status(store, "ci-1", authorize=lambda _: True)["new_turns"] == 1


def test_repair_denied_without_finite_allowance_and_after_deadline(tmp_path):
    from agent_runtime.harness import ci_ops as ci

    with RuntimeStore.open_sync(tmp_path / "runtime.db") as store:
        setup_subscription(store, tmp_path, repair_allowance=None, repair_budget=None)
        receipt = record(store, evidence(checks=[{"id": "unit", "state": "fail"}]))
        with pytest.raises(PermissionError, match="allowance"):
            ci.enqueue_repair(store, "ci-1", receipt["evidence_id"], authorize=lambda _: True, now=30)
    other = tmp_path / "other"
    other.mkdir()
    with RuntimeStore.open_sync(other / "runtime.db") as store:
        setup_subscription(store, other)
        receipt = record(store, evidence(checks=[{"id": "unit", "state": "fail"}]))
        with pytest.raises(PermissionError, match="deadline"):
            ci.enqueue_repair(store, "ci-1", receipt["evidence_id"], authorize=lambda _: True, now=1001)
        assert len(host_ops.list_commands(store)) == 1


def test_poll_reservations_survive_restart_and_stop_at_count_or_deadline(tmp_path):
    from agent_runtime.harness import ci_ops as ci

    database = tmp_path / "runtime.db"
    with RuntimeStore.open_sync(database) as store:
        setup_subscription(store, tmp_path)
        assert ci.reserve_poll(store, "ci-1", authorize=lambda _: True, now=10)["attempt"] == 1
        assert ci.reserve_poll(store, "ci-1", authorize=lambda _: True, now=11) is None
    with RuntimeStore.open_sync(database) as store:
        assert ci.reserve_poll(store, "ci-1", authorize=lambda _: True, now=12)["attempt"] == 2
        assert ci.reserve_poll(store, "ci-1", authorize=lambda _: True, now=16)["attempt"] == 3
        assert ci.reserve_poll(store, "ci-1", authorize=lambda _: True, now=100) is None
        assert ci.status(store, "ci-1", authorize=lambda _: True)["poll_attempts"] == 3


def test_old_head_success_cannot_close_after_explicit_advance(tmp_path):
    from agent_runtime.harness import ci_ops as ci

    with RuntimeStore.open_sync(tmp_path / "r.db") as store:
        setup_subscription(store, tmp_path)
        record(store, evidence())
        ci.update_head(store, "ci-1", "sha-2", authorize=lambda _: True)
        record(store, evidence("old-head", head_sha="sha-1"))
        status = ci.status(store, "ci-1", authorize=lambda _: True)
        assert status["remote_status"] == "pending"
        assert status["delivery_complete"] is False


@pytest.mark.anyio
async def test_poll_observing_old_head_never_rolls_back_selected_head(tmp_path):
    import time

    from agent_runtime.channels.ci import poll_one
    from agent_runtime.harness import channel_ops as ch
    from agent_runtime.harness import ci_ops as ci
    from tests.agent.test_channel_runtime import channel_setup

    async with await RuntimeStore.open(tmp_path / "r.db") as store:

        def setup():
            _, start, grant = channel_setup(store, tmp_path)
            ref = {k: grant[k] for k in ("source_id", "principal", "conversation_id", "revision")}
            ref["binding_revision"] = 1
            # Use a valid outbound Source for the provider polling adapter.
            ch.configure_source(
                store,
                start.thread_id,
                {
                    "source_id": "gh",
                    "kind": "github",
                    "enabled": True,
                    "repository": "o/r",
                    "webhook_secret_env": "SECRET",
                },
            )
            return ci.create_subscription(
                store,
                subscription_id="ci",
                thread_id=start.thread_id,
                workspace=tmp_path,
                source_id="gh",
                repository="o/r",
                pull_request=1,
                head_sha="new",
                check_ids=["5:unit"],
                grant_ref=ref,
                poll_policy={"deadline": time.time() + 60, "max_attempts": 1, "initial_delay": 1, "max_delay": 2},
                authorize=lambda _: True,
            )

        await store.run_async(setup)

        class Remote:
            async def poll_ci(self, *args, **kwargs):
                await kwargs["authorize"]()
                return {"head_sha": "old", "checks": [{"id": "5:unit", "state": "pass", "run": 1}]}

        assert await poll_one(store, Remote())
        sub = await store.run_async(ch.get, store, "ci_subscription", "ci")
        assert sub["head_sha"] == "new"
        assert sub["remote_status"] == "head_mismatch"
        assert sub["poll_attempts"] == 1


def test_unrelated_review_or_comment_cannot_clear_outstanding_change_request(tmp_path):
    from agent_runtime.harness import ci_ops as ci

    with RuntimeStore.open_sync(tmp_path / "r.db") as store:
        setup_subscription(store, tmp_path)
        record(store, evidence())
        record(
            store,
            evidence(
                "blocking",
                kind="review",
                review_id="10",
                reviewer_id="alice",
                submitted_at="2026-11-04T01:00:00Z",
                state="changes_requested",
                checks=[],
            ),
        )
        record(
            store,
            evidence(
                "comment",
                kind="review",
                review_id="11",
                reviewer_id="bob",
                submitted_at="2026-11-04T02:00:00Z",
                state="commented",
                checks=[],
            ),
        )
        assert ci.status(store, "ci-1", authorize=lambda _: True)["remote_status"] == "fail"
        ci.update_head(store, "ci-1", "sha-2", authorize=lambda _: True)
        record(store, evidence("new-checks", head_sha="sha-2"))
        assert ci.status(store, "ci-1", authorize=lambda _: True)["remote_status"] == "fail"
        record(
            store,
            evidence(
                "approve",
                head_sha="sha-2",
                kind="review",
                review_id="12",
                reviewer_id="alice",
                submitted_at="2026-11-04T03:00:00Z",
                state="approved",
                checks=[],
            ),
        )
        assert ci.status(store, "ci-1", authorize=lambda _: True)["remote_status"] == "pass"
        record(
            store,
            evidence(
                "late-block",
                head_sha="sha-2",
                kind="review",
                review_id="13",
                reviewer_id="alice",
                submitted_at="2026-11-04T00:00:00Z",
                state="changes_requested",
                checks=[],
            ),
        )
        # A late older change request may conservatively block; it cannot clear newer blocks.
        assert ci.status(store, "ci-1", authorize=lambda _: True)["delivery_complete"] is False


@pytest.mark.anyio
async def test_delayed_poll_cannot_overwrite_newer_webhook_observation(tmp_path):
    import time

    from agent_runtime.channels.ci import poll_one
    from agent_runtime.harness import channel_ops as ch
    from agent_runtime.harness import ci_ops as ci
    from tests.agent.test_channel_runtime import channel_setup

    async with await RuntimeStore.open(tmp_path / "r.db") as store:

        def setup():
            _, start, grant = channel_setup(store, tmp_path)
            ch.configure_source(
                store,
                start.thread_id,
                {
                    "source_id": "gh",
                    "kind": "github",
                    "enabled": True,
                    "repository": "o/r",
                    "webhook_secret_env": "SECRET",
                },
            )
            ref = {k: grant[k] for k in ("source_id", "principal", "conversation_id", "revision")}
            ref["binding_revision"] = 1
            return ci.create_subscription(
                store,
                subscription_id="ci",
                thread_id=start.thread_id,
                workspace=tmp_path,
                source_id="gh",
                repository="o/r",
                pull_request=1,
                head_sha="same",
                check_ids=["5:unit"],
                grant_ref=ref,
                poll_policy={"deadline": time.time() + 60, "max_attempts": 1, "initial_delay": 1, "max_delay": 2},
                authorize=lambda _: True,
            )

        await store.run_async(setup)

        class Remote:
            async def poll_ci(self, *args, **kwargs):
                await kwargs["authorize"]()

                def webhook():
                    sub = ch.get(store, "ci_subscription", "ci")
                    sub["check_results"] = {"5:unit": {"id": "5:unit", "state": "pending", "run": 11}}
                    ch.put(store, sub["thread_id"], "ci_subscription", "ci", sub)
                    ci.record_evidence(
                        store,
                        "ci",
                        {
                            "source_id": "gh",
                            "event_id": "webhook",
                            "repository": "o/r",
                            "pull_request": 1,
                            "head_sha": "same",
                            "kind": "checks",
                            "checks": [{"id": "5:unit", "state": "pending"}],
                        },
                        authorize=lambda _: True,
                        verify=lambda _: True,
                    )

                await store.run_async(webhook)
                return {"head_sha": "same", "checks": [{"id": "5:unit", "state": "pass", "run": 10}]}

        assert await poll_one(store, Remote())
        sub = await store.run_async(ch.get, store, "ci_subscription", "ci")
        assert sub["remote_status"] == "pending"
        assert sub["check_results"]["5:unit"]["run"] == 11
        observations = await store.run_async(ch.entries, store, "ci_poll_observation")
        assert observations[0]["reason"] == "superseded_during_query"


def test_same_provider_event_can_feed_two_explicit_subscriptions(tmp_path):
    from agent_runtime.harness import ci_ops as ci

    with RuntimeStore.open_sync(tmp_path / "r.db") as store:
        sub, turn = setup_subscription(store, tmp_path)
        config = dict(sub["configuration"])
        config.update(subscription_id="ci-2", workspace=tmp_path, authorize=lambda _: True, now=10)
        ci.create_subscription(store, **config)
        first = record(store, evidence())
        second = ci.record_evidence(store, "ci-2", evidence(), authorize=lambda _: True, verify=lambda _: True, now=20)
        assert first["evidence_id"] != second["evidence_id"]
        assert ci.status(store, "ci-1", authorize=lambda _: True)["remote_status"] == "pass"
        assert ci.status(store, "ci-2", authorize=lambda _: True)["remote_status"] == "pass"


def test_native_review_comment_id_namespace_cannot_clear_review_decision(tmp_path, monkeypatch):
    import hashlib
    import hmac
    import json
    import time

    from agent_runtime.channels.ci import consume
    from agent_runtime.channels.providers import Envelope, Source, parse_event
    from agent_runtime.harness import ci_ops as ci
    from tests.agent.test_channel_runtime import channel_setup

    monkeypatch.setenv("SECRET", "secret")
    with RuntimeStore.open_sync(tmp_path / "r.db") as store:
        ch, start, grant = channel_setup(store, tmp_path)
        ch.configure_source(
            store,
            start.thread_id,
            {"source_id": "gh", "kind": "github", "enabled": True, "repository": "o/r", "webhook_secret_env": "SECRET"},
        )
        ref = {k: grant[k] for k in ("source_id", "principal", "conversation_id", "revision")}
        ref["binding_revision"] = 1
        ci.create_subscription(
            store,
            subscription_id="ci",
            thread_id=start.thread_id,
            workspace=tmp_path,
            source_id="gh",
            repository="o/r",
            pull_request=1,
            head_sha="same",
            check_ids=["5:unit"],
            grant_ref=ref,
            poll_policy={"deadline": time.time() + 60, "max_attempts": 1, "initial_delay": 1, "max_delay": 2},
            authorize=lambda _: True,
        )
        ci.record_evidence(
            store,
            "ci",
            {
                "source_id": "gh",
                "event_id": "checks",
                "repository": "o/r",
                "pull_request": 1,
                "head_sha": "same",
                "kind": "checks",
                "checks": [{"id": "5:unit", "state": "pass"}],
            },
            authorize=lambda _: True,
            verify=lambda _: True,
        )
        source = Source(source_id="gh", kind="github", repository="o/r", webhook_secret_env="SECRET")

        def send(kind, key, state, event_id):
            payload = {
                "repository": {"full_name": "o/r"},
                "sender": {"id": 7, "type": "User"},
                "action": "submitted" if key == "review" else "created",
                "pull_request": {"number": 1, "head": {"sha": "same"}},
                key: {
                    "id": 10,
                    "user": {"id": 7},
                    "body": "feedback",
                    "state": state,
                    "commit_id": "same",
                    "submitted_at": "2026-11-04T01:00:00Z",
                },
            }
            raw = json.dumps(payload).encode()
            headers = {
                "x-hub-signature-256": "sha256=" + hmac.new(b"secret", raw, hashlib.sha256).hexdigest(),
                "x-github-delivery": event_id,
                "x-github-event": kind,
            }
            event = parse_event(source, headers, raw)
            assert isinstance(event, Envelope)
            consume(store, event)

        send("pull_request_review", "review", "changes_requested", "review")
        send("pull_request_review_comment", "comment", None, "comment")
        status = ci.status(store, "ci", authorize=lambda _: True)
        assert status["remote_status"] == "fail"
