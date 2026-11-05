from pathlib import Path

import pytest

from agent_runtime.harness import RuntimeStore, turn_ops


def test_channel_facts_rebuild_and_conflict(tmp_path: Path) -> None:
    from agent_runtime.harness import channel_ops as ch

    with RuntimeStore.open_sync(tmp_path / "runtime.sqlite") as store:
        thread = turn_ops.create_thread(store, workspace=tmp_path)
        ch.put(store, thread.thread_id, "test", "key", {"value": 1})
        assert ch.get(store, "test", "key")["value"] == 1
        ch.put(store, thread.thread_id, "test", "key", {"value": 2})
        assert store.verify().valid
        store.rebuild_projections()
        assert ch.get(store, "test", "key")["value"] == 2
        other = turn_ops.create_thread(store, workspace=tmp_path)
        with pytest.raises(PermissionError):
            ch.put(store, other.thread_id, "test", "key", {"value": 3})


def channel_setup(store, workspace):
    from agent_runtime.harness import channel_ops as ch
    from agent_runtime.harness import host_ops

    start = host_ops.receive_command(
        store, workspace=workspace, kind="start", payload={"text": "work"}, command_id="start"
    )
    ch.configure_source(store, start.thread_id, {"source_id": "gh", "kind": "github", "enabled": True})
    ch.bind(
        store, source_id="gh", conversation_id="github:o/r:1", thread_id=start.thread_id, destination="github:o/r:1"
    )
    grant = ch.set_grant(
        store,
        source_id="gh",
        principal="7",
        conversation_id="github:o/r:1",
        thread_id=start.thread_id,
        operations=["input", "status", "respond", "cancel", "notify"],
        role="owner",
        enabled=True,
    )
    return ch, start, grant


def envelope(**changes):
    from types import SimpleNamespace

    data = dict(
        source_id="gh",
        event_id="event1",
        digest="payload1",
        native_id="comment1",
        principal="7",
        principal_kind="user",
        conversation_id="github:o/r:1",
        destination="github:o/r:1",
        operation="input",
        text="follow up",
        attachments=(),
        request_id=None,
        action=None,
        evidence={},
    )
    data.update(changes)
    return SimpleNamespace(**data, model_dump=lambda **_: data)


def test_receive_is_atomic_idempotent_and_revalidates_cached_authority(tmp_path):
    from agent_runtime.harness import host_ops

    with RuntimeStore.open_sync(tmp_path / "runtime.sqlite") as store:
        ch, start, grant = channel_setup(store, tmp_path)
        receipt = ch.receive(store, envelope())
        assert ch.receive(store, envelope()) == receipt
        assert len(host_ops.list_commands(store)) == 2
        with pytest.raises(ValueError, match="conflict"):
            ch.receive(store, envelope(digest="different"))
        # Changed unsigned delivery ID cannot replay a signed native comment.
        assert ch.receive(store, envelope(event_id="changed-header")) == receipt
        ch.set_grant(
            store,
            **{k: grant[k] for k in ("source_id", "principal", "conversation_id", "thread_id", "operations", "role")},
            enabled=False,
        )
        with pytest.raises(PermissionError):
            ch.receive(store, envelope())
        cmd = host_ops.read_command(store, receipt["command_id"])
        with pytest.raises(PermissionError):
            ch.authorize_command(store, cmd)
        assert store.verify().valid


def test_failed_receive_rolls_back_inbox_and_wake(tmp_path, monkeypatch):
    from agent_runtime.harness import host_ops

    with RuntimeStore.open_sync(tmp_path / "runtime.sqlite") as store:
        ch, _, _ = channel_setup(store, tmp_path)

        def fail(*args, **kwargs):
            raise OSError("disk full")

        monkeypatch.setattr(host_ops, "receive_command", fail)
        with pytest.raises(OSError):
            ch.receive(store, envelope())
        assert ch.entries(store, "inbox") == ()


def test_bot_and_cross_audience_cannot_control(tmp_path):
    with RuntimeStore.open_sync(tmp_path / "runtime.sqlite") as store:
        ch, _, _ = channel_setup(store, tmp_path)
        with pytest.raises(PermissionError):
            ch.receive(store, envelope(principal_kind="bot", operation="cancel"))
        with pytest.raises(PermissionError):
            ch.receive(store, envelope(destination="github:o/r:2"))


def test_outbox_revoke_at_claim_and_send_unknown_never_retries(tmp_path):
    with RuntimeStore.open_sync(tmp_path / "runtime.sqlite") as store:
        ch, start, grant = channel_setup(store, tmp_path)
        note = ch.enqueue_notification(store, grant["grant_id"], start.turn_id, {"status": "needs_action"}, now=100)
        claim = ch.claim_notification(store, owner="one", now=100)
        assert claim["outbox_id"] == note["outbox_id"]
        assert ch.begin_send(store, claim, now=101)["status"] == "sending"
        # Crash after dispatch, even without ACK, must not be retried by another owner.
        assert ch.claim_notification(store, owner="two", now=200) is None
        assert ch.get(store, "outbox", note["outbox_id"])["status"] == "uncertain"
        with pytest.raises(ValueError):
            ch.retry_notification(store, note["outbox_id"], now=201)
        ch.abandon_notification(store, note["outbox_id"])
        note2 = ch.enqueue_notification(store, grant["grant_id"], start.turn_id, {"status": "changed"}, now=210)
        claim2 = ch.claim_notification(store, owner="two", now=210)
        ch.set_grant(
            store,
            **{k: grant[k] for k in ("source_id", "principal", "conversation_id", "thread_id", "operations", "role")},
            enabled=False,
        )
        with pytest.raises(PermissionError):
            ch.begin_send(store, claim2, now=211)
        assert ch.claim_notification(store, owner="three", now=300) is None
        assert ch.get(store, "outbox", note2["outbox_id"])["status"] == "blocked"


def test_outbox_finite_retry_after_and_dedup(tmp_path):
    with RuntimeStore.open_sync(tmp_path / "runtime.sqlite") as store:
        ch, start, grant = channel_setup(store, tmp_path)
        first = ch.enqueue_notification(
            store, grant["grant_id"], start.turn_id, {"status": "waiting"}, now=100, max_attempts=2, deadline_seconds=50
        )
        assert ch.enqueue_notification(store, grant["grant_id"], start.turn_id, {"status": "waiting"}, now=101) == first
        claim = ch.claim_notification(store, owner="one", now=100)
        ch.begin_send(store, claim, now=100)
        ch.finish_send(store, claim, status="unsent", retry_after=20, now=101)
        assert ch.claim_notification(store, owner="two", now=120) is None
        next_claim = ch.claim_notification(store, owner="two", now=121)
        ch.begin_send(store, next_claim, now=121)
        ch.finish_send(store, next_claim, status="unsent", now=122)
        assert ch.claim_notification(store, owner="three", now=150) is None
        assert ch.get(store, "outbox", first["outbox_id"])["status"] == "blocked"


@pytest.mark.anyio
async def test_signed_event_public_service_to_host_outbox_and_mock_provider(tmp_path, monkeypatch):
    import hashlib
    import hmac
    import json

    import httpx

    from agent_runtime import Agent
    from agent_runtime.channels import Channels
    from agent_runtime.channels.providers import ProviderClient, Source
    from agent_runtime.harness import channel_ops as ch
    from agent_runtime.host import BackgroundHost
    from tests.agent.harness.test_subagent_budget import SmallAnswerModel

    monkeypatch.setenv("GH_SECRET", "secret")
    monkeypatch.setenv("GH_TOKEN", "token")
    agent = Agent(
        workspace_path=tmp_path,
        checkpoint_db=tmp_path / "runtime.sqlite",
        model_session_path=None,
        enable_workspace_mcp=False,
    )
    handle = await agent.start_background("inspect", input_id="first", max_steps=3, autostart=False)
    channels = Channels(database=handle.database, workspace=tmp_path)
    await channels.configure(
        handle.thread_id,
        Source(
            source_id="gh",
            kind="github",
            repository="o/r",
            webhook_secret_env="GH_SECRET",
            outbound_token_env="GH_TOKEN",
            outbound_principal="99",
        ),
    )
    await channels.bind(
        source_id="gh", conversation_id="github:o/r:1", destination="github:o/r:1", thread_id=handle.thread_id
    )
    await channels.grant(
        source_id="gh",
        principal="7",
        conversation_id="github:o/r:1",
        thread_id=handle.thread_id,
        role="owner",
        operations=["input", "status", "notify"],
    )
    raw = json.dumps(
        {
            "repository": {"full_name": "o/r"},
            "sender": {"id": 7, "type": "User"},
            "action": "created",
            "issue": {"number": 1},
            "comment": {"id": 9, "body": "inspect tests", "user": {"id": 7}},
        }
    ).encode()
    headers = {
        "x-hub-signature-256": "sha256=" + hmac.new(b"secret", raw, hashlib.sha256).hexdigest(),
        "x-github-delivery": "delivery1",
        "x-github-event": "issue_comment",
    }
    receipt = await channels.receive("gh", headers, raw)
    assert await channels.receive("gh", headers, raw) == receipt
    sent = []

    async def provider(request):
        sent.append(json.loads(request.content))
        return httpx.Response(201, json={"id": 99})

    async with BackgroundHost(handle.database, model=SmallAnswerModel()) as host:
        await host.tick()
        assert (await handle.wait(timeout=4))["status"] == "completed"
    await channels.collect()
    assert await channels.deliver(client=ProviderClient(transport=httpx.MockTransport(provider)))
    assert len(sent) == 1
    await channels.collect()
    assert not await channels.deliver(client=ProviderClient(transport=httpx.MockTransport(provider)))
    assert len(sent) == 1
    async with await RuntimeStore.open(handle.database) as store:
        records = await store.run_async(ch.entries, store, "outbox")
        assert records[0]["status"] == "acknowledged"
        assert records[0]["summary"]["delivery"] == "unverified"
        assert (await store.run_async(store.verify)).valid


@pytest.mark.anyio
async def test_signed_ci_failure_enqueues_finite_turn_and_revocation_fences_it(tmp_path, monkeypatch):
    import hashlib
    import hmac
    import json
    import time

    from agent_runtime import Agent
    from agent_runtime.channels import Channels
    from agent_runtime.channels.providers import Source
    from agent_runtime.harness import host_ops
    from agent_runtime.host import BackgroundHost
    from tests.agent.harness.test_subagent_budget import SmallAnswerModel

    monkeypatch.setenv("GH_SECRET", "secret")
    agent = Agent(
        workspace_path=tmp_path,
        checkpoint_db=tmp_path / "runtime.db",
        model_session_path=None,
        enable_workspace_mcp=False,
    )
    handle = await agent.start_background("inspect", autostart=False, max_steps=3)
    async with BackgroundHost(handle.database, model=SmallAnswerModel()) as host:
        await host.tick()
        await handle.wait(timeout=3)
    channels = Channels(database=handle.database, workspace=tmp_path)
    await channels.configure(
        handle.thread_id, Source(source_id="gh", kind="github", repository="o/r", webhook_secret_env="GH_SECRET")
    )
    await channels.bind(
        source_id="gh", conversation_id="github:o/r:1", destination="github:o/r:1", thread_id=handle.thread_id
    )
    await channels.grant(
        source_id="gh",
        principal="7",
        conversation_id="github:o/r:1",
        thread_id=handle.thread_id,
        operations=["status", "input"],
        role="owner",
    )
    await channels.grant(
        source_id="gh",
        principal="8",
        conversation_id="github:o/r:1",
        thread_id=handle.thread_id,
        operations=["evidence"],
        role="ci",
    )
    await channels.subscribe_ci(
        subscription_id="ci",
        thread_id=handle.thread_id,
        source_id="gh",
        principal="7",
        conversation_id="github:o/r:1",
        repository="o/r",
        pull_request=1,
        head_sha="sha1",
        check_ids=["5:unit"],
        poll_policy={"max_attempts": 2, "deadline": time.time() + 100, "initial_delay": 5, "max_delay": 10},
        repair_allowance={
            "max_new_turns": 1,
            "deadline": time.time() + 100,
            "max_steps": 3,
            "max_tokens_total": 500,
            "max_cost_micros": 100,
        },
        repair_budget={"max_steps": 3, "max_tokens_total": 500, "max_cost_micros": 100},
    )
    raw = json.dumps(
        {
            "repository": {"full_name": "o/r"},
            "sender": {"id": 8, "type": "Bot"},
            "action": "completed",
            "check_run": {
                "id": 10,
                "name": "unit",
                "app": {"id": 5},
                "head_sha": "sha1",
                "pull_requests": [{"number": 1}],
                "status": "completed",
                "conclusion": "failure",
            },
        }
    ).encode()
    headers = {
        "x-hub-signature-256": "sha256=" + hmac.new(b"secret", raw, hashlib.sha256).hexdigest(),
        "x-github-delivery": "ci-event",
        "x-github-event": "check_run",
    }
    first_receipt = await channels.receive("gh", headers, raw)
    assert await channels.receive("gh", {**headers, "x-github-delivery": "changed-header"}, raw) == first_receipt
    async with await RuntimeStore.open(handle.database) as replay_store:
        from agent_runtime.harness import channel_ops

        assert len(await replay_store.run_async(channel_ops.entries, replay_store, "ci_evidence")) == 1
    info = await channels.ci_status("ci")
    assert info["remote_status"] == "fail"
    assert info["new_turns"] == 1
    async with await RuntimeStore.open(handle.database) as store:
        commands = await store.run_async(host_ops.list_commands, store, handle.thread_id)
        assert len(commands) == 2
        queued = commands[-1]
        assert queued.kind == "start"
        assert queued.request["previous_turn_id"] == handle.turn_id
    await channels.grant(
        source_id="gh",
        principal="7",
        conversation_id="github:o/r:1",
        thread_id=handle.thread_id,
        operations=["status", "input"],
        role="owner",
        enabled=False,
    )
    async with BackgroundHost(handle.database, model=SmallAnswerModel()) as host:
        await host.tick()
        assert (await host.store.run_async(host_ops.read_command, host.store, queued.command_id)).status == "rejected"
    with pytest.raises(PermissionError):
        await channels.ci_status("ci")


@pytest.mark.anyio
async def test_host_automatically_collects_and_delivers_without_repeating(tmp_path, monkeypatch):
    import asyncio

    import httpx

    from agent_runtime import Agent
    from agent_runtime.channels import Channels
    from agent_runtime.channels.providers import ProviderClient, Source
    from agent_runtime.host import BackgroundHost
    from tests.agent.harness.test_subagent_budget import SmallAnswerModel

    monkeypatch.setenv("TOKEN", "token")
    agent = Agent(
        workspace_path=tmp_path, checkpoint_db=tmp_path / "r.db", model_session_path=None, enable_workspace_mcp=False
    )
    handle = await agent.start_background("inspect", autostart=False)
    channels = Channels(database=handle.database, workspace=tmp_path)
    await channels.configure(
        handle.thread_id,
        Source(
            source_id="gh",
            kind="github",
            repository="o/r",
            webhook_secret_env="SECRET",
            outbound_token_env="TOKEN",
            outbound_principal="99",
        ),
    )
    await channels.bind(
        source_id="gh", conversation_id="github:o/r:1", destination="github:o/r:1", thread_id=handle.thread_id
    )
    await channels.grant(
        source_id="gh",
        principal="7",
        conversation_id="github:o/r:1",
        thread_id=handle.thread_id,
        operations=["notify"],
        role="reader",
    )
    sent = []

    async def provider(request):
        sent.append(request)
        return httpx.Response(201, json={"id": 8})

    async with BackgroundHost(
        handle.database,
        model=SmallAnswerModel(),
        channel_client=ProviderClient(transport=httpx.MockTransport(provider)),
    ) as host:
        async with asyncio.timeout(5):
            while not sent:
                await host.tick()
                await asyncio.sleep(0.01)
        await host.tick()
        assert len(sent) == 1


def test_status_query_has_current_canonical_summary_and_regrant_cannot_duplicate_unknown(tmp_path):
    with RuntimeStore.open_sync(tmp_path / "r.db") as store:
        ch, start, grant = channel_setup(store, tmp_path)
        turn = turn_ops.start_turn(
            store, thread_id=start.thread_id, turn_id=start.turn_id, user_message="work", binding_manifest={}
        )
        turn_ops.complete_turn(store, turn_id=turn.turn_id, answer="done")
        receipt = ch.receive(store, envelope(operation="status", text="", event_id="status", native_id="status"))
        assert receipt["summary"]["status"] == "completed"
        assert receipt["summary"]["delivery"] == "unverified"
        query = ch.claim_notification(store, owner="query")
        ch.begin_send(store, query)
        ch.finish_send(store, query, status="acknowledged", remote_id="query-reply")
        note = ch.enqueue_notification(store, grant["grant_id"], turn.turn_id, {"status": "ready"}, now=100)
        claim = ch.claim_notification(store, owner="one", now=100)
        ch.begin_send(store, claim, now=100)
        ch.finish_send(store, claim, status="uncertain", now=101)
        fresh = ch.set_grant(
            store,
            **{k: grant[k] for k in ("source_id", "principal", "conversation_id", "thread_id", "operations", "role")},
            enabled=True,
        )
        repeated = ch.enqueue_notification(store, fresh["grant_id"], turn.turn_id, {"status": "ready"}, now=102)
        assert repeated["outbox_id"] == note["outbox_id"]
        assert ch.claim_notification(store, owner="two", now=103) is None


def test_legacy_readonly_and_schema_fence(tmp_path):
    import sqlite3

    database = tmp_path / "r.db"
    with RuntimeStore.open_sync(database) as store:
        turn_ops.create_thread(store, workspace=tmp_path)
    with sqlite3.connect(database) as conn:
        conn.execute("DROP TABLE channel_entries")
    with RuntimeStore.open_sync(database, read_only=True) as store:
        assert store.verify().valid
    with RuntimeStore.open_sync(database) as store:
        assert store.verify().valid
        assert store.tx.metadata("minimum_writer_schema") == "8"


def test_native_id_conflict_is_not_an_unsigned_header_escape(tmp_path):
    with RuntimeStore.open_sync(tmp_path / "r.db") as store:
        ch, _, _ = channel_setup(store, tmp_path)
        ch.receive(store, envelope())
        with pytest.raises(ValueError, match="conflict"):
            ch.receive(store, envelope(event_id="new-header", digest="modified-signed-body"))


@pytest.mark.anyio
async def test_feishu_native_followup_uses_same_canonical_inbox(tmp_path, monkeypatch):
    import json
    import time

    from agent_runtime.channels import Channels
    from agent_runtime.harness import host_ops
    from tests.agent.test_channel_providers import feishu_source, fs_event, fs_headers

    monkeypatch.setenv("FS_KEY", "key")
    monkeypatch.setenv("FS_VERIFY", "verify")
    database = tmp_path / "r.db"
    async with await RuntimeStore.open(database) as store:
        start = await store.run_async(
            host_ops.receive_command, store, workspace=tmp_path, kind="start", payload={"text": "work"}
        )
    channels = Channels(database=database, workspace=tmp_path)
    await channels.configure(start.thread_id, feishu_source())
    await channels.bind(
        source_id="fs",
        conversation_id="feishu:chat-1:root-1",
        destination="feishu:chat-1:root-1",
        thread_id=start.thread_id,
    )
    await channels.grant(
        source_id="fs",
        principal="ou_42",
        conversation_id="feishu:chat-1:root-1",
        thread_id=start.thread_id,
        operations=["input"],
        role="member",
    )
    raw = json.dumps(fs_event("inspect tests")).encode()
    receipt = await channels.receive("fs", fs_headers(raw, now=int(time.time())), raw)
    assert await channels.receive("fs", fs_headers(raw, now=int(time.time())), raw) == receipt
    async with await RuntimeStore.open(database) as store:
        command = await store.run_async(host_ops.read_command, store, receipt["command_id"])
        assert command.thread_id == start.thread_id and command.kind == "input"
        assert command.request["text"] == "inspect tests"
        assert (await store.run_async(store.verify)).valid


@pytest.mark.anyio
async def test_own_notification_echo_does_not_start_a_turn_but_same_user_can_follow_up(tmp_path, monkeypatch):
    import json

    from agent_runtime.channels import Channels
    from agent_runtime.harness import channel_ops as ch
    from agent_runtime.harness import host_ops
    from tests.agent.test_channel_providers import gh_headers, github_event, github_source

    monkeypatch.setenv("GH_WEBHOOK", "secret")
    channels = Channels(database=tmp_path / "r.db", workspace=tmp_path)
    async with await RuntimeStore.open(channels.database) as store:
        start = await store.run_async(
            host_ops.receive_command, store, workspace=tmp_path, kind="start", payload={"text": "work"}
        )
    await channels.configure(start.thread_id, github_source(outbound_principal="42"))
    await channels.bind(
        source_id="gh",
        conversation_id="github:owner/repo:3",
        destination="github:owner/repo:3",
        thread_id=start.thread_id,
    )
    grant = await channels.grant(
        source_id="gh",
        principal="42",
        conversation_id="github:owner/repo:3",
        thread_id=start.thread_id,
        operations=["input", "notify"],
        role="owner",
    )
    async with await RuntimeStore.open(channels.database) as store:
        row = await store.run_async(
            ch.enqueue_notification, store, grant["grant_id"], start.turn_id, {"status": "ready"}
        )
        claim = await store.run_async(ch.claim_notification, store, owner="sender")
        await store.run_async(ch.begin_send, store, claim)
        await store.run_async(ch.finish_send, store, claim, status="acknowledged", remote_id="99")
    raw = json.dumps(github_event(row["text"] + "\n\n<!-- praxis-delivery:" + row["outbox_id"] + " -->")).encode()
    assert (await channels.receive("gh", gh_headers(raw), raw))["status"] == "ignored_echo"
    payload = github_event("inspect tests")
    payload["comment"]["id"] = 102
    raw = json.dumps(payload).encode()
    assert (await channels.receive("gh", gh_headers(raw, delivery="human-followup"), raw))["status"] == "accepted"
    async with await RuntimeStore.open(channels.database) as store:
        assert len(await store.run_async(host_ops.list_commands, store)) == 2


def test_notification_budget_keeps_concrete_request_without_unbounded_resources():
    from agent_runtime.channels.service import bounded_summary
    from agent_runtime.harness.operation_values import _canonical_json

    summary = {
        "thread_id": "thread",
        "turn_id": "turn",
        "status": "paused",
        "delivery": "blocked",
        "request_id": "request",
        "actions": ["allow_once", "deny"],
        "approval": {
            "tool": "apply_patch",
            "resources": [{"path": "x" * 500} for _ in range(100)],
            "effects": ["file_write"],
        },
    }
    bounded = bounded_summary(summary)
    assert len(_canonical_json(bounded)) <= 4000
    assert bounded["request_id"] == "request"
    assert bounded["approval"]["resource_count"] == 100
    assert bounded["details_omitted"] is True


@pytest.mark.anyio
async def test_channel_lane_failure_cannot_cancel_coding_driver(tmp_path, monkeypatch):
    from agent_runtime import Agent
    from agent_runtime.channels import Channels, service
    from agent_runtime.channels.providers import Source
    from agent_runtime.host import BackgroundHost
    from tests.agent.harness.test_subagent_budget import SmallAnswerModel

    agent = Agent(
        workspace_path=tmp_path, checkpoint_db=tmp_path / "r.db", model_session_path=None, enable_workspace_mcp=False
    )
    handle = await agent.start_background("inspect", autostart=False)
    channels = Channels(database=handle.database, workspace=tmp_path)
    await channels.configure(
        handle.thread_id, Source(source_id="gh", kind="github", repository="o/r", webhook_secret_env="SECRET")
    )

    def channel_failure(*args):
        raise ValueError("oversized channel summary")

    monkeypatch.setattr(service, "collect_notifications", channel_failure)
    async with BackgroundHost(handle.database, model=SmallAnswerModel()) as host:
        await host.serve(idle_timeout=0.01)
        assert (await handle.status())["status"] == "completed"
    diagnostics = (await channels.inspect())["channel_diagnostic"]
    assert diagnostics[0]["reason"] == "ValueError"


def test_queued_canonical_status_is_revalidated_before_send(tmp_path):
    with RuntimeStore.open_sync(tmp_path / "r.db") as store:
        ch, start, grant = channel_setup(store, tmp_path)
        turn = turn_ops.start_turn(
            store, thread_id=start.thread_id, turn_id=start.turn_id, user_message="work", binding_manifest={}
        )
        note = ch.enqueue_notification(
            store,
            grant["grant_id"],
            turn.turn_id,
            {"thread_id": start.thread_id, "turn_id": turn.turn_id, "status": "running", "delivery": "unverified"},
        )
        claim = ch.claim_notification(store, owner="sender")
        turn_ops.complete_turn(store, turn_id=turn.turn_id, answer="done")
        with pytest.raises(PermissionError, match="facts changed"):
            ch.begin_send(store, claim)
        assert ch.get(store, "outbox", note["outbox_id"])["status"] == "claimed"


def test_expired_notification_facts_do_not_prevent_positive_reconciliation(tmp_path):
    with RuntimeStore.open_sync(tmp_path / "r.db") as store:
        ch, start, grant = channel_setup(store, tmp_path)
        turn = turn_ops.start_turn(
            store, thread_id=start.thread_id, turn_id=start.turn_id, user_message="work", binding_manifest={}
        )
        note = ch.enqueue_notification(
            store,
            grant["grant_id"],
            turn.turn_id,
            {"thread_id": start.thread_id, "turn_id": turn.turn_id, "status": "running", "delivery": "unverified"},
        )
        claim = ch.claim_notification(store, owner="sender")
        ch.begin_send(store, claim)
        ch.finish_send(store, claim, status="uncertain")
        turn_ops.complete_turn(store, turn_id=turn.turn_id, answer="done")
        ch.reconcile_notification(store, note["outbox_id"], "provider-positive-id")
        assert ch.get(store, "outbox", note["outbox_id"])["status"] == "acknowledged"
