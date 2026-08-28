from __future__ import annotations

import json
import sqlite3
from dataclasses import replace
from pathlib import Path
from uuid import UUID

import pytest

from agent_runtime.core.context import AgentRunConfig
from agent_runtime.core.messages import ModelMessage, canonical_json_text
from agent_runtime.core.model_request import (
    canonical_transcript_revision,
    project_transcript_compaction,
)
from agent_runtime.knowledge import RAGKnowledgeConfig
from agent_runtime.loop.state import create_loop_state
from agent_runtime.memory.compactor import LoopContextCompactor
from agent_runtime.memory.models import MemoryPolicy
from agent_runtime.streaming.events import (
    EventType,
    ItemStatus,
    TurnItemKind,
    item_completed,
)
from agent_runtime.turns import (
    RuntimeBinding,
    TurnStateError,
    TurnStatus,
    TurnStore,
)


def _runtime(
    workspace: Path,
    *,
    model: str = "test-model",
    knowledge: RAGKnowledgeConfig | None = None,
) -> RuntimeBinding:
    return RuntimeBinding(
        model_alias=model,
        workspace_path=str(workspace.resolve()),
        knowledge=knowledge,
    )


def test_turn_store_links_followups_without_session(tmp_path: Path) -> None:
    store = TurnStore(tmp_path / "agent.sqlite")
    runtime = _runtime(tmp_path)
    first = store.begin_turn("remember alpha", runtime)
    store.sync_turn_messages(
        first.turn_id,
        [
            ModelMessage(role="user", content="remember alpha"),
            ModelMessage(role="assistant", content="alpha"),
        ],
    )
    store.mark_terminal(first.turn_id, TurnStatus.COMPLETED)
    second = store.begin_turn(
        "what did I say?",
        runtime,
        previous_turn_id=first.turn_id,
    )

    assert str(UUID(first.turn_id)) == first.turn_id
    assert first.previous_turn_id is None
    assert second.previous_turn_id == first.turn_id
    assert store.history_before_turn(second.turn_id) == (
        ModelMessage(role="user", content="remember alpha"),
        ModelMessage(role="assistant", content="alpha"),
    )
    assert store.turn_history(second.turn_id) == (
        ModelMessage(role="user", content="what did I say?"),
    )
    store.close()


def test_followup_requires_terminal_predecessor(tmp_path: Path) -> None:
    store = TurnStore(tmp_path / "agent.sqlite")
    runtime = _runtime(tmp_path)
    first = store.begin_turn("first", runtime)

    with pytest.raises(TurnStateError, match="only terminal Turns"):
        store.begin_turn("too early", runtime, previous_turn_id=first.turn_id)

    store.close()


def test_followup_may_bind_a_different_model_alias(tmp_path: Path) -> None:
    store = TurnStore(tmp_path / "agent.sqlite")
    first = store.begin_turn("first", _runtime(tmp_path, model="model-a"))
    store.mark_terminal(first.turn_id, TurnStatus.COMPLETED)

    second = store.begin_turn(
        "use another model",
        _runtime(tmp_path, model="model-b"),
        previous_turn_id=first.turn_id,
    )

    assert second.previous_turn_id == first.turn_id
    assert store.get_turn(first.turn_id).runtime.model_alias == "model-a"
    assert store.get_turn(second.turn_id).runtime.model_alias == "model-b"
    store.close()


@pytest.mark.parametrize("resource", ["workspace", "knowledge"])
def test_followup_rejects_non_model_runtime_changes(
    tmp_path: Path,
    resource: str,
) -> None:
    knowledge = RAGKnowledgeConfig(
        storage_root=tmp_path / "knowledge-a",
        vector_backend="sqlite",
    )
    store = TurnStore(tmp_path / "agent.sqlite")
    first = store.begin_turn(
        "first",
        _runtime(tmp_path, model="model-a", knowledge=knowledge),
    )
    store.mark_terminal(first.turn_id, TurnStatus.COMPLETED)
    workspace = tmp_path if resource == "knowledge" else tmp_path / "other-workspace"
    followup_knowledge = (
        RAGKnowledgeConfig(
            storage_root=tmp_path / "knowledge-b",
            vector_backend="sqlite",
        )
        if resource == "knowledge"
        else knowledge
    )

    with pytest.raises(TurnStateError, match="runtime does not match"):
        store.begin_turn(
            "different runtime",
            _runtime(
                workspace,
                model="model-b",
                knowledge=followup_knowledge,
            ),
            previous_turn_id=first.turn_id,
        )
    store.close()


def test_turn_transcript_sync_is_append_only(tmp_path: Path) -> None:
    store = TurnStore(tmp_path / "agent.sqlite")
    turn = store.begin_turn("hello", _runtime(tmp_path))
    transcript = [
        ModelMessage(role="user", content="hello"),
        ModelMessage(role="assistant", content="hi"),
    ]
    store.sync_turn_messages(turn.turn_id, transcript)
    store.sync_turn_messages(turn.turn_id, transcript)

    with pytest.raises(RuntimeError, match="canonical history conflict"):
        store.sync_turn_messages(
            turn.turn_id,
            [
                ModelMessage(role="user", content="changed"),
                ModelMessage(role="assistant", content="hi"),
            ],
        )
    assert store.turn_history(turn.turn_id) == tuple(transcript)
    store.close()


def _compacted_turn_messages(
    turn_id: str,
    user_message: str,
) -> tuple[list[ModelMessage], list[ModelMessage]]:
    policy = _compaction_policy()
    state = create_loop_state(
        current_message=user_message,
        run_config=AgentRunConfig(
            turn_id=turn_id,
            llm_budget_total=10_000,
            memory_policy=policy,
        ),
    )
    original = [
        ModelMessage(role="user", content=user_message),
        *[
            ModelMessage(
                role="assistant" if index % 2 == 0 else "user",
                content=f"persisted-{index}: " + (f"token-{index} " * 500),
            )
            for index in range(6)
        ],
    ]
    state["turn_transcript"] = list(original)
    result = LoopContextCompactor().reactive_compact(state)
    assert result.changed is True
    return original, list(state["turn_transcript"])


def _compaction_policy() -> MemoryPolicy:
    return MemoryPolicy(
        message_compaction_min_count=99,
        reactive_compact_tail_count=2,
    )


def test_turn_store_accepts_verified_canonical_compaction_rewrite(
    tmp_path: Path,
) -> None:
    store = TurnStore(tmp_path / "agent.sqlite")
    turn = store.begin_turn("compact me", _runtime(tmp_path))
    original, compacted = _compacted_turn_messages(
        turn.turn_id,
        "compact me",
    )
    store.sync_turn_messages(turn.turn_id, original)
    candidate = [
        *compacted,
        ModelMessage(role="assistant", content="continued after compaction"),
    ]

    store.sync_turn_messages(
        turn.turn_id,
        candidate,
        compaction_policy=_compaction_policy(),
    )

    assert store.turn_history(turn.turn_id) == tuple(candidate)
    store.close()


def test_turn_store_requires_trusted_policy_for_compaction_rewrite(
    tmp_path: Path,
) -> None:
    store = TurnStore(tmp_path / "agent-missing-policy.sqlite")
    turn = store.begin_turn("compact with authority", _runtime(tmp_path))
    original, compacted = _compacted_turn_messages(
        turn.turn_id,
        "compact with authority",
    )
    store.sync_turn_messages(turn.turn_id, original)

    with pytest.raises(RuntimeError, match="canonical history conflict"):
        store.sync_turn_messages(turn.turn_id, compacted)

    assert store.turn_history(turn.turn_id) == tuple(original)
    store.close()


@pytest.mark.parametrize(
    "tamper",
    (
        "summary",
        "digest",
        "parent_revision",
        "summary_limit",
        "covered_count",
        "suffix",
    ),
)
def test_turn_store_rejects_tampered_compaction_rewrite(
    tmp_path: Path,
    tamper: str,
) -> None:
    store = TurnStore(tmp_path / f"agent-{tamper}.sqlite")
    turn = store.begin_turn("compact safely", _runtime(tmp_path))
    original, compacted = _compacted_turn_messages(
        turn.turn_id,
        "compact safely",
    )
    store.sync_turn_messages(turn.turn_id, original)
    candidate = list(compacted)
    event = json.loads(candidate[1].content)
    if tamper == "summary":
        event["payload"]["summary"] = "forged summary"
        candidate[1] = replace(
            candidate[1],
            content=canonical_json_text(event),
        )
    elif tamper == "digest":
        event["payload"]["projection"]["source_digest"] = "0" * 64
        candidate[1] = replace(
            candidate[1],
            content=canonical_json_text(event),
        )
    elif tamper == "parent_revision":
        event["payload"]["parent_context_revision"] = "forged-parent"
        candidate[1] = replace(
            candidate[1],
            content=canonical_json_text(event),
        )
    elif tamper == "summary_limit":
        event["payload"]["projection"]["summary_max_chars"] = 1
        candidate[1] = replace(
            candidate[1],
            content=canonical_json_text(event),
        )
    elif tamper == "covered_count":
        event["payload"]["projection"]["covered_count"] = 1
        candidate[1] = replace(
            candidate[1],
            content=canonical_json_text(event),
        )
    else:
        candidate[2] = replace(
            candidate[2],
            content="forged retained suffix",
        )

    with pytest.raises(RuntimeError, match="canonical history conflict"):
        store.sync_turn_messages(
            turn.turn_id,
            candidate,
            compaction_policy=_compaction_policy(),
        )

    assert store.turn_history(turn.turn_id) == tuple(original)
    store.close()


def test_turn_store_rejects_self_consistent_projection_outside_runtime_policy(
    tmp_path: Path,
) -> None:
    store = TurnStore(tmp_path / "agent-untrusted-policy.sqlite")
    turn = store.begin_turn("compact by policy", _runtime(tmp_path))
    original, _ = _compacted_turn_messages(
        turn.turn_id,
        "compact by policy",
    )
    store.sync_turn_messages(turn.turn_id, original)
    forged_projection = project_transcript_compaction(
        original[1:],
        parent_context_revision=canonical_transcript_revision(original),
        tail_start=len(original) - 1,
        max_summary_chars=1,
    )
    candidate = [original[0], *forged_projection]

    with pytest.raises(RuntimeError, match="canonical history conflict"):
        store.sync_turn_messages(
            turn.turn_id,
            candidate,
            compaction_policy=_compaction_policy(),
        )

    assert store.turn_history(turn.turn_id) == tuple(original)
    store.close()


def test_turn_store_persists_runtime_lineage_and_messages(tmp_path: Path) -> None:
    database = tmp_path / "agent.sqlite"
    runtime = _runtime(tmp_path)
    first_store = TurnStore(database)
    first = first_store.begin_turn("first", runtime)
    first_store.sync_turn_messages(
        first.turn_id,
        [
            ModelMessage(role="user", content="first"),
            ModelMessage(role="assistant", content="answer"),
        ],
    )
    first_store.mark_terminal(first.turn_id, TurnStatus.COMPLETED)
    second = first_store.begin_turn("second", runtime, previous_turn_id=first.turn_id)
    first_store.close()

    restored = TurnStore(database)
    assert restored.get_turn(second.turn_id).previous_turn_id == first.turn_id
    assert restored.get_turn(first.turn_id).runtime == runtime
    assert restored.history_before_turn(second.turn_id)[-1].content == "answer"
    restored.close()


def test_turn_store_replays_one_shared_durable_order_across_items_and_lifecycle(
    tmp_path: Path,
) -> None:
    database = tmp_path / "agent.sqlite"
    store = TurnStore(database)
    turn = store.begin_turn("start", _runtime(tmp_path))
    commit_item = getattr(store, "commit_completed_item", None)
    replay_turn = getattr(store, "replay_turn_events", None)
    assert commit_item is not None
    assert replay_turn is not None

    first_item = item_completed(
        turn_id=turn.turn_id,
        item_id="agent:first",
        item_kind=TurnItemKind.AGENT_MESSAGE,
        status=ItemStatus.SUCCESS,
        iteration=1,
        data={"content": "first", "tool_calls": []},
    )
    first_ordinal = commit_item(
        first_item,
        message=ModelMessage(role="assistant", content="first"),
        started_at_ms=first_item.timestamp_ms - 5,
    )
    store.mark_paused(turn.turn_id)
    store.claim_for_resume(
        turn.turn_id,
        lease_owner="worker-2",
        lease_seconds=30,
    )
    second_item = item_completed(
        turn_id=turn.turn_id,
        item_id="agent:second",
        item_kind=TurnItemKind.AGENT_MESSAGE,
        status=ItemStatus.SUCCESS,
        iteration=2,
        data={"content": "second", "tool_calls": []},
    )
    second_ordinal = commit_item(
        second_item,
        message=ModelMessage(role="assistant", content="second"),
        started_at_ms=second_item.timestamp_ms - 5,
    )
    store.mark_terminal(turn.turn_id, TurnStatus.COMPLETED)

    replayed = replay_turn(turn.turn_id)
    assert first_ordinal == 1
    assert second_ordinal == 4
    assert [record.durable_ordinal for record in replayed] == list(range(6))
    assert [record.event.type for record in replayed] == [
        EventType.TURN_STARTED,
        EventType.ITEM_COMPLETED,
        EventType.TURN_PAUSED,
        EventType.TURN_RESUMED,
        EventType.ITEM_COMPLETED,
        EventType.TURN_COMPLETED,
    ]
    assert store.turn_history(turn.turn_id) == (
        ModelMessage(role="user", content="start"),
        ModelMessage(role="assistant", content="first"),
        ModelMessage(role="assistant", content="second"),
    )
    store.close()

    reopened = TurnStore(database)
    assert [
        record.event.type for record in reopened.replay_turn_events(turn.turn_id)
    ] == [record.event.type for record in replayed]
    reopened.close()


def test_recoverable_interruption_replays_pause_then_resume(tmp_path: Path) -> None:
    store = TurnStore(tmp_path / "agent.sqlite")
    turn = store.begin_turn("resume me", _runtime(tmp_path))

    store.mark_interrupted(turn.turn_id, reason="worker_lost")
    store.claim_for_resume(
        turn.turn_id,
        lease_owner="worker-2",
        lease_seconds=30,
    )

    replayed = store.replay_turn_events(turn.turn_id)
    assert [record.event.type for record in replayed] == [
        EventType.TURN_STARTED,
        EventType.TURN_PAUSED,
        EventType.TURN_RESUMED,
    ]
    assert replayed[1].event.data == {
        "status": "interrupted",
        "reason": "worker_lost",
    }
    store.close()


def test_completed_item_commit_is_idempotent_but_divergence_fails_closed(
    tmp_path: Path,
) -> None:
    store = TurnStore(tmp_path / "agent.sqlite")
    turn = store.begin_turn("idempotent", _runtime(tmp_path))
    commit_item = getattr(store, "commit_completed_item", None)
    assert commit_item is not None
    event = item_completed(
        turn_id=turn.turn_id,
        item_id="tool:one",
        item_kind=TurnItemKind.TOOL,
        status=ItemStatus.SUCCESS,
        data={"result": "ok"},
    )

    first = commit_item(event, started_at_ms=10)
    same_delivery_with_new_timestamp = item_completed(
        turn_id=turn.turn_id,
        item_id="tool:one",
        item_kind=TurnItemKind.TOOL,
        status=ItemStatus.SUCCESS,
        data={"result": "ok"},
    )
    assert commit_item(same_delivery_with_new_timestamp, started_at_ms=11) == first

    divergent = item_completed(
        turn_id=turn.turn_id,
        item_id="tool:one",
        item_kind=TurnItemKind.TOOL,
        status=ItemStatus.SUCCESS,
        data={"result": "different"},
    )
    with pytest.raises(RuntimeError, match="completed item conflict"):
        commit_item(divergent, started_at_ms=10)
    assert len(store.replay_turn_events(turn.turn_id)) == 2
    store.close()


def test_new_item_cannot_be_appended_after_terminal_turn(tmp_path: Path) -> None:
    store = TurnStore(tmp_path / "agent.sqlite")
    turn = store.begin_turn("terminal", _runtime(tmp_path))
    store.mark_terminal(turn.turn_id, TurnStatus.COMPLETED)

    with pytest.raises(TurnStateError, match="cannot append"):
        store.commit_completed_item(
            item_completed(
                turn_id=turn.turn_id,
                item_id="agent:late",
                item_kind=TurnItemKind.AGENT_MESSAGE,
                status=ItemStatus.SUCCESS,
                data={"content": "late", "tool_calls": []},
            )
        )

    assert [record.event.type for record in store.replay_turn_events(turn.turn_id)] == [
        EventType.TURN_STARTED,
        EventType.TURN_COMPLETED,
    ]
    store.close()


def test_pre_v2_turn_database_is_transactionally_projected_for_replay(
    tmp_path: Path,
) -> None:
    database = tmp_path / "legacy.sqlite"
    connection = sqlite3.connect(database)
    connection.executescript(
        """
        CREATE TABLE agent_turns (
            turn_id TEXT PRIMARY KEY,
            previous_turn_id TEXT,
            status TEXT NOT NULL,
            user_message TEXT NOT NULL,
            runtime_json TEXT NOT NULL,
            lease_owner TEXT,
            lease_expires_at REAL,
            created_at REAL NOT NULL,
            updated_at REAL NOT NULL
        );
        CREATE TABLE agent_turn_messages (
            turn_id TEXT NOT NULL,
            message_index INTEGER NOT NULL,
            payload_json TEXT NOT NULL,
            PRIMARY KEY(turn_id, message_index)
        );
        """
    )
    turn_id = "00000000-0000-4000-8000-000000000001"
    connection.execute(
        """
        INSERT INTO agent_turns VALUES (?, NULL, 'completed', ?, ?, NULL, NULL, 1.0, 2.0)
        """,
        (turn_id, "legacy user", _runtime(tmp_path).model_dump_json()),
    )
    for index, message in enumerate(
        (
            {"role": "user", "content": "legacy user", "tool_calls": []},
            {"role": "assistant", "content": "legacy answer", "tool_calls": []},
        )
    ):
        connection.execute(
            "INSERT INTO agent_turn_messages VALUES (?, ?, ?)",
            (turn_id, index, json.dumps(message, separators=(",", ":"))),
        )
    connection.commit()
    connection.close()

    store = TurnStore(database)
    replayed = store.replay_turn_events(turn_id)
    assert [record.durable_ordinal for record in replayed] == [0, 1, 2, 3]
    assert [record.event.type for record in replayed] == [
        EventType.TURN_STARTED,
        EventType.ITEM_COMPLETED,
        EventType.ITEM_COMPLETED,
        EventType.TURN_COMPLETED,
    ]
    assert all(
        record.event.item_kind is TurnItemKind.LEGACY_MESSAGE
        for record in replayed[1:3]
    )
    assert store.get_turn(turn_id).status is TurnStatus.COMPLETED
    store.close()


def test_turn_status_and_resume_lease_lifecycle(tmp_path: Path) -> None:
    store = TurnStore(tmp_path / "agent.sqlite")
    turn = store.begin_turn(
        "approve",
        _runtime(tmp_path),
        lease_owner="owner-a",
        lease_seconds=1,
    )
    paused = store.mark_paused(turn.turn_id)
    assert paused.status is TurnStatus.PAUSED
    claimed = store.claim_for_resume(
        turn.turn_id,
        lease_owner="owner-b",
        lease_seconds=10,
    )
    assert claimed.status is TurnStatus.RUNNING
    assert claimed.lease_owner == "owner-b"
    renewed = store.renew_lease(
        turn.turn_id,
        lease_owner="owner-b",
        lease_seconds=20,
    )
    assert renewed.lease_expires_at is not None
    completed = store.mark_terminal(turn.turn_id, TurnStatus.COMPLETED)
    assert completed.status is TurnStatus.COMPLETED
    assert completed.lease_owner is None
    store.close()


def test_prepare_resume_normalizes_expired_running_turn(tmp_path: Path) -> None:
    store = TurnStore(tmp_path / "agent.sqlite")
    turn = store.begin_turn(
        "interrupted",
        _runtime(tmp_path),
        lease_owner="dead-worker",
        lease_seconds=1,
    )
    prepared = store.prepare_turn_for_resume(
        turn.turn_id,
        now=(turn.lease_expires_at or 0) + 1,
    )
    assert prepared.status is TurnStatus.INTERRUPTED
    assert prepared.lease_owner is None
    store.close()


def test_latest_turn_queries_use_turn_runtime_not_session_join(tmp_path: Path) -> None:
    store = TurnStore(tmp_path / "agent.sqlite")
    runtime = _runtime(tmp_path)
    completed = store.begin_turn("done", runtime)
    store.mark_terminal(completed.turn_id, TurnStatus.COMPLETED)
    paused = store.begin_turn("paused", runtime)
    store.mark_paused(paused.turn_id)

    assert store.latest_turn(workspace_path=tmp_path).turn_id == completed.turn_id
    assert store.latest_resumable_turn(workspace_path=tmp_path).turn_id == paused.turn_id
    assert [item.turn_id for item in store.list_turns(workspace_path=tmp_path)] == [
        paused.turn_id,
        completed.turn_id,
    ]
    store.close()
