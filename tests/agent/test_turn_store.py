from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from uuid import UUID

import pytest

from rag.agent.core.context import AgentRunConfig
from rag.agent.core.messages import ModelMessage, canonical_json_text
from rag.agent.core.model_request import (
    canonical_transcript_revision,
    project_transcript_compaction,
)
from rag.agent.loop.state import create_loop_state
from rag.agent.memory.compactor import LoopContextCompactor
from rag.agent.memory.models import MemoryPolicy
from rag.agent.turns import (
    RuntimeBinding,
    TurnStateError,
    TurnStatus,
    TurnStore,
)


def _runtime(workspace: Path, *, model: str = "test-model") -> RuntimeBinding:
    return RuntimeBinding(
        model_alias=model,
        workspace_path=str(workspace.resolve()),
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


def test_followup_requires_terminal_predecessor_and_same_runtime(tmp_path: Path) -> None:
    store = TurnStore(tmp_path / "agent.sqlite")
    runtime = _runtime(tmp_path)
    first = store.begin_turn("first", runtime)

    with pytest.raises(TurnStateError, match="only terminal Turns"):
        store.begin_turn("too early", runtime, previous_turn_id=first.turn_id)

    store.mark_terminal(first.turn_id, TurnStatus.COMPLETED)
    with pytest.raises(TurnStateError, match="runtime does not match"):
        store.begin_turn(
            "different runtime",
            _runtime(tmp_path, model="other-model"),
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
