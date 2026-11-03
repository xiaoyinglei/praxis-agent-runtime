from pathlib import Path

import pytest

from agent_runtime.harness import RuntimeStore, Session, turn_ops
from tests.agent.harness.test_subagent_budget import AcceptAnswer, SmallAnswerModel


def test_steering_receipt_survives_reopen_and_completed_turn(tmp_path: Path) -> None:
    database = tmp_path / "rollout.sqlite3"
    with RuntimeStore.open_sync(database) as store:
        thread = turn_ops.create_thread(store, workspace=tmp_path)
        turn = turn_ops.start_turn(store, thread_id=thread.thread_id, user_message="inspect", binding_manifest={})
        first = turn_ops.receive_turn_input(
            store, thread_id=thread.thread_id, turn_id=turn.turn_id, text="also inspect tests", input_id="input-1"
        )
        turn_ops.complete_turn(store, turn_id=turn.turn_id, answer="checked")
        revision = store.user_input_revision(turn.turn_id)
        record_count = len(store.list_records(thread.thread_id))
    with RuntimeStore.open_sync(database) as store:
        repeated = turn_ops.receive_turn_input(
            store, thread_id=thread.thread_id, turn_id=turn.turn_id, text="also inspect tests", input_id="input-1"
        )
        assert repeated == first
        assert store.user_input_revision(turn.turn_id) == revision
        assert len(store.list_records(thread.thread_id)) == record_count
        with pytest.raises(ValueError, match="receipt"):
            turn_ops.receive_turn_input(
                store, thread_id=thread.thread_id, turn_id=turn.turn_id, text="different work", input_id="input-1"
            )


def test_steering_receipt_cannot_cross_turn_or_change_attachments(tmp_path: Path) -> None:
    with RuntimeStore.open_sync(tmp_path / "rollout.sqlite3") as store:
        thread = turn_ops.create_thread(store, workspace=tmp_path)
        first = turn_ops.start_turn(store, thread_id=thread.thread_id, user_message="inspect", binding_manifest={})
        turn_ops.receive_turn_input(
            store, thread_id=thread.thread_id, turn_id=first.turn_id, text="context", input_id="input-1"
        )
        with pytest.raises(ValueError, match="receipt"):
            turn_ops.receive_turn_input(
                store, thread_id=thread.thread_id, turn_id=first.turn_id, text="context", input_id="input-1",
                input_files=({"path": "different.txt"},),
            )
        turn_ops.complete_turn(store, turn_id=first.turn_id, answer="checked")
        second = turn_ops.start_turn(store, thread_id=thread.thread_id, user_message="next", binding_manifest={})
        original_revision = store.user_input_revision(second.turn_id)
        with pytest.raises(ValueError, match="receipt"):
            turn_ops.receive_turn_input(
                store, thread_id=thread.thread_id, turn_id=second.turn_id, text="context", input_id="input-1"
            )
        assert store.user_input_revision(second.turn_id) == original_revision


@pytest.mark.anyio
async def test_session_accepts_host_preallocated_turn_identity(tmp_path: Path) -> None:
    async with await Session.open(
        database=tmp_path / "rollout.sqlite3", workspace=tmp_path,
        model=SmallAnswerModel(), completion_gate=AcceptAnswer(),
    ) as session:
        result = await session.submit("inspect", turn_id="turn_host_preallocated")
        assert result.turn_id == "turn_host_preallocated"
        assert result.status == "done"
        with pytest.raises(ValueError, match="already"):
            await session.submit("must not execute again", turn_id="turn_host_preallocated")


@pytest.mark.anyio
@pytest.mark.parametrize("entry", ["session", "agent"])
async def test_public_steering_receipt_replays_after_completion_without_model_io(tmp_path: Path, entry: str):
    from agent_runtime import Agent

    database = tmp_path / "rollout.sqlite3"
    async with await Session.open(
        database=database, workspace=tmp_path, model=SmallAnswerModel(), completion_gate=AcceptAnswer()
    ) as session:
        turn = await session.store.run_async(
            lambda: turn_ops.start_turn(
                session.store, thread_id=session.thread_id, user_message="start", binding_manifest={}
            )
        )
        public = session if entry == "session" else Agent(
            workspace_path=tmp_path, checkpoint_db=database, model_session_path=None
        )
        receipt = await public.steer("inspect tests", expected_turn_id=turn.turn_id, input_id="public-input")
        await session.store.run_async(
            lambda: turn_ops.complete_turn(session.store, turn_id=turn.turn_id, answer="checked")
        )
        before = await session.store.run_async(session.store.list_records, session.thread_id)
        repeated = await public.steer("inspect tests", expected_turn_id=turn.turn_id, input_id="public-input")
        assert repeated == receipt
        assert await session.store.run_async(session.store.list_records, session.thread_id) == before
        with pytest.raises(ValueError, match="receipt"):
            await public.steer("different", expected_turn_id=turn.turn_id, input_id="public-input")
        with pytest.raises(RuntimeError):
            await public.steer("new input after completion", expected_turn_id=turn.turn_id, input_id="new-input")


@pytest.mark.anyio
async def test_attachment_receipt_replay_does_not_reread_or_regrant_file(tmp_path: Path):
    from agent_runtime import Agent

    database = tmp_path / "rollout.sqlite3"
    attachment = tmp_path / "context.txt"
    attachment.write_text("original evidence", encoding="utf-8")
    agent = Agent(workspace_path=tmp_path, checkpoint_db=database, model_session_path=None)
    agent._model_factory = SmallAnswerModel
    async with agent.session() as session:
        turn = await session.store.run_async(
            lambda: turn_ops.start_turn(
                session.store, thread_id=session.thread_id, user_message="start", binding_manifest={}
            )
        )
        receipt = await session.steer(
            "inspect attached context", expected_turn_id=turn.turn_id,
            files=["context.txt"], input_id="attachment-input",
        )
        before = await session.store.run_async(session.store.list_records, session.thread_id)
        repeated = await session.steer(
            "inspect attached context", expected_turn_id=turn.turn_id,
            files=["context.txt"], input_id="attachment-input",
        )
        assert repeated == receipt
        assert await session.store.run_async(session.store.list_records, session.thread_id) == before
        attachment.unlink()
        assert await session.steer(
            "inspect attached context", expected_turn_id=turn.turn_id,
            files=["context.txt"], input_id="attachment-input",
        ) == receipt
        with pytest.raises(ValueError, match="receipt"):
            await session.steer(
                "inspect attached context", expected_turn_id=turn.turn_id,
                files=["different.txt"], input_id="attachment-input",
            )
