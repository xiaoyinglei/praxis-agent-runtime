from pathlib import Path

import pytest

from agent_runtime.harness import Session
from agent_runtime.harness.operation_values import TurnCancellationPendingError
from tests.agent.harness.test_subagent_budget import SmallAnswerModel


@pytest.mark.anyio
async def test_revoked_host_admission_prevents_turn_and_provider_dispatch(tmp_path: Path) -> None:
    def revoked(_store) -> None:
        raise TurnCancellationPendingError("host task was cancelled")

    async with await Session.open(
        database=tmp_path / "rollout.db", workspace=tmp_path, model=SmallAnswerModel(), dispatch_guard=revoked,
    ) as session:
        with pytest.raises(TurnCancellationPendingError):
            await session.submit("must not start", turn_id="reserved_turn")
        assert await session.store.run_async(session.store.list_turns) == ()
        assert await session.store.run_async(session.store.list_model_operations) == ()


@pytest.mark.anyio
async def test_guard_recheck_at_dispatch_persists_cancellation_without_provider_call(tmp_path: Path) -> None:
    calls = 0

    def guard(_store) -> None:
        nonlocal calls
        calls += 1
        if calls > 1:
            raise TurnCancellationPendingError("revoked after turn creation")

    async with await Session.open(
        database=tmp_path / "rollout.db", workspace=tmp_path, model=SmallAnswerModel(), dispatch_guard=guard,
    ) as session:
        result = await session.submit("inspect", turn_id="reserved_turn")
        assert result.status == "failed"
        turn = await session.store.run_async(session.store.read_turn, result.turn_id)
        assert turn.status == "cancelled"
        operations = await session.store.run_async(session.store.list_model_operations)
        assert all(op.status != "completed" for op in operations)


@pytest.mark.anyio
async def test_initial_guard_and_turn_admission_share_the_write_transaction(tmp_path: Path) -> None:
    checks = []

    def guard(store) -> None:
        checks.append(store.persistence._connection.in_transaction)
        assert checks[-1], "guard must be fenced by the Turn admission write transaction"

    async with await Session.open(
        database=tmp_path / "rollout.db", workspace=tmp_path, model=SmallAnswerModel(), dispatch_guard=guard,
    ) as session:
        await session.submit("inspect")
    assert checks
