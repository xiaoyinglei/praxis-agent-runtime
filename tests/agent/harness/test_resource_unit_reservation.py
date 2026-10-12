from __future__ import annotations

import asyncio
import sqlite3
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier

import httpx
import pytest

from agent_runtime.harness.events import RolloutEventReader
from agent_runtime.harness.rollout import RolloutStore


def _turn(database: Path, workspace: Path) -> str:
    with RolloutStore(database) as store:
        thread = store.create_thread(workspace=workspace)
        return store.start_turn(thread_id=thread.thread_id, user_message="read", binding_manifest={}).turn_id


def test_two_connections_atomically_reserve_only_two_units_and_reopen(tmp_path: Path) -> None:
    database = tmp_path / "rollout.sqlite"
    turn_id = _turn(database, tmp_path)
    assert hasattr(RolloutStore, "reserve_resource_unit")
    barrier = Barrier(2)

    def reserve(count: int) -> list[dict[str, object] | str]:
        with RolloutStore(database) as store:
            barrier.wait(timeout=5)
            outcomes: list[dict[str, object] | str] = []
            for _ in range(count):
                try:
                    outcomes.append(store.reserve_resource_unit(turn_id, "network_attempts", 2))
                except ValueError as error:
                    outcomes.append(str(error))
            return outcomes

    with ThreadPoolExecutor(max_workers=2) as workers:
        first = workers.submit(reserve, 2)
        second = workers.submit(reserve, 1)
        outcomes = first.result(timeout=10) + second.result(timeout=10)
    successes = [outcome for outcome in outcomes if isinstance(outcome, dict)]
    assert sorted(outcome["used"] for outcome in successes) == [1, 2]
    assert outcomes.count("resource_limit_exceeded") == 1
    with RolloutStore(database) as reopened:
        usage = reopened.read_resource_usage(turn_id, "network_attempts", initial_used=99)
        assert usage["used"] == 2
        assert usage["remaining"] == 0
        assert reopened.read_item(usage["item_id"]).kind == "resource_usage"
        assert reopened.verify().valid
        with pytest.raises(ValueError, match="^resource_limit_mismatch$"):
            reopened.reserve_resource_unit(turn_id, "network_attempts", 3)


def test_legacy_usage_is_only_applied_to_first_reservation_and_events_are_suppressed(tmp_path: Path) -> None:
    database = tmp_path / "rollout.sqlite"
    turn_id = _turn(database, tmp_path)
    with RolloutStore(database) as store:
        assert store.read_resource_usage(turn_id, "network_attempts", initial_used=1)["used"] == 1
        mutation = store.capture_mutation(
            lambda: store.reserve_resource_unit(turn_id, "network_attempts", 3, initial_used=1)
        )
        assert mutation.value["used"] == 2
        assert RolloutEventReader(store).project_committed_batch(mutation.records) == ()
    with RolloutStore(database) as reopened:
        usage = reopened.reserve_resource_unit(turn_id, "network_attempts", 3, initial_used=200)
        assert usage["used"] == 3
        assert usage["remaining"] == 0
        assert reopened.read_item(usage["item_id"]).payload == {"resource": "network_attempts", "limit": 3, "used": 3}
        assert reopened.verify().valid
    with RolloutStore(database, read_only=True) as reader:
        assert reader.read_resource_usage(turn_id, "network_attempts")["used"] == 3
        with pytest.raises(sqlite3.OperationalError, match="readonly"):
            reader.reserve_resource_unit(turn_id, "network_attempts", 3)


def test_committed_units_survive_process_exit_without_closing_store(tmp_path: Path) -> None:
    database = tmp_path / "crash.sqlite"
    turn_id = _turn(database, tmp_path)
    program = """
import os
import sys
from pathlib import Path
from agent_runtime.harness.rollout import RolloutStore
store = RolloutStore(Path(sys.argv[1]))
store.reserve_resource_unit(sys.argv[2], "network_attempts", 2)
store.reserve_resource_unit(sys.argv[2], "network_attempts", 2)
os._exit(17)
"""
    crashed = subprocess.run([sys.executable, "-c", program, str(database), turn_id], timeout=10, check=False)
    assert crashed.returncode == 17
    with RolloutStore(database) as reopened:
        assert reopened.read_resource_usage(turn_id, "network_attempts")["used"] == 2
        assert reopened.verify().valid


@pytest.mark.parametrize("resource,limit,initial", [
    ("", 2, 0), ("x" * 121, 2, 0), ("network", 0, 0), ("network", 257, 0),
    ("network", True, 0), ("network", 2, -1), ("network", 2, True),
])
def test_invalid_reservations_do_not_append_records(tmp_path: Path, resource: str, limit: int, initial: int) -> None:
    database = tmp_path / "rollout.sqlite"
    turn_id = _turn(database, tmp_path)
    with RolloutStore(database) as store:
        before = store.list_items(turn_id)
        with pytest.raises(ValueError):
            store.reserve_resource_unit(turn_id, resource, limit, initial_used=initial)
        assert store.list_items(turn_id) == before


@pytest.mark.anyio
@pytest.mark.parametrize("legacy_kind", ["new", "running", "skipped_after_retry"])
async def test_two_product_sessions_share_one_turn_request_limit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, legacy_kind: str,
) -> None:
    from agent_runtime import Agent
    from agent_runtime.harness.session import Session
    from agent_runtime.tools import web_http
    from agent_runtime.tools.tool import ToolCall, ToolCallOrigin
    from agent_runtime.tools.web_http import PublicWebClient
    from agent_runtime.tools.web_limits import WebLimits
    from tests.agent.harness.test_public_agent_cutover import PatchThenAnswerModel

    hits: list[httpx.Request] = []
    monkeypatch.setattr(web_http, "PublicWebClient", lambda **kwargs: PublicWebClient(
        **kwargs, transport=httpx.MockTransport(lambda request: hits.append(request) or httpx.Response(
            200, text="read", headers={"content-type": "text/plain"})),
    ))
    database = tmp_path / "sessions.sqlite"
    agent = Agent(workspace_path=tmp_path / "workspace", checkpoint_db=database,
                  enable_workspace_mcp=False, web_limits=WebLimits(turn_requests=2))
    monkeypatch.setattr(agent, "_model_factory", lambda: PatchThenAnswerModel())
    async with await Session.open(agent=agent, allow_web_tools=True) as first:
        async with await Session.open(agent=agent, thread_id=first.thread_id, allow_web_tools=True) as second:
            assert first.store is not second.store
            turn_id = "turn_two_sessions"
            binding = first.binding_provider.snapshot(thread_id=first.thread_id, turn_id=turn_id)
            if legacy_kind == "new":
                binding = {**binding, "web_limits": {"turn_requests": 2}}
            else:
                binding = {key: value for key, value in binding.items() if key != "web_limits"}
            turn = first.store.start_turn(thread_id=first.thread_id, turn_id=turn_id,
                                          user_message="read", binding_manifest=binding)
            if legacy_kind != "new":
                values = dict(turn_id=turn.turn_id, operation_id="legacy-operation", tool_call_id="legacy-call",
                              tool_name="web_fetch", arguments_digest="legacy", execution_revision="legacy",
                              idempotent=True, attempt_count=(1 if legacy_kind == "skipped_after_retry" else 0),
                              error_code=None, requires_reconciliation=False,
                              effects=("network",), resources=())
                first.store.record_tool_execution_state(status="prepared", **values)
                first.store.record_tool_execution_state(status="ready", **values)
                claim = first.store.claim_tool_operation(operation_id="legacy-operation", worker_id="old-worker",
                                                          lease_seconds=300)
                if legacy_kind == "skipped_after_retry":
                    assert first.store.commit_tool_operation_result(
                        turn_id=turn.turn_id, operation_id=claim.operation_id, claim_generation=claim.claim_generation,
                        fencing_token=claim.fencing_token, status="failed", attempt_count=2,
                        error_code="web_request_budget_exceeded", requires_reconciliation=False,
                        result={"tool_call_id": "legacy-call", "tool_name": "web_fetch", "is_error": True,
                                "error_code": "web_request_budget_exceeded", "error_message": "legacy last skipped",
                                "metadata": {"web_request_attempted": False}},
                    )
            calls = [ToolCall(tool_call_id=f"session-call-{index}", tool_name="web_fetch",
                              arguments={"url": f"https://example.com/{index}", "render": False},
                              origin=ToolCallOrigin(request_id="two-sessions", toolset_revision="web",
                                                    exposed_tool_names=("web_fetch",))) for index in range(3)]
            if legacy_kind == "new":
                results = await asyncio.gather(
                    first.tool_orchestrator.execute(turn_id=turn.turn_id, call=calls[0]),
                    second.tool_orchestrator.execute(turn_id=turn.turn_id, call=calls[1]),
                    first.tool_orchestrator.execute(turn_id=turn.turn_id, call=calls[2]),
                )
            else:
                # Preserve pre-upgrade unknown exposure at the first reservation;
                # later Sessions must use that durable count rather than recount.
                initial = await first.tool_orchestrator.execute(turn_id=turn.turn_id, call=calls[0])
                results = [initial, *await asyncio.gather(
                    second.tool_orchestrator.execute(turn_id=turn.turn_id, call=calls[1]),
                    first.tool_orchestrator.execute(turn_id=turn.turn_id, call=calls[2]),
                )]
            assert len(hits) == (2 if legacy_kind == "new" else 3)
            assert sum(result.error_code == "web_request_budget_exceeded" for result in results) == (
                1 if legacy_kind == "new" else 0)
            assert first.store.read_resource_usage(turn.turn_id, "public_web_requests")["used"] == (
                2 if legacy_kind == "new" else 4)
            assert first.store.verify().valid
