from __future__ import annotations

import asyncio
import time
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Event

import pytest

import agent_runtime.harness as harness
from agent_runtime.harness import RolloutStore, ToolOrchestrator
from agent_runtime.tools.permissions import ToolExecutionContext
from agent_runtime.tools.registry import ToolRegistry
from agent_runtime.tools.tool import (
    CancellationMode,
    InterruptBehavior,
    JsonValue,
    NormalizedToolOutput,
    ResolvedToolUse,
    Tool,
    ToolCall,
    ToolCallOrigin,
    ToolContentBlock,
    ToolDefinition,
    ToolEffect,
    ToolResult,
    ToolTarget,
    json_schema_input,
)


def _ready_operation(store: RolloutStore, turn_id: str) -> str:
    values = {
        "turn_id": turn_id,
        "operation_id": "toolop-1",
        "tool_call_id": "call-1",
        "tool_name": "write_file",
        "arguments_digest": "args-v1",
        "execution_revision": "write-file-v1",
        "idempotent": False,
        "attempt_count": 0,
        "error_code": None,
        "requires_reconciliation": False,
    }
    store.record_tool_execution_state(status="prepared", **values)
    store.record_tool_execution_state(status="ready", **values)
    return "toolop-1"


def test_expired_claim_fences_late_worker_result(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    with RolloutStore(tmp_path / "rollout.sqlite3") as store:
        thread = store.create_thread(workspace=workspace)
        turn = store.start_turn(
            thread_id=thread.thread_id,
            user_message="write once",
            binding_manifest={"model_alias": "test-model"},
        )
        operation_id = _ready_operation(store, turn.turn_id)
        claim = store.claim_tool_operation(
            operation_id=operation_id,
            worker_id="worker-a",
            now=100.0,
            lease_seconds=5.0,
        )

        store.expire_tool_operation_claim(operation_id=operation_id, now=106.0)
        accepted = store.commit_tool_operation_outcome(
            operation_id=operation_id,
            claim_generation=claim.claim_generation,
            fencing_token=claim.fencing_token,
            status="succeeded",
            attempt_count=1,
            error_code=None,
            requires_reconciliation=False,
        )

        assert accepted is False
        operation = store.read_tool_operation(operation_id)
        assert operation.status == "unknown"
        assert operation.claim_generation == 1
        assert operation.claim_owner == "worker-a"
        assert any(
            record.record_type == "tool_operation_stale_result"
            for record in store.list_records(thread.thread_id)
        )
        assert store.verify().valid is True


def test_conflicting_resource_claims_are_serialized_across_threads(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    database = tmp_path / "rollout.sqlite3"
    target = workspace / "shared.txt"
    with RolloutStore(database) as store:
        turns = [
            store.start_turn(
                thread_id=store.create_thread(workspace=workspace).thread_id,
                user_message=f"writer {index}",
                binding_manifest={"model_alias": "test-model"},
            )
            for index in range(2)
        ]
        for index, turn in enumerate(turns):
            values = {
                "turn_id": turn.turn_id,
                "operation_id": f"write-op-{index}",
                "tool_call_id": f"write-call-{index}",
                "tool_name": "write_file",
                "arguments_digest": f"args-{index}",
                "execution_revision": "write-file-v1",
                "idempotent": False,
                "attempt_count": 0,
                "error_code": None,
                "requires_reconciliation": False,
                "effects": ("write_workspace",),
                "resources": (
                    {
                        "kind": "filesystem",
                        "identity": str(target.resolve()),
                        "access": "write",
                    },
                ),
            }
            store.record_tool_execution_state(status="prepared", **values)
            store.record_tool_execution_state(status="ready", **values)

        first = store.claim_tool_operation(
            operation_id="write-op-0",
            worker_id="worker-a",
            lease_seconds=30.0,
        )
        with pytest.raises(RuntimeError, match="resource claim conflict"):
            store.claim_tool_operation(
                operation_id="write-op-1",
                worker_id="worker-b",
                lease_seconds=30.0,
            )

        assert store.commit_tool_operation_outcome(
            operation_id="write-op-0",
            claim_generation=first.claim_generation,
            fencing_token=first.fencing_token,
            status="succeeded",
            attempt_count=1,
            error_code=None,
            requires_reconciliation=False,
        )
        second = store.claim_tool_operation(
            operation_id="write-op-1",
            worker_id="worker-b",
            lease_seconds=30.0,
        )

        assert second.status == "running"
        assert store.verify().valid is True


def test_compatible_read_claims_can_overlap_on_the_same_resource(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    target = workspace / "shared.txt"
    with RolloutStore(tmp_path / "rollout.sqlite3") as store:
        operation_ids: list[str] = []
        for index in range(2):
            turn = store.start_turn(
                thread_id=store.create_thread(workspace=workspace).thread_id,
                user_message=f"reader {index}",
                binding_manifest={"model_alias": "test-model"},
            )
            operation_id = f"read-op-{index}"
            operation_ids.append(operation_id)
            values = {
                "turn_id": turn.turn_id,
                "operation_id": operation_id,
                "tool_call_id": f"read-call-{index}",
                "tool_name": "read_file",
                "arguments_digest": f"args-{index}",
                "execution_revision": "read-file-v1",
                "idempotent": True,
                "attempt_count": 0,
                "error_code": None,
                "requires_reconciliation": False,
                "effects": ("read_workspace",),
                "resources": (
                    {
                        "kind": "filesystem",
                        "identity": str(target.resolve()),
                        "access": "read",
                    },
                ),
            }
            store.record_tool_execution_state(status="prepared", **values)
            store.record_tool_execution_state(status="ready", **values)

        claims = [
            store.claim_tool_operation(
                operation_id=operation_id,
                worker_id=f"worker-{index}",
                lease_seconds=30.0,
            )
            for index, operation_id in enumerate(operation_ids)
        ]

        assert [claim.status for claim in claims] == ["running", "running"]
        assert store.verify().valid is True


def test_conflicting_orchestrator_never_enters_the_second_runner(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    database = tmp_path / "rollout.sqlite3"
    target = workspace / "shared.txt"
    started = Event()
    release = Event()
    runner_calls: list[str] = []
    schema: Mapping[str, JsonValue] = {
        "type": "object",
        "properties": {"owner": {"type": "string"}},
        "required": ("owner",),
        "additionalProperties": False,
    }

    def run(arguments: Mapping[str, JsonValue]) -> dict[str, str]:
        owner = str(arguments["owner"])
        runner_calls.append(owner)
        if owner == "first":
            started.set()
            assert release.wait(timeout=5.0)
        return {"owner": owner}

    tool = Tool(
        definition=ToolDefinition(
            name="write_shared",
            description="Write the same canonical resource.",
            input_schema=schema,
        ),
        validate_input=json_schema_input(schema),
        run=run,
        normalize_output=lambda raw: NormalizedToolOutput(
            content=(ToolContentBlock(type="text", data=raw),),
        ),
        output_schema=None,
        static_effects=frozenset({ToolEffect.WRITE_WORKSPACE}),
        resolve_use=lambda _arguments: ResolvedToolUse(
            effects=frozenset({ToolEffect.WRITE_WORKSPACE}),
            targets=(
                ToolTarget(
                    kind="workspace_path",
                    value=str(target),
                ),
            ),
        ),
        execution_revision="write-shared-v1",
        idempotent=False,
        concurrency_safe=False,
        cancellation_mode=CancellationMode.COOPERATIVE,
        interrupt_behavior=InterruptBehavior.CANCEL,
        timeout_seconds=30.0,
        max_model_output_bytes=4_096,
    )
    registry = ToolRegistry()
    registry.register(tool)
    tools = registry.freeze()
    with RolloutStore(database) as setup:
        turns = [
            setup.start_turn(
                thread_id=setup.create_thread(workspace=workspace).thread_id,
                user_message=owner,
                binding_manifest={"model_alias": "test-model"},
            )
            for owner in ("first", "second")
        ]

    def execute(turn_id: str, owner: str) -> ToolResult:
        with RolloutStore(database) as store:
            return asyncio.run(
                ToolOrchestrator(
                    store=store,
                    tools=tools,
                    execution_context=ToolExecutionContext(
                        workspace_root=workspace,
                        cwd=workspace,
                        allow_write_tools=True,
                    ),
                    worker_id=f"worker-{owner}",
                ).execute(
                    turn_id=turn_id,
                    call=ToolCall(
                        tool_call_id=f"call-{owner}",
                        tool_name="write_shared",
                        arguments={"owner": owner},
                        origin=ToolCallOrigin(
                            request_id=f"request-{owner}",
                            toolset_revision="toolset-v1",
                            exposed_tool_names=("write_shared",),
                        ),
                    ),
                )
            )

    with ThreadPoolExecutor(max_workers=1) as pool:
        first_future = pool.submit(execute, turns[0].turn_id, "first")
        assert started.wait(timeout=5.0)
        second = execute(turns[1].turn_id, "second")
        release.set()
        first = first_future.result(timeout=5.0)

    assert first.is_error is False
    assert second.is_error is True
    assert second.error_code == "resource_busy"
    assert runner_calls == ["first"]
    with RolloutStore(database) as verifier:
        operations = verifier.list_tool_operations()
        assert sorted(operation.status for operation in operations) == [
            "failed",
            "succeeded",
        ]
        assert verifier.verify().valid is True


def test_fresh_store_recovers_expired_running_operation_to_unknown_pause(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    database = tmp_path / "rollout.sqlite3"
    with RolloutStore(database) as first:
        thread = first.create_thread(workspace=workspace)
        turn = first.start_turn(
            thread_id=thread.thread_id,
            user_message="write once",
            binding_manifest={"model_alias": "test-model"},
        )
        operation_id = _ready_operation(first, turn.turn_id)
        first.claim_tool_operation(
            operation_id=operation_id,
            worker_id="crashed-worker",
            now=100.0,
            lease_seconds=5.0,
        )

    with RolloutStore(database) as restarted:
        recovered = restarted.expire_tool_operation_claim(
            operation_id=operation_id,
            now=106.0,
        )

        assert recovered.status == "unknown"
        assert recovered.requires_reconciliation is True
        assert restarted.read_turn(turn.turn_id).status == "paused"
        assert restarted.read_thread(thread.thread_id).active_turn_id == turn.turn_id
        [interaction] = restarted.list_interactions(turn.turn_id)
        assert interaction.kind == "tool_reconciliation"
        assert interaction.status == "pending"
        assert interaction.operation_id == operation_id
        assert restarted.verify().valid is True


def test_trusted_reconciler_commits_unknown_success_without_replaying_runner(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    with RolloutStore(tmp_path / "rollout.sqlite3") as store:
        thread = store.create_thread(workspace=workspace)
        turn = store.start_turn(
            thread_id=thread.thread_id,
            user_message="reconcile once",
            binding_manifest={"model_alias": "test-model"},
        )
        operation_id = _ready_operation(store, turn.turn_id)
        store.claim_tool_operation(
            operation_id=operation_id,
            worker_id="crashed-worker",
            now=100.0,
            lease_seconds=5.0,
        )
        store.expire_tool_operation_claim(operation_id=operation_id, now=106.0)
        orchestrator = ToolOrchestrator(
            store=store,
            tools={},
            execution_context=ToolExecutionContext(
                workspace_root=workspace,
                cwd=workspace,
            ),
        )

        outcome_type = getattr(harness, "ToolReconciliationOutcome", None)
        assert outcome_type is not None, "Harness must expose reconciliation outcomes"
        result = orchestrator.reconcile(
            turn_id=turn.turn_id,
            outcome=outcome_type(
                status="succeeded",
                result=ToolResult(
                    tool_call_id="call-1",
                    tool_name="write_file",
                    structured_content={"verified": True},
                    metadata={"reconciler_revision": "fixture-v1"},
                ),
            ),
        )

        assert result.structured_content == {"verified": True}
        operation = store.read_tool_operation(operation_id)
        assert operation.status == "succeeded"
        assert operation.requires_reconciliation is False
        [interaction] = store.list_interactions(turn.turn_id)
        assert interaction.kind == "tool_reconciliation"
        assert interaction.status == "resolved"
        assert interaction.response == {
            "status": "succeeded",
            "reconciler_revision": "fixture-v1",
        }
        assert store.read_turn(turn.turn_id).status == "running"
        [tool_result] = [
            item for item in store.list_items(turn.turn_id) if item.kind == "tool_result"
        ]
        assert operation.result_item_id == tool_result.item_id
        assert store.verify().valid is True

        store.rebuild_projections()
        assert store.read_tool_operation(operation_id).status == "succeeded"
        assert store.verify().valid is True


def test_live_stale_worker_cannot_commit_after_recovery_fences_claim(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    database = tmp_path / "rollout.sqlite3"
    with RolloutStore(database) as setup:
        thread = setup.create_thread(workspace=workspace)
        turn = setup.start_turn(
            thread_id=thread.thread_id,
            user_message="read once",
            binding_manifest={"model_alias": "test-model"},
        )

    started = Event()
    release = Event()
    schema: Mapping[str, JsonValue] = {
        "type": "object",
        "properties": {},
        "additionalProperties": False,
    }

    def run(_arguments: Mapping[str, JsonValue]) -> dict[str, str]:
        started.set()
        assert release.wait(timeout=5.0)
        return {"text": "stale success"}

    tool = Tool(
        definition=ToolDefinition(
            name="blocking_read",
            description="Block until the recovery test releases the runner.",
            input_schema=schema,
        ),
        validate_input=json_schema_input(schema),
        run=run,
        normalize_output=lambda raw: NormalizedToolOutput(
            content=(ToolContentBlock(type="text", data=raw),),
        ),
        output_schema=None,
        static_effects=frozenset({ToolEffect.READ_WORKSPACE}),
        resolve_use=lambda _arguments: ResolvedToolUse(
            effects=frozenset({ToolEffect.READ_WORKSPACE}),
            targets=(),
        ),
        execution_revision="blocking-read-v1",
        idempotent=True,
        concurrency_safe=False,
        cancellation_mode=CancellationMode.COOPERATIVE,
        interrupt_behavior=InterruptBehavior.CANCEL,
        timeout_seconds=45.0,
        max_model_output_bytes=4_096,
    )
    registry = ToolRegistry()
    registry.register(tool)
    tools = registry.freeze()
    call = ToolCall(
        tool_call_id="blocking-call-1",
        tool_name="blocking_read",
        arguments={},
        origin=ToolCallOrigin(
            request_id="request-1",
            toolset_revision="toolset-v1",
            exposed_tool_names=("blocking_read",),
        ),
    )

    def execute_in_old_worker() -> None:
        with RolloutStore(database) as worker_store:
            orchestrator = ToolOrchestrator(
                store=worker_store,
                tools=tools,
                execution_context=ToolExecutionContext(
                    workspace_root=workspace,
                    cwd=workspace,
                ),
                worker_id="worker-a",
                lease_seconds=30.0,
            )
            asyncio.run(orchestrator.execute(turn_id=turn.turn_id, call=call))

    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(execute_in_old_worker)
        assert started.wait(timeout=5.0)
        with RolloutStore(database) as recovery:
            [running] = recovery.list_tool_operations(turn.turn_id)
            assert running.status == "running"
            assert running.lease_expires_at is not None
            assert running.lease_expires_at - time.time() >= tool.timeout_seconds
            recovery.expire_tool_operation_claim(
                operation_id=running.operation_id,
                now=running.lease_expires_at + 1.0,
            )
        release.set()
        with pytest.raises(RuntimeError, match="stale worker"):
            future.result(timeout=5.0)

    with RolloutStore(database) as verifier:
        [operation] = verifier.list_tool_operations(turn.turn_id)
        assert operation.status == "unknown"
        assert not any(
            item.kind == "tool_result" for item in verifier.list_items(turn.turn_id)
        )
        assert any(
            record.record_type == "tool_operation_stale_result"
            for record in verifier.list_records(thread.thread_id)
        )
        assert verifier.verify().valid is True
