"""Resource owner and consistent queries. Business commands live in *_ops."""

from __future__ import annotations

import asyncio
import hashlib
import json
from collections.abc import AsyncGenerator, Callable, Iterator, Mapping
from functools import wraps
from pathlib import Path
from types import MappingProxyType
from typing import Any, Literal, Self

from agent_runtime.budget import (
    BudgetLimits,
    BudgetReservation,
    BudgetState,
    ReservationStatus,
    ResourceUsage,
    resource_usage_from_model_usage,
)
from agent_runtime.harness.operation_values import (
    _canonical_json,
    _context_durable_state,
    _validate_resource_usage_input,
)
from agent_runtime.harness.protocol import (
    restore_model_response,
)
from agent_runtime.harness.records import (
    ApprovalSnapshot,
    ArtifactGCReport,
    ArtifactSnapshot,
    CommittedMutation,
    InteractionSnapshot,
    ItemSnapshot,
    ModelAttemptSnapshot,
    ModelOperationSnapshot,
    RolloutRecord,
    ThreadSnapshot,
    ToolOperationSnapshot,
    TurnSnapshot,
    VerificationReport,
    _approval_snapshot,
    _interaction_snapshot,
    _item_snapshot,
    _model_attempt_snapshot,
    _model_operation_snapshot,
    _thread_snapshot,
    _tool_operation_snapshot,
    _turn_snapshot,
)
from agent_runtime.harness.rollout import RolloutStore
from agent_runtime.harness.runtime_tx import RuntimeTx


def owner_operation[**P, T](operation: Callable[P, T]) -> Callable[P, T]:
    """Serialize the complete closure; nested commands borrow the same owner."""

    @wraps(operation)
    def owned(*args: P.args, **kwargs: P.kwargs) -> T:
        store = args[0] if args else kwargs.get("store", kwargs.get("state"))
        if not isinstance(store, RuntimeStore):
            raise TypeError("runtime operation requires RuntimeStore")
        return store.run_sync(lambda: operation(*args, **kwargs))

    return owned


def read_operation[**P, T](operation: Callable[P, T]) -> Callable[P, T]:
    """Hold a coherent snapshot across every query in the closure."""

    @wraps(operation)
    def read(*args: P.args, **kwargs: P.kwargs) -> T:
        store = args[0]
        if not isinstance(store, RuntimeStore):
            raise TypeError("runtime query requires RuntimeStore")
        return store.read_snapshot(lambda: operation(*args, **kwargs))

    return read


class RuntimeStore:
    """Own persistence lifetime and serialize complete detached operations."""

    def __init__(self) -> None:
        self._persistence: RolloutStore | None = None

    @property
    def persistence(self) -> RolloutStore:
        if self._persistence is None:
            raise RuntimeError("RuntimeStore is not opened; use open_sync(...) or await open(...)")
        return self._persistence

    @classmethod
    def open_sync(cls, path: Path, *, read_only: bool = False, max_pending_operations: int = 128) -> Self:
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            pass
        else:
            raise RuntimeError("use await RuntimeStore.open(...) on an event-loop thread")
        self = cls()
        self._persistence = RolloutStore.open_sync(
            path,
            read_only=read_only,
            max_pending_operations=max_pending_operations,
        )
        return self

    @classmethod
    async def open(cls, path: Path, *, read_only: bool = False, max_pending_operations: int = 128) -> Self:
        self = cls()
        self._persistence = await RolloutStore.open(
            path,
            read_only=read_only,
            max_pending_operations=max_pending_operations,
        )
        return self

    @property
    def tx(self) -> RuntimeTx:
        return RuntimeTx(self.persistence)

    def read_snapshot[T](self, query: Callable[[], T]) -> T:
        """Run a complete detached query in one owner SQLite read snapshot."""

        def read() -> T:
            with self.persistence.read_snapshot():
                return query()

        return self.run_sync(read)

    def transact[T](self, command: Callable[[RuntimeTx], T]) -> CommittedMutation[T]:
        """Accept one complete command and return its committed fact receipt."""

        def mutate() -> T:
            with self.persistence.transaction():
                return command(self.tx)

        return self.capture_mutation(mutate)

    def run_sync[T](self, operation: Callable[[], T]) -> T:
        return self.persistence.run_sync(operation)

    async def run_async[**P, T](self, operation: Callable[P, T], *args: P.args, **kwargs: P.kwargs) -> T:
        return await self.persistence.run_async(operation, *args, **kwargs)

    async def iterate[T](self, iterator: Iterator[T]) -> AsyncGenerator[T, None]:
        async for value in self.persistence.iterate(iterator):
            yield value

    def __enter__(self) -> Self:
        self.persistence.__enter__()
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()

    async def __aenter__(self) -> Self:
        await self.persistence.__aenter__()
        return self

    async def __aexit__(self, *_args: object) -> None:
        await self.aclose()

    def close(self) -> None:
        if self._persistence is not None:
            self._persistence.close()

    async def aclose(self) -> None:
        if self._persistence is not None:
            await self._persistence.aclose()

    @property
    def epoch(self) -> str:
        return self.persistence.epoch

    @property
    def artifact_root(self) -> Path:
        return self.persistence.artifact_root

    def capture_mutation[T](self, operation: Callable[[], T]) -> CommittedMutation[T]:
        return self.persistence.capture_mutation(operation)

    async def capture_mutation_async[T](self, operation: Callable[[], T]) -> CommittedMutation[T]:
        return await self.run_async(self.capture_mutation, operation)

    def read_artifact_metadata(self, artifact_id: str) -> ArtifactSnapshot:
        return self.persistence.read_artifact_metadata(artifact_id)

    def list_artifacts(self, turn_id: str | None = None) -> tuple[ArtifactSnapshot, ...]:
        return self.persistence.list_artifacts(turn_id)

    def artifact_blob_path(self, artifact_id: str) -> Path:
        return self.persistence.artifact_blob_path(artifact_id)

    def read_artifact(self, artifact_id: str) -> bytes:
        return self.persistence.read_artifact(artifact_id)

    def gc_unreferenced_artifacts(self, *, min_age_seconds: float = 3600.0) -> ArtifactGCReport:
        return self.persistence.gc_unreferenced_artifacts(min_age_seconds=min_age_seconds)

    def last_record(self, thread_id: str) -> RolloutRecord | None:
        return self.persistence.last_record(thread_id)

    def list_records(self, thread_id: str) -> tuple[RolloutRecord, ...]:
        return self.persistence.list_records(thread_id)

    def list_turn_records(self, turn_id: str) -> tuple[RolloutRecord, ...]:
        return self.persistence.list_turn_records(turn_id)

    def read_turn_completion(self, turn_id: str) -> RolloutRecord | None:
        return self.persistence.read_turn_completion(turn_id)

    def list_record_tail(
        self,
        thread_id: str,
        *,
        after_thread_sequence: int = 0,
        limit: int | None = None,
        through_thread_sequence: int = 9223372036854775807,
    ) -> tuple[RolloutRecord, ...]:
        return self.persistence.list_record_tail(
            thread_id,
            after_thread_sequence=after_thread_sequence,
            limit=limit,
            through_thread_sequence=through_thread_sequence,
        )

    def read_event_reference(
        self,
        record_type: Literal[
            "item_started",
            "item_completed",
            "model_attempt_completed",
            "tool_operation_result_linked",
            "tool_operation_claimed",
            "tool_operation_prepared",
            "tool_operation_outcome_committed",
            "tool_operation_migrated",
        ],
        identity: str,
        *,
        at_record_id: int | None = None,
    ) -> RolloutRecord | None:
        return self.persistence.read_event_reference(record_type, identity, at_record_id=at_record_id)

    def find_model_response_item(self, turn_id: str, text: str | None) -> str | None:
        return self.persistence.find_model_response_item(turn_id, text)

    def list_global_records(
        self, *, after_record_id: int = 0, limit: int | None = None, through_record_id: int = 9223372036854775807
    ) -> tuple[RolloutRecord, ...]:
        return self.persistence.list_global_records(
            after_record_id=after_record_id, limit=limit, through_record_id=through_record_id
        )

    def last_global_record_id(self) -> int:
        return self.persistence.last_global_record_id()

    def projection_hashes(self) -> dict[str, str]:
        return self.persistence.projection_hashes()

    def verify(self) -> VerificationReport:
        return self.persistence.verify()

    def rebuild_projections(self) -> None:
        return self.persistence.rebuild_projections()

    @read_operation
    def validate_conversation(
        self,
        *,
        thread_id: str | None = None,
        turn_id: str | None = None,
    ) -> VerificationReport:
        """Replay the conversation's canonical recovery dependencies on open.

        Discover identities from facts, never from the projections being
        checked. A fresh conversation consumes no previous conversation state.
        Full-store audit remains RolloutStore.verify(). No cached digest serves
        as evidence that projected rows or blobs are valid.
        """
        if thread_id is not None and turn_id is not None:
            raise ValueError("choose one conversation anchor")
        with self.persistence.read_snapshot():

            def canonical_turn_thread(identity: str) -> str:
                fact = self.persistence.first(
                    "rollout_records",
                    where={
                        "turn_id": identity,
                        "record_type": "turn_started",
                    },
                )
                if fact is None:
                    raise KeyError(f"unknown canonical Turn: {identity}")
                return str(fact["thread_id"])

            if turn_id is not None:
                thread_id = canonical_turn_thread(turn_id)
            if thread_id is None:
                return VerificationReport(valid=True, errors=())
            pending, checked = [thread_id], set()
            while pending:
                identity = pending.pop()
                if identity in checked:
                    continue
                checked.add(identity)
                for fact in self.persistence.list_records(identity):
                    payload = fact.payload
                    if fact.record_type == "thread_forked":
                        pending.append(str(payload["parent_thread_id"]))
                    elif fact.record_type == "turn_started":
                        parent = payload["binding_manifest"].get("budget_parent_turn_id")
                        if parent:
                            pending.append(canonical_turn_thread(parent))
                    elif fact.record_type == "budget_child_allocated":
                        pending.append(canonical_turn_thread(str(payload["child_turn_id"])))
                    elif fact.record_type == "item_completed":
                        for attachment in payload.get("payload", {}).get("attachments", ()):
                            if not isinstance(attachment, Mapping) or not isinstance(
                                attachment.get("artifact_id"),
                                str,
                            ):
                                continue  # The independent replay reports malformed attachments.
                            artifact = self.persistence.first(
                                "rollout_records",
                                where={
                                    "record_type": "artifact_committed",
                                },
                                payload_fields={"artifact_id": attachment["artifact_id"]},
                            )
                            if artifact is not None:
                                pending.append(str(artifact["thread_id"]))
            return self.persistence.verify(thread_ids=frozenset(checked))

    @read_operation
    def read_memory_state(
        self,
        workspace: Path,
        *,
        version: int | None = None,
        scope_id: str | None = None,
    ) -> dict[str, Any]:
        """Read workspace memory; old/read-only databases need no schema migration."""
        with self.persistence.read_snapshot():
            from agent_runtime.harness.memory_identity import memory_namespace

            if scope_id is not None and self.persistence.metadata("memory-scope:" + scope_id) is None:
                raise KeyError("unknown memory scope")
            key = "memory:" + (scope_id if scope_id is not None else memory_namespace(self, workspace))
            if version == 0:
                return {
                    "version": 0,
                    "entries": [],
                    "processed": {},
                    "deleted_keys": [],
                    "deleted_sources": [],
                    "use_memories": False,
                    "generate_memories": False,
                }
            if version is not None and version > 0:
                key += f":version:{version}"
            row = {"value": value} if (value := self.persistence.metadata(key)) is not None else None
            if row is None:
                if version is not None and version > 0:
                    raise KeyError("unknown workspace memory version")
                return {
                    "version": 0,
                    "entries": [],
                    "processed": {},
                    "deleted_keys": [],
                    "deleted_sources": [],
                    "use_memories": False,
                    "generate_memories": False,
                }
            envelope = json.loads(row["value"])
            state = envelope["state"]
            if hashlib.sha256(_canonical_json(state).encode()).hexdigest() != envelope["sha256"]:
                raise RuntimeError("workspace memory integrity mismatch")
            return dict(state)

    @read_operation
    def read_memory_jobs(self, workspace: Path) -> dict[str, Any]:
        with self.persistence.read_snapshot():
            from agent_runtime.harness.memory_identity import memory_namespace

            key = "memory-jobs:" + memory_namespace(self, workspace)
            row = {"value": value} if (value := self.persistence.metadata(key)) is not None else None
            return {} if row is None else dict(json.loads(row["value"]))

    def memory_scope(self, workspace: Path) -> dict[str, Any] | None:
        from agent_runtime.harness.memory_identity import memory_scope

        return memory_scope(self, workspace)

    def thread_in_memory_scope(self, workspace: Path, thread_id: str) -> bool:
        from agent_runtime.harness.memory_identity import thread_in_memory_scope

        return thread_in_memory_scope(self, workspace, thread_id)

    @read_operation
    def user_input_revision(self, turn_id: str) -> int:
        """Count canonical user messages, independently of context compaction."""
        return int(
            (
                self.persistence.count(
                    "items", where={"turn_id": turn_id, "kind": "user_message", "status": "completed"}
                ),
            )[0]
        )

    @read_operation
    def request_input_revision(self, turn_id: str, request_ref: Mapping[str, Any]) -> int:
        """Require the producer's explicit revision; never infer it from history."""
        revision = request_ref.get("input_revision")
        if type(revision) is not int or revision < 1:
            raise ValueError(
                f"Turn {turn_id}: model request input_revision must be a positive integer; "
                "unsupported requests cannot be resumed; start a new Turn."
            )
        return revision

    @read_operation
    def turn_input_status(self, item_id: str) -> str:
        """Derive input consumption from ordinary model dispatches, never preparation."""
        item = self.read_item(item_id)
        if item.kind != "user_message" or "input_revision" not in item.payload:
            raise ValueError("Item is not a steering receipt")
        dispatched = {
            row["payload"].get("operation_id")
            for row in self.persistence.rows(
                "rollout_records", where={"turn_id": item.turn_id, "record_type": "model_attempt_dispatched"}
            )
        }
        for operation in self.list_model_operations(item.turn_id):
            revision = operation.request_ref.get("input_revision")
            if (
                operation.request_ref.get("purpose") == "agent_step"
                and type(revision) is int
                and revision > 0
                and revision >= item.payload["input_revision"]
                and operation.operation_id in dispatched
            ):
                return "applied"
        status = self.read_turn(item.turn_id).status
        return "pending" if status in {"running", "paused", "interrupted"} else "unconsumed"

    @read_operation
    def context_source_revision(self, turn_id: str) -> str:
        """Version every source Thread, including non-message state transitions."""
        thread_ids = {thread_id for thread_id, _ in self._conversation_segments(turn_id)}
        thread_ids.add(self.read_turn(turn_id).thread_id)
        versions = [
            (thread_id, self.read_thread(thread_id).applied_thread_sequence) for thread_id in sorted(thread_ids)
        ]
        return hashlib.sha256(_canonical_json(versions).encode()).hexdigest()

    @read_operation
    def context_durable_state(self, turn_id: str) -> dict[str, Any]:
        return _context_durable_state(self, turn_id)

    @read_operation
    def turn_cancellation_pending(self, turn_id: str) -> bool:
        """The latest durable cancellation/resume fact decides the stop state."""
        records = self.persistence.rows(
            "rollout_records",
            where={
                "turn_id": turn_id,
                "record_type": ("turn_cancellation_requested", "turn_resumed"),
            },
            order_by="record_id",
            newest=True,
            limit=1,
        )
        return bool(records and records[0]["record_type"] == "turn_cancellation_requested")

    @read_operation
    def turn_admission_revoked_by(self, turn_id: str) -> str | None:
        """Read durable cancellation/terminal lineage without relying on live drivers."""
        seen: set[str] = set()
        current = turn_id
        while current not in seen:
            seen.add(current)
            turn = self.read_turn(current)
            if self.turn_cancellation_pending(current) or turn.status in {"completed", "failed", "cancelled"}:
                return current
            allocation = self.persistence.first("child_budget_allocations", where={"child_turn_id": current})
            if allocation is None:
                return None
            current = str(allocation["parent_turn_id"])
        raise RuntimeError("child allocation ancestry contains a cycle")

    @read_operation
    def list_model_operations(self, turn_id: str | None = None) -> tuple[ModelOperationSnapshot, ...]:
        if turn_id is None:
            rows = self.persistence.rows("model_operations", order_by="rowid")
        else:
            rows = self.persistence.rows("model_operations", where={"turn_id": turn_id}, order_by="rowid")
        return tuple(_model_operation_snapshot(row) for row in rows)

    @read_operation
    def list_model_attempts(self, operation_id: str) -> tuple[ModelAttemptSnapshot, ...]:
        rows = self.persistence.rows("model_attempts", where={"operation_id": operation_id}, order_by="generation")
        return tuple(_model_attempt_snapshot(row) for row in rows)

    @read_operation
    def read_child_budget_allocation(self, child_turn_id: str) -> Mapping[str, Any]:
        row = self.persistence.first("child_budget_allocations", where={"child_turn_id": child_turn_id})
        if row is None:
            raise KeyError(f"child Turn has no parent budget allocation: {child_turn_id}")
        return MappingProxyType(
            {
                "allocation_id": str(row["allocation_id"]),
                "parent_turn_id": str(row["parent_turn_id"]),
                "child_turn_id": str(row["child_turn_id"]),
                "allocated_tokens": row["allocated_tokens"],
                "allocated_cost_micros": row["allocated_cost_micros"],
                "actual": ResourceUsage.from_dict(row["actual"]),
                "status": str(row["status"]),
            }
        )

    @read_operation
    def read_budget_state(self, turn_id: str) -> BudgetState:
        """Derive the Turn budget from durable reservations and tool executions."""

        turn = self.read_turn(turn_id)
        raw_limit = turn.binding_manifest.get("model_token_budget_total")
        if raw_limit is None:
            token_limit: int | None = None
        elif isinstance(raw_limit, bool) or not isinstance(raw_limit, int) or raw_limit < 0:
            raise RuntimeError("frozen model token budget is invalid")
        else:
            token_limit = raw_limit

        raw_cost_limit = turn.binding_manifest.get("model_cost_budget_total_micros")
        if raw_cost_limit is None:
            cost_limit: int | None = None
        elif isinstance(raw_cost_limit, bool) or not isinstance(raw_cost_limit, int) or raw_cost_limit < 0:
            raise RuntimeError("frozen model cost budget is invalid")
        else:
            cost_limit = raw_cost_limit

        used = ResourceUsage()
        reserved = ResourceUsage()
        uncertain = ResourceUsage()
        child_reserved = ResourceUsage()
        tracked_attempt_ids: set[str] = set()
        rows = self.persistence.rows("budget_reservations", where={"turn_id": turn_id}, order_by="rowid")
        for row in rows:
            tracked_attempt_ids.add(str(row["attempt_id"]))
            reserved_usage = ResourceUsage.from_dict(row["reserved"])
            status = ReservationStatus(str(row["status"]))
            if status in {ReservationStatus.RESERVED, ReservationStatus.DISPATCHED}:
                reserved = reserved + reserved_usage
            elif status is ReservationStatus.UNKNOWN:
                uncertain = uncertain + reserved_usage
            elif status is ReservationStatus.SETTLED:
                used = used + ResourceUsage.from_dict(row["actual"])

        # Compatibility for Turns created before durable reservations existed.
        # Once an attempt is tracked by a reservation, its usage is sourced only
        # from that reservation projection to avoid double-counting.
        for operation in self.list_model_operations(turn_id):
            for attempt in self.list_model_attempts(operation.operation_id):
                if attempt.attempt_id in tracked_attempt_ids or attempt.status != "completed":
                    continue
                used = used + resource_usage_from_model_usage(attempt.usage)

        # Claims count execution attempts, including failed/unknown outcomes.
        # Result Items can be replaced during reconciliation, so they are not
        # counters. Fenced outcomes preserve each observed duration only once.
        tool_ms = sum(
            float(record.payload.get("duration_ms") or 0)
            for record in self.list_records(turn.thread_id)
            if record.turn_id == turn_id and record.record_type == "tool_operation_outcome_committed"
        )
        used = used + ResourceUsage(
            tool_calls=sum(operation.attempt_count for operation in self.list_tool_operations(turn_id)),
            tool_ms=round(tool_ms),
        )

        for row in self.persistence.rows(
            "child_budget_allocations", where={"parent_turn_id": turn_id}, order_by="rowid"
        ):
            if row["status"] == "active":
                allocated_tokens = row["allocated_tokens"]
                if allocated_tokens is not None:
                    if (
                        isinstance(allocated_tokens, bool)
                        or not isinstance(allocated_tokens, int)
                        or allocated_tokens < 0
                    ):
                        raise RuntimeError("durable child token allocation is invalid")
                child_reserved = child_reserved + ResourceUsage(
                    # Child allocations reserve fungible total-token capacity.
                    # input_tokens is only the carrier so ResourceUsage.total_tokens
                    # can participate in the existing BudgetState arithmetic.
                    input_tokens=allocated_tokens or 0,
                    cost_micros=(row["allocated_cost_micros"] or 0),
                    subagents=1,
                )
            elif row["status"] == "settled":
                used = used + ResourceUsage.from_dict(row["actual"])
            else:
                raise RuntimeError("durable child budget status is unsupported")

        parent_scope_id = turn.binding_manifest.get("budget_parent_turn_id")
        if not isinstance(parent_scope_id, str) or not parent_scope_id:
            parent_scope_id = None
        return BudgetState(
            scope_id=turn_id,
            parent_scope_id=parent_scope_id,
            limits=BudgetLimits(tokens=token_limit, cost_micros=cost_limit),
            used=used,
            reserved=reserved,
            uncertain=uncertain,
            child_reserved=child_reserved,
        )

    @read_operation
    def read_budget_reservation(self, attempt_id: str) -> BudgetReservation:
        from agent_runtime.harness import budget_ops

        row = budget_ops._budget_reservation_row_for_attempt(self, attempt_id)
        if row is None:
            raise KeyError(f"model attempt has no budget reservation: {attempt_id}")
        return BudgetReservation(
            reservation_id=str(row["reservation_id"]),
            scope_id=str(row["scope_id"]),
            operation_id=str(row["operation_id"]),
            attempt_id=str(row["attempt_id"]),
            reserved=ResourceUsage.from_dict(row["reserved"]),
            status=ReservationStatus(str(row["status"])),
        )

    @read_operation
    def read_tool_operation(self, operation_id: str) -> ToolOperationSnapshot:
        row = self.persistence.first("tool_operations", where={"operation_id": operation_id})
        if row is None:
            raise KeyError(f"unknown tool operation: {operation_id}")
        return _tool_operation_snapshot(row)

    @read_operation
    def list_tool_operations(self, turn_id: str | None = None) -> tuple[ToolOperationSnapshot, ...]:
        if turn_id is None:
            rows = self.persistence.rows("tool_operations", order_by="rowid")
        else:
            rows = self.persistence.rows("tool_operations", where={"turn_id": turn_id}, order_by="rowid")
        return tuple(_tool_operation_snapshot(row) for row in rows)

    @read_operation
    def read_interaction(self, request_id: str) -> InteractionSnapshot:
        row = self.persistence.first("interactions", where={"request_id": request_id})
        if row is None:
            raise KeyError(f"unknown interaction: {request_id}")
        return _interaction_snapshot(row)

    @read_operation
    def list_interactions(self, turn_id: str | None = None) -> tuple[InteractionSnapshot, ...]:
        if turn_id is None:
            rows = self.persistence.rows("interactions", order_by="rowid")
        else:
            rows = self.persistence.rows("interactions", where={"turn_id": turn_id}, order_by="rowid")
        return tuple(_interaction_snapshot(row) for row in rows)

    @read_operation
    def list_approvals(self, turn_id: str | None = None) -> tuple[ApprovalSnapshot, ...]:
        if turn_id is None:
            rows = self.persistence.rows("approvals", order_by="rowid")
        else:
            rows = self.persistence.rows("approvals", where={"turn_id": turn_id}, order_by="rowid")
        return tuple(_approval_snapshot(row) for row in rows)

    @read_operation
    def read_resource_usage(
        self,
        turn_id: str,
        resource: str,
        initial_used: int = 0,
    ) -> dict[str, Any]:
        """Read the latest durable counter; legacy usage applies only before its first Item."""

        _validate_resource_usage_input(resource, initial_used)
        self.read_turn(turn_id)
        rows = self.persistence.rows(
            "items",
            where={"turn_id": turn_id, "kind": "resource_usage", "status": "completed", "producer": "runtime"},
            order_by="sequence",
            newest=True,
        )
        for row in rows:
            payload = row["payload"]
            if payload.get("resource") != resource:
                continue
            used, limit = payload.get("used"), payload.get("limit")
            if type(used) is not int or type(limit) is not int or not 0 <= used <= limit <= 256 or limit < 1:
                raise RuntimeError("canonical resource_usage payload is malformed")
            return {
                "resource": resource,
                "limit": limit,
                "used": used,
                "remaining": max(0, limit - used),
                "item_id": str(row["item_id"]),
            }
        return {"resource": resource, "limit": None, "used": initial_used, "remaining": None, "item_id": None}

    @read_operation
    def read_thread(self, thread_id: str) -> ThreadSnapshot:
        row = self.persistence.first("threads", where={"thread_id": thread_id})
        if row is None:
            raise KeyError(f"unknown thread: {thread_id}")
        return _thread_snapshot(row)

    @read_operation
    def list_threads(self) -> tuple[ThreadSnapshot, ...]:
        rows = self.persistence.rows("threads", order_by="rowid")
        return tuple(_thread_snapshot(row) for row in rows)

    @read_operation
    def read_turn(self, turn_id: str) -> TurnSnapshot:
        row = self.persistence.first("turns", where={"turn_id": turn_id})
        if row is None:
            raise KeyError(f"unknown turn: {turn_id}")
        return _turn_snapshot(row)

    @read_operation
    def list_turns(self, thread_id: str | None = None) -> tuple[TurnSnapshot, ...]:
        rows = self.persistence.rows(
            "turns", where=({} if thread_id is None else {"thread_id": thread_id}), order_by="rowid"
        )
        return tuple(_turn_snapshot(row) for row in rows)

    @read_operation
    def list_items(
        self,
        turn_id: str,
        *,
        kind: str | tuple[str, ...] | None = None,
        payload_fields: Mapping[str, Any] | None = None,
    ) -> tuple[ItemSnapshot, ...]:
        rows = self.persistence.rows(
            "items",
            where={"turn_id": turn_id, **({"kind": kind} if kind is not None else {})},
            payload_fields=payload_fields,
            order_by="sequence",
        )
        return tuple(_item_snapshot(row) for row in rows)

    def _latest_item(self, turn_id: str) -> ItemSnapshot:
        row = self.persistence.first("items", where={"turn_id": turn_id}, order_by="sequence", newest=True, limit=1)
        if row is None:
            raise KeyError(f"Turn has no Items: {turn_id}")
        return _item_snapshot(row)

    @read_operation
    def read_item(self, item_id: str) -> ItemSnapshot:
        row = self.persistence.first("items", where={"item_id": item_id})
        if row is None:
            raise KeyError(f"unknown Item: {item_id}")
        return _item_snapshot(row)

    @read_operation
    def list_context_items(self, turn_id: str, *, tool_name: str | None = None) -> tuple[ItemSnapshot, ...]:
        """Visible model/recall window; the complete ledger remains separately readable."""
        from agent_runtime.harness import artifact_ops

        artifact_ops._assert_artifact_integrity(self, turn_id)
        return tuple(
            _item_snapshot(row)
            for thread_id, start, cutoff in self._context_segments(turn_id)
            for row in self.persistence.conversation_rows(
                thread_id,
                cutoff,
                from_turn_index=start,
                tool_name=tool_name,
            )
        )

    @read_operation
    def list_conversation_items(self, turn_id: str) -> tuple[ItemSnapshot, ...]:
        """Read original Items through the fork point; do not load or execute artifacts."""
        rows: list[Mapping[str, Any]] = []
        for ancestor_thread_id, cutoff in self._conversation_segments(turn_id):
            rows.extend(self.persistence.conversation_rows(ancestor_thread_id, cutoff))
        return tuple(_item_snapshot(row) for row in rows)

    def _conversation_segments(self, turn_id: str) -> tuple[tuple[str, int], ...]:
        """Resolve ancestry using metadata only, independently of history size."""
        turn = self.persistence.first("turns", where={"turn_id": turn_id})
        if turn is None:
            raise KeyError(f"unknown turn: {turn_id}")
        segments: list[tuple[str, int]] = []
        segment_thread_id = str(turn["thread_id"])
        segment_turn_index = int(turn["turn_index"])
        while True:
            segments.append((segment_thread_id, segment_turn_index))
            thread = self.persistence.first("threads", where={"thread_id": segment_thread_id})
            if thread is None:
                raise RuntimeError("context Thread projection is missing")
            parent_thread_id = thread["parent_thread_id"]
            fork_turn_id = thread["fork_turn_id"]
            if parent_thread_id is None and fork_turn_id is None:
                break
            if parent_thread_id is None or fork_turn_id is None:
                raise RuntimeError("fork ancestry projection is incomplete")
            fork_turn = self.persistence.first("turns", where={"turn_id": fork_turn_id})
            if fork_turn is None or fork_turn["thread_id"] != parent_thread_id:
                raise RuntimeError("fork point does not belong to its parent Thread")
            segment_thread_id = str(parent_thread_id)
            segment_turn_index = int(fork_turn["turn_index"])
        return tuple(reversed(segments))

    def _context_segments(self, turn_id: str) -> tuple[tuple[str, int, int], ...]:
        """Resolve context resets before loading any message bodies."""
        segments = []
        for thread_id, cutoff in self._conversation_segments(turn_id):
            start = 0
            for turn in self.persistence.rows("turns", where={"thread_id": thread_id}, order_by="turn_index"):
                reset = turn["binding_manifest"].get("context_reset")
                if (
                    turn["turn_index"] <= cutoff
                    and isinstance(reset, Mapping)
                    and reset.get("revision") == "v1"
                    and reset.get("reason") == "memory_revoked"
                ):
                    segments.clear()
                    start = turn["turn_index"]
            segments.append((thread_id, start, cutoff))
        return tuple(segments)

    @read_operation
    def user_read_references(self, turn_id: str) -> tuple[dict[str, str], ...]:
        references = {}
        for thread_id, cutoff in self._conversation_segments(turn_id):
            for reference in self.persistence.user_read_references(thread_id, cutoff):
                references[reference["path"]] = reference
        return tuple(references.values())

    @read_operation
    def conversation_artifact_ids(self, turn_id: str) -> set[str]:
        return set().union(
            *(
                self.persistence.conversation_artifact_ids(thread_id, cutoff)
                for thread_id, cutoff in self._conversation_segments(turn_id)
            )
        )

    @read_operation
    def list_model_context_items(self, turn_id: str) -> tuple[ItemSnapshot, ...]:
        """Load a complete replacement and its suffix; old Rollouts use legacy replay.

        Canonical Items stay available through list_context_items/read_item. Only
        the small set explicitly retained by the checkpoint is fetched by ID.
        """
        segments = self._context_segments(turn_id)
        for index in range(len(segments) - 1, -1, -1):
            thread_id, start, cutoff = segments[index]
            row = self.persistence.latest_context_checkpoint(thread_id, cutoff, from_turn_index=start)
            if row is None:
                continue
            checkpoint = _item_snapshot(row)
            identities = dict.fromkeys(
                entry["item_id"] for entry in checkpoint.payload["replacement_history"] if entry["item_id"]
            )
            retained = tuple(self.read_item(identity) for identity in identities)
            suffix = list(
                self.persistence.conversation_rows(
                    thread_id,
                    cutoff,
                    after_sequence=checkpoint.applied_thread_sequence,
                )
            )
            for segment_thread, segment_start, segment_cutoff in segments[index + 1 :]:
                suffix.extend(
                    self.persistence.conversation_rows(
                        segment_thread,
                        segment_cutoff,
                        from_turn_index=segment_start,
                    )
                )
            items = (*retained, checkpoint, *(_item_snapshot(row) for row in suffix))
            visible_ids = {item.item_id for item in items}
            if any(
                item.kind == "context_compaction"
                and not set(item.payload.get("covered_item_ids", ())).issubset(visible_ids)
                for item in items[len(retained) + 1 :]
            ):
                # Old elision records used whole-archive coverage. Replay them
                # under that contract until the next complete checkpoint.
                return self.list_context_items(turn_id)
            from agent_runtime.harness.artifact_ops import _assert_artifact_integrity

            _assert_artifact_integrity(self, turn_id, items=items)
            return items
        return self.list_context_items(turn_id)

    @read_operation
    def read_turn_answer(self, turn_id: str) -> ItemSnapshot:
        """Resolve the accepted output identity."""
        completion = self.read_turn_completion(turn_id)
        if completion is None:
            raise RuntimeError("Turn has no committed answer")
        item_id = completion.payload["answer_item_id"]
        answer = self.read_item(item_id)
        if (
            answer.turn_id != turn_id
            or answer.status != "completed"
            or answer.kind not in {"agent_message", "model_response"}
            or (answer.kind == "model_response" and restore_model_response(answer.payload).status != "completed")
        ):
            raise RuntimeError("completed Turn has an invalid answer reference")
        return answer
