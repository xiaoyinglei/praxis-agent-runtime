"""Business transitions over durable facts; no SQL or connection ownership."""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
import time
from collections.abc import AsyncGenerator, Callable, Iterator, Mapping
from functools import wraps
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Literal, Self
from uuid import uuid4

from agent_runtime.budget import (
    BudgetLimitExceededError,
    BudgetLimits,
    BudgetReservation,
    BudgetState,
    ReservationStatus,
    ResourceUsage,
    ensure_token_headroom,
    normal_token_remaining,
    resource_usage_from_model_usage,
)
from agent_runtime.budget import (
    reserve as reserve_budget,
)
from agent_runtime.harness.protocol import ContextSourceChangedError
from agent_runtime.harness.records import (
    ApprovalSnapshot,
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
from agent_runtime.modeling.events import ModelOutput, output_to_payload
from agent_runtime.planning import PlanStep
from agent_runtime.streaming.events import (
    derive_model_public_item_id,
    derive_operation_public_item_id,
    derive_plan_public_item_id,
    derive_reconciliation_public_item_id,
)

if TYPE_CHECKING:
    from agent_runtime.harness.memory_identity import MemoryClaim


def _state_operation[**P, T](method: Callable[P, T]) -> Callable[P, T]:
    """Execute a complete business operation on its persistence owner."""

    @wraps(method)
    def owned(*args: P.args, **kwargs: P.kwargs) -> T:
        state = args[0]
        assert isinstance(state, RuntimeState)
        return state.run_sync(lambda: method(*args, **kwargs))

    return owned


class RuntimeState:
    """Own Turn, model, tool, approval and budget transitions.

    The business operation chooses transaction scope. RolloutStore owns the
    connection and atomically commits facts and their rebuildable projections.
    Construct without I/O, open explicitly, and close/drain accepted work.
    """

    def __init__(self) -> None:
        self._persistence: RolloutStore | None = None

    @property
    def persistence(self) -> RolloutStore:
        if self._persistence is None:
            raise RuntimeError("RuntimeState is not opened; use open_sync(...) or await open(...)")
        return self._persistence

    @classmethod
    def open_sync(cls, path: Path, *, read_only: bool = False, max_pending_operations: int = 128) -> Self:
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            pass
        else:
            raise RuntimeError("use await RuntimeState.open(...) on an event-loop thread")
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

    def _append_and_reduce(self, **fact: Any) -> None:
        # Business identity enrichment belongs with commands, before storage.
        if fact["record_type"] == "item_completed":
            start = self.read_event_reference("item_started", str(fact["payload"]["item_id"]))
            if start is None:
                raise RuntimeError("completed internal Item has no durable start")
            fact["payload"] = {**start.payload, **fact["payload"]}
        elif fact["record_type"] in {"interaction_resolved", "approval_invalidated"}:
            interaction = self.persistence.first("interactions", where={"request_id": fact["payload"]["request_id"]})
            if interaction is None:
                raise RuntimeError("interaction resolution has no committed request")
            fact["payload"] = {
                "kind": interaction["kind"], "operation_id": interaction["operation_id"], **fact["payload"],
            }
        self.persistence.append_record(**fact)

    def read_artifact_metadata(self, artifact_id: str) -> ArtifactSnapshot:
        return self.persistence.read_artifact_metadata(artifact_id)

    def list_artifacts(self, turn_id: str | None = None) -> tuple[ArtifactSnapshot, ...]:
        return self.persistence.list_artifacts(turn_id)

    def artifact_blob_path(self, artifact_id: str) -> Path:
        return self.persistence.artifact_blob_path(artifact_id)

    def read_artifact(self, artifact_id: str) -> bytes:
        return self.persistence.read_artifact(artifact_id)

    def gc_unreferenced_artifacts(self, *, min_age_seconds: float = 3600.0) -> tuple[str, ...]:
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

    @_state_operation
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

    @_state_operation
    def commit_artifact(
        self,
        *,
        turn_id: str,
        content: bytes,
        media_type: str,
        name: str,
        fault_injector: Callable[[str], None] | None = None,
    ) -> ArtifactSnapshot:
        self.persistence.assert_writable()
        if not isinstance(media_type, str) or not media_type.strip():
            raise ValueError("artifact media_type must be non-empty")
        if not isinstance(name, str) or not name.strip():
            raise ValueError("artifact name must be non-empty")
        artifact_id = f"artifact_{uuid4().hex}"
        with self.persistence.transaction():
            thread_id = self._running_turn_thread_id(turn_id)
            # GC holds the same write boundary. It cannot unlink an old orphan
            # after we verify/reinstall it but before its reference commits.
            blob_sha256 = self.persistence.install_blob(content, fault_injector=fault_injector)
            self._append_and_reduce(
                thread_id=thread_id,
                turn_id=turn_id,
                record_type="artifact_committed",
                producer="runtime",
                payload={
                    "artifact_id": artifact_id,
                    "blob_sha256": blob_sha256,
                    "size_bytes": len(content),
                    "media_type": media_type,
                    "name": name,
                },
            )
            if fault_injector is not None:
                fault_injector("before_sqlite_commit")
        if fault_injector is not None:
            fault_injector("after_sqlite_commit")
        return self.read_artifact_metadata(artifact_id)

    @_state_operation
    def read_memory_state(
        self, workspace: Path, *, version: int | None = None, scope_id: str | None = None,
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

    @_state_operation
    def commit_memory_state(
        self,
        workspace: Path,
        *,
        expected_version: int,
        state: Mapping[str, Any],
        claim: MemoryClaim | None = None,
        job_thread_id: str | None = None,
        job_state: Mapping[str, Any] | None = None,
        job_revision: int | None = None,
    ) -> dict[str, Any]:
        """CAS commit; keep immutable versions for inspection and crash recovery."""
        from agent_runtime.harness.memory_identity import memory_namespace

        with self.persistence.transaction():
            if claim is not None:
                self._assert_memory_claim(workspace, claim)
            key = "memory:" + memory_namespace(self, workspace, create=True)
            if job_thread_id is not None and job_state is not None:
                self.write_memory_job(
                    workspace, job_thread_id, job_state, claim=claim, expected_revision=job_revision
                )
            if self.read_memory_state(workspace)["version"] != expected_version:
                raise ContextSourceChangedError("workspace memory changed before commit")
            committed = _json_object({**state, "version": expected_version + 1}, field="memory state")
            value = _canonical_json(
                {"state": committed, "sha256": hashlib.sha256(_canonical_json(committed).encode()).hexdigest()}
            )
            self.persistence.put_metadata(f"{key}:version:{expected_version + 1}", value, replace=False)
            self.persistence.put_metadata(key, value, replace=True)
        return committed

    @_state_operation
    def read_memory_jobs(self, workspace: Path) -> dict[str, Any]:
        with self.persistence.read_snapshot():
            from agent_runtime.harness.memory_identity import memory_namespace

            key = "memory-jobs:" + memory_namespace(self, workspace)
            row = {"value": value} if (value := self.persistence.metadata(key)) is not None else None
            return {} if row is None else dict(json.loads(row["value"]))

    @_state_operation
    def write_memory_job(
        self,
        workspace: Path,
        thread_id: str,
        state: Mapping[str, Any],
        *,
        claim: MemoryClaim | None = None,
        expected_revision: int | None = None,
    ) -> dict[str, Any]:
        from agent_runtime.harness.memory_identity import memory_namespace

        with self.persistence.transaction():
            if claim is not None:
                self._assert_memory_claim(workspace, claim)
            key = "memory-jobs:" + memory_namespace(self, workspace, create=True)
            jobs = self.read_memory_jobs(workspace)
            revision = jobs.get(thread_id, {}).get("revision", 0)
            if expected_revision is not None and revision != expected_revision:
                raise ContextSourceChangedError("memory job changed before commit")
            job = _json_object({**state, "revision": revision + 1}, field="memory job")
            jobs[thread_id] = job
            self.persistence.put_metadata(key, _canonical_json(jobs), replace=True)
        return job

    @_state_operation
    def enqueue_memory_job(self, workspace: Path, thread_id: str) -> None:
        from agent_runtime.harness.memory_identity import thread_in_memory_scope

        with self.persistence.transaction():
            if not thread_in_memory_scope(self, workspace, thread_id):
                raise ValueError("memory job belongs to a different workspace")
            job = self.read_memory_jobs(workspace).get(thread_id, {})
            if job.get("status") in {"pending", "running", "needs_reconciliation", "failed"}:
                return
            self.write_memory_job(workspace, thread_id, {"status": "pending", "turn_ids": []})

    def _assert_memory_claim(self, workspace: Path, claim: MemoryClaim) -> None:
        scope = self.memory_scope(workspace)
        if (scope is None or scope["scope_id"] != claim.scope_id
                or scope["binding_generation"] != claim.binding_generation):
            raise ContextSourceChangedError("memory workspace binding changed")
        raw = self.persistence.metadata("memory-job:" + claim.scope_id)
        lease = {} if raw is None else json.loads(raw)
        if (lease.get("owner") != claim.owner or lease.get("expires_at", 0) <= time.time()
                or lease.get("lease_id") != claim.lease_id
                or lease.get("scope_id") != claim.scope_id
                or lease.get("binding_generation") != claim.binding_generation):
            raise ContextSourceChangedError("memory workspace lease is no longer owned")

    @_state_operation
    def claim_memory_job(
        self, workspace: Path, *, owner: str, now: float, lease_seconds: float,
    ) -> MemoryClaim | None:
        from agent_runtime.harness.memory_identity import MemoryClaim, memory_namespace

        with self.persistence.transaction():
            key = "memory-job:" + memory_namespace(self, workspace)
            row = {"value": value} if (value := self.persistence.metadata(key)) is not None else None
            if row is not None and json.loads(row["value"])["expires_at"] > now:
                return None
            namespace = memory_namespace(self, workspace, create=True)
            scope = self.memory_scope(workspace)
            assert scope is not None
            claim = MemoryClaim(namespace, scope["binding_generation"], owner, uuid4().hex)
            self.persistence.put_metadata(
                "memory-job:" + namespace,
                _canonical_json({"owner": owner, "expires_at": now + lease_seconds,
                                 "scope_id": claim.scope_id, "binding_generation": claim.binding_generation,
                                 "lease_id": claim.lease_id}),
                replace=True,
            )
        return claim

    @_state_operation
    def release_memory_job(self, workspace: Path, *, claim: MemoryClaim) -> None:
        with self.persistence.transaction():
            try:
                self._assert_memory_claim(workspace, claim)
            except ContextSourceChangedError:
                return
            self.persistence.delete_metadata("memory-job:" + claim.scope_id)

    def memory_scope(self, workspace: Path) -> dict[str, Any] | None:
        from agent_runtime.harness.memory_identity import memory_scope

        return memory_scope(self, workspace)

    def thread_in_memory_scope(self, workspace: Path, thread_id: str) -> bool:
        from agent_runtime.harness.memory_identity import thread_in_memory_scope

        return thread_in_memory_scope(self, workspace, thread_id)

    def relocate_memory(
        self, *, previous_workspace: Path, workspace: Path, expected_version: int | None = None,
    ) -> dict[str, Any]:
        from agent_runtime.harness.memory_identity import relocate_memory

        return relocate_memory(self, previous_workspace, workspace, expected_version=expected_version)

    @_state_operation
    def create_thread(
        self,
        *,
        workspace: Path,
        thread_id: str | None = None,
    ) -> ThreadSnapshot:
        self.persistence.assert_writable()
        resolved_workspace = Path(workspace).resolve()
        if not resolved_workspace.is_dir():
            raise ValueError("workspace must be an existing directory")
        if thread_id is not None and (not isinstance(thread_id, str) or not thread_id.strip()):
            raise ValueError("thread_id must be non-empty when provided")
        thread_id = thread_id or f"thread_{uuid4().hex}"
        from agent_runtime.harness.workspace_stores import register_workspace_store

        register_workspace_store(resolved_workspace, self.persistence.path)
        with self.persistence.transaction():
            self._append_and_reduce(
                thread_id=thread_id,
                turn_id=None,
                record_type="thread_created",
                producer="runtime",
                payload={"workspace": str(resolved_workspace)},
            )
            from agent_runtime.harness.memory_identity import bind_memory_thread

            bind_memory_thread(self, thread_id, resolved_workspace)
        return self.read_thread(thread_id)

    @_state_operation
    def update_session_settings(
        self,
        thread_id: str,
        settings: Mapping[str, Any],
        *,
        expected_request_revision: int | None = None,
    ) -> None:
        frozen = _json_object(settings, field="settings")
        with self.persistence.transaction():
            thread = self.read_thread(thread_id)
            if (
                expected_request_revision is not None
                and thread.settings.get("request_settings_revision", 0) != expected_request_revision
            ):
                from agent_runtime.model_config_io import ConfigVersionConflict

                raise ConfigVersionConflict(
                    "Request settings changed in another Session; reopen before changing settings"
                )
            self._append_and_reduce(
                thread_id=thread_id,
                turn_id=None,
                record_type="session_settings_updated",
                producer="runtime",
                payload={"settings": frozen},
            )

    @_state_operation
    def fork_thread(self, *, from_turn_id: str) -> ThreadSnapshot:
        thread_id = f"thread_{uuid4().hex}"
        with self.persistence.transaction():
            source = self.persistence.turn_with_workspace(from_turn_id)
            if source is None:
                raise KeyError(f"unknown fork Turn: {from_turn_id}")
            if source["status"] not in {"completed", "failed", "cancelled"}:
                raise RuntimeError("only a terminal Turn can be a fork point")
            self._append_and_reduce(
                thread_id=thread_id,
                turn_id=None,
                record_type="thread_forked",
                producer="runtime",
                payload={
                    "workspace": source["workspace"],
                    "parent_thread_id": source["thread_id"],
                    "fork_turn_id": from_turn_id,
                },
            )
            from agent_runtime.harness.memory_identity import bind_memory_thread

            bind_memory_thread(self, thread_id, Path(source["workspace"]))
        return self.read_thread(thread_id)

    @_state_operation
    def start_turn(
        self,
        *,
        thread_id: str,
        user_message: str,
        binding_manifest: Mapping[str, Any],
        input_files: tuple[Mapping[str, Any], ...] = (),
        turn_id: str | None = None,
        turn_producer: str = "runtime",
    ) -> TurnSnapshot:
        if not isinstance(user_message, str) or not user_message.strip():
            raise ValueError("user_message must be non-empty")
        frozen_manifest = _json_object(binding_manifest, field="binding_manifest")
        frozen_input_files = tuple(_json_object(value, field="input file") for value in input_files)
        if turn_id is not None and (not isinstance(turn_id, str) or not turn_id):
            raise ValueError("turn_id must be non-empty when provided")
        if not isinstance(turn_producer, str) or not turn_producer:
            raise ValueError("turn_producer must be non-empty")
        turn_id = turn_id or f"turn_{uuid4().hex}"
        item_id = f"item_{uuid4().hex}"
        with self.persistence.transaction():
            thread = self.persistence.first("threads", where={"thread_id": thread_id})
            if thread is None:
                raise KeyError(f"unknown thread: {thread_id}")
            if thread["active_turn_id"] is not None:
                raise RuntimeError(f"thread already has an active turn: {thread_id}")
            turn_index = (self.persistence.maximum("turns", "turn_index", where={"thread_id": thread_id}) + 1,)[0]
            self._append_and_reduce(
                thread_id=thread_id,
                turn_id=turn_id,
                record_type="turn_started",
                producer=turn_producer,
                payload={
                    "turn_id": turn_id,
                    "predecessor_turn_id": thread["head_turn_id"],
                    "turn_index": turn_index,
                    "binding_manifest": frozen_manifest,
                },
            )
            self._append_and_reduce(
                thread_id=thread_id,
                turn_id=turn_id,
                record_type="item_started",
                producer="user",
                payload={"item_id": item_id, "kind": "user_message"},
            )
            self._append_and_reduce(
                thread_id=thread_id,
                turn_id=turn_id,
                record_type="item_completed",
                producer="user",
                payload={"item_id": item_id, "payload": {"text": user_message}},
            )
            for input_file in frozen_input_files:
                input_item_id = f"item_{uuid4().hex}"
                self._append_and_reduce(
                    thread_id=thread_id,
                    turn_id=turn_id,
                    record_type="item_started",
                    producer="user",
                    payload={"item_id": input_item_id, "kind": "input_file"},
                )
                self._append_and_reduce(
                    thread_id=thread_id,
                    turn_id=turn_id,
                    record_type="item_completed",
                    producer="user",
                    payload={"item_id": input_item_id, "payload": input_file},
                )
        return self.read_turn(turn_id)

    @_state_operation
    def user_input_revision(self, turn_id: str) -> int:
        """Count canonical user messages, independently of context compaction."""
        return int(
            (
                self.persistence.count(
                    "items", where={"turn_id": turn_id, "kind": "user_message", "status": "completed"}
                ),
            )[0]
        )

    @_state_operation
    def request_input_revision(self, turn_id: str, request_ref: Mapping[str, Any]) -> int | None:
        revision = request_ref.get("input_revision")
        if isinstance(revision, int) and not isinstance(revision, bool):
            return revision
        # Pre-steering rollouts have no version field. Their canonical request
        # Item still establishes which user messages preceded that request.
        items = self.list_items(turn_id)
        request = next(
            (
                item
                for item in items
                if item.kind == "model_request"
                and item.payload.get("request_ref", {}).get("request_id") == request_ref.get("request_id")
            ),
            None,
        )
        if request is None:
            return None
        return sum(
            item.kind == "user_message" and item.status == "completed" and item.sequence < request.sequence
            for item in items
        )

    @_state_operation
    def receive_turn_input(
        self,
        *,
        thread_id: str,
        turn_id: str,
        text: str,
        input_files: tuple[Mapping[str, Any], ...] = (),
    ) -> ItemSnapshot:
        """Durably accept steering for exactly the expected active Turn."""
        if not isinstance(text, str) or not text.strip():
            raise ValueError("input must be non-empty")
        item_id = f"item_{uuid4().hex}"
        with self.persistence.transaction():
            thread = self.read_thread(thread_id)
            turn = self.read_turn(turn_id) if thread.active_turn_id == turn_id else None
            if turn is None or turn.status not in {"running", "paused", "interrupted"}:
                raise RuntimeError("expected Turn is no longer active; input was not accepted")
            self._assert_no_pending_cancellation(turn_id)
            revision = self.user_input_revision(turn_id) + 1
            self._append_and_reduce(
                thread_id=thread_id,
                turn_id=turn_id,
                record_type="item_started",
                producer="user",
                payload={"item_id": item_id, "kind": "user_message"},
            )
            self._append_and_reduce(
                thread_id=thread_id,
                turn_id=turn_id,
                record_type="item_completed",
                producer="user",
                payload={"item_id": item_id, "payload": {"text": text, "input_revision": revision}},
            )
            for value in input_files:
                file_id = f"item_{uuid4().hex}"
                self._append_and_reduce(
                    thread_id=thread_id,
                    turn_id=turn_id,
                    record_type="item_started",
                    producer="user",
                    payload={"item_id": file_id, "kind": "input_file"},
                )
                self._append_and_reduce(
                    thread_id=thread_id,
                    turn_id=turn_id,
                    record_type="item_completed",
                    producer="user",
                    payload={"item_id": file_id, "payload": _json_object(value, field="input file")},
                )
        return self.read_item(item_id)

    @_state_operation
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
            if (
                operation.request_ref.get("purpose") == "agent_step"
                and (operation.request_ref.get("input_revision") or 0) >= item.payload["input_revision"]
                and operation.operation_id in dispatched
            ):
                return "applied"
        status = self.read_turn(item.turn_id).status
        return "pending" if status in {"running", "paused", "interrupted"} else "unconsumed"

    @_state_operation
    def complete_turn(
        self,
        *,
        turn_id: str,
        answer: str,
        producer: str = "runtime",
        expected_input_revision: int | None = None,
        source_item_id: str | None = None,
    ) -> TurnSnapshot:
        if not isinstance(answer, str) or not answer.strip():
            raise ValueError("answer must be non-empty")
        item_id = f"item_{uuid4().hex}"
        with self.persistence.transaction():
            if expected_input_revision is not None and self.user_input_revision(turn_id) != expected_input_revision:
                raise TurnInputChangedError("new user input superseded this final answer")
            turn = self.persistence.first("turns", where={"turn_id": turn_id})
            if turn is None:
                raise KeyError(f"unknown turn: {turn_id}")
            if turn["status"] != "running":
                raise RuntimeError(f"turn is not running: {turn_id}")
            thread_id = turn["thread_id"]
            active = self.persistence.first("threads", where={"thread_id": thread_id})
            if active is None or active["active_turn_id"] != turn_id:
                raise RuntimeError(f"turn does not own the thread active slot: {turn_id}")
            if source_item_id is not None:
                source = self.read_item(source_item_id)
                if (
                    source.turn_id != turn_id
                    or source.kind != "model_response"
                    or source.status != "completed"
                    or source.payload.get("text") != answer
                    or source.payload.get("response_status", "completed") != "completed"
                ):
                    raise ValueError("accepted answer requires the matching completed response from this Turn")
                item_id = source_item_id
            else:
                self._append_and_reduce(
                    thread_id=thread_id,
                    turn_id=turn_id,
                    record_type="item_started",
                    producer=producer,
                    payload={"item_id": item_id, "kind": "agent_message"},
                )
                self._append_and_reduce(
                    thread_id=thread_id,
                    turn_id=turn_id,
                    record_type="item_completed",
                    producer=producer,
                    payload={"item_id": item_id, "payload": {"text": answer}},
                )
            self._append_terminal_plan_state(
                thread_id=thread_id,
                turn_id=turn_id,
                status="complete",
            )
            self._append_and_reduce(
                thread_id=thread_id,
                turn_id=turn_id,
                record_type="turn_completed",
                producer=producer,
                payload={"turn_id": turn_id, "answer_item_id": item_id},
            )
        return self.read_turn(turn_id)

    @_state_operation
    def fail_turn(
        self,
        *,
        turn_id: str,
        reason_code: str,
        message: str,
    ) -> TurnSnapshot:
        if not isinstance(reason_code, str) or not reason_code.strip():
            raise ValueError("Turn failure reason_code must be non-empty")
        if not isinstance(message, str) or not message.strip():
            raise ValueError("Turn failure message must be non-empty")
        with self.persistence.transaction():
            turn = self.persistence.first("turns", where={"turn_id": turn_id})
            if turn is None:
                raise KeyError(f"unknown turn: {turn_id}")
            if turn["status"] != "running":
                raise RuntimeError("only a running Turn can fail")
            active = self.persistence.first("threads", where={"thread_id": turn["thread_id"]})
            if active is None or active["active_turn_id"] != turn_id:
                raise RuntimeError("failed Turn does not own the Thread active slot")
            self._append_terminal_plan_state(
                thread_id=turn["thread_id"],
                turn_id=turn_id,
                status="blocked",
                message=message.strip(),
            )
            self._append_and_reduce(
                thread_id=turn["thread_id"],
                turn_id=turn_id,
                record_type="turn_failed",
                producer="verifier",
                payload={
                    "turn_id": turn_id,
                    "reason_code": reason_code.strip(),
                    "message": message.strip(),
                },
            )
        return self.read_turn(turn_id)

    def _append_terminal_plan_state(
        self,
        *,
        thread_id: str,
        turn_id: str,
        status: str,
        message: str | None = None,
    ) -> None:
        latest = self.persistence.first(
            "items",
            where={"turn_id": turn_id, "kind": "plan_state", "status": "completed"},
            order_by="sequence",
            newest=True,
            limit=1,
        )
        if latest is None:
            return
        payload = latest["payload"]
        raw_plan = payload.get("plan")
        if not isinstance(raw_plan, Mapping):
            raise RuntimeError("canonical plan_state payload is malformed")
        revision = int(raw_plan["revision"]) + 1
        steps = [dict(step) for step in raw_plan.get("steps", ()) if isinstance(step, Mapping)]
        if status == "complete" and any(step.get("status") not in {"completed", "skipped"} for step in steps):
            status = "needs_replan"
        plan = {
            **dict(raw_plan),
            "revision": revision,
            "status": status,
            "active_step_id": None,
            "steps": steps,
        }
        event_type = "completed" if status == "complete" else status
        event = {
            "event_type": event_type,
            "message": message
            or (
                "Plan completed as reported; task validation is separate."
                if status == "complete"
                else (
                    "Turn accepted with unfinished plan steps; review remaining work."
                    if status == "needs_replan"
                    else "Plan blocked by the failed Turn; reported steps retained."
                )
            ),
            "tool_call_ids": [],
        }
        item_id = f"item_{uuid4().hex}"
        public_item_id = derive_plan_public_item_id(
            turn_id=turn_id,
            revision=revision,
        )
        self._append_and_reduce(
            thread_id=thread_id,
            turn_id=turn_id,
            record_type="item_started",
            producer="runtime",
            payload={
                "item_id": item_id,
                "kind": "plan_state",
                "public_item_id": public_item_id,
                "revision": revision,
            },
        )
        self._append_and_reduce(
            thread_id=thread_id,
            turn_id=turn_id,
            record_type="item_completed",
            producer="runtime",
            payload={
                "item_id": item_id,
                "public_item_id": public_item_id,
                "revision": revision,
                "payload": {"plan": plan, "event": event},
            },
        )

    @_state_operation
    def pause_turn(
        self,
        *,
        turn_id: str,
        reason: str,
        producer: str = "runtime",
    ) -> TurnSnapshot:
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError("Turn pause reason must be non-empty")
        if not isinstance(producer, str) or not producer.strip():
            raise ValueError("Turn pause producer must be non-empty")
        with self.persistence.transaction():
            turn = self.persistence.first("turns", where={"turn_id": turn_id})
            if turn is None:
                raise KeyError(f"unknown turn: {turn_id}")
            if turn["status"] != "running":
                raise RuntimeError("only a running Turn can pause")
            active = self.persistence.first("threads", where={"thread_id": turn["thread_id"]})
            if active is None or active["active_turn_id"] != turn_id:
                raise RuntimeError("paused Turn does not own the Thread active slot")
            self._append_and_reduce(
                thread_id=turn["thread_id"],
                turn_id=turn_id,
                record_type="turn_paused",
                producer=producer,
                payload={"turn_id": turn_id, "reason": reason},
            )
        return self.read_turn(turn_id)

    @_state_operation
    def interrupt_orphaned_turn(
        self,
        *,
        turn_id: str,
        reason: str,
        maintenance_confirmed: bool,
    ) -> TurnSnapshot:
        """Recover only the safe gap before any model/tool operation exists."""

        if not maintenance_confirmed:
            raise RuntimeError(
                "orphan recovery requires explicit maintenance confirmation that the prior worker is stopped"
            )
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError("Turn interruption reason must be non-empty")
        with self.persistence.transaction():
            turn = self.persistence.first("turns", where={"turn_id": turn_id})
            if turn is None:
                raise KeyError(f"unknown turn: {turn_id}")
            if turn["status"] != "running":
                raise RuntimeError("orphan recovery requires a running Turn")
            active = self.persistence.first("threads", where={"thread_id": turn["thread_id"]})
            if active is None or active["active_turn_id"] != turn_id:
                raise RuntimeError("orphaned Turn does not own the Thread active slot")
            operation_count = 0
            for table in (
                "model_operations",
                "tool_operations",
                "interactions",
                "approvals",
            ):
                row = {"count": self.persistence.count(table, where={"turn_id": turn_id})}
                operation_count += int(row["count"])
            if operation_count:
                raise RuntimeError(
                    "pre-dispatch orphan recovery refuses Turns with durable operations; "
                    "use operation-specific reconciliation"
                )
            self._append_and_reduce(
                thread_id=str(turn["thread_id"]),
                turn_id=turn_id,
                record_type="turn_interrupted",
                producer="recovery",
                payload={
                    "turn_id": turn_id,
                    "reason": reason,
                    "maintenance_confirmed": True,
                    "recovery_scope": "pre_dispatch_orphan",
                },
            )
        return self.read_turn(turn_id)

    @_state_operation
    def context_source_revision(self, turn_id: str) -> str:
        """Version every source Thread, including non-message state transitions."""
        thread_ids = {item.thread_id for item in self.list_context_items(turn_id)}
        thread_ids.add(self.read_turn(turn_id).thread_id)
        versions = [
            (thread_id, self.read_thread(thread_id).applied_thread_sequence) for thread_id in sorted(thread_ids)
        ]
        return hashlib.sha256(_canonical_json(versions).encode()).hexdigest()

    @_state_operation
    def context_durable_state(self, turn_id: str) -> dict[str, Any]:
        return _context_durable_state(self, turn_id)

    @_state_operation
    def record_context_compaction(
        self,
        *,
        turn_id: str,
        covered_item_ids: tuple[str, ...],
        summary: str,
        preserved_facts: Mapping[str, Any],
        artifact_refs: tuple[Mapping[str, Any], ...] = (),
        context_version: int,
        preserved_item_ids: tuple[str, ...] = (),
        expected_source_revision: str | None = None,
        durable_state: Mapping[str, Any] | None = None,
        algorithm_revision: str | None = None,
        tool_result_overrides: Mapping[str, str] | None = None,
        capacity_diagnostics: Mapping[str, Any] | None = None,
    ) -> ItemSnapshot:
        """Append an auditable replacement for one committed context prefix."""

        if not covered_item_ids or any(not isinstance(item_id, str) or not item_id for item_id in covered_item_ids):
            raise ValueError("compaction must cover non-empty Item IDs")
        if len(set(covered_item_ids)) != len(covered_item_ids):
            raise ValueError("compaction covered Item IDs must be unique")
        if not isinstance(summary, str) or not summary.strip():
            raise ValueError("compaction summary must be non-empty")
        if isinstance(context_version, bool) or not isinstance(context_version, int) or context_version < 1:
            raise ValueError("context_version must be a positive integer")

        with self.persistence.transaction():
            if (
                expected_source_revision is not None
                and self.context_source_revision(turn_id) != expected_source_revision
            ):
                raise ContextSourceChangedError("Context source changed before compaction commit.")
            turn = self.persistence.first("turns", where={"turn_id": turn_id})
            if turn is None:
                raise KeyError(f"unknown turn: {turn_id}")
            if turn["status"] != "running":
                raise RuntimeError("context compaction requires a running Turn")
            visible_items = tuple(item for item in self.list_context_items(turn_id) if item.status == "completed")
            visible_ids = tuple(item.item_id for item in visible_items)
            if visible_ids[: len(covered_item_ids)] != covered_item_ids:
                raise ValueError("compaction must cover a contiguous context prefix in exact order")
            if len(set(preserved_item_ids)) != len(preserved_item_ids) or any(
                item_id not in covered_item_ids for item_id in preserved_item_ids
            ):
                raise ValueError("preserved Items must belong to the covered prefix")
            if algorithm_revision in {"semantic-compaction-v5", "semantic-compaction-v6"}:
                pending: set[tuple[str, str]] = set()
                replay_results: set[str] = set()
                for item in visible_items:
                    if item.item_id not in preserved_item_ids:
                        continue
                    if item.kind == "model_response" and item.payload.get("tool_calls"):
                        if pending:
                            raise ValueError("preserved tool exchanges must be complete")
                        pending.update((item.turn_id, call["id"]) for call in item.payload["tool_calls"])
                    elif item.kind == "tool_result":
                        key = (item.turn_id, item.payload.get("tool_call_id"))
                        if key not in pending:
                            raise ValueError("preserved tool result must have its original call")
                        pending.remove(key)
                        replay_results.add(item.item_id)
                    elif item.kind != "user_message" or pending:
                        raise ValueError("only user Items and complete tool exchanges can be preserved")
                if pending or (
                    algorithm_revision == "semantic-compaction-v5"
                    and replay_results != set(tool_result_overrides or {})
                ):
                    raise ValueError("preserved tool exchanges require all paired archived results")
                for source_turn in {item.turn_id for item in visible_items if item.item_id in replay_results}:
                    if any(
                        op.result_item_id in replay_results
                        and (
                            op.requires_reconciliation
                            or op.status not in {"succeeded", "failed", "denied", "cancelled"}
                        )
                        for op in self.list_tool_operations(source_turn)
                    ):
                        raise ValueError("unsettled tool results cannot be archived in semantic replay")
            elif any(item.kind != "user_message" for item in visible_items if item.item_id in preserved_item_ids):
                raise ValueError("only user Items can be preserved inside a covered prefix")

            if tool_result_overrides is not None:
                covered_by_id = {item.item_id: item for item in visible_items[: len(covered_item_ids)]}
                if not tool_result_overrides or any(
                    target not in covered_by_id
                    or covered_by_id[target].kind != "tool_result"
                    or not isinstance(content, str)
                    or not content
                    for target, content in tool_result_overrides.items()
                ):
                    raise ValueError("tool result overrides require completed covered tool Items and nonempty text")

            frozen_facts = _json_object(
                preserved_facts,
                field="compaction preserved_facts",
            )
            required_fact_categories = {
                "architecture_and_safety_constraints",
                "file_changes",
                "verification_results",
                "unresolved_work",
                "uncertain_side_effects",
            }
            missing_categories = sorted(required_fact_categories - frozen_facts.keys())
            if missing_categories:
                raise ValueError(f"compaction preserved_facts lacks critical fact categories: {missing_categories}")
            frozen_artifacts = tuple(_json_object(value, field="compaction artifact ref") for value in artifact_refs)
            covered_items = visible_items[: len(covered_item_ids)]
            covered_refs = [
                {
                    "item_id": item.item_id,
                    "thread_id": item.thread_id,
                    "turn_id": item.turn_id,
                    "item_sequence": item.sequence,
                    "applied_thread_sequence": item.applied_thread_sequence,
                    "payload_sha256": hashlib.sha256(_canonical_json(dict(item.payload)).encode()).hexdigest(),
                }
                for item in covered_items
            ]
            grouped_sequences: dict[str, list[int]] = {}
            for item in covered_items:
                grouped_sequences.setdefault(item.thread_id, []).append(item.applied_thread_sequence)
            covered_ranges = [
                {
                    "thread_id": thread_id,
                    "start_thread_sequence": min(sequences),
                    "end_thread_sequence": max(sequences),
                }
                for thread_id, sequences in grouped_sequences.items()
            ]
            frozen_state = (
                _context_durable_state(self, turn_id)
                if durable_state is None
                else _json_object(durable_state, field="compaction durable_state")
            )
            item_id = f"item_{uuid4().hex}"
            payload = {
                "context_version": context_version,
                "covered_item_ids": list(covered_item_ids),
                "covered_item_refs": covered_refs,
                "covered_sequence_ranges": covered_ranges,
                "summary": summary,
                "preserved_facts": frozen_facts,
                "artifact_refs": list(frozen_artifacts),
                "durable_state": frozen_state,
            }
            if tool_result_overrides is not None:
                payload["tool_result_overrides"] = dict(tool_result_overrides)
            if capacity_diagnostics is not None:
                payload["capacity_diagnostics"] = _json_object(capacity_diagnostics, field="capacity diagnostics")
            if expected_source_revision is not None:
                payload.update(
                    source_revision=expected_source_revision,
                    preserved_item_ids=list(preserved_item_ids),
                    algorithm_revision=algorithm_revision,
                )
            self._append_and_reduce(
                thread_id=str(turn["thread_id"]),
                turn_id=turn_id,
                record_type="item_started",
                producer="runtime",
                payload={"item_id": item_id, "kind": "context_compaction"},
            )
            self._append_and_reduce(
                thread_id=str(turn["thread_id"]),
                turn_id=turn_id,
                record_type="item_completed",
                producer="runtime",
                payload={"item_id": item_id, "payload": payload},
            )
        return self._latest_item(turn_id)

    @_state_operation
    def cancel_turn(self, *, turn_id: str) -> TurnSnapshot:
        with self.persistence.transaction():
            turn = self.persistence.first("turns", where={"turn_id": turn_id})
            if turn is None:
                raise KeyError(f"unknown turn: {turn_id}")
            if turn["status"] not in {"paused", "interrupted"}:
                raise RuntimeError("only a safely paused or interrupted Turn can be cancelled")
            thread_id = str(turn["thread_id"])
            unsafe_operation = self.persistence.first(
                "tool_operations",
                where={
                    "turn_id": turn_id,
                    "status": (
                        "running",
                        "unknown",
                    ),
                },
                order_by="rowid",
                limit=1,
            )
            if unsafe_operation is not None:
                raise RuntimeError(
                    f"cancel cannot discard a running or unknown tool operation: {unsafe_operation['operation_id']}"
                )
            pending = self.persistence.rows(
                "interactions", where={"turn_id": turn_id, "status": "pending"}, order_by="rowid"
            )
            for interaction in pending:
                self._append_and_reduce(
                    thread_id=thread_id,
                    turn_id=turn_id,
                    record_type="interaction_resolved",
                    producer="user",
                    payload={
                        "request_id": interaction["request_id"],
                        "version": int(interaction["version"]) + 1,
                        "response": {"action": "abort"},
                        "approval_status": "denied",
                    },
                )
                if interaction["operation_id"] is not None:
                    operation = self.persistence.first(
                        "tool_operations", where={"operation_id": interaction["operation_id"]}
                    )
                    if operation is None or operation["status"] != "awaiting_approval":
                        raise RuntimeError("cancel cannot discard a started or unknown tool operation")
                    self._append_and_reduce(
                        thread_id=thread_id,
                        turn_id=turn_id,
                        record_type="tool_operation_approval_resolved",
                        producer="runtime",
                        payload={
                            "operation_id": interaction["operation_id"],
                            "request_id": interaction["request_id"],
                            "status": "cancelled",
                        },
                    )
            self._append_and_reduce(
                thread_id=thread_id,
                turn_id=turn_id,
                record_type="turn_cancelled",
                producer="user",
                payload={"turn_id": turn_id},
            )
        return self.read_turn(turn_id)

    @_state_operation
    def request_turn_cancellation(
        self,
        *,
        turn_id: str,
        producer: str = "runtime",
        reason: str = "controlling stream closed",
    ) -> TurnSnapshot:
        """Append the durable request before live work receives cancellation."""

        if not isinstance(producer, str) or not producer.strip():
            raise ValueError("cancellation producer must be non-empty")
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError("cancellation reason must be non-empty")
        with self.persistence.transaction():
            turn = self.persistence.first("turns", where={"turn_id": turn_id})
            if turn is None:
                raise KeyError(f"unknown turn: {turn_id}")
            if turn["status"] not in {"running", "paused", "interrupted"}:
                return self.read_turn(turn_id)
            thread_id = str(turn["thread_id"])
            active = self.persistence.first("threads", where={"thread_id": thread_id})
            if active is None or active["active_turn_id"] != turn_id:
                raise RuntimeError("cancellation request cannot target a foreign active slot")
            if not self.turn_cancellation_pending(turn_id):
                self._append_and_reduce(
                    thread_id=thread_id,
                    turn_id=turn_id,
                    record_type="turn_cancellation_requested",
                    producer=producer.strip(),
                    payload={
                        "turn_id": turn_id,
                        "reason": reason.strip()[:2_000],
                    },
                )
        return self.read_turn(turn_id)

    @_state_operation
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

    def _assert_no_pending_cancellation(self, turn_id: str) -> None:
        # Every path that resumes a paused Turn on behalf of user input must
        # respect a durable cancellation request; an explicit
        # resume_cancelled_turn is the only way past it.
        if self.turn_cancellation_pending(turn_id):
            raise RuntimeError("Turn cancellation was requested; input was not accepted")

    @_state_operation
    def resume_cancelled_turn(self, *, turn_id: str) -> TurnSnapshot:
        """Explicitly continue a quiescent cancellation without replaying results."""
        with self.persistence.transaction():
            turn = self.read_turn(turn_id)
            if turn.status != "paused" or not self.turn_cancellation_pending(turn_id):
                raise RuntimeError("Turn has no paused cancellation to resume")
            if self.read_thread(turn.thread_id).active_turn_id != turn_id:
                raise RuntimeError("cancelled Turn does not own the active slot")
            if (
                any(item.status == "pending" for item in self.list_interactions(turn_id))
                or any(op.status in {"running", "unknown"} for op in self.list_tool_operations(turn_id))
                or any(op.status in {"dispatched", "unknown"} for op in self.list_model_operations(turn_id))
            ):
                raise RuntimeError("pending or uncertain execution must be resolved before resume")
            self._append_and_reduce(
                thread_id=turn.thread_id,
                turn_id=turn_id,
                record_type="turn_resumed",
                producer="user",
                payload={"turn_id": turn_id, "reason": "explicit cancellation resume"},
            )
        return self.read_turn(turn_id)

    @_state_operation
    def request_clarification(
        self,
        *,
        turn_id: str,
        question: str,
    ) -> InteractionSnapshot:
        if not isinstance(question, str) or not question.strip():
            raise ValueError("clarification question must be non-empty")
        request_id = f"interaction_{uuid4().hex}"
        with self.persistence.transaction():
            thread_id = self._running_turn_thread_id(turn_id)
            self._append_and_reduce(
                thread_id=thread_id,
                turn_id=turn_id,
                record_type="interaction_requested",
                producer="verifier",
                payload={
                    "request_id": request_id,
                    "kind": "clarification",
                    "operation_id": None,
                    "request": {"question": question},
                },
            )
            self._append_and_reduce(
                thread_id=thread_id,
                turn_id=turn_id,
                record_type="turn_paused",
                producer="verifier",
                payload={"turn_id": turn_id, "request_id": request_id},
            )
        return self.read_interaction(request_id)

    @_state_operation
    def resolve_clarification(
        self,
        *,
        turn_id: str,
        request_id: str,
        response: str,
    ) -> InteractionSnapshot:
        if not isinstance(response, str) or not response.strip():
            raise ValueError("clarification response must be non-empty")
        self._assert_artifact_integrity(turn_id)
        item_id = f"item_{uuid4().hex}"
        with self.persistence.transaction():
            turn = self.persistence.first("turns", where={"turn_id": turn_id})
            if turn is None:
                raise KeyError(f"unknown turn: {turn_id}")
            if turn["status"] != "paused":
                raise RuntimeError("clarification response requires a paused Turn")
            self._assert_no_pending_cancellation(turn_id)
            interaction = self.persistence.first("interactions", where={"request_id": request_id, "turn_id": turn_id})
            if interaction is None:
                raise KeyError(f"unknown clarification request: {request_id}")
            if interaction["kind"] != "clarification" or interaction["status"] != "pending":
                raise RuntimeError("interaction is not a pending clarification")
            thread_id = str(turn["thread_id"])
            self._append_and_reduce(
                thread_id=thread_id,
                turn_id=turn_id,
                record_type="interaction_resolved",
                producer="user",
                payload={
                    "request_id": request_id,
                    "version": int(interaction["version"]) + 1,
                    "response": {"text": response},
                },
            )
            self._append_and_reduce(
                thread_id=thread_id,
                turn_id=turn_id,
                record_type="item_started",
                producer="user",
                payload={"item_id": item_id, "kind": "user_message"},
            )
            self._append_and_reduce(
                thread_id=thread_id,
                turn_id=turn_id,
                record_type="item_completed",
                producer="user",
                payload={"item_id": item_id, "payload": {"text": response}},
            )
            self._append_and_reduce(
                thread_id=thread_id,
                turn_id=turn_id,
                record_type="turn_resumed",
                producer="runtime",
                payload={"turn_id": turn_id, "request_id": request_id},
            )
        return self.read_interaction(request_id)

    @_state_operation
    def request_choice(
        self,
        *,
        turn_id: str,
        question: str,
        options: tuple[str, ...],
    ) -> InteractionSnapshot:
        if not isinstance(question, str) or not question.strip():
            raise ValueError("choice question must be non-empty")
        if (
            len(options) < 2
            or any(not isinstance(option, str) or not option.strip() for option in options)
            or len(set(options)) != len(options)
        ):
            raise ValueError("choice options must contain at least two unique non-empty values")
        request_id = f"interaction_{uuid4().hex}"
        with self.persistence.transaction():
            thread_id = self._running_turn_thread_id(turn_id)
            self._append_and_reduce(
                thread_id=thread_id,
                turn_id=turn_id,
                record_type="interaction_requested",
                producer="runtime",
                payload={
                    "request_id": request_id,
                    "kind": "choice",
                    "operation_id": None,
                    "request": {"question": question, "options": list(options)},
                },
            )
            self._append_and_reduce(
                thread_id=thread_id,
                turn_id=turn_id,
                record_type="turn_paused",
                producer="runtime",
                payload={"turn_id": turn_id, "request_id": request_id},
            )
        return self.read_interaction(request_id)

    @_state_operation
    def resolve_choice(
        self,
        *,
        turn_id: str,
        request_id: str,
        selection: str,
    ) -> InteractionSnapshot:
        if not isinstance(selection, str) or not selection.strip():
            raise ValueError("choice selection must be non-empty")
        self._assert_artifact_integrity(turn_id)
        item_id = f"item_{uuid4().hex}"
        with self.persistence.transaction():
            turn = self.persistence.first("turns", where={"turn_id": turn_id})
            if turn is None:
                raise KeyError(f"unknown turn: {turn_id}")
            if turn["status"] != "paused":
                raise RuntimeError("choice response requires a paused Turn")
            self._assert_no_pending_cancellation(turn_id)
            interaction = self.persistence.first("interactions", where={"request_id": request_id, "turn_id": turn_id})
            if interaction is None:
                raise KeyError(f"unknown choice request: {request_id}")
            if interaction["kind"] != "choice" or interaction["status"] != "pending":
                raise RuntimeError("interaction is not a pending choice")
            request = interaction["request"]
            options = request.get("options")
            if not isinstance(options, list) or selection not in options:
                raise ValueError("choice selection is not one of the frozen options")
            thread_id = str(turn["thread_id"])
            self._append_and_reduce(
                thread_id=thread_id,
                turn_id=turn_id,
                record_type="interaction_resolved",
                producer="user",
                payload={
                    "request_id": request_id,
                    "version": int(interaction["version"]) + 1,
                    "response": {"selection": selection},
                },
            )
            self._append_and_reduce(
                thread_id=thread_id,
                turn_id=turn_id,
                record_type="item_started",
                producer="user",
                payload={"item_id": item_id, "kind": "user_message"},
            )
            self._append_and_reduce(
                thread_id=thread_id,
                turn_id=turn_id,
                record_type="item_completed",
                producer="user",
                payload={"item_id": item_id, "payload": {"text": selection}},
            )
            self._append_and_reduce(
                thread_id=thread_id,
                turn_id=turn_id,
                record_type="turn_resumed",
                producer="runtime",
                payload={"turn_id": turn_id, "request_id": request_id},
            )
        return self.read_interaction(request_id)

    @_state_operation
    def prepare_model_operation(
        self,
        *,
        turn_id: str,
        request_hash: str,
        context_hash: str,
        tool_hash: str,
        wire_hash: str,
        request_ref: Mapping[str, Any],
    ) -> ModelOperationSnapshot:
        hashes = {
            "request_hash": request_hash,
            "context_hash": context_hash,
            "tool_hash": tool_hash,
            "wire_hash": wire_hash,
        }
        if not all(isinstance(value, str) and value for value in hashes.values()):
            raise ValueError("model operation hashes must be non-empty strings")
        frozen_ref = _json_object(request_ref, field="request_ref")
        request_id = frozen_ref.get("request_id")
        if not isinstance(request_id, str) or not request_id.strip():
            raise ValueError("model request_ref.request_id must be a non-empty string")
        operation_id = f"modelop_{uuid4().hex}"
        attempt_id = f"attempt_{uuid4().hex}"
        request_item_id = f"item_{uuid4().hex}"
        selected_operation_id = operation_id
        with self.persistence.transaction():
            turn = self.persistence.first("turns", where={"turn_id": turn_id})
            if turn is None:
                raise KeyError(f"unknown turn: {turn_id}")
            if turn["status"] != "running":
                raise RuntimeError(f"turn is not running: {turn_id}")
            thread_id = turn["thread_id"]
            matches = [
                row
                for row in self.persistence.rows("model_operations", where={"turn_id": turn_id}, order_by="rowid")
                if row["request_ref"].get("request_id") == request_id
            ]
            if len(matches) > 1:
                raise RuntimeError(f"duplicate durable model request identity: {request_id}")
            if matches:
                existing = matches[0]
                existing_hashes = {name: existing[name] for name in hashes}
                existing_ref = existing["request_ref"]
                if existing_hashes != hashes or existing_ref != frozen_ref:
                    raise RuntimeError(f"model request identity reused with conflicting payload: {request_id}")
                selected_operation_id = str(existing["operation_id"])
            else:
                self._append_and_reduce(
                    thread_id=thread_id,
                    turn_id=turn_id,
                    record_type="model_operation_prepared",
                    producer="runtime",
                    payload={
                        "operation_id": operation_id,
                        "attempt_id": attempt_id,
                        "generation": 1,
                        "public_item_ids": _model_public_item_ids(
                            turn_id=turn_id,
                            attempt_id=attempt_id,
                        ),
                        **hashes,
                        "request_ref": frozen_ref,
                    },
                )
                self._append_and_reduce(
                    thread_id=thread_id,
                    turn_id=turn_id,
                    record_type="item_started",
                    producer="runtime",
                    payload={"item_id": request_item_id, "kind": "model_request"},
                )
                self._append_and_reduce(
                    thread_id=thread_id,
                    turn_id=turn_id,
                    record_type="item_completed",
                    producer="runtime",
                    payload={
                        "item_id": request_item_id,
                        "payload": {**hashes, "request_ref": frozen_ref},
                    },
                )
        row = self.persistence.first("model_operations", where={"operation_id": selected_operation_id})
        if row is None:
            raise RuntimeError("prepared model operation projection is missing")
        return _model_operation_snapshot(row)

    @_state_operation
    def supersede_prepared_model(self, operation_id: str) -> None:
        """Close an undispatched stale request without reserving or spending budget."""
        with self.persistence.transaction():
            operation = next(op for op in self.list_model_operations() if op.operation_id == operation_id)
            if operation.status != "prepared":
                raise RuntimeError("only an undispatched prepared model can be superseded")
            self._append_and_reduce(
                thread_id=operation.thread_id,
                turn_id=operation.turn_id,
                record_type="model_attempt_rejected",
                producer="runtime",
                payload={
                    "operation_id": operation_id,
                    "attempt_id": operation.active_attempt_id,
                    "generation": operation.generation,
                    "reason": "new user input before dispatch",
                    "error_type": "input_superseded",
                },
            )

    @_state_operation
    def dispatch_model_attempt(
        self,
        operation_id: str,
        *,
        worker_id: str = "direct-model-worker",
        lease_seconds: float = 300.0,
        now: float | None = None,
        resource_request: ResourceUsage | None = None,
        allow_protected_budget: bool = False,
    ) -> ModelAttemptSnapshot:
        if not isinstance(worker_id, str) or not worker_id.strip():
            raise ValueError("model worker_id must be non-empty")
        if lease_seconds <= 0:
            raise ValueError("model lease_seconds must be positive")
        if resource_request is not None and not isinstance(resource_request, ResourceUsage):
            raise TypeError("resource_request must be ResourceUsage or None")
        if type(allow_protected_budget) is not bool:
            raise TypeError("allow_protected_budget must be a bool")
        row = self.persistence.first("model_operations", where={"operation_id": operation_id})
        if row is None:
            raise KeyError(f"unknown model operation: {operation_id}")
        self._assert_artifact_integrity(str(row["turn_id"]))
        dispatched_at = time.time() if now is None else float(now)
        with self.persistence.transaction():
            operation = self.persistence.first("model_operations", where={"operation_id": operation_id})
            if operation is None:
                raise KeyError(f"unknown model operation: {operation_id}")
            if operation["status"] != "prepared":
                raise RuntimeError(f"model operation is not prepared: {operation_id}")
            request_ref = operation["request_ref"]
            revision = self.request_input_revision(str(operation["turn_id"]), request_ref)
            if (
                request_ref.get("purpose") == "agent_step"
                and revision is not None
                and revision != self.user_input_revision(str(operation["turn_id"]))
            ):
                raise TurnInputChangedError("new user input before model dispatch")
            plan = request_ref.get("compaction_plan")
            if request_ref.get("purpose") == "context_summary" and plan is not None:
                summary_ops = [
                    op
                    for op in self.list_model_operations(str(operation["turn_id"]))
                    if op.request_ref.get("purpose") == "context_summary"
                ]
                for limit_key, selected in (
                    ("max_summary_calls", summary_ops),
                    (
                        "planned_summary_calls",
                        [
                            op
                            for op in summary_ops
                            if op.request_ref.get("compaction_plan", {}).get("plan_id") == plan.get("plan_id")
                        ],
                    ),
                ):
                    limit = plan.get(limit_key)
                    if limit is None:  # Legacy prepared records may lack the new call bound.
                        continue
                    if type(limit) is not int or limit < 1:
                        raise ValueError("summary call allowance must be a positive integer")
                    dispatched = sum(
                        attempt.status != "prepared"
                        for op in selected
                        for attempt in self.list_model_attempts(op.operation_id)
                    )
                    if dispatched >= limit:
                        raise BudgetLimitExceededError(
                            resource="model_calls", limit=limit, current_exposure=dispatched, requested=1
                        )
            reservation_id: str | None = None
            if resource_request is not None:
                attempt_id = str(operation["active_attempt_id"])
                reservation_id = f"reservation_{attempt_id}"
                budget_state = self.read_budget_state(str(operation["turn_id"]))
                continuation_reserve = 0
                if request_ref.get("purpose") == "context_summary" and plan is not None:
                    continuation_reserve = plan.get("continuation_token_reserve")
                    if type(continuation_reserve) is not int or continuation_reserve < 1:
                        raise ValueError("summary plan must preserve a positive continuation token reserve")
                    continuation_cost = plan.get("continuation_cost_reserve", 0)
                    if type(continuation_cost) is not int or continuation_cost < 0:
                        raise ValueError("summary continuation cost reserve must be non-negative")
                    cost_limit = budget_state.limits.cost_micros
                    requested_cost = resource_request.cost_micros + continuation_cost
                    if cost_limit is not None and budget_state.exposure.cost_micros + requested_cost > cost_limit:
                        raise BudgetLimitExceededError(
                            resource="cost_micros",
                            limit=cost_limit,
                            current_exposure=budget_state.exposure.cost_micros,
                            requested=requested_cost,
                        )
                ensure_token_headroom(
                    budget_state,
                    requested_tokens=resource_request.total_tokens + continuation_reserve,
                    allow_protected=allow_protected_budget and not continuation_reserve,
                )
                mutation = reserve_budget(
                    budget_state,
                    reservation_id=reservation_id,
                    operation_id=operation_id,
                    attempt_id=attempt_id,
                    amount=resource_request,
                )
                self._append_and_reduce(
                    thread_id=operation["thread_id"],
                    turn_id=operation["turn_id"],
                    record_type="budget_reserved",
                    producer="runtime",
                    payload={
                        "reservation_id": reservation_id,
                        "scope_id": mutation.reservation.scope_id,
                        "operation_id": operation_id,
                        "attempt_id": attempt_id,
                        "reserved": resource_request.to_dict(),
                    },
                )
            self._append_and_reduce(
                thread_id=operation["thread_id"],
                turn_id=operation["turn_id"],
                record_type="model_attempt_dispatched",
                producer="runtime",
                payload={
                    "operation_id": operation_id,
                    "attempt_id": operation["active_attempt_id"],
                    "generation": operation["generation"],
                    "claim_owner": worker_id,
                    "lease_expires_at": dispatched_at + lease_seconds,
                },
            )
            if reservation_id is not None:
                self._append_and_reduce(
                    thread_id=operation["thread_id"],
                    turn_id=operation["turn_id"],
                    record_type="budget_dispatched",
                    producer="runtime",
                    payload={"reservation_id": reservation_id},
                )
        return self.list_model_attempts(operation_id)[-1]

    @_state_operation
    def expire_model_attempt_dispatch(
        self,
        *,
        operation_id: str,
        now: float | None = None,
    ) -> ModelAttemptSnapshot:
        observed_at = time.time() if now is None else float(now)
        with self.persistence.transaction():
            operation = self.persistence.first("model_operations", where={"operation_id": operation_id})
            if operation is None:
                raise KeyError(f"unknown model operation: {operation_id}")
            if operation["status"] != "dispatched":
                raise RuntimeError("only a dispatched model attempt can expire")
            attempt = self.persistence.first("model_attempts", where={"attempt_id": operation["active_attempt_id"]})
            if attempt is None:
                raise RuntimeError("active model attempt projection is missing")
            expires_at = attempt["lease_expires_at"]
            if expires_at is None or observed_at <= float(expires_at):
                raise RuntimeError("model attempt dispatch lease has not expired")
            self._append_and_reduce(
                thread_id=operation["thread_id"],
                turn_id=operation["turn_id"],
                record_type="model_attempt_unknown",
                producer="recovery",
                payload={
                    "operation_id": operation_id,
                    "attempt_id": attempt["attempt_id"],
                    "generation": attempt["generation"],
                    "reason": "dispatch lease expired",
                    "observed_at": observed_at,
                },
            )
            self._append_budget_outcome(
                operation=operation,
                attempt_id=str(attempt["attempt_id"]),
                outcome="unknown",
                producer="recovery",
            )
            self._close_started_model_outputs(
                operation=operation, attempt_id=str(attempt["attempt_id"]),
                status="outcome_unknown", error="model dispatch lease expired", channel_content={},
            )
            turn = self.persistence.first("turns", where={"turn_id": operation["turn_id"]})
            if turn is not None and turn["status"] == "running":
                self._append_and_reduce(
                    thread_id=operation["thread_id"],
                    turn_id=operation["turn_id"],
                    record_type="turn_interrupted",
                    producer="recovery",
                    payload={
                        "turn_id": operation["turn_id"],
                        "operation_id": operation_id,
                        "reason": "model outcome unknown after lease expiry",
                    },
                )
        return self.list_model_attempts(operation_id)[-1]

    @_state_operation
    def mark_model_attempt_unknown(
        self,
        *,
        operation_id: str,
        attempt_id: str,
        generation: int,
        error_type: str | None = None,
        error_message: str | None = None,
        channel_content: Mapping[str, str] | None = None,
        outputs: tuple[ModelOutput, ...] = (),
    ) -> ModelAttemptSnapshot:
        if error_type is not None and (not isinstance(error_type, str) or not error_type.strip()):
            raise ValueError("model dispatch error_type must be non-empty")
        if error_message is not None and (not isinstance(error_message, str) or not error_message.strip()):
            raise ValueError("model dispatch error_message must be non-empty")
        frozen_channel_content = _model_channel_content(channel_content)
        with self.persistence.transaction():
            operation = self.persistence.first("model_operations", where={"operation_id": operation_id})
            if operation is None:
                raise KeyError(f"unknown model operation: {operation_id}")
            if not (
                operation["status"] == "dispatched"
                and operation["active_attempt_id"] == attempt_id
                and operation["generation"] == generation
            ):
                raise RuntimeError("only the active dispatched model attempt can become unknown")
            payload: dict[str, Any] = {
                "operation_id": operation_id,
                "attempt_id": attempt_id,
                "generation": generation,
            }
            if error_type is not None:
                payload["error_type"] = error_type.strip()
            if error_message is not None:
                payload["error_message"] = error_message.strip()[:2_000]
            self._append_and_reduce(
                thread_id=operation["thread_id"],
                turn_id=operation["turn_id"],
                record_type="model_attempt_unknown",
                producer="runtime",
                payload=payload,
            )
            self._append_budget_outcome(
                operation=operation,
                attempt_id=attempt_id,
                outcome="unknown",
            )
            self._close_started_model_outputs(
                operation=operation,
                attempt_id=attempt_id,
                status="outcome_unknown",
                error=(error_message.strip()[:2_000] if error_message is not None else "model outcome is unknown"),
                channel_content=frozen_channel_content,
                outputs=outputs,
            )
            turn = self.persistence.first("turns", where={"turn_id": operation["turn_id"]})
            if turn is not None and turn["status"] == "running":
                self._append_and_reduce(
                    thread_id=operation["thread_id"],
                    turn_id=operation["turn_id"],
                    record_type="turn_interrupted",
                    producer="runtime",
                    payload={
                        "turn_id": operation["turn_id"],
                        "operation_id": operation_id,
                        "reason": "outcome_unknown",
                    },
                )
        return self.list_model_attempts(operation_id)[-1]

    @_state_operation
    def reject_model_attempt(
        self,
        *,
        operation_id: str,
        attempt_id: str,
        generation: int,
        reason: str,
        error_type: str | None = None,
        error_message: str | None = None,
        channel_content: Mapping[str, str] | None = None,
        outputs: tuple[ModelOutput, ...] = (),
        budget_outcome: str = "released",
    ) -> ModelAttemptSnapshot:
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError("model rejection reason must be non-empty")
        if budget_outcome not in {"released", "unknown"}:
            raise ValueError("model rejection budget_outcome is unsupported")
        if error_type is not None and (not isinstance(error_type, str) or not error_type.strip()):
            raise ValueError("model rejection error_type must be non-empty")
        if error_message is not None and (not isinstance(error_message, str) or not error_message.strip()):
            raise ValueError("model rejection error_message must be non-empty")
        frozen_channel_content = _model_channel_content(channel_content)
        with self.persistence.transaction():
            operation = self.persistence.first("model_operations", where={"operation_id": operation_id})
            if operation is None:
                raise KeyError(f"unknown model operation: {operation_id}")
            if not (
                operation["status"] == "dispatched"
                and operation["active_attempt_id"] == attempt_id
                and operation["generation"] == generation
            ):
                raise RuntimeError("only the active dispatched model attempt can be rejected")
            payload: dict[str, Any] = {
                "operation_id": operation_id,
                "attempt_id": attempt_id,
                "generation": generation,
                "reason": reason,
            }
            if error_type is not None:
                payload["error_type"] = error_type.strip()[:120]
            if error_message is not None:
                payload["error_message"] = error_message.strip()[:2_000]
            self._append_and_reduce(
                thread_id=operation["thread_id"],
                turn_id=operation["turn_id"],
                record_type="model_attempt_rejected",
                producer="runtime",
                payload=payload,
            )
            self._append_budget_outcome(
                operation=operation,
                attempt_id=attempt_id,
                outcome=budget_outcome,
            )
            self._close_started_model_outputs(
                operation=operation,
                attempt_id=attempt_id,
                status="failed",
                error=(error_message.strip()[:2_000] if error_message is not None else reason.strip()[:2_000]),
                channel_content=frozen_channel_content,
                outputs=outputs,
            )
        return self.list_model_attempts(operation_id)[-1]

    def _close_started_model_outputs(
        self,
        *,
        operation: Mapping[str, Any],
        attempt_id: str,
        status: str,
        error: str,
        channel_content: Mapping[str, str],
        outputs: tuple[ModelOutput, ...] = (),
    ) -> None:
        from agent_runtime.harness.model_outputs import close_model_outputs

        close_model_outputs(self, attempt_id=attempt_id, status=status, error=error, partial_outputs=outputs)
        if status not in {"failed", "cancelled", "outcome_unknown"}:
            raise ValueError("model Item terminal status is unsupported")
        started = self.persistence.rows(
            "rollout_records",
            where={"turn_id": operation["turn_id"], "record_type": "item_started"},
            payload_fields={
                "attempt_id": attempt_id,
                "channel": (
                    "agent_message",
                    "reasoning",
                    "plan",
                ),
            },
            order_by="record_id",
        )
        seen_channels: set[str] = set()
        for row in started:
            start_payload = row["payload"]
            channel = str(start_payload["channel"])
            if channel in seen_channels:
                raise RuntimeError(f"model {channel} channel has duplicate durable starts")
            seen_channels.add(channel)
            item_id = str(start_payload["item_id"])
            completed = self.persistence.first(
                "rollout_records",
                where={"turn_id": operation["turn_id"], "record_type": "item_completed"},
                payload_fields={"item_id": item_id},
                limit=1,
            )
            if completed is not None:
                continue
            content = channel_content.get(channel, "")
            item_payload: dict[str, Any]
            if channel == "agent_message":
                item_payload = {"text": content, "tool_calls": ()}
            else:
                item_payload = {"content": content}
            self._append_and_reduce(
                thread_id=str(operation["thread_id"]),
                turn_id=str(operation["turn_id"]),
                record_type="item_completed",
                producer="model",
                payload={
                    "item_id": item_id,
                    "attempt_id": attempt_id,
                    "channel": channel,
                    "public_item_id": derive_model_public_item_id(
                        turn_id=str(operation["turn_id"]),
                        model_attempt_id=attempt_id,
                        channel=channel,
                    ),
                    "status": status,
                    "error": error,
                    "payload": item_payload,
                },
            )

    @_state_operation
    def cancel_model_attempt(
        self,
        *,
        operation_id: str,
        attempt_id: str,
        generation: int,
        reason: str,
        channel_content: Mapping[str, str] | None = None,
        outputs: tuple[ModelOutput, ...] = (),
    ) -> ModelAttemptSnapshot:
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError("model cancellation reason must be non-empty")
        frozen_channel_content = _model_channel_content(channel_content)
        with self.persistence.transaction():
            operation = self.persistence.first("model_operations", where={"operation_id": operation_id})
            if operation is None:
                raise KeyError(f"unknown model operation: {operation_id}")
            if not (
                operation["status"] == "dispatched"
                and operation["active_attempt_id"] == attempt_id
                and operation["generation"] == generation
            ):
                raise RuntimeError("only the active dispatched model attempt can be cancelled")
            self._append_and_reduce(
                thread_id=str(operation["thread_id"]),
                turn_id=str(operation["turn_id"]),
                record_type="model_attempt_cancelled",
                producer="runtime",
                payload={
                    "operation_id": operation_id,
                    "attempt_id": attempt_id,
                    "generation": generation,
                    "reason": reason.strip()[:2_000],
                },
            )
            # Provider-side cancellation does not prove zero billing. Keep the
            # reservation as uncertain exposure unless usage is later reconciled.
            self._append_budget_outcome(
                operation=operation,
                attempt_id=attempt_id,
                outcome="unknown",
            )
            self._close_started_model_outputs(
                operation=operation,
                attempt_id=attempt_id,
                status="cancelled",
                error=reason.strip()[:2_000],
                channel_content=frozen_channel_content,
                outputs=outputs,
            )
            self._append_and_reduce(
                thread_id=str(operation["thread_id"]),
                turn_id=str(operation["turn_id"]),
                record_type="turn_cancelled",
                producer="runtime",
                payload={
                    "turn_id": str(operation["turn_id"]),
                    "reason": reason.strip()[:2_000],
                },
            )
        return self.list_model_attempts(operation_id)[-1]

    @_state_operation
    def prepare_model_retry(self, operation_id: str) -> ModelAttemptSnapshot:
        row = self.persistence.first("model_operations", where={"operation_id": operation_id})
        if row is None:
            raise KeyError(f"unknown model operation: {operation_id}")
        self._assert_artifact_integrity(str(row["turn_id"]))
        with self.persistence.transaction():
            operation = self.persistence.first("model_operations", where={"operation_id": operation_id})
            if operation is None:
                raise KeyError(f"unknown model operation: {operation_id}")
            if operation["status"] != "unknown":
                raise RuntimeError(f"model operation is not unknown: {operation_id}")
            attempt_id = f"attempt_{uuid4().hex}"
            generation = operation["generation"] + 1
            self._append_and_reduce(
                thread_id=operation["thread_id"],
                turn_id=operation["turn_id"],
                record_type="model_retry_prepared",
                producer="runtime",
                payload={
                    "operation_id": operation_id,
                    "previous_attempt_id": operation["active_attempt_id"],
                    "attempt_id": attempt_id,
                    "generation": generation,
                    "public_item_ids": _model_public_item_ids(
                        turn_id=str(operation["turn_id"]),
                        attempt_id=attempt_id,
                    ),
                },
            )
            turn = self.persistence.first("turns", where={"turn_id": operation["turn_id"]})
            if turn is None or turn["status"] not in {"paused", "interrupted"}:
                raise RuntimeError("model retry requires a paused or interrupted Turn")
            self._append_and_reduce(
                thread_id=operation["thread_id"],
                turn_id=operation["turn_id"],
                record_type="turn_resumed",
                producer="runtime",
                payload={
                    "turn_id": operation["turn_id"],
                    "operation_id": operation_id,
                },
            )
        return self.list_model_attempts(operation_id)[-1]

    @_state_operation
    def complete_model_attempt(
        self,
        *,
        operation_id: str,
        attempt_id: str,
        generation: int,
        text: str,
        provider_response_id: str | None,
        usage: Mapping[str, Any],
        tool_calls: tuple[Mapping[str, Any], ...] = (),
        response_status: str = "completed",
        incomplete_reason: str | None = None,
        reasoning_content: str | None = None,
        plan_content: str | None = None,
        outputs: tuple[ModelOutput, ...] | None = None,
        stop_reason: str | None = None,
        now: float | None = None,
    ) -> bool:
        if not isinstance(text, str):
            raise TypeError("model response text must be a string")
        frozen_tool_calls = tuple(_json_object(call, field="model tool call") for call in tool_calls)
        if response_status not in {"completed", "incomplete"}:
            raise ValueError("model response status is unsupported")
        if response_status == "completed" and incomplete_reason is not None:
            raise ValueError("completed model response cannot have an incomplete reason")
        if response_status == "incomplete" and (
            not isinstance(incomplete_reason, str) or not incomplete_reason.strip()
        ):
            raise ValueError("incomplete model response requires a reason")
        if response_status == "completed" and not text.strip() and not frozen_tool_calls:
            raise ValueError("model response must contain text or tool calls")
        for channel, content in (
            ("reasoning", reasoning_content),
            ("plan", plan_content),
        ):
            if content is not None and not isinstance(content, str):
                raise TypeError(f"model {channel} content must be a string or None")
        frozen_usage = _json_object(usage, field="usage")
        response_item_id: str | None = None
        observed_at = time.time() if now is None else float(now)
        with self.persistence.transaction():
            operation = self.persistence.first("model_operations", where={"operation_id": operation_id})
            if operation is None:
                raise KeyError(f"unknown model operation: {operation_id}")
            # Durable state is the fencing authority. Lease expiry only opens
            # the recovery takeover (expire_model_attempt_dispatch), which
            # flips the status to unknown inside the same serialized
            # transaction. While the operation still points at this attempt
            # and generation with no committed response, this completion is
            # authoritative even past the lease clock: a billed response must
            # not be dropped on a timing boundary.
            is_current = (
                operation["status"] == "dispatched"
                and operation["active_attempt_id"] == attempt_id
                and operation["generation"] == generation
                and operation["response_item_id"] is None
            )
            if not is_current:
                return False
            internal_summary = operation["request_ref"].get("purpose") == "context_summary"
            if outputs is not None and not internal_summary:
                from agent_runtime.harness.model_outputs import finish_model_outputs

                finish_model_outputs(
                    self, operation_id=operation_id, attempt_id=attempt_id,
                    generation=generation, outputs=outputs,
                )
            started_channels = self.persistence.rows(
                "rollout_records",
                where={"turn_id": operation["turn_id"], "record_type": "item_started"},
                payload_fields={
                    "attempt_id": attempt_id,
                    "channel": (
                        "agent_message",
                        "reasoning",
                        "plan",
                    ),
                },
                order_by="record_id",
            )
            started_by_channel: dict[str, Mapping[str, Any]] = {}
            for row in started_channels:
                payload = row["payload"]
                channel = str(payload["channel"])
                if channel in started_by_channel:
                    raise RuntimeError(f"model {channel} channel has duplicate durable starts")
                started_by_channel[channel] = payload
            if internal_summary and started_by_channel:
                raise RuntimeError("context summary must not have public output channels")
            started_response = started_by_channel.get("agent_message")
            if started_response is not None:
                response_item_id = str(started_response["item_id"])
            else:
                response_item_id = f"item_{uuid4().hex}"
            public_item_id = derive_model_public_item_id(
                turn_id=str(operation["turn_id"]),
                model_attempt_id=attempt_id,
                channel="agent_message",
            )
            public_metadata = (
                {}
                if internal_summary
                else {
                    "channel": "agent_message",
                    "public_item_id": public_item_id,
                }
            )
            if outputs is not None:
                public_metadata["output_protocol"] = "model-outputs-v1"
            self._append_and_reduce(
                thread_id=operation["thread_id"],
                turn_id=operation["turn_id"],
                record_type="model_attempt_completed",
                producer="runtime",
                payload={
                    "operation_id": operation_id,
                    "attempt_id": attempt_id,
                    "generation": generation,
                    "observed_at": observed_at,
                    "provider_response_id": provider_response_id,
                    "usage": frozen_usage,
                    "response_item_id": response_item_id,
                    "response_status": response_status,
                    "incomplete_reason": incomplete_reason,
                    "stop_reason": stop_reason,
                    **({} if internal_summary else {"public_item_id": public_item_id}),
                },
            )
            self._append_budget_outcome(
                operation=operation,
                attempt_id=attempt_id,
                outcome="settled",
                usage=frozen_usage,
            )
            if started_response is None:
                self._append_and_reduce(
                    thread_id=operation["thread_id"],
                    turn_id=operation["turn_id"],
                    record_type="item_started",
                    producer="model",
                    payload={
                        "item_id": response_item_id,
                        "kind": "context_summary_response" if internal_summary else "model_response",
                        "attempt_id": attempt_id,
                        **public_metadata,
                    },
                )
            result_order = (
                self.read_turn(str(operation["turn_id"]))
                .binding_manifest.get(
                    "execution_context_policy",
                    {},
                )
                .get("tool_result_order")
            )
            self._append_and_reduce(
                thread_id=operation["thread_id"],
                turn_id=operation["turn_id"],
                record_type="item_completed",
                producer="model",
                payload={
                    "item_id": response_item_id,
                    "attempt_id": attempt_id,
                    **public_metadata,
                    "payload": {
                        "text": text,
                        "tool_calls": frozen_tool_calls,
                        **({"tool_result_order": "call-v1"} if result_order == "call-v1" else {}),
                        "provider_response_id": provider_response_id,
                        "usage": frozen_usage,
                        "response_status": response_status,
                        "incomplete_reason": incomplete_reason,
                        "stop_reason": stop_reason,
                        **({"outputs": [output_to_payload(output) for output in outputs],
                            "output_protocol": "model-outputs-v1"} if outputs is not None else {}),
                        **({"reasoning_content": reasoning_content} if reasoning_content else {}),
                        **({"plan_content": plan_content} if plan_content else {}),
                    },
                },
            )
            for channel, content in (
                ("reasoning", reasoning_content),
                ("plan", plan_content),
            ):
                started = started_by_channel.get(channel)
                if started is None:
                    continue
                public_channel_item_id = derive_model_public_item_id(
                    turn_id=str(operation["turn_id"]),
                    model_attempt_id=attempt_id,
                    channel=channel,
                )
                self._append_and_reduce(
                    thread_id=operation["thread_id"],
                    turn_id=operation["turn_id"],
                    record_type="item_completed",
                    producer="model",
                    payload={
                        "item_id": str(started["item_id"]),
                        "attempt_id": attempt_id,
                        "channel": channel,
                        "public_item_id": public_channel_item_id,
                        "payload": {"content": content or ""},
                    },
                )
        return True

    @_state_operation
    def list_model_operations(self, turn_id: str | None = None) -> tuple[ModelOperationSnapshot, ...]:
        if turn_id is None:
            rows = self.persistence.rows("model_operations", order_by="rowid")
        else:
            rows = self.persistence.rows("model_operations", where={"turn_id": turn_id}, order_by="rowid")
        return tuple(_model_operation_snapshot(row) for row in rows)

    @_state_operation
    def list_model_attempts(self, operation_id: str) -> tuple[ModelAttemptSnapshot, ...]:
        rows = self.persistence.rows("model_attempts", where={"operation_id": operation_id}, order_by="generation")
        return tuple(_model_attempt_snapshot(row) for row in rows)

    @_state_operation
    def start_budgeted_child_turn(
        self,
        *,
        parent_turn_id: str,
        child_thread_id: str,
        child_turn_id: str,
        user_message: str,
        binding_manifest: Mapping[str, Any],
        requested_tokens: int | None,
        requested_cost_micros: int | None,
    ) -> TurnSnapshot:
        """Atomically carve a child budget from the parent and start its Turn.

        The allocation record and child Turn live on the child Thread so this
        mutation remains one canonical Thread batch while SQLite's write lock
        serializes competing allocations from the same parent.
        """

        if not isinstance(parent_turn_id, str) or not parent_turn_id.strip():
            raise ValueError("parent_turn_id must be non-empty")
        if not isinstance(child_thread_id, str) or not child_thread_id.strip():
            raise ValueError("child_thread_id must be non-empty")
        if not isinstance(child_turn_id, str) or not child_turn_id.strip():
            raise ValueError("child_turn_id must be non-empty")
        if requested_tokens is not None and (
            isinstance(requested_tokens, bool) or not isinstance(requested_tokens, int) or requested_tokens < 1
        ):
            raise ValueError("requested child token budget must be positive or None")
        if requested_cost_micros is not None and (
            isinstance(requested_cost_micros, bool)
            or not isinstance(requested_cost_micros, int)
            or requested_cost_micros < 1
        ):
            raise ValueError("requested child cost budget must be positive or None")
        frozen_binding = _json_object(binding_manifest, field="binding_manifest")
        allocation_id = f"child_budget_{child_turn_id}"

        with self.persistence.transaction():
            parent = self.persistence.turn_with_workspace(parent_turn_id)
            if parent is None:
                raise KeyError(f"unknown parent Turn: {parent_turn_id}")
            if parent["status"] != "running":
                raise RuntimeError("child budget can only be allocated from a running parent Turn")

            parent_state = self.read_budget_state(parent_turn_id)
            remaining = normal_token_remaining(parent_state)
            if remaining is not None:
                if requested_tokens is None:
                    if remaining < 1:
                        limit = parent_state.limits.tokens
                        if limit is None:
                            raise RuntimeError("parent token limit disappeared")
                        raise BudgetLimitExceededError(
                            resource="tokens",
                            limit=limit,
                            current_exposure=limit,
                            requested=1,
                        )
                    granted_tokens: int | None = remaining
                else:
                    if requested_tokens > remaining:
                        limit = parent_state.limits.tokens
                        if limit is None:
                            raise RuntimeError("parent token limit disappeared")
                        raise BudgetLimitExceededError(
                            resource="tokens",
                            limit=limit,
                            current_exposure=limit - remaining,
                            requested=requested_tokens,
                        )
                    granted_tokens = requested_tokens
            else:
                granted_tokens = requested_tokens

            remaining_cost = parent_state.remaining("cost_micros")
            if remaining_cost is not None:
                if requested_cost_micros is None:
                    raise ValueError("child cost budget is required when the parent Turn has a cost cap")
                if requested_cost_micros > remaining_cost:
                    limit = parent_state.limits.cost_micros
                    if limit is None:
                        raise RuntimeError("parent cost limit disappeared")
                    raise BudgetLimitExceededError(
                        resource="cost_micros",
                        limit=limit,
                        current_exposure=limit - remaining_cost,
                        requested=requested_cost_micros,
                    )
                granted_cost_micros: int | None = requested_cost_micros
            else:
                granted_cost_micros = requested_cost_micros

            self.create_thread(
                workspace=Path(str(parent["workspace"])),
                thread_id=child_thread_id,
            )
            root_turn_id = parent_turn_id
            parent_binding = parent["binding_manifest"]
            inherited_root = parent_binding.get("budget_root_turn_id")
            if isinstance(inherited_root, str) and inherited_root:
                root_turn_id = inherited_root
            child_binding = dict(frozen_binding)
            child_binding["budget_root_turn_id"] = root_turn_id
            child_binding["budget_parent_turn_id"] = parent_turn_id
            child_binding["budget_allocation_id"] = allocation_id
            child_binding["model_token_budget_total"] = granted_tokens
            child_binding["model_cost_budget_total_micros"] = granted_cost_micros

            self._append_and_reduce(
                thread_id=child_thread_id,
                turn_id=None,
                record_type="budget_child_allocated",
                producer="runtime",
                payload={
                    "allocation_id": allocation_id,
                    "parent_turn_id": parent_turn_id,
                    "child_turn_id": child_turn_id,
                    "allocated_tokens": granted_tokens,
                    "allocated_cost_micros": granted_cost_micros,
                },
            )
            self.start_turn(
                thread_id=child_thread_id,
                turn_id=child_turn_id,
                user_message=user_message,
                binding_manifest=child_binding,
            )
        return self.read_turn(child_turn_id)

    @_state_operation
    def settle_child_budget(self, *, child_turn_id: str) -> Mapping[str, Any]:
        """Convert one terminal child's allocation into its actual subtree usage."""

        with self.persistence.transaction():
            row = self.persistence.first("child_budget_allocations", where={"child_turn_id": child_turn_id})
            if row is None:
                raise KeyError(f"child Turn has no parent budget allocation: {child_turn_id}")
            if row["status"] == "settled":
                return self.read_child_budget_allocation(child_turn_id)
            if row["status"] != "active":
                raise RuntimeError("child budget allocation is not active")
            turn = self.read_turn(child_turn_id)
            if turn.status not in {"completed", "failed", "cancelled"}:
                raise RuntimeError("child budget can only settle after a terminal Turn")
            state = self.read_budget_state(child_turn_id)
            zero = ResourceUsage()
            if state.reserved != zero or state.uncertain != zero or state.child_reserved != zero:
                raise RuntimeError("child budget still has unresolved exposure")
            actual = state.used + ResourceUsage(subagents=1)
            self._append_and_reduce(
                thread_id=str(row["thread_id"]),
                turn_id=child_turn_id,
                record_type="budget_child_settled",
                producer="runtime",
                payload={
                    "allocation_id": str(row["allocation_id"]),
                    "actual": actual.to_dict(),
                },
            )
        return self.read_child_budget_allocation(child_turn_id)

    @_state_operation
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

    @_state_operation
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

    @_state_operation
    def read_budget_reservation(self, attempt_id: str) -> BudgetReservation:
        row = self._budget_reservation_row_for_attempt(attempt_id)
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

    def _budget_reservation_row_for_attempt(
        self,
        attempt_id: str,
    ) -> Mapping[str, Any] | None:
        row = self.persistence.first("budget_reservations", where={"attempt_id": attempt_id})
        return row

    def _append_budget_outcome(
        self,
        *,
        operation: Mapping[str, Any],
        attempt_id: str,
        outcome: str,
        usage: Mapping[str, Any] | None = None,
        producer: str = "runtime",
    ) -> None:
        row = self._budget_reservation_row_for_attempt(attempt_id)
        if row is None:
            return
        reservation_id = str(row["reservation_id"])
        status = ReservationStatus(str(row["status"]))
        if outcome == "settled":
            if status not in {
                ReservationStatus.RESERVED,
                ReservationStatus.DISPATCHED,
                ReservationStatus.UNKNOWN,
            }:
                raise RuntimeError("budget reservation cannot be settled")
            if usage is None:
                raise RuntimeError("settled budget requires provider usage")
            self._append_and_reduce(
                thread_id=str(operation["thread_id"]),
                turn_id=str(operation["turn_id"]),
                record_type="budget_settled",
                producer=producer,
                payload={
                    "reservation_id": reservation_id,
                    "actual": resource_usage_from_model_usage(usage).to_dict(),
                },
            )
            return
        if outcome == "released":
            if status not in {
                ReservationStatus.RESERVED,
                ReservationStatus.DISPATCHED,
                ReservationStatus.UNKNOWN,
            }:
                raise RuntimeError("budget reservation cannot be released")
            self._append_and_reduce(
                thread_id=str(operation["thread_id"]),
                turn_id=str(operation["turn_id"]),
                record_type="budget_released",
                producer=producer,
                payload={"reservation_id": reservation_id},
            )
            return
        if outcome == "unknown":
            if status is ReservationStatus.UNKNOWN:
                return
            if status is not ReservationStatus.DISPATCHED:
                raise RuntimeError("only a dispatched budget can become unknown")
            self._append_and_reduce(
                thread_id=str(operation["thread_id"]),
                turn_id=str(operation["turn_id"]),
                record_type="budget_unknown",
                producer=producer,
                payload={"reservation_id": reservation_id},
            )
            return
        raise ValueError(f"unsupported budget outcome: {outcome}")

    @_state_operation
    def record_tool_call(
        self,
        *,
        turn_id: str,
        tool_call_id: str,
        tool_name: str,
        arguments: Mapping[str, Any],
        origin: Mapping[str, Any],
        argument_error: str | None = None,
    ) -> ItemSnapshot:
        if not tool_call_id or not tool_name:
            raise ValueError("tool call identity must be non-empty")
        frozen_arguments = _json_object(arguments, field="tool arguments")
        frozen_origin = _json_object(origin, field="tool call origin")
        expected_payload = {
            "tool_call_id": tool_call_id,
            "tool_name": tool_name,
            "arguments": frozen_arguments,
            "origin": frozen_origin,
        }
        if argument_error is not None:
            if not isinstance(argument_error, str):
                raise TypeError("argument_error must be a string or None")
            expected_payload["argument_error"] = argument_error
        item_id: str | None = None
        with self.persistence.transaction():
            thread_id = self._running_turn_thread_id(turn_id)
            matches = self.persistence.rows(
                "items", where={"turn_id": turn_id, "kind": "tool_call"}, order_by="sequence"
            )
            same_identity = [row for row in matches if row["payload"].get("tool_call_id") == tool_call_id]
            if len(same_identity) > 1:
                raise RuntimeError("tool call identity has multiple canonical Items")
            if same_identity:
                existing = same_identity[0]
                if existing["status"] != "completed" or _canonical_json(existing["payload"]) != _canonical_json(
                    expected_payload
                ):
                    raise RuntimeError("tool call identity conflicts with committed payload")
                item_id = str(existing["item_id"])
            else:
                item_id = f"item_{uuid4().hex}"
                self._append_and_reduce(
                    thread_id=thread_id,
                    turn_id=turn_id,
                    record_type="item_started",
                    producer="model",
                    payload={"item_id": item_id, "kind": "tool_call"},
                )
                self._append_and_reduce(
                    thread_id=thread_id,
                    turn_id=turn_id,
                    record_type="item_completed",
                    producer="model",
                    payload={"item_id": item_id, "payload": expected_payload},
                )
        if item_id is None:
            raise RuntimeError("tool call Item identity was not resolved")
        return self.read_item(item_id)

    @_state_operation
    def record_tool_execution_state(
        self,
        *,
        turn_id: str,
        operation_id: str,
        tool_call_id: str,
        tool_name: str,
        arguments_digest: str,
        execution_revision: str,
        idempotent: bool,
        status: str,
        attempt_count: int,
        error_code: str | None,
        requires_reconciliation: bool,
        effects: tuple[str, ...] = (),
        resources: tuple[Mapping[str, Any], ...] = (),
    ) -> ToolOperationSnapshot:
        allowed = {"prepared", "ready"}
        if status not in allowed:
            raise ValueError(f"unsupported tool operation status: {status}")
        frozen_effects = _string_tuple(effects, field="tool effects")
        frozen_resources = _resource_tuple(resources)
        with self.persistence.transaction():
            thread_id = self._running_turn_thread_id(turn_id)
            existing = self.persistence.first("tool_operations", where={"operation_id": operation_id})
            payload = {
                "operation_id": operation_id,
                "tool_call_id": tool_call_id,
                "tool_name": tool_name,
                "arguments_digest": arguments_digest,
                "execution_revision": execution_revision,
                "effects": frozen_effects,
                "resources": frozen_resources,
                "idempotent": idempotent,
                "status": status,
                "attempt_count": attempt_count,
                "error_code": error_code,
                "requires_reconciliation": requires_reconciliation,
            }
            if existing is None:
                if status != "prepared":
                    raise RuntimeError("first durable tool operation state must be prepared")
                prior_call = self.persistence.first(
                    "tool_operations", where={"turn_id": turn_id, "tool_call_id": tool_call_id}
                )
                if prior_call is not None:
                    raise RuntimeError(f"tool call already has an operation: {prior_call['operation_id']}")
                record_type = "tool_operation_prepared"
            else:
                immutable = {
                    "tool_call_id": tool_call_id,
                    "tool_name": tool_name,
                    "arguments_digest": arguments_digest,
                    "execution_revision": execution_revision,
                    "idempotent": int(idempotent),
                    "effects": list(frozen_effects),
                    "resources": [dict(resource) for resource in frozen_resources],
                }
                if any(_canonical_json(existing[name]) != _canonical_json(value) for name, value in immutable.items()):
                    raise RuntimeError("tool operation identity changed across states")
                current_status = str(existing["status"])
                allowed_next = {"prepared": {"ready"}}
                if current_status not in allowed_next:
                    raise RuntimeError("prepared-state API cannot mutate claimed or terminal operation")
                if status not in allowed_next[current_status]:
                    raise RuntimeError(f"invalid tool operation transition: {current_status} -> {status}")
                if attempt_count < int(existing["attempt_count"]):
                    raise RuntimeError("tool operation attempt count cannot decrease")
                record_type = "tool_operation_status_changed"
            self._append_and_reduce(
                thread_id=thread_id,
                turn_id=turn_id,
                record_type=record_type,
                producer="runtime",
                payload=payload,
            )
        return self.read_tool_operation(operation_id)

    @_state_operation
    def claim_tool_operation(
        self,
        *,
        operation_id: str,
        worker_id: str,
        lease_seconds: float,
        now: float | None = None,
    ) -> ToolOperationSnapshot:
        if not isinstance(worker_id, str) or not worker_id.strip():
            raise ValueError("worker_id must be non-empty")
        if lease_seconds <= 0:
            raise ValueError("lease_seconds must be positive")
        self._assert_artifact_integrity(self.read_tool_operation(operation_id).turn_id)
        claimed_at = time.time() if now is None else float(now)
        with self.persistence.transaction():
            operation = self.persistence.first("tool_operations", where={"operation_id": operation_id})
            if operation is None:
                raise KeyError(f"unknown tool operation: {operation_id}")
            if operation["status"] != "ready":
                raise RuntimeError("only a ready tool operation can be claimed")
            requested_resources = tuple(operation["resources"])
            active_operations = tuple(
                row
                for row in self.persistence.rows("tool_operations", where={"status": ("running", "unknown")})
                if row["operation_id"] != operation_id
            )
            for active in active_operations:
                active_resources = tuple(active["resources"])
                if _resource_sets_conflict(requested_resources, active_resources):
                    raise ResourceClaimConflictError(
                        f"resource claim conflict with operation: {active['operation_id']}"
                    )
            generation = int(operation["claim_generation"]) + 1
            fencing_token = f"fence_{uuid4().hex}"
            public_item_id = derive_operation_public_item_id(
                turn_id=str(operation["turn_id"]),
                operation_id=operation_id,
                attempt_generation=generation,
            )
            self._append_and_reduce(
                thread_id=operation["thread_id"],
                turn_id=operation["turn_id"],
                record_type="tool_operation_claimed",
                producer="runtime",
                payload={
                    "operation_id": operation_id,
                    "claim_generation": generation,
                    "fencing_token": fencing_token,
                    "claim_owner": worker_id,
                    "lease_expires_at": claimed_at + lease_seconds,
                    "attempt_count": int(operation["attempt_count"]) + 1,
                    "public_item_id": public_item_id,
                    "tool_name": str(operation["tool_name"]),
                    "tool_call_id": str(operation["tool_call_id"]),
                },
            )
        return self.read_tool_operation(operation_id)

    @_state_operation
    def reject_ready_tool_operation(
        self,
        *,
        operation_id: str,
        error_code: str,
    ) -> ToolOperationSnapshot:
        if not isinstance(error_code, str) or not error_code:
            raise ValueError("start rejection error_code must be non-empty")
        with self.persistence.transaction():
            operation = self.persistence.first("tool_operations", where={"operation_id": operation_id})
            if operation is None:
                raise KeyError(f"unknown tool operation: {operation_id}")
            if operation["status"] != "ready":
                raise RuntimeError("only a ready tool operation can be rejected")
            self._append_and_reduce(
                thread_id=operation["thread_id"],
                turn_id=operation["turn_id"],
                record_type="tool_operation_start_rejected",
                producer="runtime",
                payload={
                    "operation_id": operation_id,
                    "status": "failed",
                    "attempt_count": operation["attempt_count"],
                    "error_code": error_code,
                    "requires_reconciliation": False,
                },
            )
        return self.read_tool_operation(operation_id)

    @_state_operation
    def expire_tool_operation_claim(
        self,
        *,
        operation_id: str,
        now: float | None = None,
    ) -> ToolOperationSnapshot:
        observed_at = time.time() if now is None else float(now)
        with self.persistence.transaction():
            operation = self.persistence.first("tool_operations", where={"operation_id": operation_id})
            if operation is None:
                raise KeyError(f"unknown tool operation: {operation_id}")
            if operation["status"] != "running":
                raise RuntimeError("only a running tool operation claim can expire")
            expires_at = operation["lease_expires_at"]
            if expires_at is None or observed_at <= float(expires_at):
                raise RuntimeError("tool operation claim has not expired")
            self._append_and_reduce(
                thread_id=operation["thread_id"],
                turn_id=operation["turn_id"],
                record_type="tool_operation_claim_expired",
                producer="recovery",
                payload={
                    "operation_id": operation_id,
                    "claim_generation": operation["claim_generation"],
                    "fencing_token": operation["fencing_token"],
                    "observed_at": observed_at,
                },
            )
            request_id = f"interaction_{uuid4().hex}"
            self._append_and_reduce(
                thread_id=operation["thread_id"],
                turn_id=operation["turn_id"],
                record_type="interaction_requested",
                producer="recovery",
                payload={
                    "request_id": request_id,
                    "kind": "tool_reconciliation",
                    "operation_id": operation_id,
                    "request": {
                        "reason": "tool outcome unknown after lease expiry",
                        "claim_generation": operation["claim_generation"],
                        "claim_owner": operation["claim_owner"],
                        "idempotent": bool(operation["idempotent"]),
                    },
                },
            )
            turn = self.persistence.first("turns", where={"turn_id": operation["turn_id"]})
            if turn is not None and turn["status"] == "running":
                self._append_and_reduce(
                    thread_id=operation["thread_id"],
                    turn_id=operation["turn_id"],
                    record_type="turn_paused",
                    producer="recovery",
                    payload={
                        "turn_id": operation["turn_id"],
                        "operation_id": operation_id,
                        "request_id": request_id,
                        "reason": "tool outcome unknown after lease expiry",
                    },
                )
        return self.read_tool_operation(operation_id)

    @_state_operation
    def mark_tool_result_missing(
        self,
        *,
        operation_id: str,
    ) -> InteractionSnapshot:
        """Pause when execution outcome committed but its canonical result did not."""

        with self.persistence.transaction():
            operation = self.persistence.first("tool_operations", where={"operation_id": operation_id})
            if operation is None:
                raise KeyError(f"unknown tool operation: {operation_id}")
            if operation["status"] not in {"succeeded", "failed"}:
                raise RuntimeError("missing-result recovery requires a committed execution outcome")
            if operation["result_item_id"] is not None:
                raise RuntimeError("tool operation already has a canonical result")
            turn = self.persistence.first("turns", where={"turn_id": operation["turn_id"]})
            if turn is None or turn["status"] != "running":
                raise RuntimeError("missing-result recovery requires a running Turn")
            existing = self.persistence.rows(
                "interactions",
                where={"turn_id": operation["turn_id"], "kind": "tool_reconciliation", "status": "pending"},
            )
            if existing:
                raise RuntimeError("tool reconciliation is already pending")
            prior_status = str(operation["status"])
            self._append_and_reduce(
                thread_id=operation["thread_id"],
                turn_id=operation["turn_id"],
                record_type="tool_operation_result_missing",
                producer="recovery",
                payload={
                    "operation_id": operation_id,
                    "committed_execution_status": prior_status,
                },
            )
            request_id = f"interaction_{uuid4().hex}"
            self._append_and_reduce(
                thread_id=operation["thread_id"],
                turn_id=operation["turn_id"],
                record_type="interaction_requested",
                producer="recovery",
                payload={
                    "request_id": request_id,
                    "kind": "tool_reconciliation",
                    "operation_id": operation_id,
                    "request": {
                        "reason": "tool execution outcome committed but ToolResult is missing",
                        "committed_execution_status": prior_status,
                        "idempotent": bool(operation["idempotent"]),
                    },
                },
            )
            self._append_and_reduce(
                thread_id=operation["thread_id"],
                turn_id=operation["turn_id"],
                record_type="turn_paused",
                producer="recovery",
                payload={
                    "turn_id": operation["turn_id"],
                    "operation_id": operation_id,
                    "request_id": request_id,
                    "reason": "canonical ToolResult is missing after execution",
                },
            )
        return self.read_interaction(request_id)

    @_state_operation
    def commit_tool_operation_outcome(
        self,
        *,
        operation_id: str,
        claim_generation: int,
        fencing_token: str,
        status: str,
        attempt_count: int,
        error_code: str | None,
        requires_reconciliation: bool,
        duration_ms: float | None = None,
        now: float | None = None,
    ) -> bool:
        if status not in {"succeeded", "failed", "unknown"}:
            raise ValueError(f"unsupported claimed tool outcome: {status}")
        if duration_ms is not None and (
            isinstance(duration_ms, bool)
            or not isinstance(duration_ms, (int, float))
            or not math.isfinite(duration_ms)
            or duration_ms < 0
        ):
            raise ValueError("tool duration must be finite and non-negative")
        observed_at = time.time() if now is None else float(now)
        with self.persistence.transaction():
            operation = self.persistence.first("tool_operations", where={"operation_id": operation_id})
            if operation is None:
                raise KeyError(f"unknown tool operation: {operation_id}")
            # Same fencing rule as model attempts: expiry only authorizes the
            # recovery takeover, which rewrites the status to unknown under
            # the same serialized transaction. A matching claim generation and
            # fencing token on a still-running operation stay authoritative
            # past the lease clock.
            is_current = bool(
                operation["status"] == "running"
                and int(operation["claim_generation"]) == claim_generation
                and operation["fencing_token"] == fencing_token
            )
            if not is_current:
                return False
            public_item_id = derive_operation_public_item_id(
                turn_id=str(operation["turn_id"]),
                operation_id=operation_id,
                attempt_generation=claim_generation,
            )
            self._append_and_reduce(
                thread_id=operation["thread_id"],
                turn_id=operation["turn_id"],
                record_type="tool_operation_outcome_committed",
                producer="runtime",
                payload={
                    "operation_id": operation_id,
                    "claim_generation": claim_generation,
                    "fencing_token": fencing_token,
                    "status": status,
                    "attempt_count": attempt_count,
                    "error_code": error_code,
                    "requires_reconciliation": requires_reconciliation,
                    "observed_at": observed_at,
                    "public_item_id": public_item_id,
                    "duration_ms": duration_ms,
                },
            )
            if is_current and status == "unknown":
                request_id = f"interaction_{uuid4().hex}"
                self._append_and_reduce(
                    thread_id=operation["thread_id"],
                    turn_id=operation["turn_id"],
                    record_type="interaction_requested",
                    producer="runtime",
                    payload={
                        "request_id": request_id,
                        "kind": "tool_reconciliation",
                        "operation_id": operation_id,
                        "request": {
                            "reason": "tool execution outcome is unknown",
                            "error_code": error_code,
                            "claim_generation": claim_generation,
                            "claim_owner": operation["claim_owner"],
                            "idempotent": bool(operation["idempotent"]),
                        },
                    },
                )
                self._append_and_reduce(
                    thread_id=operation["thread_id"],
                    turn_id=operation["turn_id"],
                    record_type="turn_paused",
                    producer="runtime",
                    payload={
                        "turn_id": operation["turn_id"],
                        "operation_id": operation_id,
                        "request_id": request_id,
                        "reason": "tool execution outcome is unknown",
                    },
                )
        return is_current

    @_state_operation
    def commit_tool_operation_result(
        self,
        *,
        turn_id: str,
        operation_id: str,
        claim_generation: int,
        fencing_token: str,
        status: str,
        attempt_count: int,
        error_code: str | None,
        requires_reconciliation: bool,
        result: Mapping[str, Any],
        duration_ms: float | None = None,
        now: float | None = None,
    ) -> bool:
        with self.persistence.transaction():
            accepted = self.commit_tool_operation_outcome(
                operation_id=operation_id,
                claim_generation=claim_generation,
                fencing_token=fencing_token,
                status=status,
                attempt_count=attempt_count,
                error_code=error_code,
                requires_reconciliation=requires_reconciliation,
                duration_ms=duration_ms,
                now=now,
            )
            if not accepted:
                return False
            self.record_tool_result(
                turn_id=turn_id,
                operation_id=operation_id,
                result=result,
            )
        return True

    @_state_operation
    def resolve_tool_reconciliation(
        self,
        *,
        turn_id: str,
        status: str,
        result: Mapping[str, Any],
        reconciler_revision: str,
    ) -> ItemSnapshot:
        if status not in {"succeeded", "failed", "cancelled"}:
            raise ValueError("reconciled tool status is unsupported")
        self._assert_artifact_integrity(turn_id)
        if not isinstance(reconciler_revision, str) or not reconciler_revision.strip():
            raise ValueError("reconciler_revision must be non-empty")
        frozen_result = _json_object(result, field="reconciled tool result")
        item_id = f"item_{uuid4().hex}"
        with self.persistence.transaction():
            turn = self.persistence.first("turns", where={"turn_id": turn_id})
            if turn is None:
                raise KeyError(f"unknown turn: {turn_id}")
            if turn["status"] != "paused":
                raise RuntimeError("tool reconciliation requires a paused Turn")
            pending = self.persistence.rows(
                "interactions",
                where={"turn_id": turn_id, "kind": "tool_reconciliation", "status": "pending"},
                order_by="rowid",
            )
            if len(pending) != 1:
                raise RuntimeError("reconciliation requires exactly one pending interaction")
            interaction = pending[0]
            operation = self.persistence.first("tool_operations", where={"operation_id": interaction["operation_id"]})
            if operation is None or operation["status"] != "unknown":
                raise RuntimeError("reconciliation interaction is not bound to an unknown operation")
            thread_id = str(turn["thread_id"])
            parent_item_id = derive_operation_public_item_id(
                turn_id=turn_id,
                operation_id=str(operation["operation_id"]),
                attempt_generation=int(operation["claim_generation"]),
            )
            public_item_id = derive_reconciliation_public_item_id(
                turn_id=turn_id,
                operation_id=str(operation["operation_id"]),
                reconciler_revision=reconciler_revision,
            )
            self._append_and_reduce(
                thread_id=thread_id,
                turn_id=turn_id,
                record_type="interaction_resolved",
                producer="reconciler",
                payload={
                    "request_id": interaction["request_id"],
                    "version": int(interaction["version"]) + 1,
                    "response": {
                        "status": status,
                        "reconciler_revision": reconciler_revision,
                    },
                },
            )
            self._append_and_reduce(
                thread_id=thread_id,
                turn_id=turn_id,
                record_type="tool_operation_reconciled",
                producer="reconciler",
                payload={
                    "operation_id": operation["operation_id"],
                    "status": status,
                    "error_code": frozen_result.get("error_code"),
                    "reconciler_revision": reconciler_revision,
                },
            )
            self._append_and_reduce(
                thread_id=thread_id,
                turn_id=turn_id,
                record_type="item_started",
                producer="reconciler",
                payload={
                    "item_id": item_id,
                    "kind": "tool_reconciliation",
                    "operation_id": str(operation["operation_id"]),
                    "public_item_id": public_item_id,
                    "parent_item_id": parent_item_id,
                    "reconciler_revision": reconciler_revision,
                },
            )
            self._append_and_reduce(
                thread_id=thread_id,
                turn_id=turn_id,
                record_type="item_completed",
                producer="reconciler",
                payload={
                    "item_id": item_id,
                    "operation_id": str(operation["operation_id"]),
                    "public_item_id": public_item_id,
                    "parent_item_id": parent_item_id,
                    "reconciler_revision": reconciler_revision,
                    "payload": {
                        "status": status,
                        "result": frozen_result,
                        "reconciler_revision": reconciler_revision,
                    },
                },
            )
            self._append_and_reduce(
                thread_id=thread_id,
                turn_id=turn_id,
                record_type="tool_operation_result_linked",
                producer="reconciler",
                payload={
                    "operation_id": operation["operation_id"],
                    "result_item_id": item_id,
                },
            )
            self._append_and_reduce(
                thread_id=thread_id,
                turn_id=turn_id,
                record_type="turn_resumed",
                producer="reconciler",
                payload={
                    "turn_id": turn_id,
                    "request_id": interaction["request_id"],
                },
            )
        return self._latest_item(turn_id)

    @_state_operation
    def resolve_tool_approval(
        self,
        *,
        turn_id: str,
        decision: str,
        policy_recheck: bool = False,
    ) -> InteractionSnapshot:
        if decision not in {"approve", "deny"}:
            raise ValueError("approval decision must be approve or deny")
        self._assert_artifact_integrity(turn_id)
        with self.persistence.transaction():
            turn = self.persistence.first("turns", where={"turn_id": turn_id})
            if turn is None:
                raise KeyError(f"unknown turn: {turn_id}")
            if turn["status"] != "paused":
                raise RuntimeError(f"turn is not paused: {turn_id}")
            pending = self.persistence.rows(
                "interactions",
                where={"turn_id": turn_id, "kind": "tool_approval", "status": "pending"},
                order_by="rowid",
            )
            if len(pending) != 1:
                raise RuntimeError("resume requires exactly one pending tool approval")
            interaction = pending[0]
            operation = self.persistence.first("tool_operations", where={"operation_id": interaction["operation_id"]})
            if operation is None or operation["status"] != "awaiting_approval":
                raise RuntimeError("pending approval is not bound to an awaiting operation")
            thread_id = str(turn["thread_id"])
            approval_status = "approved" if decision == "approve" else "denied"
            operation_status = "ready" if decision == "approve" else "denied"
            self._append_and_reduce(
                thread_id=thread_id,
                turn_id=turn_id,
                record_type="interaction_resolved",
                producer="user",
                payload={
                    "request_id": interaction["request_id"],
                    "version": int(interaction["version"]) + 1,
                    "response": {"decision": decision, **({"policy_recheck": True} if policy_recheck else {})},
                    "approval_status": approval_status,
                },
            )
            self._append_and_reduce(
                thread_id=thread_id,
                turn_id=turn_id,
                record_type="tool_operation_approval_resolved",
                producer="runtime",
                payload={
                    "operation_id": interaction["operation_id"],
                    "request_id": interaction["request_id"],
                    "status": operation_status,
                },
            )
            self._append_and_reduce(
                thread_id=thread_id,
                turn_id=turn_id,
                record_type="turn_resumed",
                producer="runtime",
                payload={"turn_id": turn_id, "request_id": interaction["request_id"]},
            )
        return self.read_interaction(str(interaction["request_id"]))

    @_state_operation
    def refresh_tool_approval(self, *, turn_id: str, request: Mapping[str, Any]) -> InteractionSnapshot:
        """Replace an unexecuted approval after a policy recheck, atomically."""
        frozen_request = _json_object(request, field="approval request")
        request_id = f"interaction_{uuid4().hex}"
        with self.persistence.transaction():
            turn = self.read_turn(turn_id)
            candidates = [
                self.read_interaction(op.approval_request_id)
                for op in self.list_tool_operations(turn_id)
                if op.status in {"awaiting_approval", "ready"} and op.approval_request_id is not None
            ]
            if turn.status not in {"paused", "running"} or len(candidates) != 1:
                raise RuntimeError("approval refresh requires one unexecuted operation")
            previous = candidates[0]
            if previous.status == "resolved" and previous.response.get("policy_recheck") is not True:
                raise RuntimeError("only a policy recheck can reopen a resolved approval")
            if previous.status == "pending":
                self._append_and_reduce(
                    thread_id=turn.thread_id,
                    turn_id=turn_id,
                    record_type="interaction_resolved",
                    producer="runtime",
                    payload={
                        "request_id": previous.request_id,
                        "version": previous.version + 1,
                        "response": {"action": "recheck"},
                        "approval_status": "invalidated",
                    },
                )
            else:
                self._append_and_reduce(
                    thread_id=turn.thread_id,
                    turn_id=turn_id,
                    record_type="approval_invalidated",
                    producer="runtime",
                    payload={"request_id": previous.request_id, "reason": "policy recheck requires confirmation"},
                )
            self._append_and_reduce(
                thread_id=turn.thread_id,
                turn_id=turn_id,
                record_type="interaction_requested",
                producer="runtime",
                payload={
                    "request_id": request_id,
                    "kind": "tool_approval",
                    "operation_id": previous.operation_id,
                    "request": frozen_request,
                },
            )
            self._append_and_reduce(
                thread_id=turn.thread_id,
                turn_id=turn_id,
                record_type="tool_operation_approval_requested",
                producer="runtime",
                payload={"operation_id": previous.operation_id, "request_id": request_id},
            )
            if turn.status == "running":
                self._append_and_reduce(
                    thread_id=turn.thread_id,
                    turn_id=turn_id,
                    record_type="turn_paused",
                    producer="runtime",
                    payload={"turn_id": turn_id, "request_id": request_id},
                )
        return self.read_interaction(request_id)

    @_state_operation
    def invalidate_tool_approval(
        self,
        *,
        turn_id: str,
        reason: str,
    ) -> InteractionSnapshot:
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError("approval invalidation reason must be non-empty")
        with self.persistence.transaction():
            turn = self.persistence.first("turns", where={"turn_id": turn_id})
            if turn is None:
                raise KeyError(f"unknown turn: {turn_id}")
            if turn["status"] != "paused":
                raise RuntimeError(f"turn is not paused: {turn_id}")
            pending = self.persistence.rows(
                "interactions",
                where={"turn_id": turn_id, "kind": "tool_approval", "status": "pending"},
                order_by="rowid",
            )
            if len(pending) != 1:
                raise RuntimeError("invalidation requires one pending tool approval")
            interaction = pending[0]
            operation = self.persistence.first("tool_operations", where={"operation_id": interaction["operation_id"]})
            if operation is None or operation["status"] != "awaiting_approval":
                raise RuntimeError("approval is not bound to an awaiting operation")
            thread_id = str(turn["thread_id"])
            self._append_and_reduce(
                thread_id=thread_id,
                turn_id=turn_id,
                record_type="interaction_resolved",
                producer="user",
                payload={
                    "request_id": interaction["request_id"],
                    "version": int(interaction["version"]) + 1,
                    "response": {
                        "decision": "approve",
                        "invalidation_reason": reason,
                    },
                    "approval_status": "approved",
                },
            )
            self._append_and_reduce(
                thread_id=thread_id,
                turn_id=turn_id,
                record_type="approval_invalidated",
                producer="runtime",
                payload={
                    "request_id": interaction["request_id"],
                    "reason": reason,
                },
            )
            self._append_and_reduce(
                thread_id=thread_id,
                turn_id=turn_id,
                record_type="tool_operation_superseded",
                producer="runtime",
                payload={
                    "operation_id": interaction["operation_id"],
                    "request_id": interaction["request_id"],
                    "reason": reason,
                },
            )
        return self.read_interaction(str(interaction["request_id"]))

    @_state_operation
    def invalidate_resolved_tool_approval(
        self,
        *,
        turn_id: str,
        reason: str,
    ) -> InteractionSnapshot:
        """Invalidate an approved ready operation after crash-time revalidation."""

        if not isinstance(reason, str) or not reason.strip():
            raise ValueError("approval invalidation reason must be non-empty")
        with self.persistence.transaction():
            turn = self.persistence.first("turns", where={"turn_id": turn_id})
            if turn is None:
                raise KeyError(f"unknown turn: {turn_id}")
            if turn["status"] != "running":
                raise RuntimeError("resolved approval recovery requires a running Turn")
            interaction = self.persistence.first(
                "interactions",
                where={"turn_id": turn_id, "kind": "tool_approval", "status": "resolved"},
                order_by="rowid",
                newest=True,
                limit=1,
            )
            if (
                interaction is None
                or interaction["response"].get("decision") != "approve"
                or interaction["operation_id"] is None
            ):
                raise RuntimeError("no resolved approved operation can be invalidated")
            operation = self.persistence.first("tool_operations", where={"operation_id": interaction["operation_id"]})
            approval = self.persistence.first("approvals", where={"request_id": interaction["request_id"]})
            if (
                operation is None
                or operation["status"] != "ready"
                or approval is None
                or approval["status"] != "approved"
            ):
                raise RuntimeError("resolved approval is not bound to a ready operation")
            thread_id = str(turn["thread_id"])
            self._append_and_reduce(
                thread_id=thread_id,
                turn_id=turn_id,
                record_type="approval_invalidated",
                producer="runtime",
                payload={
                    "request_id": interaction["request_id"],
                    "reason": reason,
                },
            )
            self._append_and_reduce(
                thread_id=thread_id,
                turn_id=turn_id,
                record_type="tool_operation_superseded",
                producer="runtime",
                payload={"operation_id": interaction["operation_id"]},
            )
            self._append_and_reduce(
                thread_id=thread_id,
                turn_id=turn_id,
                record_type="turn_paused",
                producer="recovery",
                payload={
                    "turn_id": turn_id,
                    "request_id": interaction["request_id"],
                    "reason": reason,
                },
            )
        return self.read_interaction(str(interaction["request_id"]))

    @_state_operation
    def request_tool_approval(
        self,
        *,
        turn_id: str,
        operation_id: str,
        tool_call_id: str,
        tool_name: str,
        arguments_digest: str,
        execution_revision: str,
        idempotent: bool,
        request: Mapping[str, Any],
        effects: tuple[str, ...] = (),
        resources: tuple[Mapping[str, Any], ...] = (),
    ) -> InteractionSnapshot:
        frozen_request = _json_object(request, field="approval request")
        frozen_effects = _string_tuple(effects, field="tool effects")
        frozen_resources = _resource_tuple(resources)
        request_id = f"interaction_{uuid4().hex}"
        with self.persistence.transaction():
            thread_id = self._running_turn_thread_id(turn_id)
            prior_call = self.persistence.first(
                "tool_operations", where={"turn_id": turn_id, "tool_call_id": tool_call_id}
            )
            if prior_call is not None:
                if (
                    prior_call["status"] != "ready"
                    or prior_call["approval_request_id"] is None
                    or prior_call["arguments_digest"] != arguments_digest
                    or prior_call["execution_revision"] != execution_revision
                    or prior_call["tool_name"] != tool_name
                    or tuple(prior_call["effects"]) != frozen_effects
                    or tuple(prior_call["resources"]) != frozen_resources
                ):
                    raise RuntimeError(f"tool call already has an operation: {prior_call['operation_id']}")
                # A second scope (e.g. network) must reuse the unstarted operation.
                operation_id = str(prior_call["operation_id"])
            else:
                self._append_and_reduce(
                    thread_id=thread_id,
                    turn_id=turn_id,
                    record_type="tool_operation_prepared",
                    producer="runtime",
                    payload={
                        "operation_id": operation_id,
                        "tool_call_id": tool_call_id,
                        "tool_name": tool_name,
                        "arguments_digest": arguments_digest,
                        "execution_revision": execution_revision,
                        "effects": frozen_effects,
                        "resources": frozen_resources,
                        "idempotent": idempotent,
                        "status": "prepared",
                        "attempt_count": 0,
                        "error_code": None,
                        "requires_reconciliation": False,
                    },
                )
            self._append_and_reduce(
                thread_id=thread_id,
                turn_id=turn_id,
                record_type="interaction_requested",
                producer="runtime",
                payload={
                    "request_id": request_id,
                    "kind": "tool_approval",
                    "operation_id": operation_id,
                    "request": frozen_request,
                },
            )
            self._append_and_reduce(
                thread_id=thread_id,
                turn_id=turn_id,
                record_type="tool_operation_approval_requested",
                producer="runtime",
                payload={
                    "operation_id": operation_id,
                    "request_id": request_id,
                },
            )
            self._append_and_reduce(
                thread_id=thread_id,
                turn_id=turn_id,
                record_type="turn_paused",
                producer="runtime",
                payload={"turn_id": turn_id, "request_id": request_id},
            )
        return self.read_interaction(request_id)

    @_state_operation
    def record_tool_result(
        self,
        *,
        turn_id: str,
        operation_id: str | None,
        result: Mapping[str, Any],
    ) -> ItemSnapshot:
        frozen_result = _json_object(result, field="tool result")
        self._validate_artifact_references(frozen_result)
        operation: Mapping[str, Any] | None = None
        item_id = f"item_{uuid4().hex}"
        with self.persistence.transaction():
            turn = self.persistence.first("turns", where={"turn_id": turn_id})
            if turn is None:
                raise KeyError(f"unknown turn: {turn_id}")
            thread_id = str(turn["thread_id"])
            if operation_id is not None:
                operation = self.persistence.first(
                    "tool_operations", where={"operation_id": operation_id, "turn_id": turn_id}
                )
                if operation is None:
                    raise KeyError(f"unknown tool operation: {operation_id}")
                if operation["status"] not in {
                    "succeeded",
                    "failed",
                    "denied",
                    "cancelled",
                    "unknown",
                }:
                    raise RuntimeError("tool result requires a terminal operation state")
                paused_unknown = turn["status"] == "paused" and operation["status"] == "unknown"
                if paused_unknown:
                    reconciliation = self.persistence.rows(
                        "interactions",
                        where={
                            "turn_id": turn_id,
                            "operation_id": operation_id,
                            "kind": "tool_reconciliation",
                            "status": "pending",
                        },
                    )
                    if len(reconciliation) != 1:
                        raise RuntimeError("paused unknown tool result requires one reconciliation interaction")
                elif turn["status"] != "running":
                    raise RuntimeError(f"turn is not running: {turn_id}")
            elif turn["status"] != "running":
                raise RuntimeError(f"turn is not running: {turn_id}")
            public_item_id: str | None = None
            plan_public_item_id: str | None = None
            plan_snapshot: Mapping[str, Any] | None = None
            plan_event: Mapping[str, Any] | None = None
            plan_item_id: str | None = None
            if operation is not None:
                assert operation_id is not None
                attempt_generation = int(operation["claim_generation"])
                if attempt_generation > 0:
                    public_item_id = derive_operation_public_item_id(
                        turn_id=turn_id,
                        operation_id=operation_id,
                        attempt_generation=attempt_generation,
                    )
                if operation["tool_name"] == "update_plan" and frozen_result.get("is_error") is not True:
                    structured = frozen_result.get("structured_content")
                    provider_revision = structured.get("revision") if isinstance(structured, Mapping) else None
                    if (
                        isinstance(provider_revision, bool)
                        or not isinstance(provider_revision, int)
                        or provider_revision < 0
                    ):
                        raise RuntimeError("successful update_plan result requires a canonical revision")
                    call = self.persistence.rows(
                        "items",
                        where={"turn_id": turn_id, "kind": "tool_call", "status": "completed"},
                        order_by="sequence",
                    )
                    matching_calls = [
                        row["payload"]
                        for row in call
                        if row["payload"].get("tool_call_id") == operation["tool_call_id"]
                    ]
                    if len(matching_calls) != 1:
                        raise RuntimeError("successful update_plan result requires one canonical ToolCall")
                    arguments = matching_calls[0].get("arguments")
                    if not isinstance(arguments, Mapping):
                        raise RuntimeError("successful update_plan ToolCall arguments are malformed")
                    prior_revisions = (
                        self.persistence.count("items", where={"turn_id": turn_id, "kind": "plan_state"}),
                    )[0]
                    revision = int(prior_revisions) + 1
                    plan_public_item_id = derive_plan_public_item_id(
                        turn_id=turn_id,
                        revision=revision,
                    )
                    plan_item_id = f"item_{uuid4().hex}"
                    raw_steps = arguments.get("plan", [])
                    steps: list[dict[str, Any]] = []
                    if isinstance(raw_steps, (list, tuple)):
                        for index, raw_step in enumerate(raw_steps, start=1):
                            if not isinstance(raw_step, Mapping):
                                continue
                            status = raw_step.get("status")
                            normalized_status: Literal["pending", "in_progress", "completed"]
                            if status in {"pending", "completed"}:
                                normalized_status = status
                            elif status == "in_progress":
                                normalized_status = "in_progress"
                            else:
                                continue
                            step = PlanStep(
                                step_id=str(raw_step.get("step_id") or f"step_{index}"),
                                title=str(raw_step.get("step", "")),
                                status=normalized_status,
                            )
                            steps.append(step.model_dump())
                    objective_row = self.persistence.first(
                        "items", where={"turn_id": turn_id, "kind": "user_message"}, order_by="sequence", limit=1
                    )
                    objective_payload = objective_row["payload"] if objective_row is not None else {}
                    plan_snapshot = {
                        "revision": revision,
                        "objective": objective_payload.get("text", "Current task"),
                        "status": "active",
                        "active_step_id": next(
                            (step["step_id"] for step in steps if step["status"] == "in_progress"),
                            next(
                                (step["step_id"] for step in steps if step["status"] == "pending"),
                                None,
                            ),
                        ),
                        "steps": steps,
                        "summary": arguments.get("explanation"),
                    }
                    plan_event = {
                        "event_type": "llm_update",
                        "message": (
                            arguments.get("explanation")
                            if isinstance(arguments.get("explanation"), str)
                            else "Applied update_plan tool update."
                        ),
                        "related_step_id": plan_snapshot["active_step_id"],
                        "tool_call_ids": [operation["tool_call_id"]],
                    }
            started_payload: dict[str, Any] = {
                "item_id": item_id,
                "kind": "tool_result",
                "tool_name": (
                    str(operation["tool_name"]) if operation is not None else frozen_result.get("tool_name")
                ),
                "tool_call_id": (
                    str(operation["tool_call_id"]) if operation is not None else frozen_result.get("tool_call_id")
                ),
            }
            if public_item_id is not None:
                started_payload.update(
                    {
                        "operation_id": operation_id,
                        "attempt_generation": attempt_generation,
                        "public_item_id": public_item_id,
                    }
                )
            self._append_and_reduce(
                thread_id=thread_id,
                turn_id=turn_id,
                record_type="item_started",
                producer="tool",
                payload=started_payload,
            )
            completed_payload: dict[str, Any] = {
                "item_id": item_id,
                "payload": frozen_result,
            }
            if public_item_id is not None:
                assert operation is not None
                completed_payload.update(
                    {
                        "operation_id": operation_id,
                        "attempt_generation": attempt_generation,
                        "public_item_id": public_item_id,
                    }
                )
                public_status = {
                    "succeeded": "success",
                    "cancelled": "cancelled",
                    "unknown": "outcome_unknown",
                }.get(str(operation["status"]), "failed")
                completed_payload["status"] = public_status
                if public_status in {"failed", "outcome_unknown"}:
                    completed_payload["error"] = str(operation["error_code"] or "tool execution failed")
            self._append_and_reduce(
                thread_id=thread_id,
                turn_id=turn_id,
                record_type="item_completed",
                producer="tool",
                payload=completed_payload,
            )
            if (
                plan_item_id is not None
                and plan_public_item_id is not None
                and plan_snapshot is not None
                and plan_event is not None
            ):
                self._append_and_reduce(
                    thread_id=thread_id,
                    turn_id=turn_id,
                    record_type="item_started",
                    producer="runtime",
                    payload={
                        "item_id": plan_item_id,
                        "kind": "plan_state",
                        "public_item_id": plan_public_item_id,
                        "revision": plan_snapshot["revision"],
                    },
                )
                self._append_and_reduce(
                    thread_id=thread_id,
                    turn_id=turn_id,
                    record_type="item_completed",
                    producer="runtime",
                    payload={
                        "item_id": plan_item_id,
                        "public_item_id": plan_public_item_id,
                        "revision": plan_snapshot["revision"],
                        "payload": {
                            "plan": plan_snapshot,
                            "event": plan_event,
                        },
                    },
                )
            if operation is not None:
                self._append_and_reduce(
                    thread_id=thread_id,
                    turn_id=turn_id,
                    record_type="tool_operation_result_linked",
                    producer="runtime",
                    payload={
                        "operation_id": operation_id,
                        "result_item_id": item_id,
                    },
                )
        return self._latest_item(turn_id)


    @_state_operation
    def read_tool_operation(self, operation_id: str) -> ToolOperationSnapshot:
        row = self.persistence.first("tool_operations", where={"operation_id": operation_id})
        if row is None:
            raise KeyError(f"unknown tool operation: {operation_id}")
        return _tool_operation_snapshot(row)

    @_state_operation
    def record_tool_continuation(
        self,
        *,
        operation_id: str,
        key: str,
        result: Mapping[str, Any],
    ) -> ItemSnapshot:
        """Persist the owner's confirmed terminal result when its live scope closes."""
        operation = self.read_tool_operation(operation_id)
        if operation.result_item_id is None:
            raise RuntimeError("continuation requires a committed launch result")
        for item in self.list_items(operation.turn_id):
            if item.kind == "tool_continuation" and item.payload.get("operation_id") == operation_id:
                return item
        item_id = f"item_{uuid4().hex}"
        with self.persistence.transaction():
            self._append_and_reduce(
                thread_id=operation.thread_id,
                turn_id=operation.turn_id,
                record_type="item_started",
                producer="runtime",
                payload={"item_id": item_id, "kind": "tool_continuation"},
            )
            self._append_and_reduce(
                thread_id=operation.thread_id,
                turn_id=operation.turn_id,
                record_type="item_completed",
                producer="runtime",
                payload={
                    "item_id": item_id,
                    "payload": {
                        **dict(result),
                        "operation_id": operation_id,
                        "key": key,
                        "source_result_item_id": operation.result_item_id,
                    },
                },
            )
        return self.read_item(item_id)

    @_state_operation
    def record_verification(
        self,
        *,
        turn_id: str,
        operation_id: str,
        kind: str,
        verifier: str,
        verified_resources: tuple[str, ...] = (),
    ) -> ItemSnapshot:
        if kind not in {"test", "static_analysis", "assertion", "inspection"}:
            raise ValueError(f"unsupported verification kind: {kind}")
        if not isinstance(verifier, str) or not verifier.strip():
            raise ValueError("verification verifier must be non-empty")
        with self.persistence.transaction():
            operation = self.persistence.first(
                "tool_operations", where={"operation_id": operation_id, "turn_id": turn_id}
            )
            if operation is None:
                raise KeyError(f"unknown tool operation: {operation_id}")
            if operation["status"] != "succeeded" or operation["result_item_id"] is None:
                raise RuntimeError("verification requires a committed successful ToolResult")
            existing_rows = self.persistence.rows(
                "items", where={"turn_id": turn_id, "kind": "verification", "status": "completed"}, order_by="sequence"
            )
            for row in existing_rows:
                payload = row["payload"]
                if payload.get("operation_id") == operation_id:
                    return _item_snapshot(row)

            item_id = f"item_{uuid4().hex}"
            payload = {
                "operation_id": operation_id,
                "tool_call_id": operation["tool_call_id"],
                "arguments_digest": operation["arguments_digest"],
                "source_result_item_id": operation["result_item_id"],
                "verification_kind": kind,
                "verifier": verifier.strip(),
                "verified_resources": list(verified_resources),
            }
            self._append_and_reduce(
                thread_id=operation["thread_id"],
                turn_id=turn_id,
                record_type="item_started",
                producer="runtime",
                payload={"item_id": item_id, "kind": "verification"},
            )
            self._append_and_reduce(
                thread_id=operation["thread_id"],
                turn_id=turn_id,
                record_type="item_completed",
                producer="runtime",
                payload={"item_id": item_id, "payload": payload},
            )
        return self.read_item(item_id)

    @_state_operation
    def record_workspace_checkpoint(
        self,
        *,
        turn_id: str,
        checkpoint_id: str,
        trigger: str,
        tree_sha256: str | None = None,
        changed_file_count: int = 0,
        wall_time: float = 0.0,
    ) -> ItemSnapshot:
        """Persist the Turn's workspace checkpoint; one pre_write item per Turn."""
        if trigger not in {"pre_write", "restore"}:
            raise ValueError(f"unsupported checkpoint trigger: {trigger}")
        if not isinstance(checkpoint_id, str) or not checkpoint_id.strip():
            raise ValueError("checkpoint_id must be a non-empty string")
        with self.persistence.transaction():
            turn = self.persistence.first("turns", where={"turn_id": turn_id})
            if turn is None:
                raise KeyError(f"unknown turn: {turn_id}")
            existing_rows = self.persistence.rows(
                "items",
                where={"turn_id": turn_id, "kind": "workspace_checkpoint", "status": "completed"},
                order_by="sequence",
            )
            for row in existing_rows:
                payload = row["payload"]
                if payload.get("checkpoint_id") == checkpoint_id or (
                    trigger == "pre_write" and payload.get("trigger") == "pre_write"
                ):
                    return _item_snapshot(row)
            item_id = f"item_{uuid4().hex}"
            payload = {
                "checkpoint_id": checkpoint_id,
                "trigger": trigger,
                "tree_sha256": tree_sha256,
                "changed_file_count": changed_file_count,
                "wall_time": wall_time,
            }
            self._append_and_reduce(
                thread_id=turn["thread_id"],
                turn_id=turn_id,
                record_type="item_started",
                producer="runtime",
                payload={"item_id": item_id, "kind": "workspace_checkpoint"},
            )
            self._append_and_reduce(
                thread_id=turn["thread_id"],
                turn_id=turn_id,
                record_type="item_completed",
                producer="runtime",
                payload={"item_id": item_id, "payload": payload},
            )
        return self.read_item(item_id)

    @_state_operation
    def list_tool_operations(self, turn_id: str | None = None) -> tuple[ToolOperationSnapshot, ...]:
        if turn_id is None:
            rows = self.persistence.rows("tool_operations", order_by="rowid")
        else:
            rows = self.persistence.rows("tool_operations", where={"turn_id": turn_id}, order_by="rowid")
        return tuple(_tool_operation_snapshot(row) for row in rows)

    @_state_operation
    def read_interaction(self, request_id: str) -> InteractionSnapshot:
        row = self.persistence.first("interactions", where={"request_id": request_id})
        if row is None:
            raise KeyError(f"unknown interaction: {request_id}")
        return _interaction_snapshot(row)

    @_state_operation
    def list_interactions(self, turn_id: str | None = None) -> tuple[InteractionSnapshot, ...]:
        if turn_id is None:
            rows = self.persistence.rows("interactions", order_by="rowid")
        else:
            rows = self.persistence.rows("interactions", where={"turn_id": turn_id}, order_by="rowid")
        return tuple(_interaction_snapshot(row) for row in rows)

    @_state_operation
    def list_approvals(self, turn_id: str | None = None) -> tuple[ApprovalSnapshot, ...]:
        if turn_id is None:
            rows = self.persistence.rows("approvals", order_by="rowid")
        else:
            rows = self.persistence.rows("approvals", where={"turn_id": turn_id}, order_by="rowid")
        return tuple(_approval_snapshot(row) for row in rows)

    @_state_operation
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

    @_state_operation
    def reserve_resource_unit(
        self,
        turn_id: str,
        resource: str,
        limit: int,
        initial_used: int = 0,
    ) -> dict[str, Any]:
        """Commit one conservative unit before I/O, serialized across Store connections."""

        _validate_resource_usage_input(resource, initial_used)
        if type(limit) is not int or not 1 <= limit <= 256:
            raise ValueError("resource limit must be an integer between 1 and 256")
        item_id = f"item_{uuid4().hex}"
        with self.persistence.transaction():
            thread_id = self._running_turn_thread_id(turn_id)
            usage = self.read_resource_usage(turn_id, resource, initial_used)
            if usage["limit"] is not None and usage["limit"] != limit:
                raise ValueError("resource_limit_mismatch")
            if usage["used"] >= limit:
                raise ValueError("resource_limit_exceeded")
            used = usage["used"] + 1
            self._append_and_reduce(
                thread_id=thread_id,
                turn_id=turn_id,
                record_type="item_started",
                producer="runtime",
                payload={"item_id": item_id, "kind": "resource_usage"},
            )
            self._append_and_reduce(
                thread_id=thread_id,
                turn_id=turn_id,
                record_type="item_completed",
                producer="runtime",
                payload={"item_id": item_id, "payload": {"resource": resource, "limit": limit, "used": used}},
            )
        return {"resource": resource, "limit": limit, "used": used, "remaining": limit - used, "item_id": item_id}

    @_state_operation
    def record_final_proposal(
        self,
        *,
        turn_id: str,
        answer: str,
        source_item_id: str | None = None,
    ) -> ItemSnapshot:
        if not isinstance(answer, str) or not answer.strip():
            raise ValueError("answer must be non-empty")
        item_id = f"item_{uuid4().hex}"
        with self.persistence.transaction():
            thread_id = self._running_turn_thread_id(turn_id)
            payload = {"text": answer}
            if source_item_id is not None:
                source = self.read_item(source_item_id)
                if source.turn_id != turn_id or source.kind != "model_response" or source.status != "completed":
                    raise RuntimeError("final proposal requires a completed model response from this Turn")
                payload["source_item_id"] = source_item_id
            self._append_and_reduce(
                thread_id=thread_id,
                turn_id=turn_id,
                record_type="item_started",
                producer="model",
                payload={"item_id": item_id, "kind": "final_proposal"},
            )
            self._append_and_reduce(
                thread_id=thread_id,
                turn_id=turn_id,
                record_type="item_completed",
                producer="model",
                payload={"item_id": item_id, "payload": payload},
            )
        return self._latest_item(turn_id)

    @_state_operation
    def record_completion_decision(
        self,
        *,
        turn_id: str,
        proposal_item_id: str,
        action: str,
        reason: str,
        validation_status: str = "not_evaluated",
        validation_scope: str | None = None,
        reason_code: str | None = None,
    ) -> ItemSnapshot:
        if action not in {"accept", "continue", "pause", "fail"}:
            raise ValueError(f"unsupported completion action: {action}")
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError("completion reason must be non-empty")
        if validation_status not in {"not_evaluated", "passed", "failed"}:
            raise ValueError(f"unsupported completion validation status: {validation_status}")
        if validation_scope is not None and (not isinstance(validation_scope, str) or not validation_scope.strip()):
            raise ValueError("completion validation scope must be non-empty or None")
        if reason_code is not None and (not isinstance(reason_code, str) or not reason_code.strip()):
            raise ValueError("completion reason code must be non-empty or None")
        item_id = f"item_{uuid4().hex}"
        with self.persistence.transaction():
            thread_id = self._running_turn_thread_id(turn_id)
            proposal = self.persistence.first("items", where={"item_id": proposal_item_id, "turn_id": turn_id})
            if proposal is None or proposal["kind"] != "final_proposal" or proposal["status"] != "completed":
                raise RuntimeError("completion decision requires a completed final proposal")
            self._append_and_reduce(
                thread_id=thread_id,
                turn_id=turn_id,
                record_type="item_started",
                producer="verifier",
                payload={"item_id": item_id, "kind": "completion_decision"},
            )
            self._append_and_reduce(
                thread_id=thread_id,
                turn_id=turn_id,
                record_type="item_completed",
                producer="verifier",
                payload={
                    "item_id": item_id,
                    "payload": {
                        "proposal_item_id": proposal_item_id,
                        "action": action,
                        "reason": reason,
                        "validation_status": validation_status,
                        "validation_scope": validation_scope,
                        **({"reason_code": reason_code} if reason_code is not None else {}),
                    },
                },
            )
        return self._latest_item(turn_id)

    @_state_operation
    def record_completion_feedback(
        self,
        *,
        turn_id: str,
        reason: str,
        source_item_id: str | None = None,
        reason_code: str | None = None,
    ) -> ItemSnapshot:
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError("completion feedback reason must be non-empty")
        item_id = f"item_{uuid4().hex}"
        with self.persistence.transaction():
            thread_id = self._running_turn_thread_id(turn_id)
            payload = {"text": reason}
            if reason_code is not None:
                payload["reason_code"] = reason_code
            if source_item_id is not None:
                source = self.read_item(source_item_id)
                if (
                    source.turn_id != turn_id
                    or source.status != "completed"
                    or source.kind not in {"model_response", "completion_decision"}
                ):
                    raise RuntimeError("feedback requires a completed source from this Turn")
                payload["source_item_id"] = source_item_id
            self._append_and_reduce(
                thread_id=thread_id,
                turn_id=turn_id,
                record_type="item_started",
                producer="verifier",
                payload={"item_id": item_id, "kind": "completion_feedback"},
            )
            self._append_and_reduce(
                thread_id=thread_id,
                turn_id=turn_id,
                record_type="item_completed",
                producer="verifier",
                payload={"item_id": item_id, "payload": payload},
            )
        return self._latest_item(turn_id)

    def _running_turn_thread_id(self, turn_id: str) -> str:
        turn = self.persistence.first("turns", where={"turn_id": turn_id})
        if turn is None:
            raise KeyError(f"unknown turn: {turn_id}")
        if turn["status"] != "running":
            raise RuntimeError(f"turn is not running: {turn_id}")
        return str(turn["thread_id"])

    @_state_operation
    def read_thread(self, thread_id: str) -> ThreadSnapshot:
        row = self.persistence.first("threads", where={"thread_id": thread_id})
        if row is None:
            raise KeyError(f"unknown thread: {thread_id}")
        return _thread_snapshot(row)

    @_state_operation
    def list_threads(self) -> tuple[ThreadSnapshot, ...]:
        rows = self.persistence.rows("threads", order_by="rowid")
        return tuple(_thread_snapshot(row) for row in rows)

    @_state_operation
    def read_turn(self, turn_id: str) -> TurnSnapshot:
        row = self.persistence.first("turns", where={"turn_id": turn_id})
        if row is None:
            raise KeyError(f"unknown turn: {turn_id}")
        return _turn_snapshot(row)

    @_state_operation
    def list_turns(self, thread_id: str | None = None) -> tuple[TurnSnapshot, ...]:
        rows = self.persistence.rows(
            "turns", where=({} if thread_id is None else {"thread_id": thread_id}), order_by="rowid"
        )
        return tuple(_turn_snapshot(row) for row in rows)

    @_state_operation
    def list_items(self, turn_id: str) -> tuple[ItemSnapshot, ...]:
        rows = self.persistence.rows("items", where={"turn_id": turn_id}, order_by="sequence")
        return tuple(_item_snapshot(row) for row in rows)

    def _latest_item(self, turn_id: str) -> ItemSnapshot:
        row = self.persistence.first("items", where={"turn_id": turn_id}, order_by="sequence", newest=True, limit=1)
        if row is None:
            raise KeyError(f"Turn has no Items: {turn_id}")
        return _item_snapshot(row)

    @_state_operation
    def read_item(self, item_id: str) -> ItemSnapshot:
        row = self.persistence.first("items", where={"item_id": item_id})
        if row is None:
            raise KeyError(f"unknown Item: {item_id}")
        return _item_snapshot(row)

    @_state_operation
    def list_context_items(self, turn_id: str) -> tuple[ItemSnapshot, ...]:
        """Visible model/recall window; the complete ledger remains separately readable."""
        self._assert_artifact_integrity(turn_id)
        items = self.list_conversation_items(turn_id)
        start = 0
        seen = set()
        for index, item in enumerate(items):
            if item.turn_id not in seen:
                seen.add(item.turn_id)
                reset = self.read_turn(item.turn_id).binding_manifest.get("context_reset")
                if (
                    isinstance(reset, Mapping)
                    and reset.get("revision") == "v1"
                    and reset.get("reason") == "memory_revoked"
                ):
                    start = index
        return items[start:]

    @_state_operation
    def list_conversation_items(self, turn_id: str) -> tuple[ItemSnapshot, ...]:
        """Read original Items through the fork point; do not load or execute artifacts."""
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
        rows: list[Mapping[str, Any]] = []
        for ancestor_thread_id, cutoff in reversed(segments):
            rows.extend(self.persistence.conversation_rows(ancestor_thread_id, cutoff))
        return tuple(_item_snapshot(row) for row in rows)

    @_state_operation
    def read_turn_answer(self, turn_id: str) -> ItemSnapshot:
        """Resolve the accepted output identity; old Turns retain their original answer Item."""
        completion = self.read_turn_completion(turn_id)
        if completion is None:
            raise RuntimeError("Turn has no committed answer")
        item_id = completion.payload.get("answer_item_id")
        if item_id is None:
            if completion.payload_schema_version >= 3:
                raise RuntimeError("completed Turn is missing its canonical answer reference")
            answers = [item for item in self.list_items(turn_id) if item.kind == "agent_message"]
            if len(answers) != 1:
                raise RuntimeError("completed Turn has no unique canonical answer")
            return answers[0]
        answer = self.read_item(item_id)
        if (
            answer.turn_id != turn_id
            or answer.status != "completed"
            or answer.kind not in {"agent_message", "model_response"}
            or answer.payload.get("response_status", "completed") != "completed"
        ):
            raise RuntimeError("completed Turn has an invalid answer reference")
        return answer



    def _assert_artifact_integrity(self, turn_id: str) -> None:
        # Dispatch validates its actual conversation references. A damaged blob
        # in another Thread is still diagnosed by verify(), without disabling
        # unrelated work throughout the whole store.
        artifact_ids = {
            str(attachment["artifact_id"])
            for item in self.list_conversation_items(turn_id)
            for attachment in item.payload.get("attachments", ())
            if isinstance(attachment, Mapping) and isinstance(attachment.get("artifact_id"), str)
        }
        errors = self.persistence.artifact_integrity_errors(artifact_ids=artifact_ids)
        if errors:
            raise RuntimeError(f"artifact integrity check failed: {errors[0]}")

    def _validate_artifact_references(self, payload: Mapping[str, Any]) -> None:
        attachments = payload.get("attachments", ())
        if not isinstance(attachments, (list, tuple)):
            raise RuntimeError("tool result artifact attachments must be a list")
        for attachment in attachments:
            if not isinstance(attachment, Mapping):
                raise RuntimeError("tool result artifact attachment is malformed")
            artifact_id = attachment.get("artifact_id")
            if not isinstance(artifact_id, str):
                raise RuntimeError("tool result references an unknown artifact")
            try:
                artifact = self.read_artifact_metadata(artifact_id)
            except KeyError:
                raise RuntimeError(f"tool result references unknown artifact: {artifact_id}") from None
            if attachment.get("media_type") != artifact.media_type or attachment.get("name") != artifact.name:
                raise RuntimeError("tool result artifact metadata conflicts with durable reference")
            errors = self.persistence.artifact_integrity_errors(artifact_ids={artifact_id})
            error = errors[0] if errors else None
            if error is not None:
                raise RuntimeError(f"artifact integrity check failed: {error}")


class TurnInputChangedError(RuntimeError):
    """A response no longer covers the active Turn's received user input."""


def _validate_resource_usage_input(resource: str, initial_used: int) -> None:
    if not isinstance(resource, str) or not resource.strip() or len(resource) > 120:
        raise ValueError("resource must be a non-empty name of at most 120 characters")
    if type(initial_used) is not int or initial_used < 0:
        raise ValueError("initial resource usage must be a non-negative integer")


def _model_public_item_ids(*, turn_id: str, attempt_id: str) -> dict[str, str]:
    return {
        channel: derive_model_public_item_id(
            turn_id=turn_id,
            model_attempt_id=attempt_id,
            channel=channel,
        )
        for channel in ("agent_message", "reasoning", "plan")
    }


def _model_channel_content(
    content: Mapping[str, str] | None,
) -> dict[str, str]:
    if content is None:
        return {}
    allowed = {"agent_message", "reasoning", "plan"}
    frozen: dict[str, str] = {}
    for channel, value in content.items():
        if channel not in allowed:
            raise ValueError(f"unsupported model output channel: {channel}")
        if not isinstance(value, str):
            raise TypeError(f"model {channel} content must be a string")
        frozen[channel] = value
    return frozen


class ResourceClaimConflictError(RuntimeError):
    """Another running or unknown operation owns an incompatible resource."""


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _context_durable_state(store: RuntimeState, turn_id: str) -> dict[str, Any]:
    """Project non-text state that compaction must never silently erase."""

    return {
        "interactions": [
            {
                "request_id": interaction.request_id,
                "kind": interaction.kind,
                "status": interaction.status,
                "version": interaction.version,
                "operation_id": interaction.operation_id,
                "request": dict(interaction.request),
                "response": dict(interaction.response),
            }
            for interaction in store.list_interactions(turn_id)
            if interaction.status != "resolved"
        ],
        "approvals": [
            {
                "request_id": approval.request_id,
                "operation_id": approval.operation_id,
                "status": approval.status,
                "scope": dict(approval.scope),
            }
            for approval in store.list_approvals(turn_id)
            if approval.status in {"pending", "approved"}
        ],
        "tool_operations": [
            {
                "operation_id": operation.operation_id,
                "tool_call_id": operation.tool_call_id,
                "tool_name": operation.tool_name,
                "status": operation.status,
                "effects": list(operation.effects),
                "resources": [dict(resource) for resource in operation.resources],
                "requires_reconciliation": operation.requires_reconciliation,
                "error_code": operation.error_code,
            }
            for operation in store.list_tool_operations(turn_id)
            if operation.status not in {"succeeded", "failed", "denied", "cancelled", "superseded"}
            or operation.requires_reconciliation
        ],
    }


def _json_object(value: Mapping[str, Any], *, field: str) -> dict[str, Any]:
    try:
        normalized = json.loads(_canonical_json(dict(value)))
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{field} must be JSON-serializable") from exc
    if not isinstance(normalized, dict):
        raise TypeError(f"{field} must be an object")
    return normalized


def _string_tuple(values: tuple[str, ...], *, field: str) -> tuple[str, ...]:
    if any(not isinstance(value, str) or not value for value in values):
        raise ValueError(f"{field} must contain non-empty strings")
    return tuple(sorted(set(values)))


def _resource_tuple(
    values: tuple[Mapping[str, Any], ...],
) -> tuple[dict[str, str], ...]:
    resources: list[dict[str, str]] = []
    for value in values:
        kind = value.get("kind")
        identity = value.get("identity")
        access = value.get("access")
        if (
            not isinstance(kind, str)
            or not kind
            or not isinstance(identity, str)
            or not identity
            or access not in {"read", "write"}
        ):
            raise ValueError("tool resources require non-empty kind/identity and read/write access")
        if kind == "filesystem" and not Path(identity).is_absolute():
            raise ValueError("filesystem resource identity must be absolute")
        resources.append({"kind": kind, "identity": identity, "access": str(access)})
    unique = {(item["kind"], item["identity"], item["access"]): item for item in resources}
    return tuple(unique[key] for key in sorted(unique))


def _resource_sets_conflict(
    left: tuple[Mapping[str, Any], ...],
    right: tuple[Mapping[str, Any], ...],
) -> bool:
    for first in left:
        for second in right:
            if first.get("access") == second.get("access") == "read":
                continue
            if first.get("kind") != second.get("kind"):
                continue
            first_identity = first.get("identity")
            second_identity = second.get("identity")
            if not isinstance(first_identity, str) or not isinstance(second_identity, str):
                raise RuntimeError("durable resource claim is malformed")
            if first.get("kind") != "filesystem":
                if first_identity == second_identity:
                    return True
                continue
            first_path = Path(first_identity)
            second_path = Path(second_identity)
            if first_path.is_relative_to(second_path) or second_path.is_relative_to(first_path):
                return True
    return False
