"""Attached child Turns: durable ownership, bounded live drivers, kernel input/cancel.

The allocation ledger is authoritative. Tasks only provide wakeups and live I/O
handles; losing this manager never erases ownership or refunds unknown usage.
"""

from __future__ import annotations

import asyncio
import hashlib
import math
from collections.abc import Mapping
from functools import partial
from typing import TYPE_CHECKING, Any

from agent_runtime.budget import ResourceUsage
from agent_runtime.harness import budget_ops, turn_ops
from agent_runtime.harness.operation_values import TurnChildrenPendingError
from agent_runtime.harness.protocol import item_text
from agent_runtime.harness.records import TurnSnapshot

if TYPE_CHECKING:
    from agent_runtime.harness.session import Session

DEFAULT_LIMITS = {"max_concurrent": 4, "max_depth": 2, "max_total": 16}
TERMINAL = {"completed", "failed", "cancelled"}


def child_limits(value: Mapping[str, object] | None = None) -> dict[str, int]:
    limits = dict(DEFAULT_LIMITS)
    if value is not None:
        if set(value) - set(limits):
            raise ValueError("unknown subagent limit")
        for name, number in value.items():
            if isinstance(number, bool) or not isinstance(number, int) or number < 0:
                raise ValueError("subagent limits must be nonnegative integers")
            limits[name] = number
    return limits


def reserve_child(session: Session, **arguments: Any) -> TurnSnapshot:
    """Check lineage/count ceilings in the same write transaction as allocation."""
    store = session.store
    with store.tx.transaction():
        session._ensure_open()
        parent = store.read_turn(arguments["parent_turn_id"])
        limits = child_limits(parent.binding_manifest.get("subagent_limits"))
        limits = {k: min(v, session._subagent_limits[k]) for k, v in limits.items()}
        edges: dict[str, str] = {}
        for turn in store.list_turns():
            try:
                allocation = store.read_child_budget_allocation(turn.turn_id)
            except KeyError:
                continue
            edges[turn.turn_id] = str(allocation["parent_turn_id"])
        root, depth = parent.turn_id, 1
        while root in edges:
            root = edges[root]
            depth += 1
        if depth > limits["max_depth"]:
            raise RuntimeError("subagent depth limit exceeded")
        family = {root}
        while True:
            expanded = family | {c for c, p in edges.items() if p in family}
            if expanded == family:
                break
            family = expanded
        if len(family) - 1 >= limits["max_total"]:
            raise RuntimeError("subagent total count limit exceeded")
        active = sum(store.read_turn(c).status not in TERMINAL for c in family - {root})
        if active >= limits["max_concurrent"]:
            raise RuntimeError("subagent concurrent limit exceeded")
        binding = dict(arguments["binding_manifest"])
        binding["subagent_limits"] = limits
        return budget_ops.start_budgeted_child_turn(store, **{**arguments, "binding_manifest": binding})


class SubagentManager:
    def __init__(self, session: Session) -> None:
        self.session = session

    def _owned(self, parent_turn_id: str, child_turn_id: str) -> TurnSnapshot:
        store = self.session.store
        try:
            parent = store.read_turn(parent_turn_id) if parent_turn_id else None
        except KeyError as exc:
            raise PermissionError("parent Turn is not owned by this Session") from exc
        if parent is None or parent.thread_id != self.session.thread_id:
            raise PermissionError("parent Turn is not owned by this Session")
        try:
            allocation = store.read_child_budget_allocation(child_turn_id)
        except KeyError as exc:
            raise PermissionError("child has no owned allocation") from exc
        if allocation["parent_turn_id"] != parent_turn_id:
            raise PermissionError("child is not attached to this parent Turn")
        return store.read_turn(child_turn_id)

    async def spawn(
        self,
        *,
        parent_turn_id: str,
        user_message: str,
        max_steps: int | None = None,
        max_tokens_total: int | None = None,
        max_cost_micros: int | None = None,
    ) -> dict[str, object]:
        self.session._ensure_open()
        # Reservation commits before scheduling. The returned identity is already
        # recoverable even if the host disappears before the first child Step.
        prepared = await self.session._prepare_child(
            parent_turn_id=parent_turn_id,
            user_message=user_message,
            max_steps=max_steps,
            max_tokens_total=max_tokens_total,
            max_cost_micros=max_cost_micros,
        )
        return self._schedule(prepared)

    def _schedule(self, prepared: Mapping[str, Any]) -> dict[str, object]:
        turn_id = str(prepared["turn_id"])
        task = asyncio.create_task(self.session._drive_child(prepared), name=f"child:{turn_id}")
        self.session._subagent_tasks[turn_id] = task
        # Retrieve exceptions even when the parent deliberately doesn't wait.
        task.add_done_callback(lambda t: None if t.cancelled() else t.exception())
        return {"child_turn_id": turn_id, "thread_id": prepared["thread_id"], "status": "running"}

    async def send(
        self, *, parent_turn_id: str, child_turn_id: str, message: str, message_id: str
    ) -> dict[str, object]:
        if not isinstance(message_id, str) or not message_id.strip():
            raise ValueError("message_id must be non-empty")
        child = await self.session.store.run_async(self._owned, parent_turn_id, child_turn_id)
        receipt = (
            "child_input_" + hashlib.sha256(f"{parent_turn_id}\0{child_turn_id}\0{message_id}".encode()).hexdigest()
        )
        item = await self.session.store.run_async(
            partial(turn_ops.receive_turn_input, self.session.store),
            thread_id=child.thread_id,
            turn_id=child_turn_id,
            text=message,
            input_id=receipt,
        )
        return {
            "child_turn_id": child_turn_id,
            "message_id": message_id,
            "input_id": item.item_id,
            "status": "accepted",
        }

    async def wait(
        self, *, parent_turn_id: str, child_turn_id: str, timeout_seconds: float = 10.0
    ) -> dict[str, object]:
        if (
            isinstance(timeout_seconds, bool)
            or not isinstance(timeout_seconds, (int, float))
            or not math.isfinite(timeout_seconds)
            or not 0 <= timeout_seconds <= 30
        ):
            raise ValueError("wait timeout must be between zero and 30 seconds")
        await self.session.store.run_async(self._owned, parent_turn_id, child_turn_id)
        task = self.session._subagent_tasks.get(child_turn_id)
        if task is not None and not task.done() and timeout_seconds:
            # asyncio.wait never cancels the driver when a wait expires.
            await asyncio.wait({task}, timeout=timeout_seconds)
        return await self.session.store.run_async(self._output, parent_turn_id, child_turn_id)

    def _output(self, parent_turn_id: str, child_turn_id: str) -> dict[str, object]:
        child = self._owned(parent_turn_id, child_turn_id)
        store = self.session.store
        pending = [i for i in store.list_interactions(child_turn_id) if i.status == "pending"]
        pause = None
        if child.status == "paused" and len(pending) == 1:
            request = pending[0].request
            pause = {
                "request_id": pending[0].request_id,
                "kind": pending[0].kind,
                "question": request.get("reason")
                or request.get("question")
                or f"Input required for child {pending[0].kind}",
            }
        answer = None
        for record in reversed(store.list_turn_records(child_turn_id)):
            if record.record_type == "turn_completed":
                answer = item_text(store.read_item(record.payload["answer_item_id"]))
                break
        evidence = [
            {
                "kind": i.payload["verification_kind"],
                "verifier": i.payload["verifier"],
                "operation_id": i.payload["operation_id"],
                "verified_resources": i.payload.get("verified_resources", []),
            }
            for i in store.list_items(child_turn_id)
            if i.kind == "verification" and i.status == "completed" and i.producer == "runtime"
        ]
        return {
            "child_turn_id": child_turn_id,
            "status": "done"
            if child.status == "completed"
            else "failed"
            if child.status in {"failed", "cancelled"}
            else "paused"
            if child.status in {"paused", "interrupted"}
            else "running",
            "conclusion": answer or (str(pause["question"]) if pause else child.terminal_message or ""),
            "stop_reason": child.terminal_reason_code or child.status,
            "pause": pause,
            "execution_evidence": evidence,
        }

    async def cancel(self, *, parent_turn_id: str, child_turn_id: str) -> dict[str, object]:
        child = await self.session.store.run_async(self._owned, parent_turn_id, child_turn_id)
        if child.status not in TERMINAL:
            await self.session.store.run_async(
                partial(turn_ops.request_turn_cancellation, self.session.store),
                turn_id=child_turn_id,
                reason="Parent requested attached child cancellation",
            )
            task = self.session._subagent_tasks.get(child_turn_id)
            if task is not None and not task.done():
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)
            # A driver may be lost across restart. Do not declare its operations
            # cancelled or refund allocations without canonical reconciliation.
            live = self.session._subagent_sessions.get(child_turn_id)
            if live is not None and live._cancel_live_commands is not None:
                await live._cancel_live_commands()
            child = await self.session.store.run_async(self.session.store.read_turn, child_turn_id)
            if task is not None and task.done() and child.status == "running":
                child = await self.session.store.run_async(
                    partial(turn_ops.pause_turn, self.session.store),
                    turn_id=child_turn_id,
                    reason="Child driver stopped after cancellation request",
                )
            if child.status in {"paused", "interrupted"}:
                try:
                    await self.session.store.run_async(
                        partial(turn_ops.cancel_turn, self.session.store), turn_id=child_turn_id
                    )
                except RuntimeError:
                    pass  # unresolved model/tool exposure remains reserved
            await self._settle(child_turn_id)
        return await self.session.store.run_async(self._output, parent_turn_id, child_turn_id)

    async def _settle(self, child_turn_id: str) -> None:
        store = self.session.store

        def settle() -> None:
            child = store.read_turn(child_turn_id)
            state = store.read_budget_state(child_turn_id)
            if child.status in TERMINAL and state.reserved + state.uncertain + state.child_reserved == ResourceUsage():
                budget_ops.settle_child_budget(store, child_turn_id=child_turn_id)

        await store.run_async(settle)

    async def drain(self, parent_turn_id: str) -> None:
        children = await self.session.store.run_async(self._children, parent_turn_id)
        tasks = [self.session._subagent_tasks[c] for c in children if c in self.session._subagent_tasks]
        if tasks:
            await asyncio.gather(*(asyncio.shield(t) for t in tasks), return_exceptions=True)
        for child in children:
            await self._settle(child)
        unresolved = await self.session.store.run_async(
            lambda: [c for c in children if self.session.store.read_child_budget_allocation(c)["status"] != "settled"]
        )
        if unresolved:
            # Returning a parent answer would hide a paused/lost/unknown child.
            # This is surfaced as a durable parent pause by Session's boundary.
            raise TurnChildrenPendingError(tuple(unresolved))

    def _children(self, parent_turn_id: str) -> list[str]:
        result = []
        for turn in self.session.store.list_turns():
            try:
                edge = self.session.store.read_child_budget_allocation(turn.turn_id)
            except KeyError:
                continue
            if edge["parent_turn_id"] == parent_turn_id:
                result.append(turn.turn_id)
        return result

    async def cancel_all(self, parent_turn_id: str) -> None:
        for child in await self.session.store.run_async(self._children, parent_turn_id):
            await self.cancel(parent_turn_id=parent_turn_id, child_turn_id=child)

    async def join(self) -> None:
        """Keep borrowed resources alive until this process's drivers stop."""
        tasks = tuple(self.session._subagent_tasks.values())
        if tasks:
            await asyncio.gather(*(asyncio.shield(t) for t in tasks), return_exceptions=True)

    async def close(self, *, exclude: frozenset[str] = frozenset(), preserve: bool = False) -> None:
        # Revoke ownership before waiting for drivers. Otherwise a child blocked
        # in provider I/O can keep Session.close waiting indefinitely.
        # Canonical attachments can outlive this process's task table (including
        # a crash between allocation and scheduling). Close must explicitly
        # cancel those remaining running children, without refunding exposure or
        # pretending a lost process/remote call has been reaped.
        def remaining() -> list[tuple[str, str]]:
            return [
                (parent.turn_id, child)
                for parent in self.session.store.list_turns(self.session.thread_id)
                for child in self._children(parent.turn_id)
                if child not in exclude and self.session.store.read_turn(child).status not in TERMINAL
            ]

        for parent, child in () if preserve else await self.session.store.run_async(remaining):
            await self.cancel(parent_turn_id=parent, child_turn_id=child)
        tasks = [task for child, task in self.session._subagent_tasks.items() if child not in exclude]
        if tasks:
            await asyncio.gather(*(asyncio.shield(t) for t in tasks), return_exceptions=True)
