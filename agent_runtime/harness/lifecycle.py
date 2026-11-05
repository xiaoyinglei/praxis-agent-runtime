"""Run trusted hooks outside transactions and settle decisions in the Rollout."""

from __future__ import annotations

import asyncio
import hashlib
from collections.abc import Awaitable, Callable, Mapping
from functools import partial
from typing import Any

from agent_runtime.harness import facts, turn_ops
from agent_runtime.harness.runtime_store import RuntimeStore
from agent_runtime.hooks import HookDecision, HookEvent, HookRequest, LifecycleHook, validate_hooks


class LifecycleHooks:
    def __init__(
        self, store: RuntimeStore, hooks: tuple[LifecycleHook, ...],
        *, dispatch_guard: Callable[[RuntimeStore], None] | None = None,
    ) -> None:
        self._store = store
        self._dispatch_guard = dispatch_guard
        self._hooks = {(hook.event, hook.name): hook for hook in validate_hooks(hooks)}

    def configured(
        self,
        turn_id: str,
        event: HookEvent,
        *,
        tool_name: str | None = None,
        effects: tuple[str, ...] = (),
    ) -> tuple[Mapping[str, Any], ...]:
        binding = self._store.read_turn(turn_id).binding_manifest
        effect_names = frozenset(effects)
        return tuple(
            hook for hook in binding.get("lifecycle_hooks", ())
            if hook["event"] == event
            and (
                tool_name is None
                or (
                    (not hook.get("tool_names") or tool_name in hook["tool_names"])
                    and (not hook.get("effects") or not effect_names.isdisjoint(hook["effects"]))
                )
            )
        )

    async def run(
        self,
        commit: Callable[[Callable[[], Any]], Awaitable[Any]],
        event: HookEvent,
        *,
        turn_id: str,
        target_id: str,
        **details: Any,
    ) -> str | None:
        store = self._store
        tool_name = details.get("tool_name")
        if event != "Stop" and not tool_name:
            raise ValueError("tool hooks require a canonical tool name")
        configured = await store.run_async(
            self.configured, turn_id, event,
            tool_name=tool_name, effects=details.get("effects", ()),
        )
        if not configured:
            return None
        turn = await store.run_async(store.read_turn, turn_id)
        thread = await store.run_async(store.read_thread, turn.thread_id)
        evidence_revision = details.get("completion_revision")
        for spec in configured:
            name = spec["name"]
            key_parts = (turn_id, event, target_id, name)
            if evidence_revision is not None:
                key_parts += (evidence_revision,)
            key = "\0".join(key_parts)
            invocation_id = "hook_" + hashlib.sha256(key.encode()).hexdigest()
            request = HookRequest(event, invocation_id, thread.thread_id, turn_id, thread.workspace, **details)
            outcome = await self._run_registered(commit, request, spec, target_id)
            if event != "PostToolUse" and outcome["status"] != "allowed":
                return f"{event} hook {name!r} ({outcome['status']}): {outcome['feedback']}"
        return None

    async def _run_registered(
        self,
        commit: Callable[[Callable[[], Any]], Awaitable[Any]],
        request: HookRequest,
        spec: Mapping[str, Any],
        target_id: str,
    ) -> Mapping[str, Any]:
        store = self._store
        invocation_id = request.invocation_id
        turn_id = request.turn_id
        thread_id = request.thread_id
        identity = {"hook_event": request.event, "hook_name": spec["name"], "target_id": target_id}
        if request.completion_revision is not None:
            identity["completion_revision"] = request.completion_revision

        def start() -> str:
            if self._dispatch_guard is not None:
                self._dispatch_guard(store)
            with store.tx.transaction():
                turn_ops._assert_no_pending_cancellation(store, turn_id)
                prior = store.read_event_reference("item_started", invocation_id)
                if prior is not None:
                    return store.read_item(invocation_id).status
                facts.append_fact(
                    store, thread_id=thread_id, turn_id=turn_id,
                    producer="runtime", record_type="item_started",
                    payload={"item_id": invocation_id, "kind": "lifecycle_hook", **identity},
                )
                return "new"

        def settle(status: str, feedback: str) -> Mapping[str, Any]:
            with store.tx.transaction():
                item = store.read_item(invocation_id)
                if item.status == "completed":
                    return item.payload
                body = {
                    **identity, "status": status, "feedback": feedback,
                    "source_item_id": request.source_item_id, "operation_id": request.operation_id,
                    "tool_call_id": request.tool_call_id,
                }
                facts.append_fact(
                    store, thread_id=thread_id, turn_id=turn_id, producer="runtime",
                    record_type="item_completed", payload={"item_id": invocation_id, "payload": body},
                )
                return body

        state = await commit(start)
        if state == "completed":
            outcome = (await store.run_async(store.read_item, invocation_id)).payload
        else:
            if state == "started":
                status, feedback = "unknown", "The previous hook invocation has no confirmed outcome."
            else:
                try:
                    status, feedback = await self._invoke(request, spec)
                except asyncio.CancelledError:
                    await asyncio.shield(commit(partial(
                        settle, "unknown", "The hook was interrupted before its outcome was confirmed.",
                    )))
                    raise
            outcome = await commit(partial(settle, status, feedback))
        return outcome

    async def _invoke(self, request: HookRequest, spec: Mapping[str, Any]) -> tuple[str, str]:
        hook = self._hooks.get((request.event, spec["name"]))
        if hook is None or hook.binding() != spec:
            return "unavailable", "The frozen hook registration is unavailable in this host."
        task = asyncio.current_task()
        assert task is not None
        cancellations = task.cancelling()
        try:
            async with asyncio.timeout(hook.timeout_seconds) as deadline:
                decision = await hook.handler(request)
            if deadline.expired():
                raise TimeoutError
            if decision is None:
                decision = HookDecision()
            if not isinstance(decision, HookDecision) or (request.event == "PostToolUse" and decision.block):
                raise ValueError("invalid decision for this hook event")
        except TimeoutError:
            return "timeout", "The hook exceeded its time limit."
        except asyncio.CancelledError:
            if task.cancelling() > cancellations:
                raise
            return "error", "The hook cancelled its own work."
        except Exception:
            return "error", "The hook failed or returned an invalid decision."
        if request.event == "PostToolUse":
            return "observed", decision.feedback
        if decision.block:
            return "blocked", decision.feedback.strip() or "The host hook blocked this action."
        return "allowed", decision.feedback
