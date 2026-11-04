"""Local execution host. All coding, permissions and recovery remain in Harness."""
from __future__ import annotations

import asyncio
import fcntl
import hashlib
import math
import os
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, Self
from uuid import uuid4

from agent_runtime.background import current_configuration_permissions, permission_ceiling, restore_configuration
from agent_runtime.core.runtime_diagnostics import exception_diagnostic
from agent_runtime.harness import RuntimeStore, Session, host_ops, turn_ops
from agent_runtime.harness.execution_facts import adopt_execution_protocol
from agent_runtime.harness.operation_values import TurnCancellationPendingError
from agent_runtime.harness.ownership import ConversationBusyError
from agent_runtime.harness.protocol import BoundHarnessModel
from agent_runtime.harness.recovery import ContinuationKind, recovery_plan
from agent_runtime.permission_policy import ResolvedPermissions, relocate_workspace_permissions
from agent_runtime.tools.permissions import ToolExecutionContext, restore_tool_policy

_TERMINAL = {"completed", "failed", "cancelled"}


class BackgroundHost:
    def __init__(
        self, database: Path, *, concurrency: int = 4, model: BoundHarnessModel | None = None,
        lease_seconds: float = 30,
    ) -> None:
        if type(concurrency) is not int or not 1 <= concurrency <= 32:
            raise ValueError("host concurrency must be between 1 and 32")
        if isinstance(lease_seconds, bool) or not math.isfinite(lease_seconds) or lease_seconds < 1:
            raise ValueError("host lease must be at least one second")
        self.database = database.expanduser().resolve()
        self.concurrency, self.model, self.lease_seconds = concurrency, model, lease_seconds
        self.owner = "host_" + uuid4().hex
        self._drivers: dict[str, asyncio.Task[None]] = {}
        self._sessions: dict[str, Session] = {}
        self._workspaces: dict[str, str] = {}
        self._closing = False

    async def __aenter__(self) -> Self:
        self.database.parent.mkdir(parents=True, exist_ok=True)
        self._lock = os.open(str(self.database) + ".host.lock", os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
        try:
            fcntl.flock(self._lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            os.close(self._lock)
            raise RuntimeError("a background host already owns this database") from None
        try:
            self.store = await RuntimeStore.open(self.database)
            report = await self.store.run_async(self.store.verify)
            if not report.valid:
                raise RuntimeError("host cannot execute a corrupt rollout: " + "; ".join(report.errors))
            await self.store.run_async(host_ops.release_orphan_claims, self.store, owner=self.owner)
        except BaseException:
            if hasattr(self, "store"):
                await self.store.aclose()
            os.close(self._lock)
            raise
        return self

    async def __aexit__(self, *_args: object) -> None:
        self._closing = True
        try:
            # Shutdown cancels in-process drivers and preserves their exact
            # unknown effects. It is different from a client detaching.
            tasks = list(self._drivers.values())
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            await self.store.aclose()
        finally:
            os.close(self._lock)

    def _guard(self, claim: host_ops.HostCommand, store: RuntimeStore) -> None:
        if not host_ops.claim_current(store, claim) or (
            host_ops.background_submission(store, claim.turn_id).status == "rejected"
        ):
            raise TurnCancellationPendingError("background execution claim was superseded")

    async def _heartbeat(self, claim: host_ops.HostCommand) -> None:
        while True:
            await asyncio.sleep(self.lease_seconds / 3)
            if not await self.store.run_async(
                host_ops.renew_claim, self.store, claim, lease_seconds=self.lease_seconds
            ):
                return

    async def _execute(self, claim: host_ops.HostCommand) -> None:
        heartbeat = asyncio.create_task(self._heartbeat(claim))
        try:
            try:
                turn = await self.store.run_async(self.store.read_turn, claim.turn_id)
            except KeyError:
                turn = None
            if turn is not None and turn.status in _TERMINAL:
                await self._ack(claim)
                return
            source = await self.store.run_async(host_ops.background_submission, self.store, claim.turn_id)
            if source.status == "rejected":
                await self.store.run_async(
                    host_ops.settle_command, self.store, claim, status="rejected",
                    result={"reason": "submission_not_started"},
                )
                return
            request = source.request
            configuration = request["configuration"]
            source_configuration = configuration
            thread = await self.store.run_async(self.store.read_thread, claim.thread_id)
            ceiling = permission_ceiling(configuration)
            source_ceiling = ceiling
            environments: tuple[dict[str, Any], ...] = ()
            if turn is not None and turn.binding_manifest.get("child_environment") is not None:
                def environment_chain() -> list[dict[str, Any]]:
                    result = []
                    current = turn
                    while current.binding_manifest.get("child_environment") is not None:
                        env = current.binding_manifest["child_environment"]
                        edge = self.store.read_child_budget_allocation(current.turn_id)
                        if (env["child_turn_id"] != current.turn_id
                                or env["parent_turn_id"] != edge["parent_turn_id"]
                                or env["workspace"] != self.store.read_thread(current.thread_id).workspace):
                            raise PermissionError("child environment conflicts with canonical lineage")
                        result.append(env)
                        current = self.store.read_turn(str(edge["parent_turn_id"]))
                    if (self.store.read_thread(current.thread_id).workspace
                            != configuration["agent"]["workspace_path"]):
                        raise PermissionError("child source configuration crosses workspace boundary")
                    return result

                environments = tuple(reversed(await self.store.run_async(environment_chain)))
                for env in environments:
                    ceiling = relocate_workspace_permissions(ceiling, Path(env["source_root"]), Path(env["workspace"]))
                configuration = {
                    **configuration, "agent": {**configuration["agent"], "workspace_path": thread.workspace},
                }
            workspace = Path(configuration["agent"]["workspace_path"])
            if not workspace.is_dir():
                await self.store.run_async(
                    host_ops.settle_command, self.store, claim, status="pending",
                    result={"reason": "workspace_offline"}, retry_after=5,
                )
                return
            agent = restore_configuration(configuration)
            if agent._harness_database() != self.database:
                raise ValueError("background configuration names a different runtime database")
            if agent.workspace_path != Path(thread.workspace):
                raise PermissionError("background configuration crosses workspace boundary")
            if self.model is not None:
                model = self.model
                agent._model_factory = lambda: model
            options = dict(request["options"])
            private = tuple(str(self.database) + suffix for suffix in (
                "", "-wal", "-shm", ".artifacts", ".locks", ".host.lock", ".host.log"
            ))
            ceiling = ceiling.model_copy(update={
                "private_paths": tuple(dict.fromkeys((*ceiling.private_paths, *private)))
            })

            def inherited_permissions() -> ResolvedPermissions:
                current = current_configuration_permissions(source_configuration).intersection(source_ceiling)
                for environment in environments:
                    current = relocate_workspace_permissions(
                        current, Path(environment["source_root"]), Path(environment["workspace"])
                    )
                return current.intersection(ceiling)

            context = ToolExecutionContext(
                workspace_root=workspace, cwd=workspace, permissions=ceiling,
                permission_provider=inherited_permissions if environments else None,
                approval_mode=agent.approval_mode or "ask",
                allow_write_tools=options.pop("allow_write_tools"),
                allow_execute_tools=options.pop("allow_execute_tools"),
                allow_web_tools=options.pop("allow_web_tools"),
            )
            saved_policy = thread.settings.get("tool_execution_policy")
            if saved_policy is None and turn is not None:
                saved_policy = turn.binding_manifest.get("tool_execution_policy")
            if saved_policy is not None:
                context = restore_tool_policy(context, saved_policy, restore_session_grants=True)
                if agent.approval_mode is None:
                    context = replace(context, approval_mode=saved_policy.get("approval_mode", "ask"))
            if turn is not None:
                from agent_runtime.harness.child_workspace_ops import ancestor_context

                context = await self.store.run_async(ancestor_context, self.store, turn.turn_id, context)
            async with await Session.open(
                agent=agent, thread_id=claim.thread_id if turn is None else None,
                frozen_turn_id=None if turn is None else claim.turn_id,
                tool_execution_context=context, dispatch_guard=lambda store: self._guard(claim, store),
                approval_mode=agent.approval_mode,
                _background_memory=False, _preserve_children_on_exit=True, **options,
            ) as session:
                self._sessions[claim.turn_id] = session
                if turn is None:
                    if claim.kind != "start" and (
                        claim.request.get("action", "continue") != "continue"
                        or claim.request.get("request_id") is not None
                    ):
                        raise ValueError("unstarted submission requires an explicit continue")
                    predecessor = request.get("previous_turn_id")
                    current = await self.store.run_async(self.store.read_thread, claim.thread_id)
                    if predecessor is not None and current.head_turn_id != predecessor:
                        raise ValueError("queued predecessor is no longer the Thread head")
                    await session.submit(str(request["text"]), files=request.get("files"), turn_id=claim.turn_id)
                else:
                    await session.store.run_async(adopt_execution_protocol, session.store, claim.turn_id)
                    plan = await session.store.run_async(recovery_plan, session.store, claim.turn_id)
                    action = str(claim.request.get("action", "continue"))
                    # An unknown provider result requires an explicit retry
                    # command. Restarting a host never supplies that approval.
                    if claim.kind == "start" and (
                        plan.requires_input or plan.kind in {
                            ContinuationKind.RECONCILE, ContinuationKind.APPROVAL,
                            ContinuationKind.RETRY_MODEL, ContinuationKind.RESUME_CANCELLED,
                        }
                    ):
                        await self._ack(claim)
                        return
                    if not plan.accepts(action) and plan.kind != ContinuationKind.CANCEL:
                        await self.store.run_async(
                            host_ops.settle_command, self.store, claim, status="blocked",
                            result={"reason": plan.reason_code, "actions": list(plan.actions)},
                        )
                        return
                    if plan.kind == ContinuationKind.CANCEL:
                        async with session._active_turn():
                            await session._executor().drive(turn_id=claim.turn_id)
                    else:
                        await session.resume(
                            claim.turn_id, action, user_input=claim.request.get("text"),
                            expected_request_id=claim.request.get("request_id"),
                            automatic=claim.kind == "start" or claim.request.get("automatic") is True,
                            authorization_revision=claim.request.get("authorization_revision"),
                        )
            await self._ack(claim)
        except ConversationBusyError:
            await self.store.run_async(
                host_ops.settle_command, self.store, claim, status="pending",
                result={"reason": "execution_busy"}, retry_after=0.2,
            )
        except asyncio.CancelledError:
            await self._ack(claim)
            raise
        except Exception as error:
            # Record diagnostics, not guessed coding outcomes or raw credentials.
            await self.store.run_async(
                host_ops.settle_command, self.store, claim, status="blocked",
                result={"reason": type(error).__name__, "diagnostic": exception_diagnostic(
                    error, stage="background_execution", turn_id=claim.turn_id
                )},
            )
        finally:
            self._sessions.pop(claim.turn_id, None)
            heartbeat.cancel()
            await asyncio.gather(heartbeat, return_exceptions=True)

    async def _ack(self, claim: host_ops.HostCommand) -> None:
        def acknowledge() -> None:
            with self.store.tx.transaction():
                try:
                    turn = self.store.read_turn(claim.turn_id)
                except KeyError:
                    turn = None
                pending = turn is None or turn.status == "running"
                host_ops.settle_command(
                    self.store, claim, status="pending" if pending else "applied",
                    result={"reason": "execution_not_finished"} if pending else {"turn_id": claim.turn_id},
                )
        await self.store.run_async(acknowledge)

    async def _control(self, claim: host_ops.HostCommand) -> None:
        def apply() -> bool:
            with self.store.tx.transaction():
                self._guard(claim, self.store)
                try:
                    turn = self.store.read_turn(claim.turn_id)
                except KeyError:
                    turn = None
                if claim.kind == "cancel":
                    if turn is not None:
                        turn_ops.request_turn_cancellation(
                            self.store, turn_id=claim.turn_id, reason="Background client requested cancellation"
                        )
                        if turn.status not in _TERMINAL:
                            host_ops.receive_command(
                                self.store, workspace=Path(self.store.read_thread(claim.thread_id).workspace),
                                kind="resume", thread_id=claim.thread_id, turn_id=claim.turn_id,
                                command_id="cancel_finish_" + hashlib.sha256(claim.command_id.encode()).hexdigest(),
                                payload={"action": "abort"},
                            )
                    else:
                        for start in host_ops.list_commands(self.store, claim.thread_id):
                            if start.kind == "start" and start.turn_id == claim.turn_id:
                                host_ops.reject_queued_start(self.store, start.command_id)
                    host_ops.settle_command(
                        self.store, claim, status="applied", result={"cancellation_requested": True}
                    )
                    return True
                if turn is None:
                    if host_ops.background_submission(self.store, claim.turn_id).status in {"rejected", "blocked"}:
                        host_ops.settle_command(
                            self.store, claim, status="rejected", result={"reason": "submission_not_started"}
                        )
                        return False
                    host_ops.settle_command(
                        self.store, claim, status="pending", result={"reason": "turn_not_started"}, retry_after=0.1
                    )
                    return False
                item = turn_ops.receive_turn_input(
                    self.store, thread_id=claim.thread_id, turn_id=claim.turn_id,
                    text=str(claim.request["text"]), input_id="host_input_" + claim.command_id,
                )
                host_ops.settle_command(self.store, claim, status="applied", result={"input_id": item.item_id})
                return False

        try:
            cancel = await self.store.run_async(apply)
        except (ValueError, RuntimeError, KeyError) as error:
            await self.store.run_async(
                host_ops.settle_command, self.store, claim, status="rejected",
                result={"reason": type(error).__name__, "diagnostic": exception_diagnostic(
                    error, stage="background_control", turn_id=claim.turn_id
                )},
            )
            return
        if cancel:
            for command_id, task in tuple(self._drivers.items()):
                command = await self.store.run_async(host_ops.read_command, self.store, command_id)
                if command.turn_id == claim.turn_id:
                    task.cancel()
            # Children spawned inside a Session have their own live drivers,
            # but no separate host claim while their parent still owns them.
            sessions = list(self._sessions.values())
            while sessions:
                session = sessions.pop()
                child = session._subagent_tasks.get(claim.turn_id)
                if child is not None and not child.done():
                    child.cancel()
                sessions.extend(session._subagent_sessions.values())

    async def tick(self) -> None:
        if self._closing:
            return
        for _ in range(64):
            control = await self.store.run_async(
                host_ops.claim_next, self.store, owner=self.owner, control=True, lease_seconds=self.lease_seconds
            )
            if control is None:
                break
            await self._control(control)
        for command_id, task in tuple(self._drivers.items()):
            if task.done():
                task.result() if not task.cancelled() else None
                self._drivers.pop(command_id)
                self._workspaces.pop(command_id)
        await self.store.run_async(host_ops.enqueue_wakeups, self.store)
        for _ in range(self.concurrency * 2):
            if len(self._drivers) >= self.concurrency:
                break
            claim = await self.store.run_async(
                host_ops.claim_next, self.store, owner=self.owner, lease_seconds=self.lease_seconds,
                exclude_commands=frozenset(self._drivers),
            )
            if claim is None:
                break
            workspace = (await self.store.run_async(self.store.read_thread, claim.thread_id)).workspace
            if workspace in self._workspaces.values():
                await self.store.run_async(
                    host_ops.settle_command, self.store, claim, status="pending",
                    result={"reason": "workspace_busy"}, retry_after=0.1,
                )
                continue
            self._workspaces[claim.command_id] = workspace
            self._drivers[claim.command_id] = asyncio.create_task(self._execute(claim))

    async def serve(self, *, stop: asyncio.Event | None = None, idle_timeout: float | None = 60) -> None:
        idle_since = time.monotonic()
        while stop is None or not stop.is_set():
            await self.tick()
            commands = await self.store.run_async(host_ops.list_commands, self.store)
            pending = any(c.status in {"pending", "claimed"} for c in commands)
            if pending or self._drivers:
                idle_since = time.monotonic()
            elif idle_timeout is not None and time.monotonic() - idle_since >= idle_timeout:
                return
            await asyncio.sleep(0.05)
