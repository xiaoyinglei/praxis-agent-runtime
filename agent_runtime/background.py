"""Model-free submission and attachment to the same durable Harness database."""
from __future__ import annotations

import asyncio
import fcntl
import json
import math
import os
import subprocess
import sys
import threading
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel

from agent_runtime.harness import RuntimeStore, host_ops
from agent_runtime.harness.events import RolloutEventReader
from agent_runtime.permission_policy import PermissionPolicy, ResolvedPermissions, resolve_permission_sources

if TYPE_CHECKING:
    from agent_runtime.agent import Agent
    from agent_runtime.result import AgentResult


def _policy(value: PermissionPolicy | Path | str | None) -> dict[str, Any] | str | None:
    if isinstance(value, PermissionPolicy):
        return value.model_dump(mode="json")
    return str(Path(value).expanduser().resolve()) if value is not None else None


def freeze_configuration(agent: Agent) -> dict[str, Any]:
    """Snapshot choices and permission ceilings, without opening a model provider."""
    plane = agent._get_model_control_plane()
    spec = plane.current_model()
    settings = agent.request_settings()
    values: dict[str, Any] = {
        "model": spec.id, "workspace_path": str(agent.workspace_path),
        "checkpoint_db": str(agent._harness_database()), "model_session_path": None,
        "enable_workspace_mcp": agent.enable_workspace_mcp,
        "approval_mode": agent.approval_mode,
        "permissions": _policy(agent.permissions), "managed_permissions": _policy(agent.managed_permissions),
        "web_search_key_file": None if agent.web_search_key_file is None else str(agent.web_search_key_file),
        "web_search_provider": agent.web_search_provider, "web_proxy_url": agent.web_proxy_url,
        "reasoning_effort": settings["preferences"].get("reasoning_effort"),
        "max_output_tokens": settings["preferences"].get("max_output_tokens"),
        "_selection_requester": plane.state.selection_requester,
    }
    for name in ("knowledge", "memory", "web_limits"):
        value = getattr(agent, name)
        values[name] = value.model_dump(mode="json") if isinstance(value, BaseModel) else None
    trust = agent.mcp_config_trust
    values["mcp_config_trust"] = None if trust is None else {
        "config_path": str(trust.config_path), "source": trust.source, "config_sha256": trust.config_sha256,
    }
    return {
        "agent": values, "ceiling": agent.effective_permissions().model_dump(mode="json"),
        "model_definition": plane.catalog.definition(spec.id).model_dump(mode="json"),
    }


def restore_configuration(configuration: Mapping[str, Any]) -> Agent:
    from agent_runtime.agent import Agent
    from agent_runtime.harness.memory import MemoryConfig
    from agent_runtime.knowledge import RAGKnowledgeConfig
    from agent_runtime.runtime.mcp import MCPConfigTrustDecision
    from agent_runtime.tools.web_limits import WebLimits

    values = dict(configuration["agent"])
    models: dict[str, type[BaseModel]] = {
        "knowledge": RAGKnowledgeConfig, "memory": MemoryConfig,
        "web_limits": WebLimits,
    }
    for name, model in models.items():
        if values.get(name) is not None:
            values[name] = model.model_validate_json(json.dumps(values[name]))
    if values.get("mcp_config_trust") is not None:
        trust = dict(values["mcp_config_trust"])
        trust["config_path"] = Path(trust["config_path"])
        values["mcp_config_trust"] = MCPConfigTrustDecision(**trust)
    for name in ("permissions", "managed_permissions"):
        if isinstance(values.get(name), Mapping):
            values[name] = PermissionPolicy.model_validate_json(json.dumps(values[name]))
    agent = Agent(**values)
    plane = agent._get_model_control_plane()
    if plane.catalog.definition(plane.current_model().id).model_dump(mode="json") != configuration["model_definition"]:
        raise ValueError("queued model definition changed; resubmit with the intended model configuration")
    return agent


def permission_ceiling(configuration: Mapping[str, Any]) -> ResolvedPermissions:
    return ResolvedPermissions.model_validate_json(json.dumps(configuration["ceiling"]))


def current_configuration_permissions(configuration: Mapping[str, Any]) -> ResolvedPermissions:
    """Reload permission declarations in their original workspace domain."""
    values = configuration["agent"]

    def declaration(value: Any) -> PermissionPolicy | str | None:
        if isinstance(value, Mapping):
            return PermissionPolicy.model_validate_json(json.dumps(value))
        if value is None or isinstance(value, str):
            return value
        raise ValueError("invalid saved permission declaration")

    return resolve_permission_sources(
        Path(values["workspace_path"]), declaration(values.get("permissions")),
        declaration(values.get("managed_permissions")),
    )


def ensure_host(database: Path) -> None:
    """Start a detached process; its OS lock elects one host per database."""
    database = database.expanduser().resolve()
    fd = os.open(str(database) + ".host.lock", os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return
    finally:
        os.close(fd)
    log_fd = os.open(str(database) + ".host.log", os.O_CREAT | os.O_APPEND | os.O_WRONLY | os.O_NOFOLLOW, 0o600)
    try:
        process = subprocess.Popen(
            [sys.executable, "-m", "agent_runtime.host_worker", "--database", str(database)],
            stdin=subprocess.DEVNULL, stdout=log_fd, stderr=log_fd,
            cwd=database.parent, start_new_session=True, close_fds=True,
            env={**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parents[1])},
        )
        threading.Thread(target=process.wait, daemon=True, name="praxis-host-reaper").start()
    finally:
        os.close(log_fd)


class BackgroundHandle:
    """An attachment to Thread/Turn identities; disposing it never cancels work."""

    def __init__(
        self, *, database: Path, workspace: Path, thread_id: str, turn_id: str, autostart: bool = True,
    ) -> None:
        self.database = database.expanduser().resolve()
        self.workspace = workspace.expanduser().resolve()
        self.thread_id, self.turn_id, self.autostart = thread_id, turn_id, autostart

    def _authorize(self, store: RuntimeStore) -> None:
        actual = Path(store.read_thread(self.thread_id).workspace).resolve()
        submission = host_ops.background_submission(store, self.turn_id)
        try:
            turn = store.read_turn(self.turn_id)
        except KeyError:
            if submission.thread_id != self.thread_id:
                raise PermissionError("background target belongs to another Thread") from None
        else:
            if turn.thread_id != self.thread_id:
                raise PermissionError("background target belongs to another Thread")
        if actual == self.workspace:
            return
        # A controller in the normal project entrypoint may attach its owned
        # descendants. Only canonical edges plus matching environment provenance
        # establish that authority; claimed binding parent IDs are insufficient.
        current = self.turn_id
        seen = set()
        while current not in seen:
            seen.add(current)
            edge = store.tx.first("child_budget_allocations", where={"child_turn_id": current})
            if edge is None:
                break
            child = store.read_turn(current)
            environment = child.binding_manifest.get("child_environment")
            parent = store.read_turn(str(edge["parent_turn_id"]))
            parent_workspace = Path(store.read_thread(parent.thread_id).workspace).resolve()
            if (environment is None or environment.get("parent_turn_id") != parent.turn_id
                    or Path(environment["source_root"]).resolve() != parent_workspace
                    or environment.get("workspace") != store.read_thread(child.thread_id).workspace):
                break
            if parent_workspace == self.workspace:
                self.workspace = actual
                return
            current = parent.turn_id
        raise PermissionError("background attachment belongs to another workspace")

    async def status(self) -> dict[str, Any]:
        async with await RuntimeStore.open(self.database, read_only=True) as store:
            def read() -> dict[str, Any]:
                self._authorize(store)
                submission = host_ops.background_submission(store, self.turn_id)
                try:
                    turn = store.read_turn(self.turn_id)
                except KeyError:
                    turn = None
                from agent_runtime.harness.child_workspace_ops import state as environment_state
                from agent_runtime.harness.delivery import inspect as inspect_delivery
                from agent_runtime.harness.recovery import recovery_plan

                return {
                    "thread_id": self.thread_id, "turn_id": self.turn_id,
                    "workspace": str(self.workspace),
                    "queue_status": submission.status,
                    "status": None if turn is None else turn.status,
                    "delivery": None if turn is None else inspect_delivery(store, self.turn_id),
                    "receipt": dict(submission.result),
                    "recovery": None if turn is None else recovery_plan(store, self.turn_id).as_dict(),
                    "commands": [
                        {"command_id": c.command_id, "kind": c.kind, "status": c.status,
                         "generation": c.generation, "result": dict(c.result)}
                        for c in host_ops.list_commands(store, self.thread_id) if c.turn_id == self.turn_id
                    ],
                    "children": [
                        {
                            "turn_id": c["child_turn_id"],
                            "thread_id": store.read_turn(str(c["child_turn_id"])).thread_id,
                            "status": store.read_turn(str(c["child_turn_id"])).status,
                            "allocation_status": c["status"],
                            "environment": environment_state(store, self.turn_id, str(c["child_turn_id"])),
                            "recovery": recovery_plan(store, str(c["child_turn_id"])).as_dict(),
                        }
                        for c in store.tx.rows("child_budget_allocations", where={"parent_turn_id": self.turn_id})
                    ],
                }
            return await store.run_async(store.read_snapshot, read)

    async def wait(self, *, timeout: float | None = None) -> dict[str, Any]:
        if timeout is not None and (not math.isfinite(timeout) or timeout < 0):
            raise ValueError("timeout must be finite and nonnegative")
        deadline = None if timeout is None else time.monotonic() + timeout
        while True:
            value = await self.status()
            if value["status"] in {"paused", "interrupted", "completed", "failed", "cancelled"} or (
                value["status"] is None and value["queue_status"] in {"blocked", "rejected"}
            ):
                return value
            if deadline is not None and time.monotonic() >= deadline:
                raise TimeoutError("waiting ended; background execution was not cancelled")
            await asyncio.sleep(0.05)

    async def _command(
        self, kind: str, payload: Mapping[str, Any], command_id: str | None
    ) -> host_ops.HostCommand:
        async with await RuntimeStore.open(self.database) as store:
            def receive() -> host_ops.HostCommand:
                with store.tx.transaction():
                    self._authorize(store)
                    request = dict(payload)
                    if command_id is not None:
                        try:
                            prior = host_ops.read_command(store, command_id)
                        except KeyError:
                            prior = None
                        if prior is not None:
                            for key in ("authorization_revision",):
                                if key in prior.request:
                                    request[key] = prior.request[key]
                    if kind == "resume" and "authorization_revision" not in request:
                        try:
                            turn = store.read_turn(self.turn_id)
                        except KeyError:
                            request["authorization_revision"] = 0
                        else:
                            request["authorization_revision"] = turn.applied_thread_sequence
                    return host_ops.receive_command(
                        store, workspace=self.workspace, kind=kind, payload=request,
                        command_id=command_id, thread_id=self.thread_id, turn_id=self.turn_id,
                    )
            receipt = await store.run_async(receive)
        if self.autostart:
            await asyncio.to_thread(ensure_host, self.database)
        return receipt

    async def send(self, text: str, *, input_id: str | None = None) -> host_ops.HostCommand:
        if not text.strip():
            raise ValueError("input must be nonempty")
        return await self._command("input", {"text": text}, input_id)

    async def respond(
        self, request_id: str, action: str, *, text: str | None = None, input_id: str | None = None
    ) -> host_ops.HostCommand:
        return await self._command(
            "resume", {"request_id": request_id, "action": action, "text": text}, input_id
        )

    async def resume(self, *, action: str = "continue", input_id: str | None = None) -> host_ops.HostCommand:
        return await self._command("resume", {"action": action}, input_id)

    async def cancel(self, *, input_id: str | None = None) -> host_ops.HostCommand:
        return await self._command("cancel", {}, input_id)

    async def result(self) -> AgentResult:
        from agent_runtime.agent import Agent

        async with await RuntimeStore.open(self.database, read_only=True) as store:
            await store.run_async(self._authorize, store)
        agent = Agent(workspace_path=self.workspace, checkpoint_db=self.database, model_session_path=None)
        return await agent.read_result(self.turn_id)

    async def events(self, *, after: str | None = None, limit: int = 100) -> tuple[dict[str, Any], ...]:
        async with await RuntimeStore.open(self.database, read_only=True) as store:
            await store.run_async(self._authorize, store)
            def read() -> tuple[dict[str, Any], ...]:
                events = RolloutEventReader(store).read(self.thread_id, after=after, limit=limit)
                return tuple({"cursor": e.cursor, "event": asdict(e.event)} for e in events)
            return await store.run_async(read)


async def submit_background(
    agent: Agent, text: str, *, input_id: str | None = None, files: Sequence[str] | None = None,
    previous_turn_id: str | None = None, autostart: bool = True,
    max_steps: int | None = None, max_tokens_total: int | None = None, max_cost_micros: int | None = None,
    require_workspace_change: bool = False, acceptance_command: str | None = None,
    allow_write_tools: bool = False, allow_execute_tools: bool = False, allow_web_tools: bool = False,
) -> BackgroundHandle:
    if not isinstance(text, str) or not text.strip():
        raise ValueError("task must be nonempty")
    from agent_runtime.harness.execution_context import step_limit_from_binding

    step_limit_from_binding({"model_step_budget": max_steps})
    for name, value in (("max_tokens_total", max_tokens_total), ("max_cost_micros", max_cost_micros)):
        if value is not None and (type(value) is not int or value < 1):
            raise ValueError(f"{name} must be a positive integer or None")
    if acceptance_command is not None and (
        not isinstance(acceptance_command, str) or not acceptance_command.strip()
    ):
        raise ValueError("acceptance_command must be a non-empty string or None")
    configuration = freeze_configuration(agent)
    payload = {
        "text": text, "configuration": configuration, "files": list(files or ()),
        "previous_turn_id": previous_turn_id,
        "options": {
            "max_steps": max_steps, "max_tokens_total": max_tokens_total, "max_cost_micros": max_cost_micros,
            "require_workspace_change": require_workspace_change, "acceptance_command": acceptance_command,
            "allow_write_tools": allow_write_tools, "allow_execute_tools": allow_execute_tools,
            "allow_web_tools": allow_web_tools,
        },
    }
    database = agent._harness_database()
    async with await RuntimeStore.open(database) as store:
        def receive() -> host_ops.HostCommand:
            with store.tx.transaction():
                thread_id = None
                prior = None
                if input_id is not None:
                    try:
                        prior = host_ops.read_command(store, input_id)
                    except KeyError:
                        pass
                if previous_turn_id is not None:
                    turn = store.read_turn(previous_turn_id)
                    thread = store.read_thread(turn.thread_id)
                    if Path(thread.workspace).resolve() != agent.workspace_path:
                        raise PermissionError("predecessor belongs to another workspace")
                    if prior is None and (
                        turn.status not in {"completed", "failed", "cancelled"} or thread.head_turn_id != turn.turn_id
                    ):
                        raise ValueError("background follow-up requires the terminal thread head")
                    thread_id = thread.thread_id
                return host_ops.receive_command(
                    store, workspace=agent.workspace_path, kind="start", payload=payload,
                    command_id=input_id, thread_id=thread_id,
                )
        receipt = await store.run_async(receive)
    handle = BackgroundHandle(
        database=database, workspace=agent.workspace_path, thread_id=receipt.thread_id,
        turn_id=receipt.turn_id, autostart=autostart,
    )
    if autostart:
        await asyncio.to_thread(ensure_host, database)
    return handle
