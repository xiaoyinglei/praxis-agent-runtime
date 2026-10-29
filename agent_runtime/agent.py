"""Public Agent SDK backed exclusively by the Rollout Harness."""

from __future__ import annotations

import asyncio
import hashlib
import logging
import os
from collections.abc import AsyncIterator, Callable, Sequence
from contextlib import ExitStack, asynccontextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Protocol

from agent_runtime.approval import MODE_ORDER, ApprovalMode
from agent_runtime.harness.memory import MemoryConfig
from agent_runtime.harness.protocol import DEFAULT_MAX_STEPS, item_text
from agent_runtime.knowledge import RAGKnowledgeConfig
from agent_runtime.models import (
    ModelControlPlane,
    ModelSpec,
    ModelSwitchRequester,
    validate_model_switch_requester,
)
from agent_runtime.permission_policy import PermissionPolicy, ResolvedPermissions, resolve_permission_sources
from agent_runtime.result import AgentPause, AgentResult
from agent_runtime.streaming.events import StreamEvent
from agent_runtime.streaming.sink import TurnEventDispatcher
from agent_runtime.tools.web_limits import WebLimits
from agent_runtime.workspace import DEFAULT_CHECKPOINT_PATH, DEFAULT_MODEL_SESSION_PATH

if TYPE_CHECKING:
    from agent_runtime.harness import BoundHarnessModel, Session
    from agent_runtime.harness.records import ItemSnapshot
    from agent_runtime.runtime.mcp import MCPConfigTrustDecision
    from agent_runtime.tools.builtins.shell import RunCommandOutput

logger = logging.getLogger(__name__)
_RUNTIME_CLOSE_GRACE_SECONDS = 5.0


class AgentEventSink(Protocol):
    """Receive durable lifecycle events derived from the Rollout log."""

    async def emit(self, event: StreamEvent) -> None: ...


class Agent:
    def __init__(
        self,
        *,
        model: str | None = None,
        checkpoint_db: Path | None = DEFAULT_CHECKPOINT_PATH,
        workspace_path: Path | str | None = None,
        model_session_path: Path | None = DEFAULT_MODEL_SESSION_PATH,
        knowledge: RAGKnowledgeConfig | None = None,
        memory: MemoryConfig | None = None,
        enable_workspace_mcp: bool = True,
        mcp_config_trust: MCPConfigTrustDecision | None = None,
        web_search_key_file: Path | str | None = None,
        web_search_provider: str | None = None,
        web_proxy_url: str | None = None,
        web_limits: WebLimits | None = None,
        permissions: PermissionPolicy | Path | str | None = None,
        managed_permissions: PermissionPolicy | Path | str | None = None,
        approval_mode: ApprovalMode | None = None,
        reasoning_effort: str | None = None,
        max_output_tokens: int | None = None,
        _selection_requester: ModelSwitchRequester = "system",
    ) -> None:
        if approval_mode is not None and approval_mode not in MODE_ORDER:
            raise ValueError("approval_mode must be ask, auto, or full")
        self.approval_mode = approval_mode
        if knowledge is not None and not isinstance(knowledge, RAGKnowledgeConfig):
            raise TypeError("knowledge must be RAGKnowledgeConfig or None")
        if memory is not None and not isinstance(memory, MemoryConfig):
            raise TypeError("memory must be MemoryConfig or None")
        if not isinstance(enable_workspace_mcp, bool):
            raise TypeError("enable_workspace_mcp must be bool")
        if web_limits is not None and not isinstance(web_limits, WebLimits):
            raise TypeError("web_limits must be WebLimits or None")
        self.web_limits = web_limits or WebLimits()
        from agent_runtime.web_config import resolve_web_configuration

        web_configuration = resolve_web_configuration(web_search_provider, web_search_key_file)
        if mcp_config_trust is not None:
            from agent_runtime.runtime.mcp import MCPConfigTrustDecision

            if not isinstance(mcp_config_trust, MCPConfigTrustDecision):
                raise TypeError("mcp_config_trust must be MCPConfigTrustDecision or None")
        selection_requester = validate_model_switch_requester(_selection_requester)
        self.model = model
        self.checkpoint_db = checkpoint_db

        workspace = (Path.cwd() if workspace_path is None else Path(workspace_path))
        self.workspace_path = (workspace.expanduser().resolve())
        for value in (permissions, managed_permissions):
            if value is not None and not isinstance(value, (PermissionPolicy, Path, str)):
                raise TypeError("permissions must be PermissionPolicy, a policy file path, or None")
        self.permissions = permissions
        self.managed_permissions = managed_permissions
        self._input_read_files: tuple[str, ...] = ()
        self.effective_permissions()  # Reject invalid configuration before model/provider setup.
        self.model_session_path = model_session_path
        self.knowledge = knowledge
        self.memory = memory
        self.enable_workspace_mcp = enable_workspace_mcp
        self.mcp_config_trust = mcp_config_trust
        self.web_search_key_file = web_configuration.key_file
        self.web_search_provider = web_configuration.provider
        from agent_runtime.tools.web_http import web_proxy_configuration

        self.web_proxy_url, self.web_no_proxy = web_proxy_configuration(web_proxy_url, dict(os.environ))
        self._selection_requester = selection_requester
        self._followup_model_id: str | None = None
        self._model_factory: Callable[[], BoundHarnessModel] | None = None
        self._request_overrides: dict[str, dict[str, Any]] = {}
        self._pending_request_settings: dict[str, dict[str, Any]] = {}
        self._request_settings_revision = 0
        if reasoning_effort is not None or max_output_tokens is not None:
            self.configure_request(reasoning_effort=reasoning_effort, max_output_tokens=max_output_tokens)

    def models(self) -> list[ModelSpec]:
        return self._get_model_control_plane().list_models()

    def current_model(self) -> ModelSpec:
        return self._get_model_control_plane().current_model()

    def request_settings(self) -> dict[str, Any]:
        return self._get_model_control_plane().request_settings()

    def configure_request(
        self, *, reasoning_effort: str | None = None, max_output_tokens: int | str | None = None,
    ) -> dict[str, Any]:
        plane = self._get_model_control_plane()
        settings = plane.configure_request(
            reasoning_effort=reasoning_effort, max_output_tokens=max_output_tokens,
            persist=self.model_session_path is not None,
        )
        self._request_settings_revision = settings["revision"]
        self._request_overrides[settings["model_id"]] = {
            "model_id": settings["model_id"],
            "options": {key: ("default" if value is None else value)
                        for key, value in settings["preferences"].items()},
        }
        self._pending_request_settings[settings["model_id"]] = self._request_overrides[settings["model_id"]]
        return settings

    def switch_model(self, model_id: str) -> ModelSpec:
        spec = self._get_model_control_plane().switch_model(
            model_id,
            requested_by="user",
            persist=self.model_session_path is not None,
        )
        self.model = spec.id
        self._followup_model_id = spec.id
        self._selection_requester = "user"
        return spec

    def _request_model_switch(self, model_id: str) -> ModelSpec:
        spec = self._get_model_control_plane().request_model_switch(model_id)
        self.model = spec.id
        self._followup_model_id = spec.id
        self._selection_requester = "agent"
        return spec

    async def run(
        self,
        task: str,
        *,
        previous_turn_id: str | None = None,
        files: Sequence[str] | None = None,
        max_steps: int | None = None,
        max_turns: int | None = None,
        max_tokens_total: int | None = None,
        max_cost_micros: int | None = None,
        require_workspace_change: bool = False,
        acceptance_command: str | None = None,
        allow_write_tools: bool = False,
        allow_execute_tools: bool = False,
        allow_web_tools: bool = False,
        approval_mode: ApprovalMode | None = None,
        event_sink: AgentEventSink | None = None,
        _event_dispatcher: TurnEventDispatcher | None = None,
    ) -> AgentResult:
        async with self.session(
            previous_turn_id=previous_turn_id,
            require_workspace_change=require_workspace_change,
            acceptance_command=acceptance_command,
            allow_write_tools=allow_write_tools,
            allow_execute_tools=allow_execute_tools,
            allow_web_tools=allow_web_tools,
            approval_mode=approval_mode,
            max_steps=max_steps, max_turns=max_turns, max_tokens_total=max_tokens_total,
            max_cost_micros=max_cost_micros,
            event_sink=event_sink, _event_dispatcher=_event_dispatcher,
        ) as session:
            return await session.submit(task, files=files)

    @asynccontextmanager
    async def session(
        self, *, previous_turn_id: str | None = None, thread_id: str | None = None,
        require_workspace_change: bool = False,
        acceptance_command: str | None = None,
        allow_write_tools: bool = False, allow_execute_tools: bool = False,
        allow_web_tools: bool = False,
        approval_mode: ApprovalMode | None = None,
        max_steps: int | None = None, max_turns: int | None = None, max_tokens_total: int | None = None,
        max_cost_micros: int | None = None,
        event_sink: AgentEventSink | None = None,
        _event_dispatcher: TurnEventDispatcher | None = None,
        _frozen_turn_id: str | None = None,
        _background_memory: bool = True,
    ) -> AsyncIterator[Session]:
        from agent_runtime.harness.session import Session

        if acceptance_command is not None:
            if not isinstance(acceptance_command, str) or not acceptance_command.strip():
                raise ValueError("acceptance_command must be a non-empty string or None")
            acceptance_command = acceptance_command.strip()
        if max_steps is not None and max_turns is not None and max_steps != max_turns:
            raise ValueError("max_steps and legacy max_turns must agree when both are provided")
        step_budget = max_steps if max_steps is not None else max_turns
        session = await Session.open(
            agent=self, previous_turn_id=previous_turn_id, frozen_turn_id=_frozen_turn_id, thread_id=thread_id,
            require_workspace_change=require_workspace_change,
            acceptance_command=acceptance_command,
            allow_write_tools=allow_write_tools, allow_execute_tools=allow_execute_tools,
            allow_web_tools=allow_web_tools,
            approval_mode=approval_mode if approval_mode is not None else self.approval_mode,
            max_steps=DEFAULT_MAX_STEPS if step_budget is None else step_budget,
            max_tokens_total=max_tokens_total, max_cost_micros=max_cost_micros,
            event_sink=event_sink, event_dispatcher=_event_dispatcher,
            _background_memory=_background_memory,
        )
        async with session:
            yield session

    async def steer(self, text: str, *, expected_turn_id: str) -> ItemSnapshot:
        """Persist guidance without opening a model; target must still be active."""
        from agent_runtime.harness import RuntimeState

        async with await RuntimeState.open(self._harness_database()) as store:
            turn = await store.run_async(store.read_turn, expected_turn_id)
            if (
                Path((await store.run_async(store.read_thread, turn.thread_id)).workspace).resolve()
                != self._workspace_path()
            ):
                raise RuntimeError("Turn belongs to a different workspace security domain")
            return await store.run_async(
                store.receive_turn_input, thread_id=turn.thread_id, turn_id=expected_turn_id, text=text
            )

    def input_status(self, item_id: str) -> str:
        """Inspect a steering receipt without model I/O."""
        from agent_runtime.harness import RuntimeState

        with RuntimeState.open_sync(self._harness_database()) as store:
            item = store.read_item(item_id)
            if Path(store.read_thread(item.thread_id).workspace).resolve() != self._workspace_path():
                raise RuntimeError("input belongs to a different workspace security domain")
            return store.turn_input_status(item_id)

    def memories(self) -> dict[str, object]:
        """Inspect this workspace's memory without opening a model or creating state."""
        from agent_runtime.harness.memory import memory_snapshot
        from agent_runtime.harness.state import RuntimeState

        database = self._harness_database()
        if not database.exists():
            return {"version": 0, "workspace": str(self.workspace_path), "entries": [],
                    "use_memories": self.memory.use_memories if self.memory is not None else False,
                    "generate_memories": self.memory.generate_memories if self.memory is not None else False}
        with RuntimeState.open_sync(database, read_only=True) as store:
            def read() -> dict[str, object]:
                with store.persistence.read_snapshot():
                    state = store.read_memory_state(self.workspace_path)
                    return {**memory_snapshot(store, self.workspace_path),
                            "maintenance_jobs": {key: {name: value for name, value in job.items()
                                                       if name != "candidates"}
                                                 for key, job in store.read_memory_jobs(self.workspace_path).items()},
                            "use_memories": self.memory.use_memories if self.memory is not None
                            else state["use_memories"],
                            "generate_memories": self.memory.generate_memories if self.memory is not None
                            else state["generate_memories"]}

            return store.run_sync(read)

    def configure_memory(self, config: MemoryConfig) -> dict[str, object]:
        """Persist workspace defaults; independent use and contribution controls."""
        from agent_runtime.harness.memory import configure_memory
        from agent_runtime.harness.state import RuntimeState

        if not isinstance(config, MemoryConfig):
            raise TypeError("config must be MemoryConfig")
        with RuntimeState.open_sync(self._harness_database()) as store:
            result = configure_memory(store, self.workspace_path, config)
        self.memory = config
        return result

    def relocate_memory(self, previous_workspace: Path, *,
                        expected_version: int | None = None) -> dict[str, object]:
        """Rebind memory after moving a workspace, using its existing database.

        Both workspace paths must be idle. The source directory may no longer
        exist; historical Threads and Turns retain their original workspace.
        """
        from agent_runtime.harness.ownership import workspace_owner
        from agent_runtime.harness.state import RuntimeState

        previous = Path(previous_workspace).expanduser().resolve()
        workspace = self.workspace_path
        if previous == workspace:
            raise ValueError("previous and current workspace must be different directories")
        if not workspace.is_dir():
            raise ValueError("current workspace must be an existing directory")
        database = self._harness_database()
        if not database.is_file():
            raise FileNotFoundError(f"memory relocation requires an existing database: {database}")
        with ExitStack() as locks:
            for path in sorted((previous, workspace)):
                locks.enter_context(workspace_owner(path, exclusive=True))
            with RuntimeState.open_sync(database) as store:
                return store.relocate_memory(previous_workspace=previous, workspace=workspace,
                                             expected_version=expected_version)

    def delete_memory(self, key: str) -> dict[str, object]:
        from agent_runtime.harness.memory import delete_memory
        from agent_runtime.harness.state import RuntimeState

        with RuntimeState.open_sync(self._harness_database()) as store:
            return delete_memory(store, self.workspace_path, key)

    def remember_memory(self, key: str, text: str, *, title: str | None = None,
                        kind: Literal["preference", "fact", "reference"] = "fact",
                        expected_version: int | None = None) -> dict[str, object]:
        from agent_runtime.harness.memory_management import remember_memory
        from agent_runtime.harness.state import RuntimeState

        with RuntimeState.open_sync(self._harness_database()) as store:
            return remember_memory(store, self.workspace_path, key=key, title=title or key, text=text,
                                   kind=kind, expected_version=expected_version)

    def edit_memory(self, key: str, text: str, *, title: str | None = None,
                    kind: Literal["preference", "fact", "reference"] | None = None,
                    expected_version: int | None = None) -> dict[str, object]:
        from agent_runtime.harness.memory_management import edit_memory
        from agent_runtime.harness.state import RuntimeState

        with RuntimeState.open_sync(self._harness_database()) as store:
            return edit_memory(store, self.workspace_path, key=key, title=title, text=text,
                               kind=kind, expected_version=expected_version)

    def explain_memory(self, key: str, *, offset: int = 0, limit: int = 8) -> dict[str, object]:
        from agent_runtime.harness.memory_management import explain_memory
        from agent_runtime.harness.state import RuntimeState

        with RuntimeState.open_sync(self._harness_database(), read_only=True) as store:
            return explain_memory(store, self.workspace_path, key=key, offset=offset, limit=limit)

    async def refresh_memory(self) -> dict[str, object]:
        """Run one bounded extraction/consolidation pass with the configured model."""
        async with self.session(require_workspace_change=False, _background_memory=False) as session:
            return await session.refresh_memory()

    async def command(
        self, turn_id: str, session_id: str, *, action: Literal["status", "wait", "cancel", "read"] = "status",
        wait_seconds: float = 10, stream: Literal["stdout", "stderr"] = "stdout",
        offset: int = 0, max_bytes: int = 16000,
    ) -> RunCommandOutput:
        """Control the original committed command without calling a model or rerunning it."""
        from agent_runtime.harness.command_recovery import manage_command
        return await manage_command(self, turn_id, session_id, action=action, wait_seconds=wait_seconds,
                                    stream=stream, offset=offset, max_bytes=max_bytes)

    async def list_checkpoints(self, *, thread_id: str | None = None) -> list[dict[str, object]]:
        """List workspace checkpoints recorded for this workspace's Threads."""
        from agent_runtime.checkpoints import _numeric, thread_checkpoint_rows
        from agent_runtime.harness import RuntimeState

        database = self._harness_database()
        rows: list[dict[str, object]] = []
        if database.exists():
            async with await RuntimeState.open(database, read_only=True) as store:
                threads = (
                    (thread_id,)
                    if thread_id is not None
                    else tuple(
                        dict.fromkeys(
                            await store.run_async(lambda: tuple(turn.thread_id for turn in store.list_turns()))
                        )
                    )
                )
                rows = await store.run_async(
                    lambda: [row for current in threads for row in thread_checkpoint_rows(store, current)]
                )
        # Restore anchors live only in the shadow repository (restores happen
        # outside Turns), so merge them in or a rewind could not be undone.
        from agent_runtime.checkpoints import CheckpointService

        recorded = {str(row.get("checkpoint_id")) for row in rows}
        service_rows: list[dict[str, object]] = []
        for entry in await CheckpointService(self._workspace_path()).list_checkpoints():
            if entry.checkpoint_id in recorded or entry.trigger != "restore":
                continue
            service_rows.append({
                "checkpoint_id": entry.checkpoint_id,
                "turn_id": "",
                "trigger": "restore",
                "tree_sha256": None,
                "changed_file_count": 0,
                "wall_time": entry.timestamp,
                "sequence": entry.sequence,
                "turn_input": "回滚前状态（撤销回滚用）",
            })
        rows.extend(service_rows)

        def row_order(row: dict[str, object]) -> tuple[float, int]:
            return _numeric(row.get("wall_time")), int(_numeric(row.get("sequence")))

        rows.sort(key=row_order, reverse=True)
        return rows

    async def rewind(self, checkpoint_id: str) -> dict[str, object]:
        """Restore workspace files to a shadow checkpoint; the current state is anchored first.

        Refuses while any Turn in this workspace is still running or paused:
        restoring underneath live execution would corrupt its workspace views.
        """
        from agent_runtime.checkpoints import CheckpointService
        from agent_runtime.harness import RuntimeState

        async with await RuntimeState.open(self._harness_database(), read_only=True) as store:
            active = await store.run_async(
                lambda: [turn.turn_id for turn in store.list_turns() if turn.status in {"running", "paused"}]
            )
        if active:
            raise RuntimeError(
                "存在未结束的 Turn（运行中或等待输入），请先完成、取消或恢复后再回滚。"
            )
        result = await CheckpointService(self._workspace_path()).restore(checkpoint_id)
        return {
            "target_checkpoint_id": result.target_checkpoint_id,
            "restored": list(result.restored),
            "deleted": list(result.deleted),
            "restore_anchor_id": result.anchor_checkpoint_id,
        }

    async def commit_status(self) -> dict[str, object]:
        """Read the workspace repository's branch and pending change set."""
        from agent_runtime.git_delivery import GitDelivery

        status = await GitDelivery(self._workspace_path()).status()
        return {
            "branch": status.branch,
            "changed": [f"{code} {path}" for code, path in status.changed],
            "changed_file_count": len(status.changed),
            "preview_id": status.preview_id,
        }

    async def commit(
        self, message: str, *, branch_name: str | None = None, expected_preview_id: str | None = None,
    ) -> dict[str, object]:
        """Commit workspace changes on a ``praxis/*`` branch of the real repository.

        A user-authorized delivery operation: the sandbox and the model never
        gain Git write access. Pass ``commit_status()["preview_id"]`` to bind
        execution to that approved candidate. Without a preview ID this call
        directly authorizes the current candidate. Cancellation waits for the
        actual terminal outcome and raises DeliveryInterruptedError with its result.
        """
        from agent_runtime.git_delivery import GitDelivery
        from agent_runtime.harness import RuntimeState

        async with await RuntimeState.open(self._harness_database(), read_only=True) as store:
            active = await store.run_async(
                lambda: [
                    turn.turn_id
                    for turn in store.list_turns()
                    if turn.status in {"running", "paused"}
                    and Path(store.read_thread(turn.thread_id).workspace).resolve() == self._workspace_path()
                ]
            )
        if active:
            raise RuntimeError(
                "存在未结束的 Turn（运行中或等待输入），请先完成、取消或恢复后再提交。"
            )
        result = await GitDelivery(self._workspace_path()).commit(
            message, branch_name=branch_name, expected_preview_id=expected_preview_id,
        )
        return {
            "commit_id": result.commit_id,
            "branch": result.branch,
            "changed_file_count": result.changed_file_count,
            "created_branch": result.created_branch,
        }

    async def pull_request_status(self) -> dict[str, object]:
        """Read what a pull request from the current praxis branch would contain."""
        from agent_runtime.git_delivery import GitDelivery

        status = await GitDelivery(self._workspace_path()).pull_request_status()
        return {
            "branch": status.branch,
            "remote_url": status.remote_url,
            "base_branch": status.base_branch,
            "commit_subjects": list(status.commit_subjects),
            "changed_paths": list(status.changed_paths),
            "uncommitted_count": status.uncommitted_count,
            "open_pull_request_url": status.open_pull_request_url,
            "head_commit": status.head_commit,
            "preview_id": status.preview_id,
        }

    async def open_pull_request(
        self, *, title: str | None = None, body: str | None = None,
        base_branch: str | None = None, expected_preview_id: str | None = None,
    ) -> dict[str, object]:
        """Push the current praxis branch and open (or return) its pull request.

        Publishing is outward-facing: it runs host-side only after the caller
        confirmed the previewed change set, and refuses while Turns are active.
        """
        from agent_runtime.git_delivery import GitDelivery
        from agent_runtime.harness import RuntimeState

        async with await RuntimeState.open(self._harness_database(), read_only=True) as store:
            active = await store.run_async(
                lambda: [
                    turn.turn_id
                    for turn in store.list_turns()
                    if turn.status in {"running", "paused"}
                    and Path(store.read_thread(turn.thread_id).workspace).resolve() == self._workspace_path()
                ]
            )
        if active:
            raise RuntimeError(
                "存在未结束的 Turn（运行中或等待输入），请先完成、取消或恢复后再发布。"
            )
        result = await GitDelivery(self._workspace_path()).open_pull_request(
            title=title, body=body, base_branch=base_branch, expected_preview_id=expected_preview_id,
        )
        return {
            "url": result.url,
            "branch": result.branch,
            "created": result.created,
        }

    async def pull_request_checks(
        self, branch: str | None = None, *,
        include_failure_logs: bool = False,
        wait_seconds: float = 0.0,
        poll_interval_seconds: float = 20.0,
    ) -> dict[str, object]:
        """Read a branch's PR checks; optionally wait until they settle."""
        from agent_runtime.git_delivery import GitDelivery

        report = await GitDelivery(self._workspace_path()).pull_request_checks(
            branch,
            include_failure_logs=include_failure_logs,
            wait_seconds=wait_seconds,
            poll_interval_seconds=poll_interval_seconds,
        )
        return {
            "branch": report.branch,
            "head_commit": report.head_commit,
            "overall": report.overall,
            "pull_request_url": report.pull_request_url,
            "checks": [
                {"name": check.name, "state": check.state,
                 "workflow": check.workflow, "link": check.link}
                for check in report.checks
            ],
            "failure_logs": [
                {"workflow": name, "logs": logs}
                for name, logs in report.failure_logs
            ],
        }

    async def resume(
        self,
        turn_id: str,
        action: str,
        *,
        user_input: str | None = None,
        files: Sequence[str] | None = None,
        event_sink: AgentEventSink | None = None,
        expected_request_id: str | None = None,
    ) -> AgentResult:
        from agent_runtime.harness import RuntimeState, TurnResult

        if action == "abort":
            from agent_runtime.harness import RolloutEventReader
            from agent_runtime.harness.ownership import (
                ConversationRecoveryRequiredError,
                execution_owner,
                recover_owner_loss,
            )

            async with await RuntimeState.open(self._harness_database()) as store:
                turn = await store.run_async(store.read_turn, turn_id)
                thread = await store.run_async(store.read_thread, turn.thread_id)
                if Path(thread.workspace).resolve() != self._workspace_path():
                    raise RuntimeError("turn belongs to a different workspace security domain")
                with execution_owner(self._harness_database(), thread.thread_id):
                    if expected_request_id is not None:
                        from agent_runtime.harness.recovery import recovery_plan

                        if (await store.run_async(recovery_plan, store, turn_id)).request_id != expected_request_id:
                            raise ValueError("recovery request changed; inspect current state before responding")
                    from agent_runtime.harness.command_recovery import cancel_commands

                    recovered = await store.run_async(
                        store.capture_mutation, lambda: recover_owner_loss(store, turn_id, abort=True)
                    )
                    await cancel_commands(self, turn_id)
                    from agent_runtime.harness.recovery import recovery_plan

                    plan = await store.run_async(recovery_plan, store, turn_id)
                    if not plan.accepts("abort"):
                        raise ConversationRecoveryRequiredError(f"取消未完成：{plan.reason_code}；请先核对执行状态。")
                    mutation = await store.run_async(store.capture_mutation, lambda: store.cancel_turn(turn_id=turn_id))
                    cancelled = mutation.value
                    result = await store.run_async(
                        AgentResult._from_harness,
                        TurnResult(
                            thread_id=cancelled.thread_id, turn_id=cancelled.turn_id, answer=None, status="cancelled"
                        ),
                        store=store,
                    )
                    if event_sink is not None:
                        dispatcher = TurnEventDispatcher()
                        dispatcher.subscribe_controlling_sink(event_sink)
                        for replayed in await store.run_async(
                            RolloutEventReader(store).project_committed_batch, (*recovered.records, *mutation.records)
                        ):
                            await dispatcher.emit(replayed.event, cursor=replayed.cursor)
            return result
        async with self.session(_frozen_turn_id=turn_id, event_sink=event_sink) as session:
            return await session.resume(turn_id, action, user_input=user_input, files=files,
                                        expected_request_id=expected_request_id)

    async def recovery_state(self, turn_id: str) -> dict[str, Any]:
        """Inspect available actions without loading a model or recovering execution.

        Availability is an observation. resume rechecks it under execution ownership;
        pass request_id as expected_request_id to answer that specific request.
        """
        from agent_runtime.harness.ownership import ConversationBusyError, execution_owner
        from agent_runtime.harness.recovery import recovery_plan
        from agent_runtime.harness.state import RuntimeState

        async with await RuntimeState.open(self._harness_database(), read_only=True) as store:
            turn = await store.run_async(store.read_turn, turn_id)
            if (
                Path((await store.run_async(store.read_thread, turn.thread_id)).workspace).resolve()
                != self._workspace_path()
            ):
                raise RuntimeError("turn belongs to a different workspace security domain")
            try:
                with execution_owner(self._harness_database(), turn.thread_id):
                    return (await store.run_async(recovery_plan, store, turn_id)).as_dict()
            except ConversationBusyError:
                return {
                    "turn_id": turn_id,
                    "status": turn.status,
                    "actions": [],
                    "reason_code": "execution_active",
                    "evidence_ids": [turn_id],
                    "request_id": None,
                    "requires_input": False,
                    "question": None,
                    "choices": [],
                }

    async def read_result(self, turn_id: str) -> AgentResult:
        from agent_runtime.harness import RuntimeState, TurnResult

        async with await RuntimeState.open(self._harness_database(), read_only=True) as store:
            turn = await store.run_async(store.read_turn, turn_id)
            thread = await store.run_async(store.read_thread, turn.thread_id)
            if Path(thread.workspace).resolve() != self._workspace_path():
                raise RuntimeError("turn belongs to a different workspace security domain")
            answer: str | None = None
            if turn.status == "completed":
                answer = item_text(await store.run_async(store.read_turn_answer, turn_id))
            pending = await store.run_async(
                lambda: next(
                    (
                        interaction.request_id
                        for interaction in reversed(store.list_interactions(turn_id))
                        if interaction.status == "pending"
                    ),
                    None,
                )
            )
            return await store.run_async(
                AgentResult._from_harness,
                TurnResult(
                    thread_id=turn.thread_id,
                    turn_id=turn.turn_id,
                    answer=answer,
                    status=turn.status,
                    interaction_id=pending,
                ),
                store=store,
            )

    async def stream(
        self,
        task: str,
        *,
        previous_turn_id: str | None = None,
        files: Sequence[str] | None = None,
        max_steps: int | None = None,
        max_turns: int | None = None,
        max_tokens_total: int | None = None,
        max_cost_micros: int | None = None,
        require_workspace_change: bool = False,
        acceptance_command: str | None = None,
        allow_write_tools: bool = False,
        allow_execute_tools: bool = False,
        allow_web_tools: bool = False,
        approval_mode: ApprovalMode | None = None,
    ) -> AsyncIterator[StreamEvent]:
        """Yield committed durable events while the Turn is still running."""

        dispatcher = TurnEventDispatcher()
        stream = dispatcher.subscribe_controlling()
        run_task = asyncio.create_task(
            self.run(
                task,
                previous_turn_id=previous_turn_id,
                files=files,
                max_steps=max_steps, max_turns=max_turns,
                max_tokens_total=max_tokens_total,
                max_cost_micros=max_cost_micros,
                require_workspace_change=require_workspace_change,
                acceptance_command=acceptance_command,
                allow_write_tools=allow_write_tools,
                allow_execute_tools=allow_execute_tools,
                allow_web_tools=allow_web_tools,
                approval_mode=approval_mode,
                _event_dispatcher=dispatcher,
            )
        )
        try:
            while True:
                if not stream.empty:
                    yield stream.receive_nowait()
                    continue
                if run_task.done():
                    break
                next_event = asyncio.create_task(stream.wait_available())
                done, _pending = await asyncio.wait(
                    {run_task, next_event},
                    return_when=asyncio.FIRST_COMPLETED,
                )
                if next_event in done:
                    continue
                else:
                    next_event.cancel()
                    await asyncio.gather(next_event, return_exceptions=True)
            await run_task
        finally:
            if not run_task.done():
                run_task.cancel()
                await asyncio.gather(run_task, return_exceptions=True)
            dispatcher.close()

    async def pending_input(self, turn_id: str) -> AgentPause | None:
        from agent_runtime.harness import RuntimeState, TurnResult

        async with await RuntimeState.open(self._harness_database(), read_only=True) as store:
            turn = await store.run_async(store.read_turn, turn_id)
            thread = await store.run_async(store.read_thread, turn.thread_id)
            if Path(thread.workspace).resolve() != self._workspace_path():
                raise RuntimeError("turn belongs to a different workspace security domain")
            if turn.status != "paused":
                return None
            projected = await store.run_async(
                AgentResult._from_harness,
                TurnResult(thread_id=thread.thread_id, turn_id=turn_id, answer=None, status="paused"),
                store=store,
            )
            return projected.pause

    def _harness_database(self) -> Path:
        if self.checkpoint_db is not None:
            return Path(self.checkpoint_db).expanduser().resolve()
        return self._workspace_path() / ".praxis" / "runtime" / "rollout.sqlite3"

    def _workspace_path(self) -> Path:
        return self.workspace_path

    def effective_permissions(self) -> ResolvedPermissions:
        """Reload trusted sources; saved conversation grants cannot widen them."""
        return resolve_permission_sources(self.workspace_path, self.permissions, self.managed_permissions,
                                          read_files=self._input_read_files)

    def _reference_harness_files(
        self,
        files: Sequence[str],
        *,
        permission_ceiling: ResolvedPermissions | None = None,
    ) -> tuple[dict[str, object], ...]:
        from agent_runtime.local_files import resolve_local_file

        if not files:
            return ()
        originals = [resolve_local_file(value, base=self._workspace_path(), resolve_symlinks=False) for value in files]
        paths = [path.resolve() for path in originals]
        selected = tuple(dict.fromkeys((*self._input_read_files, *(str(path) for path in (*originals, *paths)))))
        permissions = resolve_permission_sources(self.workspace_path, self.permissions, self.managed_permissions,
                                                 read_files=selected)
        if permission_ceiling is not None:
            permissions = permissions.intersection(permission_ceiling.with_read_files(selected))
        for path in originals:
            # Validate lexical denials as well as the canonical file reference.
            lexical_permissions = permissions.with_read_files((str(path),))
            if not lexical_permissions.path_access(path):
                raise PermissionError(lexical_permissions.path_reason(path))
        for path in paths:
            if not permissions.path_access(path):
                raise PermissionError(permissions.path_reason(path))
        values: list[dict[str, object]] = []
        for path in dict.fromkeys(paths):
            digest = hashlib.sha256()
            with path.open("rb") as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(chunk)
            values.append(
                {
                    "original_path": str(path),
                    "path": str(path),
                    "storage": "reference",
                    "sha256": digest.hexdigest(),
                    "size_bytes": path.stat().st_size,
                }
            )
        return tuple(values)

    def _get_model_control_plane(self) -> ModelControlPlane:
        from agent_runtime.runtime.builder import build_model_control_plane

        workspace = self._workspace_path()
        session_path = self.model_session_path
        if session_path is not None and not session_path.is_absolute():
            session_path = workspace / session_path
        plane = build_model_control_plane(
            model_id=self.model,
            selection_requester=self._selection_requester,
            session_path=session_path,
            workspace=workspace,
        )
        if self.model_session_path is None:
            plane.state.request_settings_revision = self._request_settings_revision
            for model_id, override in self._request_overrides.items():
                from agent_runtime.core.generation_settings import ModelRequestPreferences

                plane.state.request_preferences[model_id] = ModelRequestPreferences.model_validate({
                    key: None if value == "default" else value for key, value in override["options"].items()
                })
        return plane


async def _close_owned_sync_resource(resource: object, *, label: str) -> None:
    close_method = getattr(resource, "close", None)
    if not callable(close_method):
        return
    try:
        await asyncio.wait_for(
            asyncio.to_thread(close_method),
            timeout=_RUNTIME_CLOSE_GRACE_SECONDS,
        )
    except TimeoutError:
        logger.warning(
            "%s close exceeded %.1fs grace period",
            label,
            _RUNTIME_CLOSE_GRACE_SECONDS,
        )
    except Exception as exc:
        logger.warning("%s close failed (%s)", label, type(exc).__name__[:120])
