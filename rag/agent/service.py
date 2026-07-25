from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import AsyncGenerator, AsyncIterator, Mapping, Sequence
from contextlib import aclosing, asynccontextmanager
from pathlib import Path
from typing import Literal, cast
from uuid import uuid4

from langchain_core.messages import BaseMessage
from langgraph.checkpoint.base import BaseCheckpointSaver
from pydantic import BaseModel, ConfigDict, Field, field_validator

from agent_runtime.planning import AgentPlan, PlanEvent
from rag.agent.core.checkpointing import (
    LangGraphCheckpointStore,
    aclose_agent_checkpointer,
    create_agent_checkpointer,
    reconcile_tool_manifest,
)
from rag.agent.core.context import AgentRunConfig, TurnRegistry
from rag.agent.core.definition import AgentRuntimePolicy
from rag.agent.core.finalization import FinishCandidateBuilder
from rag.agent.core.goal_contract import GoalCompatibilityConfig, GoalSpec
from rag.agent.core.human_input import (
    HumanInputRequest,
    HumanInputResponse,
)
from rag.agent.core.llm_registry import ModelResolver
from rag.agent.core.messages import ModelMessage, snapshot_model_message
from rag.agent.core.model_provider_runtime import ModelProviderResolver
from rag.agent.core.model_request import (
    ModelCallRecord,
    build_stable_context,
    build_tool_manifest,
    split_turn_context,
)
from rag.agent.core.output_finalizer import (
    StructuredOutputFinalizer,
    create_model_structured_output_finalizer,
)
from rag.agent.core.output_models import ValidatedFinalOutput, output_model_path
from rag.agent.core.runtime_diagnostics import (
    AgentLatencyProfile,
    RuntimeDiagnostic,
    ToolCallMetrics,
    merge_runtime_diagnostics,
)
from rag.agent.core.turn_contracts import (
    ToolCallPlan,
    ToolManifestDriftStatus,
)
from rag.agent.file_manifest import FileManifest, build_file_manifest
from rag.agent.loop.runtime import AgentLoop, ModelTurnProvider
from rag.agent.loop.state import (
    LoopPause,
    LoopState,
    LoopTerminal,
    LoopTransition,
    create_loop_state,
)
from rag.agent.loop.stop_hooks import StopHookRunner, build_stop_hooks
from rag.agent.memory.compactor import LoopContextCompactor, MessageCompactor
from rag.agent.memory.models import MemoryPolicy
from rag.agent.memory.store import WorkspaceMemoryStore
from rag.agent.skills.catalog import SkillCatalog
from rag.agent.skills.runtime import SkillRuntime
from rag.agent.streaming.events import StreamEvent
from rag.agent.streaming.sink import StreamEventSink
from rag.agent.tools.builtins import RESIDENT_CODING_TOOL_NAMES
from rag.agent.tools.executor import (
    ExecutionStatus,
    ToolExecutionRecord,
    ToolExecutor,
)
from rag.agent.tools.permissions import ToolExecutionContext
from rag.agent.tools.registry import ToolRegistry
from rag.agent.tools.selection import resolve_tool_options, select_tools
from rag.agent.tools.tool import JsonValue, ToolCall, ToolCallOrigin, ToolResult
from rag.agent.turns import (
    RuntimeBinding,
    TurnStateError,
    TurnStatus,
    TurnStore,
)
from rag.agent.workspace import (
    WorkspaceRuntime,
    create_temp_workspace,
    import_files,
    open_workspace,
)
from rag.schema.query import AnswerCitation, EvidenceItem

logger = logging.getLogger(__name__)
_TURN_LEASE_SECONDS = 300.0
_TURN_LEASE_HEARTBEAT_SECONDS = 60.0
type _ResumeDecision = Literal[
    "allow_once",
    "deny",
    "mark_completed",
    "mark_failed",
]


class AgentRunRequest(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    message: str = Field(min_length=1)
    previous_turn_id: str | None = None
    turn_id: str | None = None
    max_turns: int | None = Field(default=None, gt=0, strict=True)
    max_context_tokens: int | None = Field(default=None, gt=0)
    llm_budget_total: int | None = Field(default=None, gt=0)
    pending_tool_calls: list[ToolCallPlan] = Field(default_factory=list)
    approved_tool_call_ids: list[str] = Field(default_factory=list)
    denied_tool_call_ids: list[str] = Field(default_factory=list)
    messages: list[BaseMessage] = Field(default_factory=list)
    conversation_history: list[ModelMessage] = Field(default_factory=list)
    input_files: list[str] = Field(default_factory=list)
    workspace_path: str | None = None
    memory_policy: MemoryPolicy | None = None
    goal_spec: GoalSpec | None = None
    tools: tuple[str, ...] | None = None
    disabled_tools: tuple[str, ...] = ()
    allow_write_tools: bool = False
    allow_execute_tools: bool = False
    allow_discovery_tools: bool | None = None

    @field_validator("tools", "disabled_tools", mode="before")
    @classmethod
    def _tuple_names(cls, value: object) -> object:
        if value is None:
            return None
        if isinstance(value, (str, bytes)):
            raise TypeError("tool options must be sequences of names")
        return tuple(value)  # type: ignore[arg-type]

    def to_run_config(self, definition: AgentRuntimePolicy) -> AgentRunConfig:
        turn_id = self.turn_id or str(uuid4())
        return AgentRunConfig(
            turn_id=turn_id,
            max_turns=self.max_turns,
            max_context_tokens=self.max_context_tokens,
            llm_budget_total=self.llm_budget_total,
            tool_policy=definition.tool_policy,
            memory_policy=self.memory_policy or MemoryPolicy(),
        )


class AgentRunResult(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    turn_id: str
    status: str
    final_answer: str | None = None
    final_output: BaseModel | None = None
    output_validation_errors: list[dict[str, object]] = Field(default_factory=list)
    stop_reason: str | None = None
    tool_results: list[ToolResult] = Field(default_factory=list)
    tool_call_arguments: dict[str, dict[str, JsonValue]] = Field(default_factory=dict)
    model_call_records: list[ModelCallRecord] = Field(default_factory=list)
    evidence: list[EvidenceItem] = Field(default_factory=list)
    citations: list[AnswerCitation] = Field(default_factory=list)
    input_files: list[str] = Field(default_factory=list)
    iteration: int = 0
    groundedness_flag: bool = False
    insufficient_evidence_flag: bool = False
    needs_user_input: str | None = None
    human_input_request: HumanInputRequest | None = None
    pending_tool_calls_summary: list[dict[str, str]] = Field(default_factory=list)
    workspace_path: str | None = None
    runtime_diagnostics: list[RuntimeDiagnostic] = Field(default_factory=list)
    tool_call_metrics: ToolCallMetrics | None = None
    latency_profile: AgentLatencyProfile | None = None
    plan: AgentPlan | None = None
    plan_events: list[PlanEvent] = Field(default_factory=list)

    @classmethod
    def from_loop_result(
        cls,
        state: LoopState,
        *,
        definition: AgentRuntimePolicy | None = None,
        workspace_path: str | None = None,
    ) -> AgentRunResult:
        run_config = state["run_config"]
        terminal = state["terminal"]
        pause = state["pause"]
        evidence, citations = _result_provenance(state["tool_results"])
        is_terminal = state["status"] in {"completed", "failed"}
        return cls(
            turn_id=run_config.turn_id,
            status=("done" if state["status"] == "completed" else state["status"]),
            final_answer=state["finish_state"].final_answer,
            final_output=_restore_final_output(
                state["finish_state"].final_output,
                definition=definition,
            ),
            output_validation_errors=list(state["finish_state"].output_validation_errors),
            stop_reason=None if terminal is None else terminal.stop_reason,
            tool_results=list(state["tool_results"]),
            tool_call_arguments={
                result.tool_call_id: dict(call.arguments)
                for result in state["tool_results"]
                if (call := state["canonical_tool_calls"].get(result.tool_call_id)) is not None
            },
            model_call_records=list(state.get("model_call_records", ())),
            evidence=evidence,
            citations=citations,
            input_files=list(state.get("input_files", ())),
            iteration=state["iteration"],
            groundedness_flag=_result_flag(
                state["tool_results"],
                "groundedness_flag",
            ),
            insufficient_evidence_flag=(
                _result_flag(state["tool_results"], "insufficient_evidence")
                or _result_flag(
                    state["tool_results"],
                    "insufficient_evidence_flag",
                )
            ),
            needs_user_input=(None if is_terminal or pause is None else pause.reason),
            human_input_request=(None if is_terminal else state["approval_request"]),
            pending_tool_calls_summary=[
                {
                    "tool_call_id": call.tool_call_id,
                    "tool_name": call.tool_name,
                }
                for call in state["pending_tool_calls"]
            ],
            workspace_path=workspace_path,
            runtime_diagnostics=list(state["runtime_diagnostics"]),
            tool_call_metrics=cast(ToolCallMetrics | None, state.get("tool_call_metrics")),
            latency_profile=state.get("latency_profile"),
            plan=state["plan_state"].agent_plan,
            plan_events=list(state["plan_state"].plan_events),
        )


class AgentService:
    """Assemble one frozen Tool snapshot and reuse one executor everywhere."""

    def __init__(
        self,
        *,
        definition: AgentRuntimePolicy,
        tool_registry: ToolRegistry,
        model_turn_provider: ModelTurnProvider | None = None,
        output_finalizer: StructuredOutputFinalizer | None = None,
        model_registry: ModelResolver | None = None,
        checkpointer: BaseCheckpointSaver[str] | None = None,
        runtime_diagnostics: Sequence[RuntimeDiagnostic] = (),
        stream_sink: StreamEventSink | None = None,
        skill_catalog: SkillCatalog | None = None,
        skill_runtime: SkillRuntime | None = None,
        strict_model_provider: bool = True,
        latency_profile: AgentLatencyProfile | None = None,
        workspace: WorkspaceRuntime | None = None,
        configured_resident_tool_names: Sequence[str] = (),
        discoverable_tool_names: Sequence[str] = (),
        turn_store: TurnStore | None = None,
        runtime_binding: RuntimeBinding | None = None,
    ) -> None:
        self._policy = definition
        self._tool_registry = tool_registry
        self._tool_snapshot = tool_registry.freeze()
        self._tool_executor = ToolExecutor(self._tool_snapshot)
        self._configured_resident_tool_names = tuple(configured_resident_tool_names)
        self._discoverable_tool_names = tuple(discoverable_tool_names)
        self._skill_runtime = skill_runtime or (None if skill_catalog is None else SkillRuntime(skill_catalog))
        self._model_turn_provider = model_turn_provider
        self._model_registry = model_registry
        self._strict_model_provider = strict_model_provider
        self._output_finalizer = output_finalizer
        self._checkpointer = checkpointer or create_agent_checkpointer(None)
        self._runtime_diagnostics = tuple(merge_runtime_diagnostics([], runtime_diagnostics))
        self._stream_sink = stream_sink
        self._latency_profile = latency_profile or AgentLatencyProfile()
        self._workspace = workspace
        self._workspace_by_turn: dict[str, WorkspaceRuntime] = {}
        self._owns_turn_store = turn_store is None
        self._turn_store = turn_store or TurnStore()
        self._runtime_binding = runtime_binding or RuntimeBinding(
            workspace_path=(None if workspace is None else str(workspace.root)),
        )
        self._lease_owner = str(uuid4())

    async def aclose(self) -> None:
        await aclose_agent_checkpointer(self._checkpointer)
        if self._owns_turn_store:
            self._turn_store.close()

    def initial_state(self, request: AgentRunRequest) -> LoopState:
        return self._initial_state(
            request,
            run_config=request.to_run_config(self._policy),
        )

    async def run(self, request: AgentRunRequest) -> AgentRunResult:
        """Execute one Turn, optionally continuing a previous Turn."""

        async with self._open_turn(request) as effective_request:
            return await self._run_request(
                effective_request,
                streaming=False,
            )

    async def run_streaming(
        self,
        request: AgentRunRequest,
    ) -> AsyncGenerator[StreamEvent, None]:
        stream = self._stream_turn(request)
        async with aclosing(stream) as events:
            async for event in events:
                yield event

    async def _stream_turn(
        self,
        request: AgentRunRequest,
    ) -> AsyncGenerator[StreamEvent, None]:
        async with self._open_turn(request) as effective_request:
            turn_id = effective_request.turn_id
            if turn_id is None:
                raise RuntimeError("Turn allocation did not produce a turn_id")
            async for event in self._execute_streaming(effective_request):
                yield event

    @asynccontextmanager
    async def _open_turn(
        self,
        request: AgentRunRequest,
    ) -> AsyncIterator[AgentRunRequest]:
        turn_id = request.turn_id or str(uuid4())
        runtime = self._runtime_for_turn(
            turn_id=turn_id,
            previous_turn_id=request.previous_turn_id,
            requested_workspace=request.workspace_path,
        )
        turn = self._turn_store.begin_turn(
            request.message,
            runtime,
            previous_turn_id=request.previous_turn_id,
            turn_id=turn_id,
            lease_owner=self._lease_owner,
            lease_seconds=_TURN_LEASE_SECONDS,
        )
        try:
            history = self._turn_store.history_before_turn(turn.turn_id)
            if history and history[0].role != "user":
                raise RuntimeError(
                    f"History for Turn {turn.turn_id} does not begin with a user message"
                )
            effective_request = request.model_copy(
                update={
                    "turn_id": turn.turn_id,
                    "workspace_path": runtime.workspace_path,
                    "conversation_history": list(history),
                }
            )
            yield effective_request
        except (asyncio.CancelledError, GeneratorExit):
            self._interrupt_turn(turn.turn_id)
            raise
        except Exception:
            self._fail_turn(turn.turn_id)
            raise

    def _runtime_for_turn(
        self,
        *,
        turn_id: str,
        previous_turn_id: str | None,
        requested_workspace: str | None,
    ) -> RuntimeBinding:
        if previous_turn_id is not None:
            return self._turn_store.get_turn(previous_turn_id).runtime
        if self._runtime_binding.workspace_path is not None:
            return self._runtime_binding
        if requested_workspace is not None:
            workspace = open_workspace(requested_workspace, create=True)
        elif self._workspace is not None:
            workspace = self._workspace
        elif self._turn_store.path is not None:
            database = self._turn_store.path.expanduser().resolve()
            workspace = open_workspace(
                database.parent / ".agent-workspaces" / database.stem / turn_id,
                create=True,
            )
        else:
            workspace = create_temp_workspace()
        return self._runtime_binding.model_copy(
            update={"workspace_path": str(workspace.root)},
        )

    async def _execute_streaming(
        self,
        request: AgentRunRequest,
    ) -> AsyncGenerator[StreamEvent, None]:
        run_config = request.to_run_config(self._policy)
        lease_task = self._start_lease_heartbeat(run_config)
        state: LoopState | None = None
        workspace: WorkspaceRuntime | None = None
        try:
            (
                state,
                loop,
                workspace,
                started_at,
                checkpoint_store,
                tool_trace_start,
            ) = self._prepare_execution(
                request,
                run_config=run_config,
            )
            async for event in loop.run_streaming(state):
                yield event
        except (asyncio.CancelledError, GeneratorExit):
            TurnRegistry.remove(run_config.turn_id)
            raise
        except Exception:
            TurnRegistry.remove(run_config.turn_id)
            raise
        finally:
            try:
                if state is not None:
                    self._finalize_state(
                        state,
                        started_at=started_at,
                        tool_trace_start=tool_trace_start,
                    )
                    if state["status"] == "paused":
                        await checkpoint_store.save_snapshot(
                            state,
                            reason="service_pause_finalized",
                        )
                    if state["status"] in {"completed", "failed"}:
                        TurnRegistry.remove(run_config.turn_id)
                    if state["status"] != "running":
                        self._sync_turn(state)
                    if workspace is not None:
                        self._workspace_by_turn[run_config.turn_id] = workspace
            finally:
                await self._stop_lease_heartbeat(lease_task)

    async def _run_request(
        self,
        request: AgentRunRequest,
        *,
        streaming: bool,
        run_config: AgentRunConfig | None = None,
    ) -> AgentRunResult:
        del streaming
        effective_config = run_config or request.to_run_config(self._policy)
        lease_task = self._start_lease_heartbeat(effective_config)
        try:
            return await self._run_request_with_config(
                request,
                run_config=effective_config,
            )
        finally:
            await self._stop_lease_heartbeat(lease_task)

    async def _run_request_with_config(
        self,
        request: AgentRunRequest,
        *,
        run_config: AgentRunConfig,
    ) -> AgentRunResult:
        try:
            (
                state,
                loop,
                workspace,
                started_at,
                checkpoint_store,
                tool_trace_start,
            ) = self._prepare_execution(
                request,
                run_config=run_config,
            )
            result_state = await loop.run(state)
        except asyncio.CancelledError:
            TurnRegistry.remove(run_config.turn_id)
            raise
        except Exception:
            TurnRegistry.remove(run_config.turn_id)
            raise
        self._finalize_state(
            result_state,
            started_at=started_at,
            tool_trace_start=tool_trace_start,
        )
        if result_state["status"] == "paused":
            await checkpoint_store.save_snapshot(
                result_state,
                reason="service_pause_finalized",
            )
        if result_state["status"] in {"completed", "failed"}:
            TurnRegistry.remove(run_config.turn_id)
        self._sync_turn(result_state)
        self._workspace_by_turn[run_config.turn_id] = workspace
        return AgentRunResult.from_loop_result(
            result_state,
            definition=self._policy,
            workspace_path=str(workspace.root),
        )

    def _sync_turn(self, state: LoopState) -> None:
        run_config = state["run_config"]
        self._turn_store.sync_turn_messages(
            run_config.turn_id,
            state["turn_transcript"],
            compaction_policy=run_config.memory_policy,
        )
        if state["status"] == "paused":
            self._turn_store.mark_paused(run_config.turn_id)
        elif state["status"] == "completed":
            self._turn_store.mark_terminal(
                run_config.turn_id,
                TurnStatus.COMPLETED,
            )
        elif state["status"] == "failed":
            self._turn_store.mark_terminal(
                run_config.turn_id,
                TurnStatus.FAILED,
            )

    def _sync_checkpoint_turn(self, state: LoopState) -> None:
        run_config = state["run_config"]
        self._turn_store.sync_turn_messages(
            run_config.turn_id,
            state["turn_transcript"],
            compaction_policy=run_config.memory_policy,
        )

    def _interrupt_turn(self, turn_id: str) -> None:
        turn = self._turn_store.get_turn(turn_id)
        if turn.status is TurnStatus.RUNNING:
            self._turn_store.mark_interrupted(turn_id)

    def _fail_turn(self, turn_id: str) -> None:
        turn = self._turn_store.get_turn(turn_id)
        if turn.status is TurnStatus.RUNNING:
            self._turn_store.mark_terminal(turn_id, TurnStatus.FAILED)

    def _start_lease_heartbeat(
        self,
        run_config: AgentRunConfig,
    ) -> asyncio.Task[None] | None:
        return self._start_turn_lease_heartbeat(run_config.turn_id)

    def _start_turn_lease_heartbeat(
        self,
        turn_id: str,
    ) -> asyncio.Task[None] | None:
        owner_task = asyncio.current_task()
        heartbeat = asyncio.create_task(
            self._renew_turn_lease(turn_id),
            name=f"turn-lease-heartbeat:{turn_id}",
        )

        def cancel_owner_on_failure(task: asyncio.Task[None]) -> None:
            if task.cancelled() or task.exception() is None:
                return
            if owner_task is not None and not owner_task.done():
                owner_task.cancel()

        heartbeat.add_done_callback(cancel_owner_on_failure)
        return heartbeat

    async def _renew_turn_lease(self, turn_id: str) -> None:
        while True:
            await asyncio.sleep(_TURN_LEASE_HEARTBEAT_SECONDS)
            try:
                self._turn_store.renew_lease(
                    turn_id,
                    lease_owner=self._lease_owner,
                    lease_seconds=_TURN_LEASE_SECONDS,
                )
            except TurnStateError:
                if self._turn_store.get_turn(turn_id).status is not TurnStatus.RUNNING:
                    return
                raise

    @staticmethod
    async def _stop_lease_heartbeat(
        heartbeat: asyncio.Task[None] | None,
    ) -> None:
        if heartbeat is None:
            return
        if not heartbeat.done():
            heartbeat.cancel()
        try:
            await heartbeat
        except asyncio.CancelledError:
            pass

    def _prepare_execution(
        self,
        request: AgentRunRequest,
        *,
        run_config: AgentRunConfig,
    ) -> tuple[
        LoopState,
        AgentLoop,
        WorkspaceRuntime,
        float,
        LangGraphCheckpointStore,
        int,
    ]:
        started_at = time.perf_counter()
        tool_trace_start = len(self._tool_executor.traces)
        goal_spec = request.goal_spec or GoalSpec(
            original_query=request.message,
        )
        workspace = self._workspace_for_request(request)
        imported_files: list[Path] = []
        if request.input_files:
            imported_files = import_files(
                workspace,
                [Path(item) for item in request.input_files],
                namespace=(
                    None
                    if workspace.is_temporary
                    else run_config.turn_id
                ),
            )
        file_manifest = build_file_manifest(
            workspace,
            files=imported_files,
        )
        memory_store = WorkspaceMemoryStore(
            workspace=workspace,
            policy=run_config.memory_policy,
        )
        state = self._initial_state(
            request,
            run_config=run_config,
            memory_store=memory_store,
            file_manifest=file_manifest,
        )
        checkpoint_store = LangGraphCheckpointStore(
            self._checkpointer,
            run_config=run_config,
            compatibility_config=GoalCompatibilityConfig(goal_spec=goal_spec),
            snapshot_sink=self._sync_checkpoint_turn,
        )
        loop = self._build_loop(
            state=state,
            checkpoint_store=checkpoint_store,
            memory_store=memory_store,
            goal_spec=goal_spec,
            workspace=workspace,
        )
        return (
            state,
            loop,
            workspace,
            started_at,
            checkpoint_store,
            tool_trace_start,
        )

    def _initial_state(
        self,
        request: AgentRunRequest,
        *,
        run_config: AgentRunConfig,
        memory_store: WorkspaceMemoryStore | None = None,
        file_manifest: FileManifest | None = None,
    ) -> LoopState:
        TurnRegistry.remove(run_config.turn_id)
        handles = TurnRegistry.get_or_create(run_config)
        if memory_store is not None:
            handles.memory_store = memory_store
        state = create_loop_state(
            current_message=request.message,
            run_config=run_config,
            conversation_history=(
                snapshot_model_message(message)
                for message in request.conversation_history
            ),
            turn_transcript=(
                ModelMessage(role="user", content=request.message),
            ),
            pending_tool_calls=request.pending_tool_calls,
            messages=request.messages,
            runtime_diagnostics=self._runtime_diagnostics,
            input_files=request.input_files,
            file_manifest=file_manifest,
        )
        allow_discovery_tools = (
            bool(self._discoverable_tool_names)
            if request.allow_discovery_tools is None
            else request.allow_discovery_tools
        )
        policy_disabled_tools = tuple(name for name in self._tool_snapshot if name in run_config.tool_policy.deny_tools)
        options = resolve_tool_options(
            self._tool_snapshot,
            default_resident_names=self._default_resident_names(),
            configured_resident_names=self._configured_resident_tool_names,
            discoverable_names=self._discoverable_tool_names,
            tools=request.tools,
            disabled_tools=(
                *request.disabled_tools,
                *policy_disabled_tools,
            ),
            allow_discovery_tools=allow_discovery_tools,
        )
        if options.uses_default_tools:
            state["resident_tool_names"] = list(options.resident_names)
            state["explicit_tool_names"] = []
        else:
            state["resident_tool_names"] = []
            state["explicit_tool_names"] = list(options.resident_names)
        state["disabled_tool_names"] = list(options.disabled_names)
        state["allow_write_tools"] = request.allow_write_tools
        state["allow_execute_tools"] = request.allow_execute_tools
        state["allow_discovery_tools"] = allow_discovery_tools
        state["approved_tool_call_ids"] = list(request.approved_tool_call_ids)
        state["denied_tool_call_ids"] = list(request.denied_tool_call_ids)
        selected_names = (
            *state["resident_tool_names"],
            *state["explicit_tool_names"],
        )
        selected = select_tools(
            self._tool_snapshot,
            resident_names=selected_names,
            disabled_names=state["disabled_tool_names"],
        )
        state["tool_manifest"] = build_tool_manifest(
            tools=selected,
            resident_tool_names=state["resident_tool_names"],
            explicit_tool_names=state["explicit_tool_names"],
            provider_serializer_revision=state["provider_serializer_revision"],
        )
        initial_message, context_transcript = split_turn_context(
            conversation_history=state["conversation_history"],
            turn_transcript=state["turn_transcript"],
        )
        context = build_stable_context(
            instructions=(self._policy.system_instructions or "You are a helpful agent.",),
            initial_user_task=initial_message,
            transcript=context_transcript,
        )
        state["context_revision"] = context.context_revision
        exposed_names = tuple(tool.definition.name for tool in selected)
        manifest = state["tool_manifest"]
        if manifest is None:
            raise RuntimeError("initial tool manifest was not built")
        origin = ToolCallOrigin(
            request_id=f"{run_config.turn_id}:initial",
            toolset_revision=manifest.toolset_revision,
            exposed_tool_names=exposed_names,
        )
        for pending in state["pending_tool_calls"]:
            effective_origin = pending.plan.origin or origin
            pending.plan.origin = effective_origin
            state["canonical_tool_calls"][pending.tool_call_id] = ToolCall(
                tool_call_id=pending.tool_call_id,
                tool_name=pending.tool_name,
                arguments=cast(
                    Mapping[str, JsonValue],
                    pending.plan.arguments,
                ),
                origin=effective_origin,
            )
        compacted = MessageCompactor(
            policy=run_config.memory_policy,
            store=memory_store,
        ).compact_initial_state(dict(state))
        result = cast(LoopState, compacted)
        result["latency_profile"] = self._latency_profile
        return result

    def _build_loop(
        self,
        *,
        state: LoopState,
        checkpoint_store: LangGraphCheckpointStore,
        memory_store: WorkspaceMemoryStore | None,
        goal_spec: GoalSpec | None,
        workspace: WorkspaceRuntime,
    ) -> AgentLoop:
        provider = ModelProviderResolver(
            model_turn_provider=self._model_turn_provider,
            model_registry=self._model_registry,
            policy=self._policy,
            registry_snapshot=self._tool_snapshot,
            goal_spec=goal_spec,
            strict_model_provider=self._strict_model_provider,
            stream_sink=self._stream_sink,
            skill_runtime=self._skill_runtime,
        ).resolve(state)
        output_finalizer = self._resolve_output_finalizer(state)
        tool_policy = state["run_config"].tool_policy
        execution_context = ToolExecutionContext(
            workspace_root=workspace.root,
            cwd=workspace.root,
            allow_write_tools=state["allow_write_tools"],
            allow_execute_tools=state["allow_execute_tools"],
            max_parallel_calls=tool_policy.max_parallel_calls,
            require_confirmation_for=tool_policy.require_confirmation_for,
            denied_tool_names=tool_policy.deny_tools,
            auto_approve_sandboxed=tool_policy.auto_approve_sandboxed,
        )
        return AgentLoop(
            definition=self._policy,
            model_provider=provider,
            context_manager=LoopContextCompactor(store=memory_store),
            tool_executor=self._tool_executor,
            registry_snapshot=self._tool_snapshot,
            execution_context=execution_context,
            checkpoint_store=checkpoint_store,
            stop_hook_runner=StopHookRunner(
                hooks=build_stop_hooks(
                    definition=self._policy,
                    output_finalizer=output_finalizer,
                    goal_spec=goal_spec,
                    workspace_root=workspace.root,
                ),
                max_blocks=self._policy.max_stop_hook_blocks,
            ),
            finish_candidate_builder=FinishCandidateBuilder(),
            stream_sink=self._stream_sink,
            skill_runtime=self._skill_runtime,
            discoverable_tool_names=self._discoverable_tool_names,
        )

    def _default_resident_names(self) -> tuple[str, ...]:
        installed = tuple(self._tool_snapshot)
        baseline = tuple(name for name in RESIDENT_CODING_TOOL_NAMES if name in installed)
        if baseline:
            return baseline
        return tuple(name for name in self._policy.configured_tool_names if name in installed)

    def _workspace_for_request(
        self,
        request: AgentRunRequest,
    ) -> WorkspaceRuntime:
        if self._workspace is not None:
            if request.workspace_path is None:
                return self._workspace
            requested = Path(request.workspace_path).expanduser().resolve()
            if requested == self._workspace.root.resolve():
                return self._workspace
        if request.workspace_path is not None:
            return open_workspace(request.workspace_path, create=True)
        return create_temp_workspace()

    def _resolve_output_finalizer(
        self,
        state: LoopState,
    ) -> StructuredOutputFinalizer | None:
        if self._output_finalizer is not None:
            return self._output_finalizer
        if self._policy.output_model is None or self._model_registry is None:
            return None
        try:
            return create_model_structured_output_finalizer(self._model_registry)
        except Exception as exc:
            state["runtime_diagnostics"].append(
                RuntimeDiagnostic.from_exception(
                    code="structured_output_finalizer_initialization_failed",
                    component="structured_output_finalizer",
                    error=exc,
                )
            )
            return None

    def _finalize_state(
        self,
        state: LoopState,
        *,
        started_at: float,
        tool_trace_start: int,
    ) -> None:
        phase_tool_latency = sum(trace.duration_ms for trace in self._tool_executor.traces[tool_trace_start:])
        profile = state.get("latency_profile") or AgentLatencyProfile()
        tool_latency = profile.tool_latency_ms + phase_tool_latency
        total_ms = profile.total_ms + (time.perf_counter() - started_at) * 1000
        if profile.total_ms == 0:
            total_ms += profile.startup_ms + profile.build_service_ms
        total_ms = max(
            total_ms,
            profile.startup_ms + profile.build_service_ms + profile.model_latency_ms + tool_latency,
        )
        state["latency_profile"] = profile.model_copy(
            update={
                "tool_latency_ms": tool_latency,
                "total_ms": total_ms,
            }
        )

    async def resume_turn(
        self,
        *,
        turn_id: str,
        action: str,
        user_input: str | None = None,
    ) -> AgentRunResult:
        turn = self._turn_store.prepare_turn_for_resume(turn_id)
        started_at = time.perf_counter()
        checkpoint_store = LangGraphCheckpointStore(
            self._checkpointer,
            run_config=self._checkpoint_lookup_config(turn_id),
            snapshot_sink=self._sync_checkpoint_turn,
        )
        restored = await checkpoint_store.load_for_resume()
        if restored is None:
            raise KeyError(f"No checkpoint found for turn_id={turn_id}")
        if restored["run_config"].turn_id != turn.turn_id:
            raise RuntimeError(f"Checkpoint identity does not match Turn {turn_id}")
        current_request = _resume_request(restored)
        resolving_reconciliation = current_request is not None and current_request.kind == "tool_reconciliation"
        if action != "abort" and not resolving_reconciliation:
            drift_result = await self._reconcile_manifest(
                restored,
                checkpoint_store=checkpoint_store,
            )
            if drift_result is not None:
                return drift_result
        response, abort = _response_for_resume_action(
            restored,
            action=action,
            user_input=user_input,
        )
        self._turn_store.claim_for_resume(
            turn_id,
            lease_owner=self._lease_owner,
            lease_seconds=_TURN_LEASE_SECONDS,
        )
        lease_task = self._start_turn_lease_heartbeat(turn_id)
        try:
            if response is not None:
                state = await checkpoint_store.apply_human_response(response)
            else:
                state = restored
                state["status"] = "running"
                state["pause"] = None
                state["approval_request"] = None
                state["approval_response"] = None
            self._hydrate_turn_state(state, turn_id=turn_id)
            if user_input is not None and not abort:
                message = ModelMessage(role="user", content=user_input)
                state["turn_transcript"] = [
                    *state["turn_transcript"],
                    message,
                ]
            if resolving_reconciliation and not abort:
                drift_result = await self._reconcile_manifest(
                    state,
                    checkpoint_store=checkpoint_store,
                )
                if drift_result is not None:
                    self._sync_turn(state)
                    return drift_result
            if abort:
                state["status"] = "failed"
                state["pause"] = None
                state["terminal"] = LoopTerminal(
                    status="failed",
                    stop_reason="user_aborted",
                    error="The user aborted this Turn.",
                )
                state["latest_transition"] = LoopTransition(
                    reason="failed",
                    iteration=state["iteration"],
                    detail={"stop_reason": "user_aborted"},
                )
                await checkpoint_store.save_snapshot(
                    state,
                    reason="user_aborted",
                )
                self._sync_turn(state)
                TurnRegistry.remove(turn_id)
                workspace = self._workspace or create_temp_workspace()
                return AgentRunResult.from_loop_result(
                    state,
                    definition=self._policy,
                    workspace_path=str(workspace.root),
                )
            await checkpoint_store.save_snapshot(
                state,
                reason="turn_resume_prepared",
            )
            workspace = self._workspace or create_temp_workspace()
            return await self._continue_resumed_state(
                state,
                checkpoint_store=checkpoint_store,
                workspace=workspace,
                started_at=started_at,
            )
        except BaseException:
            self._interrupt_turn(turn_id)
            TurnRegistry.remove(turn_id)
            raise
        finally:
            await self._stop_lease_heartbeat(lease_task)

    async def _continue_resumed_state(
        self,
        state: LoopState,
        *,
        checkpoint_store: LangGraphCheckpointStore,
        workspace: WorkspaceRuntime,
        started_at: float,
    ) -> AgentRunResult:
        run_config = state["run_config"]
        turn_id = run_config.turn_id
        TurnRegistry.get_or_create(run_config)
        memory_store = WorkspaceMemoryStore(
            workspace=workspace,
            policy=run_config.memory_policy,
        )
        TurnRegistry.get(turn_id).memory_store = memory_store
        loop = self._build_loop(
            state=state,
            checkpoint_store=checkpoint_store,
            memory_store=memory_store,
            goal_spec=checkpoint_store.compatibility_config.goal_spec,
            workspace=workspace,
        )
        tool_trace_start = len(self._tool_executor.traces)
        result_state = await loop.run(state)
        self._finalize_state(
            result_state,
            started_at=started_at,
            tool_trace_start=tool_trace_start,
        )
        if result_state["status"] == "paused":
            await checkpoint_store.save_snapshot(
                result_state,
                reason="service_pause_finalized",
            )
        if result_state["status"] in {"completed", "failed"}:
            TurnRegistry.remove(turn_id)
        self._sync_turn(result_state)
        self._workspace_by_turn[turn_id] = workspace
        return AgentRunResult.from_loop_result(
            result_state,
            definition=self._policy,
            workspace_path=str(workspace.root),
        )

    def _hydrate_turn_state(
        self,
        state: LoopState,
        *,
        turn_id: str,
    ) -> None:
        turn = self._turn_store.get_turn(turn_id)
        before = self._turn_store.history_before_turn(turn_id)
        persisted_turn = self._turn_store.turn_history(turn_id)
        checkpoint_turn = tuple(state.get("turn_transcript", ()))
        if checkpoint_turn[: len(persisted_turn)] == persisted_turn:
            current = checkpoint_turn
        elif persisted_turn[: len(checkpoint_turn)] == checkpoint_turn:
            current = persisted_turn
        else:
            try:
                self._turn_store.sync_turn_messages(
                    turn_id,
                    checkpoint_turn,
                    compaction_policy=(
                        state["run_config"].memory_policy
                    ),
                )
            except RuntimeError as exc:
                raise RuntimeError(
                    "Checkpoint and canonical history conflict for "
                    f"Turn {turn_id}"
                ) from exc
            current = checkpoint_turn
        initial = ModelMessage(role="user", content=turn.user_message)
        if not current:
            current = (initial,)
        if current[0] != initial:
            raise RuntimeError(f"Canonical history for Turn {turn_id} does not begin with its user message")
        if before and before[0].role != "user":
            raise RuntimeError(f"History for Turn {turn_id} does not begin with a user message")
        state["current_message"] = turn.user_message
        state["conversation_history"] = list(before)
        state["turn_transcript"] = list(current)

    async def _reconcile_manifest(
        self,
        state: LoopState,
        *,
        checkpoint_store: LangGraphCheckpointStore,
    ) -> AgentRunResult | None:
        persisted = state.get("tool_manifest")
        if persisted is None:
            return None
        resident = tuple(name for name in state.get("resident_tool_names", ()) if name in self._tool_snapshot)
        explicit = tuple(name for name in state.get("explicit_tool_names", ()) if name in self._tool_snapshot)
        active = tuple(name for name in state.get("active_tool_names", ()) if name in self._tool_snapshot)
        tools = tuple(self._tool_snapshot[name] for name in (*resident, *explicit, *active))
        rebuilt = build_tool_manifest(
            tools=tools,
            resident_tool_names=resident,
            explicit_tool_names=explicit,
            active_tool_names=active,
            provider_serializer_revision=state["provider_serializer_revision"],
        )
        calls = state.get("canonical_tool_calls", {})
        records = state["tool_execution_records"]
        denied_call_ids = set(state["denied_tool_call_ids"])
        dependent = tuple(
            calls[item.tool_call_id]
            for item in state["pending_tool_calls"]
            if item.tool_call_id in calls
            and item.tool_call_id not in denied_call_ids
            and (
                (record := records.get(item.tool_call_id)) is None
                or record.status
                not in {
                    ExecutionStatus.COMPLETED,
                    ExecutionStatus.FAILED,
                }
            )
        )
        decision = reconcile_tool_manifest(
            persisted=persisted,
            rebuilt=rebuilt,
            pending_tool_calls=dependent,
            paused_tool_calls=(),
        )
        if decision.status is ToolManifestDriftStatus.MATCH:
            return None
        if decision.status is ToolManifestDriftStatus.NEW_REVISION_REQUIRED:
            state["resident_tool_names"] = list(resident)
            state["explicit_tool_names"] = list(explicit)
            state["active_tool_names"] = list(decision.active_tool_names)
            state["tool_manifest"] = rebuilt
            await checkpoint_store.save_snapshot(
                state,
                reason="tool_manifest_revision",
            )
            return None
        existing_request = _resume_request(state)
        if existing_request is not None and existing_request.kind == "tool_reconciliation":
            return AgentRunResult.from_loop_result(
                state,
                definition=self._policy,
            )
        primary_call = decision.dependent_tool_calls[0]
        current_tool = self._tool_snapshot.get(primary_call.tool_name)
        if primary_call.tool_call_id not in records and current_tool is not None:
            records[primary_call.tool_call_id] = ToolExecutionRecord.prepare(
                primary_call,
                current_tool,
            )
        request = HumanInputRequest(
            request_id=f"hir_{uuid4().hex[:12]}",
            kind="tool_reconciliation",
            question=("A pending tool definition changed; reconcile it before execution."),
            context={
                "reason": decision.reason,
                "error_code": "tool_definition_changed",
                "tool_call_id": primary_call.tool_call_id,
                "tool_call_ids": [call.tool_call_id for call in decision.dependent_tool_calls],
            },
            options=["mark_failed"],
        )
        state["status"] = "paused"
        state["approval_request"] = request
        state["pause"] = LoopPause(
            reason="tool_definition_changed",
            request=request,
        )
        state["latest_transition"] = LoopTransition(
            reason="approval_required",
            iteration=state["iteration"],
            detail={"reason": "tool_definition_changed"},
        )
        await checkpoint_store.save_snapshot(
            state,
            reason="tool_definition_changed",
        )
        return AgentRunResult.from_loop_result(
            state,
            definition=self._policy,
        )

    def pending_human_input_request(
        self,
        *,
        turn_id: str,
    ) -> HumanInputRequest:
        state = LangGraphCheckpointStore(
            self._checkpointer,
            run_config=self._checkpoint_lookup_config(turn_id),
        ).load_latest_sync()
        return _pending_request(state, turn_id=turn_id)

    async def apending_human_input_request(
        self,
        *,
        turn_id: str,
    ) -> HumanInputRequest:
        checkpoint_store = LangGraphCheckpointStore(
            self._checkpointer,
            run_config=self._checkpoint_lookup_config(turn_id),
            snapshot_sink=self._sync_checkpoint_turn,
        )
        state = await checkpoint_store.load_for_resume()
        if state is not None:
            await self._reconcile_manifest(
                state,
                checkpoint_store=checkpoint_store,
            )
        return _pending_request(state, turn_id=turn_id)

    def _checkpoint_lookup_config(self, turn_id: str) -> AgentRunConfig:
        return AgentRunConfig(
            turn_id=turn_id,
            tool_policy=self._policy.tool_policy,
        )


def _response_for_resume_action(
    state: LoopState,
    *,
    action: str,
    user_input: str | None,
) -> tuple[HumanInputResponse | None, bool]:
    if action == "abort":
        return None, True
    request = _resume_request(state)
    if request is None:
        if action != "continue":
            raise ValueError(
                "An interrupted Turn without a human-input request only supports action='continue' or action='abort'"
            )
        return None, False
    if request.kind == "tool_approval":
        if action not in {"allow_once", "deny"}:
            raise ValueError("A tool approval Turn supports allow_once, deny, or abort")
        tool_call_ids = [item.approval_id or item.tool_call_id for item in request.tool_calls]
        return (
            HumanInputResponse(
                request_id=request.request_id,
                decision=cast(_ResumeDecision, action),
                approved_tool_call_ids=(tool_call_ids if action == "allow_once" else []),
                denied_tool_call_ids=(tool_call_ids if action == "deny" else []),
                user_message=user_input,
            ),
            False,
        )
    if request.kind == "tool_reconciliation":
        allowed_actions = tuple(
            option for option in request.options if option in {"mark_completed", "mark_failed"}
        ) or ("mark_completed", "mark_failed")
        if action not in allowed_actions:
            choices = ", ".join(allowed_actions)
            raise ValueError(f"A tool reconciliation Turn only supports {choices} or abort; replay is forbidden")
        return (
            HumanInputResponse(
                request_id=request.request_id,
                decision=cast(_ResumeDecision, action),
                user_message=user_input,
            ),
            False,
        )
    if action == "continue":
        if request.kind == "clarification" and (user_input is None or not user_input.strip()):
            raise ValueError("Clarification resume requires non-empty user_input")
        message = user_input
    elif action in request.options:
        message = user_input or action
    else:
        raise ValueError(f"Unsupported action {action!r} for {request.kind} request")
    return (
        HumanInputResponse(
            request_id=request.request_id,
            decision="continue",
            user_message=message,
        ),
        False,
    )


def _resume_request(state: LoopState) -> HumanInputRequest | None:
    request = state["approval_request"]
    if request is None and state["pause"] is not None:
        request = state["pause"].request
    return request


def _pending_request(
    state: LoopState | None,
    *,
    turn_id: str,
) -> HumanInputRequest:
    if state is None:
        raise KeyError(f"No checkpoint found for turn_id={turn_id}")
    request = state["approval_request"]
    if request is None and state["pause"] is not None:
        request = state["pause"].request
    if request is None:
        raise KeyError(f"No pending human input request for turn_id={turn_id}")
    return request


def _result_mapping(result: ToolResult) -> Mapping[str, object] | None:
    value = result.structured_content
    return value if isinstance(value, Mapping) else None


def _result_provenance(
    results: Sequence[ToolResult],
) -> tuple[list[EvidenceItem], list[AnswerCitation]]:
    evidence: list[EvidenceItem] = []
    citations: list[AnswerCitation] = []
    for result in results:
        if result.is_error:
            continue
        payload = _result_mapping(result)
        if payload is None:
            continue
        raw_evidence = payload.get("evidence", ()) or ()
        if isinstance(raw_evidence, Sequence) and not isinstance(
            raw_evidence,
            (str, bytes),
        ):
            for item in raw_evidence:
                evidence.append(EvidenceItem.model_validate(item))
        raw_citations = payload.get("citations", ()) or ()
        if isinstance(raw_citations, Sequence) and not isinstance(
            raw_citations,
            (str, bytes),
        ):
            for item in raw_citations:
                if not isinstance(item, (Mapping, AnswerCitation)):
                    continue
                citations.append(AnswerCitation.model_validate(item))
    return evidence, citations


def _result_flag(results: Sequence[ToolResult], name: str) -> bool:
    for result in reversed(results):
        payload = _result_mapping(result)
        if payload is not None and bool(payload.get(name, False)):
            return True
    return False


def _restore_final_output(
    raw_output: ValidatedFinalOutput | dict[str, object] | None,
    *,
    definition: AgentRuntimePolicy | None,
) -> BaseModel | None:
    if raw_output is None or definition is None or definition.output_model is None:
        return None
    envelope = ValidatedFinalOutput.model_validate(raw_output)
    expected_path = output_model_path(definition.output_model)
    if envelope.model_path != expected_path:
        raise ValueError("Checkpoint final output model does not match configured output model")
    return definition.output_model.model_validate(envelope.data)


__all__ = ["AgentRunRequest", "AgentRunResult", "AgentService"]
