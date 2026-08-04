from __future__ import annotations

import asyncio
import logging
import os
import time
from collections.abc import AsyncIterator, Sequence
from contextlib import aclosing, asynccontextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Protocol
from uuid import uuid4

from agent_runtime.knowledge import RAGKnowledgeConfig
from agent_runtime.models import ModelControlPlane, ModelSpec
from agent_runtime.result import AgentPause, AgentResult, _project_pause
from agent_runtime.streaming.events import StreamEvent

if TYPE_CHECKING:
    from langgraph.checkpoint.base import BaseCheckpointSaver

    from agent_runtime.core.runtime_diagnostics import RuntimeDiagnostic
    from agent_runtime.knowledge_providers.rag import LazyRAGKnowledgeProvider
    from agent_runtime.service import AgentRunRequest, AgentService
    from agent_runtime.tools.tool import Tool
    from agent_runtime.turns import RuntimeBinding, TurnStore

_RUNTIME_CLOSE_GRACE_SECONDS = 5.0
DEFAULT_CHECKPOINT_PATH = Path(".praxis/checkpoints.sqlite")
DEFAULT_MODEL_SESSION_PATH = Path(".praxis/model_session.json")
logger = logging.getLogger(__name__)


class AgentEventSink(Protocol):
    """Receive the same lifecycle events exposed by ``astream``."""

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
    ) -> None:
        if knowledge is not None and not isinstance(
            knowledge,
            RAGKnowledgeConfig,
        ):
            raise TypeError("knowledge must be RAGKnowledgeConfig or None")
        self.model = model
        self.checkpoint_db = checkpoint_db
        self.workspace_path = None if workspace_path is None else Path(workspace_path).expanduser().resolve()
        self.model_session_path = model_session_path
        self.knowledge = knowledge
        self._model_control_plane: ModelControlPlane | None = None
        self._turn_store: TurnStore | None = None
        self._checkpointer: BaseCheckpointSaver[str] | None = None

    def models(self) -> list[ModelSpec]:
        return self._get_model_control_plane().list_models()

    def current_model(self) -> ModelSpec:
        return self._get_model_control_plane().current_model()

    def switch_model(self, model_id: str) -> ModelSpec:
        return self._get_model_control_plane().switch_model(
            model_id,
            requested_by="user",
            persist=self.model_session_path is not None,
        )

    def run(
        self,
        task: str,
        *,
        previous_turn_id: str | None = None,
        files: Sequence[str] | None = None,
        max_turns: int | None = None,
        max_tokens_total: int | None = None,
        require_workspace_change: bool = True,
        allow_write_tools: bool = False,
        allow_execute_tools: bool = False,
        event_sink: AgentEventSink | None = None,
    ) -> AgentResult:
        return asyncio.run(
            self.arun(
                task,
                previous_turn_id=previous_turn_id,
                files=files,
                max_turns=max_turns,
                max_tokens_total=max_tokens_total,
                require_workspace_change=require_workspace_change,
                allow_write_tools=allow_write_tools,
                allow_execute_tools=allow_execute_tools,
                event_sink=event_sink,
            )
        )

    async def arun(
        self,
        task: str,
        *,
        previous_turn_id: str | None = None,
        files: Sequence[str] | None = None,
        max_turns: int | None = None,
        max_tokens_total: int | None = None,
        require_workspace_change: bool = True,
        allow_write_tools: bool = False,
        allow_execute_tools: bool = False,
        event_sink: AgentEventSink | None = None,
    ) -> AgentResult:
        runtime_agent = self if previous_turn_id is None else self._agent_for_previous_turn(previous_turn_id)
        request = runtime_agent._turn_request(
            task,
            previous_turn_id=previous_turn_id,
            files=files,
            max_turns=max_turns,
            max_tokens_total=max_tokens_total,
            require_workspace_change=require_workspace_change,
            allow_write_tools=allow_write_tools,
            allow_execute_tools=allow_execute_tools,
        )
        async with runtime_agent._open_product_runtime(
            stream_sink=event_sink,
        ) as service:
            internal_result = await service.run(request)
        return AgentResult._from_internal(
            internal_result,
            files=tuple(request.input_files),
        )

    def resume(
        self,
        turn_id: str,
        action: str,
        *,
        user_input: str | None = None,
        event_sink: AgentEventSink | None = None,
    ) -> AgentResult:
        return asyncio.run(
            self.aresume(
                turn_id,
                action,
                user_input=user_input,
                event_sink=event_sink,
            )
        )

    async def aresume(
        self,
        turn_id: str,
        action: str,
        *,
        user_input: str | None = None,
        event_sink: AgentEventSink | None = None,
    ) -> AgentResult:
        return await self._resume_turn(
            turn_id,
            action,
            user_input=user_input,
            stream_sink=event_sink,
        )

    async def _resume_turn(
        self,
        turn_id: str,
        action: str,
        *,
        user_input: str | None = None,
        stream_sink: AgentEventSink | None = None,
    ) -> AgentResult:
        runtime_agent = self._agent_for_turn(turn_id)
        async with runtime_agent._open_product_runtime(
            stream_sink=stream_sink,
        ) as service:
            internal_result = await service.resume_turn(
                turn_id=turn_id,
                action=action,
                user_input=user_input,
            )
            return AgentResult._from_internal(internal_result)

    def _turn_request(
        self,
        message: str,
        *,
        previous_turn_id: str | None,
        files: Sequence[str] | None,
        max_turns: int | None,
        max_tokens_total: int | None,
        require_workspace_change: bool,
        allow_write_tools: bool,
        allow_execute_tools: bool,
    ) -> AgentRunRequest:
        from agent_runtime.core.goal_contract import GoalConstraint, GoalSpec
        from agent_runtime.service import AgentRunRequest

        turn_id = str(uuid4())
        goal_constraints = [
            GoalConstraint(
                constraint_id="workspace_change",
                constraint_type="workspace_change",
                expected_value=True,
            )
        ]
        if require_workspace_change:
            goal_constraints.append(
                GoalConstraint(
                    constraint_id="verification_after_change",
                    constraint_type="verification_after_change",
                    expected_value=True,
                )
            )
        goal_spec = (
            GoalSpec(
                original_query=message,
                constraints=goal_constraints,
            )
            if require_workspace_change
            else None
        )
        return AgentRunRequest(
            message=message,
            previous_turn_id=previous_turn_id,
            turn_id=turn_id,
            max_turns=max_turns,
            llm_budget_total=max_tokens_total,
            input_files=list(files or ()),
            workspace_path=(None if self.workspace_path is None else str(self.workspace_path)),
            goal_spec=goal_spec,
            allow_write_tools=allow_write_tools,
            allow_execute_tools=allow_execute_tools,
        )

    def _agent_for_previous_turn(self, turn_id: str) -> Agent:
        turn = self._get_turn_store().get_turn(turn_id)
        return self._agent_for_binding(turn.runtime)

    def _agent_for_turn(self, turn_id: str) -> Agent:
        turn = self._get_turn_store().prepare_turn_for_resume(turn_id)
        return self._agent_for_binding(turn.runtime)

    def _agent_for_binding(self, binding: RuntimeBinding) -> Agent:
        restored = Agent(
            model=binding.model_alias,
            checkpoint_db=self.checkpoint_db,
            workspace_path=binding.workspace_path,
            knowledge=binding.knowledge,
        )
        restored._turn_store = self._get_turn_store()
        restored._checkpointer = self._get_checkpointer()
        return restored

    async def astream(
        self,
        task: str,
        *,
        previous_turn_id: str | None = None,
        files: Sequence[str] | None = None,
        max_turns: int | None = None,
        max_tokens_total: int | None = None,
        require_workspace_change: bool = True,
        allow_write_tools: bool = False,
        allow_execute_tools: bool = False,
    ) -> AsyncIterator[StreamEvent]:
        runtime_agent = self if previous_turn_id is None else self._agent_for_previous_turn(previous_turn_id)
        request = runtime_agent._turn_request(
            task,
            previous_turn_id=previous_turn_id,
            files=files,
            max_turns=max_turns,
            max_tokens_total=max_tokens_total,
            require_workspace_change=require_workspace_change,
            allow_write_tools=allow_write_tools,
            allow_execute_tools=allow_execute_tools,
        )
        async with runtime_agent._open_product_runtime() as service:
            stream = service.run_streaming(request)
            async with aclosing(stream) as events:
                async for event in events:
                    yield event

    def pending_input(self, turn_id: str) -> AgentPause | None:
        """Return the durable input request blocking a Turn, if one exists."""

        return asyncio.run(self.apending_input(turn_id))

    async def apending_input(
        self,
        turn_id: str,
    ) -> AgentPause | None:
        from agent_runtime.turns import TurnStatus

        turn = self._get_turn_store().get_turn(turn_id)
        if turn.status in {TurnStatus.COMPLETED, TurnStatus.FAILED}:
            return None
        runtime_agent = self._agent_for_binding(turn.runtime)
        async with runtime_agent._open_product_runtime() as service:
            try:
                request = await service.apending_human_input_request(turn_id=turn_id)
            except KeyError:
                return None
        return _project_pause(request)

    @asynccontextmanager
    async def _open_product_runtime(
        self,
        *,
        stream_sink: AgentEventSink | None = None,
    ) -> AsyncIterator[AgentService]:
        """Own one SDK call's resources and release them in reverse order."""

        from agent_runtime.runtime.mcp import (
            open_product_mcp_tools,
            resolve_product_mcp_config,
        )

        config_path = resolve_product_mcp_config(self.workspace_path)
        runtime_diagnostics: list[RuntimeDiagnostic] = []
        async with open_product_mcp_tools(
            config_path,
            diagnostics=runtime_diagnostics,
        ) as mcp_tools:
            service, provider = self._build_service(
                mcp_tools=mcp_tools,
                runtime_diagnostics=tuple(runtime_diagnostics),
                stream_sink=stream_sink,
            )
            try:
                yield service
            finally:
                try:
                    close_method = getattr(service, "aclose", None)
                    if callable(close_method):
                        try:
                            await asyncio.wait_for(
                                close_method(),
                                timeout=_RUNTIME_CLOSE_GRACE_SECONDS,
                            )
                        except TimeoutError:
                            logger.warning(
                                "Agent runtime close exceeded %.1fs grace period",
                                _RUNTIME_CLOSE_GRACE_SECONDS,
                            )
                finally:
                    if provider is not None:
                        await _close_owned_sync_resource(
                            provider,
                            label="knowledge provider",
                        )
                    if self._model_control_plane is not None:
                        await _close_owned_sync_resource(
                            self._model_control_plane,
                            label="model control plane",
                        )

    def _build_service(
        self,
        *,
        mcp_tools: tuple[Tool, ...] = (),
        runtime_diagnostics: Sequence[RuntimeDiagnostic] = (),
        stream_sink: AgentEventSink | None = None,
    ) -> tuple[AgentService, LazyRAGKnowledgeProvider | None]:
        from agent_runtime.runtime.builder import build_agent_service
        from agent_runtime.skills.catalog import SkillCatalog
        from agent_runtime.skills.loader import scan_and_load_skills
        from agent_runtime.skills.policy import SkillPolicy
        from agent_runtime.skills.runtime import SkillRuntime
        from agent_runtime.text import load_env_file
        from agent_runtime.tools.integrations.skills import create_skill_tools
        from agent_runtime.tools.integrations.subagent import (
            SubagentInput,
            create_subagent_tool,
        )
        from agent_runtime.workspace import open_workspace

        startup_started_at = time.perf_counter()
        load_env_file(".env" if self.workspace_path is None else self.workspace_path / ".env")
        try:
            model_control_plane = self._get_model_control_plane()
        except Exception:
            if self.model is not None:
                raise
            model_control_plane = None
        provider: LazyRAGKnowledgeProvider | None = None
        knowledge_runner = None
        if self.knowledge is not None:
            from agent_runtime.knowledge_providers.rag import LazyRAGKnowledgeProvider

            provider = LazyRAGKnowledgeProvider(
                config=self.knowledge,
                model_alias=self.model,
                vector_dsn=os.environ.get("AGENT_VECTOR_DSN"),
            )
            knowledge_runner = provider.search_knowledge

        workspace = None if self.workspace_path is None else open_workspace(self.workspace_path, create=True)
        skill_runtime = None
        skill_tools: tuple[Tool, ...] = ()
        subagent_tools: tuple[Tool, ...] = ()
        if workspace is not None:
            policy = SkillPolicy()
            manifests = [
                manifest
                for manifest in scan_and_load_skills(
                    workspace.root,
                    repo_root=workspace.root,
                )
                if policy.is_skill_enabled(manifest)
            ]
            catalog = SkillCatalog(manifests)
            candidate_runtime = SkillRuntime(catalog, policy=policy)
            if candidate_runtime.has_model_invocable_skills:
                skill_runtime = candidate_runtime
                skill_tools = create_skill_tools(
                    workspace,
                    invoke_skill=skill_runtime.invoke_skill,
                    active_skill_root=skill_runtime.skill_root,
                    invoke_execution_revision=skill_runtime.catalog_revision,
                )

            async def run_subagent(arguments: object) -> dict[str, object]:
                from agent_runtime.service import AgentRunRequest

                payload = SubagentInput.model_validate(arguments)
                child_task = payload.task
                if payload.context_summary:
                    child_task += "\n\nContext supplied by the parent agent:\n" + payload.context_summary
                child_service = build_agent_service(
                    workspace,
                    checkpoint_db=None,
                    model_alias=self.model,
                    model_control_plane=model_control_plane,
                    runtime_diagnostics=(),
                    knowledge_runner=knowledge_runner,
                    mcp_tools=mcp_tools,
                    skill_tools=skill_tools,
                    skill_runtime=skill_runtime,
                )
                try:
                    child = await child_service.run(
                        AgentRunRequest(
                            message=child_task,
                            max_turns=payload.max_turns,
                            llm_budget_total=payload.llm_budget_total,
                            workspace_path=str(workspace.root),
                        )
                    )
                finally:
                    close_child = getattr(child_service, "aclose", None)
                    if callable(close_child):
                        try:
                            await asyncio.wait_for(
                                close_child(),
                                timeout=_RUNTIME_CLOSE_GRACE_SECONDS,
                            )
                        except TimeoutError:
                            logger.warning(
                                "Subagent close exceeded %.1fs grace period",
                                _RUNTIME_CLOSE_GRACE_SECONDS,
                            )
                status = child.status if child.status in {"done", "failed", "paused"} else "failed"
                return {
                    "conclusion": child.final_answer or child.needs_user_input or "",
                    "key_facts": [item.text[:2000] for item in child.evidence[:10]],
                    "evidence_refs": [
                        {
                            "evidence_id": item.evidence_id,
                            "doc_id": item.doc_id,
                            "citation_anchor": item.citation_anchor,
                        }
                        for item in child.evidence[:20]
                    ],
                    "citations": [
                        {
                            "citation_id": item.citation_id,
                            "evidence_id": item.evidence_id,
                            "record_type": item.record_type,
                            "file_name": item.file_name,
                            "section_path": list(item.section_path),
                            "page_start": item.page_start,
                            "page_end": item.page_end,
                            "doc_id": item.doc_id,
                            "benchmark_doc_id": item.benchmark_doc_id,
                            "source_id": item.source_id,
                            "source_type": item.source_type,
                            "citation_anchor": item.citation_anchor or "",
                        }
                        for item in child.citations[:20]
                    ],
                    "status": status,
                    "child_turn_id": child.turn_id,
                    "stop_reason": child.stop_reason,
                }

            subagent_tools = (create_subagent_tool(run_subagent),)
        service = build_agent_service(
            workspace,
            checkpoint_db=self.checkpoint_db,
            checkpointer=self._get_checkpointer(),
            model_alias=self.model,
            model_control_plane=model_control_plane,
            runtime_diagnostics=runtime_diagnostics,
            knowledge_runner=knowledge_runner,
            mcp_tools=mcp_tools,
            skill_tools=skill_tools,
            subagent_tools=subagent_tools,
            skill_runtime=skill_runtime,
            stream_sink=stream_sink,
            startup_ms=(time.perf_counter() - startup_started_at) * 1000,
            turn_store=self._get_turn_store(),
            runtime_binding=self._runtime_binding(),
        )
        return service, provider

    def _get_turn_store(self) -> TurnStore:
        if self._turn_store is None:
            from agent_runtime.turns import TurnStore

            self._turn_store = TurnStore(self.checkpoint_db)
        return self._turn_store

    def _get_checkpointer(self) -> BaseCheckpointSaver[str]:
        if self._checkpointer is None:
            from agent_runtime.core.checkpointing import create_agent_checkpointer

            self._checkpointer = create_agent_checkpointer(self.checkpoint_db)
        return self._checkpointer

    def _runtime_binding(self) -> RuntimeBinding:
        from agent_runtime.turns import RuntimeBinding

        model_alias = self.model
        if self._model_control_plane is not None:
            model_alias = self._model_control_plane.current_model().id
        return RuntimeBinding(
            model_alias=model_alias,
            workspace_path=(None if self.workspace_path is None else str(self.workspace_path)),
            knowledge=self.knowledge,
        )

    def _get_model_control_plane(self) -> ModelControlPlane:
        if self._model_control_plane is None:
            self._model_control_plane = ModelControlPlane.from_env(
                initial_model_id=self.model,
                session_path=self.model_session_path,
            )
        return self._model_control_plane


async def _close_owned_sync_resource(
    resource: object,
    *,
    label: str,
) -> None:
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
        logger.warning(
            "%s close failed (%s)",
            label,
            type(exc).__name__[:120],
        )
