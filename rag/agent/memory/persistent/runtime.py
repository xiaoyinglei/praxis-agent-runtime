"""Persistent memory lifecycle wiring for agent runs.

This module owns the LLM-backed persistent memory load/extract steps so
AgentService can stay focused on run orchestration.
"""

from __future__ import annotations

import logging
from typing import Any

from rag.agent.loop.state import LoopState
from rag.agent.loop.substate import (
    MemoryState,
    PersistentMemorySnapshot,
    _persistent_memory_index_digest,
)
from rag.agent.memory.persistent.consolidator import MemoryConsolidator
from rag.agent.memory.persistent.extractor import MemoryExtractor
from rag.agent.memory.persistent.selector import MemorySelector
from rag.agent.memory.persistent.store import PersistentMemoryStore

logger = logging.getLogger(__name__)


class PersistentMemoryRuntime:
    """Load, extract, and consolidate persistent workspace memories."""

    def __init__(self, *, model_registry: Any | None) -> None:
        self._model_registry = model_registry

    async def load(
        self,
        state: LoopState,
        store: PersistentMemoryStore,
        *,
        task: str,
    ) -> None:
        """Load selected persistent memories into loop state."""
        if not store.is_available:
            return

        try:
            index_content = store.read_index()
            state["memory_index"] = index_content

            if not index_content.strip():
                return

            memory_gateway = self._create_gateway("memory_select")
            if memory_gateway is None:
                selector = MemorySelector(max_selected=5, max_tokens=4000)
            else:
                selector = MemorySelector(
                    llm_gateway=memory_gateway,
                    max_selected=5,
                    max_tokens=4000,
                )

            selected = await selector.select(
                task=task,
                index_content=index_content,
                store=store,
            )
            state["persistent_memories"] = [m.to_markdown() for m in selected]
            self._write_snapshot(
                state,
                index_content=index_content,
                selected_markdown=[m.to_markdown() for m in selected],
            )
        except Exception:
            logger.warning("Failed to load persistent memories", exc_info=True)

    async def extract(
        self,
        state: LoopState,
        store: PersistentMemoryStore,
    ) -> None:
        """Extract durable memories from a completed run and consolidate."""
        if not store.is_available:
            return

        try:
            extract_gateway = self._create_gateway("memory_extract")
            if extract_gateway is None:
                return

            written = await MemoryExtractor(llm_gateway=extract_gateway).extract(
                state=state,
                store=store,
            )
            if written:
                logger.info("Extracted persistent memories: %s", written)

            consolidate_gateway = self._create_gateway("memory_consolidate")
            consolidator = MemoryConsolidator(
                llm_gateway=consolidate_gateway or extract_gateway,
            )
            result = await consolidator.consolidate(store)
            if result.action == "consolidated":
                logger.info(
                    "Consolidated memories: %d -> %d",
                    result.before_count,
                    result.after_count,
                )
        except Exception:
            logger.warning("Failed to extract persistent memories", exc_info=True)

    def _create_gateway(self, stage: str) -> Any | None:
        if self._model_registry is None:
            return None
        try:
            model_alias = self._resolve_model_alias(stage)
            resolved = self._model_registry.resolve_or_fallback(model_alias)
            if resolved.gateway is not None:
                return resolved.gateway
            if resolved.token_accounting is None:
                return None

            from rag.providers.llm_gateway import LLMGateway

            return LLMGateway(
                generator=resolved.generator,
                token_accounting=resolved.token_accounting,
                model_context_tokens=resolved.context_window_tokens,
            )
        except Exception:
            logger.debug(
                "Failed to create memory gateway for stage %s",
                stage,
                exc_info=True,
            )
            return None

    def _resolve_model_alias(self, stage: str) -> str:
        if self._model_registry is None:
            return ""
        try:
            task_config = getattr(
                self._model_registry.generation_config,
                stage,
                None,
            )
            if task_config is not None and task_config.model:
                return str(task_config.model)
        except Exception:
            logger.debug(
                "Failed to resolve memory model from runtime config",
                exc_info=True,
            )
        return str(self._model_registry.default_model)

    @staticmethod
    def _write_snapshot(
        state: LoopState,
        *,
        index_content: str,
        selected_markdown: list[str],
    ) -> None:
        snapshot = PersistentMemorySnapshot(
            index_digest=_persistent_memory_index_digest(index_content),
            selected_count=len(selected_markdown),
            selected_summaries=[text[:200] for text in selected_markdown],
        )
        memory_state = state.get("memory_state")
        if isinstance(memory_state, MemoryState):
            state["memory_state"] = memory_state.model_copy(
                update={"persistent": snapshot},
            )
        else:
            state["memory_state"] = MemoryState(persistent=snapshot)
__all__ = ["PersistentMemoryRuntime"]
