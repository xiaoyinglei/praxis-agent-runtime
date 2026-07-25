from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from hashlib import sha256
from typing import TYPE_CHECKING, Any, Protocol, cast

from langchain_core.messages import BaseMessage
from pydantic import BaseModel

from rag.agent.core.messages import ModelMessage, tool_result_message
from rag.agent.core.model_request import (
    canonical_transcript_revision,
    project_transcript_compaction,
)
from rag.agent.loop.substate import _persistent_memory_index_digest
from rag.agent.memory.models import (
    ContextBudgetSnapshot,
    EvictedStateItem,
    ExtractedFact,
    MemoryBudgetSnapshot,
    MemoryPolicy,
    MemoryRef,
    MessageBatchPayload,
    StateChannelReplacement,
    WorkingMemoryDraft,
    WorkingSummary,
)
from rag.agent.tools.tool import ToolResult
from rag.utils.text import text_unit_count

if TYPE_CHECKING:
    from rag.agent.loop.state import LoopState


class ToolOutputMemoryStore(Protocol):
    def write_tool_output(
        self,
        payload: BaseModel,
        *,
        summary: str,
        source_tool_call_id: str | None = None,
        source_tool_name: str | None = None,
        warnings: list[str] | None = None,
    ) -> MemoryRef: ...


class WorkingMemoryCompactor:
    """Deterministically compact old messages into bounded working memory.

    This component does not infer semantic facts from free text. It only carries
    forward explicitly supplied working-memory facts from message metadata.
    """

    def __init__(
        self,
        *,
        tail_message_count: int = 8,
        max_summary_chars: int = 4000,
        max_context_tokens: int = 0,
    ) -> None:
        if tail_message_count < 0:
            raise ValueError("tail_message_count must be non-negative")
        if max_summary_chars <= 0:
            raise ValueError("max_summary_chars must be positive")
        self._tail_message_count = tail_message_count
        self._max_summary_chars = max_summary_chars
        self._max_context_tokens = max_context_tokens

    def compact(
        self,
        messages: Sequence[BaseMessage],
        *,
        now_iso: str | None = None,
    ) -> WorkingMemoryDraft:
        indexed_messages = list(messages)
        tail_start = self._tail_start_index(indexed_messages)
        covered = indexed_messages[:tail_start]
        tail = indexed_messages[tail_start:]
        working_summary = self._build_summary(covered, now_iso=now_iso) if covered else None
        facts = self._extract_explicit_facts(covered)
        context_budget = ContextBudgetSnapshot(
            max_context_tokens=self._max_context_tokens,
            working_memory_tokens=0 if working_summary is None else working_summary.token_count,
            message_tail_tokens=sum(text_unit_count(self._message_text(message)) for message in tail),
        )
        return WorkingMemoryDraft(
            working_summary=working_summary,
            extracted_facts=facts,
            tail_messages=tail,
            context_budget=context_budget,
        )

    def dehydrate(
        self,
        messages: Sequence[BaseMessage],
        *,
        now_iso: str | None = None,
    ) -> WorkingMemoryDraft:
        return self.compact(messages, now_iso=now_iso)

    def _tail_start_index(self, messages: list[BaseMessage]) -> int:
        if not messages:
            return 0
        if self._tail_message_count == 0:
            return len(messages)
        start = max(0, len(messages) - self._tail_message_count)
        return self._extend_tail_for_tool_pairs(messages, start)

    @staticmethod
    def _extend_tail_for_tool_pairs(messages: list[BaseMessage], start: int) -> int:
        required_tool_call_ids = {
            tool_call_id for message in messages[start:] if (tool_call_id := getattr(message, "tool_call_id", None))
        }
        if not required_tool_call_ids:
            return start
        earliest = start
        for index in range(start - 1, -1, -1):
            tool_calls = getattr(messages[index], "tool_calls", None) or []
            call_ids = {str(call.get("id")) for call in tool_calls if call.get("id")}
            if call_ids & required_tool_call_ids:
                earliest = index
                required_tool_call_ids -= call_ids
                if not required_tool_call_ids:
                    break
        return earliest

    def _build_summary(
        self,
        messages: list[BaseMessage],
        *,
        now_iso: str | None,
    ) -> WorkingSummary:
        lines = []
        for message in messages:
            message_id = self._message_id(message)
            role = getattr(message, "type", message.__class__.__name__)
            text = self._message_text(message).replace("\n", " ").strip()
            if text:
                lines.append(f"{message_id} [{role}]: {text}")
            else:
                lines.append(f"{message_id} [{role}]: <empty>")
        summary = self._truncate("\n".join(lines))
        return WorkingSummary(
            summary=summary,
            covered_message_ids=[self._message_id(message) for message in messages],
            updated_at=now_iso or datetime.now(UTC).isoformat(),
            token_count=text_unit_count(summary),
        )

    def _extract_explicit_facts(self, messages: list[BaseMessage]) -> list[ExtractedFact]:
        facts: list[ExtractedFact] = []
        seen: set[str] = set()
        for message in messages:
            message_id = self._message_id(message)
            for raw_fact in message.additional_kwargs.get("working_memory_facts", []):
                fact = self._coerce_fact(raw_fact, source_message_id=message_id)
                if fact.fact_id in seen:
                    continue
                seen.add(fact.fact_id)
                facts.append(fact)
        return facts

    def _coerce_fact(self, raw_fact: object, *, source_message_id: str) -> ExtractedFact:
        if isinstance(raw_fact, str):
            text = raw_fact.strip()
            return ExtractedFact(
                fact_id=self._fact_id(text),
                text=text,
                source_message_ids=[source_message_id],
            )
        if isinstance(raw_fact, dict):
            payload: dict[str, Any] = dict(raw_fact)
            existing_sources = list(payload.get("source_message_ids") or [])
            if source_message_id not in existing_sources:
                existing_sources.append(source_message_id)
            payload["source_message_ids"] = existing_sources
            if "fact_id" not in payload:
                payload["fact_id"] = self._fact_id(str(payload.get("text", "")))
            return ExtractedFact.model_validate(payload)
        raise ValueError(f"Unsupported working_memory_facts item: {type(raw_fact).__name__}")

    @staticmethod
    def _fact_id(text: str) -> str:
        return f"fact_{sha256(text.encode('utf-8')).hexdigest()[:16]}"

    def _truncate(self, text: str) -> str:
        if len(text) <= self._max_summary_chars:
            return text
        return text[: self._max_summary_chars].rstrip()

    @staticmethod
    def _message_id(message: BaseMessage) -> str:
        return message.id or f"message_{sha256(repr(message).encode('utf-8')).hexdigest()[:16]}"

    @staticmethod
    def _message_text(message: BaseMessage) -> str:
        content = message.content
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            return " ".join(str(part) for part in content)
        return str(content)


class MessageCompactor:
    """Externalize old messages and keep a bounded deterministic message tail."""

    def __init__(
        self,
        *,
        policy: MemoryPolicy,
        store: ToolOutputMemoryStore | None = None,
    ) -> None:
        self._policy = policy
        self._store = store

    def compact_initial_state(self, state: dict[str, Any]) -> dict[str, Any]:
        messages = [message for message in state.get("messages", []) if isinstance(message, BaseMessage)]
        if len(messages) < self._policy.message_compaction_min_count:
            return state

        draft = WorkingMemoryCompactor(
            tail_message_count=self._policy.max_message_tail_count,
            max_summary_chars=self._policy.max_working_summary_chars,
            max_context_tokens=0,
        ).compact(messages)
        covered_ids = set(draft.working_summary.covered_message_ids) if draft.working_summary is not None else set()
        covered_messages = [message for message in messages if self._message_id(message) in covered_ids]
        if not covered_messages:
            return state

        update: dict[str, Any] = {
            "messages": list(draft.tail_messages),
            "working_summary": self._merge_summary(
                state["memory_state"].working_summary,
                draft.working_summary,
            ),
            "extracted_facts": self._bounded_facts(
                [
                    *[fact for fact in state["memory_state"].extracted_facts if isinstance(fact, ExtractedFact)],
                    *draft.extracted_facts,
                ]
            ),
        }
        warnings: list[str] = []
        ref = self._write_message_batch(covered_messages, warnings=warnings)
        if ref is not None:
            update["memory_refs"] = [
                *[ref for ref in state["memory_state"].memory_refs if isinstance(ref, MemoryRef)],
                ref,
            ]
        if warnings:
            update["memory_warnings"] = [
                *[str(item) for item in state["memory_state"].memory_warnings],
                *warnings,
            ]
        # Dual-write to structured memory_state for checkpoint/restore.
        memory_state = state["memory_state"]
        update["memory_state"] = memory_state.model_copy(
            update={
                "working_summary": update.get(
                    "working_summary",
                    memory_state.working_summary,
                ),
                "extracted_facts": list(
                    update.get("extracted_facts", memory_state.extracted_facts)
                ),
                "memory_refs": list(
                    update.get("memory_refs", memory_state.memory_refs)
                ),
                "memory_warnings": list(
                    update.get("memory_warnings", memory_state.memory_warnings)
                ),
                "persistent": memory_state.persistent.model_copy(
                    update={
                        "index_digest": _persistent_memory_index_digest(
                            state.get("memory_index", "")
                        ),
                        "selected_count": len(
                            state.get("persistent_memories", [])
                        ),
                    }
                ),
            }
        )
        return {**state, **update}

    def compact_update(self, state: dict[str, Any], update: dict[str, Any]) -> dict[str, Any]:
        combined = {**state, **update}
        compacted = self.compact_initial_state(combined)
        if compacted is combined:
            return update
        result = dict(update)
        for key in (
            "working_summary",
            "extracted_facts",
            "memory_refs",
            "memory_warnings",
        ):
            if compacted.get(key) != combined.get(key):
                result[key] = compacted.get(key)
        if compacted.get("messages") != combined.get("messages"):
            result["messages"] = [StateChannelReplacement(items=list(compacted["messages"]))]
        return result

    def _write_message_batch(
        self,
        messages: list[BaseMessage],
        *,
        warnings: list[str],
    ) -> MemoryRef | None:
        summary = (
            f"message_batch messages={len(messages)} "
            f"first_id={self._message_id(messages[0])} last_id={self._message_id(messages[-1])}"
        )
        payload = MessageBatchPayload(messages=messages)
        payload_chars = len(payload.model_dump_json())
        if payload_chars > self._policy.max_message_batch_chars:
            warnings.append("message_batch_truncated_to_policy")
            payload = MessageBatchPayload(messages=self._bounded_message_batch(messages))
        if self._store is None:
            warnings.append("memory_unavailable")
            return None
        try:
            return self._store.write_tool_output(
                payload,
                summary=summary,
                source_tool_name="message_compaction",
            )
        except Exception:
            warnings.append("message_compaction_failed")
            return None

    def _bounded_message_batch(self, messages: list[BaseMessage]) -> list[BaseMessage]:
        bounded: list[BaseMessage] = []
        total = 0
        for message in reversed(messages):
            size = len(self._message_text(message))
            if bounded and total + size > self._policy.max_message_batch_chars:
                break
            bounded.append(message)
            total += size
        return list(reversed(bounded))

    def _merge_summary(
        self,
        existing: object,
        new_summary: WorkingSummary | None,
    ) -> WorkingSummary | None:
        if new_summary is None:
            return existing if isinstance(existing, WorkingSummary) else None
        if not isinstance(existing, WorkingSummary):
            return new_summary.model_copy(update={"summary": self._truncate_working_summary(new_summary.summary)})
        merged_text = "\n".join(text for text in (existing.summary, new_summary.summary) if text.strip())
        return WorkingSummary(
            summary=self._truncate_working_summary(merged_text),
            covered_message_ids=list(dict.fromkeys([*existing.covered_message_ids, *new_summary.covered_message_ids])),
            updated_at=new_summary.updated_at,
            token_count=text_unit_count(self._truncate_working_summary(merged_text)),
        )

    def _bounded_facts(self, facts: list[ExtractedFact]) -> list[ExtractedFact]:
        by_id = {fact.fact_id: fact for fact in facts}
        return list(by_id.values())[-self._policy.max_extracted_facts :]

    def _truncate_working_summary(self, summary: str) -> str:
        if len(summary) <= self._policy.max_working_summary_chars:
            return summary
        return summary[: self._policy.max_working_summary_chars].rstrip() + " [truncated]"

    @staticmethod
    def _message_id(message: BaseMessage) -> str:
        return message.id or f"message_{sha256(repr(message).encode('utf-8')).hexdigest()[:16]}"

    @staticmethod
    def _message_text(message: BaseMessage) -> str:
        content = message.content
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            return " ".join(str(part) for part in content)
        return str(content)


class MemoryCompactor:
    """Deterministically externalize large tool outputs and cap long state channels."""

    _CAPPED_CHANNELS: dict[str, str] = {
        "tool_results": "max_tool_results",
        "memory_refs": "max_memory_refs",
        "plan_events": "max_plan_events",
    }

    def __init__(
        self,
        *,
        policy: MemoryPolicy,
        store: ToolOutputMemoryStore | None = None,
        loop_mode: bool = False,
    ) -> None:
        self._policy = policy
        self._store = store
        self._loop_mode = loop_mode

    def compact_update(
        self,
        state: dict[str, Any],
        update: dict[str, Any],
    ) -> dict[str, Any]:
        compacted = dict(update)
        warnings: list[str] = []
        new_memory_refs: list[MemoryRef] = []
        evicted_items: list[EvictedStateItem] = []
        used_channel_counts: dict[str, int] = {}
        pinned_item_count = 0

        tool_results = self._combined_items(state, compacted, "tool_results")
        compacted_tool_results = list(tool_results)

        dropped: dict[str, int] = {}
        pin_context = (
            self._pin_loop_context(
                state,
                compacted,
                new_memory_refs=new_memory_refs,
            )
            if self._loop_mode
            else self._pin_context(
                state,
                compacted,
                new_memory_refs=new_memory_refs,
            )
        )
        for channel, limit_attr in self._CAPPED_CHANNELS.items():
            channel_changed = False
            combined: list[Any]
            if channel == "tool_results":
                combined = compacted_tool_results
            elif channel == "memory_refs" and new_memory_refs:
                combined = [
                    *[ref for ref in state["memory_state"].memory_refs if isinstance(ref, MemoryRef)],
                    *new_memory_refs,
                ]
                channel_changed = True
            else:
                combined = self._combined_items(state, compacted, channel)
            sanitized, sanitized_changed, sanitize_warnings = self._sanitize_channel_items(
                channel,
                combined,
            )
            if sanitized_changed:
                channel_changed = True
                warnings.extend(sanitize_warnings)
            limit = int(getattr(self._policy, limit_attr))
            bounded, channel_evictions, channel_pins = self._bounded_with_audit(
                channel,
                sanitized,
                limit=limit,
                pinned_keys=pin_context.get(channel, set()),
            )
            used_channel_counts[channel] = len(bounded)
            pinned_item_count += channel_pins
            if channel_evictions:
                channel_changed = True
                evicted_items.extend(channel_evictions)
                dropped[channel] = len(channel_evictions)
            if not channel_changed:
                continue
            if state.get(channel):
                compacted[channel] = [StateChannelReplacement(items=bounded)]
            else:
                compacted[channel] = bounded

        compacted["memory_budget"] = MemoryBudgetSnapshot(
            max_tool_output_chars=self._policy.max_tool_output_chars,
            externalized_record_count=0,
            unavailable_record_count=0,
            memory_ref_count=len(self._combined_items(state, compacted, "memory_refs")),
            compacted_tool_result_count=0,
            dropped_state_items=dropped,
            evicted_items=evicted_items,
            used_channel_counts=used_channel_counts,
            pinned_item_count=pinned_item_count,
            warnings=list(dict.fromkeys(warnings)),
        )
        if warnings:
            compacted["memory_warnings"] = list(dict.fromkeys(warnings))
        return compacted

    def summarize_tool_result(self, result: ToolResult) -> str:
        """Summarize without changing the model-visible result representation."""

        return self._one_line(
            f"{result.tool_name} {tool_result_message(result).content}"
        )

    def _bounded_with_audit(
        self,
        channel: str,
        items: list[Any],
        *,
        limit: int,
        pinned_keys: set[str],
    ) -> tuple[list[Any], list[EvictedStateItem], int]:
        if len(items) <= limit:
            pinned_count = sum(1 for item in items if self._is_pinned(channel, item, pinned_keys=pinned_keys))
            return list(items), [], pinned_count

        selected: list[Any] = []
        selected_keys: set[str] = set()
        pinned_count = 0
        for item in items:
            key = _item_key(item)
            if self._is_pinned(channel, item, pinned_keys=pinned_keys):
                selected.append(item)
                selected_keys.add(key)
                pinned_count += 1

        for item in reversed(items):
            key = _item_key(item)
            if key in selected_keys:
                continue
            if len(selected) >= limit:
                break
            selected.append(item)
            selected_keys.add(key)

        kept_keys_in_order = {_item_key(item) for item in selected}
        kept = [item for item in items if _item_key(item) in kept_keys_in_order]
        kept_keys = {_item_key(item) for item in kept}
        evicted = [
            self._evicted_item(channel, item, reason="retention_limit")
            for item in items
            if _item_key(item) not in kept_keys
        ]
        return kept, evicted, min(pinned_count, len(kept))

    def _is_pinned(self, channel: str, item: Any, *, pinned_keys: set[str]) -> bool:
        if _must_preserve(item):
            return True
        key = _item_key(item)
        if key in pinned_keys:
            return True
        if channel == "memory_refs" and isinstance(item, MemoryRef):
            return item.ref_id in pinned_keys
        return False

    def _evicted_item(self, channel: str, item: Any, *, reason: str) -> EvictedStateItem:
        return EvictedStateItem(
            channel=cast(Any, channel),
            key=_item_key(item),
            reason=reason,
            summary=self._eviction_summary(item),
            source_tool_call_id=_source_tool_call_id(item),
            memory_ref_id=_memory_ref_id(item),
        )

    def _sanitize_channel_items(
        self,
        channel: str,
        items: list[Any],
    ) -> tuple[list[Any], bool, list[str]]:
        sanitized: list[Any] = []
        changed = False
        warnings: list[str] = []
        for item in items:
            replacement = self._sanitize_state_item(channel, item)
            if replacement is not item:
                changed = True
                warnings.append("raw_checkpoint_guard_sanitized")
            sanitized.append(replacement)
        return sanitized, changed, list(dict.fromkeys(warnings))

    def _sanitize_state_item(self, channel: str, item: Any) -> Any:
        return item

    def _sanitized_text(self, value: object) -> object:
        if not isinstance(value, str):
            return value
        if len(value) <= self._policy.max_tool_output_chars:
            return value
        return self._metadata_summary(
            "raw_checkpoint_guard_sanitized",
            original_chars=len(value),
            sha256=sha256(value.encode("utf-8")).hexdigest()[:16],
        )

    def _pin_context(
        self,
        state: dict[str, Any],
        update: dict[str, Any],
        *,
        new_memory_refs: list[MemoryRef],
    ) -> dict[str, set[str]]:
        combined = {**state, **update}
        pins: dict[str, set[str]] = {channel: set() for channel in self._CAPPED_CHANNELS}
        plan = combined.get("agent_plan")
        active_step = _active_plan_step(plan)
        active_tool_call_ids = set(getattr(active_step, "tool_call_ids", []) or [])

        for tool_call_id in active_tool_call_ids:
            key = f"tool_call_id:{tool_call_id}"
            pins["tool_results"].add(key)
        for ref_id in self._referenced_memory_ref_ids(state, update, new_memory_refs):
            pins["memory_refs"].add(ref_id)
        return pins

    def _pin_loop_context(
        self,
        state: dict[str, Any],
        update: dict[str, Any],
        *,
        new_memory_refs: list[MemoryRef],
    ) -> dict[str, set[str]]:
        combined = {**state, **update}
        pins: dict[str, set[str]] = {channel: set() for channel in self._CAPPED_CHANNELS}
        active_tool_call_ids = {
            call.tool_call_id for call in combined.get("pending_tool_calls", []) if getattr(call, "tool_call_id", None)
        }
        approval_request = combined.get("approval_request")
        active_tool_call_ids.update(
            summary.tool_call_id
            for summary in getattr(approval_request, "tool_calls", []) or []
            if getattr(summary, "tool_call_id", None)
        )
        plan = combined.get("agent_plan")
        active_step = _active_plan_step(plan)
        active_tool_call_ids.update(
            str(tool_call_id) for tool_call_id in getattr(active_step, "tool_call_ids", []) or [] if tool_call_id
        )

        for tool_call_id in active_tool_call_ids:
            key = f"tool_call_id:{tool_call_id}"
            pins["tool_results"].add(key)

        for ref_id in self._referenced_memory_ref_ids(
            state,
            update,
            new_memory_refs,
        ):
            pins["memory_refs"].add(ref_id)
        return pins

    def _referenced_memory_ref_ids(
        self,
        state: dict[str, Any],
        update: dict[str, Any],
        new_memory_refs: list[MemoryRef],
    ) -> set[str]:
        del self, state, update
        return {ref.ref_id for ref in new_memory_refs}

    def _bounded(self, channel: str, items: list[Any]) -> list[Any]:
        limit = int(getattr(self._policy, self._CAPPED_CHANNELS[channel]))
        if len(items) <= limit:
            return list(items)
        return _bounded_recent(items, limit=limit)

    @staticmethod
    def _combined_items(
        state: dict[str, Any],
        update: dict[str, Any],
        channel: str,
    ) -> list[Any]:
        current = list(state.get(channel, []))
        incoming = list(update.get(channel, []))
        if len(incoming) == 1 and isinstance(incoming[0], StateChannelReplacement):
            return list(incoming[0].items)
        return [*current, *incoming]

    def _eviction_summary(self, item: Any) -> str | None:
        if isinstance(item, ToolResult):
            return self.summarize_tool_result(item)
        for attr in ("summary", "text", "value_preview", "unit_type", "tool_name"):
            value = getattr(item, attr, None)
            if isinstance(value, str) and value.strip():
                return self._one_line(value)[: self._policy.max_memory_summary_chars]
        if isinstance(item, MemoryRef):
            return self._one_line(item.summary)[: self._policy.max_memory_summary_chars]
        return self._one_line(str(type(item).__name__))

    @staticmethod
    def _metadata_summary(label: str, **values: object) -> str:
        parts = [label]
        for key, value in values.items():
            if value in (None, "", []):
                continue
            parts.append(f"{key}={MemoryCompactor._one_line(str(value))}")
        return " ".join(parts)

    @staticmethod
    def _list_preview(values: object, *, limit: int = 6) -> str:
        if not isinstance(values, list):
            return ""
        shown = [MemoryCompactor._one_line(str(value)) for value in values[:limit]]
        remaining = len(values) - limit
        suffix = f", ...(+{remaining})" if remaining > 0 else ""
        return "[" + ", ".join(shown) + suffix + "]"

    def _truncate_summary(self, summary: str) -> str:
        if len(summary) <= self._policy.max_memory_summary_chars:
            return summary
        return summary[: self._policy.max_memory_summary_chars].rstrip() + " [truncated]"

    @staticmethod
    def _one_line(text: str) -> str:
        return " ".join(text.split())


@dataclass(frozen=True, slots=True)
class LoopCompactionResult:
    changed: bool
    channels: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()


class LoopContextCompactor:
    """Prepare bounded loop state before a model invocation."""

    def __init__(
        self,
        *,
        store: ToolOutputMemoryStore | None = None,
    ) -> None:
        self._store = store

    @staticmethod
    def _sync_memory_state_from_flat_channels(
        state: LoopState,
        state_dict: dict[str, Any],
    ) -> None:
        memory_state = state["memory_state"]
        state["memory_state"] = memory_state.model_copy(
            update={
                "working_summary": state_dict.get(
                    "working_summary",
                    memory_state.working_summary,
                ),
                "extracted_facts": list(
                    state_dict.get(
                        "extracted_facts",
                        memory_state.extracted_facts,
                    )
                ),
                "memory_refs": list(
                    state_dict.get("memory_refs", memory_state.memory_refs)
                ),
                "memory_budget": state_dict.get(
                    "memory_budget",
                    memory_state.memory_budget,
                ),
                "memory_warnings": list(
                    state_dict.get(
                        "memory_warnings",
                        memory_state.memory_warnings,
                    )
                ),
                "persistent": memory_state.persistent.model_copy(
                    update={
                        "index_digest": _persistent_memory_index_digest(
                            state.get("memory_index", "")
                        ),
                        "selected_count": len(
                            state.get("persistent_memories", [])
                        ),
                    }
                ),
            }
        )

    def prepare(self, state: LoopState) -> LoopCompactionResult:
        from rag.agent.loop.state import (
            LoopTransition,
            replace_latest_transition,
        )

        state_dict = cast(dict[str, Any], state)
        policy = state["run_config"].memory_policy
        initial_warnings = list(state["memory_state"].memory_warnings)
        changed_channels: list[str] = []

        if self._compact_turn_transcript(
            state_dict,
            policy,
            retained_tail_count=policy.max_message_tail_count,
            force=False,
        ):
            changed_channels.append("turn_transcript")

        memory_update = MemoryCompactor(
            policy=policy,
            store=self._store,
            loop_mode=True,
        ).compact_update(state_dict, {})
        meaningful_memory_update = {key: value for key, value in memory_update.items() if key != "memory_budget"}
        if meaningful_memory_update:
            changed_channels.extend(meaningful_memory_update)
        self._apply_update(state_dict, memory_update)

        # Dual-write to structured memory_state for checkpoint/restore.
        self._sync_memory_state_from_flat_channels(state, state_dict)

        changed = bool(changed_channels)
        warnings = tuple(
            warning
            for warning in state["memory_state"].memory_warnings
            if warning not in initial_warnings
        )
        channels = tuple(dict.fromkeys(changed_channels))
        if changed:
            replace_latest_transition(
                state,
                LoopTransition(
                    reason="compaction",
                    iteration=state["iteration"],
                    detail={
                        "channels": list(channels),
                        "warnings": list(warnings),
                    },
                ),
            )
        return LoopCompactionResult(
            changed=changed,
            channels=channels,
            warnings=warnings,
        )

    def reactive_compact(self, state: LoopState) -> LoopCompactionResult:
        """Aggressively shrink loop state after a provider context overflow."""

        state_dict = cast(dict[str, Any], state)
        policy = state["run_config"].memory_policy
        initial_warnings = list(state["memory_state"].memory_warnings)
        changed_channels: list[str] = []

        if self._compact_turn_transcript(
            state_dict,
            policy,
            retained_tail_count=policy.reactive_compact_tail_count,
            force=True,
        ):
            changed_channels.append("turn_transcript")

        if changed_channels:
            self._append_memory_warnings(state_dict, ["reactive_compact"])
            changed_channels.append("memory_warnings")

        # Dual-write to structured memory_state for checkpoint/restore.
        self._sync_memory_state_from_flat_channels(state, state_dict)

        warnings = tuple(
            warning
            for warning in state["memory_state"].memory_warnings
            if warning not in initial_warnings
        )
        return LoopCompactionResult(
            changed=bool(changed_channels),
            channels=tuple(dict.fromkeys(changed_channels)),
            warnings=warnings,
        )

    @staticmethod
    def _compact_turn_transcript(
        state: dict[str, Any],
        policy: MemoryPolicy,
        *,
        retained_tail_count: int,
        force: bool = False,
    ) -> bool:
        messages = [
            message
            for message in state.get("turn_transcript", [])
            if isinstance(message, ModelMessage)
        ]
        if not messages:
            return False
        anchor_count = 1 if messages[0].role == "user" else 0
        anchor = messages[:anchor_count]
        body = tuple(messages[anchor_count:])
        if not body:
            return False
        if (
            not force
            and len(messages) < policy.message_compaction_min_count
        ):
            return False

        tail_start = max(0, len(body) - retained_tail_count)
        if force and tail_start == 0:
            tail_start = 1
        projected = project_transcript_compaction(
            body,
            parent_context_revision=canonical_transcript_revision(messages),
            tail_start=tail_start,
            max_summary_chars=policy.max_working_summary_chars,
        )
        if force and projected == body:
            projected = project_transcript_compaction(
                body,
                parent_context_revision=canonical_transcript_revision(
                    messages
                ),
                tail_start=len(body),
                max_summary_chars=policy.max_working_summary_chars,
            )
        if projected == body:
            return False
        state["turn_transcript"] = [*anchor, *projected]
        return True

    @staticmethod
    def _append_memory_warnings(state: dict[str, Any], warnings: list[str]) -> None:
        state["memory_state"].memory_warnings = list(
            dict.fromkeys(
                [
                    *[str(item) for item in state["memory_state"].memory_warnings],
                    *[warning for warning in warnings if warning],
                ]
            )
        )

    @staticmethod
    def _apply_update(
        state: dict[str, Any],
        update: dict[str, Any],
    ) -> None:
        for key, value in update.items():
            if isinstance(value, list) and len(value) == 1 and isinstance(value[0], StateChannelReplacement):
                state[key] = list(value[0].items)
                continue
            state[key] = value


def _bounded_recent(items: list[Any], *, limit: int) -> list[Any]:
    if len(items) <= limit:
        return list(items)
    required: list[Any] = []
    for item in items:
        if _must_preserve(item):
            required.append(item)
    selected: list[Any] = []
    for item in reversed(items):
        key = _item_key(item)
        if key in {_item_key(existing) for existing in selected}:
            continue
        selected.append(item)
        if len(selected) >= limit:
            break
    for item in reversed(required):
        key = _item_key(item)
        if key in {_item_key(existing) for existing in selected}:
            continue
        if len(selected) >= limit:
            selected.pop(0)
        selected.append(item)
    return list(reversed(selected[-limit:]))


def _must_preserve(item: Any) -> bool:
    warnings = getattr(item, "warnings", None)
    if isinstance(warnings, list) and warnings:
        return True
    status = getattr(item, "status", None)
    return status == "error"


def _item_key(item: Any) -> str:
    key = getattr(item, "key", None)
    if isinstance(key, str) and key:
        return key
    for attr in ("tool_call_id", "source_tool_call_id", "unit_id", "evidence_id", "citation_id"):
        value = getattr(item, attr, None)
        if value:
            return f"{attr}:{value}"
    return repr(item)


def _source_tool_call_id(item: Any) -> str | None:
    for attr in ("tool_call_id", "source_tool_call_id", "content_ref"):
        value = getattr(item, attr, None)
        if isinstance(value, str) and value:
            return value
    return None


def _memory_ref_id(item: Any) -> str | None:
    if isinstance(item, MemoryRef):
        return item.ref_id
    ref = getattr(item, "raw_memory_ref", None)
    return ref.ref_id if isinstance(ref, MemoryRef) else None


def _active_plan_step(plan: Any) -> Any | None:
    active_step_id = getattr(plan, "active_step_id", None)
    steps = getattr(plan, "steps", []) or []
    if active_step_id is not None:
        for step in steps:
            if getattr(step, "step_id", None) == active_step_id:
                return step
    for step in steps:
        if getattr(step, "status", None) in {"in_progress", "pending"}:
            return step
    return None


__all__ = [
    "LoopCompactionResult",
    "LoopContextCompactor",
    "MemoryCompactor",
    "MessageCompactor",
    "ToolOutputMemoryStore",
    "WorkingMemoryCompactor",
]
