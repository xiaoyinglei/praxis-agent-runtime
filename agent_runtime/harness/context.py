"""Provider-neutral context projection from committed rollout Items."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator, Mapping
from dataclasses import asdict, replace
from typing import Any

from agent_runtime.harness.protocol import (
    ContextBudgetExceededError,
    ContextCompactionCandidate,
    ContextSourceChangedError,
    HarnessMessage,
    HarnessToolCall,
)
from agent_runtime.harness.rollout import ItemSnapshot, RolloutStore
from agent_runtime.modeling.contracts import ContextCapacityPolicy, context_capacity_policy


class RolloutContextManager:
    """Build model-visible messages without owning provider serialization."""

    def __init__(
        self,
        store: RolloutStore,
        *,
        max_item_bytes: int | None = None,
        max_total_bytes: int | None = None,
        max_messages: int | None = None,
    ) -> None:
        for name, value in (
            ("max_item_bytes", max_item_bytes),
            ("max_total_bytes", max_total_bytes),
            ("max_messages", max_messages),
        ):
            if value is not None and (isinstance(value, bool) or not isinstance(value, int) or value < 1):
                raise ValueError(f"{name} must be a positive integer")
        self._store = store
        self._max_item_bytes = max_item_bytes
        self._max_total_bytes = max_total_bytes
        self._max_messages = max_messages

    def capacity_policy(self, turn_id: str) -> ContextCapacityPolicy | None:
        return context_capacity_policy(self._store.read_turn(turn_id).binding_manifest)

    def limits(self, turn_id: str | None = None) -> dict[str, int]:
        policy = None if turn_id is None else self.capacity_policy(turn_id)
        defaults = {"max_item_bytes": 500_000, "max_total_bytes": 4_000_000, "max_messages": 2000}
        return {
            name: explicit if (explicit := getattr(self, "_" + name)) is not None
            else (getattr(policy, name) if policy is not None else value)
            for name, value in defaults.items()
        }

    def compact(
        self,
        *,
        turn_id: str,
        covered_item_ids: tuple[str, ...],
        summary: str,
        preserved_facts: Mapping[str, Any],
        artifact_refs: tuple[Mapping[str, Any], ...] = (),
        context_version: int,
    ) -> ItemSnapshot:
        """Persist a compaction Item; original Items remain canonical history."""

        return self._store.record_context_compaction(
            turn_id=turn_id,
            covered_item_ids=covered_item_ids,
            summary=summary,
            preserved_facts=preserved_facts,
            artifact_refs=artifact_refs,
            context_version=context_version,
        )

    def build(self, turn_id: str) -> tuple[HarnessMessage, ...]:
        return self._validate_budget(
            tuple(message for _, message in self._projected_messages(turn_id)), turn_id=turn_id
        )

    def current_plan(self, turn_id: str) -> ItemSnapshot | None:
        """Read canonical advisory state, including after compaction or restart."""
        return next((item for item in reversed(self._store.list_items(turn_id))
                     if item.kind == "plan_state" and item.status == "completed"), None)

    def _validate_budget(
        self, projected: tuple[HarnessMessage, ...], *, turn_id: str | None = None
    ) -> tuple[HarnessMessage, ...]:
        limits = self.limits(turn_id)
        messages: list[HarnessMessage] = []
        total_bytes = 0

        def append(message: HarnessMessage) -> None:
            nonlocal total_bytes
            item_bytes = _message_size_bytes(message)
            if item_bytes > limits["max_item_bytes"]:
                raise ContextBudgetExceededError(
                    "byte_guard: Model context single Item exceeds the configured byte limit "
                    f"({item_bytes} > {limits['max_item_bytes']}).", reason_code="byte_guard"
                )
            if len(messages) >= limits["max_messages"]:
                raise ContextBudgetExceededError(
                    "message_guard: Model context exceeds the configured message-count limit "
                    f"({len(messages) + 1} > {limits['max_messages']}).", reason_code="message_guard"
                )
            if total_bytes + item_bytes > limits["max_total_bytes"]:
                raise ContextBudgetExceededError(
                    "byte_guard: Model context exceeds the configured total byte limit "
                    f"({total_bytes + item_bytes} > {limits['max_total_bytes']}).", reason_code="byte_guard"
                )
            messages.append(message)
            total_bytes += item_bytes

        for message in projected:
            append(message)
        return tuple(messages)

    def compact_for_budget(
        self,
        *,
        turn_id: str,
        retained_tail_messages: int,
    ) -> ItemSnapshot:
        """Explicit local compaction; automatic Turns measure candidates before committing."""
        candidate = self._candidate(turn_id, retained_tail_messages, require_reduction=False)
        return self.commit_compaction(candidate)

    def compaction_candidates(
        self,
        turn_id: str,
        *,
        summary: str | None = None,
    ) -> Iterator[ContextCompactionCandidate]:
        """Project a supplied semantic summary, or try reversible tool elision.

        This synchronous method never invents a summary by slicing old messages.
        The Turn owns model I/O and supplies the durable semantic summary.
        """
        if summary is None:
            yield from self.cheap_candidates(turn_id)
            return
        seen: set[str] = set()
        failure: ContextBudgetExceededError | None = None
        for tail in (self.semantic_retained_tail(turn_id), 0):
            try:
                candidate = self._candidate(turn_id, tail, summary_override=summary)
            except ContextBudgetExceededError as exc:
                failure = exc
                continue
            if candidate.messages_json not in seen:
                seen.add(candidate.messages_json)
                yield candidate
        if not seen and failure is not None:
            raise failure

    def _candidate(
        self,
        turn_id: str,
        retained: int,
        *,
        require_reduction: bool = True,
        summary_override: str | None = None,
        validate: bool = True,
    ) -> ContextCompactionCandidate:
        if isinstance(retained, bool) or not isinstance(retained, int) or retained < 0:
            raise ValueError("retained_tail_messages must be a non-negative integer")
        revision = self._store.context_source_revision(turn_id)
        items = tuple(i for i in self._store.list_context_items(turn_id) if i.status == "completed")
        projected = _project_items(items)
        messages = tuple(m for _, m in projected)
        latest_user = next((item_id for item_id, m in reversed(projected) if m.role == "user"), None)
        if latest_user is None:
            raise ContextBudgetExceededError("Context has no protected user task.")
        uncertain_calls = {
            (source_turn, op.tool_call_id)
            for source_turn in {i.turn_id for i in items}
            for op in self._store.list_tool_operations(source_turn)
            if op.requires_reconciliation or op.status not in {"succeeded", "failed", "denied", "cancelled"}
        }
        item_turns = {i.item_id: i.turn_id for i in items}
        first_uncertain = next(
            (
                index
                for index, (item_id, message) in enumerate(projected)
                if any((item_turns[item_id], call.id) in uncertain_calls for call in message.tool_calls)
            ),
            len(messages),
        )
        cut = _tool_safe_boundary(messages, min(first_uncertain, max(0, len(messages) - retained)))
        semantic_cut = _tool_safe_boundary(messages, first_uncertain)
        item_kinds = {item.item_id: item.kind for item in items}
        for index in range(len(projected) - 1, -1, -1):
            if item_kinds.get(projected[index][0]) not in {"context_message", "completion_feedback"}:
                break
            cut = min(cut, index)
            semantic_cut = min(semantic_cut, index)
        if cut == 0:
            raise ContextBudgetExceededError(
                "No completed context is available for compaction.", reason_code="no_compactable_context"
            )
        raw_cut = (
            len(items)
            if cut == len(projected)
            else next(n for n, item in enumerate(items) if item.item_id == projected[cut][0])
        )
        # A user input may precede the results in the ledger while following
        # them in provider context. Cover the whole raw tool exchange, keeping
        # all crossed user Items protected below.
        raw_indices = {item.item_id: index for index, item in enumerate(items)}
        raw_cut = max(raw_cut, max((raw_indices[identity] + 1 for identity, _ in projected[:cut]), default=0))
        if raw_cut == 0:
            raise ContextBudgetExceededError(
                "No completed context is available for compaction.", reason_code="no_compactable_context"
            )
        semantic_tail = (summary_override is not None and self.capacity_policy(turn_id) is not None and retained > 0)
        if semantic_tail:
            # The source already summarizes this exchange. Preserve its raw
            # replay separately, so the next summary does not summarize it again.
            raw_cut = (len(items) if semantic_cut == len(projected)
                       else next(n for n, item in enumerate(items) if item.item_id == projected[semantic_cut][0]))
            raw_cut = max(raw_cut, max((raw_indices[identity] + 1
                                       for identity, _ in projected[:semantic_cut]), default=0))
        covered = items[:raw_cut]
        protected_users = {latest_user} | {
            item.item_id for item in items if item.turn_id == turn_id and item.kind == "user_message"
        }
        first_tail_role = next((m.role for _, m in projected[cut:] if m.role != "context"), None)
        if first_tail_role not in {None, "user"}:
            anchor = next((item_id for item_id, m in reversed(projected[:cut]) if m.role == "user"), None)
            if anchor is not None:
                protected_users.add(anchor)
        preserved = protected_users | (
            {item_id for item_id, _ in projected[cut:semantic_cut]} if semantic_tail else set()
        )
        protected = tuple(i.item_id for i in covered if i.item_id in preserved)
        facts_items = tuple(i for i in covered if i.item_id not in protected)
        facts = _runtime_preserved_facts(self._store, turn_id=turn_id, covered_items=facts_items)
        derived_facts = {category: list(entries) for category, entries in facts.items()}
        prior = [i for i in items if i.kind == "context_compaction"]
        historical_categories = (
            "architecture_and_safety_constraints",
            "file_changes",
            "verification_results",
            "inspectable_sources",
        )
        if summary_override is not None:
            # These facts are already present in the history being summarized.
            # Keep current runtime state exact, and make historical evidence
            # discoverable without promoting every old message to a pinned rule.
            for category in historical_categories:
                facts[category] = []
        for item in prior:
            for category in historical_categories:
                previous = item.payload.get("preserved_facts", {}).get(category, [])
                if summary_override is not None:
                    # Explicit caller-supplied facts have no equivalent in the
                    # canonical runtime-derived list, so retain them verbatim.
                    automatic = item.payload.get("algorithm_revision") in {
                        "rollout-compaction-v2",
                        "semantic-compaction-v3",
                        "semantic-compaction-v4",
                        "semantic-compaction-v5",
                        "semantic-compaction-v6",
                    }
                    previous = [
                        entry
                        for entry in previous
                        if (not automatic or entry not in derived_facts[category])
                        and not (isinstance(entry, dict) and entry.get("runtime_archive_reference") is True)
                    ]
                facts[category] = _unique_json(
                    [
                        *facts[category],
                        *previous,
                    ]
                )
        if summary_override is not None:
            for category in historical_categories:
                if derived_facts[category]:
                    facts[category].append(
                        {
                            "runtime_archive_reference": True,
                            "count": len(derived_facts[category]),
                            "lookup": "read_context(item_id='history', query=...) for original evidence",
                        }
                    )
        versions = [i.payload["context_version"] for i in items if i.kind == "context_compaction"]
        # Explicit offline compaction is a lossless regrouping, not summarization.
        # Automatic semantic compaction always supplies a model-generated summary.
        summary = (
            summary_override
            if summary_override is not None
            else _json(
                [
                    {"item_id": item_id, **asdict(message)}
                    for item_id, message in projected[:cut]
                    if item_id not in protected or message.role == "context"
                ]
            )
        )
        # Raw replay and semantic coverage are separate: references still cover
        # the retained exchange, so later summaries retain its exact coordinates.
        reference_items = covered if self.capacity_policy(turn_id) is not None else facts_items
        artifacts = [dict(i.payload) for i in reference_items if i.kind == "input_file"]
        artifacts.extend(ref for i in reference_items if (ref := _web_source_reference(i)) is not None)
        artifacts = _unique_json([*artifacts, *(ref for i in prior for ref in i.payload.get("artifact_refs", []))])
        if self.capacity_policy(turn_id) is not None:
            artifacts = [ref for ref in artifacts if ref.get("kind") != "web_acquired_ranges"]
            artifacts.extend(_web_acquired_ranges(artifacts))
        payload = {
            "covered_item_ids": [i.item_id for i in covered],
            "preserved_item_ids": list(protected),
            "summary": summary,
            "preserved_facts": facts,
            "artifact_refs": artifacts,
            "durable_state": self._store.context_durable_state(turn_id),
            "context_version": max(versions, default=0) + 1,
            "algorithm_revision": "rollout-compaction-v2" if summary_override is None else (
                "semantic-compaction-v6" if self.capacity_policy(turn_id) is not None else "semantic-compaction-v4"
            ),
        }
        synthetic = replace(items[-1], item_id="candidate", kind="context_compaction", payload=payload)
        result = tuple(m for _, m in _project_items((*items, synthetic)))
        _validate_tool_groups(result)
        if validate:
            self._validate_budget(result, turn_id=turn_id)
        if require_reduction and sum(map(_message_size_bytes, result)) >= sum(map(_message_size_bytes, messages)):
            raise ContextBudgetExceededError(
                "Compaction candidate does not reduce context size.", reason_code="candidate_not_smaller"
            )
        if self._store.context_source_revision(turn_id) != revision:
            raise ContextSourceChangedError("Context source changed during compaction planning.")
        return ContextCompactionCandidate(turn_id, revision, _json(payload), _json([asdict(m) for m in result]))

    def semantic_retained_tail(self, turn_id: str) -> int:
        """Keep the latest completed tool exchange directly usable after compaction."""
        messages = tuple(message for _, message in self._projected_messages(turn_id))
        if not messages or messages[-1].role != "tool":
            return 0
        cut = _tool_safe_boundary(messages, len(messages) - 1)
        tail = messages[cut:]
        # Let actual candidate measurement decide how much room the task and
        # summary need. A fixed half-window cap discards useful evidence too soon.
        if sum(map(_message_size_bytes, tail)) >= self.limits(turn_id)["max_total_bytes"]:
            return 0
        return len(tail)

    def semantic_floor(self, turn_id: str, *, retained_tail_messages: int = 0) -> ContextCompactionCandidate:
        """Measure the irreducible continuation before spending on a summary.

        Empty summary candidates are planning-only; the store rejects committing
        them. Local limits and the normal model adapter still apply.
        """
        return self._candidate(turn_id, retained_tail_messages, require_reduction=False, summary_override="")

    def summary_byte_allowance(self, floor: ContextCompactionCandidate) -> int:
        messages = floor.messages
        context_size = next(_message_size_bytes(m) for m in messages if m.role == "context")
        limits = self.limits(floor.turn_id)
        return min(limits["max_total_bytes"] - sum(map(_message_size_bytes, messages)),
                   limits["max_item_bytes"] - context_size)

    def semantic_source(self, turn_id: str, *, retained_tail_messages: int = 0) -> tuple[str, str]:
        """Freeze the current continuation plus new history, without truncation.

        Internal summary bookkeeping is excluded. Global revision CAS still guards
        the final commit; this fingerprint detects changes across summary calls.
        """
        if type(retained_tail_messages) is not int or retained_tail_messages < 0:
            raise ValueError("retained_tail_messages must be a non-negative integer")
        # Summarize the complete eligible history, including a recent exchange
        # that may also be retained verbatim. If the actual generated summary
        # needs its space, that redundant replay can be removed without dropping
        # unseen evidence or buying another summary of the same history.
        candidate = self._candidate(
            turn_id,
            0,
            require_reduction=False,
            summary_override="Pending semantic summary",
            validate=False,
        )
        payload = json.loads(candidate.payload_json)
        covered = set(payload["covered_item_ids"])
        protected = set(payload["preserved_item_ids"])
        history_positions = {item.item_id: index for index, item in enumerate(self._store.list_context_items(turn_id))}
        # The projection includes prior committed summaries. Replaying their raw
        # covered items AND compaction payloads recursively duplicates history.
        # Originals remain in the archive for targeted recall.
        sources = [
            {"history_index": history_positions[item_id], **asdict(message)}
            for item_id, message in self._projected_messages(turn_id)
            if item_id in covered and (item_id not in protected or message.role != "user")
        ]
        items = self._store.list_context_items(turn_id)
        previously_summarized = {
            item_id
            for item in items
            if item.kind == "context_compaction" and ("tool_result_overrides" not in item.payload
            or str(item.payload.get("algorithm_revision", "")).startswith("semantic-compaction-"))
            for item_id in item.payload.get("covered_item_ids", ())
        }
        if self.capacity_policy(turn_id) is not None:
            # Elision coverage proves archival, not semantic coverage. Recover
            # canonical evidence only after the most recent semantic summary.
            semantic = next((item for item in reversed(items) if item.kind == "context_compaction"
                             and str(item.payload.get("algorithm_revision", "")).startswith("semantic-compaction-")),
                            None)
            summarized = set() if semantic is None else set(semantic.payload["covered_item_ids"])
            sources = []
            if semantic is not None:
                sources.append({"history_index": history_positions[semantic.item_id],
                                **asdict(_compaction_message(semantic.payload))})
            for item in items:
                if (item.item_id not in covered or item.item_id in summarized or item.status != "completed"
                        or item.kind == "context_compaction"):
                    continue
                message = _item_message(item)
                if message is not None and (item.item_id not in protected or message.role != "user"):
                    sources.append({"history_index": history_positions[item.item_id], **asdict(message)})
        sources.extend(
            {
                "history_index": history_positions[item.item_id],
                "role": "context",
                "content": _json({"runtime_verification": {
                    key: item.payload[key] for key in ("verification_kind", "verified_resources", "verifier")
                    if key in item.payload
                }}),
            }
            for item in items
            if item.kind == "verification"
            and item.status == "completed"
            and item.item_id in covered
            and item.item_id not in previously_summarized
        )
        sources.sort(key=lambda entry: entry["history_index"])
        if not sources:
            raise ContextBudgetExceededError(
                "No eligible history beyond the protected user task.", reason_code="no_compactable_context"
            )
        source = _json(
            {
                "history": sources,
                "current_task": next(
                    m.content for _, m in reversed(self._projected_messages(turn_id)) if m.role == "user"
                ),
                "history_order": "Oldest to newest; later explicit revisions supersede earlier decisions.",
                "archive_lookup": "read_context(item_id='history', query=...) locates original evidence and its IDs.",
                "protected_facts": payload["preserved_facts"],
                "durable_state": payload["durable_state"],
            }
        )
        return hashlib.sha256(source.encode()).hexdigest(), source

    def semantic_candidate(
        self, turn_id: str, *, source_hash: str, summary: str, retained_tail_messages: int = 0
    ) -> ContextCompactionCandidate:
        revision = self._store.context_source_revision(turn_id)
        if self.semantic_source(turn_id, retained_tail_messages=retained_tail_messages)[0] != source_hash:
            raise ContextSourceChangedError("Context changed while generating its semantic summary.")
        candidate = self._candidate(turn_id, retained_tail_messages, summary_override=summary)
        if candidate.source_revision != revision:
            raise ContextSourceChangedError("Context changed while projecting its semantic summary.")
        payload = json.loads(candidate.payload_json)
        payload["algorithm_revision"] = (
            "semantic-compaction-v6" if self.capacity_policy(turn_id) is not None else "semantic-compaction-v4"
        )
        return replace(candidate, payload_json=_json(payload))

    def max_tool_result_bytes(self, turn_id: str) -> int:
        return max((len(m.content.encode("utf-8")) for _, m in self._projected_messages(turn_id)
                    if m.role == "tool"), default=0)

    def cheap_candidates(
        self, turn_id: str, *, include_recent: bool = False, max_result_bytes: int | None = None
    ) -> Iterator[ContextCompactionCandidate]:
        """Replace old result bodies without changing message roles or tool IDs.

        Only called under context pressure. Try increasingly large prefixes of
        eligible results, oldest first, and let the caller measure the actual
        provider request. Keep the most recent complete exchange intact unless
        the caller explicitly budgets result excerpts for batch admission.
        """
        if max_result_bytes is not None and (type(max_result_bytes) is not int or max_result_bytes < 0):
            raise ValueError("max_result_bytes must be a non-negative integer")
        revision = self._store.context_source_revision(turn_id)
        items = tuple(i for i in self._store.list_context_items(turn_id) if i.status == "completed")
        projected = _project_items(items)
        if not projected:
            return
        original = tuple(message for _, message in projected)
        last_tool = next((i for i in range(len(original) - 1, -1, -1) if original[i].role == "tool"), None)
        keep_from = len(original) if last_tool is None else _tool_safe_boundary(original, last_tool)
        eligible = projected if include_recent else projected[:keep_from]
        by_id = {item.item_id: item for item in items}
        unsettled = {
            op.result_item_id
            for source_turn in {item.turn_id for item in items}
            for op in self._store.list_tool_operations(source_turn)
            if op.requires_reconciliation or op.status not in {"succeeded", "failed", "denied", "cancelled"}
        }
        overrides: dict[str, str] = {}
        version = max((i.payload["context_version"] for i in items if i.kind == "context_compaction"), default=0) + 1
        for item_id, message in eligible:
            if message.role != "tool" or item_id in unsettled or by_id[item_id].payload.get("is_error"):
                continue
            canonical = _item_message(by_id[item_id])
            content = _web_result_excerpt(canonical.content if canonical is not None else message.content,
                                          max_result_bytes, item_id)
            if content is None:
                content = f"Tool output archived. Use read_context(item_id={item_id!r}) to recover it."
                if max_result_bytes is not None:
                    raw = message.content.encode("utf-8")
                    if len(raw) <= max_result_bytes:
                        continue
                    prefix = raw[:max_result_bytes].decode("utf-8", errors="ignore")
                    content = (
                        prefix + "\n[Tool result truncated for context capacity. "
                        "The text above is an incomplete prefix, "
                        f"not a summary. Original: read_context(item_id={item_id!r}).]"
                    )
            if len(content.encode()) >= len(message.content.encode()):
                continue
            overrides[item_id] = content
            result = tuple(
                replace(m, content=overrides[source_id]) if source_id in overrides else m for source_id, m in projected
            )
            try:
                self._validate_budget(result, turn_id=turn_id)
                _validate_tool_groups(result)
            except ContextBudgetExceededError:
                continue
            if sum(map(_message_size_bytes, result)) >= sum(map(_message_size_bytes, original)):
                continue
            if self._store.context_source_revision(turn_id) != revision:
                raise ContextSourceChangedError("Context changed while planning tool output elision.")
            payload: dict[str, Any] = {
                "covered_item_ids": [i.item_id for i in items],
                "preserved_item_ids": [],
                "summary": "Tool result bodies archived; message structure preserved.",
                "preserved_facts": {
                    name: []
                    for name in (
                        "architecture_and_safety_constraints",
                        "file_changes",
                        "verification_results",
                        "unresolved_work",
                        "uncertain_side_effects",
                    )
                },
                "artifact_refs": [],
                "durable_state": self._store.context_durable_state(turn_id),
                "context_version": version,
                "algorithm_revision": "tool-output-elision-v5",
                "tool_result_overrides": dict(overrides),
            }
            yield ContextCompactionCandidate(turn_id, revision, _json(payload), _json([asdict(m) for m in result]))

    def commit_compaction(self, candidate: ContextCompactionCandidate) -> ItemSnapshot:
        payload = json.loads(candidate.payload_json)
        return self._store.record_context_compaction(
            turn_id=candidate.turn_id,
            expected_source_revision=candidate.source_revision,
            covered_item_ids=tuple(payload["covered_item_ids"]),
            preserved_item_ids=tuple(payload["preserved_item_ids"]),
            summary=payload["summary"],
            preserved_facts=payload["preserved_facts"],
            artifact_refs=tuple(payload["artifact_refs"]),
            durable_state=payload["durable_state"],
            context_version=payload["context_version"],
            algorithm_revision=payload["algorithm_revision"],
            tool_result_overrides=payload.get("tool_result_overrides"),
            capacity_diagnostics=payload.get("capacity_diagnostics"),
        )

    def _projected_messages(self, turn_id: str) -> tuple[tuple[str, HarnessMessage], ...]:
        return _project_items(self._store.list_context_items(turn_id))


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _unique_json(values: list[Any]) -> list[Any]:
    return list({_json(value): value for value in values}.values())


def _project_items(items: tuple[ItemSnapshot, ...]) -> tuple[tuple[str, HarnessMessage], ...]:
    replacements, suppressed = _compaction_projection(items)
    projected = []
    for item in items:
        if item.status != "completed":
            continue
        if (replacement := replacements.get(item.item_id)) is not None:
            projected.append((item.item_id, replacement))
        if item.item_id not in suppressed and (message := _item_message(item)) is not None:
            projected.append((item.item_id, message))
    # Preserve receive order in the ledger; keep provider tool groups complete.
    ordered = []
    deferred = []
    pending: set[str] = set()
    for entry in projected:
        message = entry[1]
        if pending and message.role == "user":
            deferred.append(entry)
            continue
        ordered.append(entry)
        pending.update(call.id for call in message.tool_calls)
        if message.role == "tool":
            pending.discard(message.tool_call_id or "")
        if not pending:
            ordered.extend(deferred)
            deferred.clear()
    ordered.extend(deferred)
    return tuple(ordered)


def _tool_safe_boundary(messages: tuple[HarnessMessage, ...], cut: int) -> int:
    pending: dict[str, int] = {}
    safe = 0
    for index, message in enumerate(messages[:cut]):
        for call in message.tool_calls:
            pending[call.id] = index
        if message.role == "tool" and message.tool_call_id is not None:
            pending.pop(message.tool_call_id, None)
        if not pending:
            safe = index + 1
    return safe


def _validate_tool_groups(messages: tuple[HarnessMessage, ...]) -> None:
    pending: set[str] = set()
    for message in messages:
        if message.role == "tool":
            if message.tool_call_id not in pending:
                raise ContextBudgetExceededError("Context contains an orphan or duplicate tool result.")
            pending.remove(message.tool_call_id)
        else:
            if pending:
                raise ContextBudgetExceededError("Context splits an unfinished tool group.")
            ids = [c.id for c in message.tool_calls]
            if len(ids) != len(set(ids)):
                raise ContextBudgetExceededError("Context contains duplicate tool call IDs.")
            pending.update(ids)
    if pending:
        raise ContextBudgetExceededError("Context contains an unfinished tool group.")


def _item_message(item: ItemSnapshot) -> HarnessMessage | None:
    text = item.payload.get("text")
    if item.kind == "user_message" and isinstance(text, str):
        return HarnessMessage(role="user", content=text)
    if item.kind == "agent_message" and isinstance(text, str):
        return HarnessMessage(role="assistant", content=text)
    if item.kind == "model_response" and isinstance(text, str):
        if item.payload.get("response_status") == "incomplete":
            # Do not replay partial assistant text, reasoning, or a tool group
            # for which no execution/result exists. This also survives compaction.
            return HarnessMessage(role="context", content=(
                "A previous model step was incomplete (output limit or another provider stop). "
                "Its partial output was discarded; no tools from that step were executed. "
                "Generate a complete next step from the completed history. Keep it concise; "
                "do not repeat completed tool actions."
            ))
        calls = item.payload.get("tool_calls")
        if isinstance(calls, (list, tuple)) and calls:
            return HarnessMessage(
                role="assistant",
                content=text,
                reasoning_content=item.payload.get("reasoning_content"),
                tool_calls=tuple(_tool_call(call) for call in calls if isinstance(call, Mapping)),
            )
    if item.kind == "tool_result":
        model_content = item.payload.get("model_content")
        tool_call_id = item.payload.get("tool_call_id")
        if isinstance(model_content, str) and isinstance(tool_call_id, str):
            return HarnessMessage(
                role="tool",
                content=model_content,
                tool_call_id=tool_call_id,
            )
    if item.kind in {"completion_feedback", "context_message"} and isinstance(text, str):
        return HarnessMessage(role="context", content=text)
    if item.kind == "tool_continuation" and item.producer == "runtime":
        return HarnessMessage(role="context", content=(
            "The live command owner closed. Confirmed terminal outcome (the process is not reattached):\n"
            + _json(dict(item.payload))
        ))
    if item.kind == "input_file":
        if item.payload.get("storage") == "reference" and isinstance(item.payload.get("path"), str):
            return HarnessMessage(role="context", content=(
                f"Attached input file (read-only original, not copied): {_json(item.payload['path'])}. "
                f"SHA256 at attachment: {item.payload.get('sha256')}. "
                "This is a live local file reference: read its current contents with file tools; "
                "the original may change or be removed. Write derived outputs inside the workspace."
            ))
        workspace_path = item.payload.get("workspace_path")
        sha256 = item.payload.get("sha256")
        if isinstance(workspace_path, str) and isinstance(sha256, str):
            return HarnessMessage(
                role="context",
                content=(f"Attached input file available in the workspace: {workspace_path} (sha256={sha256})."),
            )
    return None


def _runtime_preserved_facts(
    store: RolloutStore,
    *,
    turn_id: str,
    covered_items: tuple[ItemSnapshot, ...],
) -> dict[str, Any]:
    covered_ids = {item.item_id for item in covered_items}
    turn_ids = {item.turn_id for item in covered_items}
    operations = tuple(
        operation for source_turn_id in sorted(turn_ids) for operation in store.list_tool_operations(source_turn_id)
    )
    operation_by_result = {
        operation.result_item_id: operation for operation in operations if operation.result_item_id is not None
    }
    constraints = [
        {
            "item_id": item.item_id,
            "kind": item.kind,
            "text": item.payload["text"],
        }
        for item in covered_items
        if item.kind in {"user_message", "context_message", "completion_feedback"}
        and isinstance(item.payload.get("text"), str)
    ]
    file_changes = [
        {
            "operation_id": operation.operation_id,
            "tool_name": operation.tool_name,
            "resources": [dict(resource) for resource in operation.resources],
        }
        for item in covered_items
        if (operation := operation_by_result.get(item.item_id)) is not None
        and "write_workspace" in operation.effects
        and operation.status == "succeeded"
    ]
    verification_results = [
        dict(item.payload) for item in covered_items if item.kind == "verification" and item.producer == "runtime"
    ]
    plan_items = [
        item
        for item in store.list_context_items(turn_id)
        if item.kind == "plan_state" and item.status == "completed" and isinstance(item.payload.get("plan"), Mapping)
    ]
    from agent_runtime.planning import model_visible_plan

    unresolved_work = [] if not plan_items else [model_visible_plan(plan_items[-1].payload["plan"])]
    uncertain_side_effects = [
        {
            "operation_id": operation.operation_id,
            "tool_name": operation.tool_name,
            "status": operation.status,
            "resources": [dict(resource) for resource in operation.resources],
        }
        for operation in operations
        if operation.status == "unknown" or operation.requires_reconciliation
    ]
    inspectable_sources = [
        {
            "operation_id": operation.operation_id,
            "tool_name": operation.tool_name,
            "resources": [dict(resource) for resource in operation.resources],
        }
        for operation in operations
        if operation.result_item_id in covered_ids
        and operation.status == "succeeded"
        and "read_workspace" in operation.effects
    ]
    return {
        "architecture_and_safety_constraints": constraints,
        "file_changes": file_changes,
        "verification_results": verification_results,
        "unresolved_work": unresolved_work,
        "uncertain_side_effects": uncertain_side_effects,
        "inspectable_sources": inspectable_sources,
    }


def _tool_call(payload: Mapping[str, object]) -> HarnessToolCall:
    call_id = payload.get("id")
    name = payload.get("name")
    arguments = payload.get("arguments")
    if not isinstance(call_id, str) or not isinstance(name, str) or not isinstance(arguments, Mapping):
        raise RuntimeError("committed model tool call is malformed")
    return HarnessToolCall(id=call_id, name=name, arguments=arguments)


def _compaction_projection(
    items: tuple[ItemSnapshot, ...],
) -> tuple[dict[str, HarnessMessage], set[str]]:
    replacements: dict[str, HarnessMessage] = {}
    suppressed: set[str] = set()
    visible_ids = {item.item_id for item in items}
    for item in items:
        if item.status != "completed" or item.kind != "context_compaction":
            continue
        covered = item.payload.get("covered_item_ids")
        if (
            not isinstance(covered, (list, tuple))
            or not covered
            or not all(isinstance(item_id, str) for item_id in covered)
            or any(item_id not in visible_ids for item_id in covered)
        ):
            raise RuntimeError("committed context compaction coverage is malformed")
        semantic_replay = item.payload.get("algorithm_revision") == "semantic-compaction-v5"
        if "tool_result_overrides" in item.payload and not semantic_replay:
            overrides = item.payload["tool_result_overrides"]
            if not isinstance(overrides, Mapping) or not overrides:
                raise RuntimeError("committed tool result overrides are malformed")
            originals = {i.item_id: i for i in items}
            for target, content in overrides.items():
                original = originals.get(target)
                if (
                    target not in covered
                    or original is None
                    or original.status != "completed"
                    or original.kind != "tool_result"
                    or not isinstance(content, str)
                    or not content
                ):
                    raise RuntimeError("committed tool result override target is malformed")
                message = _item_message(original)
                if message is None:
                    raise RuntimeError("committed tool result override has no model message")
                # A later elision must never revive an item already summarized.
                if target not in suppressed or target in replacements:
                    replacements[target] = replace(message, content=content)
                    suppressed.add(target)
            suppressed.add(item.item_id)
            continue
        for target in covered:
            replacements.pop(target, None)
        anchor = covered[0]
        if item.payload.get("algorithm_revision") in {
            "semantic-compaction-v4", "semantic-compaction-v5", "semantic-compaction-v6",
        }:
            protected = set(item.payload.get("preserved_item_ids", ()))
            anchor = next((original.item_id for original in items if original.item_id in protected
                           and original.kind == "model_response"), anchor)
            # Place execution memory at the last summarized message, so work
            # already done in this Turn follows the request that triggered it.
            anchor = next(
                (
                    original.item_id
                    for original in reversed(items)
                    if original.item_id in covered
                    and original.item_id not in protected
                    and (_item_message(original) is not None or original.kind == "context_compaction")
                ),
                anchor,
            )
        replacements[anchor] = _compaction_message(item.payload)
        suppressed.update(covered)
        suppressed.difference_update(item.payload.get("preserved_item_ids", ()))
        if semantic_replay:
            originals = {original.item_id: original for original in items}
            for target, content in item.payload.get("tool_result_overrides", {}).items():
                message = _item_message(originals[target])
                if message is None or message.role != "tool":
                    raise RuntimeError("semantic replay requires original tool results")
                replacements[target] = replace(message, content=content)
                suppressed.add(target)
        suppressed.add(item.item_id)
    return replacements, suppressed


def _compaction_message(payload: Mapping[str, Any]) -> HarnessMessage:
    projected = {
        "coverage": (
            "This memory replaces an earlier prefix of this conversation, not a sampled search result. "
            "Newer uncompressed messages follow it. The archive contains the same past events; pagination "
            "does not imply additional unseen decisions after the current conversation."
        ),
        "recall": (
            "Use read_context only for missing, ambiguous or conflicting details needed for the current task. "
            "item_id=history lists archived IDs; query searches an original item. "
            "Do not reread already established facts merely because they are summarized."
        ),
        "artifact_refs": payload.get("artifact_refs", []),
        "context_version": payload.get("context_version"),
        "durable_state": payload.get("durable_state", {}),
        "preserved_facts": payload.get("preserved_facts", {}),
        "summary": payload.get("summary"),
        "summary_authority": "Continuation memory. Runtime facts take precedence; summaries do not prove execution.",
    }
    try:
        content = json.dumps(
            projected,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise RuntimeError("committed context compaction payload is malformed") from exc
    return HarnessMessage(
        role="context", content=f"Context compaction:\n{content}", context_category="continuation_summary"
    )


def _message_size_bytes(message: HarnessMessage) -> int:
    payload = {
        "content": message.content,
        "role": message.role,
        "tool_call_id": message.tool_call_id,
        "tool_calls": [
            {
                "arguments": call.arguments,
                "id": call.id,
                "name": call.name,
            }
            for call in message.tool_calls
        ],
    }
    if message.reasoning_content is not None:
        payload["reasoning_content"] = message.reasoning_content
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    return len(encoded.encode("utf-8"))


def _web_result_excerpt(content: str, budget: int | None, item_id: str) -> str | None:
    """Shrink only a saved web body; keep its typed cursor and citation metadata intact.

    Metadata is a non-elidable floor, even for a zero body budget. Admission
    still measures the full envelope and fails closed if that floor cannot fit.
    Canonical original ToolResults and artifacts are never changed.
    """
    try:
        envelope = json.loads(content)
    except (ValueError, RecursionError):
        return None
    if not isinstance(envelope, dict):
        return None
    source = envelope.get("structured_content")
    if (not isinstance(source, dict) or not isinstance(source.get("source_id"), str)
            or not source["source_id"].startswith("artifact_") or not isinstance(source.get("content"), str)
            or not isinstance(source.get("start_line"), int) or "next_line" not in source):
        return None
    if budget is not None and len(content.encode()) <= budget:
        return content
    # Visibility may shrink; completed acquisition and its cursor never change.
    kept: list[str] = []
    used = 0
    for line in source["content"].splitlines():
        size = len(line.encode()) + bool(kept)
        if budget is None or used + size > budget:
            break
        kept.append(line)
        used += size
    omitted = len(kept) < len(source["content"].splitlines())
    projected = {**source, "content": "\n".join(kept), "links": [], "context_truncated": True}
    projected.pop("fetch_next_line", None)
    projected["visible_end_line"] = source["start_line"] + len(kept) - 1 if kept else None
    if omitted:
        projected["truncated"] = True
    return _json({**envelope, "structured_content": projected, "truncated": True,
                  "archive_lookup": f"read_context(item_id={item_id!r}) recovers the original tool message; "
                                    "web_fetch(source_id=structured_content.source_id, start_line=next_line) "
                                    "continues the saved webpage. item_id is not a source_id."})


def _web_source_reference(item: ItemSnapshot, projected_content: str | None = None) -> dict[str, Any] | None:
    if item.kind != "tool_result":
        return None
    source = item.payload.get("structured_content")
    if source is None:
        try:
            source = json.loads(item.payload.get("model_content", "")).get("structured_content")
        except (ValueError, AttributeError, TypeError):
            return None
    if not isinstance(source, Mapping):
        return None
    if isinstance(source.get('query'), str) and isinstance(source.get('provider'), str):
        return {'item_id': item.item_id, 'kind': 'search_leads',
                **{key: source[key] for key in ('query', 'provider', 'result_status', 'previous_query',
                                               'freshness_requested', 'freshness_verified', 'error_code',
                                               'error_message', 'connection_mode', 'failure_stage',
                                               'cache_hit', 'results_truncated')
                   if key in source},
                'is_error': bool(item.payload.get('is_error'))}
    if not isinstance(source.get("source_id"), str):
        return None
    if not source["source_id"].startswith("artifact_") or not source.get("url"):
        return None
    # References report canonical acquisition, never an older elision cursor.
    del projected_content
    content = source.get("content")
    start = source.get("start_line")
    lines = content.splitlines() if isinstance(content, str) else []
    acquired_end = start + len(lines) - 1 if type(start) is int and lines else None
    return {"item_id": item.item_id, "acquired_end_line": acquired_end, **{key: source[key] for key in (
        "source_id", "url", "title", "published_at", "published_at_source", "content_hash",
        "start_line", "next_line", "fetch_next_line", "context_truncated", "total_lines",
        "source_truncated", "truncated",
        "view", "line_basis", "section_id", "next_section", "next_link",
        "warning", "extraction_method", "connection_mode", "failure_stage", "render_mode",
        "render_diagnostics", "error_code", "error_message",
    ) if key in source}, 'is_error': bool(item.payload.get('is_error'))}


def _web_acquired_ranges(refs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Derive acquisition intervals from canonical evidence; gaps stay explicit."""
    grouped: dict[str, tuple[dict[str, Any], list[tuple[int, int]]]] = {}
    for ref in refs:
        start, end = ref.get("start_line"), ref.get("acquired_end_line")
        if not ref.get("source_id") or type(start) is not int or type(end) is not int or end < start:
            continue
        coordinates = {key: ref.get(key) for key in ("source_id", "content_hash", "view", "line_basis")}
        _, ranges = grouped.setdefault(_json(coordinates), (coordinates, []))
        ranges.append((start, end))
    result = []
    for coordinates, ranges in grouped.values():
        merged: list[list[int]] = []
        for start, end in sorted(ranges):
            if merged and start <= merged[-1][1] + 1:
                merged[-1][1] = max(merged[-1][1], end)
            else:
                merged.append([start, end])
        result.append({"kind": "web_acquired_ranges", **coordinates, "acquired_ranges": merged})
    return result
