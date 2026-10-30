"""Host-owned execution facts. Model-visible text is a one-way rendering."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from agent_runtime.core.messages import ModelMessage, canonical_json_text
from agent_runtime.tools.tool import _thaw_json, json_schema_output


def _freeze(value: Mapping[str, Any]) -> Mapping[str, Any]:
    frozen = json_schema_output(None, dict(value))
    if not isinstance(frozen, Mapping):
        raise TypeError("execution facts must be a JSON object")
    return frozen


@dataclass(frozen=True, slots=True)
class ContextSnapshot:
    runtime_state: Mapping[str, Any] = field(default_factory=dict)
    context_baselines: Mapping[str, Any] = field(default_factory=dict)
    input_limits: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in ("runtime_state", "context_baselines", "input_limits"):
            object.__setattr__(self, name, _freeze(getattr(self, name)))

    @classmethod
    def from_messages(cls, messages: tuple[ModelMessage, ...]) -> ContextSnapshot:
        state: dict[str, Any] = {}
        baselines: dict[str, Any] = {}
        limits: Mapping[str, Any] = {}
        for message in messages:
            if message.role != "context":
                continue
            meta = message.runtime_metadata
            if meta.get("kind") == "runtime_state":
                state.update(meta["state"])
            elif meta.get("kind") == "context_baseline":
                baselines[meta["source"]] = meta["identity"]
            elif meta.get("kind") == "input_limits":
                limits = meta["limits"]
        return cls(runtime_state=state, context_baselines=baselines, input_limits=limits)


@dataclass(frozen=True, slots=True)
class StepControl:
    """Captured before preparation; request IDs and text carry no controls."""

    retry_count: int = 0
    recovery_root_id: str | None = None
    budget_status: Mapping[str, Any] = field(default_factory=dict)
    context_snapshot: ContextSnapshot = field(default_factory=ContextSnapshot)

    def __post_init__(self) -> None:
        if isinstance(self.retry_count, bool) or not isinstance(self.retry_count, int) or self.retry_count < 0:
            raise ValueError("retry_count must be a non-negative integer")
        object.__setattr__(self, "budget_status", _freeze(self.budget_status))
        if not isinstance(self.context_snapshot, ContextSnapshot):
            raise TypeError("context_snapshot must be a ContextSnapshot")

    def freeze(self) -> dict[str, Any]:
        return {
            "retry_count": self.retry_count,
            "recovery_root_id": self.recovery_root_id,
            "budget_status": _thaw_json(self.budget_status),
            "context_snapshot": {
                name: _thaw_json(getattr(self.context_snapshot, name))
                for name in ("runtime_state", "context_baselines", "input_limits")
            },
        }

    @classmethod
    def restore(cls, payload: Mapping[str, Any] | None) -> StepControl:
        if payload is None:
            return cls()
        return cls(
            retry_count=payload.get("retry_count", 0),
            recovery_root_id=payload.get("recovery_root_id"),
            budget_status=payload.get("budget_status", {}),
            context_snapshot=ContextSnapshot(**payload.get("context_snapshot", {})),
        )


def append_context_updates(
    messages: tuple[ModelMessage, ...], instructions: Mapping[str, Any] | None, memory: Mapping[str, Any] | None
) -> tuple[tuple[ModelMessage, ...], tuple[ModelMessage, ...]]:
    from agent_runtime.harness.context_rendering import memory_navigation
    from agent_runtime.project_instructions import instruction_context

    previous = ContextSnapshot.from_messages(messages).context_baselines
    updates = []
    for source, identity, meaningful, content in (
        (
            "project_instructions",
            instructions,
            bool(instructions and instructions["entries"]),
            None if instructions is None else instruction_context(instructions, replaces_previous=True),
        ),
        (
            "memory",
            memory,
            memory is not None,
            None if memory is None else memory_navigation(memory, include_version=False),
        ),
    ):
        # The rendered memory deliberately omits the storage version: unrelated
        # memory bookkeeping must not invalidate an unchanged provider prefix.
        if source == "memory" and identity is not None:
            identity = {key: value for key, value in identity.items() if key not in {"version", "scope_id"}}
        frozen = None if identity is None else _freeze(identity)
        if content is not None and previous.get(source) != frozen and (meaningful or source in previous):
            updates.append(
                ModelMessage(
                    role="context",
                    body=content,
                    runtime_metadata={"kind": "context_baseline", "source": source, "identity": frozen},
                )
            )
    return (*messages, *updates), tuple(updates)


def append_runtime_state(
    messages: tuple[ModelMessage, ...], state: Mapping[str, Any], *, acceptance: bool = False
) -> tuple[tuple[ModelMessage, ...], ModelMessage | None]:
    previous = ContextSnapshot.from_messages(messages).runtime_state
    current = _freeze(state)
    changed = {key: value for key, value in current.items() if key not in previous or previous[key] != value}
    if not changed:
        return messages, None
    text = "Runtime state update (changed fields; latest value wins):\n" + canonical_json_text(changed)
    if not previous:
        text += (
            "\nUse the latest clock for relative dates; clarify ambiguous dates. "
            "Keep the advisory plan current; it is not execution proof."
        )
        if "resource_reminder" not in state:
            text += " Under budget pressure or <=2 requests, converge and report unfinished work."
        if acceptance:
            text += (
                " Run acceptance_contract.acceptance_command exactly via run_command in the workspace "
                "root after the latest change, subject to normal permissions and approval."
            )
    update = ModelMessage(role="context", body=text, runtime_metadata={"kind": "runtime_state", "state": changed})
    return (*messages, update), update
