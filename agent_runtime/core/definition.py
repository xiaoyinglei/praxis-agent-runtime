from __future__ import annotations

from dataclasses import dataclass, field

from pydantic import BaseModel


@dataclass(frozen=True)
class ModelSelectionPolicy:
    """每个 Agent 节点的模型选择策略。None = 使用 ModelRegistry.default_model。"""

    tool_decision_model: str | None = None
    tool_decision_temperature: float = 0.0
    tool_decision_max_tokens: int | None = None


@dataclass(frozen=True)
class ToolPolicy:
    max_parallel_calls: int = 4
    require_confirmation_for: frozenset[str] = field(default_factory=frozenset)
    deny_tools: frozenset[str] = field(default_factory=frozenset)
    # Opt-in only. The permission choke point recognizes only the canonical
    # fail-closed run_command sandbox contract.
    auto_approve_sandboxed: bool = False

    def __post_init__(self) -> None:
        if type(self.max_parallel_calls) is not int:
            raise TypeError("max_parallel_calls must be an integer")
        if self.max_parallel_calls < 1:
            raise ValueError("max_parallel_calls must be positive")
        if type(self.auto_approve_sandboxed) is not bool:
            raise TypeError("auto_approve_sandboxed must be a bool")
        for field_name in ("require_confirmation_for", "deny_tools"):
            raw_names = getattr(self, field_name)
            if isinstance(raw_names, (str, bytes)):
                raise TypeError(f"{field_name} must be a collection of tool names")
            try:
                names = frozenset(raw_names)
            except TypeError as exc:
                raise TypeError(f"{field_name} must be a collection of tool names") from exc
            if any(not isinstance(name, str) or not name for name in names):
                raise ValueError(f"{field_name} must contain non-empty tool names")
            object.__setattr__(self, field_name, names)


@dataclass(frozen=True)
class AgentRuntimePolicy:
    """The single root runtime policy consumed by AgentLoop."""

    system_instructions: str
    core_tool_names: tuple[str, ...]
    deferred_tool_names: tuple[str, ...]
    max_iterations: int
    max_active_deferred_tools: int = 10
    model_selection: ModelSelectionPolicy = field(
        default_factory=ModelSelectionPolicy,
    )
    tool_policy: ToolPolicy = field(default_factory=ToolPolicy)
    output_model: type[BaseModel] | None = None
    output_validation_max_retries: int = 2
    max_stop_hook_blocks: int = 3

    @property
    def configured_tool_names(self) -> tuple[str, ...]:
        """Configured tools in their canonical prompt/runtime order."""
        return (*self.core_tool_names, *self.deferred_tool_names)

    def __post_init__(self) -> None:
        if self.output_validation_max_retries < 0:
            raise ValueError("output_validation_max_retries must be non-negative")
        if self.max_stop_hook_blocks < 1:
            raise ValueError("max_stop_hook_blocks must be positive")

    @classmethod
    def test_factory(
        cls,
        *,
        system_prompt: str = "",
        allowed_tools: list[str] | None = None,
        model_selection: ModelSelectionPolicy | None = None,
        output_model: type[BaseModel] | None = None,
        output_validation_max_retries: int = 2,
        max_stop_hook_blocks: int = 3,
        max_iterations: int = 10,
        tool_policy: ToolPolicy | None = None,
    ) -> AgentRuntimePolicy:
        """Convenience factory — flat tool list + sensible defaults for tests."""
        tools = allowed_tools or []

        return cls(
            system_instructions=system_prompt,
            core_tool_names=tuple(tools),
            deferred_tool_names=(),
            max_iterations=max_iterations,
            model_selection=model_selection or ModelSelectionPolicy(),
            tool_policy=tool_policy or ToolPolicy(),
            output_model=output_model,
            output_validation_max_retries=output_validation_max_retries,
            max_stop_hook_blocks=max_stop_hook_blocks,
        )
