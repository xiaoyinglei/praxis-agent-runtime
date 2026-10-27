from __future__ import annotations

import os
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, replace
from enum import StrEnum
from pathlib import Path
from typing import Any, Literal

from agent_runtime.approval import MODE_ORDER, ApprovalMode, CommandGrant, command_grant
from agent_runtime.permission_policy import ResolvedPermissions
from agent_runtime.tools.tool import (
    CancellationMode,
    JsonValue,
    ResolvedToolUse,
    Tool,
    ToolApprovalProfile,
    ToolCall,
    ToolEffect,
)


class UseToolDecision(StrEnum):
    """Pure permission outcome; human interaction is deliberately external."""

    ALLOW = "allow"
    ASK = "ask"
    DENY = "deny"


@dataclass(frozen=True, slots=True)
class CanUseToolResult:
    decision: UseToolDecision
    reason: str
    approval_scope: Literal["tool", "network"] = "tool"

    def __post_init__(self) -> None:
        if not isinstance(self.decision, UseToolDecision):
            raise TypeError("decision must be a UseToolDecision")
        if not isinstance(self.reason, str) or not self.reason:
            raise ValueError("reason must be a non-empty string")
        if self.approval_scope not in {"tool", "network"}:
            raise ValueError("approval_scope must be tool or network")


@dataclass(frozen=True, slots=True)
class ToolExecutionContext:
    """Runtime facts consumed by guards, permission, and external approval."""

    workspace_root: Path | str | None = None
    cwd: Path | str | None = None
    allow_write_tools: bool = False
    allow_execute_tools: bool = False
    allow_web_tools: bool = False
    approved_tool_call_ids: frozenset[str] = frozenset()
    denied_tool_call_ids: frozenset[str] = frozenset()
    active_skill_ids: frozenset[str] = frozenset()
    deny_effects: frozenset[ToolEffect] = frozenset()
    max_parallel_calls: int = 4
    require_confirmation_for: frozenset[str] = frozenset()
    denied_tool_names: frozenset[str] = frozenset()
    auto_approve_sandboxed: bool = False
    approval_mode: ApprovalMode = "ask"
    command_grants: tuple[CommandGrant, ...] = ()
    permissions: ResolvedPermissions | None = None
    permission_provider: Callable[[], ResolvedPermissions] | None = None
    context_provider: Callable[[], ToolExecutionContext] | None = None
    invocation_guard: Callable[[ToolCall], None | Awaitable[None]] | None = None

    def refreshed(self) -> ToolExecutionContext:
        if self.context_provider is None:
            return self
        live = self.context_provider()
        effective = self.effective_permissions
        current = live.effective_permissions
        if effective is not None and current is not None:
            effective = effective.intersection(current)
        elif current is not None:
            effective = current
        return replace(
            self,
            permissions=effective,
            permission_provider=live.permission_provider,
            allow_write_tools=self.allow_write_tools and live.allow_write_tools,
            allow_execute_tools=self.allow_execute_tools and live.allow_execute_tools,
            allow_web_tools=self.allow_web_tools and live.allow_web_tools,
            auto_approve_sandboxed=self.auto_approve_sandboxed and live.auto_approve_sandboxed,
            approval_mode=min((self.approval_mode, live.approval_mode), key=MODE_ORDER.__getitem__),
            command_grants=tuple(grant for grant in self.command_grants if grant in live.command_grants),
            deny_effects=self.deny_effects | live.deny_effects,
            denied_tool_names=self.denied_tool_names | live.denied_tool_names,
            require_confirmation_for=self.require_confirmation_for | live.require_confirmation_for,
        )

    @property
    def effective_permissions(self) -> ResolvedPermissions | None:
        current = None if self.permission_provider is None else self.permission_provider()
        if current is None:
            return self.permissions
        return current if self.permissions is None else current.intersection(self.permissions)

    def __post_init__(self) -> None:
        if self.approval_mode not in MODE_ORDER:
            raise ValueError("approval_mode must be ask, auto, or full")
        if any(not isinstance(grant, CommandGrant) for grant in self.command_grants):
            raise TypeError("command_grants must contain CommandGrant values")
        if self.permissions is not None and not isinstance(self.permissions, ResolvedPermissions):
            raise TypeError("permissions must be ResolvedPermissions or None")
        if self.permission_provider is not None and not callable(self.permission_provider):
            raise TypeError("permission_provider must be callable")
        if self.invocation_guard is not None and not callable(self.invocation_guard):
            raise TypeError("invocation_guard must be callable")
        for name in (
            "allow_write_tools",
            "allow_execute_tools",
            "allow_web_tools",
            "auto_approve_sandboxed",
        ):
            if type(getattr(self, name)) is not bool:
                raise TypeError(f"{name} must be a bool")
        if type(self.max_parallel_calls) is not int:
            raise TypeError("max_parallel_calls must be an integer")
        if self.max_parallel_calls < 1:
            raise ValueError("max_parallel_calls must be positive")
        for name in (
            "approved_tool_call_ids",
            "denied_tool_call_ids",
            "active_skill_ids",
            "require_confirmation_for",
            "denied_tool_names",
        ):
            values = frozenset(getattr(self, name))
            if any(not isinstance(value, str) or not value for value in values):
                raise ValueError(f"{name} must contain non-empty strings")
            object.__setattr__(self, name, values)
        if self.approved_tool_call_ids & self.denied_tool_call_ids:
            raise ValueError("a tool call cannot be both approved and denied")

        effects = frozenset(self.deny_effects)
        if any(not isinstance(effect, ToolEffect) for effect in effects):
            raise TypeError("deny_effects must contain ToolEffect values")
        object.__setattr__(self, "deny_effects", effects)

        for name in ("workspace_root", "cwd"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, Path(value).expanduser().resolve())


class ToolGuardError(ValueError):
    """Bounded hard-guard failure that approval must never bypass."""

    def __init__(self, code: str, reason: str) -> None:
        if not isinstance(code, str) or not code:
            raise ValueError("guard code must be a non-empty string")
        if not isinstance(reason, str) or not reason:
            raise ValueError("guard reason must be a non-empty string")
        self.code = code
        self.reason = " ".join(reason.split())[:512]
        super().__init__(self.reason)


def restore_tool_policy(
    context: ToolExecutionContext, saved: Mapping[str, Any], *, restore_session_grants: bool = False,
) -> ToolExecutionContext:
    """Restore historical grants subject to current resource and runtime limits."""
    policy = dict(saved)
    # Mode is a live user control. Historical Steps must not undo a downgrade.
    policy.pop("approval_mode", None)
    policy.setdefault("allow_web_tools", False)
    if "deny_effects" in policy:
        policy["deny_effects"] = frozenset(ToolEffect(value) for value in policy["deny_effects"])
    snapshot = policy.pop("permissions", None)
    current = context.effective_permissions
    historical = None if snapshot is None else ResolvedPermissions.model_validate(snapshot)
    if current is not None and historical is not None:
        policy["permissions"] = current.intersection(historical)
    elif current is not None:
        policy["permissions"] = current
    elif historical is not None:
        policy["permissions"] = historical
    for key in ("deny_effects", "denied_tool_names", "require_confirmation_for"):
        policy[key] = frozenset(policy.get(key, ())) | getattr(context, key)
    if not restore_session_grants:
        for key in ("allow_write_tools", "allow_execute_tools", "allow_web_tools", "auto_approve_sandboxed"):
            if key in policy:
                policy[key] = policy[key] and getattr(context, key)
    return replace(context, **policy)


def can_use_tool(
    tool: Tool,
    args: Mapping[str, JsonValue],
    resolved: ResolvedToolUse,
    context: ToolExecutionContext,
) -> CanUseToolResult:
    """Return allow/ask/deny from resolved facts without performing approval."""

    permissions = context.effective_permissions
    if permissions is None:
        return _default_tool_decision(tool, args, resolved, context)
    name = tool.definition.name
    rule = permissions.tool_decision(name)
    if rule == "deny":
        return CanUseToolResult(UseToolDecision.DENY, f"permission rule denies tool: {name}")
    for target in resolved.targets:
        if target.kind == "workspace_path":
            write = ToolEffect.WRITE_WORKSPACE in resolved.effects
            if not permissions.path_access(target.value, write=write):
                return CanUseToolResult(UseToolDecision.DENY, permissions.path_reason(target.value, write=write))
    if permissions.read_only and (resolved.effects & {
        ToolEffect.WRITE_WORKSPACE, ToolEffect.DESTRUCTIVE,
    } or tool.approval_profile is ToolApprovalProfile.MCP_ACTION):
        return CanUseToolResult(UseToolDecision.DENY, "read-only permission profile forbids mutation")
    network_decision = None
    if ToolEffect.NETWORK in resolved.effects:
        urls = [target.value for target in resolved.targets if target.kind == "public_web"]
        outcomes = ([permissions.network_decision(url) for url in urls] if urls else
                    [permissions.command_network_allowed()])
        network_decision = "deny" if "deny" in outcomes else "ask" if "ask" in outcomes else "allow"
        if network_decision == "deny":
            return CanUseToolResult(UseToolDecision.DENY, "network destination denied by permission policy")
    # A policy grant never bypasses explicit runtime denies or hard effect floors.
    decision = _default_tool_decision(
        tool, args, resolved, context, network_preapproved=network_decision == "allow",
    )
    if decision.decision is UseToolDecision.DENY:
        return decision
    if rule == "ask":
        decision = CanUseToolResult(UseToolDecision.ASK, f"permission rule requires approval: {name}")
    elif (rule == "allow" and ToolEffect.DESTRUCTIVE not in resolved.effects
          and name not in context.require_confirmation_for
          and tool.approval_profile is not ToolApprovalProfile.MCP_ACTION):
        decision = CanUseToolResult(UseToolDecision.ALLOW, f"permission rule allows tool: {name}")
    preapproved_web = ((context.allow_web_tools or context.approval_mode in {"auto", "full"})
                      and tool.approval_profile is ToolApprovalProfile.PUBLIC_WEB_READ
                      and name in {"web_search", "web_fetch"})
    if (network_decision == "ask" and decision.decision is UseToolDecision.ALLOW
            and not preapproved_web and context.approval_mode != "full"):
        decision = CanUseToolResult(UseToolDecision.ASK, "permission policy requires network approval", "network")
    if permissions.effective_approval == "never" and decision.decision is UseToolDecision.ASK:
        return CanUseToolResult(UseToolDecision.DENY, "approval is required but permission policy disables prompts")
    return decision


def _default_tool_decision(
    tool: Tool,
    args: Mapping[str, JsonValue],
    resolved: ResolvedToolUse,
    context: ToolExecutionContext,
    *,
    network_preapproved: bool = False,
) -> CanUseToolResult:

    tool_name = tool.definition.name
    if tool_name in context.denied_tool_names:
        return CanUseToolResult(
            UseToolDecision.DENY,
            f"tool blocked by runtime policy: {tool_name}",
        )
    denied = resolved.effects & context.deny_effects
    if denied:
        names = ", ".join(sorted(effect.value for effect in denied))
        return CanUseToolResult(
            UseToolDecision.DENY,
            f"effects denied by runtime policy: {names}",
        )
    if tool_name in context.require_confirmation_for or tool.approval_profile is ToolApprovalProfile.MCP_ACTION:
        return CanUseToolResult(
            UseToolDecision.ASK,
            f"runtime policy requires confirmation for tool: {tool_name}",
        )
    if context.approval_mode == "full":
        return CanUseToolResult(UseToolDecision.ALLOW, "user selected full approval within permission limits")
    candidate = command_grant(tool, args, resolved, context) if context.command_grants else None
    if candidate is not None and any(grant.matches(candidate) for grant in context.command_grants):
        return CanUseToolResult(UseToolDecision.ALLOW, "command matches a current Session grant")
    if (
        (context.allow_web_tools or context.approval_mode == "auto")
        and tool.approval_profile is ToolApprovalProfile.PUBLIC_WEB_READ
        and tool_name in {"web_search", "web_fetch"}
        and resolved.effects == frozenset({ToolEffect.NETWORK})
        and any(target.kind == "public_web" for target in resolved.targets)
    ):
        return CanUseToolResult(UseToolDecision.ALLOW, "public web reading is pre-approved by runtime policy")
    if (
        (context.auto_approve_sandboxed or context.approval_mode == "auto")
        and (
            tool.approval_profile
            is ToolApprovalProfile.RESTRICTED_READ_ONLY_PROCESS
        )
        and tool.cancellation_mode is CancellationMode.MANAGED_PROCESS
        and ToolEffect.EXECUTE_PROCESS in resolved.effects
        and ToolEffect.WRITE_WORKSPACE not in resolved.effects
        and ToolEffect.DESTRUCTIVE not in resolved.effects
        and any(
            target.kind == "execution_mode"
            and target.value == "restricted_sandbox"
            for target in resolved.targets
        )
    ):
        return CanUseToolResult(
            UseToolDecision.ALLOW,
            "restricted sandbox execution is pre-approved by runtime policy",
        )

    approval_reasons: list[str] = []
    if (
        ToolEffect.WRITE_WORKSPACE in resolved.effects
        and not context.allow_write_tools
        and context.approval_mode != "auto"
    ):
        approval_reasons.append("workspace write")
    if (
        ToolEffect.EXECUTE_PROCESS in resolved.effects
        and not context.allow_execute_tools
    ):
        approval_reasons.append("process execution")
    if ToolEffect.DESTRUCTIVE in resolved.effects:
        approval_reasons.append("destructive operation")
    separate_network_approval = (
        ToolEffect.NETWORK in resolved.effects
        and ToolEffect.EXECUTE_PROCESS in resolved.effects
        and not network_preapproved
    )
    if (
        ToolEffect.NETWORK in resolved.effects
        and not network_preapproved
        and not separate_network_approval
    ):
        approval_reasons.append("network access")
    if approval_reasons:
        reason = "approval required for " + ", ".join(approval_reasons)
        if separate_network_approval:
            reason += "; network access requires a separate approval"
        return CanUseToolResult(
            UseToolDecision.ASK,
            reason,
            approval_scope="tool",
        )
    if separate_network_approval:
        return CanUseToolResult(
            UseToolDecision.ASK,
            "approval required for network access",
            approval_scope="network",
        )
    return CanUseToolResult(UseToolDecision.ALLOW, "resolved effects are allowed")


def enforce_hard_guards(
    tool: Tool,
    args: Mapping[str, JsonValue],
    resolved: ResolvedToolUse,
    context: ToolExecutionContext,
) -> None:
    """Enforce non-bypassable effect floors and workspace/cwd containment."""

    del args
    if not tool.static_effects.issubset(resolved.effects):
        raise ToolGuardError(
            "effect_floor_violation",
            "resolved effects cannot remove the tool static effect floor",
        )

    workspace_targets = [
        target for target in resolved.targets if target.kind == "workspace_path"
    ]
    if ToolEffect.WRITE_WORKSPACE in resolved.effects and not workspace_targets:
        raise ToolGuardError(
            "workspace_target_required",
            "workspace writes require a resolved workspace_path target",
        )
    for target in workspace_targets:
        if context.workspace_root is None:
            raise ToolGuardError(
                "workspace_root_unavailable",
                "workspace target cannot be checked without a workspace root",
            )
        permissions = context.effective_permissions
        if permissions is None:
            _enforce_containment(target.value, root=Path(context.workspace_root),
                                 code="workspace_escape", label="workspace")
        elif not permissions.path_access(target.value, write=ToolEffect.WRITE_WORKSPACE in resolved.effects):
            raise ToolGuardError("filesystem_denied", permissions.path_reason(
                target.value, write=ToolEffect.WRITE_WORKSPACE in resolved.effects,
            ))

    cwd_targets = [target for target in resolved.targets if target.kind == "cwd_path"]
    if ToolEffect.EXECUTE_PROCESS in resolved.effects and context.cwd is None:
        raise ToolGuardError(
            "cwd_required",
            "process execution requires a resolved runtime cwd",
        )
    for target in cwd_targets:
        if context.cwd is None:
            raise ToolGuardError(
                "cwd_required",
                "cwd target cannot be checked without a runtime cwd",
            )
        _enforce_containment(
            target.value,
            root=Path(context.cwd),
            code="cwd_escape",
            label="cwd",
        )

    for target in resolved.targets:
        if target.kind != "active_skill":
            continue
        if target.value not in context.active_skill_ids:
            raise ToolGuardError(
                "skill_not_active",
                "skill asset access requires an active checkpointed skill",
            )
    if context.workspace_root is not None and context.cwd is not None:
        _enforce_containment(
            str(context.cwd),
            root=Path(context.workspace_root),
            code="cwd_escape",
            label="workspace",
        )


def _enforce_containment(
    value: str,
    *,
    root: Path,
    code: str,
    label: str,
) -> None:
    candidate = Path(value).expanduser()
    if not candidate.is_absolute():
        candidate = root / candidate
    candidate = candidate.resolve()
    try:
        common = Path(os.path.commonpath((str(root), str(candidate))))
    except ValueError:
        common = Path()
    if common != root:
        raise ToolGuardError(code, f"target escapes {label}")

__all__ = [
    "CanUseToolResult",
    "ToolExecutionContext",
    "ToolGuardError",
    "UseToolDecision",
    "can_use_tool",
    "enforce_hard_guards",
]
