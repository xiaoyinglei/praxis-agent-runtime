"""Model-requested delivery tools follow the normal approval-mode contract."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import pytest
from pytest import MonkeyPatch

from agent_runtime.git_delivery import DeliveryError, DeliveryResult, PullRequestResult
from agent_runtime.permission_policy import NetworkPolicy, PermissionPolicy, ToolRule
from agent_runtime.tools.builtins.delivery import (
    create_commit_changes_tool,
    create_open_pull_request_tool,
)
from agent_runtime.tools.executor import ToolExecutor
from agent_runtime.tools.permissions import ToolExecutionContext, UseToolDecision, can_use_tool
from agent_runtime.tools.tool import JsonValue, Tool, ToolCall, ToolCallOrigin, ToolEffect
from agent_runtime.workspace import open_workspace

pytestmark = pytest.mark.anyio


def _call(tool: Tool, arguments: Mapping[str, JsonValue]) -> ToolCall:
    return ToolCall(
        tool_call_id="call_delivery",
        tool_name=tool.definition.name,
        arguments=dict(arguments),
        origin=ToolCallOrigin(
            request_id="request_delivery",
            toolset_revision="toolset-v1",
            exposed_tool_names=(tool.definition.name,),
        ),
    )


def _context(**overrides: object) -> ToolExecutionContext:
    fields: dict[str, object] = {"workspace_root": None, "cwd": None, "approval_mode": "ask"}
    fields.update(overrides)
    return ToolExecutionContext(**fields)  # type: ignore[arg-type]


@pytest.fixture
def workspace(tmp_path: Path):
    return open_workspace(tmp_path / "project", create=True)


def test_commit_asks_in_ask_and_auto_and_allows_in_full(workspace) -> None:
    tool = create_commit_changes_tool(workspace)
    arguments: Mapping[str, JsonValue] = {"message": "fix: timeout"}
    resolved = tool.resolve_use(arguments)

    ask = can_use_tool(tool, arguments, resolved, _context(approval_mode="ask"))
    assert ask.decision is UseToolDecision.ASK
    auto = can_use_tool(tool, arguments, resolved, _context(approval_mode="auto"))
    assert auto.decision is UseToolDecision.ASK
    full = can_use_tool(tool, arguments, resolved, _context(approval_mode="full"))
    assert full.decision is UseToolDecision.ALLOW
    preapproved = can_use_tool(
        tool, arguments, resolved, _context(approval_mode="ask", allow_execute_tools=True),
    )
    assert preapproved.decision is UseToolDecision.ALLOW


def test_permission_rules_apply_to_delivery(workspace, tmp_path: Path) -> None:
    tool = create_commit_changes_tool(workspace)
    arguments: Mapping[str, JsonValue] = {"message": "fix: timeout"}
    resolved = tool.resolve_use(arguments)
    policy = PermissionPolicy(tools=(ToolRule(pattern="commit_*", decision="deny"),)).resolve(tmp_path)

    denied = can_use_tool(
        tool, arguments, resolved,
        _context(approval_mode="full", permissions=policy),
    )
    assert denied.decision is UseToolDecision.DENY

    ask_policy = PermissionPolicy(tools=(ToolRule(pattern="commit_changes", decision="ask"),)).resolve(tmp_path)
    forced_ask = can_use_tool(
        tool, arguments, resolved,
        _context(approval_mode="full", permissions=ask_policy),
    )
    assert forced_ask.decision is UseToolDecision.ASK


def test_domain_filtered_network_policy_denies_pull(workspace, tmp_path: Path) -> None:
    tool = create_open_pull_request_tool(workspace)
    arguments: Mapping[str, JsonValue] = {}
    resolved = tool.resolve_use(arguments)
    policy = PermissionPolicy(
        network=NetworkPolicy(allowed_domains=("example.com",)),
    ).resolve(tmp_path)

    result = can_use_tool(
        tool, arguments, resolved,
        _context(approval_mode="full", permissions=policy),
    )
    assert result.decision is UseToolDecision.DENY


def test_read_only_profile_still_permits_process_delivery(workspace, tmp_path: Path) -> None:
    # Read-only forbids workspace writes; delivery executes a host process and
    # never mutates workspace files, so it stays available under approval.
    tool = create_commit_changes_tool(workspace)
    arguments: Mapping[str, JsonValue] = {"message": "fix: timeout"}
    policy = PermissionPolicy(profile="read-only").resolve(tmp_path)

    result = can_use_tool(
        tool, arguments, tool.resolve_use(arguments),
        _context(approval_mode="full", permissions=policy),
    )
    assert result.decision is UseToolDecision.ALLOW


async def test_commit_executes_host_side_after_approval(
    workspace, monkeypatch: MonkeyPatch,
) -> None:
    committed: list[tuple[str, str | None]] = []

    class _FakeDelivery:
        def __init__(self, root: Path) -> None:
            self.root = root

        async def commit(self, message: str, *, branch_name: str | None = None):
            committed.append((message, branch_name))
            return DeliveryResult(
                commit_id="c" * 40, branch="praxis/x",
                changed_file_count=2, created_branch=True,
            )

    monkeypatch.setattr("agent_runtime.tools.builtins.delivery.GitDelivery", _FakeDelivery)
    tool = create_commit_changes_tool(workspace)
    execution = await ToolExecutor({tool.definition.name: tool}).execute(
        _call(tool, {"message": "fix: timeout", "branch_name": "praxis/feature"}),
        context=_context(workspace_root=workspace.root, cwd=workspace.root, approval_mode="full"),
    )
    assert execution.result.is_error is False
    assert committed == [("fix: timeout", "praxis/feature")]
    output = execution.result.structured_content
    assert isinstance(output, Mapping)
    assert output["branch"] == "praxis/x"
    assert output["changed_file_count"] == 2


async def test_delivery_refusal_becomes_structured_error(
    workspace, monkeypatch: MonkeyPatch,
) -> None:
    class _RefusingDelivery:
        def __init__(self, root: Path) -> None:
            self.root = root

        async def commit(self, message: str, *, branch_name: str | None = None):
            raise DeliveryError("没有可提交的变更。")

    monkeypatch.setattr("agent_runtime.tools.builtins.delivery.GitDelivery", _RefusingDelivery)
    tool = create_commit_changes_tool(workspace)
    execution = await ToolExecutor({tool.definition.name: tool}).execute(
        _call(tool, {"message": "fix: timeout"}),
        context=_context(workspace_root=workspace.root, cwd=workspace.root, approval_mode="full"),
    )
    assert execution.result.is_error is True
    assert execution.result.error_code == "delivery_failed"
    assert "没有可提交的变更" in str(execution.result.error_message)


async def test_pull_returns_url(workspace, monkeypatch: MonkeyPatch) -> None:
    opened: list[dict[str, object]] = []

    class _FakeDelivery:
        def __init__(self, root: Path) -> None:
            self.root = root

        async def open_pull_request(self, *, title=None, body=None, base_branch=None):
            opened.append({"title": title, "body": body, "base_branch": base_branch})
            return PullRequestResult(
                url="https://github.com/o/r/pull/9", branch="praxis/x", created=True,
            )

    monkeypatch.setattr("agent_runtime.tools.builtins.delivery.GitDelivery", _FakeDelivery)
    tool = create_open_pull_request_tool(workspace)
    execution = await ToolExecutor({tool.definition.name: tool}).execute(
        _call(tool, {"title": "修复超时"}),
        context=_context(workspace_root=workspace.root, cwd=workspace.root, approval_mode="full"),
    )
    assert execution.result.is_error is False
    output = execution.result.structured_content
    assert isinstance(output, Mapping)
    assert output["url"] == "https://github.com/o/r/pull/9"
    assert opened == [{"title": "修复超时", "body": None, "base_branch": None}]


def test_effects_declare_host_execution_and_network(workspace) -> None:
    commit = create_commit_changes_tool(workspace)
    pull = create_open_pull_request_tool(workspace)
    assert commit.static_effects == frozenset({ToolEffect.EXECUTE_PROCESS})
    assert pull.static_effects == frozenset({ToolEffect.EXECUTE_PROCESS, ToolEffect.NETWORK})
