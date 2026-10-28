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


async def test_delivery_remains_usable_inside_an_open_session(workspace, tmp_path) -> None:
    import subprocess

    from agent_runtime.harness import Session

    root = workspace.root
    for args in (("init", "-b", "main"), ("config", "user.name", "Test"),
                 ("config", "user.email", "test@example.com")):
        subprocess.run(["git", "-C", str(root), *args], check=True, capture_output=True)
    (root / "base").write_text("base")
    subprocess.run(["git", "-C", str(root), "add", "base"], check=True)
    subprocess.run(["git", "-C", str(root), "commit", "-qm", "base"], check=True)
    tool = create_commit_changes_tool(workspace)
    async with await Session.open(database=tmp_path / "rollout.db", workspace=root,
                                  model=object(), tools={tool.definition.name: tool}):
        (root / "change").write_text("allowed")
        execution = await ToolExecutor({tool.definition.name: tool}).execute(
            _call(tool, {"message": "from session"}),
            context=_context(workspace_root=root, cwd=root, approval_mode="full"),
        )
        assert not execution.result.is_error
        assert execution.result.structured_content["commit_id"]


async def test_cancelled_delivery_tool_exposes_its_completed_result(workspace, monkeypatch) -> None:
    import asyncio

    from agent_runtime.git_delivery import DeliveryInterruptedError

    started, release = asyncio.Event(), asyncio.Event()
    class Delivery:
        def __init__(self, _root):
            pass
        async def commit(self, *_args, **_kwargs):
            started.set()
            await release.wait()
            raise DeliveryInterruptedError(DeliveryResult("a" * 40, "praxis/task", 1, True))
    monkeypatch.setattr("agent_runtime.tools.builtins.delivery.GitDelivery", Delivery)
    tool = create_commit_changes_tool(workspace)
    execution = asyncio.create_task(ToolExecutor({tool.definition.name: tool}).execute(
        _call(tool, {"message": "settled"}),
        context=_context(workspace_root=workspace.root, cwd=workspace.root, approval_mode="full"),
    ))
    await started.wait()
    execution.cancel()
    release.set()
    result = await execution
    assert not result.result.is_error
    assert result.result.structured_content["commit_id"] == "a" * 40
    assert result.result.structured_content["interruption_requested"] is True


async def test_session_cancellation_records_the_real_git_commit(workspace, tmp_path) -> None:
    import asyncio
    import subprocess

    from agent_runtime.core.model_request import toolset_revision_for_tools
    from agent_runtime.harness import HarnessModelResponse, HarnessToolCall, PreparedModelCall, Session

    root = workspace.root
    for args in (("init", "-b", "main"), ("config", "user.name", "Test"),
                 ("config", "user.email", "test@example.com")):
        subprocess.run(["git", "-C", str(root), *args], check=True, capture_output=True)
    (root / "base").write_text("base")
    subprocess.run(["git", "-C", str(root), "add", "base"], check=True)
    subprocess.run(["git", "-C", str(root), "commit", "-qm", "base"], check=True)
    (root / "change").write_text("approved")
    entered, release = tmp_path / "hook-entered", tmp_path / "hook-release"
    hook = root / ".git/hooks/pre-commit"
    hook.write_text(f'#!/bin/sh\ntouch "{entered}"\nwhile [ ! -e "{release}" ]; do sleep 0.02; done\n')
    hook.chmod(0o755)
    class Model:
        def snapshot(self, **_kwargs):
            return {"model_id": "delivery-test"}
        def prepare(self, request):
            return PreparedModelCall(request_hash="r", context_hash="c", wire_hash="w",
                tool_hash=toolset_revision_for_tools(request.tools), request_ref={
                    "request_id": "delivery-test", "toolset_revision": toolset_revision_for_tools(request.tools),
                    "exposed_tool_names": [tool.definition.name for tool in request.tools]})
        async def dispatch(self, prepared, *, event_sink=None):
            return HarnessModelResponse(
                text="", provider_response_id="delivery-response", usage={"input_tokens": 1, "output_tokens": 1},
                tool_calls=(HarnessToolCall(id="commit-call", name="commit_changes",
                                           arguments={"message": "settled in session"}),),
            )
        def ensure_available(self, binding, **kwargs):
            pass
    tool = create_commit_changes_tool(workspace)
    async with await Session.open(database=tmp_path / "session.db", workspace=root,
            model=Model(), tools={tool.definition.name: tool},
            tool_execution_context=_context(workspace_root=root, cwd=root, approval_mode="full")) as session:
        submission = asyncio.create_task(session.submit("commit this change"))
        try:
            async with asyncio.timeout(5):
                while not entered.exists():
                    if submission.done():
                        pytest.fail(f"submission ended before tool ran: {submission.result()}")
                    await asyncio.sleep(0.01)
            submission.cancel()
        finally:
            release.touch()
        with pytest.raises(asyncio.CancelledError):
            await submission
        operations = session.store.list_tool_operations()
        assert len(operations) == 1
        assert operations[0].status == "succeeded"
        result = session.store.read_item(operations[0].result_item_id).payload["structured_content"]
        head = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True, text=True, check=True,
        ).stdout.strip()
        assert result["commit_id"] == head
        assert result["error"] is None
