import pytest

from agent_runtime.permission_policy import PathRule, PermissionPolicy, ToolRule
from agent_runtime.tools.builtins.filesystem import (
    ApplyPatchInput,
    ListFilesInput,
    ReadFileInput,
    _apply_patch,
    _list_files,
    _read_file,
)
from agent_runtime.tools.builtins.search import SearchTextInput, _search_text
from agent_runtime.workspace import WorkspacePathError, open_workspace


def scoped_workspace(tmp_path):
    workspace = open_workspace(tmp_path / "project", create=True)
    source = tmp_path / "reference"
    source.mkdir()
    (source / "notes.txt").write_text("reference fact\n", encoding="utf-8")
    policy = PermissionPolicy(filesystem=(PathRule(path=str(source), access="read"),)).resolve(workspace.root)
    workspace.permission_provider = lambda: policy
    return workspace, source


@pytest.mark.anyio
async def test_read_list_and_search_share_external_read_permission(tmp_path):
    workspace, source = scoped_workspace(tmp_path)
    assert "reference fact" in _read_file(workspace, ReadFileInput(path=str(source / "notes.txt"))).content
    entries = _list_files(workspace, ListFilesInput(path=str(source))).entries
    assert entries[0].path == str(source / "notes.txt")
    matches = (await _search_text(workspace, SearchTextInput(pattern="reference fact", path=str(source)))).matches
    assert matches[0].file_path == str(source / "notes.txt")
    with pytest.raises((WorkspacePathError, PermissionError)):
        _apply_patch(
            workspace,
            ApplyPatchInput(file_path=str(source / "notes.txt"), old_string="reference", new_string="changed"),
        )
    assert (source / "notes.txt").read_text() == "reference fact\n"


@pytest.mark.anyio
async def test_recursive_tools_do_not_expose_denied_descendants(tmp_path):
    workspace, source = scoped_workspace(tmp_path)
    secret = source / "secret.txt"
    secret.write_text("reference secret")
    policy = PermissionPolicy(
        filesystem=(PathRule(path=str(source), access="read"), PathRule(path=str(secret), access="deny"))
    ).resolve(workspace.root)
    workspace.permission_provider = lambda: policy
    assert all(entry.name != "secret.txt" for entry in _list_files(workspace, ListFilesInput(path=str(source))).entries)
    matches = (await _search_text(workspace, SearchTextInput(pattern="reference", path=str(source)))).matches
    assert [match.file_path for match in matches] == [str(source / "notes.txt")]
    with pytest.raises((WorkspacePathError, PermissionError)):
        _read_file(workspace, ReadFileInput(path=str(secret)))


def test_revocation_applies_to_already_created_tools(tmp_path):
    workspace, source = scoped_workspace(tmp_path)
    from agent_runtime.tools.builtins.filesystem import create_read_file_tool

    tool = create_read_file_tool(workspace)
    args = tool.validate_input({"path": str(source / "notes.txt")})
    assert tool.run(args).content == "reference fact\n"
    workspace.permission_provider = lambda: PermissionPolicy().resolve(workspace.root)
    with pytest.raises((WorkspacePathError, PermissionError)):
        tool.run(args)


def test_explicit_external_write_permission_is_enforced(tmp_path):
    workspace, source = scoped_workspace(tmp_path)
    workspace.permission_provider = lambda: PermissionPolicy(
        filesystem=(PathRule(path=str(source), access="write"),),
    ).resolve(workspace.root)
    _apply_patch(
        workspace, ApplyPatchInput(file_path=str(source / "notes.txt"), old_string="reference", new_string="changed")
    )
    assert (source / "notes.txt").read_text() == "changed fact\n"


def test_effect_policy_deny_cannot_be_bypassed_by_allow_flags(tmp_path):
    from agent_runtime.tools.builtins.filesystem import create_read_file_tool
    from agent_runtime.tools.permissions import ToolExecutionContext, UseToolDecision, can_use_tool

    workspace, source = scoped_workspace(tmp_path)
    permissions = PermissionPolicy(tools=(ToolRule(pattern="read_*", decision="deny"),)).resolve(workspace.root)
    context = ToolExecutionContext(
        workspace_root=workspace.root,
        cwd=workspace.root,
        allow_write_tools=True,
        allow_execute_tools=True,
        permissions=permissions,
    )
    tool = create_read_file_tool(workspace)
    args = tool.validate_input({"path": str(source / "notes.txt")})
    decision = can_use_tool(tool, args, tool.resolve_use(args), context)
    assert decision.decision is UseToolDecision.DENY


def test_read_only_policy_denies_process_write_even_with_preapproval(tmp_path):
    from agent_runtime.tools.builtins.shell import create_run_command_tool
    from agent_runtime.tools.permissions import ToolExecutionContext, UseToolDecision, can_use_tool

    workspace = open_workspace(tmp_path, create=True)
    context = ToolExecutionContext(
        workspace_root=workspace.root,
        cwd=workspace.root,
        allow_write_tools=True,
        allow_execute_tools=True,
        permissions=PermissionPolicy(profile="read-only").resolve(workspace.root),
    )
    tool = create_run_command_tool(workspace)
    args = tool.validate_input({"command": "printf value > output.txt", "workspace_write": True})
    decision = can_use_tool(tool, args, tool.resolve_use(args), context)
    assert decision.decision is UseToolDecision.DENY


def test_sdk_reloads_file_policy_and_preserves_managed_ceiling(tmp_path):
    import json

    from agent_runtime import Agent

    config = tmp_path / "permissions.json"
    config.write_text(json.dumps({"profile": "trusted-local"}))
    agent = Agent(
        workspace_path=tmp_path / "project",
        permissions=config,
        managed_permissions=PermissionPolicy(profile="read-only"),
        enable_workspace_mcp=False,
    )
    assert not agent.effective_permissions().path_access(tmp_path / "project" / "result", write=True)
    config.write_text(json.dumps({"profile": "workspace", "tools": [{"pattern": "read_file", "decision": "deny"}]}))
    assert agent.effective_permissions().tool_decision("read_file") == "deny"


@pytest.mark.anyio
async def test_session_policy_snapshot_and_current_revocation(tmp_path, monkeypatch):
    from agent_runtime import Agent
    from tests.agent.harness.test_public_agent_cutover import PublicHarnessModel

    agent = Agent(
        workspace_path=tmp_path,
        checkpoint_db=tmp_path / "rollout.db",
        permissions=PermissionPolicy(profile="trusted-local"),
        enable_workspace_mcp=False,
    )
    monkeypatch.setattr(agent, "_model_factory", PublicHarnessModel)
    async with agent.session(require_workspace_change=False) as session:
        result = await session.submit("read-only analysis")
        snapshot = session.store.read_turn(result.turn_id).binding_manifest["tool_execution_policy"]["permissions"]
        assert snapshot["profile"] == "trusted-local"
        session.update_permissions(PermissionPolicy(profile="read-only"))
        assert not session.permissions().path_access(tmp_path / "result", write=True)
        assert session.store.read_thread(session.thread_id).settings["tool_execution_policy"]["permissions"]


def test_current_restrictions_override_frozen_step_policy(tmp_path):
    from agent_runtime.harness.services import restore_tool_policy
    from agent_runtime.tools.permissions import ToolExecutionContext

    current = ToolExecutionContext(
        workspace_root=tmp_path, cwd=tmp_path, permissions=PermissionPolicy(profile="read-only").resolve(tmp_path)
    )
    restored = restore_tool_policy(
        current,
        {
            "allow_write_tools": True,
            "allow_execute_tools": True,
            "permissions": PermissionPolicy().resolve(tmp_path).model_dump(mode="json"),
        },
    )
    assert not restored.effective_permissions.path_access(tmp_path / "result", write=True)


def test_restore_does_not_reinstate_revoked_preapproval(tmp_path):
    from agent_runtime.harness.services import restore_tool_policy
    from agent_runtime.tools.permissions import ToolExecutionContext

    current = ToolExecutionContext(workspace_root=tmp_path, cwd=tmp_path)
    restored = restore_tool_policy(
        current, {"allow_write_tools": True, "allow_execute_tools": True, "allow_web_tools": True}
    )
    assert not restored.allow_write_tools
    assert not restored.allow_execute_tools
    assert not restored.allow_web_tools


def test_intersection_fingerprint_is_stable_under_live_refresh(tmp_path):
    from dataclasses import replace

    from agent_runtime.tools.permissions import ToolExecutionContext

    policy = PermissionPolicy().resolve(tmp_path)
    history = PermissionPolicy(profile="trusted-local").resolve(tmp_path)
    live = ToolExecutionContext(permissions=policy, permission_provider=lambda: policy)
    frozen = replace(live, permissions=policy.intersection(history), context_provider=lambda: live)
    assert frozen.effective_permissions.revision == frozen.refreshed().effective_permissions.revision
    assert policy.intersection(history).revision == history.intersection(policy).revision


def test_read_only_ceiling_blocks_destructive_effect_without_path(tmp_path):
    from agent_runtime.tools.permissions import ToolExecutionContext, UseToolDecision, can_use_tool
    from agent_runtime.tools.tool import ResolvedToolUse, ToolEffect
    from tests.agent.test_single_tool_executor import _tool

    permissions = (
        PermissionPolicy()
        .resolve(tmp_path)
        .intersection(
            PermissionPolicy(profile="read-only").resolve(tmp_path),
        )
    )
    tool = _tool(static_effects=frozenset({ToolEffect.DESTRUCTIVE}))
    decision = can_use_tool(
        tool,
        {},
        ResolvedToolUse(frozenset({ToolEffect.DESTRUCTIVE}), ()),
        ToolExecutionContext(permissions=permissions, approved_tool_call_ids=frozenset({"call_1"})),
    )
    assert decision.decision is UseToolDecision.DENY


@pytest.mark.anyio
async def test_prepared_batch_rechecks_live_tool_revocation():
    from dataclasses import replace

    from agent_runtime.tools.executor import ToolExecutor
    from agent_runtime.tools.permissions import ToolExecutionContext
    from tests.agent.test_single_tool_executor import _call, _normalized, _tool

    calls = []
    current = [ToolExecutionContext()]
    initial = replace(current[0], context_provider=lambda: current[0])

    def first(_arguments):
        calls.append("first")
        current[0] = replace(current[0], denied_tool_names=frozenset({"second"}))
        return _normalized()

    def second(_arguments):
        calls.append("second")
        return _normalized()

    tools = {
        name: _tool(
            name=name,
            static_effects=frozenset(),
            run=runner,
            normalize_output=lambda value: value,
            idempotent=False,
            concurrency_safe=False,
        )
        for name, runner in (("first", first), ("second", second))
    }
    await ToolExecutor(tools).execute_batch(
        (
            _call(name="first", exposed_names=("first", "second")),
            _call(name="second", call_id="call_2", exposed_names=("first", "second")),
        ),
        context=initial,
    )
    assert calls == ["first"]


def test_case_alias_cannot_read_denied_file_or_rewrite_policy(tmp_path):
    import json

    from agent_runtime import Agent

    secret = tmp_path / "secret.txt"
    secret.write_text("secret")
    alias = tmp_path / "SECRET.TXT"
    if not alias.exists():
        pytest.skip("filesystem is case-sensitive")
    policy = PermissionPolicy(filesystem=(PathRule(path=str(secret), access="deny"),)).resolve(tmp_path)
    assert not policy.path_access(alias)
    metadata = tmp_path / ".codex"
    metadata.mkdir()
    (metadata / "policy").write_text("original")
    assert not policy.path_access(tmp_path / ".CODEX" / "policy", write=True)
    config = tmp_path / "permissions.json"
    config.write_text(json.dumps({"profile": "workspace"}))
    effective = Agent(workspace_path=tmp_path, permissions=config).effective_permissions()
    assert not effective.path_access(tmp_path / "PERMISSIONS.JSON", write=True)


def test_file_tools_cannot_install_higher_priority_automatic_policy(tmp_path, monkeypatch):
    from pathlib import Path

    from agent_runtime.permission_policy import resolve_permission_sources

    workspace = open_workspace(tmp_path / "project", create=True)
    monkeypatch.setattr(Path, "home", lambda: workspace.root)
    monkeypatch.setenv("PRAXIS_CONFIG_DIR", str(workspace.root / ".config/praxis"))
    directory = workspace.root / ".config/praxis"
    directory.mkdir(parents=True)
    source = directory / "permissions.json"
    source.write_text('{"tools":[{"pattern":"execute_python","decision":"deny"}]}')
    workspace.permission_provider = lambda: resolve_permission_sources(workspace.root)
    with pytest.raises((WorkspacePathError, PermissionError)):
        _apply_patch(workspace, ApplyPatchInput(file_path=".config/praxis/permissions.toml",
                                               new_string='profile="trusted-local"'))
    assert not (directory / "permissions.toml").exists()
    assert workspace.permissions.tool_decision("execute_python") == "deny"
