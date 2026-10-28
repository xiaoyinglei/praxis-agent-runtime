from dataclasses import replace

import pytest

from agent_runtime.permission_policy import NetworkPolicy, PermissionPolicy, ToolRule
from agent_runtime.tools.builtins.shell import create_run_command_tool
from agent_runtime.tools.permissions import ToolExecutionContext, can_use_tool
from agent_runtime.workspace import open_workspace
from tests.agent.harness.test_approval_resume import _write_tool


def shell(tmp_path, mode="ask", **policy):
    workspace = open_workspace(tmp_path, create=True)
    tool = create_run_command_tool(workspace)
    context = ToolExecutionContext(
        workspace_root=workspace.root, cwd=workspace.root, approval_mode=mode,
        permissions=PermissionPolicy(**policy).resolve(workspace.root),
    )
    return tool, context


def decision(tool, context, **kwargs):
    args = tool.validate_input({"command": "printf hello", **kwargs})
    return can_use_tool(tool, args, tool.resolve_use(args), context).decision


@pytest.mark.parametrize("mode,expected", [("ask", "ask"), ("auto", "allow"), ("full", "allow")])
def test_mode_controls_sandbox_command(tmp_path, mode, expected):
    tool, context = shell(tmp_path, mode)
    assert decision(tool, context) == expected


@pytest.mark.parametrize("mode,expected", [("ask", "ask"), ("auto", "allow"), ("full", "allow")])
def test_mode_controls_non_destructive_file_edit(tmp_path, mode, expected):
    _, context = shell(tmp_path, mode)
    tool = _write_tool(tmp_path, [])
    args = tool.validate_input({"path": "result.txt", "text": "hello"})
    assert can_use_tool(tool, args, tool.resolve_use(args), context).decision == expected


@pytest.mark.parametrize("mode,expected", [("auto", "ask"), ("full", "allow")])
def test_mode_controls_destructive_and_network_scopes(tmp_path, mode, expected):
    tool, context = shell(tmp_path, mode)
    assert decision(tool, context, workspace_write=True) == expected
    assert decision(tool, context, network=True) == expected


def test_full_does_not_override_denials_or_forced_confirmation(tmp_path):
    tool, context = shell(tmp_path, "full", network=NetworkPolicy(decision="deny"))
    assert decision(tool, context, network=True) == "deny"
    assert decision(tool, replace(context, require_confirmation_for=frozenset({"run_command"}))) == "ask"
    assert decision(tool, replace(context, denied_tool_names=frozenset({"run_command"}))) == "deny"
    readonly = PermissionPolicy(profile="read-only").resolve(tmp_path)
    assert decision(tool, replace(context, permissions=readonly), workspace_write=True) == "deny"
    explicit = PermissionPolicy(tools=(ToolRule(pattern="run_command", decision="ask"),)).resolve(tmp_path)
    assert decision(tool, replace(context, permissions=explicit)) == "ask"


def test_mode_validated_and_never_means_deny_not_autoapprove(tmp_path):
    with pytest.raises(ValueError, match="approval_mode"):
        ToolExecutionContext(approval_mode="invalid")
    tool, context = shell(tmp_path, approval="never")
    assert decision(tool, context) == "deny"


def test_grant_matches_only_bounded_command_family(tmp_path):
    from agent_runtime.approval import command_grant

    tool, context = shell(tmp_path)
    args = tool.validate_input({"command": "printf hello"})
    grant = command_grant(tool, args, tool.resolve_use(args), context)
    assert grant is not None
    granted = replace(context, command_grants=(grant,))
    assert decision(tool, granted, command="printf hello world") == "allow"
    for command in ("printf helloworld", "printf hello; touch pwn", "printf hello $(id)", "printf hello | sh"):
        assert decision(tool, granted, command=command) == "ask"
    assert decision(tool, granted, network=True) == "ask"
    assert decision(tool, granted, workspace_write=True) == "ask"
    assert decision(tool, replace(granted, command_grants=())) == "ask"
    changed = PermissionPolicy(network=NetworkPolicy(decision="allow")).resolve(tmp_path)
    assert decision(tool, replace(granted, permissions=changed)) == "ask"
    assert decision(replace(tool, execution_revision="changed"), granted) == "ask"
    assert decision(tool, replace(granted, require_confirmation_for=frozenset({"run_command"}))) == "ask"
    (tmp_path / "other").mkdir()
    assert decision(tool, granted, working_dir="other") == "ask"


@pytest.mark.parametrize("args", [
    {"command": "printf hello; id"}, {"command": "sh -c 'printf hello'"},
    {"command": "python -c 'print(1)'"}, {"command": "printf hello", "network": True},
    {"command": "printf hello", "workspace_write": True},
])
def test_complex_or_elevated_commands_do_not_offer_reusable_grant(tmp_path, args):
    from agent_runtime.approval import command_grant

    tool, context = shell(tmp_path)
    validated = tool.validate_input(args)
    assert command_grant(tool, validated, tool.resolve_use(validated), context) is None


class CommandModel:
    """Use the existing deterministic model contract, executing real tool IO."""

    def __new__(cls, arguments, tool_name="run_command"):
        from agent_runtime.harness import HarnessToolCall
        from tests.agent.harness.test_approval_resume import WriteThenAnswerModel

        class Model(WriteThenAnswerModel):
            async def dispatch(self, prepared, *, event_sink=None):
                response = await super().dispatch(prepared, event_sink=event_sink)
                if response.tool_calls:
                    return replace(response, outputs=(), tool_calls=(HarnessToolCall(
                        id="command-call", name=tool_name, arguments=arguments,
                    ),))
                return response

        return Model()


async def command_session(tmp_path, *, mode=None, thread_id=None, **arguments):
    from agent_runtime.harness import Session
    from agent_runtime.tools.registry import ToolRegistry
    from tests.agent.harness.test_approval_resume import AcceptAnswer

    tool, context = shell(tmp_path)
    registry = ToolRegistry()
    registry.register(tool)
    return await Session.open(
        database=tmp_path / "rollout.sqlite3", workspace=tmp_path, thread_id=thread_id,
        model=CommandModel({"command": "printf hello", **arguments}),
        tools=registry.freeze(), completion_gate=AcceptAnswer(), tool_execution_context=context,
        approval_mode=mode,
    )


@pytest.mark.anyio
async def test_session_grant_applies_next_turn_can_revoke_and_is_not_saved(tmp_path):
    async with await command_session(tmp_path, mode="ask") as session:
        result = await session.submit("run")
        assert result.status == "paused"
        assert "allow_session" in result.pause.options
        result = await session.resume(result.turn_id, "allow_session")
        assert result.status == "done"
        assert len(session.approval_grants()) == 1
        with pytest.raises(RuntimeError, match="approval_already_resolved"):
            await session.resume(result.turn_id, "allow_session")
        assert len(session.approval_grants()) == 1
        assert (await session.submit("run again")).status == "done"
        saved = session.store.read_thread(session.thread_id).settings["tool_execution_policy"]
        assert "command_grants" not in saved
        session.revoke_approval_grants()
        assert session.approval_grants() == ()
        result = await session.submit("run after revoke")
        assert result.status == "paused"
        await session.resume(result.turn_id, "deny")


@pytest.mark.anyio
async def test_reopen_restores_mode_but_not_command_grants(tmp_path):
    async with await command_session(tmp_path, mode="ask") as session:
        result = await session.submit("run")
        await session.resume(result.turn_id, "allow_session")
        thread_id = session.thread_id
    async with await command_session(tmp_path, thread_id=thread_id) as reopened:
        assert reopened.tool_execution_context.approval_mode == "ask"
        assert reopened.approval_grants() == ()
        assert (await reopened.submit("run again")).status == "paused"


@pytest.mark.anyio
async def test_full_process_network_has_no_second_approval(tmp_path):
    async with await command_session(tmp_path, mode="full", network=True) as session:
        result = await session.submit("run")
        assert result.status == "done"
        assert not session.store.list_interactions(result.turn_id)
        assert all(op.status == "succeeded" for op in session.store.list_tool_operations(result.turn_id))


@pytest.mark.anyio
async def test_session_grant_is_rejected_if_not_offered(tmp_path):
    async with await command_session(tmp_path, mode="ask", network=True) as session:
        result = await session.submit("run")
        assert "allow_session" not in result.pause.options
        with pytest.raises(RuntimeError, match="tool_approval_required"):
            await session.resume(result.turn_id, "allow_session")
        assert session.approval_grants() == ()
        assert session.store.read_turn(result.turn_id).status == "paused"


@pytest.mark.anyio
async def test_auto_edit_performs_real_file_io(tmp_path):
    from agent_runtime.harness import Session
    from agent_runtime.tools.registry import ToolRegistry
    from tests.agent.harness.test_approval_resume import AcceptAnswer, WriteThenAnswerModel

    calls = []
    registry = ToolRegistry()
    registry.register(_write_tool(tmp_path, calls))
    async with await Session.open(database=tmp_path / "edit.sqlite3", workspace=tmp_path,
                                  model=WriteThenAnswerModel(), tools=registry.freeze(),
                                  completion_gate=AcceptAnswer(), approval_mode="auto") as session:
        result = await session.submit("write")
        assert result.status == "done"
        assert calls == ["approved.txt"]
        assert (tmp_path / "approved.txt").read_text() == "approved content"


@pytest.mark.anyio
async def test_public_sdk_mode_reaches_real_tool_executor(tmp_path, monkeypatch):
    from agent_runtime import Agent
    from agent_runtime.harness import RuntimeState

    agent = Agent(workspace_path=tmp_path, checkpoint_db=tmp_path / "public.sqlite3",
                  model_session_path=None, enable_workspace_mcp=False, approval_mode="auto")
    monkeypatch.setattr(agent, "_model_factory", lambda: CommandModel({"command": "printf hello"}))
    result = await agent.run("run bounded command")
    assert result.status == "done"
    async with (await RuntimeState.open(agent.checkpoint_db)) as store:
        [operation] = store.list_tool_operations(result.turn_id)
        assert operation.status == "succeeded"
        assert store.read_turn(result.turn_id).binding_manifest["tool_execution_policy"]["approval_mode"] == "auto"


@pytest.mark.anyio
async def test_explicit_mode_overrides_saved_mode_and_resume_cannot_undo_downgrade(tmp_path):
    async with await command_session(tmp_path, mode="full") as session:
        result = await session.submit("run")
        thread_id = session.thread_id
    async with await command_session(tmp_path, mode="ask", thread_id=thread_id) as session:
        assert session.tool_execution_context.approval_mode == "ask"
        result = await session.submit("run")
        assert result.status == "paused"
        session.update_tool_policy(approval_mode="auto")
        changed = await session.resume(result.turn_id, "approve")
        assert changed.status == "paused"  # The old approval scope is invalidated.
        assert session.tool_execution_context.approval_mode == "auto"


@pytest.mark.anyio
@pytest.mark.parametrize("mode", ["auto", "full"])
async def test_deny_pending_call_still_denies_after_mode_changes(tmp_path, mode):
    async with await command_session(tmp_path, mode="ask") as session:
        result = await session.submit("run")
        session.update_tool_policy(approval_mode=mode)
        result = await session.resume(result.turn_id, "deny")
        assert result.status == "done"
        [operation] = session.store.list_tool_operations(result.turn_id)
        assert operation.status == "denied"


@pytest.mark.anyio
async def test_recheck_after_mode_selection_executes_current_call_once(tmp_path):
    async with await command_session(tmp_path, mode="auto", workspace_write=True) as session:
        result = await session.submit("run")
        assert result.status == "paused"
        session.update_tool_policy(approval_mode="full")
        result = await session.resume(result.turn_id, "recheck")
        assert result.status == "done"
        [operation] = session.store.list_tool_operations(result.turn_id)
        assert operation.status == "succeeded"
        assert operation.attempt_count == 1
        assert not any(i.status == "pending" for i in session.store.list_interactions(result.turn_id))


@pytest.mark.anyio
async def test_recheck_is_not_a_one_time_approval_and_preserves_forced_ask(tmp_path):
    async with await command_session(tmp_path, mode="ask") as session:
        result = await session.submit("run")
        unchanged = await session.resume(result.turn_id, "recheck")
        assert unchanged.status == "paused"
        assert unchanged.pause.request_id == result.pause.request_id
        session.update_tool_policy(approval_mode="full", require_confirmation_for=frozenset({"run_command"}))
        unchanged = await session.resume(result.turn_id, "recheck")
        assert unchanged.status == "paused"
        [operation] = session.store.list_tool_operations(result.turn_id)
        assert operation.attempt_count == 0
        # The refreshed request can now be explicitly approved exactly once.
        result = await session.resume(result.turn_id, "approve")
        assert result.status == "done"


@pytest.mark.anyio
async def test_recheck_honors_current_explicit_denial(tmp_path):
    async with await command_session(tmp_path, mode="ask") as session:
        result = await session.submit("run")
        session.update_tool_policy(approval_mode="full", denied_tool_names=frozenset({"run_command"}))
        result = await session.resume(result.turn_id, "recheck")
        assert result.status == "done"
        [operation] = session.store.list_tool_operations(result.turn_id)
        assert operation.status == "denied"
        assert operation.attempt_count == 0


@pytest.mark.anyio
async def test_recheck_crash_recovery_does_not_become_a_one_time_approval(tmp_path, monkeypatch):
    async with await command_session(tmp_path, mode="ask") as session:
        result = await session.submit("run")
        thread_id = session.thread_id
        session.update_tool_policy(approval_mode="full")

        async def crash(**kwargs):
            raise RuntimeError("crash before execution")

        monkeypatch.setattr(session.tool_orchestrator, "_execute_approved_record", crash)
        with pytest.raises(RuntimeError, match="crash before execution"):
            await session.resume(result.turn_id, "recheck")

    async with await command_session(tmp_path, mode="ask", thread_id=thread_id) as session:
        recovered = await session.resume(result.turn_id, "continue")
        assert recovered.status == "paused"
        assert recovered.pause is not None
        [operation] = session.store.list_tool_operations(result.turn_id)
        assert operation.attempt_count == 0
        # Rechecking under ask still cannot grant the operation.
        assert (await session.resume(result.turn_id, "recheck")).status == "paused"
        assert (await session.resume(result.turn_id, "approve")).status == "done"


@pytest.mark.anyio
async def test_report_python_write_continues_after_selecting_full_and_persists_mode(tmp_path):
    from agent_runtime.harness import Session
    from agent_runtime.tools.builtins.shell import create_execute_python_tool
    from agent_runtime.tools.registry import ToolRegistry
    from tests.agent.harness.test_approval_resume import AcceptAnswer

    workspace = open_workspace(tmp_path, create=True)
    registry = ToolRegistry()
    registry.register(create_execute_python_tool(workspace))
    arguments = {"code": "from pathlib import Path\nPath('report.md').write_text('sales report')",
                 "workspace_write": True, "output_paths": ["report.md"]}
    async with await Session.open(
        database=tmp_path / "report.sqlite3", workspace=tmp_path,
        model=CommandModel(arguments, tool_name="execute_python"), tools=registry.freeze(),
        completion_gate=AcceptAnswer(), approval_mode="auto",
    ) as session:
        result = await session.submit("generate report")
        assert result.status == "paused" and not (tmp_path / "report.md").exists()
        session.update_tool_policy(approval_mode="full")
        result = await session.resume(result.turn_id, "recheck")
        assert result.status == "done"
        assert (tmp_path / "report.md").read_text() == "sales report"
        [operation] = session.store.list_tool_operations(result.turn_id)
        assert operation.status == "succeeded" and operation.attempt_count == 1
        settings = session.store.read_thread(session.thread_id).settings
        assert settings["tool_execution_policy"]["approval_mode"] == "full"


@pytest.mark.anyio
async def test_repeated_crashes_recover_only_the_current_approval(tmp_path, monkeypatch):
    async def crash(**kwargs):
        raise RuntimeError("crash before execution")

    async with await command_session(tmp_path, mode="ask") as session:
        result = await session.submit("run")
        thread_id = session.thread_id
        session.update_tool_policy(approval_mode="full")
        monkeypatch.setattr(session.tool_orchestrator, "_execute_approved_record", crash)
        with pytest.raises(RuntimeError, match="crash before execution"):
            await session.resume(result.turn_id, "recheck")

    async with await command_session(tmp_path, mode="ask", thread_id=thread_id) as session:
        result = await session.resume(result.turn_id, "continue")
        assert result.status == "paused"
        monkeypatch.setattr(session.tool_orchestrator, "_execute_approved_record", crash)
        with pytest.raises(RuntimeError, match="crash before execution"):
            await session.resume(result.turn_id, "approve")

    async with await command_session(tmp_path, mode="ask", thread_id=thread_id) as session:
        result = await session.resume(result.turn_id, "continue")
        assert result.status == "done"
        [operation] = session.store.list_tool_operations(result.turn_id)
        assert operation.status == "succeeded" and operation.attempt_count == 1


@pytest.mark.anyio
async def test_repeated_mode_changes_keep_network_approval_pending(tmp_path):
    async with await command_session(tmp_path, mode="ask", network=True) as session:
        result = await session.submit("run")
        for mode in ("auto", "ask", "auto"):
            session.update_tool_policy(approval_mode=mode)
            result = await session.resume(result.turn_id, "recheck")
            assert result.status == "paused" and result.pause is not None
            [operation] = session.store.list_tool_operations(result.turn_id)
            assert operation.attempt_count == 0
        session.update_tool_policy(approval_mode="full")
        result = await session.resume(result.turn_id, "recheck")
        assert result.status == "done"
        before = session.store.projection_hashes()
        session.store.rebuild_projections()
        assert session.store.projection_hashes() == before


@pytest.mark.anyio
async def test_single_run_mode_picker_reaches_public_agent_resume(tmp_path, monkeypatch):
    from agent_runtime import Agent, cli

    choices = iter(["permissions", "full"])

    class Dialog:
        async def run_async(self):
            return next(choices)

    monkeypatch.setattr(cli, "_is_interactive_terminal", lambda: True)
    monkeypatch.setattr("prompt_toolkit.shortcuts.radiolist_dialog", lambda **kwargs: Dialog())
    agent = Agent(workspace_path=tmp_path, checkpoint_db=tmp_path / "run.sqlite3",
                  model_session_path=None, enable_workspace_mcp=False, approval_mode="auto")
    agent._model_factory = lambda: CommandModel({
        "code": "from pathlib import Path\nPath('report.md').write_text('sales report')",
        "workspace_write": True, "output_paths": ["report.md"],
    }, tool_name="execute_python")
    result = await cli._run_facade_command(agent, task="generate report", files=(),
                                         max_tokens_total=None, interactive_approval=True)
    assert result.status == "done"
    assert (tmp_path / "report.md").read_text() == "sales report"
    assert agent.approval_mode == "full"


@pytest.mark.anyio
@pytest.mark.parametrize("no_permissions", [False, True])
async def test_recheck_then_network_approval_reuses_operation_without_early_execution(tmp_path, no_permissions):
    async with await command_session(tmp_path, mode="ask", network=True) as session:
        if no_permissions:
            session.tool_execution_context = replace(session.tool_execution_context,
                                                     permissions=None, permission_provider=None)
            session._apply_tool_context()
        result = await session.submit("run")
        [original] = session.store.list_tool_operations(result.turn_id)
        session.update_tool_policy(approval_mode="auto")
        result = await session.resume(result.turn_id, "recheck")
        if result.pause.context["approval_scope"] == "tool":
            result = await session.resume(result.turn_id, "approve")
        assert result.status == "paused" and result.pause.context["approval_scope"] == "network"
        [waiting] = session.store.list_tool_operations(result.turn_id)
        assert waiting.operation_id == original.operation_id and waiting.attempt_count == 0
        result = await session.resume(result.turn_id, "approve")
        assert result.status == "done"
        [completed] = session.store.list_tool_operations(result.turn_id)
        assert completed.status == "succeeded" and completed.attempt_count == 1


@pytest.mark.anyio
async def test_network_scope_can_be_denied_without_execution(tmp_path):
    async with await command_session(tmp_path, mode="ask", network=True) as session:
        result = await session.submit("run")
        result = await session.resume(result.turn_id, "approve")
        assert result.pause.context["approval_scope"] == "network"
        result = await session.resume(result.turn_id, "deny")
        assert result.status == "done"
        [operation] = session.store.list_tool_operations(result.turn_id)
        assert operation.status == "denied" and operation.attempt_count == 0


@pytest.mark.anyio
@pytest.mark.parametrize("crash_scope", ["tool", "network"])
async def test_network_scope_approval_survives_crash_without_duplicate_execution(tmp_path, monkeypatch, crash_scope):
    async with await command_session(tmp_path, mode="ask", network=True) as session:
        result = await session.submit("run")
        thread_id = session.thread_id
        if crash_scope == "network":
            result = await session.resume(result.turn_id, "approve")
            assert result.pause.context["approval_scope"] == "network"

        async def crash(**kwargs):
            raise RuntimeError("crash before execution")

        monkeypatch.setattr(session.tool_orchestrator, "_execute_approved_record", crash)
        with pytest.raises(RuntimeError, match="crash before execution"):
            await session.resume(result.turn_id, "approve")

    async with await command_session(tmp_path, mode="ask", thread_id=thread_id, network=True) as session:
        result = await session.resume(result.turn_id, "continue")
        if crash_scope == "tool":
            assert result.status == "paused" and result.pause.context["approval_scope"] == "network"
            result = await session.resume(result.turn_id, "approve")
        assert result.status == "done"
        [operation] = session.store.list_tool_operations(result.turn_id)
        assert operation.status == "succeeded" and operation.attempt_count == 1


def test_sdk_accepts_mode_and_cli_exposes_modes_for_all_entrypoints(tmp_path):
    from typer.main import get_command

    from agent_runtime import Agent, cli

    assert Agent(workspace_path=tmp_path, approval_mode="auto").approval_mode == "auto"
    group = get_command(cli.agent_app)
    for name in ("run", "chat", "resume", "recover"):
        assert any("--approval-mode" in parameter.opts for parameter in group.commands[name].params)


@pytest.mark.anyio
async def test_permissions_picker_changes_mode_without_loading_model(tmp_path):
    from agent_runtime import cli
    from agent_runtime.terminal_app import TerminalChatApp
    from tests.agent.test_cli_chat_commands import _model_spec, _SessionFactory

    class Facade(_SessionFactory):
        workspace_path = tmp_path
        approval_mode = None

        def current_model(self):
            return _model_spec("test")

        async def submit(self, *args, **kwargs):
            pytest.fail("mode selection must not dispatch a model request")

    class UI(TerminalChatApp):
        commands = iter(["/permissions", "/exit"])

        async def prompt(self, **kwargs):
            return next(self.commands)

        async def choose_menu(self, choices, *, title, current=""):
            assert len(choices) == 3
            assert any("帮我批准" in choice for choice in choices)
            assert "审批" in title
            return choices[0]

    facade = Facade()
    await cli._chat_facade_loop(facade, max_tokens_total=None, _ui=UI(model="test", workspace=str(tmp_path)))
    assert facade.approval_mode == "ask"
    assert facade.opened == 0


@pytest.mark.anyio
async def test_chat_f4_callback_updates_live_session_and_rechecks_without_extra_approval(tmp_path):
    from agent_runtime import Agent, cli
    from agent_runtime.terminal_app import TerminalChatApp

    agent = Agent(workspace_path=tmp_path, checkpoint_db=tmp_path / "chat.sqlite3",
                  model_session_path=None, enable_workspace_mcp=False, approval_mode="auto")
    agent._model_factory = lambda: CommandModel({"command": "printf hello", "workspace_write": True})

    class UI(TerminalChatApp):
        commands = iter(["run", "/exit"])
        approvals = 0

        async def prompt(self, **kwargs):
            return next(self.commands)

        async def choose_menu(self, choices, *, title, current=""):
            assert current == cli.APPROVAL_MODES["auto"]
            return cli.APPROVAL_MODES["full"]

        async def choose_approval(self, request):
            self.approvals += 1
            assert self.approvals == 1
            assert await self.on_permissions()
            return "recheck"

    ui = UI(model="test", workspace=str(tmp_path))
    await cli._chat_facade_loop(agent, max_tokens_total=None, _ui=ui)
    assert ui.approvals == 1
    assert ui.approval_mode == agent.approval_mode == "full"
    from agent_runtime.harness import RuntimeState

    async with (await RuntimeState.open(agent.checkpoint_db, read_only=True)) as store:
        turns = store.list_turns()
        [operation] = store.list_tool_operations(turns[0].turn_id)
        assert operation.status == "succeeded" and operation.attempt_count == 1


@pytest.mark.anyio
@pytest.mark.parametrize("before_reopen", [[], ["/permissions mode full"]])
async def test_chat_reopen_does_not_inherit_another_chats_mode(tmp_path, monkeypatch, before_reopen):
    from agent_runtime import cli
    from agent_runtime.harness import RuntimeState
    from tests.agent.test_agent_cli_resume import _result
    from tests.agent.test_cli_chat_commands import _model_spec, _SessionFactory

    database = tmp_path / "chat.sqlite3"
    async with (await RuntimeState.open(database)) as store:
        thread = store.create_thread(workspace=tmp_path)
        turn = store.start_turn(thread_id=thread.thread_id, user_message="saved ask chat",
                                binding_manifest={"model_id": None})
        store.complete_turn(turn_id=turn.turn_id, answer="saved")
        store.update_session_settings(thread.thread_id, {"tool_execution_policy": {"approval_mode": "ask"}})

    class Facade(_SessionFactory):
        workspace_path = tmp_path
        checkpoint_db = database
        knowledge = None
        approval_mode = None

        def current_model(self):
            return _model_spec("test")

        async def submit(self, *args, **kwargs):
            assert self.options.get("approval_mode") is None
            assert self.approval_mode is None
            return _result(turn_id=turn.turn_id, answer="done")

    commands = iter([*before_reopen, f"/resume {thread.thread_id}", "go", "/exit"])
    monkeypatch.setattr("builtins.input", lambda _: next(commands))
    facade = Facade()
    await cli._chat_facade_loop(facade, max_tokens_total=None)
    assert facade.opened == 1
