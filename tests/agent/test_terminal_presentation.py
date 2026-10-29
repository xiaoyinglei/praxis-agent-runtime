from __future__ import annotations

from dataclasses import replace

import pytest

from agent_runtime.result import AgentResult, AgentUsage
from agent_runtime.streaming.events import (
    EventType,
    ItemDeltaKind,
    ItemStatus,
    StreamEvent,
    TurnItemKind,
    item_completed,
    item_delta,
    item_started,
)
from agent_runtime.terminal_ui import Conversation, ConversationEventDisplay


def result(**changes):
    value = AgentResult(
        answer=None,
        status="done",
        files=(),
        tool_calls=(),
        evidence=(),
        citations=(),
        usage=AgentUsage(),
        diagnostics=(),
        turn_id="t",
        stop_reason=None,
        pause=None,
        workspace_path=None,
        groundedness=False,
        insufficient_evidence=False,
        plan=None,
        plan_events=(),
    )
    return replace(value, **changes)


async def complete(
    display, identity, *, name="read_file", status=ItemStatus.SUCCESS, output="read complete", error=None
):
    await display.emit(
        item_completed(
            turn_id="t",
            item_id=identity,
            item_kind=TurnItemKind.TOOL,
            status=status,
            error=error,
            data={"result": {"tool_name": name, "structured_content": output}},
        )
    )


@pytest.mark.anyio
async def test_compact_operations_have_two_line_previews_and_full_details():
    view = Conversation()
    display = ConversationEventDisplay(view)
    display.begin_turn()
    await complete(display, "a", output="first\nsecond\nthird\nfourth")
    assert "执行记录" not in view.plain_text()
    assert "first" in view.plain_text() and "third" not in view.plain_text()
    display.set_verbose(True)
    assert "third" in view.plain_text() and "fourth" in view.plain_text()


@pytest.mark.anyio
async def test_compact_tail_keeps_problems_without_duplicating_errors():
    view = Conversation()
    display = ConversationEventDisplay(view)
    display.begin_turn()
    await complete(display, "bad", status=ItemStatus.FAILED, error="missing-file", output=None)
    for i in range(7):
        await complete(display, str(i), output=f"body-{i}")
    text = view.plain_text()
    assert text.count("missing-file") == 1
    assert "body-0" not in text and "body-6" in text
    assert "另有 2 项操作" in text
    assert "正在恢复" not in display.activity_text()
    await display.emit(StreamEvent(type=EventType.RECOVERY, turn_id="t", data={"strategy": "retry_model"}))
    assert "正在恢复" in display.activity_text()
    display.set_verbose(True)
    assert "body-0" in view.plain_text()


@pytest.mark.anyio
async def test_command_session_success_is_not_process_completion():
    view = Conversation()
    display = ConversationEventDisplay(view)
    display.begin_turn()
    await complete(
        display, "start", name="run_command", output={"status": "running", "session_id": "cmd_1", "exit_code": None}
    )
    first = display.group.children[0]
    assert first.operation_status == "running"
    assert "命令运行中" in view.plain_text() and "cmd_1" not in view.plain_text()
    await complete(
        display, "wait", name="manage_command", output={"status": "completed", "session_id": "cmd_1", "exit_code": 0}
    )
    assert first.operation_status == "success"
    assert "命令运行中" not in view.plain_text()


@pytest.mark.anyio
@pytest.mark.parametrize("status,label", [(ItemStatus.CANCELLED, "已取消"), (ItemStatus.OUTCOME_UNKNOWN, "结果未确认")])
async def test_cancelled_and_unknown_are_distinct(status, label):
    view = Conversation()
    display = ConversationEventDisplay(view)
    display.begin_turn()
    await complete(display, "one", status=status, error="process outcome", output=None)
    assert label in view.plain_text()
    if status == ItemStatus.OUTCOME_UNKNOWN:
        assert "已停止" not in view.plain_text()


@pytest.mark.anyio
async def test_private_reasoning_activity_and_silence_use_actual_clock():
    now = [100.0]
    view = Conversation()
    display = ConversationEventDisplay(view, clock=lambda: now[0])
    display.begin_turn()
    assert "等待模型响应" in display.activity_text()
    await display.emit(
        item_delta(
            turn_id="t",
            item_id="r",
            item_kind=TurnItemKind.REASONING,
            delta_kind=ItemDeltaKind.REASONING,
            delta="private reasoning",
        )
    )
    assert "思考中" in display.activity_text() and "private reasoning" not in view.plain_text()
    now[0] += 30
    assert "距上次活动 30 秒" in display.activity_text()
    assert "失败" not in display.activity_text()


@pytest.mark.anyio
async def test_conversation_order_is_not_rewritten_by_later_operations():
    view = Conversation()
    display = ConversationEventDisplay(view)
    display.begin_turn()
    await complete(display, "first", output="first operation")
    await display.emit(
        item_delta(
            turn_id="t",
            item_id="a",
            item_kind=TurnItemKind.AGENT_MESSAGE,
            delta_kind=ItemDeltaKind.TEXT,
            delta="assistant update",
        )
    )
    await complete(display, "second", output="second operation")
    text = view.plain_text()
    assert text.index("first operation") < text.index("assistant update") < text.index("second operation")


def test_result_receipt_never_turns_done_into_verified_or_claims_file_exists():
    view = Conversation()
    display = ConversationEventDisplay(view)
    display.begin_turn()
    display.present_result(result(files=("/not-existing/report.docx",)))
    assert "未运行验收" not in view.plain_text()
    assert "本轮累计 Token" in view.plain_text()
    assert "已验证" not in view.plain_text() and "已生成" not in view.plain_text()
    display.present_result(
        result(files=("/not-existing/report.docx",), validation_status="passed", validation_scope="acceptance_command")
    )
    text = view.plain_text()
    assert "已验证" in text and "用户指定的验收命令" in text
    assert text.count("report.docx") == 1


@pytest.mark.anyio
async def test_parallel_start_and_failure_keeps_other_work_running():
    view = Conversation()
    display = ConversationEventDisplay(view)
    display.begin_turn()
    for identity in ("one", "two"):
        await display.emit(
            item_started(turn_id="t", item_id=identity, item_kind=TurnItemKind.TOOL, data={"tool_name": "read_file"})
        )
    assert "另有 1 项进行中" in display.activity_text()
    await complete(display, "one", status=ItemStatus.FAILED, error="failed one", output=None)
    assert "进行中" in display.activity_text()
    assert "正在恢复" not in display.activity_text()


def test_technical_accounting_is_available_only_in_detail_view():
    from prompt_toolkit.output import DummyOutput

    from agent_runtime.terminal_app import TerminalChatApp

    ui = TerminalChatApp(model="demo", workspace="/tmp", output=DummyOutput())
    ui.display.update_usage(AgentUsage(input_tokens=10, total_tokens=10, usage_complete=False))
    assert ui._status_fragments() == []
    ui.display.set_verbose(True)
    technical = "".join(t for _, t in ui._status_fragments())
    assert "demo" in technical and "不完整" in technical and "缓存" in technical
    ui.display.set_verbose(False)
    assert ui._status_fragments() == []


@pytest.mark.anyio
async def test_details_roundtrip_preserves_draft_cursor_and_escape_precedence():
    import asyncio
    from contextlib import suppress

    from prompt_toolkit.input import create_pipe_input
    from prompt_toolkit.output import DummyOutput

    from agent_runtime.terminal_app import TerminalChatApp

    with create_pipe_input() as pipe:
        ui = TerminalChatApp(model="demo", workspace="/tmp", input=pipe, output=DummyOutput())
        ui.display.begin_turn()
        await complete(ui.display, "one", output="short\nother\nlong detail")
        ready = asyncio.Event()
        cancelled = []

        async def conversation():
            ready.set()
            try:
                await ui.operation(asyncio.Event().wait())
            except asyncio.CancelledError:
                cancelled.append(True)

        task = asyncio.create_task(ui.run(conversation))
        try:
            await ready.wait()
            ui.editor.text = "中文 draft"
            ui.editor.buffer.cursor_position = 2
            before = (ui._row, ui._follow, ui._show_history)
            pipe.send_text("\x0f")
            await asyncio.sleep(0.08)
            assert ui.view.details and "long detail" in ui.view.plain_text()
            pipe.send_text("\x1b")
            await asyncio.sleep(0.6)
            assert not ui.view.details and not cancelled
            assert ui.editor.text == "中文 draft" and ui.editor.buffer.cursor_position == 2
            assert (ui._row, ui._follow, ui._show_history) == before
            pipe.send_text("\x1b")
            await asyncio.wait_for(task, 2)
            assert cancelled == [True]
        finally:
            if not task.done():
                task.cancel()
                with suppress(asyncio.CancelledError):
                    await task


def test_inline_welcome_is_replaced_by_conversation_and_fullscreen_is_one_line():
    from prompt_toolkit.output import DummyOutput

    from agent_runtime.terminal_app import TerminalChatApp

    ui = TerminalChatApp(model="demo", workspace="/tmp", output=DummyOutput())
    assert "Praxis" in "".join(t for _, t in ui._header())
    ui.view.message("question", "user")
    assert ui._header() == []
    ui.fullscreen = True
    header = "".join(t for _, t in ui._header())
    assert "Praxis" in header and header.count("\n") <= 1


@pytest.mark.anyio
async def test_cli_cancel_waits_for_abort_and_passes_events_before_confirming():
    import asyncio

    from prompt_toolkit.output import DummyOutput

    from agent_runtime import cli
    from agent_runtime.terminal_app import TerminalChatApp

    ui = TerminalChatApp(model="demo", workspace="/tmp", output=DummyOutput())
    ui.display.begin_turn()
    await ui.display.emit(
        item_started(turn_id="t", item_id="cmd", item_kind=TurnItemKind.COMMAND, data={"tool_name": "run_command"})
    )
    gate = asyncio.Event()
    started = asyncio.Event()

    class Driver:
        async def resume(self, turn_id, action, *, event_sink):
            assert turn_id == "t" and action == "abort" and event_sink is ui.display
            started.set()
            await gate.wait()
            return result(status="failed", stop_reason="cancelled")

    pending = asyncio.create_task(cli._cancel_ui_execution(Driver(), ui, "t"))
    await started.wait()
    assert "正在停止" in ui.display.activity_text()
    assert "已停止" not in ui.view.plain_text()
    gate.set()
    await pending
    assert "已停止" in ui.view.plain_text()
    assert "结果未确认" not in ui.view.plain_text()


@pytest.mark.anyio
async def test_cli_abort_failure_preserves_unknown_operation_result():
    from prompt_toolkit.output import DummyOutput

    from agent_runtime import cli
    from agent_runtime.terminal_app import TerminalChatApp

    ui = TerminalChatApp(model="demo", workspace="/tmp", output=DummyOutput())
    ui.display.begin_turn()
    await ui.display.emit(
        item_started(turn_id="t", item_id="cmd", item_kind=TurnItemKind.COMMAND, data={"tool_name": "run_command"})
    )

    class Driver:
        async def resume(self, *args, **kwargs):
            raise RuntimeError("unknown tool operation")

    with pytest.raises(RuntimeError, match="unknown tool operation"):
        await cli._cancel_ui_execution(Driver(), ui, "t")
    assert "结果未确认" in ui.view.plain_text()
    assert "已停止" not in ui.view.plain_text()


def test_recorded_workspace_change_does_not_claim_task_validation():
    view = Conversation()
    display = ConversationEventDisplay(view)
    display.present_result(result(validation_status="passed", validation_scope="workspace_change_recorded"))
    assert "已验证" not in view.plain_text()
    assert "已记录工作区修改" in view.plain_text()
    assert "未运行任务验收" not in view.plain_text()


@pytest.mark.anyio
async def test_manage_running_session_is_one_process_in_activity_count():
    view = Conversation()
    display = ConversationEventDisplay(view)
    display.begin_turn()
    for identity, name in (("start", "run_command"), ("poll", "manage_command")):
        await complete(display, identity, name=name, output={"status": "running", "session_id": "cmd_1"})
    assert "另有 1 项进行中" not in display.activity_text()


@pytest.mark.anyio
async def test_detail_visibility_does_not_duplicate_incoming_command_output():
    view = Conversation()
    display = ConversationEventDisplay(view)
    display.begin_turn()
    await display.emit(
        item_started(turn_id="t", item_id="cmd", item_kind=TurnItemKind.COMMAND, data={"tool_name": "run_command"})
    )
    for verbose, text in ((False, "before"), (True, "during"), (False, "after")):
        display.set_verbose(verbose)
        await display.emit(
            item_delta(
                turn_id="t",
                item_id="cmd",
                item_kind=TurnItemKind.COMMAND,
                delta_kind=ItemDeltaKind.COMMAND_STDOUT,
                delta=text + "\n",
            )
        )
    content = display.group.children[0].text
    for text in ("before", "during", "after"):
        assert content.count(text) == 1


@pytest.mark.anyio
async def test_escape_closes_details_before_resolving_approval():
    import asyncio
    from contextlib import suppress

    from prompt_toolkit.input import create_pipe_input
    from prompt_toolkit.output import DummyOutput

    from agent_runtime.result import AgentPause
    from agent_runtime.terminal_app import TerminalChatApp

    with create_pipe_input() as pipe:
        ui = TerminalChatApp(model="demo", workspace="/tmp", input=pipe, output=DummyOutput())
        request = AgentPause(
            request_id="confirm", kind="tool_approval", question="允许写文件？", options=("allow_once", "deny")
        )
        selected = []

        async def conversation():
            selected.append(await ui.choose_approval(request))

        task = asyncio.create_task(ui.run(conversation))
        try:
            async with asyncio.timeout(3):
                while ui._approval is None:
                    await asyncio.sleep(0.01)
            pipe.send_text("\x0f")
            await asyncio.sleep(0.05)
            assert ui.view.details
            pipe.send_text("\x1b")
            async with asyncio.timeout(3):
                while ui.view.details and not selected:
                    await asyncio.sleep(0.01)
            assert not ui.view.details and not selected
            assert ui._approval is not None and not ui._approval.future.done()
            pipe.send_text("\x1b")
            await asyncio.wait_for(task, 2)
            assert selected == [None]
        finally:
            if not task.done():
                task.cancel()
                with suppress(asyncio.CancelledError):
                    await task


@pytest.mark.anyio
async def test_later_turn_terminal_session_result_updates_prior_running_preview():
    view = Conversation()
    display = ConversationEventDisplay(view)
    display.begin_turn()
    await complete(display, "start", name="run_command", output={"status": "running", "session_id": "cmd_1"})
    previous = display.group.children[0]
    display.finish()
    display.begin_turn()
    await complete(
        display, "wait", name="manage_command", output={"status": "completed", "session_id": "cmd_1", "exit_code": 0}
    )
    assert previous.operation_status == "success"


@pytest.mark.anyio
async def test_business_status_fields_do_not_override_tool_lifecycle():
    view = Conversation()
    display = ConversationEventDisplay(view)
    display.begin_turn()
    await complete(display, "business", name="mcp.orders.get", output={"status": "cancelled", "session_id": "order"})
    assert display.group.children[0].operation_status == "success"
    assert "已取消" not in view.plain_text()


@pytest.mark.anyio
async def test_committed_tool_cancellation_is_readable_even_with_failed_item_status():
    view = Conversation()
    display = ConversationEventDisplay(view)
    display.begin_turn()
    await display.emit(
        item_completed(
            turn_id="t",
            item_id="cmd",
            item_kind=TurnItemKind.COMMAND,
            status=ItemStatus.FAILED,
            error="cancelled",
            data={"result": {"tool_name": "run_command", "error_code": "cancelled"}},
        )
    )
    assert "已取消" in view.plain_text()
    assert "继续处理中" not in display.activity_text()
