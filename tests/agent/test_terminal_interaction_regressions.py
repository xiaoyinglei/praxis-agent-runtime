from __future__ import annotations

import asyncio
import io
from contextlib import suppress

import pytest
from prompt_toolkit.data_structures import Size
from prompt_toolkit.input import create_pipe_input
from prompt_toolkit.output.vt100 import Vt100_Output

from agent_runtime.result import AgentUsage
from agent_runtime.terminal_app import TerminalChatApp
from agent_runtime.terminal_render import format_usage_summary, web_error_message
from agent_runtime.terminal_ui import Conversation, ConversationEventDisplay
from tests.agent.test_terminal_presentation import result


@pytest.mark.anyio
@pytest.mark.parametrize("navigation", ["wheel", "page"])
async def test_fullscreen_scrolls_the_viewport_immediately_while_composing(navigation):
    with create_pipe_input() as pipe:
        terminal = io.StringIO()
        ui = TerminalChatApp(
            model="test", workspace="/tmp", ui_mode="fullscreen", input=pipe,
            output=Vt100_Output(terminal, lambda: Size(rows=24, columns=90), enable_cpr=False),
        )
        ui.view.message("\n".join(f"history row {index:03d}" for index in range(100)), "notice")
        task = asyncio.create_task(ui.run(ui.prompt))
        try:
            async with asyncio.timeout(3):
                while not ui.waiting or ui.app.renderer._last_screen is None:
                    await asyncio.sleep(0.01)
                ui.editor.text = "unfinished 中文 draft"
                ui.invalidate()
                await asyncio.sleep(0.08)
                window = next(window for window in ui.app.layout.find_all_windows() if window.content is ui.body)
                # Use the actual rendered conversation viewport, not a cursor counter.
                original = window.vertical_scroll
                assert original > 0
                if navigation == "wheel":
                    assert ui.app.mouse_support(), "fullscreen has no native terminal scrollback"
                    assert "\x1b[?1006h" in terminal.getvalue()
                    pipe.send_text("\x1b[<64;3;4M")
                else:
                    pipe.send_text("\x1b[5~")
                while window.vertical_scroll == original:
                    await asyncio.sleep(0.01)
                earlier = window.vertical_scroll
                assert earlier < original
                assert ui.app.layout.has_focus(ui.editor)
                assert ui.editor.text == "unfinished 中文 draft"
                ui.view.message("\nnew streamed output", "notice")
                await asyncio.sleep(0.08)
                assert window.vertical_scroll == earlier, "incoming output must not jump back to the bottom"
                pipe.send_text("\x1b[<65;3;4M" if navigation == "wheel" else "\x1b[6~")
                while window.vertical_scroll == earlier:
                    await asyncio.sleep(0.01)
                assert window.vertical_scroll > earlier
                pipe.send_text("\x1b[6~")
                while not ui._follow:
                    await asyncio.sleep(0.01)
                latest = window.vertical_scroll
                ui.view.message("\nfollowed output", "notice")
                while window.vertical_scroll == latest:
                    await asyncio.sleep(0.01)
                assert window.vertical_scroll > latest
        finally:
            task.cancel()
            with suppress(asyncio.CancelledError):
                await task


@pytest.mark.parametrize("status", ["done", "paused"])
def test_normal_result_shows_usage_and_cache_without_an_unrequested_acceptance_notice(status):
    view = Conversation()
    display = ConversationEventDisplay(view)
    display.begin_turn()
    value = result(status=status, usage=AgentUsage(
        input_tokens=1000, output_tokens=200, total_tokens=1200, cache_read_input_tokens=800,
    ))
    display.present_result(value)
    display.present_result(value)
    text = view.plain_text()
    assert "输入 1,000" in text and "输出 200" in text and "总计 1,200" in text
    assert "缓存命中 800" in text and "80.0%" in text
    assert text.count("输入 1,000") == 1
    assert "未运行验收" not in text and "本轮已结束" not in text


def test_missing_cache_usage_is_reported_as_unknown_instead_of_zero_or_hidden():
    assert "缓存命中 未报告" in format_usage_summary(AgentUsage(input_tokens=10, total_tokens=10))
    reported = format_usage_summary(AgentUsage(input_tokens=10, total_tokens=10, cache_read_input_tokens=0))
    assert "缓存命中 0" in reported and "0.0%" in reported


def test_resumed_long_answer_keeps_final_usage_and_validation_at_the_bottom():
    view = Conversation()
    display = ConversationEventDisplay(view)
    display.begin_turn()
    display.present_result(result(status="paused", usage=AgentUsage(input_tokens=1000, total_tokens=1000)))
    pause_receipt = view.blocks[-1]
    display.begin_turn(reset=False)
    view.message("\n".join(f"resumed answer {line}" for line in range(100)))
    display.present_result(result(
        usage=AgentUsage(input_tokens=2000, output_tokens=100, total_tokens=2100),
        validation_status="passed", validation_scope="acceptance_command",
    ))
    assert view.blocks[-1].kind == "receipt"
    assert "输入 2,000" in view.blocks[-1].text and "已验证" in view.blocks[-1].text
    assert "输入 1,000" in pause_receipt.text and "输入 2,000" not in pause_receipt.text
    assert "暂停时" in pause_receipt.text


@pytest.mark.anyio
@pytest.mark.parametrize("mode", ["inline", "fullscreen"])
async def test_real_chat_loop_outputs_turn_usage_without_verbose_or_acceptance_noise(tmp_path, monkeypatch, mode):
    from agent_runtime import Agent
    from agent_runtime.cli import _chat_facade_loop
    from agent_runtime.harness import HarnessModelResponse
    from tests.agent.harness.test_steering import SteeringModel

    class CachedAnswer(SteeringModel):
        async def dispatch(self, prepared, *, event_sink=None):
            return HarnessModelResponse.from_parts(text="a normal chat answer", provider_response_id=None, usage={
                "input_tokens": 1000, "output_tokens": 200, "total_tokens": 1200, "cache_read_input_tokens": 800,
            })

    model = CachedAnswer()
    agent = Agent(checkpoint_db=tmp_path / "run.db", workspace_path=tmp_path, enable_workspace_mcp=False)
    monkeypatch.setattr(agent, "_model_factory", lambda: model)
    with create_pipe_input() as pipe:
        ui = TerminalChatApp(
            model="test", workspace=str(tmp_path), ui_mode=mode, input=pipe,
            output=Vt100_Output(io.StringIO(), lambda: Size(rows=24, columns=90), enable_cpr=False),
        )
        task = asyncio.create_task(ui.run(lambda: _chat_facade_loop(agent, max_tokens_total=None, _ui=ui)))
        try:
            async with asyncio.timeout(5):
                while not ui.waiting:
                    await asyncio.sleep(0.01)
                pipe.send_text("hello\r")
                while "本轮累计 Token" not in ui.view.plain_text() or not ui.waiting:
                    await asyncio.sleep(0.01)
                text = ui.view.plain_text()
                assert "输入 1,000" in text and "输出 200" in text and "总计 1,200" in text
                assert "缓存命中 800" in text and "80.0%" in text
                assert "未运行验收" not in text and "本轮已结束" not in text
                assert not ui.view.details
                pipe.send_text("/exit\r")
                await task
        finally:
            if not task.done():
                task.cancel()
                with suppress(asyncio.CancelledError):
                    await task


def test_search_configuration_and_browser_capability_errors_are_actionable():
    assert "未配置" in web_error_message("web_search_not_configured")
    assert "渲染" in web_error_message("web_browser_unavailable")
    assert "web_search_not_configured" not in web_error_message("web_search_not_configured")


@pytest.mark.anyio
async def test_visible_search_setup_shortcut_preserves_the_input_draft():
    with create_pipe_input() as pipe:
        output = Vt100_Output(io.StringIO(), lambda: Size(rows=24, columns=100), enable_cpr=False)
        ui = TerminalChatApp(model="test", workspace="/tmp", ui_mode="fullscreen", input=pipe, output=output)
        ui.display.search_configuration_required = True
        commands = []

        async def command():
            commands.append(await ui.prompt())

        task = asyncio.create_task(ui.run(command))
        try:
            async with asyncio.timeout(3):
                while not ui.waiting or ui.app.renderer._last_screen is None:
                    await asyncio.sleep(0.01)
                screen = ui.app.renderer._last_screen
                text = "\n".join("".join(cell.char for cell in row.values()) for row in screen.data_buffer.values())
                assert "设置搜索" in text and "F5" in text
                ui.editor.text = "尚未发送的草稿"
                pipe.send_text("\x1b[15~")
                await task
                assert commands == ["/web"]
                assert ui.editor.text == "尚未发送的草稿"
        finally:
            if not task.done():
                task.cancel()
                with suppress(asyncio.CancelledError):
                    await task


@pytest.mark.anyio
@pytest.mark.parametrize("cancel", [False, True])
async def test_native_search_password_input_can_save_or_cancel_without_leaking(tmp_path, cancel):
    from agent_runtime.models import ModelSpec
    from agent_runtime.web_config import resolve_web_configuration

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    secret = "native-password-test-credential"
    outcomes = []
    stream = io.StringIO()
    with create_pipe_input() as pipe:
        output = Vt100_Output(stream, lambda: Size(rows=24, columns=100), enable_cpr=False)
        ui = TerminalChatApp(model="test", workspace=str(workspace), ui_mode="fullscreen", input=pipe, output=output)
        ui.editor.text = "保留这份草稿"

        async def configure_then_chat():
            model = ModelSpec(id="test", provider="local", context_window=4096,
                              supports_tools=True, supports_structured_output=True, location="local")
            try:
                await ui.setup_search(model)
                outcomes.append("saved")
            except KeyboardInterrupt:
                outcomes.append("cancelled")
            assert await ui.prompt() == "保留这份草稿"

        task = asyncio.create_task(ui.run(configure_then_chat))
        try:
            async with asyncio.timeout(5):
                while "API Key:" not in stream.getvalue():
                    await asyncio.sleep(0.01)
                pipe.send_text("\x03" if cancel else secret + "\r")
                while not ui.waiting:
                    await asyncio.sleep(0.01)
                assert ui.editor.text == "保留这份草稿"
                pipe.send_text("\r")
                await task
            assert outcomes == (["cancelled"] if cancel else ["saved"])
            assert resolve_web_configuration().provider == (None if cancel else "deepseek")
            assert secret not in stream.getvalue()
            assert secret not in ui.view.plain_text()
        finally:
            if not task.done():
                task.cancel()
                with suppress(asyncio.CancelledError):
                    await task
