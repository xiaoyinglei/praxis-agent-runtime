from __future__ import annotations

import asyncio
import io
from contextlib import suppress

import pytest
from prompt_toolkit.data_structures import Size
from prompt_toolkit.input import create_pipe_input
from prompt_toolkit.output.vt100 import Vt100_Output
from wcwidth import wcswidth

from agent_runtime.result import AgentUsage
from agent_runtime.streaming.events import EventType, StreamEvent
from agent_runtime.terminal_app import TerminalChatApp
from agent_runtime.terminal_ui import Conversation, ConversationEventDisplay


def attempt(identity, status="completed", **usage):
    return StreamEvent(type=EventType.MODEL_ATTEMPT_UPDATED, turn_id="t", data={
        "operation_id": "op-" + identity, "attempt_id": identity,
        "generation": 1, "status": status, "usage": usage,
    })


@pytest.mark.anyio
async def test_committed_attempt_usage_is_visible_before_the_final_result_and_deduplicated():
    display = ConversationEventDisplay(Conversation())
    display.begin_turn()
    assert "待报告" in display.status_lines("test")[1]
    await display.emit(attempt("first", input_tokens=1000, output_tokens=200, cache_read_input_tokens=800))
    await display.emit(attempt("first", input_tokens=1000, output_tokens=200, cache_read_input_tokens=800))
    # Summaries are model attempts too: no purpose filter may drop their cost.
    await display.emit(attempt("summary", input_tokens=300, output_tokens=40, cache_read_input_tokens=100))
    usage = display.status_lines("test")[1]
    assert "输入 1,300" in usage and "输出 240" in usage and "总计 1,540" in usage
    assert "缓存 900" in usage
    display.update_usage(AgentUsage(input_tokens=1300, output_tokens=240, total_tokens=1540,
                                    cache_read_input_tokens=900))
    display.finish()
    display.begin_turn(reset=False)
    await display.emit(attempt("first", input_tokens=1000, output_tokens=200, cache_read_input_tokens=800))
    await display.emit(attempt("next", input_tokens=500, output_tokens=60, cache_read_input_tokens=200))
    usage = display.status_lines("test")[1]
    assert "总计 2,100" in usage and "缓存 1,100" in usage
    display.begin_turn()
    assert "待报告" in display.status_lines("test")[1]


@pytest.mark.anyio
async def test_unreported_attempt_usage_and_cache_are_never_presented_as_zero():
    display = ConversationEventDisplay(Conversation())
    display.begin_turn()
    await display.emit(attempt("unknown"))
    assert "待报告" in display.status_lines("test")[1]
    await display.emit(attempt("reported", input_tokens=80, output_tokens=20))
    usage = display.status_lines("test")[1]
    assert "总计 100" in usage and "缓存 未知" in usage and "不完整" in usage


@pytest.mark.anyio
async def test_partial_usage_does_not_invent_missing_output_or_total_counts():
    display = ConversationEventDisplay(Conversation())
    display.begin_turn()
    await display.emit(attempt("partial", input_tokens=80))
    usage = display.status_lines("test")[1]
    assert "输入 80" in usage and "输出 待报告" in usage and "总计 待报告" in usage
    assert "输出 待报告" in display.runtime_stats()
    await display.emit(attempt("complete", input_tokens=100, output_tokens=20))
    usage = display.status_lines("test")[1]
    assert "输入 180" in usage and "总计 120" in usage and "不完整" in usage


@pytest.mark.anyio
async def test_usage_completed_after_a_pause_checkpoint_is_added_once():
    display = ConversationEventDisplay(Conversation())
    display.begin_turn()
    await display.emit(attempt("first", input_tokens=100, output_tokens=20))
    await display.emit(attempt("late", status="unknown"))
    display.update_usage(AgentUsage(input_tokens=100, output_tokens=20, total_tokens=120,
                                    model_calls=2, usage_complete=False))
    display.finish()
    display.begin_turn(reset=False)
    await display.emit(attempt("late", input_tokens=300, output_tokens=40))
    await display.emit(attempt("late", input_tokens=300, output_tokens=40))
    assert "总计 460" in display.status_lines("test")[1]


def test_elapsed_time_excludes_pause_and_resets_for_new_turn():
    now = [0.0]
    display = ConversationEventDisplay(Conversation(), clock=lambda: now[0])
    display.begin_turn()
    now[0] = 12.0
    assert display.runtime_stats().startswith("12 秒")
    display.finish()
    now[0] = 120.0
    assert display.runtime_stats().startswith("12 秒")
    display.begin_turn(reset=False)
    now[0] = 125.0
    assert display.runtime_stats().startswith("17 秒")
    display.begin_turn()
    assert display.runtime_stats().startswith("0 秒")


@pytest.mark.anyio
async def test_activity_bar_places_reported_usage_on_right_and_uses_cell_width():
    now = [0.0]
    columns = [120]
    ui = TerminalChatApp(model="test", workspace="/tmp", output=Vt100_Output(
        io.StringIO(), lambda: Size(rows=24, columns=columns[0]), enable_cpr=False))
    ui.display = ConversationEventDisplay(ui.view, clock=lambda: now[0])
    ui.display.begin_turn()
    await ui.display.emit(attempt("reported", input_tokens=1000, output_tokens=200, cache_read_input_tokens=800))
    now[0] = 12.0
    fragments = ui._activity_fragments()
    text = "".join(value for _, value, *_ in fragments)
    assert "输入 1,000" in text and "输出 200" in text and "缓存 800" in text
    assert text.endswith("缓存 800") and "12 秒" in text
    assert wcswidth(text) == 119
    first = fragments[0][1]
    now[0] += 1 / 6
    assert ui._activity_fragments()[0][1] != first
    for width in (80, 55, 30, 15, 4, 1):
        columns[0] = width
        text = "".join(value for _, value, *_ in ui._activity_fragments())
        assert 0 <= wcswidth(text) <= max(1, width - 1)
        assert "\n" not in text


@pytest.mark.anyio
async def test_active_animation_repaints_without_changing_editor_or_scroll_and_stops_when_idle():
    with create_pipe_input() as pipe:
        ui = TerminalChatApp(model="test", workspace="/tmp", ui_mode="fullscreen", input=pipe,
                             output=Vt100_Output(io.StringIO(), lambda: Size(rows=24, columns=100),
                                                 enable_cpr=False))
        gate = asyncio.Event()
        async def work():
            ui.display.begin_turn()
            await gate.wait()
            ui.display.finish()
        async def run():
            await ui.operation(work())
            await ui.prompt()
        task = asyncio.create_task(ui.run(run))
        try:
            async with asyncio.timeout(4):
                while ui._operation is None or ui.app.renderer._last_screen is None:
                    await asyncio.sleep(0.01)
                ui.editor.text = "unfinished 中文 draft"
                ui.view.message("\n".join(f"history {n}" for n in range(100)), "notice")
                await asyncio.sleep(0.05)
                pipe.send_text("\x1b[5~")
                await asyncio.sleep(0.05)
                position = (ui._row, ui._follow, ui._body_window.vertical_scroll)
                counter = ui.app.render_counter
                await asyncio.sleep(0.55)
                assert 2 <= ui.app.render_counter - counter <= 5
                assert ui.editor.text == "unfinished 中文 draft"
                assert (ui._row, ui._follow, ui._body_window.vertical_scroll) == position
                gate.set()
                while not ui.waiting:
                    await asyncio.sleep(0.01)
                await asyncio.sleep(0.08)
                counter = ui.app.render_counter
                await asyncio.sleep(0.4)
                assert ui.app.render_counter == counter
        finally:
            task.cancel()
            with suppress(asyncio.CancelledError):
                await task
