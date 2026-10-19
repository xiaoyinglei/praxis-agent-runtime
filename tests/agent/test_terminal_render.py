from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
from wcwidth import wcswidth

from agent_runtime.harness import RolloutStore
from agent_runtime.streaming.events import (
    ItemDeltaKind,
    ItemStatus,
    TurnItemKind,
    item_completed,
    item_delta,
    item_started,
    text_delta,
    tool_use_progress,
    tool_use_result,
    tool_use_start,
)
from agent_runtime.terminal_render import (
    BoundedCommandPreview,
    BoundedProgressPreview,
    TerminalToolEventDisplay,
    bounded_result_lines,
    display_rows,
    safe_terminal_text,
)


def test_safe_terminal_text_removes_ansi_and_unsafe_controls() -> None:
    value = "\x1b[31mred\x1b[0m\x00\x07\nnext\tvalue"

    rendered = safe_terminal_text(value)

    assert rendered == "red\nnext\tvalue"


@pytest.mark.anyio
async def test_web_failure_shows_actionable_message_without_raw_payload(capsys):
    display = TerminalToolEventDisplay(width=100)
    await display.emit(item_completed(
        turn_id="t", item_id="s", item_kind=TurnItemKind.TOOL, status=ItemStatus.FAILED,
        error="RAW provider error",
        data={"result": {"tool_name": "web_search", "error_code": "web_search_blocked",
                         "error_message": "RAW provider error", "structured_content": {
                             "error_code": "web_search_blocked", "error_message": "RAW provider error"}}},
    ))
    output = capsys.readouterr().out
    assert "搜索服务暂时限制访问" in output
    assert "RAW provider error" not in output
    assert "error_code" not in output


def test_display_rows_respects_cjk_cell_width() -> None:
    assert display_rows("你好a", width=4) == ["你好", "a"]


def test_display_rows_does_not_split_combining_graphemes() -> None:
    assert display_rows("e\u0301e\u0301", width=1) == ["e\u0301", "e\u0301"]


def test_display_rows_measures_zwj_emoji_as_one_cluster() -> None:
    family = "👨‍👩‍👧‍👦"

    assert display_rows(f"{family}{family}", width=2) == [family, family]


def test_bounded_result_lines_formats_short_json() -> None:
    assert bounded_result_lines({"ok": True}, width=80) == ["{", '  "ok": true', "}"]


def test_bounded_result_lines_keeps_head_and_tail_with_exact_marker() -> None:
    value = "\n".join(f"line {index}" for index in range(12))

    lines = bounded_result_lines(value, width=80, max_rows=8)

    assert lines == [
        "line 0",
        "line 1",
        "line 2",
        "… +5 lines (/verbose 查看完整结果)",
        "line 8",
        "line 9",
        "line 10",
        "line 11",
    ]


def test_bounded_result_lines_verbose_returns_every_row() -> None:
    value = "\n".join(f"line {index}" for index in range(12))

    lines = bounded_result_lines(value, width=80, max_rows=8, verbose=True)

    assert lines == [f"line {index}" for index in range(12)]


def test_bounded_result_lines_keeps_markers_within_narrow_width() -> None:
    value = "\n".join(f"line {index}" for index in range(12))

    lines = bounded_result_lines(value, width=10, max_rows=8)

    assert all(wcswidth(line) <= 10 for line in lines)
    assert any("+" in line for line in lines)


def test_command_preview_preserves_chunk_order_and_equal_deltas() -> None:
    preview = BoundedCommandPreview(width=80)

    visible = [
        *preview.feed("same\n"),
        *preview.feed("same\nerr"),
        *preview.feed("or\nlast\n"),
        *preview.finish(),
    ]

    assert visible == ["same", "same", "error", "last"]


def test_command_preview_treats_split_crlf_as_one_newline() -> None:
    preview = BoundedCommandPreview(width=80)

    visible = [*preview.feed("a\r"), *preview.feed("\nb\n"), *preview.finish()]

    assert visible == ["a", "b"]


def test_command_preview_keeps_six_head_and_three_tail_rows() -> None:
    preview = BoundedCommandPreview(width=80)
    visible: list[str] = []

    for index in range(12):
        visible.extend(preview.feed(f"line {index}\n"))
    visible.extend(preview.finish())

    assert visible == [
        "line 0",
        "line 1",
        "line 2",
        "line 3",
        "line 4",
        "line 5",
        "… +3 lines (/verbose 查看完整结果)",
        "line 9",
        "line 10",
        "line 11",
    ]


def test_command_preview_keeps_markers_within_narrow_width() -> None:
    preview = BoundedCommandPreview(width=8)
    visible: list[str] = []

    for index in range(12):
        visible.extend(preview.feed(f"line {index}\n"))
    visible.extend(preview.finish())

    assert all(wcswidth(line) <= 8 for line in visible)
    assert any("+" in line for line in visible)


def test_command_preview_bounds_a_long_line_and_reports_omitted_characters() -> None:
    preview = BoundedCommandPreview(width=80, max_partial_bytes=16 * 1024)

    visible = [*preview.feed("x" * 100_000)]

    assert visible == []
    assert preview.retained_bytes <= 16 * 1024

    visible.extend(preview.finish())

    assert any("… +83616 chars" in line for line in visible)
    assert visible[0] == "x" * 80
    assert visible[-1] == "x" * 32


def test_progress_preview_limits_messages_and_reports_suppression() -> None:
    preview = BoundedProgressPreview(max_messages=8)

    visible = [preview.feed(f"progress {index}") for index in range(11)]

    assert visible[:8] == [f"progress {index}" for index in range(8)]
    assert visible[8:] == [None, None, None]
    assert preview.finish() == "… +3 progress updates"


@pytest.mark.anyio
async def test_renderer_start_is_best_effort_turn_scoped_and_idempotent(
    capsys: pytest.CaptureFixture[str],
) -> None:
    display = TerminalToolEventDisplay(width=80)
    with_preview = item_started(
        turn_id="turn-a",
        item_id="tool-shared",
        item_kind=TurnItemKind.TOOL,
        data={"tool_name": "read_file", "input_preview": "path='a.py'"},
    )
    without_preview = item_started(
        turn_id="turn-b",
        item_id="tool-shared",
        item_kind=TurnItemKind.TOOL,
        data={"tool_name": "read_file"},
    )
    original = copy.deepcopy(with_preview.data)

    await display.emit(with_preview)
    await display.emit(with_preview)
    await display.emit(without_preview)

    output = capsys.readouterr().out
    assert output.count("→ read_file: path='a.py'") == 1
    assert output.count("→ read_file") == 2
    assert with_preview.data == original


@pytest.mark.anyio
async def test_renderer_projects_default_and_verbose_results_without_mutation(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    payload = {
        "tool_name": "read_file",
        "structured_content": {"lines": [f"line {index}" for index in range(12)]},
        "truncated": False,
        "metadata": {},
    }
    database = tmp_path / "rollout.sqlite3"
    with RolloutStore(database) as store:
        thread = store.create_thread(workspace=tmp_path)
        turn = store.start_turn(
            thread_id=thread.thread_id,
            user_message="read",
            binding_manifest={"model_id": "test-model"},
        )
        stored = store.record_tool_result(
            turn_id=turn.turn_id,
            operation_id=None,
            result=payload,
        )
        durable_before = json.loads(json.dumps(dict(stored.payload)))

    event = item_completed(
        turn_id="turn-default",
        item_id="tool-default",
        item_kind=TurnItemKind.TOOL,
        status=ItemStatus.SUCCESS,
        data={"result": payload},
    )
    event_before = copy.deepcopy(event.data)
    display = TerminalToolEventDisplay(width=80)

    await display.emit(event)

    default_output = capsys.readouterr().out
    assert "✓ read_file" in default_output
    assert "… +" in default_output
    assert "/verbose 查看完整结果" in default_output
    assert event.data == event_before

    verbose_event = item_completed(
        turn_id="turn-verbose",
        item_id="tool-verbose",
        item_kind=TurnItemKind.TOOL,
        status=ItemStatus.SUCCESS,
        data={"result": payload},
    )
    display.set_verbose(True)
    await display.emit(verbose_event)

    verbose_output = capsys.readouterr().out
    assert "line 0" in verbose_output
    assert "line 11" in verbose_output
    assert "/verbose 查看完整结果" not in verbose_output
    with RolloutStore(database) as reopened:
        persisted = next(
            item for item in reopened.list_items(turn.turn_id) if item.item_id == stored.item_id
        )
    assert dict(persisted.payload) == durable_before


@pytest.mark.anyio
async def test_renderer_bounds_lifecycle_keys_and_deduplicates_completion(
    capsys: pytest.CaptureFixture[str],
) -> None:
    display = TerminalToolEventDisplay(width=80, max_lifecycle_keys=256)
    duplicate = item_completed(
        turn_id="turn-0",
        item_id="tool-0",
        item_kind=TurnItemKind.TOOL,
        status=ItemStatus.SUCCESS,
        data={"result": {"tool_name": "read_file", "structured_content": {"ok": True}}},
    )

    await display.emit(duplicate)
    await display.emit(duplicate)
    for index in range(300):
        await display.emit(
            item_started(
                turn_id=f"turn-{index + 1}",
                item_id="shared",
                item_kind=TurnItemKind.TOOL,
                data={"tool_name": "read_file"},
            )
        )

    output = capsys.readouterr().out
    assert output.count("✓ read_file") == 1
    assert display.lifecycle_key_count == 256


@pytest.mark.anyio
async def test_renderer_distinguishes_aci_and_command_truncation(
    capsys: pytest.CaptureFixture[str],
) -> None:
    display = TerminalToolEventDisplay(width=80)
    await display.emit(
        item_completed(
            turn_id="turn-tool",
            item_id="tool-result",
            item_kind=TurnItemKind.TOOL,
            status=ItemStatus.SUCCESS,
            data={
                "result": {
                    "tool_name": "read_file",
                    "structured_content": {"truncated": True},
                    "truncated": True,
                }
            },
        )
    )
    await display.emit(
        item_completed(
            turn_id="turn-command",
            item_id="command-result",
            item_kind=TurnItemKind.COMMAND,
            status=ItemStatus.SUCCESS,
            data={
                "result": {
                    "tool_name": "run_command",
                    "structured_content": {"exit_code": 0, "truncated": True},
                    "truncated": False,
                }
            },
        )
    )

    output = capsys.readouterr().out
    assert output.count("ACI 截断") == 1
    assert output.count("命令输出达到工具预算") == 1


@pytest.mark.anyio
async def test_renderer_sanitizes_streamed_model_text_controls(
    capsys: pytest.CaptureFixture[str],
) -> None:
    display = TerminalToolEventDisplay(width=80)

    await display.emit(text_delta("before\x1b[2J\x07after"))

    output = capsys.readouterr().out
    assert output == "beforeafter"
    assert "\x1b" not in output
    assert "\x07" not in output


@pytest.mark.anyio
async def test_renderer_failed_completion_keeps_result_metadata_and_warnings(
    capsys: pytest.CaptureFixture[str],
) -> None:
    display = TerminalToolEventDisplay(width=80)

    await display.emit(
        item_completed(
            turn_id="turn-failed-command",
            item_id="command-failed",
            item_kind=TurnItemKind.COMMAND,
            status=ItemStatus.FAILED,
            error="command failed",
            data={
                "result": {
                    "tool_name": "run_command",
                    "structured_content": {
                        "stdout": "partial output",
                        "stderr": "fatal detail",
                        "exit_code": 2,
                        "truncated": True,
                    },
                    "truncated": True,
                }
            },
        )
    )

    output = capsys.readouterr().out
    assert "✗ run_command: exit_code=2: command failed" in output
    assert "partial output" in output
    assert "fatal detail" in output
    assert output.count("ACI 截断") == 1
    assert output.count("命令输出达到工具预算") == 1


@pytest.mark.anyio
async def test_renderer_narrow_terminal_accepts_long_tool_name_preview(
    capsys: pytest.CaptureFixture[str],
) -> None:
    display = TerminalToolEventDisplay(width=20)

    await display.emit(
        item_started(
            turn_id="turn-narrow",
            item_id="tool-narrow",
            item_kind=TurnItemKind.TOOL,
            data={
                "tool_name": "a_very_long_tool_name",
                "input_preview": "path='fixture.txt'",
            },
        )
    )

    output = capsys.readouterr().out
    assert all(wcswidth(line) <= 20 for line in output.splitlines())


@pytest.mark.anyio
async def test_renderer_uses_actual_width_below_twenty_cells(
    capsys: pytest.CaptureFixture[str],
) -> None:
    display = TerminalToolEventDisplay(width=10)

    await display.emit(
        item_completed(
            turn_id="turn-tiny",
            item_id="tool-tiny",
            item_kind=TurnItemKind.TOOL,
            status=ItemStatus.SUCCESS,
            data={
                "result": {
                    "tool_name": "read_file",
                    "structured_content": {
                        "lines": [f"value {index}" for index in range(20)]
                    },
                }
            },
        )
    )

    output = capsys.readouterr().out
    assert any("+" in line for line in output.splitlines())
    assert all(wcswidth(line) <= 10 for line in output.splitlines())


@pytest.mark.anyio
async def test_renderer_bounds_legacy_progress_and_reports_suppression(
    capsys: pytest.CaptureFixture[str],
) -> None:
    display = TerminalToolEventDisplay(width=80)
    await display.emit(tool_use_start("read_file", "legacy-call"))
    for index in range(12):
        await display.emit(tool_use_progress("legacy-call", f"progress {index}"))
    await display.emit(tool_use_result("read_file", "legacy-call", {"ok": True}))

    output = capsys.readouterr().out
    assert output.count("… read_file: progress ") == 8
    assert "… read_file: … +4 progress updates" in output


@pytest.mark.anyio
async def test_renderer_streams_bounded_command_once_and_cleans_item_state(
    capsys: pytest.CaptureFixture[str],
) -> None:
    display = TerminalToolEventDisplay(width=80)
    start = item_started(
        turn_id="turn-command",
        item_id="command-1",
        item_kind=TurnItemKind.COMMAND,
        data={"tool_name": "run_command"},
    )
    await display.emit(start)
    for index in range(12):
        await display.emit(
            item_delta(
                turn_id="turn-command",
                item_id="command-1",
                item_kind=TurnItemKind.COMMAND,
                delta_kind=ItemDeltaKind.COMMAND_STDOUT,
                delta=f"line {index}\n",
            )
        )
    await display.emit(
        item_completed(
            turn_id="turn-command",
            item_id="command-1",
            item_kind=TurnItemKind.COMMAND,
            status=ItemStatus.SUCCESS,
            data={
                "result": {
                    "tool_name": "run_command",
                    "structured_content": {
                        "stdout": "\n".join(f"line {index}" for index in range(12)),
                        "stderr": "",
                        "exit_code": 0,
                        "truncated": False,
                    },
                    "truncated": False,
                }
            },
        )
    )

    output = capsys.readouterr().out
    assert output.count("line 0") == 1
    assert output.count("line 11") == 1
    assert "… +3 lines" in output
    assert display.active_item_count == 0


@pytest.mark.anyio
@pytest.mark.parametrize("status", [ItemStatus.SUCCESS, ItemStatus.FAILED])
async def test_renderer_verbose_command_does_not_repeat_streamed_output(
    status: ItemStatus,
    capsys: pytest.CaptureFixture[str],
) -> None:
    display = TerminalToolEventDisplay(width=80)
    display.set_verbose(True)
    await display.emit(
        item_started(
            turn_id="turn-verbose-command",
            item_id="command-verbose",
            item_kind=TurnItemKind.COMMAND,
            data={"tool_name": "run_command"},
        )
    )
    await display.emit(
        item_delta(
            turn_id="turn-verbose-command",
            item_id="command-verbose",
            item_kind=TurnItemKind.COMMAND,
            delta_kind=ItemDeltaKind.COMMAND_STDOUT,
            delta="unique streamed line\n",
        )
    )
    await display.emit(
        item_completed(
            turn_id="turn-verbose-command",
            item_id="command-verbose",
            item_kind=TurnItemKind.COMMAND,
            status=status,
            error="command failed" if status is ItemStatus.FAILED else None,
            data={
                "result": {
                    "tool_name": "run_command",
                    "structured_content": {
                        "stdout": "unique streamed line\n",
                        "stderr": "",
                        "exit_code": 0 if status is ItemStatus.SUCCESS else 1,
                        "truncated": False,
                    },
                    "truncated": False,
                }
            },
        )
    )

    assert capsys.readouterr().out.count("unique streamed line") == 1


@pytest.mark.anyio
async def test_interactive_activity_collapses_and_keeps_details(monkeypatch, capsys):
    clock = [10.0]
    display = TerminalToolEventDisplay(width=100, interactive=True, clock=lambda: clock[0])
    monkeypatch.setattr(display, "_start_live", lambda: None)
    display.begin_turn()
    await display.emit(item_delta(
        turn_id="t", item_id="r", item_kind=TurnItemKind.REASONING,
        delta_kind=ItemDeltaKind.REASONING, delta="private reasoning text",
    ))
    assert "思考中" in display.activity_text()
    await display.emit(item_started(
        turn_id="t", item_id="edit", item_kind=TurnItemKind.TOOL,
        data={"tool_name": "apply_patch", "input_preview": "file_path='demo.py'"},
    ))
    event = item_completed(
        turn_id="t", item_id="edit", item_kind=TurnItemKind.TOOL,
        status=ItemStatus.SUCCESS,
        data={"result": {"tool_name": "apply_patch", "structured_content": {
            "file_path": "demo.py", "created": True,
        }, "metadata": {"file_path": "demo.py", "diff": "--- /dev/null\n+++ demo.py\n@@ -0,0 +1 @@\n+print(1)"}}},
    )
    await display.emit(event)
    await display.emit(event)
    clock[0] = 12.5
    display.finish()
    summary = capsys.readouterr().out
    assert "2.5 秒" in summary
    assert "1 次工具" in summary
    assert "demo.py" in summary
    assert "private reasoning text" not in summary
    assert "+print(1)" not in summary
    display.show_details()
    details = capsys.readouterr().out
    assert "+print(1)" in details
    assert "private reasoning text" not in details
    display.finish()
    assert capsys.readouterr().out == ""


@pytest.mark.anyio
@pytest.mark.parametrize("verbose", [False, True])
async def test_details_retains_command_middle_in_both_modes(monkeypatch, capsys, verbose):
    display = TerminalToolEventDisplay(interactive=True)
    monkeypatch.setattr(display, "_start_live", lambda: None)
    display.set_verbose(verbose)
    display.begin_turn()
    await display.emit(item_started(turn_id="t", item_id="c", item_kind=TurnItemKind.COMMAND,
                                   data={"tool_name": "run_command"}))
    await display.emit(item_delta(turn_id="t", item_id="c", item_kind=TurnItemKind.COMMAND,
                                 delta_kind=ItemDeltaKind.COMMAND_STDOUT,
                                 delta="\n".join(f"unique-line-{i:02}" for i in range(20)) + "\n"))
    await display.emit(item_completed(turn_id="t", item_id="c", item_kind=TurnItemKind.COMMAND,
                                     status=ItemStatus.SUCCESS, data={"result": {"tool_name": "run_command"}}))
    display.finish()
    capsys.readouterr()
    display.show_details()
    details = capsys.readouterr().out
    for i in range(20):
        assert details.count(f"unique-line-{i:02}") == 1


def test_resume_preserves_activity_details_and_active_elapsed(monkeypatch, capsys):
    now = [0.0]
    display = TerminalToolEventDisplay(interactive=True, clock=lambda: now[0])
    monkeypatch.setattr(display, "_start_live", lambda: None)
    display.begin_turn()
    display._render_diff({
        "file_path": "notes.md",
        "diff": "--- a/notes.md\n+++ b/notes.md\n@@ -1 +1 @@\n----\n+++value",
    })
    now[0] = 2.0
    display.finish()
    first = capsys.readouterr().out
    assert "+1 / -1" in first
    now[0] = 100.0  # Approval waiting time is excluded from active execution.
    display.begin_turn(reset=False)
    now[0] = 103.0
    display.finish()
    summary = capsys.readouterr().out
    assert "5.0 秒" in summary
    assert "notes.md" in summary
    display.show_details()
    assert "+value" in capsys.readouterr().out


@pytest.mark.anyio
async def test_whitespace_only_model_steps_do_not_print_or_count_as_answer(capsys):
    display = TerminalToolEventDisplay(interactive=False)
    display.begin_turn()
    for i in range(3):
        await display.emit(item_started(turn_id="t", item_id=f"a{i}", item_kind=TurnItemKind.AGENT_MESSAGE))
        await display.emit(text_delta("\n"))
        await display.emit(text_delta("\n"))
    assert capsys.readouterr().out == ""
    assert not display.answer_streamed
    # A new meaningful message preserves indentation split across deltas.
    await display.emit(item_started(turn_id="t", item_id="final", item_kind=TurnItemKind.AGENT_MESSAGE))
    await display.emit(text_delta("    "))
    await display.emit(text_delta("print(1)"))
    await display.emit(text_delta("\n\n"))
    assert capsys.readouterr().out == "    print(1)\n\n"
    assert display.answer_streamed


@pytest.mark.anyio
@pytest.mark.parametrize("name", ["run_command", "manage_command"])
async def test_running_command_is_shown_as_running_with_handle(capsys, name):
    display = TerminalToolEventDisplay(width=100)
    await display.emit(item_completed(
        turn_id="t", item_id="session-call", item_kind=TurnItemKind.TOOL, status=ItemStatus.SUCCESS,
        data={"result": {"tool_name": name, "structured_content": {
            "status": "running", "session_id": "cmd_live", "exit_code": None,
        }}},
    ))
    output = capsys.readouterr().out
    assert "→ " + name in output
    assert "运行中" in output and "cmd_live" in output
    assert "✓ " + name not in output


@pytest.mark.anyio
async def test_effective_request_settings_stay_in_status_across_turns(capsys):
    from agent_runtime.streaming.events import EventType, StreamEvent
    display = TerminalToolEventDisplay(width=120)
    event = StreamEvent(type=EventType.BUDGET_UPDATE, turn_id='turn', data={
        'purpose': 'agent_step', 'effective_settings': {'revision': 1, 'reasoning_effort': 'high',
            'max_output_tokens': 65536, 'requested_output_tokens': 98304,
            'output_constraints': ['model_max_output_tokens']},
    })
    await display.emit(event)
    assert capsys.readouterr().out == ''
    first, _ = display.status_lines('demo')
    assert 'high' in first and '65,536' in first
    display.begin_turn()
    assert display.status_lines('demo')[0] == first
    await display.emit(event)
    display.finish()
    assert capsys.readouterr().out == ''
    display.show_context_status()
    details = capsys.readouterr().out
    assert '98,304' in details and '模型上限' in details
    display.update_request_settings({
        "preferences": {"reasoning_effort": "medium"}, "model_defaults": {"max_output_tokens": 8192},
    })
    display.show_context_status()
    assert "high" not in capsys.readouterr().out


@pytest.mark.anyio
async def test_status_separates_request_context_from_authoritative_turn_usage(capsys):
    from agent_runtime.result import AgentUsage
    from agent_runtime.streaming.events import EventType, StreamEvent

    display = TerminalToolEventDisplay(interactive=False)
    display.update_request_settings({
        "preferences": {"reasoning_effort": "high", "max_output_tokens": None},
        "model_defaults": {"reasoning_effort": "medium", "max_output_tokens": 8192},
    })
    assert "high" in display.status_lines("demo")[0]
    assert "待报告" in display.status_lines("demo")[1]
    await display.emit(StreamEvent(type=EventType.BUDGET_UPDATE, turn_id="t", data={
        "purpose": "agent_step",
        "context_projection": {"input_tokens": 1200, "max_input_tokens": 10000,
                               "count_source": "tokenizer_estimate"},
        "budget_status": {"cost_remaining_micros": 999},
    }))
    usage = AgentUsage(input_tokens=4000, output_tokens=500, total_tokens=4500,
                       cache_read_input_tokens=3000, reasoning_tokens=100)
    display.update_usage(usage)
    display.update_usage(usage)  # Replaying a cumulative result must not add it twice.
    settings, tokens = display.status_lines("demo")
    assert "上下文输入≈1,200/10,000" in settings
    assert "本轮累计" in tokens and "输入 4,000" in tokens and "输出 500" in tokens
    assert "缓存 3,000（输入内）" in tokens and "推理 100（输出内）" in tokens
    assert "4,500" in tokens
    assert "费用" not in settings + tokens
    display.set_verbose(True)
    display.show_context_status()
    details = capsys.readouterr().out
    assert "费用" not in details
    assert "本轮累计 Token" in details and "4,000" in details


def test_status_reports_unknown_cache_and_partial_usage_without_assuming_zero():
    from agent_runtime.result import AgentUsage

    display = TerminalToolEventDisplay(interactive=False)
    display.update_usage(AgentUsage(input_tokens=10, output_tokens=2, total_tokens=12, usage_complete=False))
    _, tokens = display.status_lines("demo")
    assert "不完整" in tokens and "缓存 未知" in tokens and "推理 未知" in tokens
    display.begin_turn()
    assert "待报告" in display.status_lines("demo")[1]
    display.finish()


@pytest.mark.anyio
@pytest.mark.parametrize(("source", "label", "approximate"), [
    ("utf8_bytes_conservative", "字节估算", True),
    ("tiktoken:cl100k_base", "本地分词", False),
    ("external_counter", "外部计数", False),
])
async def test_context_status_identifies_count_source(source, label, approximate):
    from agent_runtime.streaming.events import EventType, StreamEvent

    display = TerminalToolEventDisplay(interactive=False)
    await display.emit(StreamEvent(type=EventType.BUDGET_UPDATE, turn_id="t", data={
        "purpose": "agent_step",
        "context_projection": {"input_tokens": 1234, "max_input_tokens": 10000, "count_source": source},
    }))
    first, _ = display.status_lines("demo")
    assert label in first and ("≈" in first) is approximate
