from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from agent_runtime.harness import RolloutStore
from agent_runtime.streaming.events import (
    ItemDeltaKind,
    ItemStatus,
    TurnItemKind,
    item_completed,
    item_delta,
    item_started,
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


def test_command_preview_preserves_chunk_order_and_equal_deltas() -> None:
    preview = BoundedCommandPreview(width=80)

    visible = [
        *preview.feed("same\n"),
        *preview.feed("same\nerr"),
        *preview.feed("or\nlast\n"),
        *preview.finish(),
    ]

    assert visible == ["same", "same", "error", "last"]


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
            binding_manifest={"model_alias": "test-model"},
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
    display = TerminalToolEventDisplay(width=32)

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
