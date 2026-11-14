"""Task 12/13 内置工具:inspect_data_file(本文件 Task 12 建,Task 13 追加 apply_patch)。

全部用例跑**真实解析**:xlsx/xlsm 用 openpyxl 现造,pdf 用 pymupdf 现造,
csv/tsv/json 直接写文本;不 mock openpyxl/fitz/csv。

R-T12-2 的核心回归:表头按 1-based ``start_row`` 直接定位,绝不叠加旧 B8 形态的
"header = row-1 又 skiprows=range(header)" 双重偏移。表头在第 1/3/5 行的
CSV 与工作表各有用例,列名逐字断言。

仓库无 pytest-asyncio:与 test_tools_files_search.py 同风格,asyncio.run 包装。
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import re
import time
from pathlib import Path
from typing import Any

import pytest

from praxis.core import permission
from praxis.core.pipeline import ExecCtx, execute
from praxis.protocol.items import ItemKind
from praxis.protocol.permission import PermissionContext
from praxis.protocol.policy import PermissionPolicy, ToolRule
from praxis.protocol.sandbox import ToolExecContext
from praxis.protocol.tools import (
    CancellationMode,
    InterruptBehavior,
    ResultStatus,
    ToolCall,
    ToolEffect,
    ToolValidationError,
)
from praxis.store.turns import create_turn
from praxis.tools.builtins import data as data_module
from praxis.tools.builtins.data import _MAX_INSPECTABLE_BYTES, create_inspect_data_file_tool
from praxis.tools.builtins.patch import (
    AddFile,
    DeleteFile,
    UpdateFile,
    create_apply_patch_tool,
    parse_patch,
)

_METADATA_KEYS = {
    "format",
    "size_bytes",
    "sha256",
    "valid",
    "row_count",
    "column_count",
    "column_names",
    "next_row",
    "next_column",
    "next_sheet_offset",
    "page_count",
    "json_type",
    "json_keys",
    "truncated",
    "truncated_cells",
}

_ROW_LINE = re.compile(r"^(\d+) \|")


def run(tool: Any, arguments: dict[str, Any]) -> Any:
    return asyncio.run(tool.run(arguments, ToolExecContext()))


def inspect(workspace: Path, name: str, **arguments: Any) -> Any:
    return run(create_inspect_data_file_tool(workspace), {"path": name, **arguments})


def row_numbers(result: Any) -> list[int]:
    return [int(match.group(1)) for line in result.text.splitlines() if (match := _ROW_LINE.match(line))]


_SHEET_SECTION = re.compile(r'^\[sheet "([^"]+)"')


def sheet_rows(text: str) -> dict[str, list[int]]:
    """把 content 按 sheet 段切开,返回每张表出现的源行号(顺序保留)。"""
    sections: dict[str, list[int]] = {}
    current: str | None = None
    for line in text.splitlines():
        section = _SHEET_SECTION.match(line)
        if section:
            current = section.group(1)
            sections.setdefault(current, [])
            continue
        match = _ROW_LINE.match(line)
        if match and current is not None:
            sections[current].append(int(match.group(1)))
    return sections


def make_workbook(path: Path, sheets: dict[str, list[list[Any]]]) -> None:
    from openpyxl import Workbook

    workbook = Workbook()
    workbook.remove(workbook.active)
    for title, rows in sheets.items():
        worksheet = workbook.create_sheet(title)
        for row_index, row in enumerate(rows, start=1):
            for column_index, value in enumerate(row, start=1):
                if value is not None:
                    worksheet.cell(row=row_index, column=column_index, value=value)
    workbook.save(path)


def make_pdf(path: Path, pages: int) -> None:
    import pymupdf

    document = pymupdf.open()
    for index in range(pages):
        page = document.new_page()
        page.insert_text((72, 72), f"page-{index + 1}-text")
    document.save(path)
    document.close()


# ---------------------------------------------------------------------------
# R-T12-2:表头按 1-based start_row 直接定位(不得双重偏移)
# ---------------------------------------------------------------------------


def test_csv_header_at_row_one(tmp_path: Path) -> None:
    (tmp_path / "t.csv").write_text("name,age\na,1\nb,2\n", encoding="utf-8")
    r = inspect(tmp_path, "t.csv")
    assert r.status is ResultStatus.OK
    assert r.metadata["column_names"] == ("name", "age")
    assert r.metadata["row_count"] == 3
    assert r.metadata["column_count"] == 2
    assert not r.truncated


def test_csv_header_at_row_three_not_double_shifted(tmp_path: Path) -> None:
    # 旧 B8 形态(header=row-1 又 skiprows=range(header))在这里会把列名读成第 1 行
    (tmp_path / "t.csv").write_text(
        "meta line\nanother\nname,age\na,1\nb,2\n", encoding="utf-8"
    )
    r = inspect(tmp_path, "t.csv", start_row=3)
    assert r.status is ResultStatus.OK
    assert r.metadata["column_names"] == ("name", "age")
    assert r.metadata["row_count"] == 5
    assert r.metadata["column_count"] == 2
    # 预览从第 3 行起,行号是原文件行号
    assert row_numbers(r) == [3, 4, 5]
    assert r.text.count("header: name | age") == 1


def test_csv_header_at_row_five_not_double_shifted(tmp_path: Path) -> None:
    (tmp_path / "t.csv").write_text(
        "n1\nn2\nn3\nn4\nname,age\na,1\n", encoding="utf-8"
    )
    r = inspect(tmp_path, "t.csv", start_row=5)
    assert r.metadata["column_names"] == ("name", "age")
    assert row_numbers(r) == [5, 6]
    assert r.metadata["row_count"] == 6


def test_xlsx_header_at_row_one(tmp_path: Path) -> None:
    make_workbook(tmp_path / "t.xlsx", {"Data": [["name", "age"], ["a", 1]]})
    r = inspect(tmp_path, "t.xlsx")
    assert r.status is ResultStatus.OK
    assert r.metadata["column_names"] == ("name", "age")
    assert r.metadata["row_count"] == 2


def test_xlsx_header_at_row_three_not_double_shifted(tmp_path: Path) -> None:
    make_workbook(
        tmp_path / "t.xlsx",
        {"Data": [["noise"], ["noise2"], ["name", "age"], ["a", 1], ["b", 2]]},
    )
    r = inspect(tmp_path, "t.xlsx", start_row=3)
    assert r.metadata["column_names"] == ("name", "age")
    assert row_numbers(r) == [3, 4, 5]


def test_xlsx_header_at_row_five_not_double_shifted(tmp_path: Path) -> None:
    make_workbook(
        tmp_path / "t.xlsx",
        {"Data": [["n1"], ["n2"], ["n3"], ["n4"], ["name", "age"], ["a", 1]]},
    )
    r = inspect(tmp_path, "t.xlsx", start_row=5)
    assert r.metadata["column_names"] == ("name", "age")
    assert row_numbers(r) == [5, 6]
    assert r.metadata["row_count"] == 6


# ---------------------------------------------------------------------------
# R-T12-3:预算口径唯一(2000 格跨 sheet 共享)与可前进的翻页游标
# ---------------------------------------------------------------------------


def test_grid_cell_budget_is_shared_across_sheets(tmp_path: Path) -> None:
    # 每张 sheet 100 行 × 20 列 = 恰好 2000 格:第一张吃掉全部预算,
    # 第二张必须留给 next_sheet_offset(每 sheet 各 2000 的第二套口径会一次返回两张)。
    first_rows = [[f"first-r{row}c{column}" for column in range(20)] for row in range(100)]
    second_rows = [[f"second-r{row}c{column}" for column in range(20)] for row in range(100)]
    make_workbook(tmp_path / "t.xlsx", {"Data": first_rows, "Second": second_rows})
    first = inspect(tmp_path, "t.xlsx", max_rows=100, max_columns=20, max_characters=32_000)
    assert first.status is ResultStatus.OK
    assert first.truncated
    assert first.metadata["next_sheet_offset"] == 1
    assert "first-r0c0" in first.text
    assert "second-r0c0" not in first.text
    assert "next_sheet_offset=1" in first.text
    second = inspect(
        tmp_path, "t.xlsx", sheet_offset=1, max_rows=100, max_columns=20, max_characters=32_000
    )
    assert second.status is ResultStatus.OK
    assert "second-r0c0" in second.text
    assert not second.truncated
    assert second.metadata["next_sheet_offset"] is None


def test_default_parameters_page_two_sheets_without_loss_or_repeat(tmp_path: Path) -> None:
    # 复审复现形状:两表各 20 行 × 2 列 + 默认参数。按返回游标翻页,
    # A/B 的 1..20 各覆盖恰好一次(修 round 1 的 next_sheet_offset 覆盖曾丢 A 11-20、重读 B 1-10)。
    rows_a = [[f"a-{row}-{column}" for column in range(2)] for row in range(1, 21)]
    rows_b = [[f"b-{row}-{column}" for column in range(2)] for row in range(1, 21)]
    make_workbook(tmp_path / "t.xlsx", {"A": rows_a, "B": rows_b})
    tool = create_inspect_data_file_tool(tmp_path)
    arguments: dict[str, Any] = {"path": "t.xlsx"}
    seen: dict[str, list[int]] = {"A": [], "B": []}
    calls = 0
    while True:
        result = run(tool, arguments)
        assert result.status is ResultStatus.OK
        calls += 1
        for name, numbers in sheet_rows(result.text).items():
            seen[name].extend(numbers)
        next_sheet = result.metadata["next_sheet_offset"]
        next_row = result.metadata["next_row"]
        if next_sheet is None and next_row is None:
            break
        assert calls <= 10
        if next_row is not None:
            arguments = {
                "path": "t.xlsx",
                "sheet_offset": next_sheet if next_sheet is not None else 0,
                "start_row": next_row,
            }
        else:
            assert next_sheet is not None
            arguments = {"path": "t.xlsx", "sheet_offset": next_sheet}
    assert calls == 3
    assert seen["A"] == list(range(1, 21))
    assert seen["B"] == list(range(1, 21))


def test_column_truncated_table_owns_the_sheet_cursor(tmp_path: Path) -> None:
    # 表 0 未截断、表 1 列截断:截断表必须是本页最后一张,next_sheet_offset 指向它自己
    make_workbook(
        tmp_path / "t.xlsx",
        {"A": [["a1"]], "B": [["b1", "b2"], ["b3", "b4"]]},
    )
    first = inspect(tmp_path, "t.xlsx", max_columns=1)
    assert first.status is ResultStatus.OK
    assert first.metadata["next_row"] is None
    assert first.metadata["next_column"] == 2
    assert first.metadata["next_sheet_offset"] == 1
    assert "a1" in first.text
    assert "b1" in first.text
    # 行形态断言:裸 "b2" 会随机命中 sha256(openpyxl 造包时间戳使哈希每次不同)
    assert "1 | b2" not in first.text
    second = inspect(tmp_path, "t.xlsx", sheet_offset=1, start_column=2, max_columns=1)
    assert second.status is ResultStatus.OK
    assert second.metadata["next_column"] is None
    assert second.metadata["next_sheet_offset"] is None
    assert "b2" in second.text
    # 用行形态断言:裸 "b1" 会随机命中 sha256 子串(openpyxl 造包时间戳使哈希每次不同)
    assert "1 | b1" not in second.text


def test_multi_sheet_row_truncation_never_skips_rows(tmp_path: Path) -> None:
    # I-2:sheet1 超 2000 格被行截断时,next_sheet_offset 不得越过它;
    # 按返回游标依次翻页,两张表所有行必须各覆盖一次、不重不漏。
    first_rows = [[f"a-{row}-{column}" for column in range(20)] for row in range(120)]
    second_rows = [[f"b-{row}-{column}" for column in range(20)] for row in range(30)]
    make_workbook(tmp_path / "t.xlsx", {"Data": first_rows, "Second": second_rows})
    tool = create_inspect_data_file_tool(tmp_path)
    arguments: dict[str, Any] = {"path": "t.xlsx", "max_rows": 100, "max_columns": 20}
    seen: dict[str, list[int]] = {"Data": [], "Second": []}
    pages = 0
    while True:
        result = run(tool, arguments)
        assert result.status is ResultStatus.OK
        pages += 1
        for name, numbers in sheet_rows(result.text).items():
            seen[name].extend(numbers)
        next_sheet = result.metadata["next_sheet_offset"]
        next_row = result.metadata["next_row"]
        if next_sheet is None and next_row is None:
            break
        assert pages <= 5
        if next_row is not None:
            arguments = {
                "path": "t.xlsx",
                "max_rows": 100,
                "max_columns": 20,
                "sheet_offset": next_sheet if next_sheet is not None else 0,
                "start_row": next_row,
            }
        else:
            assert next_sheet is not None
            arguments = {
                "path": "t.xlsx",
                "max_rows": 100,
                "max_columns": 20,
                "sheet_offset": next_sheet,
            }
    assert seen["Data"] == list(range(1, 121))
    assert seen["Second"] == list(range(1, 31))


def test_first_call_points_next_sheet_at_the_truncated_sheet(tmp_path: Path) -> None:
    first_rows = [[f"a-{row}-{column}" for column in range(20)] for row in range(120)]
    second_rows = [[f"b-{row}-{column}" for column in range(20)] for row in range(5)]
    make_workbook(tmp_path / "t.xlsx", {"Data": first_rows, "Second": second_rows})
    r = inspect(tmp_path, "t.xlsx", max_rows=100, max_columns=20)
    assert r.metadata["next_row"] == 101
    # 不越过仍有剩余行的 Data(next_sheet_offset 停在 0)
    assert r.metadata["next_sheet_offset"] == 0
    assert "next_sheet_offset=0" in r.text


def test_max_sheets_paging_advances_next_sheet_offset(tmp_path: Path) -> None:
    make_workbook(
        tmp_path / "t.xlsx",
        {"One": [["one"]], "Two": [["two"]], "Three": [["three"]]},
    )
    first = inspect(tmp_path, "t.xlsx", max_sheets=2)
    assert first.status is ResultStatus.OK
    assert first.metadata["next_sheet_offset"] == 2
    assert "one" in first.text
    assert "two" in first.text
    assert "three" not in first.text
    second = inspect(tmp_path, "t.xlsx", sheet_offset=2)
    assert second.status is ResultStatus.OK
    assert second.metadata["next_sheet_offset"] is None
    assert "three" in second.text
    assert "one" not in second.text


def test_grid_budget_truncated_cells_and_row_cursor(tmp_path: Path) -> None:
    # >2000 格 → truncated + truncated_cells 记 (row,col),next_row 可继续翻页
    rows = [["v" for _ in range(20)] for _ in range(120)]
    make_workbook(tmp_path / "t.xlsx", {"Data": rows})
    r = inspect(tmp_path, "t.xlsx", max_rows=100, max_columns=20)
    assert r.truncated
    assert r.metadata["truncated"] is True
    assert r.metadata["next_row"] == 101  # 2000 // 20 = 100 行上限,1-based 下一行
    assert (101, 1) in r.metadata["truncated_cells"]
    # 绝不出现"截断但不前进"
    assert r.metadata["next_row"] > 1


def test_row_pagination_advances_without_gaps_or_repeats(tmp_path: Path) -> None:
    lines = ["name,value"] + [f"row{i},{i}" for i in range(2, 31)]
    (tmp_path / "t.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")
    first = inspect(tmp_path, "t.csv", max_rows=10)
    assert first.truncated
    assert first.metadata["row_count"] is None  # 未到 EOF
    assert first.metadata["next_row"] == 11
    assert row_numbers(first) == list(range(1, 11))
    second = inspect(tmp_path, "t.csv", start_row=first.metadata["next_row"], max_rows=10)
    assert second.metadata["next_row"] == 21
    assert row_numbers(second) == list(range(11, 21))
    third = inspect(tmp_path, "t.csv", start_row=second.metadata["next_row"], max_rows=10)
    assert third.metadata["next_row"] is None
    assert third.metadata["row_count"] == 30
    assert row_numbers(third) == list(range(21, 31))
    assert not third.truncated


def test_max_rows_is_bounded_by_the_2000_cell_budget(tmp_path: Path) -> None:
    # brief 的公式:实际 max_rows = min(max_rows, 2000 // max_columns)
    rows = [[f"{row}-{column}" for column in range(100)] for row in range(50)]
    make_workbook(tmp_path / "t.xlsx", {"Data": rows})
    r = inspect(tmp_path, "t.xlsx", max_rows=100, max_columns=100, max_characters=32_000)
    assert row_numbers(r) == list(range(1, 21))
    assert r.metadata["next_row"] == 21
    assert r.truncated


def test_start_row_beyond_eof_is_empty_result_not_error(tmp_path: Path) -> None:
    (tmp_path / "t.csv").write_text("a,b\n1,2\n", encoding="utf-8")
    r = inspect(tmp_path, "t.csv", start_row=9)
    assert r.status is ResultStatus.OK
    assert not r.is_error
    assert r.metadata["column_names"] == ()
    assert r.metadata["row_count"] == 2
    assert r.metadata["column_count"] == 2
    assert r.metadata["next_row"] is None
    assert not r.truncated
    assert "no rows from row 9" in r.text


def test_long_cell_text_is_shortened_and_recorded(tmp_path: Path) -> None:
    (tmp_path / "t.csv").write_text("a,b\n" + "x" * 3_000 + ",short\n", encoding="utf-8")
    r = inspect(tmp_path, "t.csv")
    assert r.truncated
    assert r.metadata["truncated_cells"] == ((2, 1),)
    assert "…" in r.text
    assert r.metadata["next_row"] is None


def test_column_pagination_reports_next_column(tmp_path: Path) -> None:
    (tmp_path / "t.csv").write_text("a,b,c\n1,2,3\n", encoding="utf-8")
    first = inspect(tmp_path, "t.csv", max_columns=2)
    assert first.metadata["column_names"] == ("a", "b")
    assert first.metadata["next_column"] == 3
    assert first.metadata["column_count"] == 3
    assert first.truncated
    second = inspect(tmp_path, "t.csv", start_column=3, max_columns=2)
    assert second.metadata["column_names"] == ("c",)
    assert second.metadata["next_column"] is None
    assert not second.truncated


# ---------------------------------------------------------------------------
# R-T12-1:解析必须在线程里跑,事件循环不能被占死
# ---------------------------------------------------------------------------


def test_parse_runs_in_thread_and_event_loop_stays_responsive(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "t.csv").write_text("name\nvalue\n", encoding="utf-8")
    real = data_module._inspect_data_file

    def slowed(*args: Any, **kwargs: Any) -> Any:
        time.sleep(0.4)
        return real(*args, **kwargs)

    monkeypatch.setattr(data_module, "_inspect_data_file", slowed)
    tool = create_inspect_data_file_tool(tmp_path)

    async def scenario() -> tuple[int, Any]:
        ticks = 0

        async def spinner() -> None:
            nonlocal ticks
            while True:
                await asyncio.sleep(0.01)
                ticks += 1

        task = asyncio.create_task(spinner())
        try:
            result = await tool.run({"path": "t.csv"}, ToolExecContext())
        finally:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task
        return ticks, result

    ticks, result = asyncio.run(scenario())
    # 0.4s 的同步解析若占死循环,ticks 只能是 0;线程里跑则应 ~40 次
    assert ticks >= 10
    assert result.status is ResultStatus.OK


def test_tool_contract_declares_thread_limits(tmp_path: Path) -> None:
    tool = create_inspect_data_file_tool(tmp_path)
    assert tool.definition.name == "inspect_data_file"
    assert tool.interrupt_behavior is InterruptBehavior.FINISH_CURRENT
    assert tool.cancellation_mode is CancellationMode.COOPERATIVE
    assert tool.timeout_seconds == 60
    assert tool.max_model_output_bytes == 500_000
    assert tool.idempotent is True
    assert tool.concurrency_safe is True
    assert tool.static_effects == frozenset({ToolEffect.READ_WORKSPACE})
    # 线程不可杀的事实必须写进模型可见的工具描述
    assert "cannot be killed" in tool.definition.description


def test_resolve_use_targets_the_file(tmp_path: Path) -> None:
    tool = create_inspect_data_file_tool(tmp_path)
    use = tool.resolve_use({"path": "t.csv"})
    assert use.effects == frozenset({ToolEffect.READ_WORKSPACE})
    assert [target.value for target in use.targets] == [str((tmp_path / "t.csv").resolve())]


# ---------------------------------------------------------------------------
# 错误码:data_inspection_failed / not_found
# ---------------------------------------------------------------------------


def test_oversized_file_rejected(tmp_path: Path) -> None:
    # 稀疏文件:stat 即超限,不读 200 MiB 内容
    target = tmp_path / "big.csv"
    with target.open("wb") as stream:
        stream.truncate(_MAX_INSPECTABLE_BYTES + 1)
    r = inspect(tmp_path, "big.csv")
    assert r.status is ResultStatus.ERROR
    assert r.is_error
    assert r.error_code == "data_inspection_failed"
    assert "200 MiB" in r.text
    assert r.metadata["valid"] is False
    assert r.metadata["sha256"] == ""
    assert not r.truncated


@pytest.mark.parametrize("name", ["t.txt", "t.xls", "t.doc"])
def test_unsupported_format_is_data_inspection_failed(tmp_path: Path, name: str) -> None:
    (tmp_path / name).write_text("whatever", encoding="utf-8")
    r = inspect(tmp_path, name)
    assert r.status is ResultStatus.ERROR
    assert r.error_code == "data_inspection_failed"
    assert "unsupported" in r.text
    assert r.metadata["format"] == "unsupported"
    assert r.metadata["valid"] is False


def test_missing_file_is_not_found(tmp_path: Path) -> None:
    r = inspect(tmp_path, "nope.csv")
    assert r.status is ResultStatus.ERROR
    assert r.error_code == "not_found"
    assert "not found" in r.text
    assert set(r.metadata) == _METADATA_KEYS


def test_parse_error_message_is_bounded_and_does_not_echo_content(tmp_path: Path) -> None:
    # 畸形 JSON 的错误消息只描述位置,不得回显文件内容
    secret = "SECRET-" * 2_000
    (tmp_path / "t.json").write_text('{"k": ' + secret + "}", encoding="utf-8")
    r = inspect(tmp_path, "t.json")
    assert r.status is ResultStatus.ERROR
    assert r.error_code == "data_inspection_failed"
    assert "SECRET" not in r.text
    assert len(r.text) <= 1_000
    assert "  " not in r.text
    assert r.metadata["valid"] is False


def test_sheet_name_with_sheet_offset_is_invalid_input(tmp_path: Path) -> None:
    tool = create_inspect_data_file_tool(tmp_path)
    with pytest.raises(ToolValidationError) as excinfo:
        tool.validate_input({"path": "t.xlsx", "sheet_name": "Data", "sheet_offset": 1})
    assert "sheet_name and sheet_offset cannot be combined" in str(excinfo.value)


def test_unknown_sheet_name_is_data_inspection_failed(tmp_path: Path) -> None:
    make_workbook(tmp_path / "t.xlsx", {"Data": [["a"]]})
    r = inspect(tmp_path, "t.xlsx", sheet_name="Nope")
    assert r.status is ResultStatus.ERROR
    assert r.error_code == "data_inspection_failed"


# ---------------------------------------------------------------------------
# 格式覆盖:json / tsv / pdf / xlsx(公式不求值、按 sheet_name 选择)
# ---------------------------------------------------------------------------


def test_json_shape_and_sorted_keys(tmp_path: Path) -> None:
    (tmp_path / "t.json").write_text(
        json.dumps({"b": 1, "a": [1, 2], "c": None}), encoding="utf-8"
    )
    r = inspect(tmp_path, "t.json")
    assert r.status is ResultStatus.OK
    assert r.metadata["format"] == "json"
    assert r.metadata["json_type"] == "object"
    assert r.metadata["json_keys"] == ("a", "b", "c")
    assert not r.truncated
    assert '"a"' in r.text


def test_json_array_and_preview_truncation(tmp_path: Path) -> None:
    (tmp_path / "t.json").write_text(json.dumps(list(range(500))), encoding="utf-8")
    r = inspect(tmp_path, "t.json", max_characters=100)
    assert r.metadata["json_type"] == "array"
    assert r.metadata["json_keys"] == ()
    assert r.truncated
    assert "preview truncated" in r.text
    # metadata 契约里没有 json_preview 键(预览只在 content 里,受 max_characters 约束)
    assert "json_preview" not in r.metadata


def test_json_over_8mib_is_rejected_before_parsing(tmp_path: Path) -> None:
    # I-1:JSON 全量物化 → 专属 8 MiB 准入,超限走结果而不是 OOM
    target = tmp_path / "big.json"
    with target.open("wb") as stream:
        stream.truncate(8 * 1024 * 1024 + 1)
    r = inspect(tmp_path, "big.json")
    assert r.status is ResultStatus.ERROR
    assert r.error_code == "data_inspection_failed"
    assert "8 MiB" in r.text
    assert "preview limit" in r.text
    assert r.metadata["valid"] is False
    # 准入在哈希之前:拒绝路径不会把整个文件再流一遍
    assert r.metadata["sha256"] == ""


def test_json_nan_has_dedicated_error(tmp_path: Path) -> None:
    # json.load 接受 NaN/Infinity,但 allow_nan=False 的 dump 会抛;专门消息明确化
    (tmp_path / "t.json").write_text('{"v": NaN, "n": Infinity}', encoding="utf-8")
    r = inspect(tmp_path, "t.json")
    assert r.status is ResultStatus.ERROR
    assert r.error_code == "data_inspection_failed"
    assert "NaN" in r.text
    assert r.metadata["valid"] is False


def test_tsv_uses_tab_delimiter(tmp_path: Path) -> None:
    (tmp_path / "t.tsv").write_text("name\tage\na\t1\n", encoding="utf-8")
    r = inspect(tmp_path, "t.tsv")
    assert r.status is ResultStatus.OK
    assert r.metadata["column_names"] == ("name", "age")
    assert r.metadata["format"] == "tsv"


def test_pdf_pages_and_truncation(tmp_path: Path) -> None:
    make_pdf(tmp_path / "t.pdf", pages=3)
    r = inspect(tmp_path, "t.pdf", max_pages=2)
    assert r.status is ResultStatus.OK
    assert r.metadata["format"] == "pdf"
    assert r.metadata["page_count"] == 3
    assert r.truncated
    assert "page-1-text" in r.text
    assert "page-2-text" in r.text
    assert "page-3-text" not in r.text


def test_csv_field_beyond_default_limit_is_truncated_not_failed(tmp_path: Path) -> None:
    # 默认 csv 字段上限 128 KiB 会让 200k 字符的字段炸掉整份文件;抬高后应正常截断
    (tmp_path / "t.csv").write_text("a,b\n" + "x" * 200_000 + ",1\n", encoding="utf-8")
    r = inspect(tmp_path, "t.csv")
    assert r.status is ResultStatus.OK
    assert r.error_code is None
    assert r.truncated
    assert "…" in r.text


def test_pdf_total_text_is_bounded_by_max_characters(tmp_path: Path) -> None:
    import pymupdf

    target = tmp_path / "t.pdf"
    document = pymupdf.open()
    first = document.new_page()
    first.insert_textbox(pymupdf.Rect(20, 20, 580, 800), "A" * 2_000, fontsize=8)
    second = document.new_page()
    second.insert_text((72, 72), "SECOND-PAGE")
    document.save(target)
    document.close()
    r = inspect(tmp_path, "t.pdf", max_characters=100)
    assert r.status is ResultStatus.OK
    assert r.metadata["page_count"] == 2
    assert r.truncated
    assert r.text.count("A") == 100  # 聚合文本总量不超过 max_characters
    assert "SECOND-PAGE" not in r.text
    assert "[page 1/2: 100 characters, truncated]" in r.text


def test_xlsx_formula_is_not_evaluated(tmp_path: Path) -> None:
    make_workbook(tmp_path / "t.xlsx", {"Data": [["a", "b"], [1, 2], ["=SUM(A2:B2)"]]})
    r = inspect(tmp_path, "t.xlsx")
    assert r.status is ResultStatus.OK
    assert "=SUM(A2:B2)" in r.text


def test_xlsx_sheet_name_selects_one_sheet(tmp_path: Path) -> None:
    make_workbook(
        tmp_path / "t.xlsx",
        {"Data": [["first-only"]], "Second": [["second-only"]]},
    )
    r = inspect(tmp_path, "t.xlsx", sheet_name="Second")
    assert r.status is ResultStatus.OK
    assert "second-only" in r.text
    assert "first-only" not in r.text
    assert r.metadata["next_sheet_offset"] is None


def test_success_metadata_has_exactly_the_contract_keys(tmp_path: Path) -> None:
    (tmp_path / "t.csv").write_text("a,b\n1,2\n", encoding="utf-8")
    r = inspect(tmp_path, "t.csv")
    assert set(r.metadata) == _METADATA_KEYS
    assert r.metadata["valid"] is True
    assert len(r.metadata["sha256"]) == 64
    assert r.metadata["size_bytes"] == (tmp_path / "t.csv").stat().st_size


# ===========================================================================
# Task 13: apply_patch(codex 多文件 hunk 格式的进程内实现)
#
# R-T13-1 effects 动态化;R-T13-2 逐文件 target;R-T13-3 跨文件非事务 +
# 单文件原子;R-T13-4 完整性检查留在工具内,权限/地板一概交给管道。
# ===========================================================================

_FAILURE_MESSAGE = (
    "patch failed. Earlier hunks may already have changed files; "
    "inspect the workspace before retrying."
)


def _patch(*sections: str) -> str:
    return "*** Begin Patch\n" + "\n".join(sections) + "\n*** End Patch\n"


def run_patch(workspace: Path, patch: str) -> Any:
    return run(create_apply_patch_tool(workspace), {"patch": patch})


def _abs(workspace: Path, *parts: str) -> str:
    return str((workspace / Path(*parts)).resolve())


def _permission_ctx(
    workspace: Path,
    *,
    mode: str = "on-request",
    tools: tuple[ToolRule, ...] = (),
) -> PermissionContext:
    return PermissionContext(
        workspace=str(workspace),
        policy=PermissionPolicy(tools=tools),
        mode=mode,  # type: ignore[arg-type]
    )


# ---------------------------------------------------------------------------
# R-T13-2:逐个涉及路径(含 Move to 目标)都是 kind="path" 的 target
# ---------------------------------------------------------------------------


def test_apply_patch_targets_are_per_file_for_permission(tmp_path: Path) -> None:
    patch = _patch(
        "*** Update File: a.py\n@@\n-old-a\n+new-a",
        "*** Update File: b.py\n@@\n-old-b\n+new-b",
    )
    use = create_apply_patch_tool(tmp_path).resolve_use({"patch": patch})
    assert sorted(t.value for t in use.targets) == sorted(_abs(tmp_path, name) for name in ("a.py", "b.py"))
    # R-T13-1:纯 Update 只声明读写,不声明 DESTRUCTIVE
    assert use.effects == frozenset({ToolEffect.READ_WORKSPACE, ToolEffect.WRITE_WORKSPACE})


def test_move_destination_is_also_a_target(tmp_path: Path) -> None:
    patch = _patch("*** Update File: old.py\n*** Move to: moved/new.py\n@@\n-a\n+b")
    use = create_apply_patch_tool(tmp_path).resolve_use({"patch": patch})
    assert sorted(t.value for t in use.targets) == sorted(
        (_abs(tmp_path, "old.py"), _abs(tmp_path, "moved", "new.py"))
    )


# ---------------------------------------------------------------------------
# R-T13-1:effects 动态化——Delete 才追加 DESTRUCTIVE(S5 强制审批)
# ---------------------------------------------------------------------------


def test_update_only_patch_is_allowed_by_workspace_write_grant(tmp_path: Path) -> None:
    tool = create_apply_patch_tool(tmp_path)
    use = tool.resolve_use({"patch": _patch("*** Update File: a.py\n@@\n-old\n+new")})
    decision = permission.decide(tool, use, _permission_ctx(tmp_path))
    assert decision.is_allow
    assert decision.reason == "policy-grant"


def test_update_only_patch_is_allowed_by_star_rule(tmp_path: Path) -> None:
    tool = create_apply_patch_tool(tmp_path)
    use = tool.resolve_use({"patch": _patch("*** Add File: a.py\n+new")})
    decision = permission.decide(
        tool, use, _permission_ctx(tmp_path, tools=(ToolRule(pattern="*", decision="allow"),))
    )
    assert decision.is_allow
    assert decision.reason == "rule"


def test_delete_patch_asks_even_with_star_allow_rule(tmp_path: Path) -> None:
    tool = create_apply_patch_tool(tmp_path)
    use = tool.resolve_use({"patch": _patch("*** Delete File: a.py")})
    assert use.effects == frozenset(
        {ToolEffect.READ_WORKSPACE, ToolEffect.WRITE_WORKSPACE, ToolEffect.DESTRUCTIVE}
    )
    ctx = _permission_ctx(tmp_path, tools=(ToolRule(pattern="*", decision="allow"),))
    assert permission.decide(tool, use, ctx).is_ask


def test_bypass_allows_delete_but_floor_still_denies(tmp_path: Path) -> None:
    tool = create_apply_patch_tool(tmp_path)
    ordinary = tool.resolve_use({"patch": _patch("*** Delete File: a.py")})
    assert permission.decide(tool, ordinary, _permission_ctx(tmp_path, mode="bypass")).is_allow
    protected = tool.resolve_use({"patch": _patch("*** Delete File: .git/config")})
    decision = permission.decide(tool, protected, _permission_ctx(tmp_path, mode="bypass"))
    assert decision.is_deny
    assert decision.reason.startswith("floor:")


# ---------------------------------------------------------------------------
# 应用语义:多文件/多 hunk、Add/Delete/Move、摘要与 metadata
# ---------------------------------------------------------------------------


def test_hunks_apply_and_report_applied_files(tmp_path: Path) -> None:
    (tmp_path / "a.py").write_text("one\ntwo\nthree\nfour\n", encoding="utf-8")
    patch = _patch(
        "*** Update File: a.py\n@@\n one\n-two\n+TWO\n@@\n four\n+five",
    )
    r = run_patch(tmp_path, patch)
    assert r.status is ResultStatus.OK
    assert not r.is_error
    assert (tmp_path / "a.py").read_text(encoding="utf-8") == "one\nTWO\nthree\nfour\nfive\n"
    assert r.metadata["applied_files"] == (_abs(tmp_path, "a.py"),)
    assert "updated a.py" in r.text


def test_add_delete_move(tmp_path: Path) -> None:
    (tmp_path / "old.py").write_text("alpha\nbeta\n", encoding="utf-8")
    (tmp_path / "gone.py").write_text("bye\n", encoding="utf-8")
    patch = _patch(
        "*** Add File: new.py\n+line1\n+line2",
        "*** Update File: old.py\n*** Move to: moved.py\n@@\n-alpha\n+ALPHA",
        "*** Delete File: gone.py",
    )
    r = run_patch(tmp_path, patch)
    assert r.status is ResultStatus.OK
    assert (tmp_path / "new.py").read_text(encoding="utf-8") == "line1\nline2\n"
    assert not (tmp_path / "old.py").exists()
    assert (tmp_path / "moved.py").read_text(encoding="utf-8") == "ALPHA\nbeta\n"
    assert not (tmp_path / "gone.py").exists()
    assert set(r.metadata["applied_files"]) == {
        _abs(tmp_path, "new.py"),
        _abs(tmp_path, "old.py"),
        _abs(tmp_path, "moved.py"),
        _abs(tmp_path, "gone.py"),
    }
    assert "moved old.py -> moved.py" in r.text


# ---------------------------------------------------------------------------
# R-T13-3:跨文件非事务(不回滚、不残留),单文件原子(tmp + replace)
# ---------------------------------------------------------------------------


def test_failure_is_not_transactional_and_says_so(tmp_path: Path) -> None:
    (tmp_path / "a.py").write_text("a-old\n", encoding="utf-8")
    (tmp_path / "b.py").write_text("b-old\n", encoding="utf-8")
    patch = _patch(
        "*** Update File: a.py\n@@\n-a-old\n+a-new",
        "*** Update File: b.py\n@@\n-wrong\n+b-new",
    )
    r = run_patch(tmp_path, patch)
    assert r.status is ResultStatus.ERROR
    assert r.is_error
    assert r.error_code == "patch_failed"
    assert r.retryable is True
    assert r.error_message == _FAILURE_MESSAGE
    assert _FAILURE_MESSAGE in r.text
    # 已改的不回滚;失败文件保持完整原样(绝不出现半个文件)
    assert (tmp_path / "a.py").read_text(encoding="utf-8") == "a-new\n"
    assert (tmp_path / "b.py").read_text(encoding="utf-8") == "b-old\n"
    assert r.metadata["applied_files"] == (_abs(tmp_path, "a.py"),)
    assert r.metadata["failed_file"] == _abs(tmp_path, "b.py")
    assert r.metadata["hunk_index"] == 0
    assert r.metadata["expected_context_line"] == "wrong"
    # tmp 文件已清理(os.replace 消费);目录里不残留隐藏临时文件
    assert [child.name for child in tmp_path.iterdir() if child.name.startswith(".")] == []


# ---------------------------------------------------------------------------
# R-T13-4:完整性拒绝(与权限无关),权限判定不在这里
# ---------------------------------------------------------------------------


def test_symlink_target_is_rejected(tmp_path: Path) -> None:
    (tmp_path / "real.py").write_text("hello\n", encoding="utf-8")
    (tmp_path / "link.py").symlink_to(tmp_path / "real.py")
    r = run_patch(tmp_path, _patch("*** Update File: link.py\n@@\n-hello\n+bye"))
    assert r.status is ResultStatus.ERROR
    assert r.error_code == "invalid_input"
    assert "patch target is a symlink" in r.text
    assert (tmp_path / "real.py").read_text(encoding="utf-8") == "hello\n"


def test_escape_targets_are_rejected_before_use(tmp_path: Path) -> None:
    tool = create_apply_patch_tool(tmp_path)
    # tmp_path.parent 在同一 pytest session 内被多个用例共享:用本用例唯一的名字避免碰撞
    outside = tmp_path.parent / f"{tmp_path.name}-outside"
    outside.mkdir(exist_ok=True)
    (tmp_path / "dirlink").symlink_to(outside, target_is_directory=True)
    for bad in ("../escape.py", str(outside / "abs.py"), "dirlink/x.py"):
        with pytest.raises(ToolValidationError):
            tool.resolve_use({"patch": _patch(f"*** Add File: {bad}\n+x")})


def test_directory_target_is_rejected(tmp_path: Path) -> None:
    (tmp_path / "adir").mkdir()
    r = run_patch(tmp_path, _patch("*** Update File: adir\n@@\n-x\n+y"))
    assert r.status is ResultStatus.ERROR
    assert r.error_code == "invalid_input"
    assert "not a regular file" in r.text


def test_malformed_patch_becomes_invalid_input_result(tmp_path: Path) -> None:
    r = run_patch(tmp_path, "not a patch")
    assert r.status is ResultStatus.ERROR
    assert r.error_code == "invalid_input"
    assert r.metadata["failed_file"] is None


def test_patch_over_30000_chars_is_invalid_input(tmp_path: Path) -> None:
    tool = create_apply_patch_tool(tmp_path)
    with pytest.raises(ToolValidationError):
        tool.validate_input({"patch": "x" * 30_001})


# ---------------------------------------------------------------------------
# 解析器为纯函数:畸形输入穷举
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("text", ["", "\n\n", "   \n\t\n"])
def test_empty_or_whitespace_patch_is_rejected(text: str) -> None:
    with pytest.raises(ToolValidationError):
        parse_patch(text)


@pytest.mark.parametrize(
    "text",
    [
        # 缺 Begin/End
        "*** Update File: a.py\n@@\n-a\n+b\n",
        # 未知头
        "*** Begin Patch\n*** Frobnicate: a.py\n*** End Patch\n",
        # 缺 @@
        "*** Begin Patch\n*** Update File: a.py\n-old\n+new\n*** End Patch\n",
        # 裸 *** Move to: 没有所属 Update
        "*** Begin Patch\n*** Move to: b.py\n*** End Patch\n",
        # 只有 Begin/End,没有操作
        "*** Begin Patch\n*** End Patch\n",
    ],
)
def test_structural_malformations_are_rejected(text: str) -> None:
    with pytest.raises(ToolValidationError):
        parse_patch(text)


def test_hunk_line_count_mismatch_is_rejected() -> None:
    # 头声明 old=2/new=2,实际 old=1/new=1
    with pytest.raises(ToolValidationError):
        parse_patch("*** Begin Patch\n*** Update File: a.py\n@@ -2,2 +2,2 @@\n-a\n+b\n*** End Patch\n")


def test_parse_patch_returns_typed_operations() -> None:
    ops = parse_patch(
        _patch(
            "*** Add File: a.py\n+x",
            "*** Delete File: b.py",
            "*** Update File: c.py\n*** Move to: d.py\n@@\n-y\n+z",
        )
    )
    assert isinstance(ops[0], AddFile)
    assert ops[0].lines == ("x",)
    assert isinstance(ops[1], DeleteFile)
    assert isinstance(ops[2], UpdateFile)
    assert ops[2].move_to == "d.py"


def test_heredoc_wrapper_is_stripped(tmp_path: Path) -> None:
    (tmp_path / "a.py").write_text("x\n", encoding="utf-8")
    wrapped = (
        "<<EOF\n*** Begin Patch\n*** Update File: a.py\n@@\n-x\n+y\n*** End Patch\nEOF"
    )
    ops = parse_patch(wrapped)
    assert len(ops) == 1
    assert isinstance(ops[0], UpdateFile)
    assert ops[0].path == "a.py"
    r = run_patch(tmp_path, wrapped)
    assert r.status is ResultStatus.OK
    assert (tmp_path / "a.py").read_text(encoding="utf-8") == "y\n"


def test_apply_patch_tool_contract(tmp_path: Path) -> None:
    tool = create_apply_patch_tool(tmp_path)
    assert tool.definition.name == "apply_patch"
    assert tool.idempotent is False
    assert tool.concurrency_safe is False
    assert tool.cancellation_mode is CancellationMode.MANAGED_PROCESS
    assert tool.interrupt_behavior is InterruptBehavior.CANCEL
    assert tool.timeout_seconds == 35
    assert tool.max_model_output_bytes == 50_000
    assert tool.static_effects == frozenset(
        {ToolEffect.READ_WORKSPACE, ToolEffect.WRITE_WORKSPACE, ToolEffect.DESTRUCTIVE}
    )


# ---------------------------------------------------------------------------
# 复审 I-1:Move 目标策略(自移动/已存在都是数据丢失,必须 invalid_input)
# ---------------------------------------------------------------------------

_PATCH_METADATA_KEYS = {
    "applied_files",
    "failed_file",
    "hunk_index",
    "expected_context_line",
    "move_incomplete",
}


def test_self_move_is_rejected(tmp_path: Path) -> None:
    (tmp_path / "a.py").write_text("hello\n", encoding="utf-8")
    patch = _patch("*** Update File: a.py\n*** Move to: a.py\n@@\n-hello\n+bye")
    tool = create_apply_patch_tool(tmp_path)
    with pytest.raises(ToolValidationError) as excinfo:
        tool.resolve_use({"patch": patch})
    assert "must differ" in str(excinfo.value)
    r = run_patch(tmp_path, patch)
    assert r.status is ResultStatus.ERROR
    assert r.error_code == "invalid_input"
    assert "must differ" in r.text
    # 源文件既未被改写也未被删除
    assert (tmp_path / "a.py").read_text(encoding="utf-8") == "hello\n"
    assert r.metadata["applied_files"] == ()


def test_move_to_existing_destination_is_rejected(tmp_path: Path) -> None:
    (tmp_path / "a.py").write_text("src\n", encoding="utf-8")
    (tmp_path / "b.py").write_text("dest\n", encoding="utf-8")
    patch = _patch("*** Update File: a.py\n*** Move to: b.py\n@@\n-src\n+SRC")
    tool = create_apply_patch_tool(tmp_path)
    with pytest.raises(ToolValidationError) as excinfo:
        tool.resolve_use({"patch": patch})
    assert "already exists" in str(excinfo.value)
    r = run_patch(tmp_path, patch)
    assert r.status is ResultStatus.ERROR
    assert r.error_code == "invalid_input"
    assert "already exists" in r.text
    # 目标未被覆盖、源未被移动
    assert (tmp_path / "b.py").read_text(encoding="utf-8") == "dest\n"
    assert (tmp_path / "a.py").read_text(encoding="utf-8") == "src\n"


# ---------------------------------------------------------------------------
# 复审 #2/#3:解析计数溢出与 preflight 完整性(预检不过就不写)
# ---------------------------------------------------------------------------


def test_huge_hunk_count_is_validation_error() -> None:
    # CPython 4300 位 int 转换上限不得泄漏成裸 ValueError
    huge = "9" * 5_000
    with pytest.raises(ToolValidationError):
        parse_patch(
            "*** Begin Patch\n*** Update File: a.py\n"
            f"@@ -1,{huge} +1,1 @@\n-x\n+y\n*** End Patch\n"
        )


def test_preflight_failure_leaves_everything_untouched(tmp_path: Path) -> None:
    (tmp_path / "first.py").write_text("one\n", encoding="utf-8")
    patch = _patch(
        "*** Update File: first.py\n@@\n-one\n+ONE",
        "*** Update File: missing/a.py\n@@\n-x\n+y",
    )
    r = run_patch(tmp_path, patch)
    assert r.status is ResultStatus.ERROR
    assert r.error_code == "invalid_input"
    assert "does not exist" in r.text
    # 第二个目标父目录缺失在 preflight 就拒绝:第一个文件一个字节都没动
    assert (tmp_path / "first.py").read_text(encoding="utf-8") == "one\n"
    assert r.metadata["applied_files"] == ()


# ---------------------------------------------------------------------------
# 复审 #6/#7:Update 尺寸准入与换行风格保留
# ---------------------------------------------------------------------------


def test_oversized_update_target_is_rejected(tmp_path: Path) -> None:
    target = tmp_path / "big.txt"
    with target.open("wb") as stream:
        stream.truncate(64 * 1024 * 1024 + 1)
    r = run_patch(tmp_path, _patch("*** Update File: big.txt\n@@\n-a\n+b"))
    assert r.status is ResultStatus.ERROR
    assert r.error_code == "invalid_input"
    assert "too large" in r.text
    assert r.metadata["applied_files"] == ()


def test_update_preserves_crlf_line_endings(tmp_path: Path) -> None:
    (tmp_path / "w.py").write_bytes(b"one\r\ntwo\r\n")
    r = run_patch(tmp_path, _patch("*** Update File: w.py\n@@\n-two\n+TWO"))
    assert r.status is ResultStatus.OK
    # 只改一行,不得把整文件换行风格换掉
    assert (tmp_path / "w.py").read_bytes() == b"one\r\nTWO\r\n"


# ---------------------------------------------------------------------------
# 复审 #8:Move 半成功(已写 dest、删源失败)的 metadata 必须诚实
# ---------------------------------------------------------------------------


def test_move_cleanup_failure_reports_destination(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "a.py").write_text("x\n", encoding="utf-8")
    real_unlink = Path.unlink

    def failing_unlink(self: Path, *args: Any, **kwargs: Any) -> None:
        if self.name == "a.py":
            raise OSError("simulated unlink failure")
        real_unlink(self, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", failing_unlink)
    r = run_patch(
        tmp_path, _patch("*** Update File: a.py\n*** Move to: moved.py\n@@\n-x\n+y")
    )
    assert r.status is ResultStatus.ERROR
    assert r.error_code == "patch_failed"
    assert r.metadata["move_incomplete"] is True
    assert r.metadata["applied_files"] == (_abs(tmp_path, "moved.py"),)
    # dest 内容已写成,源文件仍在(半移动状态对调用方可见)
    assert (tmp_path / "moved.py").read_text(encoding="utf-8") == "y\n"
    assert (tmp_path / "a.py").exists()


# ---------------------------------------------------------------------------
# 复审 #10:真 pipeline.execute 级别的 R-T13-1 两面
# ---------------------------------------------------------------------------


def test_pipeline_update_patch_executes(store: Any, bus: Any, tmp_path: Path) -> None:
    (tmp_path / "a.py").write_text("old\n", encoding="utf-8")
    policy = PermissionPolicy()
    permissions = PermissionContext(workspace=str(tmp_path), policy=policy, mode="on-request")
    created = create_turn(store, turn_id="turn-apply-up", thread_id="thread-apply-up", binding={})
    ctx = ExecCtx(
        store=store,
        bus=bus,
        workspace=tmp_path,
        tools={"apply_patch": create_apply_patch_tool(tmp_path)},
        permissions=permissions,
        turn_id=created.turn_id,
        thread_id=created.thread_id,
        policy_revision=policy.revision,
    )
    result = asyncio.run(
        execute(
            ToolCall(
                tool_call_id="c1",
                tool_name="apply_patch",
                arguments={"patch": _patch("*** Update File: a.py\n@@\n-old\n+new")},
            ),
            ctx,
        )
    )
    assert result.status is ResultStatus.OK
    assert (tmp_path / "a.py").read_text(encoding="utf-8") == "new\n"


def test_pipeline_delete_patch_pauses_for_approval(
    store: Any, bus: Any, tmp_path: Path
) -> None:
    (tmp_path / "a.py").write_text("x\n", encoding="utf-8")
    policy = PermissionPolicy(tools=(ToolRule(pattern="*", decision="allow"),))
    permissions = PermissionContext(workspace=str(tmp_path), policy=policy, mode="on-request")
    created = create_turn(store, turn_id="turn-apply-del", thread_id="thread-apply-del", binding={})
    ctx = ExecCtx(
        store=store,
        bus=bus,
        workspace=tmp_path,
        tools={"apply_patch": create_apply_patch_tool(tmp_path)},
        permissions=permissions,
        turn_id=created.turn_id,
        thread_id=created.thread_id,
        policy_revision=policy.revision,
    )
    result = asyncio.run(
        execute(
            ToolCall(
                tool_call_id="c1",
                tool_name="apply_patch",
                arguments={"patch": _patch("*** Delete File: a.py")},
            ),
            ctx,
        )
    )
    assert result.status is ResultStatus.PAUSED
    # 审批前绝不执行:删除没有发生,并留下 approval_request
    assert (tmp_path / "a.py").exists()
    kinds = [item.kind for item in store.read_turn_items(created.turn_id)]
    assert ItemKind.APPROVAL_REQUEST in kinds


def test_patch_metadata_key_set_is_exact(tmp_path: Path) -> None:
    (tmp_path / "a.py").write_text("x\n", encoding="utf-8")
    success = run_patch(tmp_path, _patch("*** Update File: a.py\n@@\n-x\n+y"))
    failure = run_patch(tmp_path, _patch("*** Update File: a.py\n@@\n-wrong\n+z"))
    invalid = run_patch(tmp_path, "not a patch")
    for result in (success, failure, invalid):
        assert set(result.metadata) == _PATCH_METADATA_KEYS
    assert success.metadata["move_incomplete"] is False
