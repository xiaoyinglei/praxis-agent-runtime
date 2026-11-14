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

from praxis.protocol.sandbox import ToolExecContext
from praxis.protocol.tools import (
    CancellationMode,
    InterruptBehavior,
    ResultStatus,
    ToolEffect,
    ToolValidationError,
)
from praxis.tools.builtins import data as data_module
from praxis.tools.builtins.data import _MAX_INSPECTABLE_BYTES, create_inspect_data_file_tool

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
