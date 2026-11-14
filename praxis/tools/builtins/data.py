"""inspect_data_file 内置工具(移植旧 agent_runtime/tools/builtins/data.py 读取语义)。

支持 .xlsx/.xlsm/.csv/.tsv/.json/.pdf;输出人类可读摘要(每 sheet/页一段),
``metadata`` 是**扁平摘要**:``format``/``size_bytes``/``sha256``/``valid``/
``row_count``/``column_count``/``column_names``/``next_row``/``next_column``/
``next_sheet_offset``/``page_count``/``json_type``/``json_keys``/``truncated``/
``truncated_cells``。多 sheet 时标量字段取自第一个返回的表,每张表的逐页游标
在 content 的各段里如实给出。

R-T12-2(B8 不带病移植):``column_names`` 就是 1-based ``start_row`` 这一行的
单元格文本,由迭代器行号**直接定位**,绝不叠加任何 skiprows/header-1 语义
(旧 agent_runtime/file_manifest.py:637-645,700-704 的双重偏移缺陷不复现)。
不猜表头、不求值公式、不做合并单元格填充。

R-T12-1(不堵事件循环):解析主体在 ``await asyncio.to_thread(...)`` 里执行,
事件循环保持可调度(并发任务、超时定时器、取消信号都能送达)。**线程不可杀**:
超时/取消后解析必须等它自然结束——继承 pipeline.run_with_timeout 的既有限制,
工具用 ``InterruptBehavior.FINISH_CURRENT`` 如实声明。

阈值口径(唯一一套):``_MAX_GRID_CELLS`` 是**跨本次调用所有 sheet 共享**的
预算;``max_rows`` 实际值 = ``min(max_rows, _MAX_GRID_CELLS // max_columns)``。
截断时 ``truncated_cells`` 记录源 ``(row, col)``:被字符预算缩短的单元格,以及
翻页边界上第一个未返回的单元格;``next_row``/``next_column``/``next_sheet_offset``
一律是可继续翻页的真值(绝不出现"截断但不前进")。错误码:
``data_inspection_failed``(超 200 MiB / 扩展名不支持 / 解析异常,消息经
``_bounded_error`` 空白折叠且 ≤1000 字符)、``not_found``。
"""

from __future__ import annotations

import asyncio
import csv
import hashlib
import json
import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime, time
from pathlib import Path
from typing import Any, Literal, cast

import fitz  # type: ignore[import-untyped]
from openpyxl import load_workbook
from openpyxl.worksheet.formula import ArrayFormula, DataTableFormula

from praxis.protocol.json import JsonValue
from praxis.protocol.sandbox import ToolExecContext
from praxis.protocol.tools import (
    CancellationMode,
    InterruptBehavior,
    ResolvedUse,
    ResultStatus,
    Tool,
    ToolContentBlock,
    ToolDefinition,
    ToolEffect,
    ToolResult,
    ToolTarget,
    ToolValidationError,
    json_schema_input,
)
from praxis.tools.builtins.paths import checked_path

# run() 返回的占位 call id 与 filesystem.py 同一管道约定(protocol 要求非空)。
PENDING_TOOL_CALL_ID = "pending"

_MAX_INSPECTABLE_BYTES = 200 * 1024 * 1024
_MAX_CELL_CHARACTERS = 2_000
_MAX_GRID_CELLS = 2_000
_MAX_GRID_CHARACTERS = 32_000
_MAX_ERROR_CHARACTERS = 1_000
_CSV_SNIFF_CHARACTERS = 16_384

type _Format = Literal["xlsx", "xlsm", "pdf", "csv", "tsv", "json", "unsupported"]


@dataclass
class _GridBudget:
    """本次调用所有 sheet 共享的单元格与字符预算(唯一口径)。"""

    cells: int = _MAX_GRID_CELLS
    characters: int = _MAX_GRID_CHARACTERS

    def text(self, value: object) -> tuple[str, bool]:
        rendered = _cell_text(value)
        limit = min(_MAX_CELL_CHARACTERS, self.characters)
        if limit == 0:
            return "", bool(rendered)
        shortened = len(rendered) > limit
        result = rendered[: max(0, limit - 1)] + "…" if shortened else rendered
        self.characters = max(0, self.characters - len(result))
        return result, shortened


@dataclass(frozen=True, slots=True)
class _TablePreview:
    name: str
    start_row: int
    start_column: int
    width: int
    rows: list[list[str]]
    row_numbers: list[int]
    column_names: list[str]
    row_count: int | None
    column_count: int | None
    next_row: int | None
    next_column: int | None
    truncated_cells: list[tuple[int, int]]
    truncated: bool


@dataclass(frozen=True, slots=True)
class _PagePreview:
    page_number: int
    text: str
    truncated: bool


@dataclass(frozen=True, slots=True)
class _Inspection:
    content: str
    metadata: dict[str, Any]
    error_code: str | None = None


_INSPECT_INPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "path": {
            "type": "string",
            "minLength": 1,
            "maxLength": 4_096,
            "description": "Workspace-relative .xlsx/.xlsm/.csv/.tsv/.json/.pdf file.",
        },
        "max_rows": {
            "type": "integer",
            "minimum": 1,
            "maximum": 100,
            "description": "Maximum raw rows per grid, including titles, headers and blanks (default 10).",
        },
        "start_row": {
            "type": "integer",
            "minimum": 1,
            "description": "One-based source row (worksheet row or logical CSV/TSV record).",
        },
        "start_column": {
            "type": "integer",
            "minimum": 1,
            "maximum": 16_384,
            "description": "One-based source column.",
        },
        "max_columns": {
            "type": "integer",
            "minimum": 1,
            "maximum": 100,
            "description": "Maximum source columns per grid (default 20).",
        },
        "sheet_name": {
            "type": "string",
            "minLength": 1,
            "maxLength": 31,
            "description": "Read only this Excel worksheet; otherwise page through worksheets.",
        },
        "sheet_offset": {
            "type": "integer",
            "minimum": 0,
            "description": "Zero-based worksheet offset when sheet_name is omitted.",
        },
        "max_sheets": {
            "type": "integer",
            "minimum": 1,
            "maximum": 100,
            "description": "Maximum worksheets returned from one workbook (default 20).",
        },
        "max_pages": {
            "type": "integer",
            "minimum": 1,
            "maximum": 100,
            "description": "Maximum PDF pages returned (default 10).",
        },
        "max_characters": {
            "type": "integer",
            "minimum": 100,
            "maximum": 100_000,
            "description": (
                "Maximum characters per PDF page/JSON preview, or combined grid cell text "
                "(hard cap 32000; default 8000)."
            ),
        },
    },
    "required": ("path",),
    "additionalProperties": False,
}

_validate_inspect_schema = json_schema_input(_INSPECT_INPUT_SCHEMA)


def _validate_inspect_input(arguments: Mapping[str, JsonValue]) -> Mapping[str, JsonValue]:
    args = _validate_inspect_schema(arguments)
    # "同时给"按原始参数键判定:schema 不注入默认值,缺省 sheet_offset 不构成冲突。
    if args.get("sheet_name") is not None and "sheet_offset" in arguments:
        raise ToolValidationError(
            path="$.sheet_offset",
            message="sheet_name and sheet_offset cannot be combined",
        )
    return args


def create_inspect_data_file_tool(workspace: Path) -> Tool:
    validate_input = _validate_inspect_input

    async def run(
        arguments: Mapping[str, JsonValue], exec_ctx: ToolExecContext  # 进程内工具:不消费沙箱
    ) -> ToolResult:
        args = validate_input(arguments)
        target = checked_path(workspace, cast("str", args["path"]))
        # 解析是 CPU+IO 重活:放进线程,循环保持可调度(R-T12-1)。
        inspection = await asyncio.to_thread(_inspect_data_file, target, args)
        failed = inspection.error_code is not None
        return _result(
            "inspect_data_file",
            content=inspection.content,
            status=ResultStatus.ERROR if failed else ResultStatus.OK,
            metadata=inspection.metadata,
            truncated=False if failed else inspection.metadata.get("truncated") is True,
            error_code=inspection.error_code,
        )

    def resolve_use(arguments: Mapping[str, JsonValue]) -> ResolvedUse:
        target = checked_path(workspace, cast("str", arguments["path"]))
        return ResolvedUse(
            effects=frozenset({ToolEffect.READ_WORKSPACE}),
            targets=(ToolTarget(kind="path", value=str(target)),),
        )

    return Tool(
        definition=ToolDefinition(
            name="inspect_data_file",
            description=(
                "Inspect a workspace spreadsheet (.xlsx/.xlsm), PDF, CSV, TSV, or JSON "
                "file without returning raw binary bytes. The result includes a "
                "runtime-computed SHA-256 plus raw worksheet/CSV/TSV grids, extracted PDF "
                "text, or a JSON preview. Grid rows preserve titles, units, repeated/multiple "
                "header rows and blanks; no row is assumed to be a header (column_names is "
                "exactly the start_row row) and merged-cell blanks are not filled. Formula "
                "expressions are never evaluated. Use start_row/start_column with "
                "max_rows/max_columns to page a grid (1-based source coordinates; CSV rows "
                "are logical records), sheet_name to select one worksheet, or sheet_offset "
                "with max_sheets for worksheet pages. next_row/next_column continue this "
                "grid page; next_sheet_offset continues worksheet pages. One call shares a "
                "2000-cell budget across all returned sheets; max_characters bounds combined "
                "cell text with a hard cap of 32000 characters, and truncated_cells lists "
                "source (row, column) coordinates shortened by limits or left beyond the "
                "page (use bash to read values longer than the 2000-character cell cap in "
                "full). Files above 200 MiB are rejected. Parsing runs in an in-process "
                "worker thread; threads cannot be killed, so after a timeout or cancellation "
                "the parse must finish naturally before the turn can continue."
            ),
            input_schema=_INSPECT_INPUT_SCHEMA,
        ),
        validate_input=validate_input,
        run=run,
        static_effects=frozenset({ToolEffect.READ_WORKSPACE}),
        resolve_use=resolve_use,
        idempotent=True,
        concurrency_safe=True,
        cancellation_mode=CancellationMode.COOPERATIVE,
        interrupt_behavior=InterruptBehavior.FINISH_CURRENT,
        timeout_seconds=60,
        max_model_output_bytes=500_000,
    )


def _result(
    tool_name: str,
    *,
    content: str,
    status: ResultStatus,
    metadata: Mapping[str, Any],
    truncated: bool = False,
    error_code: str | None = None,
) -> ToolResult:
    is_error = status in {ResultStatus.ERROR, ResultStatus.TIMEOUT}
    return ToolResult(
        tool_call_id=PENDING_TOOL_CALL_ID,
        tool_name=tool_name,
        content=(ToolContentBlock(type="text", data={"text": content}),),
        status=status,
        is_error=is_error,
        error_code=error_code if is_error else None,
        truncated=truncated,
        metadata=metadata,
    )


# ---------------------------------------------------------------------------
# 解析入口(同步,由 run 放进线程)
# ---------------------------------------------------------------------------


def _inspect_data_file(target: Path, requests: Mapping[str, JsonValue]) -> _Inspection:
    path_value = cast("str", requests["path"])
    file_format = _format_from_path(target)
    try:
        info = target.stat()
    except FileNotFoundError:
        return _not_found(path_value, file_format)
    except OSError as exc:
        return _failure(
            file_format, 0, "", f"data inspection failed: {_bounded_error(exc)}"
        )
    if not target.is_file():
        return _not_found(path_value, file_format)

    size_bytes = info.st_size
    if size_bytes > _MAX_INSPECTABLE_BYTES:
        return _failure(
            file_format,
            size_bytes,
            "",
            (
                f"data file exceeds the {_MAX_INSPECTABLE_BYTES // (1024 * 1024)} MiB "
                "inspection limit; create a bounded sample with bash"
            ),
        )
    if file_format == "unsupported":
        return _failure(
            file_format,
            size_bytes,
            "",
            "unsupported data format; expected .xlsx, .xlsm, .pdf, .csv, .tsv, or .json",
        )

    digest = ""
    try:
        digest = _file_sha256(target)
        if file_format in ("xlsx", "xlsm"):
            return _inspect_workbook(target, path_value, requests, file_format, size_bytes, digest)
        if file_format == "csv":
            return _inspect_delimited(target, path_value, requests, "csv", size_bytes, digest)
        if file_format == "tsv":
            return _inspect_delimited(target, path_value, requests, "tsv", size_bytes, digest)
        if file_format == "pdf":
            return _inspect_pdf(target, path_value, requests, size_bytes, digest)
        return _inspect_json(target, path_value, requests, size_bytes, digest)
    except Exception as exc:  # noqa: BLE001 — 解析异常一律归一为 data_inspection_failed
        return _failure(file_format, size_bytes, digest, _bounded_error(exc))


def _inspect_workbook(
    target: Path,
    path_value: str,
    requests: Mapping[str, JsonValue],
    file_format: Literal["xlsx", "xlsm"],
    size_bytes: int,
    digest: str,
) -> _Inspection:
    sheet_name = _optional_str(requests, "sheet_name")
    sheet_offset = _int_argument(requests, "sheet_offset", 0)
    max_sheets = _int_argument(requests, "max_sheets", 20)
    workbook = load_workbook(target, read_only=True, data_only=False, keep_links=False)
    try:
        sheet_names = list(workbook.sheetnames)
        next_sheet: int | None = None
        if sheet_name is not None:
            selected = [workbook[sheet_name]]
        else:
            end = sheet_offset + max_sheets
            selected = list(workbook.worksheets[sheet_offset:end])
            next_sheet = end if end < len(sheet_names) else None
        budget = _GridBudget(
            characters=min(_int_argument(requests, "max_characters", 8_000), _MAX_GRID_CHARACTERS)
        )
        previews: list[_TablePreview] = []
        for index, worksheet in enumerate(selected):
            if budget.cells == 0:
                # 共享预算耗尽:剩余 sheet 留给下一次 sheet_offset 页。
                if sheet_name is None:
                    next_sheet = sheet_offset + index
                break
            worksheet.reset_dimensions()
            records = worksheet.iter_rows()
            try:
                previews.append(
                    _preview_grid(worksheet.title, records, requests, excel=True, budget=budget)
                )
            finally:
                records.close()
        # 同样的"截断必须前进"纪律:sheet 游标只会往后指。
        assert next_sheet is None or next_sheet > sheet_offset, "sheet paging must advance"
    finally:
        workbook.close()
    lines = [
        _header_line(file_format, path_value, size_bytes, digest),
        "sheets: " + ", ".join(sheet_names),
    ]
    for preview in previews:
        lines.extend(_render_grid(preview, f'sheet "{preview.name}"'))
    if next_sheet is not None:
        lines.append(f"[next_sheet_offset={next_sheet}]")
    return _Inspection(
        content="\n".join(lines),
        metadata=_tables_metadata(file_format, size_bytes, digest, previews, next_sheet),
    )


def _inspect_delimited(
    target: Path,
    path_value: str,
    requests: Mapping[str, JsonValue],
    file_format: Literal["csv", "tsv"],
    size_bytes: int,
    digest: str,
) -> _Inspection:
    delimiter = "\t" if file_format == "tsv" else ","
    with target.open("r", encoding="utf-8-sig", newline="") as stream:
        sample = stream.read(_CSV_SNIFF_CHARACTERS)
        stream.seek(0)
        if file_format == "csv" and sample:
            try:
                delimiter = csv.Sniffer().sniff(sample, delimiters=",;\t|").delimiter
            except csv.Error:
                pass
        preview = _preview_grid(
            target.name,
            csv.reader(stream, delimiter=delimiter, strict=True),
            requests,
            excel=False,
            budget=_GridBudget(
                characters=min(
                    _int_argument(requests, "max_characters", 8_000), _MAX_GRID_CHARACTERS
                )
            ),
        )
    lines = [
        _header_line(file_format, path_value, size_bytes, digest),
        *_render_grid(preview, f'table "{preview.name}"'),
    ]
    return _Inspection(
        content="\n".join(lines),
        metadata=_tables_metadata(file_format, size_bytes, digest, [preview], None),
    )


def _inspect_pdf(
    target: Path,
    path_value: str,
    requests: Mapping[str, JsonValue],
    size_bytes: int,
    digest: str,
) -> _Inspection:
    max_pages = _int_argument(requests, "max_pages", 10)
    max_characters = _int_argument(requests, "max_characters", 8_000)
    document = fitz.open(target)
    try:
        if document.needs_pass:
            raise ValueError("encrypted PDF requires a password")
        page_count = document.page_count
        pages: list[_PagePreview] = []
        for page_index in range(min(page_count, max_pages)):
            text = document.load_page(page_index).get_text("text")
            pages.append(
                _PagePreview(
                    page_number=page_index + 1,
                    text=text[:max_characters],
                    truncated=len(text) > max_characters,
                )
            )
    finally:
        document.close()
    lines = [_header_line("pdf", path_value, size_bytes, digest), f"page_count={page_count}"]
    for page in pages:
        note = ", truncated" if page.truncated else ""
        lines.append(
            f"[page {page.page_number}/{page_count}: {len(page.text)} characters{note}]"
        )
        lines.append(page.text)
    if page_count > len(pages):
        lines.append(f"[only {len(pages)} of {page_count} pages returned; increase max_pages]")
    truncated = page_count > len(pages) or any(page.truncated for page in pages)
    metadata = _empty_metadata("pdf", size_bytes, digest)
    metadata.update(
        {
            "page_count": page_count,
            "truncated": truncated,
        }
    )
    return _Inspection(content="\n".join(lines), metadata=metadata)


def _inspect_json(
    target: Path,
    path_value: str,
    requests: Mapping[str, JsonValue],
    size_bytes: int,
    digest: str,
) -> _Inspection:
    max_characters = _int_argument(requests, "max_characters", 8_000)
    with target.open("r", encoding="utf-8-sig") as stream:
        value = json.load(stream)
    json_type = _json_type(value)
    json_keys = sorted(str(key) for key in value) if isinstance(value, dict) else []
    rendered = json.dumps(
        value, ensure_ascii=False, indent=2, allow_nan=False, default=_json_default
    )
    truncated = len(rendered) > max_characters
    lines = [
        _header_line("json", path_value, size_bytes, digest),
        f"json_type={json_type}" + (f" keys={', '.join(json_keys)}" if json_keys else ""),
        rendered[:max_characters],
    ]
    if truncated:
        lines.append("[preview truncated; increase max_characters]")
    metadata = _empty_metadata("json", size_bytes, digest)
    metadata.update(
        {
            "json_type": json_type,
            "json_keys": json_keys,
            "truncated": truncated,
        }
    )
    return _Inspection(content="\n".join(lines), metadata=metadata)


# ---------------------------------------------------------------------------
# 网格限界与渲染
# ---------------------------------------------------------------------------


def _preview_grid(
    name: str,
    records: Iterable[Sequence[object]],
    requests: Mapping[str, JsonValue],
    *,
    excel: bool,
    budget: _GridBudget,
) -> _TablePreview:
    """按 1-based 源坐标限界网格;不解释行语义、不信任 XLSX 声明尺寸。"""
    start_row = _int_argument(requests, "start_row", 1)
    start_column = _int_argument(requests, "start_column", 1)
    max_rows = _int_argument(requests, "max_rows", 10)
    max_columns = _int_argument(requests, "max_columns", 20)

    width_limit = min(max_columns, budget.cells)
    row_limit = min(max_rows, budget.cells // width_limit)
    assert width_limit >= 1 and row_limit >= 1  # 共享预算只在完全耗尽时才停

    rows: list[list[str]] = []
    numbers: list[int] = []
    shortened: list[tuple[int, int]] = []
    total_rows = total_columns = page_columns = 0
    more_rows = False
    for row_number, record in enumerate(records, start=1):
        if row_number >= start_row and len(rows) == row_limit:
            more_rows = True
            break
        total_rows = row_number
        total_columns = max(total_columns, len(record))
        if row_number < start_row:
            continue
        page_columns = max(page_columns, len(record))
        values: list[str] = []
        begin = start_column - 1
        for column, cell in enumerate(record[begin : begin + width_limit], start_column):
            value = getattr(cell, "value", None) if excel else cell
            rendered, was_shortened = budget.text(value)
            if was_shortened:
                shortened.append((row_number, column))
            values.append(rendered)
        rows.append(values)
        numbers.append(row_number)

    width = min(width_limit, max(0, page_columns - start_column + 1))
    budget.cells -= len(rows) * width
    column_names = list(rows[0]) if rows else []
    for values in rows:
        values.extend([""] * (width - len(values)))
    next_column = start_column + width if page_columns >= start_column + width else None
    next_row = start_row + len(rows) if more_rows else None
    # Task 10 教训:截断必须伴随前进的游标;断言把"截断但不前进"变成不可能。
    if more_rows:
        assert next_row is not None and next_row > start_row, "row truncation must advance"
    if next_column is not None:
        assert next_column > start_column, "column truncation must advance"
    truncated_cells = list(shortened)
    if more_rows:
        truncated_cells.append((cast(int, next_row), start_column))
    if next_column is not None:
        truncated_cells.append((start_row, next_column))
    return _TablePreview(
        name=name,
        start_row=start_row,
        start_column=start_column,
        width=width,
        rows=rows,
        row_numbers=numbers,
        column_names=column_names,
        row_count=None if more_rows else total_rows,
        column_count=None if more_rows else total_columns,
        next_row=next_row,
        next_column=next_column,
        truncated_cells=truncated_cells,
        truncated=more_rows or next_column is not None or bool(shortened),
    )


def _render_grid(preview: _TablePreview, label: str) -> list[str]:
    if preview.row_numbers:
        rows_span = f"rows {preview.row_numbers[0]}-{preview.row_numbers[-1]}"
        columns_span = (
            f"columns {preview.start_column}-{preview.start_column + preview.width - 1}"
            if preview.width
            else f"columns from {preview.start_column}"
        )
    else:
        rows_span = f"no rows from row {preview.start_row}"
        columns_span = f"columns from {preview.start_column}"
    row_count = "unknown" if preview.row_count is None else str(preview.row_count)
    column_count = "unknown" if preview.column_count is None else str(preview.column_count)
    lines = [
        f"[{label}: {rows_span}, {columns_span}, "
        f"row_count={row_count}, column_count={column_count}]"
    ]
    if preview.column_names:
        lines.append("header: " + " | ".join(preview.column_names))
    for number, values in zip(preview.row_numbers, preview.rows, strict=True):
        lines.append(f"{number} | " + " | ".join(values))
    cursors: list[str] = []
    if preview.next_row is not None:
        cursors.append(f"next_row={preview.next_row}")
    if preview.next_column is not None:
        cursors.append(f"next_column={preview.next_column}")
    if cursors:
        lines.append("[" + ", ".join(cursors) + "]")
    return lines


def _header_line(file_format: str, path_value: str, size_bytes: int, digest: str) -> str:
    return (
        f"format={file_format} path={path_value} "
        f"size_bytes={size_bytes} sha256={digest}"
    )


def _tables_metadata(
    file_format: str,
    size_bytes: int,
    digest: str,
    previews: Sequence[_TablePreview],
    next_sheet: int | None,
) -> dict[str, Any]:
    """扁平摘要:标量字段取自第一个返回的表;truncated/truncated_cells 覆盖全部表。"""
    primary = previews[0] if previews else None
    truncated_cells: list[tuple[int, int]] = []
    for preview in previews:
        truncated_cells.extend(preview.truncated_cells)
    truncated = next_sheet is not None or any(preview.truncated for preview in previews)
    metadata = _empty_metadata(file_format, size_bytes, digest)
    metadata.update(
        {
            "row_count": primary.row_count if primary is not None else None,
            "column_count": primary.column_count if primary is not None else None,
            "column_names": list(primary.column_names) if primary is not None else [],
            "next_row": primary.next_row if primary is not None else None,
            "next_column": primary.next_column if primary is not None else None,
            "next_sheet_offset": next_sheet,
            "truncated": truncated,
            "truncated_cells": truncated_cells,
        }
    )
    return metadata


def _empty_metadata(file_format: str, size_bytes: int, digest: str) -> dict[str, Any]:
    return {
        "format": file_format,
        "size_bytes": size_bytes,
        "sha256": digest,
        "valid": True,
        "row_count": None,
        "column_count": None,
        "column_names": [],
        "next_row": None,
        "next_column": None,
        "next_sheet_offset": None,
        "page_count": None,
        "json_type": None,
        "json_keys": [],
        "truncated": False,
        "truncated_cells": [],
    }


def _failure(
    file_format: _Format,
    size_bytes: int,
    digest: str,
    message: str,
) -> _Inspection:
    bounded = _bounded_error_text(message)
    metadata = _empty_metadata(file_format, size_bytes, digest)
    metadata["valid"] = False
    return _Inspection(content=bounded, metadata=metadata, error_code="data_inspection_failed")


def _not_found(path_value: str, file_format: _Format) -> _Inspection:
    message = f"workspace data file not found: {path_value}"
    metadata = _empty_metadata(file_format, 0, "")
    metadata["valid"] = False
    return _Inspection(content=message, metadata=metadata, error_code="not_found")


# ---------------------------------------------------------------------------
# 小工具
# ---------------------------------------------------------------------------


def _int_argument(
    requests: Mapping[str, JsonValue], name: str, default: int
) -> int:
    value = requests.get(name)
    return default if value is None else cast("int", value)


def _optional_str(requests: Mapping[str, JsonValue], name: str) -> str | None:
    value = requests.get(name)
    return None if value is None else cast("str", value)


def _format_from_path(path: Path) -> _Format:
    suffix = path.suffix.casefold()
    if suffix == ".xlsx":
        return "xlsx"
    if suffix == ".xlsm":
        return "xlsm"
    if suffix == ".pdf":
        return "pdf"
    if suffix == ".csv":
        return "csv"
    if suffix == ".tsv":
        return "tsv"
    if suffix == ".json":
        return "json"
    return "unsupported"


def _cell_text(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, ArrayFormula):
        rendered = value.text or json.dumps({"formula_type": "array", **dict(value)}, sort_keys=True)
    elif isinstance(value, DataTableFormula):
        rendered = json.dumps({"formula_type": "data_table", **dict(value)}, sort_keys=True)
    elif isinstance(value, (datetime, date, time)):
        rendered = value.isoformat()
    elif isinstance(value, float) and not math.isfinite(value):
        rendered = str(value)
    elif isinstance(value, bytes):
        rendered = f"<binary value: {len(value)} bytes>"
    else:
        rendered = str(value)
    return rendered


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _bounded_error(error: Exception) -> str:
    message = " ".join(str(error).split()) or type(error).__name__
    return message[:_MAX_ERROR_CHARACTERS]


def _bounded_error_text(message: str) -> str:
    collapsed = " ".join(message.split())
    return collapsed[:_MAX_ERROR_CHARACTERS]


def _json_type(value: object) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, dict):
        return "object"
    if isinstance(value, list):
        return "array"
    if isinstance(value, str):
        return "string"
    if isinstance(value, (int, float)):
        return "number"
    return type(value).__name__


def _json_default(value: object) -> str:
    if isinstance(value, (datetime, date, time)):
        return value.isoformat()
    return str(value)


__all__ = ["create_inspect_data_file_tool"]