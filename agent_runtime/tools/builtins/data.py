from __future__ import annotations

import asyncio
import csv
import hashlib
import json
import math
import os
import signal
import sys
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime, time
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from agent_runtime.tools.tool import (
    CancellationMode,
    InterruptBehavior,
    JsonValue,
    NormalizedToolOutput,
    ResolvedToolUse,
    Tool,
    ToolDefinition,
    ToolEffect,
    ToolTarget,
    json_schema_output,
    pydantic_input,
)
from agent_runtime.workspace import WorkspacePathError, WorkspaceRuntime

_MAX_INSPECTABLE_BYTES = 200 * 1024 * 1024
_MAX_CELL_CHARACTERS = 2_000
_MAX_GRID_CELLS = 2_000
_MAX_GRID_CHARACTERS = 32_000
_MAX_WORKER_ERROR_CHARACTERS = 1_024
_WORKER_FAILURE_MESSAGE = "Data inspection worker failed; no preview was produced."
_WORKER_ERRORS: dict[str, type[Exception]] = {
    "FileNotFoundError": FileNotFoundError,
    "PermissionError": PermissionError,
    "WorkspacePathError": WorkspacePathError,
}


@dataclass
class _GridBudget:
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


class InspectDataFileInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    path: str = Field(
        min_length=1,
        max_length=4096,
        description=(
            "Workspace-relative .xlsx, .xlsm, .pdf, .csv, .tsv, or .json "
            "file to inspect."
        ),
    )
    max_rows: int = Field(
        default=10,
        ge=1,
        le=100,
        description="Maximum raw rows per grid, including titles, headers and blank rows.",
    )
    start_row: int = Field(default=1, ge=1, description="One-based worksheet row or logical CSV/TSV record.")
    start_column: int = Field(default=1, ge=1, le=16_384, description="One-based source column.")
    max_columns: int = Field(default=20, ge=1, le=100, description="Maximum source columns per grid.")
    sheet_name: str | None = Field(default=None, min_length=1, max_length=31,
                                  description="Read only this Excel worksheet; otherwise page through worksheets.")
    sheet_offset: int = Field(default=0, ge=0, description="Zero-based worksheet offset when sheet_name is omitted.")
    max_sheets: int = Field(
        default=20,
        ge=1,
        le=100,
        description="Maximum worksheets returned from one workbook.",
    )
    max_pages: int = Field(
        default=10,
        ge=1,
        le=100,
        description="Maximum PDF pages whose extracted text is returned.",
    )
    max_characters: int = Field(
        default=8_000,
        ge=100,
        le=100_000,
        description=(
            "Maximum characters per PDF page/JSON preview, or combined grid text (hard cap 32000)."
        ),
    )


class TablePreview(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    rows: list[list[str]]
    row_numbers: list[int]
    column_numbers: list[int]
    cell_types: list[list[str]]
    number_formats: list[list[str]]
    row_count: int | None = Field(ge=0, description="Total source rows, or null when the scan has not reached EOF.")
    column_count: int | None = Field(ge=0, description="Total source columns, or null before EOF.")
    next_row: int | None
    next_column: int | None
    truncated_cells: list[tuple[int, int]]
    truncated: bool


class PagePreview(BaseModel):
    model_config = ConfigDict(extra="forbid")

    page_number: int = Field(ge=1)
    text: str
    truncated: bool


class InspectDataFileOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    path: str
    format: Literal["xlsx", "xlsm", "pdf", "csv", "tsv", "json", "unsupported"]
    size_bytes: int = Field(ge=0)
    sha256: str
    valid: bool
    summary: str
    tables: list[TablePreview]
    sheet_names: list[str]
    next_sheet_offset: int | None = None
    page_count: int | None
    pages: list[PagePreview]
    json_type: str | None
    json_keys: list[str]
    json_preview: str | None
    truncated: bool
    error: str | None


_INSPECT_INPUT_SCHEMA, _validate_inspect_input = pydantic_input(
    InspectDataFileInput
)
_INSPECT_OUTPUT_SCHEMA, _unused_inspect_output_validator = pydantic_input(
    InspectDataFileOutput
)


def create_inspect_data_file_tool(workspace: WorkspaceRuntime) -> Tool:
    async def run(
        arguments: Mapping[str, JsonValue],
    ) -> InspectDataFileOutput:
        request = InspectDataFileInput.model_validate(arguments)
        return await _inspect_in_worker(workspace, request)

    return Tool(
        definition=ToolDefinition(
            name="inspect_data_file",
            description=(
                "Inspect a workspace spreadsheet, PDF, CSV, TSV, or JSON file "
                "without returning raw binary bytes. The result includes a "
                "runtime-computed SHA-256 plus raw worksheet/CSV/TSV grids, extracted PDF text, or a JSON preview. "
                "Grid rows preserve titles, units, repeated/multiple header rows and blanks; no row is assumed "
                "to be a header and merged-cell blanks are not filled. row_numbers and column_numbers give "
                "one-based source coordinates (CSV rows are logical records, including quoted newlines). "
                "Use start_row/start_column with max_rows/max_columns to page, sheet_name to select a sheet, "
                "or sheet_offset with max_sheets for sheet pages. next_row/next_column refer to additional "
                "rows/columns in this grid page; next_sheet_offset continues sheet pages. null counts mean "
                "the scan has not reached EOF. Counts include every raw row, including any headers. "
                "Cell text is stored content, not Excel display formatting. Excel cell_types retain formula/error "
                "types; number_formats preserves Excel format codes (empty means General/not applicable). Formula "
                "expressions are never evaluated or replaced by cached blanks. One call returns at most 2000 grid "
                "cells; max_characters bounds combined cell text and formats, with a hard cap of 32000 characters. "
                "truncated_cells lists source row/column pairs with shortened text/format; retrieve full values "
                "with execute_python and openpyxl if needed. Empty/out-of-range grids provide no content evidence. "
                "Parsing runs in an isolated worker that is terminated on cancellation or timeout. "
                "A successful inspection establishes parseability and the displayed content, not analysis correctness."
            ),
            input_schema=_INSPECT_INPUT_SCHEMA,
        ),
        validate_input=_validate_inspect_input,
        run=run,
        normalize_output=_normalize_inspect_output,
        output_schema=_INSPECT_OUTPUT_SCHEMA,
        static_effects=frozenset({ToolEffect.READ_WORKSPACE}),
        resolve_use=lambda arguments: _resolve_inspect_use(workspace, arguments),
        execution_revision="builtin-inspect-data-file-v4-worker-errors",
        idempotent=True,
        concurrency_safe=True,
        cancellation_mode=CancellationMode.COOPERATIVE,
        interrupt_behavior=InterruptBehavior.CANCEL,
        timeout_seconds=60.0,
        max_model_output_bytes=500_000,
    )


async def _inspect_in_worker(
    workspace: WorkspaceRuntime, request: InspectDataFileInput,
) -> InspectDataFileOutput:
    workspace.ensure_readable(workspace.resolve_path(request.path))
    package_root = Path(__file__).resolve().parents[3]
    bootstrap = (
        f"import sys; sys.path.insert(0, {str(package_root)!r}); "
        "from agent_runtime.tools.builtins.data import _inspection_worker; _inspection_worker()"
    )
    permissions = workspace.permissions
    payload = json.dumps({
        "root": str(workspace.root.resolve()),
        "permissions": permissions.model_dump(mode="json") if permissions is not None else None,
        "request": request.model_dump(mode="json"),
    }).encode()
    process = await asyncio.create_subprocess_exec(
        sys.executable, "-I", "-B", "-c", bootstrap,
        stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.DEVNULL, start_new_session=True,
        env={"PATH": os.defpath, "LANG": "C.UTF-8"}, cwd="/",
    )
    try:
        output, _ = await process.communicate(payload)
        if process.returncode:
            raise RuntimeError(_WORKER_FAILURE_MESSAGE)
        try:
            response = json.loads(output)
        except (ValueError, UnicodeError):
            raise RuntimeError(_WORKER_FAILURE_MESSAGE) from None
        if isinstance(response, dict) and "worker_error" in response:
            error = response["worker_error"]
            if isinstance(error, dict):
                kind, message = error.get("type"), error.get("message")
                if isinstance(kind, str) and kind in _WORKER_ERRORS and isinstance(message, str):
                    raise _WORKER_ERRORS[kind](message[:_MAX_WORKER_ERROR_CHARACTERS])
            raise RuntimeError(_WORKER_FAILURE_MESSAGE)
        try:
            return InspectDataFileOutput.model_validate(response)
        except ValueError:
            raise RuntimeError(_WORKER_FAILURE_MESSAGE) from None
    finally:
        if process.returncode is None:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        cleanup = asyncio.create_task(process.wait())
        interrupted = False
        while not cleanup.done():
            try:
                await asyncio.shield(cleanup)
            except asyncio.CancelledError:
                # Repeated interrupts must not abandon a still-unreaped parser.
                interrupted = True
                continue
        cleanup.result()
        if interrupted:
            raise asyncio.CancelledError


def _inspection_worker() -> None:
    from agent_runtime.permission_policy import ResolvedPermissions

    payload = json.load(sys.stdin)
    raw_permissions = payload["permissions"]
    permissions = None if raw_permissions is None else ResolvedPermissions.model_validate(raw_permissions)
    workspace = WorkspaceRuntime(
        root=Path(payload["root"]), is_temporary=False,
        permission_provider=(lambda: permissions) if permissions else None,
    )
    try:
        result = _inspect_data_file(workspace, InspectDataFileInput.model_validate(payload["request"]))
    except (FileNotFoundError, PermissionError, WorkspacePathError) as error:
        sys.stdout.write(json.dumps({"worker_error": {
            "type": type(error).__name__, "message": str(error)[:_MAX_WORKER_ERROR_CHARACTERS],
        }}))
    except Exception:
        # Only known resource errors cross the boundary. Parser/runtime failures
        # must not leak arbitrary exception details, stderr or worker tracebacks.
        sys.stdout.write(json.dumps({"worker_error": {"type": "worker_failed"}}))
    else:
        sys.stdout.write(result.model_dump_json())


def _inspect_data_file(
    workspace: WorkspaceRuntime,
    request: InspectDataFileInput,
) -> InspectDataFileOutput:
    target = workspace.ensure_readable(
        workspace.resolve_path(request.path)
    )
    if not target.is_file():
        raise FileNotFoundError(
            f"workspace data file not found: {request.path}"
        )
    size_bytes = target.stat().st_size
    if size_bytes > _MAX_INSPECTABLE_BYTES:
        return _inspection_failure(
            request=request,
            target=target,
            file_format=_format_from_path(target),
            size_bytes=size_bytes,
            error=(
                "data file exceeds the 200 MiB inspection limit; create a "
                "bounded sample with execute_python"
            ),
        )

    file_format = _format_from_path(target)
    try:
        if file_format == "xlsx":
            return _inspect_workbook(
                target,
                request=request,
                file_format="xlsx",
                size_bytes=size_bytes,
            )
        if file_format == "xlsm":
            return _inspect_workbook(
                target,
                request=request,
                file_format="xlsm",
                size_bytes=size_bytes,
            )
        if file_format == "csv":
            return _inspect_delimited(
                target,
                request=request,
                file_format="csv",
                size_bytes=size_bytes,
            )
        if file_format == "tsv":
            return _inspect_delimited(
                target,
                request=request,
                file_format="tsv",
                size_bytes=size_bytes,
            )
        if file_format == "pdf":
            return _inspect_pdf(
                target,
                request=request,
                size_bytes=size_bytes,
            )
        if file_format == "json":
            return _inspect_json(
                target,
                request=request,
                size_bytes=size_bytes,
            )
        return _inspection_failure(
            request=request,
            target=target,
            file_format="unsupported",
            size_bytes=size_bytes,
            error=(
                "unsupported data format; expected .xlsx, .xlsm, .pdf, .csv, "
                ".tsv, or .json"
            ),
        )
    except Exception as exc:
        return _inspection_failure(
            request=request,
            target=target,
            file_format=file_format,
            size_bytes=size_bytes,
            error=_bounded_error(exc),
        )


def _preview_grid(
    name: str,
    records: Iterable[Sequence[object]],
    request: InspectDataFileInput,
    *,
    excel: bool,
    budget: _GridBudget,
) -> TablePreview:
    """Bound a source grid without interpreting its rows or trusting XLSX dimensions."""
    rows: list[list[str]] = []
    types: list[list[str]] = []
    formats: list[list[str]] = []
    numbers: list[int] = []
    shortened: list[tuple[int, int]] = []
    total_rows = total_columns = page_columns = 0
    more_rows = False
    max_columns = min(request.max_columns, budget.cells)
    max_rows = min(request.max_rows, budget.cells // max_columns)
    for row_number, record in enumerate(records, start=1):
        if row_number >= request.start_row and len(rows) == max_rows:
            more_rows = True
            break
        total_rows = row_number
        total_columns = max(total_columns, len(record))
        if row_number < request.start_row:
            continue
        page_columns = max(page_columns, len(record))
        values, cell_types, number_formats = [], [], []
        begin = request.start_column - 1
        for column, cell in enumerate(record[begin : begin + max_columns], request.start_column):
            value = getattr(cell, "value", None) if excel else cell
            rendered, text_shortened = budget.text(value)
            number_format = str(getattr(cell, "number_format", "General"))
            rendered_format, format_shortened = budget.text("" if number_format == "General" else number_format)
            if text_shortened or format_shortened:
                shortened.append((row_number, column))
            values.append(rendered)
            cell_types.append("empty" if value is None else str(getattr(cell, "data_type", "s")))
            number_formats.append(rendered_format)
        rows.append(values)
        types.append(cell_types)
        formats.append(number_formats)
        numbers.append(row_number)
    width = min(max_columns, max(0, page_columns - request.start_column + 1))
    budget.cells -= len(rows) * width
    for values, cell_types, number_formats in zip(rows, types, formats, strict=True):
        values.extend([""] * (width - len(values)))
        cell_types.extend(["empty"] * (width - len(cell_types)))
        number_formats.extend([""] * (width - len(number_formats)))
    next_column = request.start_column + width if page_columns >= request.start_column + width else None
    return TablePreview(
        name=name, rows=rows, row_numbers=numbers,
        column_numbers=list(range(request.start_column, request.start_column + width)),
        cell_types=types, number_formats=formats, row_count=None if more_rows else total_rows,
        column_count=None if more_rows else total_columns,
        next_row=request.start_row + len(rows) if more_rows else None,
        next_column=next_column, truncated_cells=shortened,
        truncated=more_rows or next_column is not None or bool(shortened),
    )


def _inspect_workbook(
    target: Path,
    *,
    request: InspectDataFileInput,
    file_format: Literal["xlsx", "xlsm"],
    size_bytes: int,
) -> InspectDataFileOutput:
    from openpyxl import load_workbook

    # Public openpyxl streaming API: retain formulas and ignore potentially wrong
    # producer-supplied dimensions, as recommended by its optimised-mode docs.
    workbook = load_workbook(target, read_only=True, data_only=False, keep_links=False)
    try:
        sheet_names = list(workbook.sheetnames)
        if request.sheet_name is not None:
            if request.sheet_offset:
                raise ValueError("sheet_name and sheet_offset cannot be combined")
            selected = [workbook[request.sheet_name]]
            next_sheet = None
        else:
            end = request.sheet_offset + request.max_sheets
            selected = workbook.worksheets[request.sheet_offset:end]
            next_sheet = end if end < len(sheet_names) else None
        previews = []
        budget = _GridBudget(characters=min(request.max_characters, _MAX_GRID_CHARACTERS))
        for index, worksheet in enumerate(selected):
            if budget.cells == 0:
                next_sheet = request.sheet_offset + index
                break
            worksheet.reset_dimensions()
            records = worksheet.iter_rows()
            try:
                previews.append(_preview_grid(worksheet.title, records, request, excel=True, budget=budget))
            finally:
                records.close()
        return InspectDataFileOutput(
            path=request.path, format=file_format, size_bytes=size_bytes, sha256=_file_sha256(target),
            valid=True, summary="Raw worksheet cells; header roles are not inferred; formulas are not evaluated.",
            tables=previews, sheet_names=sheet_names, next_sheet_offset=next_sheet,
            page_count=None, pages=[], json_type=None, json_keys=[], json_preview=None,
            truncated=next_sheet is not None or any(p.truncated for p in previews), error=None,
        )
    finally:
        workbook.close()


def _inspect_delimited(
    target: Path,
    *,
    request: InspectDataFileInput,
    file_format: Literal["csv", "tsv"],
    size_bytes: int,
) -> InspectDataFileOutput:
    delimiter = "\t" if file_format == "tsv" else ","
    with target.open("r", encoding="utf-8-sig", newline="") as stream:
        sample = stream.read(16_384)
        stream.seek(0)
        if file_format == "csv" and sample:
            try:
                delimiter = csv.Sniffer().sniff(sample, delimiters=",;\t|").delimiter
            except csv.Error:
                pass
        table = _preview_grid(
            target.name, csv.reader(stream, delimiter=delimiter, strict=True), request, excel=False,
            budget=_GridBudget(characters=min(request.max_characters, _MAX_GRID_CHARACTERS)),
        )
    return InspectDataFileOutput(
        path=request.path, format=file_format, size_bytes=size_bytes, sha256=_file_sha256(target),
        valid=True, summary="Raw delimited records; header roles are not inferred; row numbers count logical records.",
        tables=[table], sheet_names=[], page_count=None, pages=[], json_type=None, json_keys=[], json_preview=None,
        truncated=table.truncated, error=None,
    )


def _inspect_pdf(
    target: Path,
    *,
    request: InspectDataFileInput,
    size_bytes: int,
) -> InspectDataFileOutput:
    import fitz  # type: ignore[import-untyped]

    document = fitz.open(target)
    try:
        if document.needs_pass:
            raise ValueError("encrypted PDF requires a password")
        page_count = document.page_count
        pages: list[PagePreview] = []
        for page_index in range(min(page_count, request.max_pages)):
            text = document.load_page(page_index).get_text("text")
            truncated = len(text) > request.max_characters
            pages.append(
                PagePreview(
                    page_number=page_index + 1,
                    text=text[: request.max_characters],
                    truncated=truncated,
                )
            )
        truncated = page_count > len(pages) or any(page.truncated for page in pages)
        return InspectDataFileOutput(
            path=request.path,
            format="pdf",
            size_bytes=size_bytes,
            sha256=_file_sha256(target),
            valid=True,
            summary=f"Valid PDF with {page_count} page(s).",
            tables=[],
            sheet_names=[],
            page_count=page_count,
            pages=pages,
            json_type=None,
            json_keys=[],
            json_preview=None,
            truncated=truncated,
            error=None,
        )
    finally:
        document.close()


def _inspect_json(
    target: Path,
    *,
    request: InspectDataFileInput,
    size_bytes: int,
) -> InspectDataFileOutput:
    with target.open("r", encoding="utf-8-sig") as stream:
        value = json.load(stream)
    json_type = _json_type(value)
    json_keys = sorted(str(key) for key in value) if isinstance(value, dict) else []
    rendered = json.dumps(
        value,
        ensure_ascii=False,
        indent=2,
        allow_nan=False,
        default=_json_default,
    )
    truncated = len(rendered) > request.max_characters
    return InspectDataFileOutput(
        path=request.path,
        format="json",
        size_bytes=size_bytes,
        sha256=_file_sha256(target),
        valid=True,
        summary=(
            f"Valid JSON {json_type}"
            + (f" with {len(value)} item(s)." if isinstance(value, Sequence) and not isinstance(value, str) else ".")
        ),
        tables=[],
        sheet_names=[],
        page_count=None,
        pages=[],
        json_type=json_type,
        json_keys=json_keys,
        json_preview=rendered[: request.max_characters],
        truncated=truncated,
        error=None,
    )


def _inspection_failure(
    *,
    request: InspectDataFileInput,
    target: Path,
    file_format: Literal[
        "xlsx", "xlsm", "pdf", "csv", "tsv", "json", "unsupported"
    ],
    size_bytes: int,
    error: str,
) -> InspectDataFileOutput:
    return InspectDataFileOutput(
        path=request.path,
        format=file_format,
        size_bytes=size_bytes,
        sha256=_file_sha256(target),
        valid=False,
        summary="Data file inspection failed.",
        tables=[],
        sheet_names=[],
        page_count=None,
        pages=[],
        json_type=None,
        json_keys=[],
        json_preview=None,
        truncated=False,
        error=error,
    )


def _normalize_inspect_output(raw: object) -> NormalizedToolOutput:
    validated = InspectDataFileOutput.model_validate(raw)
    structured = json_schema_output(
        _INSPECT_OUTPUT_SCHEMA,
        validated.model_dump(mode="json"),
    )
    if not validated.valid:
        return NormalizedToolOutput(
            structured_content=structured,
            is_error=True,
            error_code="data_inspection_failed",
            error_message=validated.error or "data file inspection failed",
            retryable=False,
        )
    return NormalizedToolOutput(structured_content=structured)


def _resolve_inspect_use(
    workspace: WorkspaceRuntime,
    arguments: Mapping[str, JsonValue],
) -> ResolvedToolUse:
    target = workspace.resolve_path(str(arguments["path"]))
    return ResolvedToolUse(
        effects=frozenset({ToolEffect.READ_WORKSPACE}),
        targets=(ToolTarget(kind="workspace_path", value=str(target)),),
    )


def _format_from_path(
    path: Path,
) -> Literal["xlsx", "xlsm", "pdf", "csv", "tsv", "json", "unsupported"]:
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
    from openpyxl.worksheet.formula import ArrayFormula, DataTableFormula

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
    return message[:1_000]


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


__all__ = [
    "InspectDataFileInput",
    "InspectDataFileOutput",
    "PagePreview",
    "TablePreview",
    "create_inspect_data_file_tool",
]
