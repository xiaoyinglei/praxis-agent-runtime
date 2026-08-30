from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from agent_runtime.text import text_unit_count
from rag.ingest.table_semantics import deduplicate_table_columns

SUMMARY_SAMPLE_ROWS = 3
TYPE_INFERENCE_ROWS = 50

TABLE_POLICY_COMPUTE_ONLY = "compute_only"


class TokenAccountingClient(Protocol):
    def count(self, text: str) -> int: ...


@dataclass(frozen=True, slots=True)
class TableAssetProfile:
    row_count: int
    column_count: int
    estimated_tokens: int
    table_policy: str
    summary_sample: str
    preview_text: str
    columns: list[str]
    schema: list[dict[str, str]]
    sample_rows: list[dict[str, str]]


def profile_markdown_table(
    markdown: str,
    *,
    token_accounting: TokenAccountingClient | None = None,
) -> TableAssetProfile:
    normalized = _normalize_table_text(markdown)
    rows = _parse_markdown_rows(normalized)
    header, data_rows = _split_header_and_data(rows)
    return profile_table_data(
        columns=header,
        rows=data_rows,
        token_accounting=token_accounting,
        full_text=normalized,
    )


def profile_table_data(
    *,
    columns: list[str],
    rows: list[list[str]],
    token_accounting: TokenAccountingClient | None = None,
    full_text: str | None = None,
    total_row_count: int | None = None,
    total_column_count: int | None = None,
) -> TableAssetProfile:
    data_rows = [[str(cell).strip() for cell in row] for row in rows]
    sampled_column_count = _column_count(columns, data_rows)
    column_count = max(_safe_count(total_column_count), sampled_column_count)
    normalized_columns = _columns(columns, sampled_column_count)
    row_count = max(_safe_count(total_row_count), len(data_rows))
    estimated_tokens = _estimated_tokens(
        columns=normalized_columns,
        data_rows=data_rows,
        row_count=row_count,
        column_count=column_count,
        full_text=full_text,
        token_accounting=token_accounting,
    )
    table_policy = _table_policy(row_count=row_count, estimated_tokens=estimated_tokens)
    schema = _schema(normalized_columns, data_rows[:TYPE_INFERENCE_ROWS])
    sample_rows = _sample_row_dicts(normalized_columns, data_rows[:SUMMARY_SAMPLE_ROWS])
    summary_sample = _summary_sample(
        columns=normalized_columns,
        data_rows=data_rows,
        row_count=row_count,
        column_count=column_count,
        estimated_tokens=estimated_tokens,
        table_policy=table_policy,
        schema=schema,
    )
    preview_text = summary_sample
    return TableAssetProfile(
        row_count=row_count,
        column_count=column_count,
        estimated_tokens=estimated_tokens,
        table_policy=table_policy,
        summary_sample=summary_sample,
        preview_text=preview_text,
        columns=normalized_columns,
        schema=schema,
        sample_rows=sample_rows,
    )


def _count_tokens(text: str, *, token_accounting: TokenAccountingClient | None) -> int:
    if token_accounting is not None:
        try:
            return token_accounting.count(text)
        except Exception:
            pass
    return text_unit_count(text)


def _estimated_tokens(
    *,
    columns: list[str],
    data_rows: list[list[str]],
    row_count: int,
    column_count: int,
    full_text: str | None,
    token_accounting: TokenAccountingClient | None,
) -> int:
    if full_text is not None:
        return _count_tokens(full_text, token_accounting=token_accounting)
    column_scale = max(column_count, 1) / max(len(columns), 1)
    if not data_rows:
        header_tokens = max(_count_tokens(" | ".join(columns), token_accounting=token_accounting), 1)
        return max(int(header_tokens * max(row_count, 1) * column_scale), header_tokens)
    sampled_rows = data_rows[: min(len(data_rows), TYPE_INFERENCE_ROWS)]
    sample_text = "\n".join(_markdown_rows(columns, sampled_rows))
    sample_tokens = max(_count_tokens(sample_text, token_accounting=token_accounting), 1)
    row_scale = max(row_count, len(data_rows), 1) / max(len(sampled_rows), 1)
    return max(int(sample_tokens * row_scale * column_scale), sample_tokens)


def _normalize_table_text(text: str) -> str:
    return str(text or "").replace("\r\n", "\n").replace("\r", "\n").strip()


def _parse_markdown_rows(text: str) -> list[list[str]]:
    rows: list[list[str]] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped.startswith("|") or not stripped.endswith("|"):
            continue
        cells = [cell.strip() for cell in stripped.strip("|").split("|")]
        if _is_separator_row(cells):
            continue
        rows.append(cells)
    return rows


def _is_separator_row(cells: list[str]) -> bool:
    if not cells:
        return False
    return all(cell and set(cell) <= {"-", ":", " "} for cell in cells)


def _split_header_and_data(rows: list[list[str]]) -> tuple[list[str], list[list[str]]]:
    if not rows:
        return [], []
    return rows[0], rows[1:]


def _column_count(header: list[str], data_rows: list[list[str]]) -> int:
    return max([len(header), *(len(row) for row in data_rows)] or [0])


def _safe_count(value: int | None) -> int:
    if value is None:
        return 0
    return max(int(value), 0)


def _columns(header: list[str], column_count: int) -> list[str]:
    if not header:
        return [f"column_{index + 1}" for index in range(column_count)]
    return deduplicate_table_columns(
        [
            header[index].strip() if index < len(header) and header[index].strip() else f"column_{index + 1}"
            for index in range(column_count)
        ]
    )


def _table_policy(*, row_count: int, estimated_tokens: int) -> str:
    return TABLE_POLICY_COMPUTE_ONLY


def _summary_sample(
    *,
    columns: list[str],
    data_rows: list[list[str]],
    row_count: int,
    column_count: int,
    estimated_tokens: int,
    table_policy: str,
    schema: list[dict[str, str]],
) -> str:
    sample_rows = data_rows[:SUMMARY_SAMPLE_ROWS]
    field_type_text = "; ".join(f"{field['name']}={field['type']}" for field in schema)
    lines = [
        f"Table policy: {table_policy}",
        f"Table shape: rows={row_count}, columns={column_count}, estimated_tokens={estimated_tokens}",
        f"Table columns: {_columns_text(columns, column_count)}",
        f"Field types: {field_type_text}",
        "Sample rows:",
    ]
    if sample_rows:
        lines.extend(_markdown_rows(columns, sample_rows))
    else:
        lines.append("(no data rows)")
    return "\n".join(lines).strip()


def _field_types(columns: list[str], rows: list[list[str]]) -> list[str]:
    return [f"{field['name']}={field['type']}" for field in _schema(columns, rows)]


def _columns_text(columns: list[str], column_count: int) -> str:
    if not columns:
        return ""
    text = " | ".join(columns)
    if column_count > len(columns):
        text = f"{text} | ... ({column_count - len(columns)} more columns)"
    return text


def _schema(columns: list[str], rows: list[list[str]]) -> list[dict[str, str]]:
    inferred: list[str] = []
    for index, _column in enumerate(columns):
        values = [_cell(row, index) for row in rows]
        non_empty = [value for value in values if value]
        if not non_empty:
            inferred.append("empty")
            continue
        numeric_values = [_as_float(value) for value in non_empty]
        numeric = [value for value in numeric_values if value is not None]
        if len(numeric) >= max(1, int(len(non_empty) * 0.7)):
            inferred.append(f"number(min={min(numeric):g}, max={max(numeric):g})")
            continue
        distinct = []
        seen = set()
        for value in non_empty:
            if value in seen:
                continue
            seen.add(value)
            distinct.append(value)
        if len(distinct) <= 10 and len(distinct) <= max(3, len(non_empty) // 2):
            inferred.append(f"enum({', '.join(distinct[:6])})")
            continue
        inferred.append("text")
    return [{"name": column, "type": inferred[index]} for index, column in enumerate(columns)]


def _sample_row_dicts(columns: list[str], rows: list[list[str]]) -> list[dict[str, str]]:
    return [{column: _cell(row, index) for index, column in enumerate(columns)} for row in rows]


def _cell(row: list[str], index: int) -> str:
    if index >= len(row):
        return ""
    return row[index].strip()


def _as_float(value: str) -> float | None:
    normalized = value.replace(",", "").strip()
    if normalized.endswith("%"):
        normalized = normalized[:-1].strip()
    try:
        return float(normalized)
    except ValueError:
        return None


def _markdown_rows(columns: list[str], rows: list[list[str]]) -> list[str]:
    lines = [
        "| " + " | ".join(columns) + " |",
        "|" + "|".join(["---"] * len(columns)) + "|",
    ]
    for row in rows:
        padded = [_cell(row, index) for index in range(len(columns))]
        lines.append("| " + " | ".join(padded) + " |")
    return lines


__all__ = [
    "TABLE_POLICY_COMPUTE_ONLY",
    "TableAssetProfile",
    "profile_markdown_table",
    "profile_table_data",
]
