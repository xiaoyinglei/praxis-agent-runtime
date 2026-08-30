from __future__ import annotations

import tempfile
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from rag.ingest.table_executor import MAX_RESULT_ROWS, ComputeResult, TableExecutor


class _FakeObjectStore:
    def __init__(self, files: dict[str, bytes] | None = None) -> None:
        self._files = files or {}

    def read_byte_range(self, key: str, start: int, end: int) -> bytes:
        return self._files.get(key, b"")

    def path_for_key(self, key: str) -> str | None:
        return None


class _FakeMetadataRepo:
    def __init__(self) -> None:
        self._assets: dict[int, _FakeAsset] = {}

    def register(self, asset_id: int, *, storage_key: str, sheet_name: str = "Sheet1") -> None:
        self._assets[asset_id] = _FakeAsset(asset_id=asset_id, storage_key=storage_key, sheet_name=sheet_name)

    def get_asset(self, asset_id: int) -> _FakeAsset | None:
        return self._assets.get(asset_id)


@dataclass(frozen=True, slots=True)
class _FakeAsset:
    asset_id: int
    storage_key: str
    sheet_name: str


def _create_test_xlsx(data: dict[str, list[object]], sheet_name: str = "Sheet1") -> bytes:
    df = pd.DataFrame(data)
    with tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False) as f:
        path = f.name
    try:
        df.to_excel(path, sheet_name=sheet_name, index=False)
        return Path(path).read_bytes()
    finally:
        Path(path).unlink(missing_ok=True)


def _create_test_parquet(data: dict[str, list[object]]) -> bytes:
    df = pd.DataFrame(data)
    with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f:
        path = f.name
    try:
        df.to_parquet(path)
        return Path(path).read_bytes()
    finally:
        Path(path).unlink(missing_ok=True)


def test_validate_sql_rejects_non_select() -> None:
    assert TableExecutor._validate_sql("") is False
    assert TableExecutor._validate_sql("   ") is False
    assert TableExecutor._validate_sql("INSERT INTO sheet VALUES (1)") is False
    assert TableExecutor._validate_sql("DROP TABLE sheet") is False
    assert TableExecutor._validate_sql("CREATE TABLE t (a INT)") is False
    assert TableExecutor._validate_sql("SELECT * FROM sheet") is True
    assert TableExecutor._validate_sql("  SELECT 1  ") is True


def test_validate_sql_rejects_forbidden_keywords() -> None:
    assert TableExecutor._validate_sql("SELECT * FROM sheet; DELETE FROM sheet") is False
    assert TableExecutor._validate_sql("SELECT * FROM sheet; ALTER TABLE sheet ADD COLUMN x INT") is False
    assert TableExecutor._validate_sql("SELECT * FROM sheet WHERE 1=1; PRAGMA threads=1") is False


def test_execute_returns_none_for_missing_asset() -> None:
    repo = _FakeMetadataRepo()
    store = _FakeObjectStore()
    executor = TableExecutor(object_store=store, metadata_repo=repo)
    result = executor.execute(asset_id=999, sql="SELECT 1")
    assert result is None


def test_execute_returns_none_for_missing_storage_key() -> None:
    repo = _FakeMetadataRepo()
    repo.register(1, storage_key="")
    store = _FakeObjectStore()
    executor = TableExecutor(object_store=store, metadata_repo=repo)
    result = executor.execute(asset_id=1, sql="SELECT 1")
    assert result is None


def test_execute_returns_none_when_file_not_in_store() -> None:
    repo = _FakeMetadataRepo()
    repo.register(1, storage_key="missing.xlsx")
    store = _FakeObjectStore()
    executor = TableExecutor(object_store=store, metadata_repo=repo)
    result = executor.execute(asset_id=1, sql="SELECT 1")
    assert result is None


def test_execute_simple_select_returns_result() -> None:
    xlsx_bytes = _create_test_xlsx({"name": ["Alice", "Bob", "Carol"], "score": [90, 85, 95]})
    repo = _FakeMetadataRepo()
    repo.register(1, storage_key="test.xlsx")
    store = _FakeObjectStore({"test.xlsx": xlsx_bytes})
    executor = TableExecutor(object_store=store, metadata_repo=repo)

    result = executor.execute(asset_id=1, sql='SELECT * FROM sheet WHERE "score" > 88')
    assert result is not None
    assert result.asset_id == 1
    assert "name" in result.columns
    assert result.raw_row_count == 2
    assert result.truncated is False
    assert result.elapsed_ms >= 0
    assert "Alice" in result.markdown
    assert "Carol" in result.markdown
    assert "Bob" not in result.markdown
    assert "[TABLE_COMPUTE_RESULT:asset_id=1]" in result.markdown


def test_execute_truncates_to_max_rows() -> None:
    data = {"id": list(range(200)), "value": list(range(200))}
    xlsx_bytes = _create_test_xlsx(data)
    repo = _FakeMetadataRepo()
    repo.register(1, storage_key="large.xlsx")
    store = _FakeObjectStore({"large.xlsx": xlsx_bytes})
    executor = TableExecutor(object_store=store, metadata_repo=repo)

    result = executor.execute(asset_id=1, sql="SELECT * FROM sheet ORDER BY id")
    assert result is not None
    assert result.raw_row_count == 200
    assert result.truncated is True
    assert len(result.rows) == MAX_RESULT_ROWS
    assert f"Showing first {MAX_RESULT_ROWS} rows" in result.markdown


def test_execute_handles_empty_result() -> None:
    xlsx_bytes = _create_test_xlsx({"name": ["A", "B"], "score": [1, 2]})
    repo = _FakeMetadataRepo()
    repo.register(1, storage_key="empty.xlsx")
    store = _FakeObjectStore({"empty.xlsx": xlsx_bytes})
    executor = TableExecutor(object_store=store, metadata_repo=repo)

    result = executor.execute(asset_id=1, sql="SELECT * FROM sheet WHERE score > 999")
    assert result is not None
    assert result.raw_row_count == 0
    assert result.truncated is False


def test_execute_aggregates_numeric_strings_from_parquet() -> None:
    parquet_bytes = _create_test_parquet(
        {
            "区域公司": ["总计（不含一体化）", "北方", "东北"],
            "日_日提货": ["131.074462", "19.22484", "6.307968"],
        }
    )
    repo = _FakeMetadataRepo()
    repo.register(14, storage_key="daily.parquet")
    store = _FakeObjectStore({"daily.parquet": parquet_bytes})
    executor = TableExecutor(object_store=store, metadata_repo=repo)

    result = executor.execute(
        asset_id=14,
        sql='SELECT SUM("日_日提货") AS total FROM sheet WHERE "区域公司" = \'总计（不含一体化）\'',
    )

    assert result is not None
    assert result.columns == ["total"]
    assert result.rows == [["131.074462"]]
    assert "TABLE_COMPUTE_RESULT:asset_id=14" in result.markdown


def test_execute_normalizes_multiline_parquet_column_names() -> None:
    parquet_bytes = _create_test_parquet(
        {
            "区域公司": ["总计", "北方"],
            "月累计\n提货量": ["9125.1182", "1693.432"],
        }
    )
    repo = _FakeMetadataRepo()
    repo.register(14, storage_key="daily.parquet")
    store = _FakeObjectStore({"daily.parquet": parquet_bytes})
    executor = TableExecutor(object_store=store, metadata_repo=repo)

    result = executor.execute(
        asset_id=14,
        sql='SELECT "月累计 提货量" FROM sheet WHERE "区域公司" = \'总计\'',
    )

    assert result is not None
    assert result.columns == ["月累计 提货量"]
    assert result.rows == [["9125.1182"]]


def test_execute_trims_trailing_sparse_footer_rows_before_ranking() -> None:
    parquet_bytes = _create_test_parquet(
        {
            "区域公司": [
                "总计（不含一体化）",
                "北方",
                "东北",
                "华东",
                "制表：",
                "",
                "日",
                "月",
            ],
            "日_日提货": ["131.074462", "19.22484", "6.307968", "29.137104", "", "0", "131.074462", "1868.351764"],
            "月累计_月累计提货": [
                "1868.351764",
                "281.059297",
                "66.189168",
                "435.377389",
                "",
                "0",
                "6.33998",
                "6.35034",
            ],
            "月提货同比": ["-0.024129", "0.029459", "0.006566", "0.064241", "", "", "", ""],
            "年累计_年累计提货": ["12354.342395", "1661.394877", "511.311048", "3040.898458", "", "", "", ""],
            "年提货同比": ["-0.024033", "-0.073234", "0.05879", "0.120539", "", "", "", ""],
        }
    )
    repo = _FakeMetadataRepo()
    repo.register(14, storage_key="daily.parquet")
    store = _FakeObjectStore({"daily.parquet": parquet_bytes})
    executor = TableExecutor(object_store=store, metadata_repo=repo)

    result = executor.execute(
        asset_id=14,
        sql='SELECT "区域公司", "日_日提货" FROM sheet ORDER BY "日_日提货" DESC LIMIT 1',
    )

    assert result is not None
    assert result.rows == [["总计（不含一体化）", "131.074462"]]


def test_execute_preserves_wide_sparse_report_rows() -> None:
    data: dict[str, list[object]] = {
        "区域": ["总计", "北方", "东北", "制表：", "日", "月"],
        "品类": ["全部", "全部", "全部", "", "", ""],
        "当日 提货": ["99.0", "19.2", "6.3", "", "99.0", "1800.0"],
    }
    for index in range(1, 32):
        data[f"空列{index}"] = ["", "", "", "", "", ""]
    parquet_bytes = _create_test_parquet(data)
    repo = _FakeMetadataRepo()
    repo.register(88, storage_key="wide-sparse.parquet")
    store = _FakeObjectStore({"wide-sparse.parquet": parquet_bytes})
    executor = TableExecutor(object_store=store, metadata_repo=repo)

    inspected = executor.inspect(asset_id=88, head_rows=3, tail_rows=0)
    result = executor.execute(
        asset_id=88,
        sql="SELECT SUM(\"当日 提货\") AS total FROM sheet WHERE \"区域\" IN ('北方', '东北')",
    )

    assert inspected is not None
    assert inspected.row_count == 3
    assert inspected.head_rows[0]["区域"] == "总计"
    assert result is not None
    assert result.rows == [["25.5"]]


def test_execute_returns_none_for_invalid_sql() -> None:
    xlsx_bytes = _create_test_xlsx({"a": [1]})
    repo = _FakeMetadataRepo()
    repo.register(1, storage_key="bad.xlsx")
    store = _FakeObjectStore({"bad.xlsx": xlsx_bytes})
    executor = TableExecutor(object_store=store, metadata_repo=repo)

    result = executor.execute(asset_id=1, sql="THIS IS NOT VALID SQL")
    assert result is None


def test_execute_handles_sql_error_gracefully() -> None:
    xlsx_bytes = _create_test_xlsx({"a": [1]})
    repo = _FakeMetadataRepo()
    repo.register(1, storage_key="bad_sql.xlsx")
    store = _FakeObjectStore({"bad_sql.xlsx": xlsx_bytes})
    executor = TableExecutor(object_store=store, metadata_repo=repo)

    result = executor.execute(asset_id=1, sql="SELECT nonexistent_column FROM sheet")
    assert result is None


def test_markdown_format_includes_metadata() -> None:
    result = ComputeResult(
        asset_id=42,
        columns=["col_a", "col_b"],
        rows=[["1", "x"], ["2", "y"]],
        raw_row_count=2,
        elapsed_ms=234.5,
        truncated=False,
    )
    md = result.markdown
    assert "[TABLE_COMPUTE_RESULT:asset_id=42]" in md
    assert "234ms" in md or "235ms" in md
    assert "Returned 2 rows" in md
    assert "| col_a | col_b |" in md
    assert "| 1 | x |" in md


def test_compute_result_columns_and_rows_properties() -> None:
    result = ComputeResult(
        asset_id=1,
        columns=["a", "b"],
        rows=[],
        raw_row_count=0,
        elapsed_ms=0.0,
        truncated=False,
    )
    assert result.columns == ["a", "b"]
    assert result.rows == []
    assert result.truncated is False


__all__ = [
    "test_validate_sql_rejects_non_select",
    "test_validate_sql_rejects_forbidden_keywords",
    "test_execute_returns_none_for_missing_asset",
    "test_execute_returns_none_for_missing_storage_key",
    "test_execute_returns_none_when_file_not_in_store",
    "test_execute_simple_select_returns_result",
    "test_execute_truncates_to_max_rows",
    "test_execute_handles_empty_result",
    "test_execute_aggregates_numeric_strings_from_parquet",
    "test_execute_normalizes_multiline_parquet_column_names",
    "test_execute_returns_none_for_invalid_sql",
    "test_execute_handles_sql_error_gracefully",
    "test_markdown_format_includes_metadata",
    "test_compute_result_columns_and_rows_properties",
]
