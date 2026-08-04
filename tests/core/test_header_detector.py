"""测试通用 Excel 表头识别器，涵盖 5 种场景。"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from rag.ingest.header_detector import (
    HeaderKind,
    detect_header,
)


def _sheet_from_rows(rows: list[list[object]], tmp_path: Path) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    path = tmp_path / "test.xlsx"
    df.to_excel(path, index=False, header=False)
    return pd.read_excel(path, header=None)


# ── 场景 1: 普通单行表头 ──

def test_single_row_header(tmp_path: Path) -> None:
    df = _sheet_from_rows([
        ["Name", "Age", "Score"],
        ["Alice", 30, 95.5],
        ["Bob", 25, 88.0],
        ["Carol", 28, 92.3],
    ], tmp_path)
    result = detect_header(df)
    assert result.header_kind == HeaderKind.SINGLE
    assert result.header_row_indices == (0,)
    assert result.data_start_row == 1
    assert result.normalized_columns == ["Name", "Age", "Score"]
    assert result.confidence > 0.6


# ── 场景 2: 标题行 + 表头 ──

def test_title_then_header(tmp_path: Path) -> None:
    df = _sheet_from_rows([
        ["Sales Report 2026", "", ""],
        ["Region", "Revenue", "Qty"],
        ["East", 15000, 120],
        ["West", 22000, 200],
    ], tmp_path)
    result = detect_header(df)
    assert result.header_kind in (HeaderKind.TITLED, HeaderKind.SINGLE)
    assert result.data_start_row >= 2
    assert result.confidence > 0.5
    # 列名应该来自第二行
    assert "Region" in result.normalized_columns or "Revenue" in result.normalized_columns


# ── 场景 3: Unnamed — 真实表头在第二行 ──

def test_real_header_in_second_row(tmp_path: Path) -> None:
    """模拟石膏板文件的场景：第一行是元数据，第二行才是表头。"""
    df = _sheet_from_rows([
        ["", "数据来源：全产品销售台账", "", ""],
        ["辅助", "排名", "区域", "销量"],
        ["龙牌", 1, "北方", 6467.96],
        ["龙牌", 2, "华东", 2163.72],
        ["龙牌", 3, "西南", 641.47],
    ], tmp_path)
    result = detect_header(df)
    assert result.header_kind in (HeaderKind.TITLED, HeaderKind.INFERRED, HeaderKind.SINGLE)
    assert result.data_start_row <= 3
    # 列名应该来自第二行
    assert any("区域" in c or "排名" in c or "辅助" in c for c in result.normalized_columns)
    assert result.confidence > 0.45


# ── 场景 4: 多级表头 ──

def test_multi_level_header(tmp_path: Path) -> None:
    df = _sheet_from_rows([
        ["", "2026年", "2026年", "2025年"],
        ["区域", "销量", "销售额", "销量"],
        ["北方", 706.22, 6435.83, 653.74],
        ["华东", 2163.72, 17337.50, 1541.64],
    ], tmp_path)
    result = detect_header(df)
    assert result.header_kind == HeaderKind.MULTI_LEVEL
    assert len(result.header_row_indices) >= 2
    assert result.data_start_row == result.header_row_indices[-1] + 1
    assert result.confidence > 0.5


# ── 场景 5: 无表头纯数据 ──

def test_no_header_pure_data(tmp_path: Path) -> None:
    df = _sheet_from_rows([
        [15000, 120, 3.5],
        [22000, 200, 4.2],
        [18000, 150, 3.8],
        [30000, 300, 5.1],
    ], tmp_path)
    result = detect_header(df)
    assert result.header_kind == HeaderKind.NONE
    assert result.confidence < 0.6
    assert result.normalized_columns == [f"column_{c}" for c in range(df.shape[1])]


# ── 边界情况 ──

def test_empty_sheet(tmp_path: Path) -> None:
    df = _sheet_from_rows([], tmp_path)
    result = detect_header(df)
    assert result.header_kind == HeaderKind.NONE
    assert result.confidence == 0.0
    assert result.normalized_columns == []


def test_single_row_only(tmp_path: Path) -> None:
    df = _sheet_from_rows([["A", "B", "C"]], tmp_path)
    result = detect_header(df)
    # 单行也可能是表头，但置信度不高
    if result.header_kind != HeaderKind.NONE:
        assert result.confidence < 0.7


def test_all_text_columns(tmp_path: Path) -> None:
    """全部是文本列——在纯文本表中所有行都像表头。接受 SINGLE 或 MULTI_LEVEL。"""
    df = _sheet_from_rows([
        ["City", "Country", "Continent"],
        ["Beijing", "China", "Asia"],
        ["Paris", "France", "Europe"],
    ], tmp_path)
    result = detect_header(df)
    assert result.header_kind in (HeaderKind.SINGLE, HeaderKind.MULTI_LEVEL)
    assert result.confidence > 0.5


def test_sparse_title_rows(tmp_path: Path) -> None:
    """稀疏的标题行（大部分单元格为空）。"""
    df = _sheet_from_rows([
        ["Quarterly Report", "", "", "", ""],
        ["", "", "", "", ""],
        ["Division", "Q1", "Q2", "Q3", "Q4"],
        ["A", 100, 200, 300, 400],
        ["B", 150, 250, 350, 450],
    ], tmp_path)
    result = detect_header(df)
    assert result.header_kind in (HeaderKind.TITLED, HeaderKind.SINGLE)
    assert result.data_start_row >= 3
    assert "Q1" in result.normalized_columns or "Q2" in result.normalized_columns


def test_result_has_all_fields(tmp_path: Path) -> None:
    df = _sheet_from_rows([
        ["Name", "Value"],
        ["X", 1],
    ], tmp_path)
    result = detect_header(df)
    assert isinstance(result.header_kind, HeaderKind)
    assert isinstance(result.header_row_indices, tuple)
    assert isinstance(result.data_start_row, int)
    assert isinstance(result.normalized_columns, list)
    assert isinstance(result.confidence, float)
    assert 0.0 <= result.confidence <= 1.0
    assert isinstance(result.warnings, tuple)
    assert result.column_count == len(result.normalized_columns)


def test_detect_header_does_not_hardcode_anything(tmp_path: Path) -> None:
    """验证检测器不依赖硬编码的列名/文件名/sheet名。"""
    df = _sheet_from_rows([
        ["Foo", "Bar", "Baz"],
        [1, 2, 3],
    ], tmp_path)
    result = detect_header(df)
    assert result.normalized_columns == ["Foo", "Bar", "Baz"]
    # 换一组完全不同的列名
    df2 = _sheet_from_rows([
        ["Xyzzy", "Quux", "Corge"],
        [10, 20, 30],
    ], tmp_path)
    result2 = detect_header(df2)
    assert result2.normalized_columns == ["Xyzzy", "Quux", "Corge"]


def test_detect_header_normalizes_multiline_column_names(tmp_path: Path) -> None:
    df = _sheet_from_rows([
        ["区域公司", "月累计\n提货量", "月提货量\n同比"],
        ["总计", 9125.1182, 0.127967],
    ], tmp_path)

    result = detect_header(df)

    assert result.normalized_columns == ["区域公司", "月累计 提货量", "月提货量 同比"]


__all__ = [
    "test_single_row_header",
    "test_title_then_header",
    "test_real_header_in_second_row",
    "test_multi_level_header",
    "test_no_header_pure_data",
    "test_empty_sheet",
    "test_single_row_only",
    "test_all_text_columns",
    "test_sparse_title_rows",
    "test_result_has_all_fields",
    "test_detect_header_does_not_hardcode_anything",
    "test_detect_header_normalizes_multiline_column_names",
]
