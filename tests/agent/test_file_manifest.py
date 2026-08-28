"""Tests for file manifest structured previews and file-first processing.

Covers:
- FileManifest model and build_file_manifest
- Pandas preview for CSV and XLSX
- Structured probe with merged cells and formulas
- Context block generation
"""

from __future__ import annotations

from pathlib import Path

import pytest

from agent_runtime.file_manifest import (
    FileManifest,
    FileManifestEntry,
    SheetPreview,
    build_file_manifest,
)
from agent_runtime.workspace import WorkspaceRuntime

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def ws(tmp_path: Path) -> WorkspaceRuntime:
    root = tmp_path / "workspace"
    root.mkdir()
    runtime = WorkspaceRuntime(root=root, is_temporary=True)
    runtime.initialize()
    return runtime


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_csv(ws: WorkspaceRuntime, name: str, content: str) -> Path:
    p = ws.input_files / name
    p.write_text(content, encoding="utf-8")
    return p


def _make_xlsx(ws: WorkspaceRuntime, name: str, sheets: dict[str, list[list]]):
    """Create an XLSX with multiple sheets."""
    openpyxl = pytest.importorskip("openpyxl")
    wb = openpyxl.Workbook()
    for i, (sheet_name, rows) in enumerate(sheets.items()):
        if i == 0:
            ws_obj = wb.active
            ws_obj.title = sheet_name
        else:
            ws_obj = wb.create_sheet(sheet_name)
        for row in rows:
            ws_obj.append(row)
    path = ws.input_files / name
    wb.save(path)
    return path


# ===================================================================
# FileManifest model
# ===================================================================


class TestFileManifestModel:
    def test_empty_manifest(self) -> None:
        m = FileManifest(
            files=[],
            total_size_bytes=0,
            has_structured_files=False,
            has_probeable_files=False,
        )
        assert m.to_context_block() == ""

    def test_single_csv_entry(self) -> None:
        entry = FileManifestEntry(
            path="input_files/data.csv",
            filename="data.csv",
            size_bytes=1024,
            mime_type="text/csv",
            file_kind="csv",
            hash="abc123",
            structured=True,
            probeable=True,
            sheets=[
                SheetPreview(
                    sheet_name="data.csv",
                    total_rows=100,
                    total_columns=5,
                    columns=[],
                    head=[],
                    dtypes={},
                )
            ],
        )
        m = FileManifest(
            files=[entry],
            total_size_bytes=1024,
            has_structured_files=True,
            has_probeable_files=True,
        )
        block = m.to_context_block()
        assert "data.csv" in block
        assert "100 rows" in block
        assert "5 columns" in block
        assert "Available Tools" not in block
        assert "structured_probe" not in block
        assert "run_python" not in block
        assert "write_file" not in block

    def test_context_block_shows_warnings(self) -> None:
        entry = FileManifestEntry(
            path="input_files/report.xlsx",
            filename="report.xlsx",
            size_bytes=2048,
            mime_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            file_kind="xlsx",
            hash="def456",
            structured=True,
            probeable=True,
            sheets=[
                SheetPreview(
                    sheet_name="Sheet1",
                    total_rows=50,
                    total_columns=3,
                    columns=[],
                    head=[],
                    dtypes={},
                    merged_cells=True,
                    formula_columns=["C"],
                )
            ],
        )
        m = FileManifest(
            files=[entry],
            total_size_bytes=2048,
            has_structured_files=True,
            has_probeable_files=True,
        )
        block = m.to_context_block()
        assert "merged cells" in block
        assert "formulas" in block


# ===================================================================
# build_file_manifest with CSV
# ===================================================================


class TestBuildManifestCSV:
    def test_simple_csv(self, ws: WorkspaceRuntime) -> None:
        _make_csv(ws, "sales.csv", "name,amount\nAlice,100\nBob,200\n")
        manifest = build_file_manifest(ws)

        assert len(manifest.files) == 1
        entry = manifest.files[0]
        assert entry.file_kind == "csv"
        assert entry.structured is True
        assert entry.probeable is True
        assert entry.error is None

        # Should have pandas preview
        assert len(entry.sheets) == 1
        sheet = entry.sheets[0]
        assert sheet.total_rows == 3  # header + 2 data rows
        assert sheet.total_columns == 2
        assert len(sheet.columns) == 2
        assert sheet.columns[0].name == "name"
        assert sheet.columns[1].name == "amount"
        assert len(sheet.head) == 2

    def test_csv_with_chinese_filename(self, ws: WorkspaceRuntime) -> None:
        _make_csv(ws, "销售数据.csv", "产品,数量\nA,10\nB,20\n")
        manifest = build_file_manifest(ws)

        assert len(manifest.files) == 1
        assert manifest.files[0].filename == "销售数据.csv"
        assert manifest.files[0].structured is True

    def test_empty_input_dir(self, ws: WorkspaceRuntime) -> None:
        manifest = build_file_manifest(ws)
        assert manifest.files == []
        assert manifest.has_structured_files is False


# ===================================================================
# build_file_manifest with XLSX
# ===================================================================


class TestBuildManifestXLSX:
    def test_single_sheet_xlsx(self, ws: WorkspaceRuntime) -> None:
        _make_xlsx(
            ws,
            "data.xlsx",
            {
                "Sales": [
                    ["product", "qty", "price"],
                    ["Widget", 10, 25.0],
                    ["Gadget", 20, 30.0],
                ],
            },
        )
        manifest = build_file_manifest(ws)

        assert len(manifest.files) == 1
        entry = manifest.files[0]
        assert entry.file_kind == "xlsx"
        assert entry.structured is True
        assert entry.probeable is True

        assert len(entry.sheets) == 1
        sheet = entry.sheets[0]
        assert sheet.sheet_name == "Sales"
        assert sheet.total_rows == 3
        assert sheet.total_columns == 3
        assert sheet.columns[0].name == "product"
        assert sheet.head[0]["product"] == "Widget"

    def test_multi_sheet_xlsx(self, ws: WorkspaceRuntime) -> None:
        _make_xlsx(
            ws,
            "report.xlsx",
            {
                "Summary": [["total", "count"], [1000, 50]],
                "Detail": [["id", "value"], [1, 100], [2, 200]],
            },
        )
        manifest = build_file_manifest(ws)

        assert len(manifest.files) == 1
        assert len(manifest.files[0].sheets) == 2
        names = [s.sheet_name for s in manifest.files[0].sheets]
        assert "Summary" in names
        assert "Detail" in names

    def test_xlsx_with_merged_cells(self, ws: WorkspaceRuntime) -> None:
        openpyxl = pytest.importorskip("openpyxl")
        wb = openpyxl.Workbook()
        sheet = wb.active
        sheet.title = "Merged"
        sheet.append(["Header1", "Header2"])
        sheet.merge_cells("A1:B1")
        sheet.append([1, 2])
        wb.save(ws.input_files / "merged.xlsx")

        manifest = build_file_manifest(ws)
        entry = manifest.files[0]
        assert entry.sheets[0].merged_cells is True

    def test_xlsx_with_formulas(self, ws: WorkspaceRuntime) -> None:
        openpyxl = pytest.importorskip("openpyxl")
        wb = openpyxl.Workbook()
        sheet = wb.active
        sheet.title = "Calc"
        sheet.append(["A", "B", "Sum"])
        sheet.append([10, 20, None])
        # Write formula in C2
        sheet["C2"] = "=A2+B2"
        wb.save(ws.input_files / "formula.xlsx")

        manifest = build_file_manifest(ws)
        entry = manifest.files[0]
        assert "C" in entry.sheets[0].formula_columns


# ===================================================================
# Structured probe enhancements
# ===================================================================


class TestProbeEnhancements:
    def test_probe_xlsx_merged_cells(self, ws: WorkspaceRuntime) -> None:
        openpyxl = pytest.importorskip("openpyxl")
        wb = openpyxl.Workbook()
        sheet = wb.active
        sheet.title = "Test"
        sheet.append(["A", "B"])
        sheet.merge_cells("A1:B1")
        sheet.append([1, 2])
        wb.save(ws.input_files / "test.xlsx")

        probe = build_file_manifest(ws).files[0].probe
        assert probe is not None
        assert len(probe.tables) == 1
        assert probe.tables[0].merged_cells is True

    def test_probe_xlsx_formulas(self, ws: WorkspaceRuntime) -> None:
        openpyxl = pytest.importorskip("openpyxl")
        wb = openpyxl.Workbook()
        sheet = wb.active
        sheet.title = "Test"
        sheet.append(["X", "Y", "Z"])
        sheet.append([1, 2, None])
        sheet["C2"] = "=A2+B2"
        wb.save(ws.input_files / "formula.xlsx")

        probe = build_file_manifest(ws).files[0].probe
        assert probe is not None
        assert len(probe.tables) == 1
        assert "C" in probe.tables[0].formula_columns

    def test_probe_csv_no_merged_or_formulas(self, ws: WorkspaceRuntime) -> None:
        _make_csv(ws, "plain.csv", "a,b\n1,2\n")
        probe = build_file_manifest(ws).files[0].probe
        assert probe is not None
        assert len(probe.tables) == 1
        assert probe.tables[0].merged_cells is False
        assert probe.tables[0].formula_columns == []


# ===================================================================
# Context block rendering
# ===================================================================


class TestContextBlock:
    def test_manifest_with_no_files_renders_empty(self) -> None:
        m = FileManifest(
            files=[],
            total_size_bytes=0,
            has_structured_files=False,
            has_probeable_files=False,
        )
        assert m.to_context_block() == ""

    def test_manifest_shows_available_packages(self) -> None:
        entry = FileManifestEntry(
            path="input_files/data.csv",
            filename="data.csv",
            size_bytes=100,
            mime_type="text/csv",
            file_kind="csv",
            hash="abc",
            structured=True,
            probeable=True,
            sheets=[
                SheetPreview(
                    sheet_name="data.csv",
                    total_rows=5,
                    total_columns=2,
                    columns=[],
                    head=[{"a": 1, "b": 2}],
                    dtypes={},
                )
            ],
        )
        m = FileManifest(
            files=[entry],
            total_size_bytes=100,
            has_structured_files=True,
            has_probeable_files=True,
        )
        block = m.to_context_block()
        assert "Available Python Packages" in block
        # pandas should be listed since it's installed
        assert "pandas" in block

    def test_manifest_error_entry(self) -> None:
        entry = FileManifestEntry(
            path="input_files/bad.xlsx",
            filename="bad.xlsx",
            size_bytes=100,
            mime_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            file_kind="xlsx",
            hash="abc",
            structured=True,
            probeable=True,
            error="probe_failed: corrupt file",
        )
        m = FileManifest(
            files=[entry],
            total_size_bytes=100,
            has_structured_files=True,
            has_probeable_files=True,
        )
        block = m.to_context_block()
        assert "probe_failed" in block


# ===================================================================
# End-to-end: file-first processing path
# ===================================================================


class TestFileFirstPath:
    def test_csv_manifest_enables_first_turn_analysis(
        self,
        ws: WorkspaceRuntime,
    ) -> None:
        """Full path: CSV → manifest → model has everything it needs."""
        _make_csv(ws, "orders.csv", "order_id,product,amount\n1,Widget,100\n2,Gadget,200\n3,Widget,150\n")
        manifest = build_file_manifest(ws)

        # Manifest should have complete info
        assert manifest.has_structured_files is True
        assert manifest.has_probeable_files is True
        entry = manifest.files[0]
        assert entry.structured is True
        assert len(entry.sheets) == 1

        sheet = entry.sheets[0]
        assert sheet.total_rows == 4
        assert sheet.total_columns == 3
        col_names = [c.name for c in sheet.columns]
        assert "order_id" in col_names
        assert "product" in col_names
        assert "amount" in col_names

        # Context block should be usable
        block = manifest.to_context_block()
        assert "orders.csv" in block
        assert "Available Python Packages" in block

    def test_xlsx_manifest_shows_all_sheets(
        self,
        ws: WorkspaceRuntime,
    ) -> None:
        """Multi-sheet XLSX → manifest lists all sheets with previews."""
        _make_xlsx(
            ws,
            "report.xlsx",
            {
                "Sales": [["date", "amount"], ["2024-01", 1000], ["2024-02", 1500]],
                "Inventory": [["item", "stock"], ["Widget", 50], ["Gadget", 30]],
            },
        )
        manifest = build_file_manifest(ws)

        entry = manifest.files[0]
        assert len(entry.sheets) == 2
        sheet_names = [s.sheet_name for s in entry.sheets]
        assert "Sales" in sheet_names
        assert "Inventory" in sheet_names

        # Each sheet should have columns
        for sheet in entry.sheets:
            assert len(sheet.columns) == 2
            assert len(sheet.head) >= 1


class TestLegacyPrimitiveClosure:
    def test_primitive_ops_exports_checkpoint_stable_models_only(self) -> None:
        import agent_runtime.primitive_ops as primitive_ops

        assert set(primitive_ops.__all__) == {
            "CandidateHeaderRow",
            "CellValue",
            "FileKind",
            "StructuredProbeOutput",
            "StructuredTableProbe",
        }
        assert not hasattr(primitive_ops, "PrimitiveOps")

    def test_primitive_runner_files_are_removed(self) -> None:
        root = Path(__file__).resolve().parents[2]

        assert not (root / "agent_runtime/runner/__init__.py").exists()
        assert not (root / "agent_runtime/runner/python_runner.py").exists()

    def test_production_has_no_primitive_executor_references(self) -> None:
        root = Path(__file__).resolve().parents[2]
        forbidden = (
            "PrimitiveOps",
            "agent_runtime.runner",
            "RunPythonInput",
            "RunPythonInlineInput",
            "RunPythonOutput",
            "StructuredProbeInput",
            "WriteFileInput",
            "WriteFileOutput",
        )
        offenders: dict[str, tuple[str, ...]] = {}

        for production_root in (root / "rag", root / "agent_runtime"):
            for path in production_root.rglob("*.py"):
                source = path.read_text(encoding="utf-8")
                matches = tuple(name for name in forbidden if name in source)
                if matches:
                    offenders[str(path.relative_to(root))] = matches

        assert offenders == {}
