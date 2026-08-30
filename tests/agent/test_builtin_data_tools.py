from __future__ import annotations

import json
import os
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import fitz
import pytest
from openpyxl import Workbook

from agent_runtime.tools.builtins import create_resident_coding_tools
from agent_runtime.tools.evidence import runtime_workspace_file_changes
from agent_runtime.tools.executor import ToolExecution, ToolExecutor
from agent_runtime.tools.permissions import ToolExecutionContext
from agent_runtime.tools.tool import Tool, ToolCall, ToolCallOrigin
from agent_runtime.workspace import WorkspaceRuntime, open_workspace


def _tools(workspace: WorkspaceRuntime) -> dict[str, Tool]:
    return {
        tool.definition.name: tool
        for tool in create_resident_coding_tools(
            workspace,
            plan_updater=lambda _arguments: {
                "accepted": True,
                "revision": 1,
                "message": "ok",
            },
        )
    }


async def _execute(
    tool: Tool,
    arguments: Mapping[str, Any],
    *,
    workspace: WorkspaceRuntime,
    approve: bool = False,
) -> ToolExecution:
    call = ToolCall(
        tool_call_id=f"call-{tool.definition.name}",
        tool_name=tool.definition.name,
        arguments=arguments,
        origin=ToolCallOrigin(
            request_id="data-aci-request",
            toolset_revision="data-aci-tools-v1",
            exposed_tool_names=(tool.definition.name,),
        ),
    )
    return await ToolExecutor({tool.definition.name: tool}).execute(
        call,
        context=ToolExecutionContext(
            workspace_root=workspace.root,
            cwd=workspace.root,
            allow_write_tools=True,
            allow_execute_tools=True,
            approved_tool_call_ids=(
                frozenset({call.tool_call_id}) if approve else frozenset()
            ),
        ),
    )


@pytest.mark.anyio
async def test_inspect_data_file_reads_spreadsheet_pdf_csv_and_json(
    tmp_path: Path,
) -> None:
    workbook = Workbook()
    sheet = workbook.active
    sheet.title = "Sales"
    sheet.append(["region", "revenue"])
    sheet.append(["East", 120])
    sheet.append(["West", 180])
    workbook.save(tmp_path / "sales.xlsx")

    document = fitz.open()
    page = document.new_page()
    page.insert_text((72, 72), "Operations uptime was 99.95 percent.")
    document.save(tmp_path / "operations.pdf")
    document.close()

    (tmp_path / "experiment.csv").write_text(
        "variant,converted\nA,1\nB,0\nB,1\n",
        encoding="utf-8",
    )
    (tmp_path / "summary.json").write_text(
        json.dumps({"winner": "B", "lift": 0.12}),
        encoding="utf-8",
    )
    workspace = open_workspace(tmp_path)
    inspect = _tools(workspace)["inspect_data_file"]

    spreadsheet = await _execute(
        inspect,
        {"path": "sales.xlsx"},
        workspace=workspace,
    )
    pdf = await _execute(
        inspect,
        {"path": "operations.pdf"},
        workspace=workspace,
    )
    csv_result = await _execute(
        inspect,
        {"path": "experiment.csv"},
        workspace=workspace,
    )
    json_result = await _execute(
        inspect,
        {"path": "summary.json"},
        workspace=workspace,
    )

    assert spreadsheet.result.is_error is False
    spreadsheet_output = spreadsheet.result.structured_content
    assert isinstance(spreadsheet_output, Mapping)
    assert spreadsheet_output["valid"] is True
    assert spreadsheet_output["sheet_names"] == ("Sales",)
    assert spreadsheet_output["tables"][0]["headers"] == (
        "region",
        "revenue",
    )
    assert spreadsheet_output["tables"][0]["rows"] == (
        ("East", "120"),
        ("West", "180"),
    )

    pdf_output = pdf.result.structured_content
    assert isinstance(pdf_output, Mapping)
    assert pdf_output["valid"] is True
    assert pdf_output["page_count"] == 1
    assert "99.95 percent" in pdf_output["pages"][0]["text"]

    csv_output = csv_result.result.structured_content
    assert isinstance(csv_output, Mapping)
    assert csv_output["tables"][0]["row_count"] == 3
    assert csv_output["tables"][0]["headers"] == (
        "variant",
        "converted",
    )

    json_output = json_result.result.structured_content
    assert isinstance(json_output, Mapping)
    assert json_output["json_type"] == "object"
    assert json_output["json_keys"] == ("lift", "winner")
    assert '"winner": "B"' in json_output["json_preview"]


@pytest.mark.anyio
async def test_read_file_routes_pdf_and_xlsx_to_structured_inspection(
    tmp_path: Path,
) -> None:
    (tmp_path / "report.pdf").write_bytes(b"%PDF-1.7\nnot-a-full-pdf")
    (tmp_path / "book.xlsx").write_bytes(b"PK\x03\x04fake-workbook")
    workspace = open_workspace(tmp_path)
    read_file = _tools(workspace)["read_file"]

    pdf = await _execute(
        read_file,
        {"path": "report.pdf"},
        workspace=workspace,
    )
    workbook = await _execute(
        read_file,
        {"path": "book.xlsx", "start_line": 1},
        workspace=workspace,
    )

    for execution, expected_format in ((pdf, "pdf"), (workbook, "xlsx")):
        assert execution.result.is_error is True
        assert execution.result.error_code == "binary_file_requires_inspection"
        assert execution.result.retryable is False
        output = execution.result.structured_content
        assert isinstance(output, Mapping)
        assert output["content"] == ""
        assert output["is_binary"] is True
        assert output["binary_format"] == expected_format
        assert output["recommended_tool"] == "inspect_data_file"
        assert "Do not retry read_file" in (
            execution.result.error_message or ""
        )


@pytest.mark.anyio
@pytest.mark.usefixtures("fake_sandbox_exec")
async def test_execute_python_uses_managed_runtime_and_attests_new_output(
    tmp_path: Path,
) -> None:
    workspace = open_workspace(tmp_path)
    assert not (workspace.root / ".venv").exists()
    tools = _tools(workspace)
    code = "\n".join(
        (
            "import fitz, pandas, scipy",
            "from openpyxl import Workbook",
            "book = Workbook()",
            "sheet = book.active",
            "sheet.append(['metric', 'value'])",
            "sheet.append(['mean', 42.5])",
            "book.save('analysis.xlsx')",
            "print('data-stack-ok')",
        )
    )

    execution = await _execute(
        tools["execute_python"],
        {
            "code": code,
            "workspace_write": True,
            "output_paths": ["analysis.xlsx"],
        },
        workspace=workspace,
        approve=True,
    )

    assert execution.result.is_error is False
    output = execution.result.structured_content
    assert isinstance(output, Mapping)
    assert output["stdout"] == "data-stack-ok\n"
    assert output["execution_mode"] == "managed_python_sandbox"
    [(path, before_sha256, after_sha256)] = runtime_workspace_file_changes(
        execution.result
    )
    assert path == "analysis.xlsx"
    assert before_sha256 != after_sha256

    inspected = await _execute(
        tools["inspect_data_file"],
        {"path": "analysis.xlsx"},
        workspace=workspace,
    )
    inspected_output = inspected.result.structured_content
    assert isinstance(inspected_output, Mapping)
    assert inspected_output["valid"] is True
    assert inspected_output["sha256"] == after_sha256
    assert inspected_output["tables"][0]["rows"] == (("mean", "42.5"),)


@pytest.mark.anyio
async def test_execute_python_requires_declared_outputs_for_workspace_write(
    tmp_path: Path,
) -> None:
    workspace = open_workspace(tmp_path)
    execution = await _execute(
        _tools(workspace)["execute_python"],
        {"code": "open('undeclared.txt', 'w').write('no')", "workspace_write": True},
        workspace=workspace,
        approve=True,
    )

    assert execution.result.is_error is True
    assert execution.result.error_code == "invalid_arguments"
    assert "output_paths" in (execution.result.error_message or "")
    assert not (workspace.root / "undeclared.txt").exists()


@pytest.mark.anyio
@pytest.mark.skipif(
    sys.platform != "darwin" or not os.path.isfile("/usr/bin/sandbox-exec"),
    reason="real managed Python sandbox requires macOS Seatbelt",
)
async def test_execute_python_real_sandbox_imports_runtime_data_stack(
    tmp_path: Path,
) -> None:
    workspace = open_workspace(tmp_path)
    execution = await _execute(
        _tools(workspace)["execute_python"],
        {
            "code": (
                "import fitz, openpyxl, pandas, scipy; "
                "print('seatbelt-data-stack-ok')"
            )
        },
        workspace=workspace,
    )

    assert execution.result.is_error is False
    output = execution.result.structured_content
    assert isinstance(output, Mapping)
    assert output["stdout"] == "seatbelt-data-stack-ok\n"
    assert not (workspace.root / ".venv").exists()
