from __future__ import annotations

import json
import os
import sys
from collections.abc import Mapping
from dataclasses import replace
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
    assert "headers" not in spreadsheet_output["tables"][0]
    assert spreadsheet_output["tables"][0]["rows"] == (
        ("region", "revenue"),
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
    assert csv_output["tables"][0]["row_count"] == 4
    assert csv_output["tables"][0]["rows"][0] == ("variant", "converted")
    assert "headers" not in csv_output["tables"][0]

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
async def test_managed_python_fresh_process_and_saved_script_share_interpreter(tmp_path: Path) -> None:
    workspace = open_workspace(tmp_path)
    tool = _tools(workspace)["execute_python"]
    (workspace.root / "probe.py").write_text(
        "import os, sys, json, openpyxl\n"
        "assert __name__ == '__main__'\n"
        "print(json.dumps({'pid': os.getpid(), 'executable': sys.executable, 'package': openpyxl.__file__}))\n"
    )
    first = await _execute(tool, {"code": (
        "import os, sys, json, openpyxl\nnonce = 42\n"
        "print(json.dumps({'pid': os.getpid(), 'executable': sys.executable, 'package': openpyxl.__file__}))"
    )}, workspace=workspace, approve=True)
    missing = await _execute(tool, {"code": "print(nonce)"}, workspace=workspace, approve=True)
    assert missing.result.error_code == "python_failed"
    assert "NameError" in missing.result.structured_content["stderr"]
    second = await _execute(tool, {"code": (
        "assert 'nonce' not in globals()\n"
        "import runpy\nrunpy.run_path('probe.py', run_name='__main__')"
    )}, workspace=workspace, approve=True)
    assert not first.result.is_error, first.result.error_message
    assert not second.result.is_error, second.result.error_message
    before = json.loads(first.result.structured_content["stdout"])
    after = json.loads(second.result.structured_content["stdout"])
    assert before["pid"] != after["pid"]
    assert before["executable"] == after["executable"]
    assert before["package"] == after["package"]


@pytest.mark.anyio
@pytest.mark.usefixtures("fake_sandbox_exec")
async def test_managed_python_script_imports_siblings_without_instrumentation(tmp_path: Path) -> None:
    workspace = open_workspace(tmp_path)
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    (scripts / "helper.py").write_text("VALUE = 42\n")
    (scripts / "check.py").write_text(
        "import helper, sys\nassert helper.VALUE == 42\n"
        "assert __name__ == '__main__'\n"
        "assert sys.argv == [__file__]\nprint('script-ok')\n"
    )
    execution = await _execute(_tools(workspace)["execute_python"],
                               {"script_path": "scripts/check.py"}, workspace=workspace)
    assert not execution.result.is_error, execution.result.error_message
    output = execution.result.structured_content
    assert output["stdout"] == "script-ok\n"
    assert output["verification"] is None


@pytest.mark.anyio
@pytest.mark.usefixtures("fake_sandbox_exec")
@pytest.mark.parametrize("encoding,declaration", [("utf-8-sig", "utf-8"), ("latin-1", "latin-1")])
async def test_managed_python_script_respects_source_encoding(tmp_path: Path, encoding, declaration) -> None:
    workspace = open_workspace(tmp_path)
    (tmp_path / "encoded.py").write_bytes((f"# coding: {declaration}\nprint('café')\n").encode(encoding))
    execution = await _execute(_tools(workspace)["execute_python"],
                               {"script_path": "encoded.py"}, workspace=workspace)
    assert not execution.result.is_error, execution.result.error_message
    assert execution.result.structured_content["stdout"] == "café\n"


@pytest.mark.anyio
@pytest.mark.usefixtures("fake_sandbox_exec")
async def test_managed_python_legacy_runpy_can_import_workspace_modules(tmp_path: Path) -> None:
    workspace = open_workspace(tmp_path)
    (tmp_path / "helper.py").write_text("VALUE = 42\n")
    (tmp_path / "check.py").write_text("import helper\nprint(helper.VALUE)\n")
    execution = await _execute(_tools(workspace)["execute_python"],
        {"code": "import runpy; runpy.run_path('check.py', run_name='__main__')"}, workspace=workspace)
    assert not execution.result.is_error, execution.result.error_message
    assert execution.result.structured_content["stdout"] == "42\n"


@pytest.mark.anyio
@pytest.mark.usefixtures("fake_sandbox_exec")
async def test_managed_python_bounds_output_while_reading_both_streams(tmp_path: Path, monkeypatch) -> None:
    from agent_runtime.tools.builtins import shell

    workspace = open_workspace(tmp_path)
    retained_sizes = []
    original = shell._bounded_stream

    def observe(value):
        retained_sizes.append(len(value))
        return original(value)

    monkeypatch.setattr(shell, "_bounded_stream", observe)
    execution = await _execute(_tools(workspace)["execute_python"], {"code": (
        "import sys\nprint('HEAD' + 'x' * 262144 + 'TAIL')\n"
        "print('ERR_HEAD' + 'y' * 262144 + 'ERR_TAIL', file=sys.stderr)\n"
    )}, workspace=workspace)
    assert not execution.result.is_error
    output = execution.result.structured_content
    assert output["truncated"]
    assert output["stdout"].startswith("HEAD") and output["stdout"].endswith("TAIL\n")
    assert output["stderr"].startswith("ERR_HEAD") and output["stderr"].endswith("ERR_TAIL\n")
    assert retained_sizes and max(retained_sizes) <= shell._MAX_STREAM_BYTES + 1


@pytest.mark.anyio
@pytest.mark.parametrize("arguments", [{}, {"code": "pass", "script_path": "check.py"}])
async def test_managed_python_requires_exactly_one_source(tmp_path: Path, arguments) -> None:
    workspace = open_workspace(tmp_path)
    execution = await _execute(_tools(workspace)["execute_python"], arguments, workspace=workspace)
    assert execution.result.error_code == "invalid_arguments"


@pytest.mark.anyio
@pytest.mark.usefixtures("fake_sandbox_exec")
async def test_managed_python_script_respects_denied_alias(tmp_path: Path) -> None:
    from agent_runtime.permission_policy import PathRule, PermissionPolicy

    workspace = open_workspace(tmp_path)
    (tmp_path / "entry.py").write_text("print('must-not-run')\n")
    (tmp_path / "denied.py").symlink_to("entry.py")
    policy = PermissionPolicy(filesystem=(PathRule(path="denied.py", access="deny"),)).resolve(tmp_path)
    workspace.permission_provider = lambda: policy
    execution = await _execute(_tools(workspace)["execute_python"],
                               {"script_path": "denied.py"}, workspace=workspace)
    assert execution.result.is_error
    assert execution.result.structured_content["sandbox_error"] == "sandbox_filesystem_denied"
    assert execution.result.structured_content["stdout"] == ""


@pytest.mark.anyio
@pytest.mark.usefixtures("fake_sandbox_exec")
async def test_managed_python_script_cannot_escape_workspace(tmp_path: Path) -> None:
    workspace = open_workspace(tmp_path / "workspace", create=True)
    (tmp_path / "outside.py").write_text("print('must-not-run')\n")
    execution = await _execute(_tools(workspace)["execute_python"],
                               {"script_path": "../outside.py"}, workspace=workspace)
    assert execution.result.is_error
    assert "must-not-run" not in str(execution.result.structured_content)


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
    assert inspected_output["tables"][0]["rows"] == (("metric", "value"), ("mean", "42.5"))


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
@pytest.mark.parametrize("code", [
    "print(f'{(1 + 2)*100:+.0f}%')\nreturn 1",
    "from pathlib import Path\nPath('unexpected.txt').touch()\nprint(f'{1)')",
    "if True:\nprint('missing indentation')",
])
async def test_execute_python_rejects_invalid_source_before_write_approval(
    tmp_path: Path, code: str,
) -> None:
    workspace = open_workspace(tmp_path)
    boundary_calls = []

    def unexpected(*_args: object, **_kwargs: object) -> Any:
        boundary_calls.append("permission_or_execution")
        raise AssertionError("Invalid source must not reach permission or execution")

    tool = replace(_tools(workspace)["execute_python"], run=unexpected)
    execution = await ToolExecutor({"execute_python": tool}, permission_decider=unexpected).execute(
        ToolCall(
            tool_call_id="invalid-source", tool_name="execute_python",
            arguments={"code": code, "workspace_write": True, "output_paths": ["unexpected.txt"]},
            origin=ToolCallOrigin(request_id="syntax", toolset_revision="syntax-v1",
                                  exposed_tool_names=("execute_python",)),
        ),
        context=ToolExecutionContext(workspace_root=workspace.root, cwd=workspace.root),
    )

    assert execution.result.error_code == "invalid_arguments"
    message = execution.result.error_message or ""
    assert "$.code" in message
    assert "line" in message and "column" in message
    assert not boundary_calls
    assert execution.record is None
    assert not (workspace.root / "unexpected.txt").exists()


@pytest.mark.anyio
async def test_execute_python_valid_dynamic_source_still_requires_write_approval(tmp_path: Path) -> None:
    workspace = open_workspace(tmp_path)
    execution = await _execute(
        _tools(workspace)["execute_python"],
        {"code": "globals()['value'] = 7\nprint(value)",
         "workspace_write": True, "output_paths": ["output.txt"]},
        workspace=workspace,
    )

    assert execution.result.error_code == "approval_required"


@pytest.mark.anyio
@pytest.mark.usefixtures("fake_sandbox_exec")
async def test_execute_python_runs_valid_dynamic_definitions_after_approval(tmp_path: Path) -> None:
    workspace = open_workspace(tmp_path)
    execution = await _execute(
        _tools(workspace)["execute_python"],
        {"code": "globals()['value'] = 7\nassert value == 7\nprint(value)"},
        workspace=workspace, approve=True,
    )

    assert not execution.result.is_error
    assert execution.result.structured_content["stdout"] == "7\n"


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
                "import tempfile; from pathlib import Path; "
                "temp = tempfile.TemporaryDirectory(); "
                "probe = Path(temp.name) / 'probe.txt'; probe.write_text('scratch-ok'); "
                "assert probe.read_text() == 'scratch-ok'; temp.cleanup(); "
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
