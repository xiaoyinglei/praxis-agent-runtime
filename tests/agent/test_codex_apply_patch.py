from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

from agent_runtime.tools.builtins import create_resident_coding_tools
from agent_runtime.tools.executor import ToolExecutor
from agent_runtime.tools.permissions import ToolExecutionContext
from agent_runtime.tools.tool import ToolCall, ToolCallOrigin
from agent_runtime.workspace import open_workspace


@pytest.fixture
def codex_binary(monkeypatch):
    manifest = Path(__file__).resolve().parents[2] / "evals/answer_evidence/2026-10-10-codex-patch-ab/dependency.json"
    supplied = os.environ.get("PRAXIS_TEST_CODEX_BINARY")
    if supplied:
        binary = Path(supplied)
    elif sys.platform != "darwin" or not manifest.exists():
        pytest.skip("optional Codex integration requires the pinned evaluation dependency")
    else:
        binary = Path(json.loads(manifest.read_text())["binary"])
    assert binary.is_file()
    monkeypatch.setenv("PRAXIS_CODEX_APPLY_PATCH", str(binary))
    return binary


def _tool(workspace):
    return next(
        t
        for t in create_resident_coding_tools(workspace, plan_updater=lambda value: value)
        if t.definition.name == "apply_patch"
    )


async def _execute(workspace, patch, *, allow_write=True):
    tool = _tool(workspace)
    call = ToolCall(
        tool_call_id="call_patch",
        tool_name="apply_patch",
        arguments={"patch": patch},
        origin=ToolCallOrigin(request_id="req_patch", toolset_revision="patch_v1", exposed_tool_names=("apply_patch",)),
    )
    return await ToolExecutor({"apply_patch": tool}).execute(
        call,
        context=ToolExecutionContext(
            workspace_root=workspace.root,
            cwd=workspace.root,
            allow_write_tools=allow_write,
            approved_tool_call_ids=(frozenset({"call_patch"}) if allow_write else frozenset()),
        ),
    )


def test_opt_in_replaces_schema_without_changing_other_tools(tmp_path, codex_binary, monkeypatch):
    workspace = open_workspace(tmp_path)
    enabled = create_resident_coding_tools(workspace, plan_updater=lambda value: value)
    assert _tool(workspace).definition.input_schema["required"] == ("patch",)
    monkeypatch.delenv("PRAXIS_CODEX_APPLY_PATCH")
    default = create_resident_coding_tools(workspace, plan_updater=lambda value: value)
    assert "file_path" in _tool(workspace).definition.input_schema["properties"]
    for before, after in zip(default, enabled, strict=True):
        if before.definition.name != "apply_patch":
            assert before.definition == after.definition


@pytest.mark.anyio
async def test_real_binary_applies_multiple_operations(tmp_path, codex_binary):
    workspace = open_workspace(tmp_path)
    (tmp_path / "edit.txt").write_text("one\ntwo\nthree\nfour\n")
    (tmp_path / "old.txt").write_text("old\n")
    (tmp_path / "delete.txt").write_text("delete\n")
    patch = (
        "*** Begin Patch\n*** Update File: edit.txt\n@@\n-one\n+ONE\n two\n@@\n three\n-four\n+FOUR\n"
        "*** Add File: new/note.txt\n+literal $(touch bad) 'quote'\n*** Update File: old.txt\n"
        "*** Move to: moved.txt\n@@\n-old\n+new\n*** Delete File: delete.txt\n*** End Patch"
    )
    result = await _execute(workspace, patch)
    assert not result.result.is_error, result.result
    assert (tmp_path / "edit.txt").read_text() == "ONE\ntwo\nthree\nFOUR\n"
    assert (tmp_path / "new/note.txt").read_text() == "literal $(touch bad) 'quote'\n"
    assert (tmp_path / "moved.txt").read_text() == "new\n"
    assert not (tmp_path / "old.txt").exists()
    assert not (tmp_path / "delete.txt").exists()
    assert not (tmp_path / "bad").exists()
    assert result.result.metadata["workspace_tree_changed"] is True


@pytest.mark.anyio
async def test_missing_context_fails_without_claiming_success(tmp_path, codex_binary):
    workspace = open_workspace(tmp_path)
    path = tmp_path / "edit.txt"
    path.write_text("original\n")
    result = await _execute(workspace, "*** Begin Patch\n*** Update File: edit.txt\n@@\n-missing\n+new\n*** End Patch")
    assert result.result.is_error
    assert result.result.error_code == "patch_failed"
    assert path.read_text() == "original\n"
    assert result.result.metadata["workspace_tree_changed"] is False


@pytest.mark.anyio
async def test_partial_failure_reports_error_and_actual_change(tmp_path, codex_binary):
    workspace = open_workspace(tmp_path)
    result = await _execute(
        workspace,
        (
            "*** Begin Patch\n*** Add File: created.txt\n+hello\n"
            "*** Update File: missing.txt\n@@\n-old\n+new\n*** End Patch"
        ),
    )
    assert result.result.is_error
    assert result.result.structured_content["exit_code"] != 0
    assert (tmp_path / "created.txt").read_text() == "hello\n"
    assert result.result.metadata["workspace_tree_changed"] is True


@pytest.mark.anyio
@pytest.mark.parametrize(
    "relative",
    [
        ".git/config",
        ".venv/keep.txt",
        "node_modules/keep.txt",
        ".agent_memory/keep.txt",
    ],
)
async def test_protected_paths_remain_read_only(tmp_path, codex_binary, relative):
    workspace = open_workspace(tmp_path)
    path = tmp_path / relative
    path.parent.mkdir(exist_ok=True)
    path.write_text("keep\n")
    result = await _execute(
        workspace, (f"*** Begin Patch\n*** Update File: {relative}\n@@\n-keep\n+changed\n*** End Patch")
    )
    assert result.result.is_error
    assert result.result.error_code == "patch_failed"
    assert path.read_text() == "keep\n"


@pytest.mark.anyio
@pytest.mark.parametrize("mode", ["absolute", "parent", "symlink"])
async def test_external_paths_are_denied_by_sandbox(tmp_path, codex_binary, mode):
    root = tmp_path / "workspace"
    root.mkdir()
    workspace = open_workspace(root)
    outside = tmp_path / "outside.txt"
    outside.write_text("keep\n")
    if mode == "symlink":
        (root / "link.txt").symlink_to(outside)
    name = str(outside) if mode == "absolute" else "../outside.txt" if mode == "parent" else "link.txt"
    result = await _execute(workspace, f"*** Begin Patch\n*** Update File: {name}\n@@\n-keep\n+changed\n*** End Patch")
    assert result.result.is_error
    assert result.result.error_code == "patch_failed"
    assert outside.read_text() == "keep\n"


@pytest.mark.anyio
async def test_write_permission_is_required_before_binary_runs(tmp_path, codex_binary):
    workspace = open_workspace(tmp_path)
    result = await _execute(
        workspace,
        "*** Begin Patch\n*** Add File: denied.txt\n+no\n*** End Patch",
        allow_write=False,
    )
    assert result.result.is_error
    assert not (tmp_path / "denied.txt").exists()
