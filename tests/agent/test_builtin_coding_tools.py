from __future__ import annotations

import ast
import shlex
import shutil
import socket
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest

from rag.agent.tools.builtins import (
    RESIDENT_CODING_TOOL_NAMES,
    create_resident_coding_tools,
)
from rag.agent.tools.builtins import search as search_module
from rag.agent.tools.builtins import shell as shell_module
from rag.agent.tools.executor import ToolExecution, ToolExecutor
from rag.agent.tools.permissions import ToolExecutionContext
from rag.agent.tools.tool import Tool, ToolCall, ToolCallOrigin, ToolEffect
from rag.agent.workspace import WorkspaceRuntime, open_workspace


def _origin(tool_name: str) -> ToolCallOrigin:
    return ToolCallOrigin(
        request_id="req_builtin",
        toolset_revision="tools_builtin_v1",
        exposed_tool_names=(tool_name,),
    )


async def _execute(
    tool: Tool,
    arguments: Mapping[str, Any],
    *,
    workspace: WorkspaceRuntime,
) -> ToolExecution:
    call = ToolCall(
        tool_call_id=f"call_{tool.definition.name}",
        tool_name=tool.definition.name,
        arguments=arguments,
        origin=_origin(tool.definition.name),
    )
    return await ToolExecutor({tool.definition.name: tool}).execute(
        call,
        context=ToolExecutionContext(
            workspace_root=workspace.root,
            cwd=workspace.root,
            allow_write_tools=True,
            allow_execute_tools=True,
        ),
    )


def _tools_by_name(
    workspace: WorkspaceRuntime,
    *,
    updates: list[Mapping[str, Any]] | None = None,
) -> dict[str, Tool]:
    captured = updates if updates is not None else []

    def update_plan(arguments: Mapping[str, Any]) -> Mapping[str, Any]:
        captured.append(arguments)
        return {
            "accepted": True,
            "revision": len(captured),
            "message": "plan updated",
        }

    tools = create_resident_coding_tools(
        workspace,
        plan_updater=update_plan,
    )
    return {tool.definition.name: tool for tool in tools}


def test_resident_coding_tool_baseline_is_exact_and_ordered(tmp_path: Path) -> None:
    workspace = open_workspace(tmp_path, create=True)

    tools = create_resident_coding_tools(
        workspace,
        plan_updater=lambda _arguments: {
            "accepted": True,
            "revision": 1,
            "message": "ok",
        },
    )

    assert RESIDENT_CODING_TOOL_NAMES == (
        "search_text",
        "list_files",
        "read_file",
        "apply_patch",
        "run_command",
        "update_plan",
    )
    assert tuple(tool.definition.name for tool in tools) == RESIDENT_CODING_TOOL_NAMES
    assert all(isinstance(tool, Tool) for tool in tools)
    assert len({tool.definition.name for tool in tools}) == len(tools)
    assert "write_file" not in {tool.definition.name for tool in tools}
    for tool in tools:
        assert len(tool.definition.description) >= 80
        properties = tool.definition.input_schema.get("properties", {})
        assert isinstance(properties, Mapping)
        assert properties
        assert all(
            isinstance(schema, Mapping) and schema.get("description")
            for schema in properties.values()
        )


def test_run_command_declares_requested_network_as_dynamic_effect(
    tmp_path: Path,
) -> None:
    workspace = open_workspace(tmp_path, create=True)
    command = _tools_by_name(workspace)["run_command"]

    resolved = command.resolve_use(
        {
            "command": "curl https://example.com",
            "working_dir": ".",
            "timeout_seconds": 1,
            "network": True,
        }
    )

    assert ToolEffect.NETWORK in resolved.effects


@pytest.mark.anyio
async def test_filesystem_tools_list_read_patch_and_expose_changes_immediately(
    tmp_path: Path,
) -> None:
    workspace = open_workspace(tmp_path, create=True)
    source = workspace.root / "src"
    source.mkdir()
    target = source / "example.py"
    target.write_text("before\nneedle_one()\nafter\n", encoding="utf-8")
    (workspace.root / ".venv").mkdir()
    (workspace.root / ".rag").mkdir(exist_ok=True)
    tools = _tools_by_name(workspace)

    listed = await _execute(
        tools["list_files"],
        {"path": "src", "glob": "*.py", "limit": 10},
        workspace=workspace,
    )
    read_before = await _execute(
        tools["read_file"],
        {"path": "src/example.py", "max_bytes": 1000},
        workspace=workspace,
    )
    patched = await _execute(
        tools["apply_patch"],
        {
            "file_path": "src/example.py",
            "old_string": "needle_one",
            "new_string": "fresh_symbol",
        },
        workspace=workspace,
    )
    old_search = await _execute(
        tools["search_text"],
        {"pattern": "needle_one", "path": "src", "glob": "*.py"},
        workspace=workspace,
    )
    new_search = await _execute(
        tools["search_text"],
        {"pattern": "fresh_symbol", "path": "src", "glob": "*.py"},
        workspace=workspace,
    )

    assert listed.result.is_error is False
    assert listed.result.structured_content is not None
    assert [entry["path"] for entry in listed.result.structured_content["entries"]] == [
        "src/example.py"
    ]
    assert read_before.result.structured_content is not None
    assert read_before.result.structured_content["content"] == (
        "before\nneedle_one()\nafter\n"
    )
    assert patched.result.structured_content is not None
    assert patched.result.structured_content["replaced"] is True
    assert set(patched.result.structured_content) == {
        "file_path",
        "replaced",
        "occurrences",
        "message",
    }
    assert patched.result.metadata["file_path"] == "src/example.py"
    patch_diff = patched.result.metadata["diff"]
    assert isinstance(patch_diff, str)
    assert "-needle_one()" in patch_diff
    assert "+fresh_symbol()" in patch_diff
    assert patched.result.metadata["diff_truncated"] is False
    assert patched.result.metadata["workspace_changed"] is True
    assert len(str(patched.result.metadata["before_sha256"])) == 64
    assert len(str(patched.result.metadata["after_sha256"])) == 64
    assert (
        patched.result.metadata["before_sha256"]
        != patched.result.metadata["after_sha256"]
    )
    assert old_search.result.structured_content is not None
    assert old_search.result.structured_content["matches"] == ()
    assert new_search.result.structured_content is not None
    assert new_search.result.structured_content["total_matches"] == 1
    assert target.read_text(encoding="utf-8") == (
        "before\nfresh_symbol()\nafter\n"
    )


@pytest.mark.anyio
async def test_read_file_default_window_bounds_model_context(
    tmp_path: Path,
) -> None:
    workspace = open_workspace(tmp_path, create=True)
    (workspace.root / "large.txt").write_text("x" * 20_000, encoding="utf-8")

    execution = await _execute(
        _tools_by_name(workspace)["read_file"],
        {"path": "large.txt"},
        workspace=workspace,
    )

    assert execution.result.is_error is False
    assert execution.result.structured_content is not None
    assert len(execution.result.structured_content["content"]) == 16_000
    assert execution.result.structured_content["truncated"] is True


@pytest.mark.anyio
async def test_read_file_supports_source_line_windows_and_continuation(
    tmp_path: Path,
) -> None:
    workspace = open_workspace(tmp_path, create=True)
    (workspace.root / "lines.py").write_text(
        "line_1\nline_2\nline_3\nline_4\nline_5\n",
        encoding="utf-8",
    )
    read_file = _tools_by_name(workspace)["read_file"]

    execution = await _execute(
        read_file,
        {"path": "lines.py", "start_line": 3, "max_lines": 2},
        workspace=workspace,
    )

    assert execution.result.structured_content is not None
    output = execution.result.structured_content
    assert output["content"] == "line_3\nline_4\n"
    assert output["start_line"] == 3
    assert output["end_line"] == 4
    assert output["next_line"] == 5
    assert output["next_offset"] == len("line_1\nline_2\nline_3\nline_4\n")
    assert output["truncated"] is True

    mixed_modes = await _execute(
        read_file,
        {"path": "lines.py", "offset": 2, "start_line": 3},
        workspace=workspace,
    )
    assert mixed_modes.result.error_code == "invalid_arguments"


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("arguments", "error_code"),
    [
        (
            {
                "file_path": "missing.txt",
                "old_string": "before",
                "new_string": "after",
            },
            "file_not_found",
        ),
        (
            {
                "file_path": "notes.txt",
                "old_string": "absent",
                "new_string": "after",
            },
            "old_string_not_found",
        ),
        (
            {
                "file_path": "notes.txt",
                "old_string": "same",
                "new_string": "after",
            },
            "old_string_not_unique",
        ),
    ],
)
async def test_apply_patch_non_effect_is_a_canonical_tool_error(
    tmp_path: Path,
    arguments: Mapping[str, Any],
    error_code: str,
) -> None:
    workspace = open_workspace(tmp_path, create=True)
    (workspace.root / "notes.txt").write_text("same same", encoding="utf-8")

    execution = await _execute(
        _tools_by_name(workspace)["apply_patch"],
        arguments,
        workspace=workspace,
    )

    assert execution.result.is_error is True
    assert execution.result.error_code == error_code
    assert execution.result.error_message
    assert execution.result.structured_content is not None
    assert execution.result.structured_content["replaced"] is False


@pytest.mark.anyio
async def test_apply_patch_rejects_replacement_with_identical_content(
    tmp_path: Path,
) -> None:
    workspace = open_workspace(tmp_path, create=True)
    target = workspace.root / "notes.txt"
    target.write_text("same", encoding="utf-8")

    execution = await _execute(
        _tools_by_name(workspace)["apply_patch"],
        {
            "file_path": "notes.txt",
            "old_string": "same",
            "new_string": "same",
        },
        workspace=workspace,
    )

    assert execution.result.is_error is True
    assert execution.result.error_code == "patch_no_change"
    assert execution.result.metadata.get("workspace_changed") is not True
    assert target.read_text(encoding="utf-8") == "same"


@pytest.mark.anyio
async def test_search_text_supports_literal_regex_path_glob_context_and_limits(
    tmp_path: Path,
) -> None:
    workspace = open_workspace(tmp_path, create=True)
    source = workspace.root / "src"
    source.mkdir()
    (source / "one.py").write_text(
        "context before\nneedle_alpha()\ncontext after\n",
        encoding="utf-8",
    )
    (source / "two.py").write_text("needle_beta()\n", encoding="utf-8")
    (source / "notes.md").write_text("needle_markdown\n", encoding="utf-8")
    search = _tools_by_name(workspace)["search_text"]

    literal = await _execute(
        search,
        {
            "pattern": "needle_alpha",
            "path": "src/one.py",
            "glob": "*.py",
            "context_lines": 1,
            "max_results": 10,
        },
        workspace=workspace,
    )
    regex = await _execute(
        search,
        {
            "pattern": r"needle_[a-z]+\(",
            "path": "src",
            "glob": "*.py",
            "regex": True,
            "max_results": 10,
        },
        workspace=workspace,
    )
    auto_regex = await _execute(
        search,
        {
            "pattern": r"needle_[a-z]+\(",
            "path": "src",
            "glob": "*.py",
            "max_results": 10,
        },
        workspace=workspace,
    )
    limited = await _execute(
        search,
        {
            "pattern": "needle_",
            "path": "src",
            "glob": "*.py",
            "max_results": 1,
        },
        workspace=workspace,
    )
    invalid_regex = await _execute(
        search,
        {"pattern": "(", "path": "src", "regex": True},
        workspace=workspace,
    )

    assert literal.result.structured_content is not None
    literal_matches = literal.result.structured_content["matches"]
    assert len(literal_matches) == 1
    assert literal_matches[0]["file_path"] == "src/one.py"
    assert literal_matches[0]["line_number"] == 2
    assert literal_matches[0]["match_start"] == 0
    assert literal_matches[0]["context_before"] == ("context before",)
    assert literal_matches[0]["context_after"] == ("context after",)

    assert regex.result.structured_content is not None
    assert [
        match["file_path"] for match in regex.result.structured_content["matches"]
    ] == ["src/one.py", "src/two.py"]
    assert auto_regex.result.structured_content is not None
    assert [
        match["file_path"]
        for match in auto_regex.result.structured_content["matches"]
    ] == ["src/one.py", "src/two.py"]
    assert limited.result.structured_content is not None
    assert limited.result.structured_content["total_matches"] == 1
    assert limited.result.structured_content["truncated"] is True
    assert invalid_regex.result.error_code == "invalid_arguments"


@pytest.mark.anyio
async def test_search_text_prioritizes_source_and_returns_local_context(
    tmp_path: Path,
) -> None:
    workspace = open_workspace(tmp_path, create=True)
    source = workspace.root / "src"
    docs = workspace.root / "docs"
    source.mkdir()
    docs.mkdir()
    (source / "state.py").write_text(
        "from pydantic import BaseModel\n\nclass PlanState(BaseModel):\n    revision: int = 0\n",
        encoding="utf-8",
    )
    (docs / "history.md").write_text(
        "PlanState\n" * 20,
        encoding="utf-8",
    )

    execution = await _execute(
        _tools_by_name(workspace)["search_text"],
        {"pattern": "PlanState", "path": ".", "max_results": 1},
        workspace=workspace,
    )

    assert execution.result.structured_content is not None
    [match] = execution.result.structured_content["matches"]
    assert match["file_path"] == "src/state.py"
    assert match["line_number"] == 3
    assert match["context_before"] == (
        "from pydantic import BaseModel",
        "",
    )
    assert match["context_after"] == ("    revision: int = 0",)
    assert execution.result.structured_content["truncated"] is True


@pytest.mark.anyio
async def test_search_text_deprioritizes_agent_support_sources_at_workspace_root(
    tmp_path: Path,
) -> None:
    workspace = open_workspace(tmp_path, create=True)
    product_source = workspace.root / "src"
    support_source = workspace.root / ".agents" / "skills" / "helper"
    product_source.mkdir()
    support_source.mkdir(parents=True)
    (product_source / "runtime.py").write_text(
        "class RuntimeSystem:\n    pass\n",
        encoding="utf-8",
    )
    (support_source / "scripts.py").write_text(
        "class SupportSystem:\n    pass\n",
        encoding="utf-8",
    )

    execution = await _execute(
        _tools_by_name(workspace)["search_text"],
        {"pattern": "System", "path": ".", "max_results": 1},
        workspace=workspace,
    )

    assert execution.result.structured_content is not None
    [match] = execution.result.structured_content["matches"]
    assert match["file_path"] == "src/runtime.py"


@pytest.mark.anyio
async def test_search_text_directory_results_are_diverse_across_matching_files(
    tmp_path: Path,
) -> None:
    workspace = open_workspace(tmp_path, create=True)
    source = workspace.root / "src"
    source.mkdir()
    (source / "noisy.py").write_text(
        "\n".join(f"OpenAI noisy reference {index}" for index in range(20)),
        encoding="utf-8",
    )
    (source / "relevant.py").write_text(
        "class OpenAIWireRequest:\n    pass\n",
        encoding="utf-8",
    )

    execution = await _execute(
        _tools_by_name(workspace)["search_text"],
        {
            "pattern": "OpenAI",
            "path": "src",
            "max_results": 5,
        },
        workspace=workspace,
    )

    assert execution.result.structured_content is not None
    matches = execution.result.structured_content["matches"]
    assert len(matches) == 5
    assert {match["file_path"] for match in matches} == {
        "src/noisy.py",
        "src/relevant.py",
    }
    assert execution.result.structured_content["truncated"] is True


@pytest.mark.anyio
async def test_search_text_skips_generated_directories_by_default(
    tmp_path: Path,
) -> None:
    workspace = open_workspace(tmp_path, create=True)
    for directory in ("src", ".venv/lib", ".rag/cache", "node_modules/pkg"):
        (workspace.root / directory).mkdir(parents=True)
        (workspace.root / directory / "target.py").write_text(
            "generated_directory_marker\n",
            encoding="utf-8",
        )

    execution = await _execute(
        _tools_by_name(workspace)["search_text"],
        {"pattern": "generated_directory_marker", "path": ".", "glob": "*.py"},
        workspace=workspace,
    )

    assert execution.result.structured_content is not None
    assert [
        match["file_path"]
        for match in execution.result.structured_content["matches"]
    ] == ["src/target.py"]


@pytest.mark.anyio
async def test_search_text_does_not_follow_symlinks_or_escape_workspace(
    tmp_path: Path,
) -> None:
    workspace = open_workspace(tmp_path / "workspace", create=True)
    source = workspace.root / "src"
    source.mkdir()
    (source / "inside.py").write_text("safe_marker\n", encoding="utf-8")
    outside = tmp_path / "outside.py"
    outside.write_text("outside_marker\n", encoding="utf-8")
    try:
        (source / "external.py").symlink_to(outside)
        (workspace.root / "linked_src").symlink_to(source, target_is_directory=True)
    except OSError:
        pytest.skip("symlink creation is unavailable")
    search = _tools_by_name(workspace)["search_text"]

    safe = await _execute(
        search,
        {"pattern": "marker", "path": ".", "glob": "*.py"},
        workspace=workspace,
    )
    escaped = await _execute(
        search,
        {"pattern": "outside", "path": "../outside.py"},
        workspace=workspace,
    )

    assert safe.result.structured_content is not None
    assert [
        match["file_path"] for match in safe.result.structured_content["matches"]
    ] == ["src/inside.py"]
    assert escaped.result.error_code == "workspace_escape"


@pytest.mark.anyio
async def test_update_plan_uses_the_injected_state_callback(tmp_path: Path) -> None:
    workspace = open_workspace(tmp_path, create=True)
    updates: list[Mapping[str, Any]] = []
    update_plan = _tools_by_name(workspace, updates=updates)["update_plan"]

    execution = await _execute(
        update_plan,
        {
            "explanation": "Show the next implementation checkpoint.",
            "plan": [
                {
                    "step_id": "step_implement",
                    "step": "Implement the resident tools",
                    "status": "in_progress",
                },
                {
                    "step": "Run focused tests",
                    "status": "pending",
                },
            ],
        },
        workspace=workspace,
    )

    assert execution.result.is_error is False
    assert execution.result.structured_content == {
        "accepted": True,
        "revision": 1,
        "message": "plan updated",
        "authority": "advisory",
    }
    assert len(updates) == 1
    assert updates[0]["plan"][0]["step_id"] == "step_implement"
    assert updates[0]["plan"][0]["step"] == "Implement the resident tools"
    assert updates[0]["explanation"] == (
        "Show the next implementation checkpoint."
    )


@pytest.mark.anyio
async def test_update_plan_accepts_strategy_without_task_choreography(
    tmp_path: Path,
) -> None:
    workspace = open_workspace(tmp_path, create=True)
    updates: list[Mapping[str, Any]] = []
    update_plan = _tools_by_name(workspace, updates=updates)["update_plan"]

    execution = await _execute(
        update_plan,
        {
            "plan": [
                {
                    "step": "Try the next viable strategy",
                    "status": "in_progress",
                }
            ],
        },
        workspace=workspace,
    )

    assert execution.result.is_error is False
    assert set(updates[0]) == {"plan", "explanation"}
    assert set(updates[0]["plan"][0]) == {
        "step_id",
        "step",
        "status",
    }


@pytest.mark.anyio
async def test_update_plan_requires_only_a_visible_plan(
    tmp_path: Path,
) -> None:
    workspace = open_workspace(tmp_path, create=True)
    update_plan = _tools_by_name(workspace)["update_plan"]

    missing_plan = await _execute(
        update_plan,
        {},
        workspace=workspace,
    )

    assert missing_plan.result.error_code == "invalid_arguments"
    assert "plan" in (missing_plan.result.error_message or "")
    assert set(update_plan.definition.input_schema["required"]) == {"plan"}
    assert "goal_id" not in update_plan.definition.input_schema["properties"]


@pytest.mark.anyio
async def test_update_plan_rejects_removed_plan_authority_fields(
    tmp_path: Path,
) -> None:
    workspace = open_workspace(tmp_path, create=True)
    update_plan = _tools_by_name(workspace)["update_plan"]

    execution = await _execute(
        update_plan,
        {
            "target_files": ["../outside.py"],
            "plan": [
                {
                    "step": "Read the target",
                    "status": "in_progress",
                }
            ],
        },
        workspace=workspace,
    )

    assert execution.result.is_error is True
    assert execution.result.error_code == "invalid_arguments"
    assert "target_files" in (execution.result.error_message or "")


@pytest.mark.anyio
@pytest.mark.usefixtures("fake_sandbox_exec")
async def test_run_command_returns_bounded_structured_process_output(
    tmp_path: Path,
) -> None:
    workspace = open_workspace(tmp_path, create=True)
    tools = _tools_by_name(workspace)

    execution = await _execute(
        tools["run_command"],
        {
            "command": "printf 'hello'; printf 'warning' >&2",
            "working_dir": ".",
            "timeout_seconds": 1,
        },
        workspace=workspace,
    )

    assert execution.result.is_error is False
    assert execution.result.structured_content is not None
    assert execution.result.structured_content["stdout"] == "hello"
    assert execution.result.structured_content["stderr"] == "warning"
    assert execution.result.structured_content["exit_code"] == 0
    assert execution.result.structured_content["timed_out"] is False
    assert execution.result.structured_content["execution_mode"] == "restricted_sandbox"
    assert execution.result.structured_content["network_enabled"] is False


@pytest.mark.anyio
@pytest.mark.usefixtures("fake_sandbox_exec")
async def test_run_command_nonzero_exit_is_a_canonical_tool_failure(
    tmp_path: Path,
) -> None:
    workspace = open_workspace(tmp_path, create=True)
    tools = _tools_by_name(workspace)

    execution = await _execute(
        tools["run_command"],
        {
            "command": "printf 'failure evidence' >&2; exit 7",
            "working_dir": ".",
            "timeout_seconds": 1,
        },
        workspace=workspace,
    )

    assert execution.result.is_error is True
    assert execution.result.error_code == "command_failed"
    assert execution.result.error_message == "command exited with status 7"
    assert execution.result.retryable is False
    assert execution.result.structured_content is not None
    assert execution.result.structured_content["stderr"] == "failure evidence"
    assert execution.result.structured_content["exit_code"] == 7


@pytest.mark.anyio
@pytest.mark.usefixtures("fake_sandbox_exec")
async def test_run_command_does_not_inherit_host_environment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "host-secret-must-not-cross-boundary")
    monkeypatch.setenv("GROQ_API_KEY", "host-secret-must-not-cross-boundary")
    monkeypatch.setenv("SSH_AUTH_SOCK", "/private/tmp/host-agent.sock")
    monkeypatch.setenv("DOCKER_HOST", "unix:///var/run/docker.sock")
    monkeypatch.setenv("DATABASE_PASSWORD", "host-password")
    workspace = open_workspace(tmp_path, create=True)
    tools = _tools_by_name(workspace)

    execution = await _execute(
        tools["run_command"],
        {
            "command": (
                "printf '%s|%s|%s|%s|%s' "
                '"${OPENAI_API_KEY-unset}" '
                '"${GROQ_API_KEY-unset}" '
                '"${SSH_AUTH_SOCK-unset}" '
                '"${DOCKER_HOST-unset}" '
                '"${DATABASE_PASSWORD-unset}"'
            ),
            "working_dir": ".",
            "timeout_seconds": 1,
        },
        workspace=workspace,
    )

    assert execution.result.structured_content is not None
    assert execution.result.structured_content["stdout"] == (
        "unset|unset|unset|unset|unset"
    )


@pytest.mark.anyio
async def test_run_command_rejects_environment_injection(tmp_path: Path) -> None:
    workspace = open_workspace(tmp_path, create=True)
    tools = _tools_by_name(workspace)

    execution = await _execute(
        tools["run_command"],
        {
            "command": "printf safe",
            "working_dir": ".",
            "timeout_seconds": 1,
            "env": {"API_TOKEN": "must-not-enter-command"},
        },
        workspace=workspace,
    )

    assert execution.result.error_code == "invalid_arguments"


@pytest.mark.anyio
@pytest.mark.skipif(
    shutil.which("sandbox-exec") is None,
    reason="Seatbelt sandbox-exec is not available on this platform",
)
async def test_run_command_cannot_read_outside_workspace(
    tmp_path: Path,
) -> None:
    workspace = open_workspace(tmp_path / "workspace", create=True)
    outside = tmp_path / "outside-secret.txt"
    outside.write_text("must stay outside", encoding="utf-8")
    tools = _tools_by_name(workspace)

    execution = await _execute(
        tools["run_command"],
        {
            "command": f"cat {shlex.quote(str(outside))} >/dev/null",
            "working_dir": ".",
            "timeout_seconds": 1,
        },
        workspace=workspace,
    )

    assert execution.result.structured_content is not None
    assert execution.result.structured_content["exit_code"] != 0
    assert "Operation not permitted" in execution.result.structured_content["stderr"]


@pytest.mark.anyio
@pytest.mark.skipif(
    shutil.which("sandbox-exec") is None,
    reason="Seatbelt sandbox-exec is not available on this platform",
)
async def test_run_command_cannot_list_host_home(tmp_path: Path) -> None:
    workspace = open_workspace(tmp_path, create=True)
    command = _tools_by_name(workspace)["run_command"]

    execution = await _execute(
        command,
        {
            "command": f"ls {shlex.quote(str(Path.home()))} >/dev/null",
            "working_dir": ".",
            "timeout_seconds": 1,
        },
        workspace=workspace,
    )

    assert execution.result.structured_content is not None
    assert execution.result.structured_content["exit_code"] != 0
    assert "Operation not permitted" in execution.result.structured_content["stderr"]


@pytest.mark.anyio
@pytest.mark.skipif(
    shutil.which("sandbox-exec") is None,
    reason="Seatbelt sandbox-exec is not available on this platform",
)
async def test_run_command_cannot_follow_workspace_symlink_outside(
    tmp_path: Path,
) -> None:
    workspace = open_workspace(tmp_path / "workspace", create=True)
    outside = tmp_path / "outside-secret.txt"
    outside.write_text("must stay outside", encoding="utf-8")
    (workspace.root / "outside-link").symlink_to(outside)
    command = _tools_by_name(workspace)["run_command"]

    execution = await _execute(
        command,
        {
            "command": "cat outside-link >/dev/null",
            "working_dir": ".",
            "timeout_seconds": 1,
        },
        workspace=workspace,
    )

    assert execution.result.structured_content is not None
    assert execution.result.structured_content["exit_code"] != 0
    assert "Operation not permitted" in execution.result.structured_content["stderr"]


@pytest.mark.anyio
@pytest.mark.skipif(
    shutil.which("sandbox-exec") is None,
    reason="Seatbelt sandbox-exec is not available on this platform",
)
async def test_run_command_uses_and_removes_private_command_temp(
    tmp_path: Path,
) -> None:
    workspace = open_workspace(tmp_path, create=True)
    command = _tools_by_name(workspace)["run_command"]

    raw = await command.run(
        {
            "command": "printf '%s' \"$TMPDIR\"; touch \"$TMPDIR/probe\"",
            "working_dir": ".",
            "timeout_seconds": 1,
        }
    )

    temp_path = Path(raw.stdout)
    assert temp_path.parent == workspace.scratch
    assert temp_path.name.startswith("run-command-")
    assert temp_path.exists() is False


@pytest.mark.anyio
@pytest.mark.skipif(
    shutil.which("sandbox-exec") is None,
    reason="Seatbelt sandbox-exec is not available on this platform",
)
async def test_run_command_network_flag_controls_ip_network_only(
    tmp_path: Path,
) -> None:
    workspace = open_workspace(tmp_path, create=True)
    command = _tools_by_name(workspace)["run_command"]
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.bind(("127.0.0.1", 0))
    listener.listen()
    port = listener.getsockname()[1]
    probe = f"/usr/bin/nc -z -w 1 127.0.0.1 {port}"
    try:
        blocked = await command.run(
            {
                "command": probe,
                "working_dir": ".",
                "timeout_seconds": 2,
            }
        )
        allowed = await command.run(
            {
                "command": probe,
                "working_dir": ".",
                "timeout_seconds": 2,
                "network": True,
            }
        )
    finally:
        listener.close()

    assert blocked.exit_code != 0
    assert blocked.network_enabled is False
    assert allowed.exit_code == 0, allowed.stderr
    assert allowed.network_enabled is True


@pytest.mark.anyio
@pytest.mark.skipif(
    shutil.which("sandbox-exec") is None,
    reason="Seatbelt sandbox-exec is not available on this platform",
)
async def test_run_command_network_approval_does_not_enable_unix_sockets(
) -> None:
    with tempfile.TemporaryDirectory(prefix="run-command-test-", dir="/tmp") as root:
        workspace = open_workspace(root, create=True)
        command = _tools_by_name(workspace)["run_command"]
        socket_path = workspace.root / "service.sock"
        listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        listener.bind(str(socket_path))
        listener.listen()
        python_code = (
            "import socket; "
            "client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM); "
            f"client.connect({str(socket_path)!r})"
        )
        try:
            result = await command.run(
                {
                    "command": (
                        "/Library/Developer/CommandLineTools/usr/bin/python3 "
                        f"-c {shlex.quote(python_code)}"
                    ),
                    "working_dir": ".",
                    "timeout_seconds": 2,
                    "network": True,
                }
            )
        finally:
            listener.close()

    assert result.exit_code != 0
    assert "Operation not permitted" in result.stderr


def test_run_command_network_profile_limits_unix_sockets_to_dns(
    tmp_path: Path,
) -> None:
    profile = shell_module._build_command_sandbox_profile(
        workspace_root=tmp_path / "workspace",
        temporary_root=tmp_path / "temporary",
        allow_network=True,
    )

    assert '(allow network-outbound (remote ip "*:*"))' in profile
    assert 'literal "/private/var/run/mDNSResponder"' in profile
    assert "docker.sock" not in profile
    assert "com.apple.SecurityServer" not in profile


def test_run_command_profile_mounts_trusted_toolchain_read_only(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    toolchain = Path("/opt/uv/python/cpython-3.12-test")
    monkeypatch.setattr(
        shell_module,
        "_trusted_toolchain_roots",
        lambda _workspace_root: (toolchain,),
    )

    profile = shell_module._build_command_sandbox_profile(
        workspace_root=tmp_path / "workspace",
        temporary_root=tmp_path / "temporary",
        allow_network=False,
    )

    assert f'(subpath "{toolchain}")' in profile
    write_section = profile.split("(allow file-write*", maxsplit=1)[1]
    assert str(toolchain) not in write_section


@pytest.mark.anyio
async def test_run_command_fails_closed_when_sandbox_is_unavailable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = open_workspace(tmp_path, create=True)
    sentinel = workspace.root / "must-not-exist.txt"
    monkeypatch.setattr(
        shell_module,
        "_SANDBOX_EXEC_PATH",
        str(tmp_path / "missing-sandbox-exec"),
    )
    tools = _tools_by_name(workspace)

    execution = await _execute(
        tools["run_command"],
        {
            "command": f"touch {shlex.quote(str(sentinel))}",
            "working_dir": ".",
            "timeout_seconds": 1,
        },
        workspace=workspace,
    )

    assert execution.result.error_code == "sandbox_unavailable"
    assert sentinel.exists() is False


def test_search_builtin_has_no_embedding_or_retrieval_import_dependency() -> None:
    module_path = Path(search_module.__file__ or "")
    tree = ast.parse(module_path.read_text(encoding="utf-8"))
    imported_modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            imported_modules.add(node.module)

    forbidden_prefixes = (
        "rag.retrieval",
        "rag.ingestion",
        "chromadb",
        "faiss",
        "sentence_transformers",
    )
    assert not any(
        module.startswith(forbidden_prefixes) for module in imported_modules
    )
