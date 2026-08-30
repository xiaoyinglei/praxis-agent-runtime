from __future__ import annotations

from agent_runtime.tools.builtins.data import create_inspect_data_file_tool
from agent_runtime.tools.builtins.filesystem import (
    create_apply_patch_tool,
    create_list_files_tool,
    create_read_file_tool,
)
from agent_runtime.tools.builtins.planning import PlanUpdater, create_update_plan_tool
from agent_runtime.tools.builtins.search import create_search_text_tool
from agent_runtime.tools.builtins.shell import (
    create_execute_python_tool,
    create_run_command_tool,
)
from agent_runtime.tools.tool import Tool
from agent_runtime.workspace import WorkspaceRuntime

RESIDENT_CODING_TOOL_NAMES = (
    "search_text",
    "list_files",
    "read_file",
    "inspect_data_file",
    "apply_patch",
    "run_command",
    "execute_python",
    "update_plan",
)


def create_resident_coding_tools(
    workspace: WorkspaceRuntime,
    *,
    plan_updater: PlanUpdater,
    command_hard_timeout_seconds: float = 605.0,
) -> tuple[Tool, ...]:
    """Build the fixed baseline as ordinary Tool values in product order."""

    return (
        create_search_text_tool(workspace),
        create_list_files_tool(workspace),
        create_read_file_tool(workspace),
        create_inspect_data_file_tool(workspace),
        create_apply_patch_tool(workspace),
        create_run_command_tool(
            workspace,
            hard_timeout_seconds=command_hard_timeout_seconds,
        ),
        create_execute_python_tool(
            workspace,
            hard_timeout_seconds=command_hard_timeout_seconds,
        ),
        create_update_plan_tool(plan_updater),
    )


__all__ = [
    "RESIDENT_CODING_TOOL_NAMES",
    "create_apply_patch_tool",
    "create_execute_python_tool",
    "create_inspect_data_file_tool",
    "create_list_files_tool",
    "create_read_file_tool",
    "create_resident_coding_tools",
    "create_run_command_tool",
    "create_search_text_tool",
    "create_update_plan_tool",
]
