from __future__ import annotations

from rag.agent.tools.builtins.filesystem import (
    create_apply_patch_tool,
    create_list_files_tool,
    create_read_file_tool,
)
from rag.agent.tools.builtins.planning import PlanUpdater, create_update_plan_tool
from rag.agent.tools.builtins.search import create_search_text_tool
from rag.agent.tools.builtins.shell import create_run_command_tool
from rag.agent.tools.tool import Tool
from rag.agent.workspace import WorkspaceRuntime

RESIDENT_CODING_TOOL_NAMES = (
    "search_text",
    "list_files",
    "read_file",
    "apply_patch",
    "run_command",
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
        create_apply_patch_tool(workspace),
        create_run_command_tool(
            workspace,
            hard_timeout_seconds=command_hard_timeout_seconds,
        ),
        create_update_plan_tool(plan_updater),
    )


__all__ = [
    "RESIDENT_CODING_TOOL_NAMES",
    "create_apply_patch_tool",
    "create_list_files_tool",
    "create_read_file_tool",
    "create_resident_coding_tools",
    "create_run_command_tool",
    "create_search_text_tool",
    "create_update_plan_tool",
]
