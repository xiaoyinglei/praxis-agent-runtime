"""Capture host-owned environment facts for append-only runtime context."""

from __future__ import annotations

import sys


def capture_runtime_environment(workspace: str) -> dict[str, str]:
    return {
        "workspace_root": workspace,
        "default_tool_working_directory": workspace,
        "host_platform": sys.platform,
    }
