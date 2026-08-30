from __future__ import annotations

import ast
import re
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).parents[2]
PRODUCTION_ROOTS = (
    REPOSITORY_ROOT / "rag",
    REPOSITORY_ROOT / "agent_runtime",
    REPOSITORY_ROOT / "scripts",
)
LEGACY_REFERENCE = re.compile(
    r"rag\.agent\.tooling|"
    r"ToolSpec|"
    r"BaseTool|"
    r"ToolCard|"
    r"MCPToolRegistry|"
    r"ToolExecutionService|"
    r"ToolExecutorLoopAdapter|"
    r"ToolSurface(?:Request|Decision|Policy)|"
    r"DiscoveryPolicy|"
    r"ModelRequestBuilder|"
    r"ToolCatalog|"
    r"DeferredToolStore|"
    r"RuntimeToolRegistryBuilder|"
    r"resolve_visible_tools|"
    r"tool_search|"
    r"activate_tools|"
    r"tool_repl|"
    r"ToolOutputFormatter|"
    r"ToolOutputFormatterResolver|"
    r"register_formatter|"
    r"get_formatter|"
    r"format_tool_result_fallback"
)
FINAL_DEFINITIONS = {
    "Tool": ("class", "agent_runtime/tools/tool.py"),
    "ToolRegistry": ("class", "agent_runtime/tools/registry.py"),
    "select_tools": ("function", "agent_runtime/tools/selection.py"),
    "can_use_tool": ("function", "agent_runtime/tools/permissions.py"),
    "ToolExecutor": ("class", "agent_runtime/tools/executor.py"),
    "ToolResult": ("class", "agent_runtime/tools/tool.py"),
}


def _production_files() -> tuple[Path, ...]:
    return tuple(
        sorted(
            path
            for root in PRODUCTION_ROOTS
            for path in root.rglob("*.py")
            if "__pycache__" not in path.parts
        )
    )


def test_no_legacy_tool_runtime_references_remain() -> None:
    violations: list[str] = []
    for path in _production_files():
        relative = path.relative_to(REPOSITORY_ROOT).as_posix()
        for line_number, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(),
            start=1,
        ):
            matches = tuple(
                dict.fromkeys(match.group(0) for match in LEGACY_REFERENCE.finditer(line))
            )
            if matches:
                violations.append(
                    f"{relative}:{line_number}: {', '.join(matches)}"
                )

    assert not violations, "legacy tool runtime references remain:\n" + "\n".join(
        violations
    )


def test_final_tool_runtime_concepts_have_one_definition() -> None:
    definitions: dict[str, list[tuple[str, str]]] = {
        name: [] for name in FINAL_DEFINITIONS
    }
    for path in _production_files():
        relative = path.relative_to(REPOSITORY_ROOT).as_posix()
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name in definitions:
                definitions[node.name].append(("class", relative))
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and (
                node.name in definitions
            ):
                definitions[node.name].append(("function", relative))

    assert definitions == {
        name: [expected]
        for name, expected in FINAL_DEFINITIONS.items()
    }
