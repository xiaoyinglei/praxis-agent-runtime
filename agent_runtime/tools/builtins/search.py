from __future__ import annotations

import asyncio
import importlib.metadata
import json
import os
import signal
import sys
from collections.abc import Mapping
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from agent_runtime.tools.tool import (
    CancellationMode,
    InterruptBehavior,
    JsonValue,
    NormalizedToolOutput,
    ResolvedToolUse,
    Tool,
    ToolDefinition,
    ToolEffect,
    ToolTarget,
    json_schema_output,
    pydantic_input,
)
from agent_runtime.workspace import WorkspaceRuntime

_MAX_SEARCH_FILE_BYTES = 2_000_000


class SearchTextInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    pattern: str = Field(
        min_length=1,
        max_length=2000,
        description="Literal text or regular expression to find.",
    )
    path: str = Field(
        default=".",
        max_length=4096,
        description="Workspace-relative or authorized absolute file or directory to search.",
    )
    glob: str | None = Field(
        default=None,
        max_length=512,
        description="Optional file glob such as '*.py' or 'src/**/*.ts'.",
    )
    regex: bool | None = Field(
        default=None,
        description=(
            "Matching mode. Use true for a ripgrep (Rust) regular expression and false "
            "for exact literal text. Omit it to auto-detect common regex syntax "
            "such as '.*', '|', '^', '$', character classes, or escaped classes."
        ),
    )
    context_lines: int = Field(
        default=2,
        ge=0,
        le=50,
        description=(
            "Lines of context to return before and after each match; defaults "
            "to two so symbol searches usually reveal the local definition."
        ),
    )
    max_results: int = Field(
        default=40,
        ge=1,
        le=200,
        description="Maximum matching lines to return.",
    )


class SearchTextMatch(BaseModel):
    model_config = ConfigDict(extra="forbid")

    file_path: str
    line_number: int = Field(ge=1)
    line_content: str
    match_start: int = Field(ge=0)
    match_end: int = Field(ge=0)
    context_before: list[str] = Field(default_factory=list)
    context_after: list[str] = Field(default_factory=list)


class SearchTextOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    matches: list[SearchTextMatch]
    total_matches: int = Field(ge=0)
    truncated: bool = False
    searched_file_path: str | None = None


_SEARCH_INPUT_SCHEMA, _validate_search_model = pydantic_input(SearchTextInput)
_SEARCH_OUTPUT_SCHEMA, _unused_search_output_validator = pydantic_input(SearchTextOutput)


def create_search_text_tool(workspace: WorkspaceRuntime) -> Tool:
    async def run(arguments: Mapping[str, JsonValue]) -> object:
        return await _search_text(workspace, SearchTextInput.model_validate(arguments))

    return Tool(
        definition=ToolDefinition(
            name="search_text",
            description=(
                "Search current workspace files without an index. Use it to locate "
                "relevant symbols, text, or files. Supports literal or regex matching, one file or directory "
                "path, an optional glob, bounded context lines, and a result limit. "
                "Files are searched in deterministic path order using the bundled ripgrep engine. "
                "Traversal, file bytes, context and output are bounded; truncated means "
                "the search or displayed text is incomplete. Symlinks are not followed; "
                "each call observes current authorized file contents. A targeted post-write search can "
                "confirm literal presence, or absence only when its exact-file scope is "
                "reported and non-truncated. Reuse that result while the workspace is "
                "unchanged. Search verifies matching text within the reported scope, "
                "not execution behavior."
            ),
            input_schema=_SEARCH_INPUT_SCHEMA,
        ),
        validate_input=_validate_search_input,
        run=run,
        normalize_output=_normalize_search_output,
        output_schema=_SEARCH_OUTPUT_SCHEMA,
        static_effects=frozenset({ToolEffect.READ_WORKSPACE}),
        resolve_use=lambda arguments: ResolvedToolUse(
            effects=frozenset({ToolEffect.READ_WORKSPACE}),
            targets=(
                ToolTarget(
                    kind="workspace_path",
                    value=str(workspace.resolve_path(str(arguments["path"]) or ".")),
                ),
            ),
        ),
        execution_revision="builtin-search-text-v6-ripgrep-worker",
        idempotent=True,
        concurrency_safe=True,
        cancellation_mode=CancellationMode.COOPERATIVE,
        interrupt_behavior=InterruptBehavior.CANCEL,
        timeout_seconds=15.0,
        max_model_output_bytes=2_000_000,
    )


def _validate_search_input(
    arguments: Mapping[str, JsonValue],
) -> Mapping[str, JsonValue]:
    return _validate_search_model(arguments)


def _ripgrep_binary() -> Path:
    """Locate our locked dependency, never an unrelated executable on PATH."""
    distribution = importlib.metadata.distribution("ripgrep-bin")
    for entry in distribution.files or ():
        if entry.name in {"rg", "rg.exe"}:
            binary = Path(str(distribution.locate_file(entry))).resolve()
            if binary.is_file() and os.access(binary, os.X_OK):
                return binary
    raise FileNotFoundError("Installed ripgrep-bin has no executable rg")


async def _stop_worker(process: asyncio.subprocess.Process) -> None:
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        await asyncio.wait_for(process.wait(), timeout=0.5)
    except TimeoutError:
        pass
    finally:
        # The worker may exit before a stubborn descendant. Always finish the
        # whole process group, even when waiting for its leader already returned.
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        await process.wait()


async def _search_text(workspace: WorkspaceRuntime, request: SearchTextInput) -> object:
    # Keep authorization at the public boundary; traversal repeats it in the worker.
    workspace.ensure_readable(workspace.resolve_path(request.path or "."))
    try:
        binary = _ripgrep_binary()
    except (importlib.metadata.PackageNotFoundError, FileNotFoundError):
        return {
            "error_code": "search_backend_unavailable",
            "error_message": "Search requires the packaged ripgrep-bin dependency; reinstall Praxis.",
        }
    package_root = Path(__file__).resolve().parents[3]
    bootstrap = (
        f"import sys; sys.path.insert(0, {str(package_root)!r}); "
        "from agent_runtime.tools.builtins.search_worker import main; main()"
    )
    permissions = workspace.permissions
    payload = json.dumps(
        {
            "root": str(workspace.root.resolve()),
            "permissions": permissions.model_dump(mode="json") if permissions is not None else None,
            "request": request.model_dump(mode="json"),
            "regex": _uses_regular_expression(request.pattern, request.regex),
            "binary": str(binary),
        }
    ).encode()
    process = await asyncio.create_subprocess_exec(
        sys.executable,
        "-I",
        "-c",
        bootstrap,
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.DEVNULL,
        start_new_session=True,
    )
    try:
        stdout, _ = await process.communicate(payload)
        if process.returncode != 0:
            return {"error_code": "search_worker_failed", "error_message": "Search worker failed; retry the search."}
        result = json.loads(stdout)
        return result if "error_code" in result else SearchTextOutput.model_validate(result)
    finally:
        # A cancelled caller must not leave a worker or backend touching the tree.
        cleanup = asyncio.create_task(_stop_worker(process))
        while not cleanup.done():
            try:
                await asyncio.shield(cleanup)
            except asyncio.CancelledError:
                continue
        cleanup.result()


def _uses_regular_expression(pattern: str, requested: bool | None) -> bool:
    if requested is not None:
        return requested
    return any(
        marker in pattern
        for marker in (
            ".*",
            ".+",
            ".?",
            "|",
            "^",
            "$",
            "(?",
            "[",
            r"\b",
            r"\d",
            r"\s",
            r"\w",
        )
    )


def _normalize_search_output(raw: object) -> NormalizedToolOutput:
    if isinstance(raw, dict) and "error_code" in raw:
        return NormalizedToolOutput(
            structured_content={"matches": (), "total_matches": 0, "truncated": True, "searched_file_path": None},
            is_error=True,
            error_code=str(raw["error_code"]),
            error_message=str(raw["error_message"]),
            retryable=False,
        )
    validated = SearchTextOutput.model_validate(raw)
    structured = json_schema_output(
        _SEARCH_OUTPUT_SCHEMA,
        validated.model_dump(mode="json"),
    )
    return NormalizedToolOutput(structured_content=structured)


__all__ = [
    "SearchTextInput",
    "SearchTextMatch",
    "SearchTextOutput",
    "create_search_text_tool",
]
