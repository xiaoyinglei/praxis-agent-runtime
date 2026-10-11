"""Optional Codex patch executor using the existing workspace sandbox."""

from __future__ import annotations

import asyncio
import hashlib
import os
import stat
import sys
import tempfile
from collections.abc import Mapping
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from agent_runtime.tools.builtins.filesystem import _atomic_create_text, _atomic_write_text
from agent_runtime.tools.builtins.shell import (
    RunCommandInput,
    _bounded_stream,
    _build_command_sandbox_profile,
    _command_environment,
    _escape_seatbelt_string,
    _linux_sandbox_argv,
    _sandbox_availability_issue,
    _terminate_process_group,
    _workspace_sandbox_preflight_issue,
)
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


class CodexPatchInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    patch: str = Field(min_length=1, max_length=30_000, description="Codex patch text, at most 30,000 characters.")


class CodexPatchOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    stdout: str
    stderr: str
    exit_code: int
    timed_out: bool = False
    truncated: bool = False


_INPUT_SCHEMA, _VALIDATE_INPUT = pydantic_input(CodexPatchInput)
_OUTPUT_SCHEMA, _UNUSED_OUTPUT_VALIDATOR = pydantic_input(CodexPatchOutput)
_EFFECTS = frozenset({ToolEffect.READ_WORKSPACE, ToolEffect.WRITE_WORKSPACE, ToolEffect.DESTRUCTIVE})


def create_codex_apply_patch_tool(workspace: WorkspaceRuntime, executable: str | Path) -> Tool:
    binary = Path(executable).expanduser().resolve(strict=True)
    if not binary.is_file() or not os.access(binary, os.X_OK):
        raise ValueError("PRAXIS_CODEX_APPLY_PATCH must name an executable Codex binary")
    if sys.platform not in {"darwin", "linux"}:
        raise ValueError("the optional Codex patch adapter requires macOS or Linux Bubblewrap")
    if sys.platform == "linux" and binary.is_relative_to(workspace.root):
        raise ValueError("on Linux, store the trusted Codex binary outside the workspace")
    with binary.open("rb") as stream:
        binary_sha256 = hashlib.file_digest(stream, "sha256").hexdigest()

    def normalize(raw: object) -> NormalizedToolOutput:
        output = CodexPatchOutput.model_validate(raw)
        failed = output.exit_code != 0 or output.timed_out
        return NormalizedToolOutput(
            structured_content=json_schema_output(_OUTPUT_SCHEMA, output.model_dump(mode="json")),
            is_error=failed,
            error_code="patch_failed" if failed else None,
            error_message=(
                "Codex patch failed. Earlier hunks may already have changed files; "
                "inspect the workspace before retrying."
                if failed
                else None
            ),
            retryable=failed,
        )

    return Tool(
        definition=ToolDefinition(
            name="apply_patch",
            description=(
                "Edit workspace files with the Codex patch executor. Send JSON with one patch string. "
                "Start with *** Begin Patch and end with *** End Patch. "
                "Use *** Add File: path with every content line prefixed +; *** Delete File: path; "
                "or *** Update File: path followed by @@ and hunks using space for unchanged context, "
                "- for removed lines and + for added lines. Updates require matching context. "
                "An update may include *** Move to: destination before its hunks. "
                "Multiple files and hunks are supported. Add and move operations can overwrite existing files. "
                "Paths must remain in the workspace; .git, .venv, node_modules and agent state are protected. "
                "A failed patch may have applied earlier hunks; the result is not transactional. "
                "Success confirms file edits, not execution behavior. Maximum patch length is 30,000 characters. "
                "Linux stages only declared regular files, rejects symlink paths and parent traversal, "
                "and limits each patch to 128 target files and 64 MiB of input or output files."
            ),
            input_schema=_INPUT_SCHEMA,
        ),
        validate_input=_VALIDATE_INPUT,
        run=lambda arguments: _run_patch(workspace, binary, str(arguments["patch"])),
        normalize_output=normalize,
        output_schema=_OUTPUT_SCHEMA,
        static_effects=_EFFECTS,
        resolve_use=lambda arguments: _resolve_use(workspace, arguments),
        execution_revision=f"codex-apply-patch-sandbox-v2:{sys.platform}:{binary_sha256}",
        idempotent=False,
        concurrency_safe=False,
        cancellation_mode=CancellationMode.MANAGED_PROCESS,
        interrupt_behavior=InterruptBehavior.CANCEL,
        timeout_seconds=35.0,
        max_model_output_bytes=50_000,
    )


def _resolve_use(workspace: WorkspaceRuntime, arguments: Mapping[str, JsonValue]) -> ResolvedToolUse:
    del arguments
    # Claim the workspace conservatively; native Codex owns hunk application.
    # Linux additionally validates declared paths and commits from a private view.
    return ResolvedToolUse(
        effects=_EFFECTS,
        targets=(ToolTarget(kind="workspace_path", value=str(workspace.root)),),
    )


_PROTECTED_PARTS = frozenset({".git", ".venv", "node_modules", ".agent_memory", ".praxis"})
_MAX_TARGETS = 128
_MAX_FILE_BYTES = 64 * 1024 * 1024


def _linux_patch_targets(patch: str) -> set[str]:
    """Project native header paths only; Codex still parses and applies the patch.

    Match the upstream framing and header whitespace rules. In an update,
    leading spaces are context, including lines that look like file headers.
    """
    lines = patch.strip().split("\n")
    if len(lines) >= 4 and lines[0].strip() in {"<<EOF", "<<'EOF'", '<<"EOF"'} and lines[-1].strip().endswith("EOF"):
        lines = lines[1:-1]
    in_update = False
    targets = set()
    for line in lines[1:-1]:
        header = line.rstrip() if in_update else line.strip()
        for marker in ("*** Add File: ", "*** Delete File: ", "*** Update File: ", "*** Move to: "):
            if header.startswith(marker):
                targets.add(header[len(marker) :])
                if marker != "*** Move to: ":
                    in_update = marker == "*** Update File: "
                break
    return targets


def _linux_target(workspace: WorkspaceRuntime, name: str) -> Path:
    raw = Path(name)
    if not name or ".." in raw.parts:
        raise PermissionError("patch paths must stay within the workspace without parent traversal")
    path = raw if raw.is_absolute() else workspace.root / raw
    if not path.is_relative_to(workspace.root):
        raise PermissionError("patch target is outside the workspace")
    relative = path.relative_to(workspace.root)
    if not relative.parts or any(part in _PROTECTED_PARTS for part in relative.parts):
        raise PermissionError("patch target is a protected path")
    current = workspace.root
    for part in relative.parts:
        current /= part
        if current.is_symlink():
            raise PermissionError("Linux patch targets and their parents cannot be symlinks")
        if current.exists() and (not current.is_dir() if current != path else not current.is_file()):
            raise PermissionError("Linux patch targets must be regular files with directory parents")
    return path


def _read_limited_file(path: Path, remaining: int) -> bytes:
    if path.stat().st_size > remaining:
        raise ValueError("Linux patch file data exceeds the 64 MiB limit")
    with path.open("rb") as stream:
        data = stream.read(remaining + 1)
    if len(data) > remaining:
        raise ValueError("Linux patch file data exceeds the 64 MiB limit")
    return data


def _prepare_linux_patch_view(workspace: WorkspaceRuntime, view: Path, patch: str) -> dict[str, bytes | None]:
    targets = _linux_patch_targets(patch)
    if len(targets) > _MAX_TARGETS:
        raise ValueError("Linux patches are limited to 128 target files")
    initial: dict[str, bytes | None] = {}
    remaining = _MAX_FILE_BYTES
    # Validate and read all targets before constructing the private view.
    for name in sorted(targets):
        path = _linux_target(workspace, name)
        data = _read_limited_file(path, remaining) if path.exists() else None
        if data is not None:
            remaining -= len(data)
        initial[str(path.relative_to(workspace.root))] = data
    view.mkdir()
    for name, data in initial.items():
        if data is not None:
            target = view / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
    return initial


def _commit_linux_patch_view(workspace: WorkspaceRuntime, view: Path, initial: dict[str, bytes | None]) -> None:
    if view.is_symlink() or not view.is_dir():
        raise PermissionError("native patch view root is missing or invalid")

    def raise_walk_error(error: OSError) -> None:
        raise error

    resulting: dict[str, bytes | None] = dict.fromkeys(initial)
    remaining = _MAX_FILE_BYTES
    for directory, dirs, files in os.walk(view, followlinks=False, onerror=raise_walk_error):
        for name in dirs + files:
            path = Path(directory) / name
            mode = path.lstat().st_mode
            relative = str(path.relative_to(view))
            if stat.S_ISDIR(mode):
                if not any(Path(target).is_relative_to(relative) for target in initial):
                    raise PermissionError("native patch produced an undeclared directory")
                continue
            if not stat.S_ISREG(mode) or relative not in initial:
                raise PermissionError("native patch produced an undeclared or nonregular file")
            data = _read_limited_file(path, remaining)
            remaining -= len(data)
            resulting[relative] = data
    changes: dict[str, str | None] = {}
    for name, before in initial.items():
        host = _linux_target(workspace, name)
        current = _read_limited_file(host, _MAX_FILE_BYTES) if host.exists() else None
        if current != before:
            raise PermissionError("workspace target changed while the patch was running; retry after inspecting it")
        after = resulting[name]
        if after != before:
            changes[name] = after.decode("utf-8") if after is not None else None
    # All outputs, paths and snapshots pass validation before the first host edit.
    for name, content in changes.items():
        path = _linux_target(workspace, name)
        current = _read_limited_file(path, _MAX_FILE_BYTES) if path.exists() else None
        if current != initial[name]:
            raise PermissionError("workspace target changed before commit")
        if content is None:
            path.unlink()
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            (_atomic_create_text if initial[name] is None else _atomic_write_text)(path, content)


async def _run_patch(workspace: WorkspaceRuntime, binary: Path, patch: str) -> CodexPatchOutput:
    linux = sys.platform == "linux"
    issue = _sandbox_availability_issue(RunCommandInput(command="codex apply_patch", workspace_write=not linux))
    issue = issue or _workspace_sandbox_preflight_issue(workspace.root, allow_workspace_write=True)
    if issue is not None:
        return CodexPatchOutput(stdout="", stderr=issue[1], exit_code=-1)
    with tempfile.TemporaryDirectory(prefix="codex-patch-", dir=None if linux else workspace.scratch) as temporary:
        initial = None
        view = Path(temporary) / "view"
        if linux:
            try:
                initial = _prepare_linux_patch_view(workspace, view, patch)
            except (OSError, ValueError) as error:
                return CodexPatchOutput(stdout="", stderr=str(error), exit_code=-1)
            argv = _linux_sandbox_argv(
                workspace_root=workspace.root,
                temporary_root=temporary,
                additional_read_roots=(binary,),
            )
            # Overlay the original root with private candidate files. The child
            # has no writable mount of the host workspace or agent state.
            argv.extend(("--bind", str(view), str(workspace.root)))
        else:
            profile = _build_command_sandbox_profile(
                workspace_root=workspace.root,
                temporary_root=temporary,
                allow_network=False,
                allow_workspace_write=True,
                additional_read_roots=(binary,),
            )
            for internal in (workspace.agent_memory, workspace.root / ".praxis"):
                protected = _escape_seatbelt_string(str(internal.resolve()))
                scratch = _escape_seatbelt_string(temporary)
                profile += (
                    f'\n(deny file-read* file-write* (require-all (subpath "{protected}") '
                    f'(require-not (subpath "{scratch}"))))\n'
                )
            argv = ["/usr/bin/sandbox-exec", "-p", profile]
        argv.extend((str(binary), "--codex-run-as-apply-patch", patch))
        process = await asyncio.create_subprocess_exec(
            *argv,
            cwd=workspace.root,
            env=_command_environment(workspace_root=workspace.root, temporary_root=temporary),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            start_new_session=True,
        )
        communication = asyncio.create_task(process.communicate())
        timed_out = False
        try:
            done, _pending = await asyncio.wait({communication}, timeout=30.0)
            if communication not in done:
                timed_out = True
                await _terminate_process_group(process, grace_seconds=0.5)
            stdout_bytes, stderr_bytes = await communication
        except BaseException:
            await _terminate_process_group(process, grace_seconds=0.5)
            await asyncio.gather(communication, return_exceptions=True)
            raise
        stdout, stdout_truncated = _bounded_stream(stdout_bytes)
        stderr, stderr_truncated = _bounded_stream(stderr_bytes)
        exit_code = process.returncode if process.returncode is not None else -1
        if initial is not None and not timed_out:
            try:
                _commit_linux_patch_view(workspace, view, initial)
            except (OSError, ValueError) as error:
                stderr, commit_truncated = _bounded_stream((stderr + "\nCommit refused: " + str(error)).encode())
                stderr_truncated = stderr_truncated or commit_truncated
                exit_code = -1
        return CodexPatchOutput(
            stdout=stdout,
            stderr=stderr,
            exit_code=exit_code,
            timed_out=timed_out,
            truncated=stdout_truncated or stderr_truncated,
        )
