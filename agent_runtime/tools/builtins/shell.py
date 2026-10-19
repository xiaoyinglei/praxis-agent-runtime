from __future__ import annotations

import asyncio
import codecs
import ctypes
import errno
import hashlib
import json
import math
import os
import platform
import secrets
import shutil
import signal
import socket
import stat
import sys
import tempfile
import threading
import time
from collections.abc import Awaitable, Callable, Mapping, Sequence
from contextlib import ExitStack
from contextvars import ContextVar
from pathlib import Path
from typing import TYPE_CHECKING, BinaryIO, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from agent_runtime.permission_policy import ResolvedPermissions
from agent_runtime.tools.tool import (
    CancellationMode,
    InterruptBehavior,
    JsonValue,
    NormalizedToolOutput,
    ResolvedToolUse,
    Tool,
    ToolApprovalProfile,
    ToolDefinition,
    ToolEffect,
    ToolProgress,
    ToolProgressKind,
    ToolProgressSink,
    ToolTarget,
    ToolValidationError,
    json_schema_output,
    pydantic_input,
)
from agent_runtime.user_config import command_state_directory
from agent_runtime.workspace import WorkspaceRuntime

if TYPE_CHECKING:
    from agent_runtime.tools.builtins.command_sessions import CommandSessions

_MAX_STREAM_BYTES = 16_000
_MAX_COMMAND_TIMEOUT_SECONDS = 600.0
_COPY_CANCEL: ContextVar[threading.Event | None] = ContextVar("linux_copy_cancel", default=None)
_STREAM_TRUNCATION_MARKER = b"\n... output truncated; preserved head and tail ...\n"
_MAX_DELTA_BYTES = 200_000
_MAX_DELTA_CHUNKS = 512
_DELTA_CHUNK_BYTES = 4_096
_SANDBOX_EXEC_PATH = "/usr/bin/sandbox-exec"
_BWRAP_PATH = "/usr/bin/bwrap"
_SANDBOX_PLATFORM = sys.platform
_PROTECTED_VERIFICATION_DIRECTORIES = frozenset({".venv", "node_modules"})
_PROTECTED_PERMISSION_DIRECTORIES = frozenset({".git", ".agents", ".codex", ".aws"})
_LINUX_VIEW_PROTECTED = _PROTECTED_PERMISSION_DIRECTORIES | _PROTECTED_VERIFICATION_DIRECTORIES | {
    ".praxis", ".agent_memory",
}
_LINUX_COPY_CHUNK_BYTES = 1024 * 1024
_LinuxSnapshot = dict[str, tuple[int, bytes | str, tuple[int, int], int]]
_FAILURE_PUBLICATION_DESCRIPTION = (
    "Linux only: with workspace_write=true, publish validated workspace changes even after "
    "nonzero exit or timeout, so incremental builds can resume from their files. "
    "The command still fails; cancellation discards changes. Default false. "
    "All path, permission and host-change checks still apply."
)


class RunCommandInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    command: str = Field(
        min_length=1,
        max_length=8000,
        description="Shell command to execute in an isolated process group.",
    )
    working_dir: str = Field(
        default=".",
        max_length=4096,
        description="Workspace-relative working directory.",
    )
    timeout_seconds: float | None = Field(
        default=120.0,
        gt=0,
        allow_inf_nan=False,
        description=(
            "Execution deadline in seconds, never milliseconds. Default 120. "
            "Deadlines above 600 seconds, or null for no deadline, require yield_time_seconds. "
            "A response wait expiring does not kill a background command."
        ),
    )
    yield_time_seconds: float | None = Field(
        default=None, gt=0, le=60,
        description=(
            "Live macOS/Linux Sessions: return after this many seconds with status=running and a session_id "
            "if still executing. Use manage_command to wait, query or cancel the same process. "
            "This is a response wait, not an execution deadline; timeout_seconds still applies unless null. "
            "Omit to wait for completion with the existing bounded command timeout."
        ),
    )
    network: bool = Field(
        default=False,
        description=(
            "Request outbound network access. Network is disabled by default "
            "and requires a separate approval from command execution."
        ),
    )
    workspace_write: bool = Field(
        default=False,
        description=(
            "Request write access to the workspace. The workspace is read-only "
            "by default; enabling writes is destructive and requires explicit "
            "approval. .git, .agents, .codex, and .aws paths remain read-only; "
            "aliases that bypass these restrictions are refused."
        ),
    )
    publish_on_failure: bool = Field(default=False, description=_FAILURE_PUBLICATION_DESCRIPTION)

    @model_validator(mode="after")
    def require_yield_for_unbounded_command(self) -> RunCommandInput:
        if self.timeout_seconds is None and self.yield_time_seconds is None:
            raise ValueError("timeout_seconds=null requires yield_time_seconds and a live Session")
        if self.timeout_seconds is not None and self.timeout_seconds > 600 and self.yield_time_seconds is None:
            raise ValueError("timeout_seconds above 600 requires yield_time_seconds; units are seconds")
        return self


class RunCommandOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    stdout: str
    stderr: str
    exit_code: int | None
    timed_out: bool
    truncated: bool
    duration_ms: float = Field(ge=0)
    execution_mode: Literal["restricted_sandbox"] = "restricted_sandbox"
    network_enabled: bool = False
    sandbox_error: str | None = None
    workspace_published: bool | None = None
    status: Literal["running", "completed", "cancelled"] = "completed"
    session_id: str | None = None
    request_digest: str | None = None
    log_chunk: str | None = None
    log_next_offset: int | None = None
    log_end_offset: int | None = None
    log_truncated: bool = False

    @model_validator(mode="after")
    def validate_process_status(self) -> RunCommandOutput:
        if self.status == "running":
            if not self.session_id or self.exit_code is not None or self.timed_out or self.sandbox_error:
                raise ValueError("running commands require a session_id and no terminal exit status")
        elif self.exit_code is None:
            raise ValueError("terminal commands require an exit_code")
        return self


class ManagedPythonInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    code: str = Field(
        min_length=1,
        max_length=100_000,
        description=(
            "Python source code executed directly by the managed Praxis "
            "interpreter. Pass plain code, not a shell command or heredoc."
        ),
    )
    working_dir: str = Field(
        default=".",
        max_length=4096,
        description="Workspace-relative working directory for the Python code.",
    )
    output_paths: list[str] = Field(
        default_factory=list,
        max_length=20,
        description=(
            "Workspace-relative or explicitly authorized absolute files to create or replace. "
            "Required when workspace_write=true so the runtime can attest the "
            "generated artifacts."
        ),
    )
    timeout_seconds: float = Field(
        default=120.0,
        gt=0,
        le=_MAX_COMMAND_TIMEOUT_SECONDS, allow_inf_nan=False,
        description=(
            "Execution timeout in seconds, at most 600. Milliseconds are not accepted."
        ),
    )
    network: bool = Field(
        default=False,
        description=("Request outbound network access. It is disabled by default and requires separate approval."),
    )
    workspace_write: bool = Field(
        default=False,
        description=(
            "Request workspace write access. Enabling it is destructive and "
            "requires explicit approval; .git, .venv, and node_modules remain "
            "read-only."
        ),
    )

    publish_on_failure: bool = Field(default=False, description=_FAILURE_PUBLICATION_DESCRIPTION)


class ManagedPythonOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    stdout: str
    stderr: str
    exit_code: int
    timed_out: bool
    truncated: bool
    duration_ms: float = Field(ge=0)
    execution_mode: Literal["managed_python_sandbox"] = "managed_python_sandbox"
    python_version: str
    network_enabled: bool = False
    sandbox_error: str | None = None
    workspace_published: bool | None = None
    verification: dict[str, JsonValue] | None = None


_COMMAND_INPUT_SCHEMA, _validate_command_input = pydantic_input(RunCommandInput)
_COMMAND_OUTPUT_SCHEMA, _unused_command_output_validator = pydantic_input(RunCommandOutput)
_PYTHON_INPUT_SCHEMA, _validate_python_input_base = pydantic_input(ManagedPythonInput)
_PYTHON_OUTPUT_SCHEMA, _unused_python_output_validator = pydantic_input(ManagedPythonOutput)


def _validate_python_input(
    arguments: Mapping[str, JsonValue],
) -> Mapping[str, JsonValue]:
    validated = _validate_python_input_base(arguments)
    raw_paths = validated.get("output_paths")
    output_paths = raw_paths if isinstance(raw_paths, tuple) else ()
    if validated.get("workspace_write") is True and not output_paths:
        raise ToolValidationError(
            path="$.output_paths",
            message="at least one output path is required when workspace_write=true",
        )
    for index, value in enumerate(output_paths):
        if not isinstance(value, str) or not value.strip():
            raise ToolValidationError(
                path=f"$.output_paths[{index}]",
                message="output path must be a non-empty string",
            )
        if len(value) > 4096:
            raise ToolValidationError(
                path=f"$.output_paths[{index}]",
                message="output path must contain at most 4096 characters",
            )
    if len(set(output_paths)) != len(output_paths):
        raise ToolValidationError(
            path="$.output_paths",
            message="output paths must be unique",
        )
    return validated


def create_run_command_tool(
    workspace: WorkspaceRuntime,
    *,
    hard_timeout_seconds: float = 605.0,
    termination_grace_seconds: float = 0.5,
    command_sessions: CommandSessions | None = None,
) -> Tool:
    if command_sessions is not None and command_sessions.workspace is not workspace:
        raise ValueError("command session owner belongs to a different workspace runtime")
    if (
        isinstance(termination_grace_seconds, bool)
        or not isinstance(termination_grace_seconds, (int, float))
        or not math.isfinite(termination_grace_seconds)
        or termination_grace_seconds <= 0
    ):
        raise ValueError("termination_grace_seconds must be positive and finite")

    async def run(arguments: Mapping[str, JsonValue]) -> RunCommandOutput:
        request = RunCommandInput.model_validate(arguments)
        if request.yield_time_seconds is not None:
            if command_sessions is None:
                return RunCommandOutput(
                    stdout="", stderr="command sessions require a live Agent Session",
                    exit_code=-1, timed_out=False, truncated=False, duration_ms=0,
                    sandbox_error="command_session_unavailable",
                )
            return await command_sessions.start(request)
        if command_sessions is not None and command_sessions.running:
            return command_sessions.busy_output()
        return await _run_command(
            workspace,
            request,
            termination_grace_seconds=float(termination_grace_seconds),
        )

    async def stream(
        arguments: Mapping[str, JsonValue],
        progress_sink: ToolProgressSink,
    ) -> RunCommandOutput:
        if arguments.get("yield_time_seconds") is not None or (
            command_sessions is not None and command_sessions.running
        ):
            return await run(arguments)
        return await _run_command(
            workspace,
            RunCommandInput.model_validate(arguments),
            termination_grace_seconds=float(termination_grace_seconds),
            progress_sink=progress_sink,
        )

    return Tool(
        definition=ToolDefinition(
            name="run_command",
            description=(
                "On macOS/Linux live Sessions, set yield_time_seconds to return a session_id while a command runs; "
                "manage_command queries, waits or cancels it. timeout_seconds=null removes the execution "
                "deadline only in this mode; closing the Session cancels its processes. One running command "
                "per Session. macOS writes go directly to the workspace and remain after failure, timeout "
                "or cancellation; publish_on_failure has no effect there. "
                "On Linux, writes use a private candidate workspace with streaming copies and fingerprints. "
                "Only exit 0 with all path, permission and host-change checks passing publishes changes by default; "
                "publish_on_failure=true also publishes validated partial build files on failure or timeout. "
                "Cancellation discards them. Internal relative symlinks are supported; external/absolute links "
                "and protected targets are refused. File mtimes are preserved for incremental builds. "
                "External writes are unsupported. "
                "Linux network access requires libseccomp and blocks host Unix/local IPC; "
                "anonymous stream socketpairs remain available for async event loops. "
                "Run one shell command inside a restricted OS sandbox from a "
                "workspace-relative directory. The workspace is read-only by default "
                "and a private temporary directory is writable. Set "
                "workspace_write=true to request destructive workspace write access; "
                ".git, .agents, .codex, .aws, .venv, and node_modules remain read-only. Host environment "
                "variables and network access are disabled. Set network=true to "
                "request a separate network approval. Timeout or cancellation "
                "terminates and reaps the complete process group. Commands fail "
                "closed before execution when hardlinked workspace files could "
                "bypass path containment. For implementation tasks, locate the "
                "affected code before execution and prefer focused verification "
                "over a full-repository test run. Use recognized read-only commands "
                "to verify code behavior; do not execute a process solely to reconfirm "
                "unchanged literal file content already established by a read or search. "
                "Invoke the project's test runner directly, without pipes, redirects or success fallbacks. "
                "Running a pytest file with python does not execute its tests; use pytest or python -m pytest. "
                "uv run --no-sync pytest is supported. For standalone assertion checks use execute_python, "
                "which observes actual assertion execution."
            ),
            input_schema=_COMMAND_INPUT_SCHEMA,
        ),
        validate_input=_validate_command_input,
        run=run,
        normalize_output=_normalize_command_output,
        output_schema=_COMMAND_OUTPUT_SCHEMA,
        static_effects=frozenset(
            {
                ToolEffect.READ_WORKSPACE,
                ToolEffect.EXECUTE_PROCESS,
            }
        ),
        resolve_use=lambda arguments: _resolve_command_use(workspace, arguments),
        execution_revision="builtin-run-command-v12-durable-sessions",
        idempotent=False,
        concurrency_safe=False,
        cancellation_mode=CancellationMode.MANAGED_PROCESS,
        interrupt_behavior=InterruptBehavior.CANCEL,
        timeout_seconds=hard_timeout_seconds,
        max_model_output_bytes=150_000,
        stream=stream,
        approval_profile=ToolApprovalProfile.RESTRICTED_READ_ONLY_PROCESS,
        continuation=None if command_sessions is None else command_sessions.continuation,
        after_commit=None if command_sessions is None else command_sessions.after_commit,
    )


def create_execute_python_tool(
    workspace: WorkspaceRuntime,
    *,
    hard_timeout_seconds: float = 605.0,
    termination_grace_seconds: float = 0.5,
) -> Tool:
    if (
        isinstance(termination_grace_seconds, bool)
        or not isinstance(termination_grace_seconds, (int, float))
        or not math.isfinite(termination_grace_seconds)
        or termination_grace_seconds <= 0
    ):
        raise ValueError("termination_grace_seconds must be positive and finite")

    async def run(arguments: Mapping[str, JsonValue]) -> ManagedPythonOutput:
        return await _execute_managed_python(
            workspace,
            ManagedPythonInput.model_validate(arguments),
            termination_grace_seconds=float(termination_grace_seconds),
        )

    return Tool(
        definition=ToolDefinition(
            name="execute_python",
            description=(
                "On Linux, writes use a private candidate workspace with streaming copies and fingerprints. "
                "Only exit 0 with all path, permission and host-change checks passing publishes changes by default; "
                "publish_on_failure=true also publishes validated partial build files on failure or timeout. "
                "Cancellation discards them. Internal relative symlinks are supported; external/absolute links "
                "and protected targets are refused. File mtimes are preserved for incremental builds. "
                "External writes are unsupported. "
                "Linux network access requires libseccomp and blocks host Unix/local IPC; "
                "anonymous stream socketpairs remain available for async event loops. "
                "Run plain Python source with the managed Praxis interpreter in "
                "a restricted OS sandbox. Unlike run_command, this works when the "
                "task workspace has no .venv and does not use shell heredocs. Use "
                "it for spreadsheet transforms, PDF analysis, CSV/JSON processing, "
                "statistics, and charts with the runtime's installed libraries "
                "such as pandas, openpyxl, PyMuPDF, SciPy, and matplotlib. The "
                "workspace is read-only by default. For generated files, set "
                "workspace_write=true and list every intended output in "
                "output_paths. Use inspect_data_file to inspect generated artifacts "
                "and check the requested results. A valid file structure does not "
                "verify calculation or transformation correctness. Do not invoke a shell or create a heredoc "
                "from the Python code merely to repeat inspection. For behavioral verification, execute real "
                "assertions that check the requested result. Output records only assertions actually reached; "
                "uncalled functions and skipped branches are not checks. Assertion counts do not establish "
                "which files or requirements were verified. For delivery, inspect changed artifacts or run "
                "the project's relevant checks after the write."
            ),
            input_schema=_PYTHON_INPUT_SCHEMA,
        ),
        validate_input=_validate_python_input,
        run=run,
        normalize_output=_normalize_managed_python_output,
        output_schema=_PYTHON_OUTPUT_SCHEMA,
        static_effects=frozenset({ToolEffect.READ_WORKSPACE, ToolEffect.EXECUTE_PROCESS}),
        resolve_use=lambda arguments: _resolve_managed_python_use(workspace, arguments),
        execution_revision="builtin-execute-python-v8-declared-targets",
        idempotent=False,
        concurrency_safe=False,
        cancellation_mode=CancellationMode.MANAGED_PROCESS,
        interrupt_behavior=InterruptBehavior.CANCEL,
        timeout_seconds=hard_timeout_seconds,
        max_model_output_bytes=150_000,
        approval_profile=ToolApprovalProfile.RESTRICTED_READ_ONLY_PROCESS,
    )


async def _run_command(
    workspace: WorkspaceRuntime,
    request: RunCommandInput,
    *,
    termination_grace_seconds: float,
    progress_sink: ToolProgressSink | None = None,
    capture_output: Callable[[ToolProgressKind, bytes], None] | None = None,
    authorize_publish: Callable[[], Awaitable[None]] | None = None,
) -> RunCommandOutput:
    cwd = workspace.ensure_within_workspace(
        workspace.resolve_path(request.working_dir or "."),
    )
    if not cwd.is_dir():
        raise NotADirectoryError(f"workspace working directory not found: {request.working_dir}")

    started_at = time.monotonic()
    permissions = workspace.permissions
    sandbox_issue = (
        _permission_sandbox_issue(permissions, network=request.network) or _sandbox_availability_issue(request)
    )
    if sandbox_issue is not None:
        return RunCommandOutput(
            stdout="",
            stderr=sandbox_issue[1],
            exit_code=-1,
            timed_out=False,
            truncated=False,
            duration_ms=(time.monotonic() - started_at) * 1000,
            sandbox_error=sandbox_issue[0],
        )

    preflight_issue = _workspace_sandbox_preflight_issue(
        workspace.root,
        allow_workspace_write=request.workspace_write,
    )
    if preflight_issue is not None:
        error_code, error_message = preflight_issue
        return RunCommandOutput(
            stdout="",
            stderr=error_message,
            exit_code=-1,
            timed_out=False,
            truncated=False,
            duration_ms=(time.monotonic() - started_at) * 1000,
            sandbox_error=error_code,
        )

    with tempfile.TemporaryDirectory(
        prefix="run-command-",
        dir=workspace.scratch,
    ) as temporary_root, ExitStack() as resources:
        environment = _command_environment(
            workspace_root=workspace.root,
            temporary_root=temporary_root,
        )
        view = None
        initial: _LinuxSnapshot = {}
        protected: set[str] = set()
        network_filter = None
        try:
            if _SANDBOX_PLATFORM == "linux":
                if request.network:
                    network_filter = resources.enter_context(_linux_network_filter_file())
                if request.workspace_write:
                    view = Path(temporary_root) / "workspace-view"
                    initial, protected = await _blocking_phase(_prepare_linux_workspace_view, workspace, view)
            sandbox_argv = _sandbox_argv(
                workspace_root=workspace.root,
                temporary_root=temporary_root,
                allow_network=request.network,
                allow_workspace_write=request.workspace_write,
                permissions=permissions,
                network_filter_fd=None if network_filter is None else network_filter.fileno(),
                workspace_view=view,
                protected_view_paths=sorted(protected),
            )
        except (OSError, ValueError) as error:
            return RunCommandOutput(stdout="", stderr=str(error), exit_code=-1, timed_out=False,
                                    truncated=False, duration_ms=(time.monotonic() - started_at) * 1000,
                                    sandbox_error="sandbox_policy_unsupported")
        process = await asyncio.create_subprocess_exec(
            *sandbox_argv,
            "/bin/sh",
            "-c",
            request.command,
            cwd=str(cwd),
            env=environment,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            start_new_session=True,
            pass_fds=() if network_filter is None else (network_filter.fileno(),),
        )
        if process.stdout is None or process.stderr is None:
            raise RuntimeError("command process pipes were not created")
        progress_budget = _CommandProgressBudget()
        stdout_task = asyncio.create_task(
            _read_command_stream(
                process.stdout,
                kind=ToolProgressKind.STDOUT,
                progress_sink=progress_sink,
                progress_budget=progress_budget,
                capture_output=capture_output,
            )
        )
        stderr_task = asyncio.create_task(
            _read_command_stream(
                process.stderr,
                kind=ToolProgressKind.STDERR,
                progress_sink=progress_sink,
                progress_budget=progress_budget,
                capture_output=capture_output,
            )
        )
        wait_task = asyncio.create_task(process.wait())
        communication = asyncio.gather(stdout_task, stderr_task, wait_task)
        timed_out = False
        try:
            done, _pending = await asyncio.wait(
                {communication},
                timeout=request.timeout_seconds,
            )
            if communication not in done:
                timed_out = True
                await _terminate_process_group(
                    process,
                    grace_seconds=termination_grace_seconds,
                )
            stdout_bytes, stderr_bytes, _returncode = await communication
            # A shell can exit while a background child has closed its pipes.
            # Finishing the command also ends the rest of its process group.
            if _SANDBOX_PLATFORM == "darwin" and not timed_out:
                await _terminate_process_group(process, grace_seconds=termination_grace_seconds)
        except asyncio.CancelledError:
            await _terminate_process_group(
                process,
                grace_seconds=termination_grace_seconds,
            )
            try:
                await communication
            except (asyncio.CancelledError, Exception):
                pass
            raise
        except BaseException:
            await _terminate_process_group(
                process,
                grace_seconds=termination_grace_seconds,
            )
            communication.cancel()
            await asyncio.gather(communication, return_exceptions=True)
            raise

        publish_error = None
        workspace_published = False if view is not None else None
        startup_failed = _SANDBOX_PLATFORM == "linux" and process.returncode != 0 and stderr_bytes.startswith(b"bwrap:")
        if (view is not None and not startup_failed
                and (request.publish_on_failure or process.returncode == 0 and not timed_out)):
            try:
                if authorize_publish is not None:
                    await authorize_publish()
                await _blocking_phase(_commit_linux_workspace_view, workspace, view, initial, protected, permissions)
                workspace_published = True
            except (OSError, ValueError) as error:
                publish_error = str(error)
        stdout, stdout_truncated = _bounded_stream(stdout_bytes)
        stderr, stderr_truncated = _bounded_stream(stderr_bytes)
        if publish_error is not None:
            stderr += ("\n" if stderr else "") + publish_error
        if timed_out and not stderr:
            stderr = "command timed out and its process group was terminated"
        output = RunCommandOutput(
            stdout=stdout,
            stderr=stderr,
            exit_code=(process.returncode if process.returncode is not None and publish_error is None else -1),
            timed_out=timed_out,
            truncated=stdout_truncated or stderr_truncated,
            duration_ms=(time.monotonic() - started_at) * 1000,
            network_enabled=request.network,
            workspace_published=workspace_published,
            sandbox_error=(
                "sandbox_publish_denied" if publish_error is not None else "sandbox_start_failed"
                if _SANDBOX_PLATFORM == "linux" and process.returncode != 0 and stderr.startswith("bwrap:")
                else None
            ),
        )
    return output


async def _execute_managed_python(
    workspace: WorkspaceRuntime,
    request: ManagedPythonInput,
    *,
    termination_grace_seconds: float,
) -> ManagedPythonOutput:
    cwd = workspace.ensure_within_workspace(workspace.resolve_path(request.working_dir or "."))
    if not cwd.is_dir():
        raise NotADirectoryError(f"workspace working directory not found: {request.working_dir}")

    started_at = time.monotonic()
    permissions = workspace.permissions
    sandbox_issue = (
        _permission_sandbox_issue(permissions, network=request.network) or _sandbox_availability_issue(request)
    )
    if sandbox_issue is not None:
        return _python_sandbox_error(
            started_at=started_at,
            message=sandbox_issue[1],
            code=sandbox_issue[0],
        )

    preflight_issue = _workspace_sandbox_preflight_issue(
        workspace.root,
        allow_workspace_write=request.workspace_write,
    )
    if preflight_issue is not None:
        error_code, error_message = preflight_issue
        return _python_sandbox_error(
            started_at=started_at,
            message=error_message,
            code=error_code,
        )

    with tempfile.TemporaryDirectory(
        prefix="run-python-",
        dir=workspace.scratch,
    ) as temporary_root, ExitStack() as resources:
        script_path = Path(temporary_root) / "praxis_task.py"
        script_path.write_text(request.code, encoding="utf-8")
        runner_path = Path(temporary_root) / "praxis_python_runner.py"
        report_path = Path(temporary_root) / "verification.json"
        shutil.copyfile(Path(__file__).parents[1] / "python_verification.py", runner_path)
        environment = _command_environment(
            workspace_root=workspace.root,
            temporary_root=temporary_root,
        )
        environment["PYTHONNOUSERSITE"] = "1"
        view = None
        initial: _LinuxSnapshot = {}
        protected: set[str] = set()
        network_filter = None
        try:
            if _SANDBOX_PLATFORM == "linux":
                if request.network:
                    network_filter = resources.enter_context(_linux_network_filter_file())
                if request.workspace_write:
                    view = Path(temporary_root) / "workspace-view"
                    initial, protected = await _blocking_phase(_prepare_linux_workspace_view, workspace, view)
            sandbox_argv = _sandbox_argv(
                workspace_root=workspace.root,
                temporary_root=temporary_root,
                allow_network=request.network,
                allow_workspace_write=request.workspace_write,
                additional_read_roots=_runtime_python_read_roots(),
                permissions=permissions,
                network_filter_fd=None if network_filter is None else network_filter.fileno(),
                workspace_view=view,
                protected_view_paths=sorted(protected),
            )
        except (OSError, ValueError) as error:
            return _python_sandbox_error(started_at=started_at, message=str(error), code="sandbox_policy_unsupported")
        process = await asyncio.create_subprocess_exec(
            *sandbox_argv,
            sys.executable,
            "-I",
            "-B",
            str(runner_path),
            str(script_path),
            str(report_path),
            cwd=str(cwd),
            env=environment,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            start_new_session=True,
            pass_fds=() if network_filter is None else (network_filter.fileno(),),
        )
        communication = asyncio.create_task(process.communicate())
        timed_out = False
        try:
            done, _pending = await asyncio.wait(
                {communication},
                timeout=request.timeout_seconds,
            )
            if communication not in done:
                timed_out = True
                await _terminate_process_group(
                    process,
                    grace_seconds=termination_grace_seconds,
                )
            stdout_bytes, stderr_bytes = await communication
        except asyncio.CancelledError:
            await _terminate_process_group(
                process,
                grace_seconds=termination_grace_seconds,
            )
            try:
                await communication
            except Exception:
                pass
            raise

        publish_error = None
        workspace_published = False if view is not None else None
        startup_failed = _SANDBOX_PLATFORM == "linux" and process.returncode != 0 and stderr_bytes.startswith(b"bwrap:")
        if (view is not None and not startup_failed
                and (request.publish_on_failure or process.returncode == 0 and not timed_out)):
            try:
                _commit_linux_workspace_view(workspace, view, initial, protected, permissions)
                workspace_published = True
            except (OSError, ValueError) as error:
                publish_error = str(error)
        stdout, stdout_truncated = _bounded_stream(stdout_bytes)
        stderr, stderr_truncated = _bounded_stream(stderr_bytes)
        if publish_error is not None:
            stderr += ("\n" if stderr else "") + publish_error
        if timed_out and not stderr:
            stderr = "Python code timed out and its process group was terminated"
        verification = None
        if process.returncode == 0 and not timed_out and report_path.is_file():
            try:
                if report_path.stat().st_size <= 1_000_000:
                    verification = json.loads(report_path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                pass
        return ManagedPythonOutput(
            stdout=stdout,
            stderr=stderr,
            exit_code=(process.returncode if process.returncode is not None and publish_error is None else -1),
            timed_out=timed_out,
            truncated=stdout_truncated or stderr_truncated,
            duration_ms=(time.monotonic() - started_at) * 1000,
            python_version=platform.python_version(),
            network_enabled=request.network,
            workspace_published=workspace_published,
            verification=verification,
            sandbox_error=(
                "sandbox_publish_denied" if publish_error is not None else "sandbox_start_failed"
                if _SANDBOX_PLATFORM == "linux" and process.returncode != 0 and stderr.startswith("bwrap:")
                else None
            ),
        )


def _python_sandbox_error(
    *,
    started_at: float,
    message: str,
    code: str,
) -> ManagedPythonOutput:
    return ManagedPythonOutput(
        stdout="",
        stderr=message,
        exit_code=-1,
        timed_out=False,
        truncated=False,
        duration_ms=(time.monotonic() - started_at) * 1000,
        python_version=platform.python_version(),
        sandbox_error=code,
    )


class _CommandProgressBudget:
    def __init__(self) -> None:
        self._remaining_bytes = _MAX_DELTA_BYTES
        self._remaining_chunks = _MAX_DELTA_CHUNKS
        self._lock = asyncio.Lock()

    async def claim(self, value: bytes) -> bytes:
        async with self._lock:
            if self._remaining_chunks <= 0 or self._remaining_bytes <= 0:
                return b""
            claimed = value[: self._remaining_bytes]
            self._remaining_bytes -= len(claimed)
            self._remaining_chunks -= 1
            return claimed


class _CommandStreamBuffer:
    """Bounded head and rolling tail, including after progress events run out."""

    def __init__(self) -> None:
        self.data = bytearray()

    def append(self, chunk: bytes) -> None:
        limit = _MAX_STREAM_BYTES + 1
        self.data.extend(chunk)
        if len(self.data) > limit:
            head = limit // 2
            del self.data[head:len(self.data) - (limit - head)]

    def value(self) -> bytes:
        return bytes(self.data)


async def _read_command_stream(
    stream: asyncio.StreamReader,
    *,
    kind: ToolProgressKind,
    progress_sink: ToolProgressSink | None,
    progress_budget: _CommandProgressBudget,
    capture_output: Callable[[ToolProgressKind, bytes], None] | None = None,
) -> bytes:
    retained = _CommandStreamBuffer()
    decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
    while True:
        chunk = await stream.read(_DELTA_CHUNK_BYTES)
        if not chunk:
            break
        retained.append(chunk)
        if capture_output is not None:
            capture_output(kind, chunk)
        if progress_sink is None:
            continue
        claimed = await progress_budget.claim(chunk)
        if not claimed:
            continue
        content = decoder.decode(claimed, final=False)
        if content:
            await progress_sink(ToolProgress(kind=kind, content=content))
    if progress_sink is not None:
        final_content = decoder.decode(b"", final=True)
        if final_content:
            await progress_sink(ToolProgress(kind=kind, content=final_content))
    return retained.value()


def _command_environment(
    *,
    workspace_root: os.PathLike[str] | str,
    temporary_root: os.PathLike[str] | str,
) -> dict[str, str]:
    workspace = os.path.realpath(workspace_root)
    temporary = os.path.realpath(temporary_root)
    # Fixed toolchain paths are intentionally not derived from the host PATH.
    path = os.pathsep.join(
        (
            os.path.join(workspace, ".venv", "bin"),
            os.path.join(workspace, "node_modules", ".bin"),
            "/opt/homebrew/bin",
            "/usr/local/bin",
            "/Library/Developer/CommandLineTools/usr/bin",
            "/usr/bin",
            "/bin",
            "/usr/sbin",
            "/sbin",
        )
    )
    return {
        "GIT_CONFIG_GLOBAL": "/dev/null",
        "GIT_CONFIG_NOSYSTEM": "1",
        "HOME": temporary,
        "LANG": "C",
        "LC_ALL": "C",
        "PATH": path,
        "PIP_CACHE_DIR": temporary,
        "PYTHONDONTWRITEBYTECODE": "1",
        "SHELL": "/bin/sh",
        "TEMP": temporary,
        "TMP": temporary,
        "TMPDIR": temporary,
        "UV_CACHE_DIR": temporary,
        "XDG_CACHE_HOME": temporary,
    }


def _sandbox_availability_issue(
    request: RunCommandInput | ManagedPythonInput,
) -> tuple[str, str] | None:
    executable = _BWRAP_PATH if _SANDBOX_PLATFORM == "linux" else _SANDBOX_EXEC_PATH
    if _SANDBOX_PLATFORM not in {"linux", "darwin"} or not (
        os.path.isfile(executable) and os.access(executable, os.X_OK)
    ):
        return (
            "sandbox_unavailable",
            f"restricted sandbox is unavailable ({executable}; "
            "Linux requires bubblewrap); refusing unsandboxed execution",
        )
    return None


def _permission_sandbox_issue(
    permissions: ResolvedPermissions | None, *, network: bool,
) -> tuple[str, str] | None:
    if permissions is not None and network and permissions.command_network_allowed() == "deny":
        return (
            "sandbox_network_denied",
            "Permission policy denies command network access; domain filters require an enforcing proxy. "
            "No command was executed.",
        )
    return None


def _seatbelt_permission_rules(
    permissions: ResolvedPermissions | None, *, temporary_root: os.PathLike[str] | str, allow_write: bool,
) -> tuple[str, str]:
    """Render disjoint subtree regions so a parent deny cannot shadow child grants."""
    if permissions is None:
        return "", ""
    rules = permissions.sandbox_filesystem
    broad_root_read = any(rule.path == "/" and rule.access != "deny" for rule in rules)
    if broad_root_read and allow_write:
        raise ValueError(
            "A broad host-root read grant cannot safely combine with subprocess writes: "
            "Seatbelt resolves host symlink aliases. Use path tools or scoped filesystem grants for edits."
        )
    _validate_permission_aliases(permissions, allow_write=allow_write)
    temporary = _escape_seatbelt_string(os.path.realpath(temporary_root))
    reads = []
    policies = []
    for rule in rules:
        path = Path(rule.path)
        if path.resolve() != path:
            raise ValueError("Seatbelt cannot represent a symlink policy boundary")
        escaped = _escape_seatbelt_string(rule.path)
        children = [child for child in rules if child.path != rule.path and Path(child.path).is_relative_to(path)]
        exclusions = "".join(
            f'    (require-not (subpath "{_escape_seatbelt_string(child.path)}"))\n' for child in children
        )
        region = f'(require-all (subpath "{escaped}")\n{exclusions}'
        region += f'    (require-not (subpath "{temporary}")))'
        if rule.access != "deny":
            reads.append(f"  {region}\n")
            if path != Path("/"):
                policies.append(f'(allow file-read-metadata file-test-existence (path-ancestors "{escaped}"))\n')
        elif path != Path("/"):
            policies.append(f"(deny file-read* {region})\n")
        if allow_write and rule.access == "write":
            policies.append(f"(allow file-write* {region})\n")
        elif path != Path("/"):
            policies.append(f"(deny file-write* {region})\n")
    # Renaming an ancestor would replace a protected configuration namespace
    # without writing its file. Protect directory identity, not sibling content.
    ancestors = {parent for source in permissions.sources if Path(source).is_absolute()
                 for parent in Path(source).parents if parent != Path("/")}
    for parent in sorted(ancestors):
        escaped = _escape_seatbelt_string(str(parent))
        policies.append(f'(deny file-write-unlink (literal "{escaped}"))\n')
    return "".join(reads), "".join(policies)


def _validate_permission_aliases(permissions: ResolvedPermissions, *, allow_write: bool) -> None:
    """Refuse symlink and hardlink aliases that erase a lexical restriction."""
    rules = permissions.sandbox_filesystem
    broad_root_read = any(rule.path == "/" and rule.access != "deny" for rule in rules)
    for rule in rules:
        path = Path(rule.path)
        if rule.access != "deny" and path.is_file() and path.stat().st_nlink > 1:
            raise ValueError(f"sandbox cannot safely expose a hardlinked file grant: {path}")
    roots = {
        Path(rule.path) for rule in rules if Path(rule.path) != Path("/") and (
            (rule.access == "deny" if broad_root_read else rule.access != "deny") or (not broad_root_read and any(
                parent.access != "deny" and Path(rule.path).is_relative_to(parent.path) for parent in rules
            ))
        )
    }
    roots = {root for root in roots if root.is_dir() and not root.is_symlink()}
    roots = {root for root in roots if not any(root != parent and root.is_relative_to(parent) for parent in roots)}
    for root in roots:
        if root != Path(permissions.workspace):
            issue = _workspace_sandbox_preflight_issue(root, allow_workspace_write=allow_write)
            if issue is not None:
                raise ValueError(f"external sandbox permission grant rejected: {issue[1]}")

        def scan_error(error: OSError) -> None:
            raise ValueError(f"cannot validate sandbox permission aliases: {error.filename}") from error

        for directory, directories, files in os.walk(root, followlinks=False, onerror=scan_error):
            for name in (*directories, *files):
                alias = Path(directory) / name
                if not alias.is_symlink():
                    continue
                try:
                    target = alias.resolve()
                except (OSError, RuntimeError) as error:
                    raise ValueError("cannot resolve a sandbox permission alias") from error
                for write in (False, True) if allow_write else (False,):
                    if not permissions.path_access(alias, write=write) and permissions.path_access(target, write=write):
                        raise ValueError(f"sandbox cannot enforce lexical permission restriction on symlink: {alias}")


def _sandbox_argv(
    *,
    workspace_root: os.PathLike[str] | str,
    temporary_root: os.PathLike[str] | str,
    allow_network: bool,
    allow_workspace_write: bool = False,
    additional_read_roots: Sequence[os.PathLike[str] | str] = (),
    permissions: ResolvedPermissions | None = None,
    network_filter_fd: int | None = None,
    workspace_view: Path | None = None,
    protected_view_paths: Sequence[str] = (),
) -> list[str]:
    if _SANDBOX_PLATFORM == "linux":
        return _linux_sandbox_argv(
            workspace_root=workspace_root,
            temporary_root=temporary_root,
            additional_read_roots=additional_read_roots,
            permissions=permissions,
            allow_network=allow_network,
            allow_workspace_write=allow_workspace_write,
            network_filter_fd=network_filter_fd,
            workspace_view=workspace_view,
            protected_view_paths=protected_view_paths,
        )
    return [
        _SANDBOX_EXEC_PATH,
        "-p",
        _build_command_sandbox_profile(
            workspace_root=workspace_root,
            temporary_root=temporary_root,
            allow_network=allow_network,
            allow_workspace_write=allow_workspace_write,
            additional_read_roots=additional_read_roots,
            permissions=permissions,
        ),
    ]


def _linux_sandbox_argv(
    *,
    workspace_root: os.PathLike[str] | str,
    temporary_root: os.PathLike[str] | str,
    additional_read_roots: Sequence[os.PathLike[str] | str],
    permissions: ResolvedPermissions | None = None,
    allow_network: bool = False,
    allow_workspace_write: bool = False,
    network_filter_fd: int | None = None,
    workspace_view: Path | None = None,
    protected_view_paths: Sequence[str] = (),
) -> list[str]:
    """Keep host grants read-only; optional writes target a private candidate.

    Sharing networking requires a seccomp filter which allows only IPv4/IPv6
    sockets, permits only anonymous Unix stream socketpairs for self-pipes,
    denies io_uring and listeners, and rejects other syscall ABIs.
    Without it abstract Unix sockets would expose host IPC through --share-net.
    Denies inside exposed trees and external writes remain fail-closed.
    """
    workspace = os.path.realpath(workspace_root)
    temporary = os.path.realpath(temporary_root)
    argv = [
        _BWRAP_PATH,
        "--unshare-all",
        "--unshare-user",
        "--die-with-parent",
        "--new-session",
        "--disable-userns",
        "--cap-drop",
        "ALL",
    ]
    if allow_network:
        if network_filter_fd is None:
            raise ValueError("Linux networking requires a seccomp IPC filter")
        argv.extend(("--share-net", "--seccomp", str(network_filter_fd)))
    for name in ("usr", "bin", "sbin", "lib", "lib64"):
        path = Path("/") / name
        if path.is_symlink():
            argv.extend(("--symlink", os.readlink(path), str(path)))
        elif path.is_dir():
            argv.extend(("--ro-bind", str(path), str(path)))
    for name in ("/etc/ld.so.cache", "/etc/alternatives", "/etc/ssl/certs"):
        if Path(name).exists():
            argv.extend(("--ro-bind", name, name))
    if allow_network:
        for name in ("/etc/resolv.conf", "/etc/hosts", "/etc/nsswitch.conf"):
            if Path(name).is_file():
                argv.extend(("--ro-bind", name, name))
    argv.extend(("--proc", "/proc", "--dev", "/dev", "--tmpfs", "/tmp"))
    roots = set(_trusted_toolchain_roots(workspace_root))
    roots.update(Path(root).resolve() for root in additional_read_roots)
    for root in sorted(roots, key=str):
        if root == Path("/"):
            raise ValueError("the host root cannot be a sandbox runtime read root")
        if root.exists() and not root.is_relative_to("/usr"):
            argv.extend(("--ro-bind", str(root), str(root)))
    argv.extend(
        (
            "--ro-bind",
            workspace,
            workspace,
            "--bind",
            temporary,
            temporary,
        )
    )
    if permissions is not None:
        _validate_permission_aliases(permissions, allow_write=allow_workspace_write)
        permission_rules = permissions.sandbox_filesystem
        for rule in permission_rules:
            path = Path(rule.path)
            if allow_workspace_write and rule.access == "write" and not path.is_relative_to(workspace):
                raise ValueError("Linux subprocess writes outside the workspace are unsupported")
            if path == Path("/"):
                if rule.access != "deny":
                    raise ValueError("Linux cannot represent a host root read grant")
                continue
            if path.resolve() != path:
                raise ValueError("Linux cannot represent a symlink policy boundary")
            if not path.exists() or path.is_relative_to(temporary):
                continue
            if rule.access == "deny":
                if path == command_state_directory():
                    continue  # Explicit final namespace mask below all grants.
                # A fresh namespace has nothing at unexposed outside paths.
                exposed = path.is_relative_to(workspace) or any(path.is_relative_to(root) for root in roots)
                exposed = exposed or any(
                    parent.access != "deny" and path.is_relative_to(parent.path)
                    for parent in permission_rules
                )
                exposed = exposed or any(path.is_relative_to(root) for root in ("/usr", "/bin", "/sbin", "/lib"))
                if exposed:
                    raise ValueError("Linux cannot enforce a filesystem deny inside an exposed subtree")
            else:
                argv.extend(("--ro-bind", str(path), str(path)))
        # Restored child grants must not restore a writable temporary root's parent.
        argv.extend(("--bind", temporary, temporary))
    if workspace_view is not None:
        argv.extend(("--bind", str(workspace_view), workspace))
        for relative in protected_view_paths:
            protected_path = str(Path(workspace) / relative)
            argv.extend(("--ro-bind", protected_path, protected_path))
        # The real command scratch must remain visible after overlaying workspace.
        argv.extend(("--bind", temporary, temporary))
    # The private supervisor journal must stay hidden even under a broad grant.
    private = command_state_directory()
    if private.exists():
        argv.extend(("--tmpfs", str(private), "--remount-ro", str(private)))
    return argv


def _linux_network_filter_file() -> BinaryIO:
    """Export a native-ABI libseccomp filter for bwrap; never install it here."""
    class Comparison(ctypes.Structure):
        _fields_ = [("arg", ctypes.c_uint), ("op", ctypes.c_int),
                    ("datum_a", ctypes.c_uint64), ("datum_b", ctypes.c_uint64)]

    try:
        library = ctypes.CDLL("libseccomp.so.2")
    except OSError as error:
        raise ValueError("Linux network isolation requires libseccomp.so.2") from error
    library.seccomp_init.argtypes = [ctypes.c_uint32]
    library.seccomp_init.restype = ctypes.c_void_p
    library.seccomp_release.argtypes = [ctypes.c_void_p]
    library.seccomp_syscall_resolve_name.argtypes = [ctypes.c_char_p]
    library.seccomp_syscall_resolve_name.restype = ctypes.c_int
    library.seccomp_rule_add_array.argtypes = [ctypes.c_void_p, ctypes.c_uint32, ctypes.c_int,
                                              ctypes.c_uint, ctypes.POINTER(Comparison)]
    library.seccomp_export_bpf.argtypes = [ctypes.c_void_p, ctypes.c_int]
    context = library.seccomp_init(0x7FFF0000)  # SCMP_ACT_ALLOW; bad architectures are killed.
    if not context:
        raise ValueError("cannot allocate Linux seccomp IPC filter")
    output = tempfile.TemporaryFile()
    try:
        def deny(name: bytes, comparison: Comparison | None = None) -> None:
            number = library.seccomp_syscall_resolve_name(name)
            if number == -1:
                raise ValueError(f"libseccomp cannot resolve required syscall {name.decode()}")
            result = library.seccomp_rule_add_array(
                context, 0x00050000 | errno.EPERM, number, 0 if comparison is None else 1,
                None if comparison is None else ctypes.pointer(comparison))
            if result != 0:
                raise ValueError(f"cannot build Linux seccomp IPC rule for {name.decode()}: {result}")

        # Socket domains 2 and 10 are AF_INET and AF_INET6. Reject every other
        # domain, including future ones, AF_UNIX, AF_VSOCK, AF_NETLINK and PACKET.
        deny(b"socket", Comparison(0, 2, 2, 0))  # SCMP_CMP_LT
        deny(b"socket", Comparison(0, 6, 10, 0))  # SCMP_CMP_GT
        for domain in range(3, 10):
            deny(b"socket", Comparison(0, 4, domain, 0))  # SCMP_CMP_EQ
        # asyncio uses an anonymous, already-connected AF_UNIX stream pair.
        # It cannot reconnect to host IPC; bind/listen remain denied. Datagram
        # pairs can change their peer and must remain blocked. Permit exactly
        # SOCK_STREAM with only the two known descriptor flags, protocol zero.
        deny(b"socketpair", Comparison(0, 1, int(socket.AF_UNIX), 0))  # SCMP_CMP_NE
        deny(b"socketpair", Comparison(2, 1, 0, 0))
        allowed_type_bits = (int(socket.SOCK_STREAM) | int(getattr(socket, "SOCK_NONBLOCK", 0))
                             | int(getattr(socket, "SOCK_CLOEXEC", 0)))
        deny(b"socketpair", Comparison(1, 7, int(socket.SOCK_STREAM), 0))  # SCMP_CMP_MASKED_EQ
        for bit in range(64):
            mask = 1 << bit
            if not allowed_type_bits & mask:
                deny(b"socketpair", Comparison(1, 7, mask, mask))
        for name in (b"bind", b"listen", b"accept", b"accept4"):
            deny(name)
        # io_uring socket operations do not pass through socket() seccomp rules.
        deny(b"io_uring_setup")
        if library.seccomp_export_bpf(context, output.fileno()) != 0:
            raise ValueError("cannot export Linux seccomp IPC filter")
        output.seek(0)
        return output
    except BaseException:
        output.close()
        raise
    finally:
        library.seccomp_release(context)


async def _blocking_phase[T](function: Callable[..., T], *arguments: object) -> T:
    """Keep cancellation schedulable; never remove scratch while a thread uses it."""
    cancelled = threading.Event()

    def invoke() -> T:
        token = _COPY_CANCEL.set(cancelled)
        try:
            return function(*arguments)
        finally:
            _COPY_CANCEL.reset(token)

    task = asyncio.create_task(asyncio.to_thread(invoke))
    try:
        return await asyncio.shield(task)
    except asyncio.CancelledError:
        cancelled.set()
        await asyncio.gather(task, return_exceptions=True)
        raise


def _check_copy_cancelled() -> None:
    cancelled = _COPY_CANCEL.get()
    if cancelled is not None and cancelled.is_set():
        raise OSError("Linux workspace copy cancelled")


def _linux_view_snapshot(root: Path, protected: set[str]) -> _LinuxSnapshot:
    snapshot: _LinuxSnapshot = {}

    def walk(descriptor: int, prefix: Path) -> None:
        with os.scandir(descriptor) as entries:
            names = sorted(entry.name for entry in entries)
        for name in names:
            _check_copy_cancelled()
            relative = (prefix / name).as_posix()
            if relative in protected:
                continue
            metadata = os.stat(name, dir_fd=descriptor, follow_symlinks=False)
            mode = metadata.st_mode
            identity = (metadata.st_dev, metadata.st_ino)
            if stat.S_ISDIR(mode):
                child = os.open(name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=descriptor)
                try:
                    opened = os.fstat(child)
                    if (opened.st_dev, opened.st_ino) != identity:
                        raise ValueError("workspace directory changed during Linux snapshot")
                    snapshot[relative] = (mode, "", identity, 0)
                    walk(child, prefix / name)
                finally:
                    os.close(child)
            elif stat.S_ISREG(mode):
                child = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=descriptor)
                with os.fdopen(child, "rb") as stream:
                    opened = os.fstat(stream.fileno())
                    if not stat.S_ISREG(opened.st_mode) or (opened.st_dev, opened.st_ino) != identity:
                        raise ValueError("workspace file changed during Linux snapshot")
                    if opened.st_nlink != 1:
                        raise ValueError("Linux candidate workspace refuses hardlinked files")
                    data = _linux_file_digest(stream)
                snapshot[relative] = (mode, data, identity, opened.st_mtime_ns)
            elif stat.S_ISLNK(mode):
                snapshot[relative] = (mode, os.readlink(name, dir_fd=descriptor), identity, 0)
            else:
                raise ValueError("Linux candidate workspace refuses nonregular files")
    descriptor = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        walk(descriptor, Path())
    finally:
        os.close(descriptor)
    return snapshot


def _linux_file_digest(stream: BinaryIO, output: BinaryIO | None = None) -> bytes:
    """Hash/copy a regular file with bounded memory and a stable open identity."""
    before = os.fstat(stream.fileno())
    if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
        raise ValueError("Linux candidate workspace refuses hardlinked or nonregular files")
    digest = hashlib.sha256()
    while chunk := stream.read(_LINUX_COPY_CHUNK_BYTES):
        _check_copy_cancelled()
        digest.update(chunk)
        if output is not None:
            output.write(chunk)
    after = os.fstat(stream.fileno())
    if any(getattr(before, field) != getattr(after, field) for field in (
        "st_dev", "st_ino", "st_mode", "st_size", "st_mtime_ns", "st_ctime_ns", "st_nlink",
    )):
        raise ValueError("workspace file changed during Linux snapshot or copy")
    return digest.digest()


def _copy_linux_file(root: Path, name: str, expected: tuple[int, bytes | str, tuple[int, int], int],
                     output: BinaryIO) -> None:
    """Do not follow source ancestors or replace a verified file with another inode."""
    with ExitStack() as sources:
        descriptor = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        sources.callback(os.close, descriptor)
        for component in Path(name).parts[:-1]:
            descriptor = os.open(component, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=descriptor)
            sources.callback(os.close, descriptor)
        child = os.open(Path(name).name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=descriptor)
        with os.fdopen(child, "rb") as stream:
            opened = os.fstat(stream.fileno())
            if ((opened.st_dev, opened.st_ino) != expected[2] or opened.st_mode != expected[0]
                    or opened.st_mtime_ns != expected[3] or _linux_file_digest(stream, output) != expected[1]):
                raise ValueError("workspace source changed before Linux copy")


def _prepare_linux_workspace_view(workspace: WorkspaceRuntime, view: Path) -> tuple[_LinuxSnapshot, set[str]]:
    """Copy ordinary files; metadata, dependencies and agent state stay host RO."""
    protected: set[str] = set()
    for directory, directories, files in os.walk(workspace.root, followlinks=False):
        for name in (*directories, *files):
            path = Path(directory) / name
            if name.casefold() in _LINUX_VIEW_PROTECTED or path == workspace.scratch:
                protected.add(path.relative_to(workspace.root).as_posix())
                if name in directories:
                    directories.remove(name)
    initial = _linux_view_snapshot(workspace.root, protected)
    view.mkdir()
    for relative, (mode, content, _identity, mtime_ns) in initial.items():
        target = view / relative
        if stat.S_ISDIR(mode):
            target.mkdir(parents=True, exist_ok=True)
        elif stat.S_ISLNK(mode):
            target.symlink_to(str(content))
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("xb") as output:
                _copy_linux_file(workspace.root, relative, initial[relative], output)
            target.chmod(stat.S_IMODE(mode))
            os.utime(target, ns=(mtime_ns, mtime_ns))
    for relative in protected:
        host = workspace.root / relative
        target = view / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if host.is_dir():
            target.mkdir(exist_ok=True)
        else:
            target.touch()
    for relative, (mode, _content, _identity, _mtime_ns) in reversed(tuple(initial.items())):
        if stat.S_ISDIR(mode):
            (view / relative).chmod(stat.S_IMODE(mode))
    return initial, protected


def _validate_linux_symlink(
    name: str, snapshot: _LinuxSnapshot, protected: set[str], workspace: WorkspaceRuntime,
    permissions: ResolvedPermissions | None,
) -> None:
    """Resolve the candidate graph, never the untrusted host/view pathname.

    Relative dangling links inside the workspace are useful to build systems.
    Absolute links, escapes, cycles and protected targets fail before mutation.
    """
    pending = list(reversed(Path(name).parts))
    resolved: list[str] = []
    followed = 0
    while pending:
        component = pending.pop()
        if component == "..":
            if not resolved:
                raise ValueError(f"Linux candidate symlink escapes workspace: {name}")
            resolved.pop()
            continue
        if component == ".":
            continue
        resolved.append(component)
        relative = Path(*resolved).as_posix()
        if component.casefold() in _LINUX_VIEW_PROTECTED or relative in protected:
            raise ValueError(f"Linux candidate symlink targets a protected path: {name}")
        entry = snapshot.get(relative)
        if entry is not None and stat.S_ISLNK(entry[0]):
            content = str(entry[1])
            followed += 1
            if Path(content).is_absolute() or followed > 40:
                raise ValueError(f"Linux candidate symlink is absolute or cyclic: {name}")
            resolved.pop()
            pending.extend(reversed(Path(content).parts))
    target = workspace.root.joinpath(*resolved)
    if permissions is not None and not permissions.path_access(target):
        raise ValueError(f"Linux candidate symlink target denied by permission policy: {name}")


def _commit_linux_workspace_view(
    workspace: WorkspaceRuntime, view: Path, initial: _LinuxSnapshot, protected: set[str],
    permissions: ResolvedPermissions | None,
) -> None:
    """Validate all changes before publishing any host mutation.

    Each replacement is atomic. A multi-file commit is not a filesystem
    transaction; concurrent host writers must not modify the workspace.
    """
    if view.is_symlink() or not view.is_dir():
        raise ValueError("Linux candidate workspace root is invalid")
    resulting = _linux_view_snapshot(view, protected)
    changes = {name for name in initial.keys() | resulting.keys()
               if (initial.get(name, ())[:2], initial.get(name, ())[3:])
               != (resulting.get(name, ())[:2], resulting.get(name, ())[3:])}
    current_permissions = workspace.permissions
    effective = (permissions.intersection(current_permissions) if permissions is not None
                 and current_permissions is not None else permissions or current_permissions)
    for name in sorted(changes):
        relative = Path(name)
        if any(part.casefold() in _LINUX_VIEW_PROTECTED for part in relative.parts):
            raise ValueError(f"Linux candidate changed a protected path: {name}")
        if any(Path(path).is_relative_to(relative) for path in protected):
            raise ValueError(f"Linux candidate changed a protected directory ancestor: {name}")
        host = workspace.root / relative
        if effective is not None and not effective.path_access(host, write=True):
            raise ValueError(f"Linux candidate write denied by permission policy: {name}")
        workspace.ensure_writable(host)
        for parent in host.parents:
            if parent == workspace.root:
                break
            if parent.is_symlink():
                raise ValueError(f"Linux candidate cannot publish through a symlink: {name}")
        after = resulting.get(name)
        before = initial.get(name)
        if after is not None and stat.S_ISLNK(after[0]):
            _validate_linux_symlink(name, resulting, protected, workspace, effective)
        if before is not None and after is not None and stat.S_IFMT(before[0]) != stat.S_IFMT(after[0]):
            raise ValueError(f"Linux candidate cannot change path type: {name}")
    if _linux_view_snapshot(workspace.root, protected) != initial:
        raise ValueError("workspace changed while Linux command was running; inspect and retry")

    def recheck(name: str, parent_fd: int) -> None:
        target = workspace.root / name
        current = workspace.permissions
        if ((permissions is not None and not permissions.path_access(target, write=True))
                or (current is not None and not current.path_access(target, write=True))):
            raise ValueError(f"Linux publish permission changed before commit: {name}")
        workspace.ensure_writable(target)
        after = resulting.get(name)
        if after is not None and stat.S_ISLNK(after[0]):
            latest = permissions.intersection(current) if permissions is not None and current is not None else (
                permissions or current
            )
            _validate_linux_symlink(name, resulting, protected, workspace, latest)
        before = initial.get(name)
        try:
            metadata = os.stat(target.name, dir_fd=parent_fd, follow_symlinks=False)
        except FileNotFoundError:
            metadata = None
        if before is None:
            if metadata is not None:
                raise ValueError(f"workspace target changed before commit: {name}")
            return
        if (metadata is None or metadata.st_mode != before[0]
                or (metadata.st_dev, metadata.st_ino) != before[2]):
            raise ValueError(f"workspace target identity changed before commit: {name}")
        if stat.S_ISREG(metadata.st_mode):
            descriptor = os.open(target.name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=parent_fd)
            with os.fdopen(descriptor, "rb") as stream:
                opened = os.fstat(stream.fileno())
                if ((opened.st_dev, opened.st_ino) != before[2] or opened.st_nlink != 1
                        or opened.st_mtime_ns != before[3]):
                    raise ValueError(f"workspace target identity changed before commit: {name}")
                assert isinstance(before[1], bytes)
                if _linux_file_digest(stream) != before[1]:
                    raise ValueError(f"workspace target content changed before commit: {name}")
        elif stat.S_ISLNK(metadata.st_mode):
            if os.readlink(target.name, dir_fd=parent_fd) != before[1]:
                raise ValueError(f"workspace symlink changed before commit: {name}")

    # Every host mutation is relative to pinned directory descriptors. Path
    # validation alone cannot prevent an ancestor becoming a symlink between
    # validation and replace/mkdir/unlink. O_NOFOLLOW applies at every component.
    with ExitStack() as publishing:
        root_fd = os.open(workspace.root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        publishing.callback(os.close, root_fd)
        created_directories: dict[str, tuple[int, int]] = {}

        def open_parent(name: str, resources: ExitStack) -> int:
            descriptor = root_fd
            parts = Path(name).parts[:-1]
            for index, component in enumerate(parts):
                descriptor = os.open(component, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                                     dir_fd=descriptor)
                resources.callback(os.close, descriptor)
                metadata = os.fstat(descriptor)
                relative = Path(*parts[:index + 1]).as_posix()
                expected = initial[relative][2] if relative in initial else created_directories.get(relative)
                if expected is None or (metadata.st_dev, metadata.st_ino) != expected:
                    raise ValueError(f"workspace parent identity changed before commit: {name}")
            return descriptor

        # Validate first, remove leaves before parents, create parents before leaves.
        for name in sorted(changes, key=lambda value: (-len(Path(value).parts), value)):
            if name in resulting:
                continue
            with ExitStack() as parents:
                parent_fd = open_parent(name, parents)
                recheck(name, parent_fd)
                if stat.S_ISDIR(initial[name][0]):
                    os.rmdir(Path(name).name, dir_fd=parent_fd)
                else:
                    os.unlink(Path(name).name, dir_fd=parent_fd)
        for name in sorted(changes, key=lambda value: (len(Path(value).parts), value)):
            after = resulting.get(name)
            if after is None:
                continue
            with ExitStack() as parents:
                parent_fd = open_parent(name, parents)
                recheck(name, parent_fd)
                mode, content, _identity, mtime_ns = after
                leaf = Path(name).name
                if stat.S_ISDIR(mode):
                    if name not in initial:
                        os.mkdir(leaf, mode=0o700, dir_fd=parent_fd)
                    descriptor = os.open(leaf, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=parent_fd)
                    parents.callback(os.close, descriptor)
                    metadata = os.fstat(descriptor)
                    if name in initial and (metadata.st_dev, metadata.st_ino) != initial[name][2]:
                        raise ValueError(f"workspace directory identity changed before chmod: {name}")
                    os.fchmod(descriptor, stat.S_IMODE(mode) & 0o777)
                    created_directories[name] = (metadata.st_dev, metadata.st_ino)
                else:
                    temporary = ".praxis-publish-" + secrets.token_hex(12)
                    try:
                        if stat.S_ISLNK(mode):
                            assert isinstance(content, str)
                            os.symlink(content, temporary, dir_fd=parent_fd)
                        else:
                            assert isinstance(content, bytes)
                            descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                                                 mode=0o600, dir_fd=parent_fd)
                            with os.fdopen(descriptor, "wb") as output:
                                _copy_linux_file(view, name, after, output)
                                output.flush()
                                os.fchmod(output.fileno(), stat.S_IMODE(mode) & 0o777)
                                os.utime(output.fileno(), ns=(mtime_ns, mtime_ns))
                        recheck(name, parent_fd)
                        if name in initial:
                            os.replace(temporary, leaf, src_dir_fd=parent_fd, dst_dir_fd=parent_fd)
                        else:
                            os.link(temporary, leaf, src_dir_fd=parent_fd, dst_dir_fd=parent_fd,
                                    follow_symlinks=False)
                    finally:
                        try:
                            os.unlink(temporary, dir_fd=parent_fd)
                        except FileNotFoundError:
                            pass


def _build_command_sandbox_profile(
    *,
    workspace_root: os.PathLike[str] | str,
    temporary_root: os.PathLike[str] | str,
    allow_network: bool,
    allow_workspace_write: bool = False,
    additional_read_roots: Sequence[os.PathLike[str] | str] = (),
    permissions: ResolvedPermissions | None = None,
) -> str:
    workspace = _escape_seatbelt_string(str(os.path.realpath(workspace_root)))
    temporary = _escape_seatbelt_string(str(os.path.realpath(temporary_root)))
    trusted_roots = {path.resolve() for path in _trusted_toolchain_roots(workspace_root)}
    trusted_roots.update(
        Path(path).expanduser().resolve() for path in additional_read_roots if Path(path).expanduser().exists()
    )
    trusted_toolchains = tuple(_escape_seatbelt_string(str(path)) for path in sorted(trusted_roots, key=str))
    trusted_toolchain_reads = "".join(f'  (subpath "{path}")\n' for path in trusted_toolchains)
    trusted_toolchain_ancestors = "".join(f'  (path-ancestors "{path}")\n' for path in trusted_toolchains)
    policy_reads, policy_rules = _seatbelt_permission_rules(
        permissions, temporary_root=temporary_root, allow_write=allow_workspace_write,
    )
    workspace_reads = f'  (subpath "{workspace}")\n' if permissions is None else policy_reads
    workspace_write_policy = f'  (subpath "{workspace}")\n' if allow_workspace_write and permissions is None else ""
    network_policy = (
        # DNS is the only approved AF_UNIX endpoint; local service and Docker
        # sockets remain denied even when IP networking is approved.
        '(allow network-outbound (remote ip "*:*"))\n'
        "(allow system-socket\n"
        "  (require-all\n"
        "    (socket-domain AF_SYSTEM)\n"
        "    (socket-protocol 2)))\n"
        "(allow system-socket (socket-domain AF_UNIX))\n"
        "(allow network-outbound\n"
        "  (remote unix-socket\n"
        '    (literal "/private/var/run/mDNSResponder"))\n'
        "  (remote unix-socket\n"
        '    (literal "/var/run/mDNSResponder")))\n'
        "(allow mach-lookup\n"
        '  (global-name "com.apple.bsd.dirhelper")\n'
        '  (global-name "com.apple.dnssd.service")\n'
        '  (global-name "com.apple.system.opendirectoryd.membership")\n'
        '  (global-name "com.apple.networkd")\n'
        '  (global-name "com.apple.ocspd")\n'
        '  (global-name "com.apple.trustd.agent")\n'
        '  (global-name "com.apple.SystemConfiguration.DNSConfiguration")\n'
        '  (global-name "com.apple.SystemConfiguration.configd"))\n'
        '(allow sysctl-read (sysctl-name-regex #"^net.routetable"))'
        if allow_network
        else ""
    )
    return (
        "(version 1)\n"
        "(deny default)\n"
        "(allow process-exec)\n"
        "(allow process-fork)\n"
        "(allow signal (target same-sandbox))\n"
        "(allow process-info* (target same-sandbox))\n"
        "(allow sysctl-read)\n"
        "(allow mach-lookup\n"
        '  (global-name "com.apple.system.opendirectoryd.libinfo"))\n'
        "(allow file-read* file-test-existence\n"
        f"{workspace_reads}"
        f'  (subpath "{temporary}")\n'
        f"{trusted_toolchain_reads}"
        '  (literal "/")\n'
        '  (subpath "/bin")\n'
        '  (subpath "/sbin")\n'
        '  (subpath "/usr/bin")\n'
        '  (subpath "/usr/sbin")\n'
        '  (subpath "/usr/lib")\n'
        '  (subpath "/usr/libexec")\n'
        '  (subpath "/usr/share")\n'
        '  (subpath "/System/Library")\n'
        '  (subpath "/Library/Apple")\n'
        '  (subpath "/Library/Developer/CommandLineTools")\n'
        '  (subpath "/opt/homebrew")\n'
        '  (subpath "/usr/local")\n'
        '  (literal "/private/var/select/sh")\n'
        '  (literal "/private/etc/hosts")\n'
        '  (literal "/private/etc/protocols")\n'
        '  (literal "/private/etc/resolv.conf")\n'
        '  (literal "/private/etc/services")\n'
        '  (literal "/private/etc/ssl/cert.pem")\n'
        '  (literal "/private/etc/ssl/openssl.cnf")\n'
        '  (literal "/Library/Preferences/com.apple.networkd.plist")\n'
        '  (subpath "/private/var/db/timezone/zoneinfo")\n'
        '  (literal "/dev/null")\n'
        '  (literal "/dev/random")\n'
        '  (literal "/dev/urandom"))\n'
        "(allow file-read-metadata file-test-existence\n"
        f'  (path-ancestors "{workspace}")\n'
        f'  (path-ancestors "{temporary}")\n'
        f"{trusted_toolchain_ancestors}"
        '  (path-ancestors "/Library/Apple")\n'
        '  (path-ancestors "/Library/Developer/CommandLineTools")\n'
        '  (path-ancestors "/Library/Preferences/com.apple.networkd.plist")\n'
        '  (path-ancestors "/etc/hosts")\n'
        '  (path-ancestors "/opt/homebrew")\n'
        '  (path-ancestors "/usr/local")\n'
        '  (path-ancestors "/var/run/mDNSResponder"))\n'
        "(allow file-write*\n"
        f"{workspace_write_policy}"
        f'  (subpath "{temporary}")\n'
        '  (literal "/dev/null"))\n'
        f"{policy_rules}"
        "(deny file-write*\n"
        "  (require-all\n"
        f'    (subpath "{workspace}")\n'
        f'    (require-not (subpath "{temporary}"))\n'
        '    (regex #"/[.][gG][iI][tT]($|/)")))\n'
        "(deny file-write*\n"
        "  (require-all\n"
        f'    (subpath "{workspace}")\n'
        '    (regex #"/[.][vV][eE][nN][vV]($|/)")))\n'
        "(deny file-write*\n"
        "  (require-all\n"
        f'    (subpath "{workspace}")\n'
        '    (regex #"/[nN][oO][dD][eE]_[mM][oO][dD][uU][lL][eE][sS]'
        '($|/)")))\n'
        '(deny file-write* (require-all\n'
        f'  (require-not (subpath "{temporary}"))\n'
        '  (regex #"/[.]([gG][iI][tT]|[aA][gG][eE][nN][tT][sS]|'
        '[cC][oO][dD][eE][xX]|[aA][wW][sS])($|/)")))\n'
        f"{network_policy}\n"
        f'(deny file-read* file-write* (subpath "{_escape_seatbelt_string(str(command_state_directory()))}"))\n'
    )


def _trusted_toolchain_roots(
    workspace_root: os.PathLike[str] | str,
) -> tuple[Path, ...]:
    """Resolve recognized read-only toolchains used by a workspace venv.

    A uv-created virtual environment commonly links ``.venv/bin/python`` to
    the managed CPython installation outside the workspace. Seatbelt resolves
    that symlink before applying path rules, so the interpreter must be
    readable explicitly. Only the one referenced CPython distribution is
    admitted, and only when it lives below a standard uv-managed Python root.
    """

    workspace = Path(workspace_root).resolve()
    uv_python_roots = tuple(
        path.resolve()
        for path in (
            Path.home() / "Library" / "Application Support" / "uv" / "python",
            Path.home() / ".local" / "share" / "uv" / "python",
        )
        if path.is_dir()
    )
    trusted: set[Path] = set()
    for name in ("python", "python3"):
        executable = workspace / ".venv" / "bin" / name
        if not executable.exists():
            continue
        resolved = executable.resolve()
        for root in uv_python_roots:
            if not resolved.is_relative_to(root):
                continue
            relative = resolved.relative_to(root)
            if relative.parts:
                trusted.add(root / relative.parts[0])
            break
    return tuple(sorted(trusted, key=str))


def _runtime_python_read_roots() -> tuple[Path, ...]:
    candidates = (
        Path(sys.prefix),
        Path(sys.base_prefix),
        Path(sys.executable).parent.parent,
    )
    return tuple(
        sorted(
            {path.expanduser().resolve() for path in candidates if path.exists()},
            key=str,
        )
    )


def _workspace_sandbox_preflight_issue(
    workspace_root: Path,
    *,
    allow_workspace_write: bool,
) -> tuple[str, str] | None:
    root_path = workspace_root.resolve()
    root = os.fspath(root_path)
    pending = [(root, False, False)]
    hardlinks: dict[
        tuple[int, int],
        tuple[int, int, str, bool, bool],
    ] = {}
    while pending:
        (
            directory,
            directory_is_git_metadata,
            directory_is_verification_toolchain,
        ) = pending.pop()
        try:
            entries = os.scandir(directory)
        except OSError:
            relative = Path(os.path.relpath(directory, root)).as_posix()
            return (
                "workspace_scan_failed",
                f"workspace containment scan failed at {relative}",
            )
        with entries:
            for entry in entries:
                try:
                    is_symlink = entry.is_symlink()
                    is_directory = entry.is_dir(follow_symlinks=False)
                except OSError:
                    relative = Path(os.path.relpath(entry.path, root)).as_posix()
                    return (
                        "workspace_scan_failed",
                        f"workspace containment scan failed at {relative}",
                    )
                entry_is_git_metadata = (
                    directory_is_git_metadata or entry.name.casefold() in _PROTECTED_PERMISSION_DIRECTORIES
                )
                entry_is_verification_toolchain = bool(
                    directory_is_verification_toolchain or entry.name.casefold() in _PROTECTED_VERIFICATION_DIRECTORIES
                )
                if entry_is_verification_toolchain and is_symlink:
                    try:
                        resolved_target = Path(entry.path).resolve()
                    except (OSError, RuntimeError):
                        relative = Path(os.path.relpath(entry.path, root)).as_posix()
                        return (
                            "workspace_scan_failed",
                            (f"workspace containment scan failed at {relative}"),
                        )
                    if resolved_target.is_relative_to(root_path):
                        target_relative = resolved_target.relative_to(root_path)
                        if not any(
                            part.casefold() in _PROTECTED_VERIFICATION_DIRECTORIES for part in target_relative.parts
                        ):
                            relative = Path(os.path.relpath(entry.path, root)).as_posix()
                            return (
                                "workspace_verifier_alias",
                                (
                                    "workspace access refused because "
                                    f"verification toolchain path {relative} "
                                    "resolves to writable workspace path "
                                    f"{target_relative.as_posix()}"
                                ),
                            )
                if allow_workspace_write and entry_is_git_metadata and is_symlink:
                    relative = Path(os.path.relpath(entry.path, root)).as_posix()
                    return (
                        "workspace_git_alias",
                        (f"workspace write refused because Git metadata path {relative} is a symlink"),
                    )
                if is_symlink:
                    continue
                if is_directory:
                    pending.append(
                        (
                            entry.path,
                            entry_is_git_metadata,
                            entry_is_verification_toolchain,
                        )
                    )
                    continue
                try:
                    metadata = entry.stat(follow_symlinks=False)
                except OSError:
                    relative = Path(os.path.relpath(entry.path, root)).as_posix()
                    return (
                        "workspace_scan_failed",
                        f"workspace containment scan failed at {relative}",
                    )
                if _SANDBOX_PLATFORM == "linux" and stat.S_ISSOCK(metadata.st_mode):
                    return (
                        "workspace_socket_detected",
                        "Linux sandbox refuses pathname sockets in the workspace",
                    )
                if _SANDBOX_PLATFORM == "linux" and not stat.S_ISREG(metadata.st_mode):
                    return (
                        "workspace_special_file_detected",
                        "Linux sandbox refuses FIFOs and device files in the workspace",
                    )
                if stat.S_ISREG(metadata.st_mode) and metadata.st_nlink > 1:
                    relative = Path(os.path.relpath(entry.path, root)).as_posix()
                    key = (metadata.st_dev, metadata.st_ino)
                    (
                        expected_count,
                        observed_count,
                        first_relative,
                        seen_protected,
                        seen_unprotected,
                    ) = hardlinks.get(
                        key,
                        (
                            metadata.st_nlink,
                            0,
                            relative,
                            False,
                            False,
                        ),
                    )
                    hardlinks[key] = (
                        max(expected_count, metadata.st_nlink),
                        observed_count + 1,
                        first_relative,
                        seen_protected or entry_is_git_metadata or entry_is_verification_toolchain,
                        seen_unprotected or not (entry_is_git_metadata or entry_is_verification_toolchain),
                    )
                pointer_kind: Literal["gitdir", "commondir"] | None = None
                if entry.name.casefold() == ".git":
                    pointer_kind = "gitdir"
                elif directory_is_git_metadata and entry.name.casefold() == "commondir":
                    pointer_kind = "commondir"
                if allow_workspace_write and stat.S_ISREG(metadata.st_mode) and pointer_kind is not None:
                    try:
                        pointer_targets = _git_metadata_pointer_targets(
                            Path(entry.path),
                            kind=pointer_kind,
                        )
                    except (OSError, UnicodeError, RuntimeError):
                        relative = Path(os.path.relpath(entry.path, root)).as_posix()
                        return (
                            "workspace_scan_failed",
                            (f"workspace containment scan failed at {relative}"),
                        )
                    for pointer_target in pointer_targets:
                        if not pointer_target.is_relative_to(root_path):
                            continue
                        target_relative = pointer_target.relative_to(root_path)
                        if not any(part.casefold() == ".git" for part in target_relative.parts):
                            relative = Path(os.path.relpath(entry.path, root)).as_posix()
                            return (
                                "workspace_git_alias",
                                (
                                    "workspace write refused because Git "
                                    f"metadata pointer {relative} targets "
                                    f"{target_relative.as_posix()}"
                                ),
                            )
    for (
        expected_count,
        observed_count,
        first_relative,
        seen_protected,
        seen_unprotected,
    ) in hardlinks.values():
        if observed_count < expected_count or (allow_workspace_write and seen_protected and seen_unprotected):
            return (
                "workspace_hardlink_detected",
                (f"workspace access refused because {first_relative} has a hardlink outside its protected scope"),
            )
    return None


def _git_metadata_pointer_targets(
    path: Path,
    *,
    kind: Literal["gitdir", "commondir"],
) -> tuple[Path, ...]:
    with path.open(encoding="utf-8") as stream:
        content = stream.read(4097)
    if len(content) > 4096:
        raise OSError("Git metadata pointer is unexpectedly large")
    first_line = content.splitlines()[0].strip() if content else ""
    if kind == "gitdir":
        marker = "gitdir:"
        if not first_line.casefold().startswith(marker):
            return ()
        value = first_line[len(marker) :].strip()
    else:
        value = first_line
    if not value:
        return ()
    target = Path(value)
    if not target.is_absolute():
        target = path.parent / target
    lexical_target = Path(os.path.abspath(target))
    return tuple(dict.fromkeys((lexical_target, lexical_target.resolve())))


def _escape_seatbelt_string(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"')


async def _terminate_process_group(
    process: asyncio.subprocess.Process,
    *,
    grace_seconds: float,
) -> None:
    process_group_id = process.pid
    _signal_process_group(process_group_id, signal.SIGTERM)
    if not await _wait_for_process_group_exit(
        process_group_id,
        timeout=grace_seconds,
    ):
        _signal_process_group(process_group_id, signal.SIGKILL)
    if process.returncode is None:
        await process.wait()
    if not await _wait_for_process_group_exit(
        process_group_id,
        timeout=grace_seconds,
    ):
        _signal_process_group(process_group_id, signal.SIGKILL)
        if not await _wait_for_process_group_exit(
            process_group_id,
            timeout=grace_seconds,
        ):
            raise RuntimeError("command process group did not terminate")


def _signal_process_group(process_group_id: int, value: signal.Signals) -> None:
    try:
        os.killpg(process_group_id, value)
    except ProcessLookupError:
        pass


async def _wait_for_process_group_exit(
    process_group_id: int,
    *,
    timeout: float,
) -> bool:
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while _process_group_exists(process_group_id):
        remaining = deadline - loop.time()
        if remaining <= 0:
            return False
        await asyncio.sleep(min(0.01, remaining))
    return True


def _process_group_exists(process_group_id: int) -> bool:
    try:
        os.killpg(process_group_id, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _bounded_stream(value: bytes) -> tuple[str, bool]:
    truncated = len(value) > _MAX_STREAM_BYTES
    if not truncated:
        bounded = value
    else:
        available = _MAX_STREAM_BYTES - len(_STREAM_TRUNCATION_MARKER)
        head_size = available // 2
        tail_size = available - head_size
        bounded = value[:head_size] + _STREAM_TRUNCATION_MARKER + value[-tail_size:]
    return bounded.decode("utf-8", errors="replace"), truncated


def _resolve_command_use(
    workspace: WorkspaceRuntime,
    arguments: Mapping[str, JsonValue],
) -> ResolvedToolUse:
    cwd = workspace.resolve_path(str(arguments["working_dir"]) or ".").resolve()
    effects = {
        ToolEffect.READ_WORKSPACE,
        ToolEffect.EXECUTE_PROCESS,
    }
    if arguments.get("workspace_write") is True:
        effects.update(
            {
                ToolEffect.WRITE_WORKSPACE,
                ToolEffect.DESTRUCTIVE,
            }
        )
    if arguments.get("network") is True:
        effects.add(ToolEffect.NETWORK)
    return ResolvedToolUse(
        effects=frozenset(effects),
        targets=(
            ToolTarget(kind="workspace_path", value=str(workspace.root)),
            ToolTarget(kind="cwd_path", value=str(cwd)),
            ToolTarget(
                kind="execution_mode",
                value="restricted_sandbox",
            ),
        ),
    )


def _resolve_managed_python_use(
    workspace: WorkspaceRuntime,
    arguments: Mapping[str, JsonValue],
) -> ResolvedToolUse:
    cwd = workspace.ensure_within_workspace(workspace.resolve_path(str(arguments["working_dir"]) or "."))
    effects = {
        ToolEffect.READ_WORKSPACE,
        ToolEffect.EXECUTE_PROCESS,
    }
    output_targets: list[ToolTarget] = []
    raw_output_paths = arguments.get("output_paths")
    output_paths = raw_output_paths if isinstance(raw_output_paths, tuple) else ()
    if arguments.get("workspace_write") is True:
        effects.update({ToolEffect.WRITE_WORKSPACE, ToolEffect.DESTRUCTIVE})
        for raw_path in output_paths:
            if not isinstance(raw_path, str):
                raise ValueError("output path must be a string")
            # Resolution declares intent. The executor owns permission checks
            # and their structured, tool-independent denial feedback.
            target = workspace.resolve_path(raw_path)
            output_targets.append(ToolTarget(kind="workspace_path", value=str(target)))
    else:
        output_targets.append(ToolTarget(kind="workspace_path", value=str(workspace.root)))
    if arguments.get("network") is True:
        effects.add(ToolEffect.NETWORK)
    return ResolvedToolUse(
        effects=frozenset(effects),
        targets=(
            *output_targets,
            ToolTarget(kind="cwd_path", value=str(cwd)),
            ToolTarget(kind="execution_mode", value="restricted_sandbox"),
        ),
    )


def _normalize_command_output(raw: object) -> NormalizedToolOutput:
    validated = RunCommandOutput.model_validate(raw)
    structured = json_schema_output(
        _COMMAND_OUTPUT_SCHEMA,
        validated.model_dump(mode="json"),
    )
    if validated.status == "running":
        return NormalizedToolOutput(structured_content=structured)
    if validated.status == "cancelled":
        return NormalizedToolOutput(
            structured_content=structured, is_error=True, error_code="command_cancelled",
            error_message=validated.stderr or "command cancelled and its process group was terminated",
        )
    if validated.timed_out:
        return NormalizedToolOutput(
            structured_content=structured,
            is_error=True,
            error_code="timeout_cancelled",
            error_message="command timed out and its process group was terminated",
            retryable=False,
        )
    if validated.sandbox_error is not None:
        return NormalizedToolOutput(
            structured_content=structured,
            is_error=True,
            error_code=validated.sandbox_error,
            error_message=validated.stderr,
            retryable=False,
        )
    if validated.exit_code != 0:
        return NormalizedToolOutput(
            structured_content=structured,
            is_error=True,
            error_code="command_failed",
            error_message=(
                f"command exited with status {validated.exit_code}; changing "
                "timeout alone cannot fix a completed command, so inspect "
                "stdout/stderr or change the command or code"
            ),
            retryable=False,
        )
    return NormalizedToolOutput(structured_content=structured)


def _normalize_managed_python_output(raw: object) -> NormalizedToolOutput:
    validated = ManagedPythonOutput.model_validate(raw)
    structured = json_schema_output(
        _PYTHON_OUTPUT_SCHEMA,
        validated.model_dump(mode="json"),
    )
    if validated.timed_out:
        return NormalizedToolOutput(
            structured_content=structured,
            is_error=True,
            error_code="timeout_cancelled",
            error_message=("Python code timed out and its process group was terminated"),
            retryable=False,
        )
    if validated.sandbox_error is not None:
        return NormalizedToolOutput(
            structured_content=structured,
            is_error=True,
            error_code=validated.sandbox_error,
            error_message=validated.stderr,
            retryable=False,
        )
    if validated.exit_code != 0:
        return NormalizedToolOutput(
            structured_content=structured,
            is_error=True,
            error_code="python_failed",
            error_message=(
                f"Python code exited with status {validated.exit_code}; inspect "
                "stdout/stderr and change the code rather than retrying unchanged"
            ),
            retryable=False,
        )
    return NormalizedToolOutput(structured_content=structured)


__all__ = [
    "RunCommandInput",
    "RunCommandOutput",
    "ManagedPythonInput",
    "ManagedPythonOutput",
    "create_execute_python_tool",
    "create_run_command_tool",
]
