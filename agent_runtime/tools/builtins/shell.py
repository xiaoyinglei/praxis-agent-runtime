from __future__ import annotations

import asyncio
import codecs
import math
import os
import platform
import signal
import stat
import sys
import tempfile
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

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
from agent_runtime.workspace import WorkspaceRuntime

_MAX_STREAM_BYTES = 16_000
_MAX_COMMAND_TIMEOUT_SECONDS = 600.0
_MILLISECONDS_PER_SECOND = 1_000.0
_STREAM_TRUNCATION_MARKER = (
    b"\n... output truncated; preserved head and tail ...\n"
)
_MAX_DELTA_BYTES = 200_000
_MAX_DELTA_CHUNKS = 512
_DELTA_CHUNK_BYTES = 4_096
_SANDBOX_EXEC_PATH = "/usr/bin/sandbox-exec"
_PROTECTED_VERIFICATION_DIRECTORIES = frozenset(
    {".venv", "node_modules"}
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
    timeout_seconds: float = Field(
        default=120.0,
        gt=0,
        le=_MAX_COMMAND_TIMEOUT_SECONDS * _MILLISECONDS_PER_SECOND,
        description=(
            "Per-command timeout in seconds before the whole process group is "
            "terminated; use 120, not 120000. Values from 1000 through 600000 "
            "are accepted as an unambiguous millisecond form and normalized to "
            "seconds. The effective maximum is 600 seconds."
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
            "approval. .git paths always remain read-only, and .git symlink "
            "aliases are refused."
        ),
    )

    @field_validator("timeout_seconds")
    @classmethod
    def normalize_timeout_milliseconds(cls, value: float) -> float:
        if value <= _MAX_COMMAND_TIMEOUT_SECONDS:
            return value
        if value < _MILLISECONDS_PER_SECOND:
            raise ValueError(
                "timeout_seconds must be at most 600 seconds; "
                "millisecond values must be at least 1000"
            )
        return value / _MILLISECONDS_PER_SECOND


class RunCommandOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    stdout: str
    stderr: str
    exit_code: int
    timed_out: bool
    truncated: bool
    duration_ms: float = Field(ge=0)
    execution_mode: Literal["restricted_sandbox"] = "restricted_sandbox"
    network_enabled: bool = False
    sandbox_error: str | None = None


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
            "Workspace-relative files the code intends to create or replace. "
            "Required when workspace_write=true so the runtime can attest the "
            "generated artifacts."
        ),
    )
    timeout_seconds: float = Field(
        default=120.0,
        gt=0,
        le=_MAX_COMMAND_TIMEOUT_SECONDS * _MILLISECONDS_PER_SECOND,
        description=(
            "Execution timeout in seconds. Values from 1000 through 600000 are "
            "accepted as an unambiguous millisecond form and normalized to seconds."
        ),
    )
    network: bool = Field(
        default=False,
        description=(
            "Request outbound network access. It is disabled by default and "
            "requires separate approval."
        ),
    )
    workspace_write: bool = Field(
        default=False,
        description=(
            "Request workspace write access. Enabling it is destructive and "
            "requires explicit approval; .git, .venv, and node_modules remain "
            "read-only."
        ),
    )

    @field_validator("timeout_seconds")
    @classmethod
    def normalize_timeout_milliseconds(cls, value: float) -> float:
        return RunCommandInput.normalize_timeout_milliseconds(value)


class ManagedPythonOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    stdout: str
    stderr: str
    exit_code: int
    timed_out: bool
    truncated: bool
    duration_ms: float = Field(ge=0)
    execution_mode: Literal["managed_python_sandbox"] = (
        "managed_python_sandbox"
    )
    python_version: str
    network_enabled: bool = False
    sandbox_error: str | None = None


_COMMAND_INPUT_SCHEMA, _validate_command_input = pydantic_input(RunCommandInput)
_COMMAND_OUTPUT_SCHEMA, _unused_command_output_validator = pydantic_input(
    RunCommandOutput
)
_PYTHON_INPUT_SCHEMA, _validate_python_input_base = pydantic_input(
    ManagedPythonInput
)
_PYTHON_OUTPUT_SCHEMA, _unused_python_output_validator = pydantic_input(
    ManagedPythonOutput
)


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
) -> Tool:
    if (
        isinstance(termination_grace_seconds, bool)
        or not isinstance(termination_grace_seconds, (int, float))
        or not math.isfinite(termination_grace_seconds)
        or termination_grace_seconds <= 0
    ):
        raise ValueError("termination_grace_seconds must be positive and finite")

    async def run(arguments: Mapping[str, JsonValue]) -> RunCommandOutput:
        return await _run_command(
            workspace,
            RunCommandInput.model_validate(arguments),
            termination_grace_seconds=float(termination_grace_seconds),
        )

    async def stream(
        arguments: Mapping[str, JsonValue],
        progress_sink: ToolProgressSink,
    ) -> RunCommandOutput:
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
                "Run one shell command inside a restricted OS sandbox from a "
                "workspace-relative directory. The workspace is read-only by default "
                "and a private temporary directory is writable. Set "
                "workspace_write=true to request destructive workspace write access; "
                ".git, .venv, and node_modules remain read-only. Host environment "
                "variables and network access are disabled. Set network=true to "
                "request a separate network approval. Timeout or cancellation "
                "terminates and reaps the complete process group. Commands fail "
                "closed before execution when hardlinked workspace files could "
                "bypass path containment. For implementation tasks, locate the "
                "affected code before execution and prefer focused verification "
                "over a full-repository test run. Use recognized read-only commands "
                "to verify code behavior; do not execute a process solely to reconfirm "
                "unchanged literal file content already established by a read or search."
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
        execution_revision="builtin-run-command-v6-protected-toolchain",
        idempotent=False,
        concurrency_safe=False,
        cancellation_mode=CancellationMode.MANAGED_PROCESS,
        interrupt_behavior=InterruptBehavior.CANCEL,
        timeout_seconds=hard_timeout_seconds,
        max_model_output_bytes=150_000,
        stream=stream,
        approval_profile=ToolApprovalProfile.RESTRICTED_READ_ONLY_PROCESS,
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
                "Run plain Python source with the managed Praxis interpreter in "
                "a restricted OS sandbox. Unlike run_command, this works when the "
                "task workspace has no .venv and does not use shell heredocs. Use "
                "it for spreadsheet transforms, PDF analysis, CSV/JSON processing, "
                "statistics, and charts with the runtime's installed libraries "
                "such as pandas, openpyxl, PyMuPDF, SciPy, and matplotlib. The "
                "workspace is read-only by default. For generated files, set "
                "workspace_write=true and list every intended output in "
                "output_paths; then call inspect_data_file once on each required "
                "artifact and return a final answer. Do not invoke a shell or create a heredoc "
                "from the Python code merely to repeat inspection."
            ),
            input_schema=_PYTHON_INPUT_SCHEMA,
        ),
        validate_input=_validate_python_input,
        run=run,
        normalize_output=_normalize_managed_python_output,
        output_schema=_PYTHON_OUTPUT_SCHEMA,
        static_effects=frozenset(
            {ToolEffect.READ_WORKSPACE, ToolEffect.EXECUTE_PROCESS}
        ),
        resolve_use=lambda arguments: _resolve_managed_python_use(
            workspace, arguments
        ),
        execution_revision="builtin-execute-python-v1-managed-runtime",
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
) -> RunCommandOutput:
    cwd = workspace.ensure_within_workspace(
        workspace.resolve_path(request.working_dir or "."),
    )
    if not cwd.is_dir():
        raise NotADirectoryError(
            f"workspace working directory not found: {request.working_dir}"
        )

    started_at = time.monotonic()
    if not (
        os.path.isfile(_SANDBOX_EXEC_PATH)
        and os.access(_SANDBOX_EXEC_PATH, os.X_OK)
    ):
        return RunCommandOutput(
            stdout="",
            stderr=(
                "restricted command sandbox is unavailable; refusing "
                "unsandboxed execution"
            ),
            exit_code=-1,
            timed_out=False,
            truncated=False,
            duration_ms=(time.monotonic() - started_at) * 1000,
            sandbox_error="sandbox_unavailable",
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
    ) as temporary_root:
        environment = _command_environment(
            workspace_root=workspace.root,
            temporary_root=temporary_root,
        )
        sandbox_profile = _build_command_sandbox_profile(
            workspace_root=workspace.root,
            temporary_root=temporary_root,
            allow_network=request.network,
            allow_workspace_write=request.workspace_write,
        )
        process = await asyncio.create_subprocess_exec(
            _SANDBOX_EXEC_PATH,
            "-p",
            sandbox_profile,
            "/bin/sh",
            "-c",
            request.command,
            cwd=str(cwd),
            env=environment,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            start_new_session=True,
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
            )
        )
        stderr_task = asyncio.create_task(
            _read_command_stream(
                process.stderr,
                kind=ToolProgressKind.STDERR,
                progress_sink=progress_sink,
                progress_budget=progress_budget,
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

        stdout, stdout_truncated = _bounded_stream(stdout_bytes)
        stderr, stderr_truncated = _bounded_stream(stderr_bytes)
        if timed_out and not stderr:
            stderr = "command timed out and its process group was terminated"
        output = RunCommandOutput(
            stdout=stdout,
            stderr=stderr,
            exit_code=(
                process.returncode if process.returncode is not None else -1
            ),
            timed_out=timed_out,
            truncated=stdout_truncated or stderr_truncated,
            duration_ms=(time.monotonic() - started_at) * 1000,
            network_enabled=request.network,
        )
    return output


async def _execute_managed_python(
    workspace: WorkspaceRuntime,
    request: ManagedPythonInput,
    *,
    termination_grace_seconds: float,
) -> ManagedPythonOutput:
    cwd = workspace.ensure_within_workspace(
        workspace.resolve_path(request.working_dir or ".")
    )
    if not cwd.is_dir():
        raise NotADirectoryError(
            f"workspace working directory not found: {request.working_dir}"
        )

    started_at = time.monotonic()
    if not (
        os.path.isfile(_SANDBOX_EXEC_PATH)
        and os.access(_SANDBOX_EXEC_PATH, os.X_OK)
    ):
        return _python_sandbox_error(
            started_at=started_at,
            message=(
                "managed Python sandbox is unavailable; refusing unsandboxed "
                "execution"
            ),
            code="sandbox_unavailable",
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
    ) as temporary_root:
        script_path = Path(temporary_root) / "praxis_task.py"
        script_path.write_text(request.code, encoding="utf-8")
        environment = _command_environment(
            workspace_root=workspace.root,
            temporary_root=temporary_root,
        )
        environment["PYTHONNOUSERSITE"] = "1"
        sandbox_profile = _build_command_sandbox_profile(
            workspace_root=workspace.root,
            temporary_root=temporary_root,
            allow_network=request.network,
            allow_workspace_write=request.workspace_write,
            additional_read_roots=_runtime_python_read_roots(),
        )
        process = await asyncio.create_subprocess_exec(
            _SANDBOX_EXEC_PATH,
            "-p",
            sandbox_profile,
            sys.executable,
            "-I",
            "-B",
            str(script_path),
            cwd=str(cwd),
            env=environment,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            start_new_session=True,
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

        stdout, stdout_truncated = _bounded_stream(stdout_bytes)
        stderr, stderr_truncated = _bounded_stream(stderr_bytes)
        if timed_out and not stderr:
            stderr = "Python code timed out and its process group was terminated"
        return ManagedPythonOutput(
            stdout=stdout,
            stderr=stderr,
            exit_code=(
                process.returncode if process.returncode is not None else -1
            ),
            timed_out=timed_out,
            truncated=stdout_truncated or stderr_truncated,
            duration_ms=(time.monotonic() - started_at) * 1000,
            python_version=platform.python_version(),
            network_enabled=request.network,
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


async def _read_command_stream(
    stream: asyncio.StreamReader,
    *,
    kind: ToolProgressKind,
    progress_sink: ToolProgressSink | None,
    progress_budget: _CommandProgressBudget,
) -> bytes:
    retained = bytearray()
    decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
    while True:
        chunk = await stream.read(_DELTA_CHUNK_BYTES)
        if not chunk:
            break
        retained_limit = _MAX_STREAM_BYTES + 1
        if len(retained) < retained_limit:
            retained.extend(chunk[: retained_limit - len(retained)])
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
    return bytes(retained)


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


def _build_command_sandbox_profile(
    *,
    workspace_root: os.PathLike[str] | str,
    temporary_root: os.PathLike[str] | str,
    allow_network: bool,
    allow_workspace_write: bool = False,
    additional_read_roots: Sequence[os.PathLike[str] | str] = (),
) -> str:
    workspace = _escape_seatbelt_string(
        str(os.path.realpath(workspace_root))
    )
    temporary = _escape_seatbelt_string(
        str(os.path.realpath(temporary_root))
    )
    trusted_roots = {
        path.resolve() for path in _trusted_toolchain_roots(workspace_root)
    }
    trusted_roots.update(
        Path(path).expanduser().resolve()
        for path in additional_read_roots
        if Path(path).expanduser().exists()
    )
    trusted_toolchains = tuple(
        _escape_seatbelt_string(str(path))
        for path in sorted(trusted_roots, key=str)
    )
    trusted_toolchain_reads = "".join(
        f'  (subpath "{path}")\n' for path in trusted_toolchains
    )
    trusted_toolchain_ancestors = "".join(
        f'  (path-ancestors "{path}")\n' for path in trusted_toolchains
    )
    workspace_write_policy = f'  (subpath "{workspace}")\n' if allow_workspace_write else ""
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
        "(allow sysctl-read (sysctl-name-regex #\"^net.routetable\"))"
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
        f'  (subpath "{workspace}")\n'
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
        "    (regex #\"/[nN][oO][dD][eE]_[mM][oO][dD][uU][lL][eE][sS]"
        '($|/)")))\n'
        f"{network_policy}\n"
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
                    relative = Path(
                        os.path.relpath(entry.path, root)
                    ).as_posix()
                    return (
                        "workspace_scan_failed",
                        f"workspace containment scan failed at {relative}",
                    )
                entry_is_git_metadata = (
                    directory_is_git_metadata
                    or entry.name.casefold() == ".git"
                )
                entry_is_verification_toolchain = bool(
                    directory_is_verification_toolchain
                    or entry.name.casefold()
                    in _PROTECTED_VERIFICATION_DIRECTORIES
                )
                if entry_is_verification_toolchain and is_symlink:
                    try:
                        resolved_target = Path(entry.path).resolve()
                    except (OSError, RuntimeError):
                        relative = Path(
                            os.path.relpath(entry.path, root)
                        ).as_posix()
                        return (
                            "workspace_scan_failed",
                            (
                                "workspace containment scan failed at "
                                f"{relative}"
                            ),
                        )
                    if resolved_target.is_relative_to(root_path):
                        target_relative = resolved_target.relative_to(
                            root_path
                        )
                        if not any(
                            part.casefold()
                            in _PROTECTED_VERIFICATION_DIRECTORIES
                            for part in target_relative.parts
                        ):
                            relative = Path(
                                os.path.relpath(entry.path, root)
                            ).as_posix()
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
                    relative = Path(
                        os.path.relpath(entry.path, root)
                    ).as_posix()
                    return (
                        "workspace_git_alias",
                        (
                            "workspace write refused because Git metadata "
                            f"path {relative} is a symlink"
                        ),
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
                    relative = Path(
                        os.path.relpath(entry.path, root)
                    ).as_posix()
                    return (
                        "workspace_scan_failed",
                        f"workspace containment scan failed at {relative}",
                    )
                if stat.S_ISREG(metadata.st_mode) and metadata.st_nlink > 1:
                    relative = Path(
                        os.path.relpath(entry.path, root)
                    ).as_posix()
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
                        seen_protected
                        or entry_is_git_metadata
                        or entry_is_verification_toolchain,
                        seen_unprotected
                        or not (
                            entry_is_git_metadata
                            or entry_is_verification_toolchain
                        ),
                    )
                pointer_kind: Literal["gitdir", "commondir"] | None = None
                if entry.name.casefold() == ".git":
                    pointer_kind = "gitdir"
                elif (
                    directory_is_git_metadata
                    and entry.name.casefold() == "commondir"
                ):
                    pointer_kind = "commondir"
                if (
                    allow_workspace_write
                    and stat.S_ISREG(metadata.st_mode)
                    and pointer_kind is not None
                ):
                    try:
                        pointer_targets = _git_metadata_pointer_targets(
                            Path(entry.path),
                            kind=pointer_kind,
                        )
                    except (OSError, UnicodeError, RuntimeError):
                        relative = Path(
                            os.path.relpath(entry.path, root)
                        ).as_posix()
                        return (
                            "workspace_scan_failed",
                            (
                                "workspace containment scan failed at "
                                f"{relative}"
                            ),
                        )
                    for pointer_target in pointer_targets:
                        if not pointer_target.is_relative_to(root_path):
                            continue
                        target_relative = pointer_target.relative_to(root_path)
                        if not any(
                            part.casefold() == ".git"
                            for part in target_relative.parts
                        ):
                            relative = Path(
                                os.path.relpath(entry.path, root)
                            ).as_posix()
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
        if observed_count < expected_count or (
            allow_workspace_write
            and seen_protected
            and seen_unprotected
        ):
            return (
                "workspace_hardlink_detected",
                (
                    "workspace access refused because "
                    f"{first_relative} has a hardlink outside its protected scope"
                ),
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
        bounded = (
            value[:head_size]
            + _STREAM_TRUNCATION_MARKER
            + value[-tail_size:]
        )
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
    cwd = workspace.ensure_within_workspace(
        workspace.resolve_path(str(arguments["working_dir"]) or ".")
    )
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
            target = workspace.ensure_within_workspace(
                workspace.resolve_path(raw_path)
            )
            output_targets.append(
                ToolTarget(kind="workspace_path", value=str(target))
            )
    else:
        output_targets.append(
            ToolTarget(kind="workspace_path", value=str(workspace.root))
        )
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
            error_message=(
                "Python code timed out and its process group was terminated"
            ),
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
