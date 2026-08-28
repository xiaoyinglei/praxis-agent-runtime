from __future__ import annotations

import asyncio
import codecs
import math
import os
import signal
import tempfile
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from rag.agent.tools.tool import (
    CancellationMode,
    InterruptBehavior,
    JsonValue,
    NormalizedToolOutput,
    ResolvedToolUse,
    Tool,
    ToolDefinition,
    ToolEffect,
    ToolProgress,
    ToolProgressKind,
    ToolProgressSink,
    ToolTarget,
    json_schema_output,
    pydantic_input,
)
from rag.agent.workspace import WorkspaceRuntime

_MAX_STREAM_BYTES = 50_000
_MAX_DELTA_BYTES = 200_000
_MAX_DELTA_CHUNKS = 512
_DELTA_CHUNK_BYTES = 4_096
_SANDBOX_EXEC_PATH = "/usr/bin/sandbox-exec"


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
        le=600.0,
        description="Per-command timeout before the whole process group is terminated.",
    )
    network: bool = Field(
        default=False,
        description=(
            "Request outbound network access. Network is disabled by default "
            "and requires a separate approval from command execution."
        ),
    )


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


_COMMAND_INPUT_SCHEMA, _validate_command_input = pydantic_input(RunCommandInput)
_COMMAND_OUTPUT_SCHEMA, _unused_command_output_validator = pydantic_input(
    RunCommandOutput
)


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
                "workspace-relative directory. Only the workspace and a private "
                "temporary directory are writable; host environment variables and "
                "network access are disabled. Set network=true to request a separate "
                "network approval. Timeout or cancellation terminates and reaps the "
                "complete process group."
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
                ToolEffect.WRITE_WORKSPACE,
                ToolEffect.EXECUTE_PROCESS,
            }
        ),
        resolve_use=lambda arguments: _resolve_command_use(workspace, arguments),
        execution_revision="builtin-run-command-v3-trusted-toolchain",
        idempotent=False,
        concurrency_safe=False,
        cancellation_mode=CancellationMode.MANAGED_PROCESS,
        interrupt_behavior=InterruptBehavior.CANCEL,
        timeout_seconds=hard_timeout_seconds,
        max_model_output_bytes=150_000,
        stream=stream,
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
) -> str:
    workspace = _escape_seatbelt_string(
        str(os.path.realpath(workspace_root))
    )
    temporary = _escape_seatbelt_string(
        str(os.path.realpath(temporary_root))
    )
    trusted_toolchains = tuple(
        _escape_seatbelt_string(str(path))
        for path in _trusted_toolchain_roots(workspace_root)
    )
    trusted_toolchain_reads = "".join(
        f'  (subpath "{path}")\n' for path in trusted_toolchains
    )
    trusted_toolchain_ancestors = "".join(
        f'  (path-ancestors "{path}")\n' for path in trusted_toolchains
    )
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
        f'  (subpath "{workspace}")\n'
        f'  (subpath "{temporary}")\n'
        '  (literal "/dev/null"))\n'
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
    bounded = value[:_MAX_STREAM_BYTES]
    return bounded.decode("utf-8", errors="replace"), truncated


def _resolve_command_use(
    workspace: WorkspaceRuntime,
    arguments: Mapping[str, JsonValue],
) -> ResolvedToolUse:
    cwd = workspace.resolve_path(str(arguments["working_dir"]) or ".").resolve()
    effects = {
        ToolEffect.READ_WORKSPACE,
        ToolEffect.WRITE_WORKSPACE,
        ToolEffect.EXECUTE_PROCESS,
    }
    if arguments.get("network") is True:
        effects.add(ToolEffect.NETWORK)
    return ResolvedToolUse(
        effects=frozenset(effects),
        targets=(
            ToolTarget(kind="workspace_path", value=str(cwd)),
            ToolTarget(kind="cwd_path", value=str(cwd)),
            ToolTarget(
                kind="execution_mode",
                value="restricted_sandbox",
            ),
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
    return NormalizedToolOutput(structured_content=structured)


__all__ = [
    "RunCommandInput",
    "RunCommandOutput",
    "create_run_command_tool",
]
