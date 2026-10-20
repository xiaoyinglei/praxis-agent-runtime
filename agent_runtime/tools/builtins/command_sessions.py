"""macOS/Linux command handles; process execution and OS policy stay in shell.

Yielding returns control while the process remains alive; the execution
deadline bounds its lifetime across subsequent polls.
"""
from __future__ import annotations

import asyncio
import hashlib
import math
import os
import secrets
import subprocess
import sys
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import BinaryIO, Literal

from pydantic import BaseModel, ConfigDict, Field

from agent_runtime.tools.builtins import shell
from agent_runtime.tools.tool import (
    CancellationMode,
    InterruptBehavior,
    JsonValue,
    ResolvedToolUse,
    Tool,
    ToolContinuation,
    ToolDefinition,
    ToolEffect,
    ToolProgressKind,
    ToolTarget,
    pydantic_input,
)
from agent_runtime.workspace import WorkspaceRuntime

_MAX_RETAINED_SESSIONS = 32


class ManageCommandInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    session_id: str = Field(min_length=1, max_length=100, description="Live session_id returned by run_command.")
    action: Literal["status", "wait", "cancel", "read"] = Field(
        default="wait", description="Query immediately, wait for completion, or terminate and reap the process group.",
    )
    stream: Literal["stdout", "stderr"] = Field(default="stdout", description="Log stream for action=read.")
    offset: int = Field(default=0, ge=0, description="Byte offset in the retained log, for action=read.")
    max_bytes: int = Field(default=16000, ge=1, le=64000, description="Maximum bytes to read from the log.")
    wait_seconds: float = Field(
        default=10, ge=0, le=60,
        description="Response wait for action=wait, in seconds; expiry returns running without killing the command.",
    )


@dataclass
class _CommandSession:
    session_id: str
    started_at: float
    network_enabled: bool
    task: asyncio.Task[shell.RunCommandOutput] | None = None
    stdout: shell._CommandStreamBuffer = field(default_factory=shell._CommandStreamBuffer)
    stderr: shell._CommandStreamBuffer = field(default_factory=shell._CommandStreamBuffer)
    observed_terminal: bool = False
    state_path: Path | None = None
    owner_lock: BinaryIO | None = None
    released: bool = False
    request_digest: str | None = None
    detaching: bool = False

    def capture(self, kind: ToolProgressKind, chunk: bytes) -> None:
        (self.stdout if kind == ToolProgressKind.STDOUT else self.stderr).append(chunk)

    def snapshot(self, *, cancelled: bool = False) -> shell.RunCommandOutput:
        stdout, out_truncated = shell._bounded_stream(self.stdout.value())
        stderr, err_truncated = shell._bounded_stream(self.stderr.value())
        return shell.RunCommandOutput(
            stdout=stdout, stderr=stderr, exit_code=-1 if cancelled else None,
            timed_out=False, truncated=out_truncated or err_truncated,
            duration_ms=(time.monotonic() - self.started_at) * 1000,
            status="cancelled" if cancelled else "running", session_id=self.session_id,
            network_enabled=self.network_enabled,
            request_digest=self.request_digest,
        )


class CommandSessions:
    """One active shell job per live Session, with bounded terminal result retention.

    Only the owning Session may address these opaque handles. No process is
    restarted by a poll; close/cancel await the existing shell cleanup path.
    """

    def __init__(self, workspace: WorkspaceRuntime, *, termination_grace_seconds: float = 0.5) -> None:
        if not math.isfinite(termination_grace_seconds) or termination_grace_seconds <= 0:
            raise ValueError("termination_grace_seconds must be positive and finite")
        self.workspace = workspace
        self.termination_grace_seconds = termination_grace_seconds
        self._sessions: dict[str, _CommandSession] = {}
        self._closed = False

    @property
    def running(self) -> bool:
        return any(entry.task is not None and not entry.task.done() for entry in self._sessions.values())

    def _error(self, code: str, message: str) -> shell.RunCommandOutput:
        return shell.RunCommandOutput(stdout="", stderr=message, exit_code=-1, timed_out=False,
                                      truncated=False, duration_ms=0, sandbox_error=code)

    def busy_output(self) -> shell.RunCommandOutput:
        return self._error("command_session_busy", "A command is running; use manage_command to wait or cancel it.")

    async def start(self, request: shell.RunCommandInput) -> shell.RunCommandOutput:
        if self._closed or shell._SANDBOX_PLATFORM not in {"darwin", "linux"}:
            return self._error("command_session_unavailable", "Live command sessions require macOS or Linux.")
        if self.running:
            return self.busy_output()
        # Never evict a running job; retain a bounded set of completed results.
        while len(self._sessions) >= _MAX_RETAINED_SESSIONS:
            collected = next((key for key, value in self._sessions.items() if value.observed_terminal), None)
            if collected is None:
                return self._error(
                    "command_session_capacity", "Collect an outstanding terminal result before starting more.",
                )
            retired = self._sessions.pop(collected)
            if retired.owner_lock is not None:
                retired.owner_lock.close()
        entry = _CommandSession("cmd_" + secrets.token_hex(16), time.monotonic(), request.network)
        entry.request_digest = hashlib.sha256(request.model_dump_json().encode()).hexdigest()
        self._sessions[entry.session_id] = entry
        entry.task = asyncio.create_task(self._execute(entry, request))
        assert request.yield_time_seconds is not None
        return await self._collect(entry, request.yield_time_seconds)

    async def after_commit(self, arguments: Mapping[str, JsonValue], output: JsonValue | None,
                           operation_id: str) -> None:
        """In-memory standalone sessions need no durable release handshake."""

    async def _execute(self, entry: _CommandSession, request: shell.RunCommandInput) -> shell.RunCommandOutput:
        try:
            result = await shell._run_command(
                self.workspace, request, termination_grace_seconds=self.termination_grace_seconds,
                capture_output=entry.capture,
            )
            return result.model_copy(update={"session_id": entry.session_id})
        except asyncio.CancelledError:
            return entry.snapshot(cancelled=True)
        except Exception as error:
            result = self._error("command_session_failed", str(error))
            return result.model_copy(update={"session_id": entry.session_id})

    async def _cancel(self, entry: _CommandSession) -> shell.RunCommandOutput:
        assert entry.task is not None
        if not entry.task.done():
            entry.task.cancel()
        try:
            return self._identity(entry, await asyncio.shield(entry.task))
        except asyncio.CancelledError:
            # Task cancelled before its coroutine started: no child exists yet.
            if entry.task.cancelled():
                return entry.snapshot(cancelled=True)
            raise

    @staticmethod
    def _identity(entry: _CommandSession, result: shell.RunCommandOutput) -> shell.RunCommandOutput:
        return result.model_copy(update={"session_id": entry.session_id, "request_digest": entry.request_digest})

    async def _collect(self, entry: _CommandSession, wait_seconds: float) -> shell.RunCommandOutput:
        assert entry.task is not None
        try:
            if not entry.task.done() and wait_seconds:
                await asyncio.wait({entry.task}, timeout=wait_seconds)
            if entry.task.cancelled():
                entry.observed_terminal = True
                return entry.snapshot(cancelled=True)
            if entry.task.done():
                entry.observed_terminal = True
                return self._identity(entry, entry.task.result())
            return entry.snapshot()
        except asyncio.CancelledError:
            await self._cancel(entry)
            raise

    async def manage(
        self, session_id: str, *, action: Literal["status", "wait", "cancel"], wait_seconds: float,
    ) -> shell.RunCommandOutput:
        entry = self._sessions.get(session_id)
        if self._closed or entry is None:
            return self._error(
                "command_session_not_found", "Unknown handle; validated ledger recovery is required to attach a job.",
            )
        if action == "cancel":
            result = await self._cancel(entry)
            entry.observed_terminal = True
            return result
        return await self._collect(entry, wait_seconds if action == "wait" else 0)

    async def aclose(self) -> None:
        await self.close_results()

    async def close_results(self) -> tuple[tuple[ToolContinuation, shell.RunCommandOutput], ...]:
        self._closed = True
        return await self.cancel_results()

    async def cancel_results(self) -> tuple[tuple[ToolContinuation, shell.RunCommandOutput], ...]:
        outcomes = []
        for entry in self._sessions.values():
            result = await self._cancel(entry)
            if not entry.observed_terminal:
                outcomes.append((ToolContinuation(self, entry.session_id, "observe", True), result))
        self._sessions.clear()
        return tuple(outcomes)

    def continuation(
        self, arguments: Mapping[str, JsonValue], output: JsonValue | None,
    ) -> ToolContinuation | None:
        if not isinstance(output, Mapping):
            return None
        session_id = output.get("session_id")
        if not isinstance(session_id, str) or session_id not in self._sessions:
            return None
        phase: Literal["start", "observe"] = "observe" if arguments.get("session_id") == session_id else "start"
        return ToolContinuation(self, session_id, phase, output.get("status") != "running")


def create_manage_command_tool(workspace: WorkspaceRuntime, sessions: CommandSessions | None) -> Tool:
    if sessions is not None and sessions.workspace is not workspace:
        raise ValueError("command session owner belongs to a different workspace runtime")
    input_schema, validate = pydantic_input(ManageCommandInput)

    async def run(arguments: Mapping[str, JsonValue]) -> shell.RunCommandOutput:
        request = ManageCommandInput.model_validate(arguments)
        if sessions is None:
            return shell.RunCommandOutput(
                stdout="", stderr="No live command session owner", exit_code=-1, timed_out=False,
                truncated=False, duration_ms=0, sandbox_error="command_session_unavailable",
            )
        if request.action == "read":
            if not isinstance(sessions, DurableCommandSessions):
                return sessions._error("command_logs_unavailable", "Durable logs require an Agent Session.")
            return await sessions.read_log(request.session_id, request.stream, request.offset, request.max_bytes)
        return await sessions.manage(request.session_id, action=request.action, wait_seconds=request.wait_seconds)

    return Tool(
        definition=ToolDefinition(
            name="manage_command",
            description=(
                "Manage an existing macOS/Linux run_command session without starting another process. "
                "action=status returns immediately; wait returns after completion or wait_seconds, "
                "and an expired wait leaves the process running. cancel terminates and reaps its process group. "
                "Cancelling a wait also cancels the command. Outputs are bounded cumulative head/tail snapshots, "
                "not new deltas. A running result is not completion evidence. Handles expire when the live "
                "Turn ends, the Agent Session closes or 32 collected results are retained; "
                "Uncollected results are never evicted. Crash recovery reattaches committed jobs without rerunning; "
                "worker loss is reported as unknown. action=read reads durable output by stream and byte offset, "
                "at most 64000 bytes per response; each stream retains its first 16 MiB and reports quota truncation. "
                "macOS workspace files already written remain after failure, timeout, cancel or Session close. "
                "This tool grants no new path or network access and cannot alter the original command."
            ), input_schema=input_schema,
        ),
        validate_input=validate, run=run, normalize_output=shell._normalize_command_output,
        output_schema=shell._COMMAND_OUTPUT_SCHEMA,
        static_effects=frozenset({ToolEffect.READ_WORKSPACE}),
        resolve_use=lambda _args: ResolvedToolUse(
            effects=frozenset({ToolEffect.READ_WORKSPACE}),
            targets=(ToolTarget(kind="workspace_path", value=str(workspace.root)),),
        ),
        execution_revision="builtin-manage-command-v2-durable", idempotent=False, concurrency_safe=False,
        cancellation_mode=CancellationMode.MANAGED_PROCESS, interrupt_behavior=InterruptBehavior.CANCEL,
        timeout_seconds=65, max_model_output_bytes=150_000,
        continuation=None if sessions is None else sessions.continuation,
    )


class DurableCommandSessions(CommandSessions):
    """A committed launch releases one detached worker; recovery never replays it.

    The stable owner lock prevents two Agents controlling one job. A separate
    worker lock and durable started marker distinguish prepared, live and lost.
    """

    def __init__(self, workspace: WorkspaceRuntime, *, termination_grace_seconds: float = 0.5) -> None:
        super().__init__(workspace, termination_grace_seconds=termination_grace_seconds)
        from agent_runtime.tools.builtins import command_worker as journal
        self._worker_code = journal.freeze_code(workspace.root)

    async def manage(
        self, session_id: str, *, action: Literal["status", "wait", "cancel"], wait_seconds: float,
    ) -> shell.RunCommandOutput:
        from agent_runtime.tools.builtins import command_worker as journal
        entry = self._sessions.get(session_id)
        if entry is not None and entry.state_path is not None:
            original = journal.read_json(entry.state_path / "manifest.json")
            if not journal.policy_matches(self.workspace, original["permissions"]):
                await self._cancel(entry)
                entry.observed_terminal = True
                return self._identity(entry, self._error(
                    "command_permissions_changed", "Command cancelled after policy changed; old output withheld.",
                ))
        result = await super().manage(session_id, action=action, wait_seconds=wait_seconds)
        return result if entry is None else await self._authorized_result(entry, result)

    async def _authorized_result(
        self, entry: _CommandSession, result: shell.RunCommandOutput,
    ) -> shell.RunCommandOutput:
        from agent_runtime.tools.builtins import command_worker as journal
        if entry.state_path is not None:
            original = journal.read_json(entry.state_path / "manifest.json")
            if not journal.policy_matches(self.workspace, original["permissions"]):
                await super()._cancel(entry)  # Confirm cleanup before reporting a terminal revocation.
                return self._identity(entry, self._error(
                    "command_permissions_changed", "Command cancelled after policy changed; old output withheld.",
                ))
        return result

    async def _cancel(self, entry: _CommandSession) -> shell.RunCommandOutput:
        return await self._authorized_result(entry, await super()._cancel(entry))

    async def _execute(self, entry: _CommandSession, request: shell.RunCommandInput) -> shell.RunCommandOutput:
        from agent_runtime.tools.builtins import command_worker as journal

        if entry.state_path is None:
            root = journal.private_root(self.workspace.root)
            entry.state_path = root / entry.session_id
            entry.state_path.mkdir(mode=0o700)
            entry.owner_lock = journal.lock(entry.state_path / "owner.lock")
            metadata = self.workspace.root.stat()
            journal.write_json(entry.state_path / "manifest.json", {
                "revision": journal.REVISION, "session_id": entry.session_id,
                "workspace": str(self.workspace.root.resolve()), "identity": [metadata.st_dev, metadata.st_ino],
                "request": request.model_dump(mode="json"), "request_digest": journal.request_digest(request),
                "permissions": journal.policy_value(self.workspace), "grace": self.termination_grace_seconds,
                "platform": shell._SANDBOX_PLATFORM, "sandbox_exec": shell._SANDBOX_EXEC_PATH,
                "bwrap": shell._BWRAP_PATH,
                "worker_code": str(self._worker_code),
            })
        path = entry.state_path
        process: subprocess.Popen[bytes] | None = None
        revoked = False
        try:
            while True:
                manifest = journal.read_json(path / "manifest.json")
                if not journal.policy_matches(self.workspace, manifest["permissions"]):
                    revoked = True
                    journal.write_json(path / "cancel.json", {})
                if not entry.released:
                    await asyncio.sleep(0.02)
                    continue
                if process is None and not (path / "started.json").exists() and not (path / "result.json").exists():
                    # -I excludes cwd, PYTHONPATH and user sitecustomize. The
                    # installation source is explicit, never inferred from cwd.
                    code = ("import sys; sys.path.insert(0," + repr(manifest["worker_code"])
                            + "); from agent_runtime.tools.builtins.command_worker import main; main(sys.argv[1])")
                    with open(path / "worker.log", "ab", buffering=0) as errors:
                        os.chmod(path / "worker.log", 0o600)
                        process = subprocess.Popen(
                            [sys.executable, "-I", "-c", code, str(path)], cwd="/",
                            stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=errors,
                            start_new_session=True, close_fds=True,
                        )
                if process is not None:
                    process.poll()  # reap an exited worker owned by this interpreter
                result_path = path / "result.json"
                if result_path.exists():
                    result = shell.RunCommandOutput.model_validate(journal.read_json(result_path))
                    if result.request_digest != manifest["request_digest"] or result.session_id != entry.session_id:
                        return self._error("command_state_invalid", "Worker result identity mismatch")
                    if revoked:
                        return self._error("command_permissions_changed", "Command cancelled after policy changed.")
                    return result
                if (path / "started.json").exists():
                    held = journal.lock(path / "run.lock")
                    if held is not None:
                        with held:
                            if result_path.exists():
                                continue
                        return self._error("command_outcome_unknown", "Worker lost its final result; not rerun.")
                elif process is not None and process.returncode is not None:
                    return self._error("command_worker_unavailable", "Worker could not start; inspect worker log.")
                progress = path / "progress.json"
                if progress.exists() and not revoked:
                    snapshot = shell.RunCommandOutput.model_validate(journal.read_json(progress))
                    # Parent snapshots are replaced, not accumulated twice.
                    entry.stdout = shell._CommandStreamBuffer()
                    entry.stderr = shell._CommandStreamBuffer()
                    entry.stdout.append(snapshot.stdout.encode())
                    entry.stderr.append(snapshot.stderr.encode())
                publish = path / "publish-request.json"
                if publish.exists() and not revoked:
                    journal.write_json(path / "publish-permit.json", journal.read_json(publish))
                await asyncio.sleep(0.05)
        except asyncio.CancelledError:
            if entry.detaching:
                return entry.snapshot()
            if not entry.released:
                return entry.snapshot(cancelled=True)
            journal.write_json(path / "cancel.json", {})
            if process is None and not (path / "started.json").exists():
                held = journal.lock(path / "run.lock")
                if held is not None:
                    with held:
                        result = entry.snapshot(cancelled=True)
                        journal.write_json(path / "result.json", result.model_dump(mode="json"))
                        return result
            # Let the worker reap descendants and discard its candidate. This
            # does not send a signal to a stale/reused PID.
            while not (path / "result.json").exists():
                held = journal.lock(path / "run.lock")
                if (path / "started.json").exists() and held is not None:
                    with held:
                        if (path / "result.json").exists():
                            break
                    return self._error("command_outcome_unknown", "Worker was lost during cancellation")
                if held is not None:
                    held.close()
                if process is not None and process.poll() is not None:
                    return self._error("command_worker_unavailable", "Worker exited during cancellation")
                await asyncio.sleep(0.05)
            return shell.RunCommandOutput.model_validate(journal.read_json(path / "result.json"))
        finally:
            if process is not None:
                process.poll()

    async def after_commit(self, arguments: Mapping[str, JsonValue], output: JsonValue | None,
                           operation_id: str) -> None:
        from agent_runtime.tools.builtins import command_worker as journal

        if not isinstance(output, Mapping):
            return
        key = output.get("session_id")
        if not isinstance(key, str) or key not in self._sessions:
            return
        entry = self._sessions[key]
        if entry.state_path is None:
            raise RuntimeError("command was not prepared before launch commit")
        journal.write_json(entry.state_path / "release.json", {"operation_id": operation_id})
        entry.released = True

    async def start(self, request: shell.RunCommandInput) -> shell.RunCommandOutput:
        from agent_runtime.tools.builtins import command_worker as journal
        result = await super().start(request)
        if result.session_id is not None:
            result = result.model_copy(update={"request_digest": journal.request_digest(request)})
        return result

    async def restore(self, key: str, *, request: shell.RunCommandInput, digest: str,
                      operation_id: str) -> None:
        from agent_runtime.tools.builtins import command_worker as journal

        if key in self._sessions:
            return
        path = journal.job_path(self.workspace.root, key)
        manifest = journal.read_json(path / "manifest.json")
        original = journal.validate_manifest(self.workspace, manifest)
        if digest != manifest["request_digest"] or journal.request_digest(request) != digest or original != request:
            raise ValueError("durable launch does not match the command request")
        release = path / "release.json"
        if release.exists() and journal.read_json(release).get("operation_id") != operation_id:
            raise ValueError("command operation identity mismatch")
        owner = journal.lock(path / "owner.lock")
        if owner is None:
            raise RuntimeError("command session is owned by another live Agent")
        entry = _CommandSession(key, time.monotonic(), request.network, state_path=path, owner_lock=owner)
        entry.request_digest = digest
        self._sessions[key] = entry
        entry.task = asyncio.create_task(self._execute(entry, request))
        await self.after_commit({}, {"session_id": key}, operation_id)
        await asyncio.sleep(0)  # status can observe an already committed final result

    async def read_log(self, key: str, stream: str, offset: int, max_bytes: int) -> shell.RunCommandOutput:
        from agent_runtime.tools.builtins import command_worker as journal
        if stream not in {"stdout", "stderr"} or offset < 0 or not 1 <= max_bytes <= 64000:
            raise ValueError("invalid command log range")
        result = await self.manage(key, action="status", wait_seconds=0)
        entry = self._sessions.get(key)
        if entry is None or entry.state_path is None or result.sandbox_error is not None:
            return result
        path = entry.state_path / (stream + ".log")
        data = b""
        end = 0
        if path.exists():
            journal.check_private(path)
            fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC)
            with os.fdopen(fd, "rb") as output:
                end = os.fstat(output.fileno()).st_size
                output.seek(min(offset, end))
                data = output.read(max_bytes)
        return await self._authorized_result(entry, result.model_copy(update={
            "log_chunk": data.decode("utf-8", errors="replace"),
            "log_next_offset": min(offset, end) + len(data), "log_end_offset": end,
            "log_truncated": end >= journal.LOG_QUOTA_BYTES,
        }))

    async def cancel_results(self) -> tuple[tuple[ToolContinuation, shell.RunCommandOutput], ...]:
        entries = tuple(self._sessions.values())
        try:
            return await super().cancel_results()
        finally:
            for entry in entries:
                if entry.owner_lock is not None:
                    entry.owner_lock.close()

    async def detach(self) -> None:
        """Host recovery inspector releases ownership without cancelling jobs.

        This is never a model tool or Session.close behavior.
        """
        self._closed = True
        for entry in self._sessions.values():
            entry.detaching = True
            if entry.task is not None and not entry.task.done():
                entry.task.cancel()
                await asyncio.gather(entry.task, return_exceptions=True)
            if entry.owner_lock is not None:
                entry.owner_lock.close()
        self._sessions.clear()
