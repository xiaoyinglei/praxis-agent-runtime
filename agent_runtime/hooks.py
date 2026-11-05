"""Trusted host registration for the Harness lifecycle hooks."""

from __future__ import annotations

import asyncio
import json
import math
import os
import signal
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, fields
from functools import partial
from pathlib import Path
from typing import Any, Literal

from agent_runtime.core.model_request import canonical_hash
from agent_runtime.tools.tool import ToolEffect, _thaw_json

HookEvent = Literal["PreToolUse", "PostToolUse", "Stop"]
EVENTS = ("PreToolUse", "PostToolUse", "Stop")
MAX_HOOKS = 16
MAX_FEEDBACK_BYTES = 2048


@dataclass(frozen=True, slots=True)
class HookRequest:
    event: HookEvent
    invocation_id: str
    thread_id: str
    turn_id: str
    workspace: str
    source_item_id: str | None = None
    operation_id: str | None = None
    tool_call_id: str | None = None
    tool_name: str | None = None
    arguments: Mapping[str, Any] | None = None
    effects: tuple[str, ...] = ()
    result: Mapping[str, Any] | None = None
    answer: str | None = None
    input_revision: int | None = None
    resources: tuple[Mapping[str, Any], ...] = ()
    completion: Mapping[str, Any] | None = None
    completion_revision: str | None = None


@dataclass(frozen=True, slots=True)
class HookDecision:
    block: bool = False
    feedback: str = ""

    def __post_init__(self) -> None:
        if type(self.block) is not bool or not isinstance(self.feedback, str):
            raise ValueError("hook decision requires a boolean block and string feedback")
        if len(self.feedback.encode()) > MAX_FEEDBACK_BYTES:
            raise ValueError("hook feedback exceeds 2048 bytes")


@dataclass(frozen=True, slots=True)
class LifecycleHook:
    event: HookEvent
    name: str
    handler: Callable[[HookRequest], Awaitable[HookDecision | None]]
    revision: str = "1"
    timeout_seconds: float = 5.0
    tool_names: tuple[str, ...] = ()
    effects: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.event not in EVENTS:
            raise ValueError(f"invalid hook event: {self.event}")
        if any(not isinstance(value, str) or not value.strip() or len(value) > 128
               for value in (self.name, self.revision)):
            raise ValueError("hook name and revision must have 1–128 characters")
        if not callable(self.handler):
            raise TypeError("hook handler must be an async callable")
        seconds = self.timeout_seconds
        if type(seconds) not in {int, float} or not math.isfinite(seconds) or not 0 < seconds <= 30:
            raise ValueError("hook timeout must be greater than 0 and at most 30 seconds")

        for field in ("tool_names", "effects"):
            values = getattr(self, field)
            if not isinstance(values, (list, tuple)) or any(
                not isinstance(value, str) or not value.strip() or value.strip() != value for value in values
            ):
                raise ValueError(f"hook {field} must contain exact non-empty names")
            if field == "effects":
                values = tuple(ToolEffect(value).value for value in values)
            object.__setattr__(self, field, tuple(sorted(set(values))))
        if self.event == "Stop" and (self.tool_names or self.effects):
            raise ValueError("tool selectors apply only to tool hooks")

    def binding(self) -> dict[str, Any]:
        return {
            "event": self.event, "name": self.name, "revision": self.revision,
            "timeout_seconds": self.timeout_seconds,
            **({"tool_names": list(self.tool_names)} if self.tool_names else {}),
            **({"effects": list(self.effects)} if self.effects else {}),
        }


def validate_hooks(hooks: tuple[LifecycleHook, ...]) -> tuple[LifecycleHook, ...]:
    if len(hooks) > MAX_HOOKS or any(not isinstance(hook, LifecycleHook) for hook in hooks):
        raise ValueError("hooks must contain at most 16 LifecycleHook registrations")
    keys = {(hook.event, hook.name) for hook in hooks}
    if len(keys) != len(hooks):
        raise ValueError("hook names must be unique within each event")
    return hooks


def load_host_hooks(path: str, workspace: Path) -> tuple[LifecycleHook, ...]:
    """Load only an explicitly selected host file outside the project."""
    source = Path(path).expanduser()
    if not source.is_absolute():
        raise ValueError("PRAXIS_HOOKS_CONFIG must name an absolute file outside the workspace")
    source = source.resolve(strict=True)
    if source.is_relative_to(workspace):
        raise ValueError("PRAXIS_HOOKS_CONFIG must name an absolute file outside the workspace")
    with source.open("rb") as file:
        raw = file.read(65_537)
    if len(raw) > 65_536:
        raise ValueError("host hook configuration exceeds 64 KiB")
    entries = json.loads(raw)
    if not isinstance(entries, list) or len(entries) > MAX_HOOKS:
        raise ValueError("host hook configuration must be an array of at most 16 entries")
    hooks = []
    for entry in entries:
        if not isinstance(entry, dict) or not {"event", "name", "argv"} <= entry.keys() or (
            entry.keys() - {"event", "name", "argv", "revision", "timeout_seconds", "tool_names", "effects"}
        ):
            raise ValueError("invalid host hook entry; event, name and argv are required")
        argv = entry["argv"]
        if (
            not isinstance(argv, list) or not argv
            or any(not isinstance(arg, str) or "\0" in arg for arg in argv)
            or not Path(argv[0]).is_absolute()
            or Path(argv[0]).resolve().is_relative_to(workspace)
        ):
            raise ValueError("hook argv requires an absolute executable outside the workspace")
        revision = entry.get("revision", "1")
        if not isinstance(revision, str) or not revision.strip() or len(revision) > 128:
            raise ValueError("command hook revision must have 1–128 characters")
        executable = Path(argv[0]).resolve(strict=True)
        argv = [str(executable), *argv[1:]]
        hooks.append(LifecycleHook(
            entry["event"],
            entry["name"],
            partial(_run_command, tuple(argv), source.parent),
            revision=canonical_hash({"argv": argv, "directory": str(source.parent), "revision": revision}),
            timeout_seconds=entry.get("timeout_seconds", 5.0),
            tool_names=entry.get("tool_names", ()),
            effects=entry.get("effects", ()),
        ))
    return validate_hooks(tuple(hooks))


async def _run_command(argv: tuple[str, ...], cwd: Path, request: HookRequest) -> HookDecision | None:
    data = {
        field.name: _thaw_json(getattr(request, field.name))
        for field in fields(request)
        if request.completion_revision is not None or field.name not in {"completion", "completion_revision"}
    }
    payload = json.dumps(data, ensure_ascii=False).encode()
    proc = await asyncio.create_subprocess_exec(
        *argv,
        cwd=cwd,
        env={"PATH": os.defpath},
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.DEVNULL,
        start_new_session=True,
    )
    assert proc.stdin is not None and proc.stdout is not None

    async def send() -> None:
        try:
            proc.stdin.write(payload + b"\n")
            await proc.stdin.drain()
        except (BrokenPipeError, ConnectionResetError):
            pass
        finally:
            proc.stdin.close()

    sender = asyncio.create_task(send())
    try:
        # read(n) can return early; readexactly enforces a total output limit.
        try:
            await proc.stdout.readexactly(16_385)
        except asyncio.IncompleteReadError as end:
            output = end.partial
        else:
            raise ValueError("hook command output exceeds 16 KiB")
        await sender
        if await proc.wait() != 0:
            raise ValueError("hook command exited unsuccessfully")
        if not output.strip():
            return None
        result = json.loads(output)
        if not isinstance(result, dict):
            raise ValueError("hook command must return a JSON decision object")
        return HookDecision(**result)
    finally:
        sender.cancel()
        if proc.returncode is None or not proc.stdout.at_eof():
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        await asyncio.gather(sender, return_exceptions=True)
        # Drain the pipe after signalling the process group. A descendant
        # that deliberately detached must not keep the Turn waiting for EOF.
        try:
            async with asyncio.timeout(0.5):
                while await proc.stdout.read(16_384):
                    pass
                await proc.wait()
        except TimeoutError:
            pass
