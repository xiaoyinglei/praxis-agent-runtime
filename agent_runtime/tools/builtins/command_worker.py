"""Private local command supervisor. No model/API client or RPC server.

Ownership split inspired by tmux and Supervisor; independently implemented.
Only the trusted host writes this directory. Sandbox processes cannot see it.
"""
from __future__ import annotations

import asyncio
import fcntl
import hashlib
import json
import os
import re
import shutil
import stat
import sys
import time
from pathlib import Path
from typing import Any, BinaryIO

from agent_runtime.permission_policy import ResolvedPermissions
from agent_runtime.tools.builtins import shell
from agent_runtime.tools.tool import ToolProgressKind
from agent_runtime.user_config import command_state_directory
from agent_runtime.workspace import WorkspaceRuntime

REVISION = "command-worker-v1"
LOG_QUOTA_BYTES = 16 * 1024 * 1024  # per stream, operator resource budget


def freeze_code(workspace: Path) -> Path:
    """Snapshot this trusted installation before running any workspace command.

    Editable installs may themselves be the workspace. Future workers must not
    import code that a previous sandbox command has just edited there.
    """
    source = Path(__file__).resolve().parents[2]
    files = sorted(path for path in source.rglob("*.py") if "__pycache__" not in path.parts)
    digest = hashlib.sha256()
    contents = []
    for path in files:
        if path.is_symlink():
            raise ValueError("worker installation contains a symlink")
        data = path.read_bytes()
        relative = path.relative_to(source)
        digest.update(str(relative).encode() + b"\0" + data)
        contents.append((relative, data))
    root = private_root(workspace)
    bundle = root / ("code_" + digest.hexdigest())
    if bundle.exists():
        check_private(bundle, directory=True)
        return bundle
    temporary = root / ("code_build_" + os.urandom(12).hex())
    temporary.mkdir(mode=0o700)
    try:
        package = temporary / "agent_runtime"
        package.mkdir(mode=0o700)
        for relative, data in contents:
            target = package / relative
            target.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
            target.write_bytes(data)
            target.chmod(0o600)
        try:
            os.rename(temporary, bundle)
        except OSError:
            if not bundle.is_dir():
                raise
    finally:
        shutil.rmtree(temporary, ignore_errors=True)
    return bundle


def private_root(workspace: Path) -> Path:
    root = command_state_directory()
    if root.is_relative_to(workspace.resolve()):
        raise ValueError("command journal must be outside the workspace")
    root.mkdir(mode=0o700, exist_ok=True)
    check_private(root, directory=True)
    return root


def check_private(path: Path, *, directory: bool = False) -> None:
    metadata = path.lstat()
    kind = stat.S_ISDIR if directory else stat.S_ISREG
    if not kind(metadata.st_mode) or metadata.st_uid != os.getuid() or metadata.st_mode & 0o077:
        raise ValueError("unsafe private command state")
    if not directory and metadata.st_nlink != 1:
        raise ValueError("hardlinked private command state")


def job_path(workspace: Path, key: str) -> Path:
    if not re.fullmatch(r"cmd_[0-9a-f]{32}", key):
        raise ValueError("invalid command session ID")
    path = private_root(workspace) / key
    check_private(path, directory=True)
    return path


def read_json(path: Path) -> dict[str, Any]:
    check_private(path)
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC)
    with os.fdopen(descriptor, "r") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError("invalid command state")
    return value


def write_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_name(path.name + "." + os.urandom(8).hex())
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC, 0o600)
    try:
        with os.fdopen(descriptor, "w") as stream:
            json.dump(value, stream, allow_nan=False, separators=(",", ":"))
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)


def lock(path: Path) -> BinaryIO | None:
    # Never replace/unlink lock inodes. close-on-exec keeps sandbox descendants out.
    descriptor = os.open(path, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW | os.O_CLOEXEC, 0o600)
    stream = os.fdopen(descriptor, "r+b")
    check_private(path)
    try:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return stream
    except BlockingIOError:
        stream.close()
        return None


def request_digest(request: shell.RunCommandInput) -> str:
    return hashlib.sha256(request.model_dump_json().encode()).hexdigest()


def policy_value(workspace: WorkspaceRuntime) -> dict[str, Any] | None:
    permissions = workspace.permissions
    return None if permissions is None else permissions.model_dump(mode="json")


def policy_matches(workspace: WorkspaceRuntime, original: dict[str, Any] | None) -> bool:
    try:
        return policy_value(workspace) == original
    except (OSError, ValueError, RuntimeError):
        # Missing/malformed/revoked authority cannot leave an unmonitored job.
        return False


def validate_manifest(workspace: WorkspaceRuntime, manifest: dict[str, Any]) -> shell.RunCommandInput:
    metadata = workspace.root.stat()
    if (manifest["revision"] != REVISION or manifest["workspace"] != str(workspace.root.resolve())
            or manifest["identity"] != [metadata.st_dev, metadata.st_ino]):
        raise ValueError("command workspace identity or worker revision changed")
    request = shell.RunCommandInput.model_validate(manifest["request"])
    if request_digest(request) != manifest["request_digest"]:
        raise ValueError("command request digest mismatch")
    return request


async def execute(path: Path, manifest: dict[str, Any]) -> None:
    request = shell.RunCommandInput.model_validate(manifest["request"])
    permissions = manifest["permissions"]
    resolved = None if permissions is None else ResolvedPermissions.model_validate(permissions)
    workspace = WorkspaceRuntime(Path(manifest["workspace"]), False,
                                 None if resolved is None else lambda: resolved)
    validate_manifest(workspace, manifest)
    entry = shell._CommandStreamBuffer(), shell._CommandStreamBuffer()
    started = time.monotonic()
    last_snapshot = 0.0
    counts = [0, 0]
    truncated = [False, False]
    streams = []
    for name in ("stdout", "stderr"):
        fd = os.open(path / (name + ".log"), os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC, 0o600)
        streams.append(os.fdopen(fd, "wb", buffering=0))

    def snapshot(*, cancelled: bool = False) -> shell.RunCommandOutput:
        out, out_cut = shell._bounded_stream(entry[0].value())
        err, err_cut = shell._bounded_stream(entry[1].value())
        return shell.RunCommandOutput(
            stdout=out, stderr=err, exit_code=-1 if cancelled else None,
            timed_out=False, truncated=out_cut or err_cut, duration_ms=(time.monotonic() - started) * 1000,
            session_id=manifest["session_id"], request_digest=manifest["request_digest"],
            status="cancelled" if cancelled else "running", network_enabled=request.network,
        )

    def capture(kind: ToolProgressKind, chunk: bytes) -> None:
        nonlocal last_snapshot
        index = 0 if kind == ToolProgressKind.STDOUT else 1
        entry[index].append(chunk)
        available = max(0, LOG_QUOTA_BYTES - counts[index])
        streams[index].write(chunk[:available])
        counts[index] += min(available, len(chunk))
        truncated[index] |= len(chunk) > available
        now = time.monotonic()
        if now - last_snapshot >= 0.05:
            write_json(path / "progress.json", snapshot().model_dump(mode="json"))
            last_snapshot = now

    async def authorize_publish() -> None:
        # Never publish under a crashed owner's frozen authorization. Ask the
        # current owner, using a fresh nonce for this exact publication request.
        nonce = os.urandom(16).hex()
        write_json(path / "publish-request.json", {"nonce": nonce})
        while True:
            permission = path / "publish-permit.json"
            if permission.exists() and read_json(permission).get("nonce") == nonce:
                return
            await asyncio.sleep(0.05)

    task = asyncio.create_task(shell._run_command(
        workspace, request, termination_grace_seconds=manifest["grace"],
        capture_output=capture, authorize_publish=authorize_publish,
    ))
    try:
        while not task.done():
            if (path / "cancel.json").exists():
                task.cancel()
                break
            await asyncio.sleep(0.05)
        try:
            result = await task
            result = result.model_copy(update={"session_id": manifest["session_id"],
                                               "request_digest": manifest["request_digest"]})
        except asyncio.CancelledError:
            result = snapshot(cancelled=True)
        except Exception as error:
            result = shell.RunCommandOutput(
                stdout=snapshot().stdout, stderr=str(error), exit_code=-1, timed_out=False,
                truncated=True, duration_ms=(time.monotonic() - started) * 1000,
                sandbox_error="command_session_failed", session_id=manifest["session_id"],
                request_digest=manifest["request_digest"],
            )
        write_json(path / "logs.json", {"truncated": truncated})
        # Result is durable before releasing run.lock. No result + free lock = lost.
        write_json(path / "result.json", result.model_dump(mode="json"))
    finally:
        for stream in streams:
            stream.close()


def main(path_string: str) -> None:
    os.umask(0o077)
    path = Path(path_string)
    check_private(path.parent, directory=True)
    check_private(path, directory=True)
    held = lock(path / "run.lock")
    if held is None:
        return
    with held:
        # A killed worker may have produced effects. Never replay it.
        if (path / "started.json").exists() or (path / "result.json").exists():
            return
        manifest = read_json(path / "manifest.json")
        if not (path / "release.json").exists():
            return
        write_json(path / "started.json", {"revision": REVISION})
        # Paths are supplied only by the trusted parent, not tool arguments.
        shell._SANDBOX_PLATFORM = manifest["platform"]
        shell._SANDBOX_EXEC_PATH = manifest["sandbox_exec"]
        shell._BWRAP_PATH = manifest["bwrap"]
        asyncio.run(execute(path, manifest))


if __name__ == "__main__":
    main(sys.argv[1])
