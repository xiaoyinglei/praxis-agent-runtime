"""Cancellable search worker: authorized bounded reads, ripgrep matching, bounded JSON.

Only the immutable permission snapshot and request cross the process boundary.
No shell, PATH lookup, user ripgrep configuration, or Python regex evaluation is used.
"""

from __future__ import annotations

import base64
import fnmatch
import json
import os
import resource
import signal
import stat
import subprocess
import sys
import tempfile
from collections.abc import Iterator
from pathlib import Path
from types import FrameType
from typing import Any

from agent_runtime.permission_policy import ResolvedPermissions
from agent_runtime.workspace import WorkspaceRuntime

_MAX_FILE_BYTES = 2_000_000
_MAX_TOTAL_BYTES = 64_000_000
_MAX_FILES = 20_000
_MAX_OUTPUT_BYTES = 1_500_000
_MAX_BACKEND_BYTES = 8_000_000
_MAX_LINE_CHARS = 500
_IGNORED = frozenset(
    {
        ".agent_memory",
        ".git",
        ".mypy_cache",
        ".nox",
        ".pytest_cache",
        ".praxis",
        ".rag",
        ".ruff_cache",
        ".tox",
        ".venv",
        "__pycache__",
        "node_modules",
    }
)


def _has_symlink(path: Path, root: Path) -> bool:
    while path != root and path != path.parent:
        if path.is_symlink():
            return True
        path = path.parent
    return path.is_symlink()


class _TraversalLimitError(Exception):
    pass


def _files(workspace: WorkspaceRuntime, value: str, glob: str | None) -> Iterator[Path]:
    lexical = workspace.resolve_path(value)
    target = workspace.ensure_readable(lexical)
    if lexical.is_relative_to(workspace.root) and ".agent_memory" in lexical.relative_to(workspace.root).parts:
        raise PermissionError("agent memory is not searchable")
    if _has_symlink(lexical, workspace.root):
        return
    pending = [target]
    enumerated = 0
    while pending:
        path = pending.pop()
        if path.is_symlink() or not workspace.is_readable(path):
            continue
        if path.is_dir():
            children = []
            try:
                with os.scandir(path) as entries:
                    for entry in entries:
                        enumerated += 1
                        if enumerated > _MAX_FILES:
                            raise _TraversalLimitError
                        child = Path(entry.path)
                        if entry.name not in _IGNORED:
                            children.append(child)
            except OSError:
                raise _TraversalLimitError from None
            pending.extend(sorted(children, reverse=True))
        elif path.is_file():
            label = workspace.path_label(path)
            if glob is None or fnmatch.fnmatch(path.name, glob) or fnmatch.fnmatch(label, glob):
                yield path


def _read_file(path: Path, limit: int) -> bytes:
    # Open each component without following links. Replacing an enumerated parent
    # with a symlink cannot widen the actual read to an unauthorized subtree.
    descriptor = os.open(path.anchor, os.O_RDONLY | os.O_DIRECTORY)
    try:
        for component in path.parts[1:-1]:
            following = os.open(component, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = following
        file_descriptor = os.open(path.name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=descriptor)
        with os.fdopen(file_descriptor, "rb") as handle:
            if not stat.S_ISREG(os.fstat(handle.fileno()).st_mode):
                raise PermissionError("search target is not a regular file")
            return handle.read(limit + 1)
    finally:
        os.close(descriptor)


def _text(value: dict[str, str]) -> str:
    return value["text"] if "text" in value else base64.b64decode(value["bytes"]).decode("utf-8", "replace")


def _backend(
    binary: str,
    pattern: str,
    regex: bool,
    raw: bytes,
    limit: int,
    context: int,
) -> tuple[list[dict[str, Any]], bool] | dict[str, str]:
    arguments = [
        binary,
        "--no-config",
        "--json",
        "--text",
        "--encoding",
        "utf-8",
        "--no-mmap",
        "--max-count",
        str(limit + 1),
        "--context",
        str(context),
    ]
    if regex:
        # Recognize CRLF line anchors without changing literal CR matching.
        arguments.append("--crlf")
    else:
        arguments.append("--fixed-strings")
    arguments.extend(("--regexp", pattern, "--", "-"))

    def limit_output() -> None:
        resource.setrlimit(resource.RLIMIT_FSIZE, (_MAX_BACKEND_BYTES, _MAX_BACKEND_BYTES))

    # JSON can contain a submatch for every character of a long line. Cap the
    # native writer too, before decoding, rather than trusting communicate() to
    # collect an arbitrarily large JSON record in memory.
    with tempfile.TemporaryFile() as output_file, tempfile.TemporaryFile() as error_file:
        process = subprocess.Popen(
            arguments, stdin=subprocess.PIPE, stdout=output_file, stderr=error_file, preexec_fn=limit_output
        )
        try:
            process.communicate(raw)
            output_file.seek(0)
            output = output_file.read(_MAX_BACKEND_BYTES)
            error_file.seek(0)
            errors = error_file.read(2000)
            capped = len(output) >= _MAX_BACKEND_BYTES or process.returncode == -signal.SIGXFSZ
            if process.returncode not in (0, 1) and not capped:
                return {"error_code": "invalid_arguments", "error_message": errors.decode("utf-8", "replace")[:2000]}
            events = []
            for line in output.splitlines():
                try:
                    events.append(json.loads(line))
                except (ValueError, UnicodeError):
                    capped = True
                    break
            return events, capped
        finally:
            if process.poll() is None:
                process.kill()
            process.wait()


def _search(payload: dict[str, Any]) -> dict[str, Any]:
    permissions = ResolvedPermissions.model_validate(payload["permissions"]) if payload["permissions"] else None
    workspace = WorkspaceRuntime(
        root=Path(payload["root"]),
        is_temporary=False,
        permission_provider=(lambda: permissions) if permissions else None,
    )
    request = payload["request"]
    matches: list[dict[str, Any]] = []
    truncated = False
    searched_file_path = None
    total_bytes = 0
    output_bytes = 0
    context = request["context_lines"]
    limit = request["max_results"]
    target = workspace.ensure_readable(workspace.resolve_path(request["path"]))
    exact = target.is_file()
    # Validate even if there are no candidate text files. Rust's regex syntax is
    # the public contract; unsupported constructs fail explicitly in the backend.
    validated = _backend(payload["binary"], request["pattern"], payload["regex"], b"", 1, 0)
    if isinstance(validated, dict):
        return validated
    try:
        for path in _files(workspace, request["path"], request["glob"]):
            if total_bytes >= _MAX_TOTAL_BYTES:
                truncated = True
                break
            workspace.ensure_readable(path)
            remaining = min(_MAX_FILE_BYTES, _MAX_TOTAL_BYTES - total_bytes)
            try:
                raw = _read_file(path, remaining)
            except (OSError, ValueError):
                # A file disappearing or being replaced is an incomplete observation.
                truncated = True
                continue
            total_bytes += len(raw)
            if b"\x00" in raw:
                continue
            label = workspace.path_label(path)
            if exact:
                searched_file_path = label
            if len(raw) > remaining:
                raw = raw[:remaining]
                truncated = True
            raw = raw.decode("utf-8", "replace").encode("utf-8")
            backend_result = _backend(
                payload["binary"], request["pattern"], payload["regex"], raw, limit - len(matches), context
            )
            if isinstance(backend_result, dict):
                return backend_result
            events, backend_truncated = backend_result
            truncated = truncated or backend_truncated
            lines: dict[int, str] = {}
            found = []
            for event in events:
                if event["type"] in {"match", "context"}:
                    data = event["data"]
                    lines[data["line_number"]] = _text(data["lines"]).rstrip("\r\n")
                    if event["type"] == "match":
                        found.append(data)
            for data in found:
                if len(matches) >= limit:
                    truncated = True
                    break
                number = data["line_number"]
                line = lines[number]
                span = data["submatches"][0]
                encoded_line = line.encode("utf-8")
                match = {
                    "file_path": label,
                    "line_number": number,
                    "line_content": line[:_MAX_LINE_CHARS],
                    "match_start": len(encoded_line[: span["start"]].decode("utf-8", "replace")),
                    "match_end": len(encoded_line[: span["end"]].decode("utf-8", "replace")),
                    "context_before": [
                        lines[n][:_MAX_LINE_CHARS] for n in range(max(1, number - context), number) if n in lines
                    ],
                    "context_after": [
                        lines[n][:_MAX_LINE_CHARS] for n in range(number + 1, number + context + 1) if n in lines
                    ],
                }
                size = len(json.dumps(match).encode())
                if output_bytes + size > _MAX_OUTPUT_BYTES:
                    truncated = True
                    return {
                        "matches": matches,
                        "total_matches": len(matches),
                        "truncated": truncated,
                        "searched_file_path": searched_file_path,
                    }
                output_bytes += size
                matches.append(match)
                if any(
                    len(lines[n]) > _MAX_LINE_CHARS
                    for n in range(max(1, number - context), number + context + 1)
                    if n in lines
                ):
                    truncated = True
            if len(matches) >= limit and (len(found) > limit or not exact):
                truncated = True
                break
    except _TraversalLimitError:
        truncated = True
    return {
        "matches": matches,
        "total_matches": len(matches),
        "truncated": truncated,
        "searched_file_path": searched_file_path,
    }


def main() -> None:
    def terminate(_signum: int, _frame: FrameType | None) -> None:
        raise SystemExit(143)

    signal.signal(signal.SIGTERM, terminate)
    payload = json.load(sys.stdin)
    result = _search(payload)
    sys.stdout.write(json.dumps(result))


if __name__ == "__main__":
    main()
