"""read/write 内置工具(移植旧 agent_runtime filesystem.py:421-608、:818-856 行为,按 brief 简化)。

Phase 1 简化:read 只做字节分页(行模式在阶段 3);二进制探测用首块空字节启发式;
write 用同目录 tmp + os.replace 原子落盘。

ToolResult 占位符约定(供 Task 12 管道):run() 返回的 ``tool_call_id`` 固定为
"pending"(protocol 要求非空),``tool_name`` 为真实工具名;管道在持久化 item 时
替换为真实 call id。

Controller ruling R1:三个工具的 run 均为 async(旧注释"read/write 同步"作废)。
"""

from __future__ import annotations

import codecs
import hashlib
import os
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

from praxis.protocol.json import JsonValue
from praxis.protocol.tools import (
    CancellationMode,
    InterruptBehavior,
    ResolvedUse,
    ResultStatus,
    Tool,
    ToolContentBlock,
    ToolDefinition,
    ToolEffect,
    ToolResult,
    ToolTarget,
    json_schema_input,
)
from praxis.tools.builtins.paths import checked_path

PENDING_TOOL_CALL_ID = "pending"

_READ_SNIFF_BYTES = 8_192
_READ_DEFAULT_MAX_BYTES = 16_000
_READ_MAX_LIMIT = 1_000_000


def _complete_text_page(raw: bytes, limit: int) -> bytes:
    """Keep UTF-8 continuation bytes attached to their code point (ported heuristic)."""
    bounded = raw[:limit]
    decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
    decoder.decode(bounded, final=False)
    pending = decoder.getstate()[0]
    if not pending:
        return bounded
    complete = len(bounded) - len(pending)
    if complete:
        return bounded[:complete]
    for end in range(len(bounded) + 1, min(len(raw), len(bounded) + 3) + 1):
        decoder.decode(raw[end - 1 : end], final=False)
        if not decoder.getstate()[0]:
            return raw[:end]
    return bounded


def _result(
    tool_name: str,
    *,
    content: str,
    status: ResultStatus,
    metadata: Mapping[str, Any],
    truncated: bool = False,
    error_code: str | None = None,
) -> ToolResult:
    is_error = status in {ResultStatus.ERROR, ResultStatus.TIMEOUT}
    return ToolResult(
        tool_call_id=PENDING_TOOL_CALL_ID,
        tool_name=tool_name,
        content=(ToolContentBlock(type="text", data={"text": content}),),
        status=status,
        is_error=is_error,
        error_code=error_code if is_error else None,
        truncated=truncated,
        metadata=metadata,
    )


def create_read_tool(workspace: Path) -> Tool:
    validate_input = json_schema_input(
        {
            "type": "object",
            "properties": {
                "path": {"type": "string", "minLength": 1},
                "offset": {"type": "integer", "minimum": 0},
                "max_bytes": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": _READ_MAX_LIMIT,
                },
            },
            "required": ("path",),
            "additionalProperties": False,
        }
    )

    async def run(arguments: Mapping[str, JsonValue]) -> ToolResult:
        args = validate_input(arguments)
        path_value = cast("str", args["path"])
        offset = cast("int", args.get("offset", 0))
        max_bytes = cast("int", args.get("max_bytes", _READ_DEFAULT_MAX_BYTES))
        target = checked_path(workspace, path_value)
        if not target.is_file():
            return _result(
                "read",
                content=f"workspace file not found: {path_value}",
                status=ResultStatus.ERROR,
                metadata={
                    "is_binary": False,
                    "size_bytes": 0,
                    "next_offset": None,
                    "truncated": False,
                },
                error_code="not_found",
            )
        size = target.stat().st_size
        with open(target, "rb") as stream:
            prefix = stream.read(_READ_SNIFF_BYTES)
            stream.seek(offset)
            raw = stream.read(max_bytes + 4)
        bounded = _complete_text_page(raw, max_bytes)
        truncated = len(raw) > len(bounded)
        is_binary = b"\x00" in prefix or b"\x00" in bounded
        if is_binary:
            return _result(
                "read",
                content=f"binary file, not returned as text: {path_value}",
                status=ResultStatus.ERROR,
                metadata={
                    "is_binary": True,
                    "size_bytes": size,
                    "next_offset": None,
                    "truncated": False,
                },
                error_code="binary_file",
            )
        content = bounded.decode("utf-8", errors="replace")
        next_offset = offset + len(bounded) if truncated else None
        return _result(
            "read",
            content=content,
            status=ResultStatus.OK,
            metadata={
                "is_binary": False,
                "size_bytes": size,
                "next_offset": next_offset,
                "truncated": truncated,
            },
            truncated=truncated,
        )

    def resolve_use(arguments: Mapping[str, JsonValue]) -> ResolvedUse:
        target = checked_path(workspace, cast("str", arguments["path"]))
        return ResolvedUse(
            effects=frozenset({ToolEffect.READ_WORKSPACE}),
            targets=(ToolTarget(kind="path", value=str(target)),),
        )

    return Tool(
        definition=ToolDefinition(
            name="read",
            description="Read a workspace text file in byte pages; returns next_offset when truncated.",
            input_schema={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "File path relative to the workspace."},
                    "offset": {"type": "integer", "description": "Byte offset to start reading (default 0)."},
                    "max_bytes": {
                        "type": "integer",
                        "description": (
                            f"Maximum bytes to return, 1..{_READ_MAX_LIMIT} "
                            f"(default {_READ_DEFAULT_MAX_BYTES})."
                        ),
                    },
                },
                "required": ("path",),
                "additionalProperties": False,
            },
        ),
        validate_input=validate_input,
        run=run,
        static_effects=frozenset({ToolEffect.READ_WORKSPACE}),
        resolve_use=resolve_use,
        idempotent=True,
        concurrency_safe=True,
        cancellation_mode=CancellationMode.COOPERATIVE,
        interrupt_behavior=InterruptBehavior.FINISH_CURRENT,
        timeout_seconds=30,
        max_model_output_bytes=_READ_MAX_LIMIT,
    )


def _atomic_write_text(path: Path, content: str) -> None:
    """同目录 tmp + os.replace(移植旧 _atomic_write_text/atomic_bytes 的原子语义)。"""
    encoded = content.encode("utf-8")
    directory = path.parent
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(directory))
    tmp = Path(tmp_name)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def create_write_tool(workspace: Path) -> Tool:
    validate_input = json_schema_input(
        {
            "type": "object",
            "properties": {
                "path": {"type": "string", "minLength": 1},
                "content": {"type": "string"},
            },
            "required": ("path", "content"),
            "additionalProperties": False,
        }
    )

    async def run(arguments: Mapping[str, JsonValue]) -> ToolResult:
        args = validate_input(arguments)
        content = cast("str", args["content"])
        target = checked_path(workspace, cast("str", args["path"]), write=True)
        target.parent.mkdir(parents=True, exist_ok=True)
        _atomic_write_text(target, content)
        return _result(
            "write",
            content=f"wrote {target.name}",
            status=ResultStatus.OK,
            metadata={
                "path": str(target),
                "bytes_written": len(content.encode("utf-8")),
                "sha256": hashlib.sha256(content.encode("utf-8")).hexdigest(),
            },
        )

    def resolve_use(arguments: Mapping[str, JsonValue]) -> ResolvedUse:
        target = checked_path(workspace, cast("str", arguments["path"]), write=True)
        return ResolvedUse(
            effects=frozenset({ToolEffect.WRITE_WORKSPACE}),
            targets=(ToolTarget(kind="path", value=str(target)),),
        )

    return Tool(
        definition=ToolDefinition(
            name="write",
            description="Atomically write a text file inside the workspace (creates parent directories).",
            input_schema={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "File path relative to the workspace."},
                    "content": {"type": "string", "description": "Full UTF-8 file content."},
                },
                "required": ("path", "content"),
                "additionalProperties": False,
            },
        ),
        validate_input=validate_input,
        run=run,
        static_effects=frozenset({ToolEffect.WRITE_WORKSPACE}),
        resolve_use=resolve_use,
        idempotent=True,
        concurrency_safe=False,
        cancellation_mode=CancellationMode.COOPERATIVE,
        interrupt_behavior=InterruptBehavior.FINISH_CURRENT,
        timeout_seconds=30,
        max_model_output_bytes=4_096,
    )
