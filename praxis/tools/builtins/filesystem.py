"""read/write/list_files 内置工具(移植旧 agent_runtime filesystem.py:421-608、:818-856 行为)。

read 支持两种**互斥**模式:字节分页(``offset``/``max_bytes``)与行分页
(``start_line``/``max_lines``,1-based,按 ``\\n`` 切行)。行模式只在行边界切分,
跨 UTF-8 多字节边界不产出替换字符;内部另有字节预算(``_READ_LINE_MAX_BYTES``),
页首超预算单行返回保码点边界的有界预览并把游标推进过该行,避免无前进死路。
二进制探测沿用首块空字节启发式。
list_files 输出确定序的 ``<kind>\\t<bytes>\\t<相对路径>`` 清单:目录 ``d``、文件 ``f``、
符号链接 ``l`` 且一律不跟随;地板目录子树从 ``FLOOR_ENTRIES`` 派生后跳过。
write 用同目录 tmp + os.replace 原子落盘。

ToolResult 占位符约定(供 Task 12 管道):run() 返回的 ``tool_call_id`` 固定为
"pending"(protocol 要求非空),``tool_name`` 为真实工具名;管道在持久化 item 时
替换为真实 call id。

Controller ruling R1:三个工具的 run 均为 async(旧注释"read/write 同步"作废)。

R-T11-2:list_files 在输出每个具体条目前用 ``read_access`` 做执行侧 deny 过滤
(deny 子树整棵不进结果);read 的 target 就是该文件,``decide`` 已拦,行为不变。
"""

from __future__ import annotations

import codecs
import hashlib
import os
import stat
import unicodedata
from collections.abc import Callable, Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any, BinaryIO, cast

from praxis.protocol.floor import FLOOR_ENTRIES, host_private_paths
from praxis.protocol.json import JsonValue
from praxis.protocol.sandbox import ToolExecContext
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
    ToolValidationError,
    json_schema_input,
)
from praxis.tools.builtins.paths import atomic_write_text, checked_path
from praxis.tools.builtins.read_access import is_read_denied, permission_sources

PENDING_TOOL_CALL_ID = "pending"

_READ_SNIFF_BYTES = 8_192
_READ_DEFAULT_MAX_BYTES = 16_000
_READ_MAX_LIMIT = 1_000_000
_READ_DEFAULT_MAX_LINES = 200
_READ_MAX_LINES = 2_000
# 行模式内部字节预算(不暴露为参数):与 max_lines 同时生效,防止巨大文件/超长单行
# 在 async run 里无界阻塞事件循环(超时/取消都送不进去)。超预算时在整行边界收口。
_READ_LINE_MAX_BYTES = 1_000_000
_READ_LINE_CHUNK_BYTES = 65_536
# 页首超预算单行的可见提示:预览必须显式说明"只看到开头",并指向字节模式。
_LINE_BUDGET_HINT = (
    "[line exceeds the line-mode byte budget; preview only — "
    "use read with offset/max_bytes for the remainder]"
)

_LIST_DEFAULT_MAX_ENTRIES = 200
_LIST_MAX_ENTRIES = 2_000
_LIST_MAX_OUTPUT_BYTES = 200_000
_LIST_TIMEOUT_SECONDS = 15.0

# 地板目录名的唯一来源仍是 protocol/floor.py:这里只做"派生投影",不手抄清单。
# 归一化与 floor_reason 的成分匹配一致(NFC + casefold)。
_FLOOR_DIRECTORY_NAMES = frozenset(
    unicodedata.normalize("NFC", entry.name).casefold() for entry in FLOOR_ENTRIES
)
_FLOOR_DIRECTORY_TEXT = "/".join(entry.name for entry in FLOOR_ENTRIES)


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


_read_schema_validator = json_schema_input(
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
            "start_line": {"type": "integer", "minimum": 1},
            "max_lines": {
                "type": "integer",
                "minimum": 1,
                "maximum": _READ_MAX_LINES,
            },
        },
        "required": ("path",),
        "additionalProperties": False,
    }
)


def _validate_read_input(arguments: Mapping[str, JsonValue]) -> Mapping[str, JsonValue]:
    """行/字节两种模式互斥:任一 ``start_line``/``max_lines`` 与任一 ``offset``/``max_bytes``
    同时出现即拒绝(管道把 ToolValidationError 归一为 ``invalid_input`` 结果)。
    """
    args = _read_schema_validator(arguments)
    line_mode = args.get("start_line") is not None or args.get("max_lines") is not None
    byte_mode = args.get("offset") is not None or args.get("max_bytes") is not None
    if line_mode and byte_mode:
        raise ToolValidationError(
            path="$.start_line",
            message="start_line/max_lines cannot be combined with offset/max_bytes",
        )
    return args


def _decode_line(raw: bytes) -> str:
    """解一行(含行尾):先解码再剥 ``\\n`` / CRLF。

    按 ``\\n`` 切分字节流,而 UTF-8 多字节序列不可能包含 0x0A,所以行模式
    永远在完整码点边界解码,不产生 U+FFFD(源文件本身非法字节仍按 replace 兜底)。
    """
    text = raw.decode("utf-8", errors="replace")
    if text.endswith("\n"):
        text = text[:-1]
        if text.endswith("\r"):
            text = text[:-1]
    return text


def _line_metadata(
    size: int,
    *,
    start_line: int,
    lines: Sequence[str],
    truncated: bool,
    line_exceeds_budget: bool = False,
) -> dict[str, Any]:
    line_count = len(lines)
    return {
        "is_binary": False,
        "size_bytes": size,
        "start_line": start_line,
        "end_line": start_line + line_count - 1 if line_count else None,
        "next_start_line": start_line + line_count if truncated else None,
        "line_count": line_count,
        "line_exceeds_budget": line_exceeds_budget,
        "truncated": truncated,
    }


def _binary_result(path_value: str, size: int) -> ToolResult:
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


def _read_line_page(
    workspace: Path,
    path_value: str,
    *,
    start_line: int,
    max_lines: int,
) -> ToolResult:
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
        if b"\x00" in prefix:
            return _binary_result(path_value, size)
        # 嗅探消耗了前缀,行模式从文件头重新开始(等价于字节模式的 seek(offset))
        stream.seek(0)
        _skip_lines(stream, start_line - 1)
        lines: list[str] = []
        bytes_used = 0
        truncated = False
        line_exceeds_budget = False
        while len(lines) < max_lines:
            remaining = _READ_LINE_MAX_BYTES - bytes_used
            if remaining <= 0:
                # 页内字节预算用尽:是否还有后续内容决定 truncated
                truncated = bool(stream.read(1))
                break
            # 有界读行:最多读 remaining+1 字节。
            raw = stream.readline(remaining + 1)
            if not raw:
                break
            if len(raw) > remaining:
                if not lines:
                    # 页首即超预算单行(remaining 此时是完整预算):返回保代码点边界的
                    # 有界预览 + 可见提示,并**把游标推进过这一行**——绝不出现
                    # truncated=True 且 next_start_line==start_line 的无前进死路。
                    preview = _complete_text_page(raw, remaining)
                    lines.append(f"{_decode_line(preview)}\n{_LINE_BUDGET_HINT}")
                    line_exceeds_budget = True
                # 已有整行时停在当前行之前:下一页以完整预算重新开始,可完整返回该行。
                truncated = True
                break
            lines.append(_decode_line(raw))
            bytes_used += len(raw)
        else:
            # 达到 max_lines:多读 1 字节判断是否还有下一行
            truncated = bool(stream.read(1))
    if any("\x00" in line for line in lines):
        return _binary_result(path_value, size)
    # start_line 超界:空内容 + next_start_line=None,是结果不是错误。
    return _result(
        "read",
        content="\n".join(lines),
        status=ResultStatus.OK,
        metadata=_line_metadata(
            size,
            start_line=start_line,
            lines=lines,
            truncated=truncated,
            line_exceeds_budget=line_exceeds_budget,
        ),
        truncated=truncated,
    )


def _skip_lines(stream: BinaryIO, count: int) -> None:
    """跳过 ``count`` 行;按固定块 readline,超长行不会一次性读进内存。"""
    for _ in range(count):
        while True:
            chunk = stream.readline(_READ_LINE_CHUNK_BYTES)
            if not chunk:
                return
            if chunk.endswith(b"\n"):
                break


def create_read_tool(workspace: Path) -> Tool:
    validate_input = _validate_read_input

    async def run(
        arguments: Mapping[str, JsonValue], exec_ctx: ToolExecContext  # 进程内工具:不消费沙箱(R9)
    ) -> ToolResult:
        args = validate_input(arguments)
        path_value = cast("str", args["path"])
        if args.get("start_line") is not None or args.get("max_lines") is not None:
            return _read_line_page(
                workspace,
                path_value,
                start_line=cast("int", args.get("start_line", 1)),
                max_lines=cast("int", args.get("max_lines", _READ_DEFAULT_MAX_LINES)),
            )
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
            return _binary_result(path_value, size)
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
            description=(
                "Read a workspace text file in byte pages (offset/max_bytes, returns "
                "next_offset when truncated) or line pages (start_line/max_lines, returns "
                "next_start_line when truncated). The two modes are mutually exclusive; "
                "line pages are also bounded by an internal byte budget, and a line "
                "larger than that budget is returned as a bounded preview with the "
                "cursor advanced past it (use byte pages for the remainder). Binary "
                "files are reported without their bytes."
            ),
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
                    "start_line": {
                        "type": "integer",
                        "description": "One-based line to start reading (line mode).",
                    },
                    "max_lines": {
                        "type": "integer",
                        "description": (
                            f"Maximum lines to return, 1..{_READ_MAX_LINES} "
                            f"(default {_READ_DEFAULT_MAX_LINES}, line mode)."
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

    async def run(
        arguments: Mapping[str, JsonValue], exec_ctx: ToolExecContext  # 进程内工具:不消费沙箱(R9)
    ) -> ToolResult:
        args = validate_input(arguments)
        content = cast("str", args["content"])
        target = checked_path(workspace, cast("str", args["path"]), write=True)
        target.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_text(target, content)
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


def _is_floor_directory_name(name: str) -> bool:
    """地板目录名判定(唯一来源 FLOOR_ENTRIES;归一化与 floor_reason 一致)。"""
    return unicodedata.normalize("NFC", name).casefold() in _FLOOR_DIRECTORY_NAMES


def _sorted_children(directory: Path) -> tuple[Path, ...]:
    """单层按名字排序;不可读目录静默跳过(读路径对畸形输入是结果不是异常)。"""
    try:
        return tuple(sorted(directory.iterdir(), key=lambda item: item.name))
    except OSError:
        return ()


def _iter_listed_entries(
    directory: Path,
    *,
    include_hidden: bool,
    denied: Callable[[Path], bool] | None = None,
) -> Iterator[tuple[Path, os.stat_result]]:
    """前序深搜清单条目:每层名字排序,符号链接只产出自身、绝不进入。

    每条目只 ``lstat`` 一次(与下降判定共用);不可列条目直接跳过,因此
    ``max_entries`` 的截断/游标判定只看真正可列出的条目。
    地板目录(``FLOOR_ENTRIES`` 派生的 ``.git``/``.praxis``/``.agent_memory``)
    整棵子树跳过;``include_hidden=False`` 时跳过所有点开头条目。
    ``denied``(R-T11-2)对每个具体条目做执行侧 deny 过滤:命中者**连名字都不
    输出**,目录被拒时整棵子树不进入——deny 规则不能因目录级入口而失效。
    """
    stack: list[Iterator[Path]] = [iter(_sorted_children(directory))]
    while stack:
        try:
            child = next(stack[-1])
        except StopIteration:
            stack.pop()
            continue
        name = child.name
        if _is_floor_directory_name(name):
            continue
        if not include_hidden and name.startswith("."):
            continue
        if denied is not None and denied(child):
            continue
        try:
            info = child.lstat()
        except OSError:
            continue
        yield child, info
        if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
            continue
        stack.append(iter(_sorted_children(child)))


def create_list_files_tool(workspace: Path) -> Tool:
    validate_input = json_schema_input(
        {
            "type": "object",
            "properties": {
                "path": {"type": "string", "minLength": 1, "maxLength": 4_096},
                "max_entries": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": _LIST_MAX_ENTRIES,
                },
                "include_hidden": {"type": "boolean"},
            },
            "additionalProperties": False,
        }
    )

    async def run(
        arguments: Mapping[str, JsonValue], exec_ctx: ToolExecContext  # 进程内工具:不消费沙箱
    ) -> ToolResult:
        args = validate_input(arguments)
        path_value = cast("str", args.get("path", "."))
        max_entries = cast("int", args.get("max_entries", _LIST_DEFAULT_MAX_ENTRIES))
        include_hidden = cast("bool", args.get("include_hidden", False))
        target = checked_path(workspace, path_value, allow_root=True)
        if not target.is_dir():
            return _result(
                "list_files",
                content=f"workspace directory not found: {path_value}",
                status=ResultStatus.ERROR,
                metadata={"entry_count": 0, "truncated": False, "next_cursor": None},
                error_code="not_found",
            )
        root = workspace.resolve()
        # S-2:地板/私有目录逐条目判定(不只按目录名剪枝)。
        private_paths = host_private_paths()
        sources = permission_sources(
            exec_ctx.permissions, workspace, private_paths=private_paths
        )
        def denied(path: Path) -> bool:
            return is_read_denied(sources, path, private_paths=private_paths)
        lines: list[str] = []
        next_cursor: str | None = None
        truncated = False
        for child, info in _iter_listed_entries(
            target, include_hidden=include_hidden, denied=denied
        ):
            relative = child.relative_to(root).as_posix()
            # 截断/游标只在真正可列出的条目上判定(不可列条目已在迭代器里跳过)
            if len(lines) == max_entries:
                truncated = True
                next_cursor = relative
                break
            if stat.S_ISLNK(info.st_mode):
                kind = "l"
            elif stat.S_ISDIR(info.st_mode):
                kind = "d"
            else:
                kind = "f"
            lines.append(f"{kind}\t{info.st_size}\t{relative}")
        return _result(
            "list_files",
            content="\n".join(lines),
            status=ResultStatus.OK,
            metadata={
                "entry_count": len(lines),
                "truncated": truncated,
                "next_cursor": next_cursor,
            },
            truncated=truncated,
        )

    def resolve_use(arguments: Mapping[str, JsonValue]) -> ResolvedUse:
        value = arguments.get("path", ".")
        path_value = value if isinstance(value, str) and value else "."
        target = checked_path(workspace, path_value, allow_root=True)
        return ResolvedUse(
            effects=frozenset({ToolEffect.READ_WORKSPACE}),
            targets=(ToolTarget(kind="path", value=str(target)),),
        )

    return Tool(
        definition=ToolDefinition(
            name="list_files",
            description=(
                "List a workspace directory tree in deterministic order: one "
                "<kind>\\t<bytes>\\t<relative path> line per entry (d=directory, f=file, "
                "l=symlink; symlinks are never followed). Hidden entries are skipped "
                f"unless include_hidden=true; protected trees ({_FLOOR_DIRECTORY_TEXT}) "
                "are always skipped. Entries denied by the permission policy "
                "(including symlinks whose real target is outside permitted paths) "
                "are omitted. Use max_entries to bound the listing."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "minLength": 1,
                        "maxLength": 4_096,
                        "description": "Directory path relative to the workspace (default '.').",
                    },
                    "max_entries": {
                        "type": "integer",
                        "description": (
                            f"Maximum entries to return, 1..{_LIST_MAX_ENTRIES} "
                            f"(default {_LIST_DEFAULT_MAX_ENTRIES})."
                        ),
                    },
                    "include_hidden": {
                        "type": "boolean",
                        "description": "Include dot-prefixed entries (default false).",
                    },
                },
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
        interrupt_behavior=InterruptBehavior.CANCEL,
        timeout_seconds=_LIST_TIMEOUT_SECONDS,
        max_model_output_bytes=_LIST_MAX_OUTPUT_BYTES,
    )
