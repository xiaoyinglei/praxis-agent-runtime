"""apply_patch 内置工具:codex 多文件 hunk 格式的**进程内**实现(findings S1 收口)。

旧 agent_runtime 的 codex_patch 自己拼 seatbelt/bwrap profile、直写宿主路径,
并在 Linux 上另建私有视图提交——那是 S1(第二套沙箱/权限构造)的病根。本实现
只做补丁解析与文件内容应用:

- 权限/地板/审批**全部**交给管道:``resolve_use`` 为每个涉及路径(含 Move to
  目标)声明 ``kind="path"`` 的 ``ToolTarget``,于是 decide 的规则、执行时地板
  守卫与 deny 规则逐文件生效;本模块不 import core/store/adapters,也不构造
  任何沙箱或 profile。
- effects 按实际动作给(R-T13-1):含 Add/Update/Move 声明读写;只有含
  ``*** Delete File:`` 才追加 ``DESTRUCTIVE``(删除不可逆,应走强制审批)。
- 跨文件**非事务**(R-T13-3):逐文件顺序应用,失败即停、已改的不回滚,
  返回 ``error_code="patch_failed"`` 并逐字保留旧语义提示;单文件内写入一律
  走 ``paths.atomic_write_text``(同目录 tmp + ``os.replace``),绝不出现半个文件。
- 完整性检查留在工具内且与权限无关(R-T13-4):路径不得含 ``..``、必须在
  workspace 内、路径各级不得是 symlink、目标必须是常规文件、patch ≤30_000
  字符、``<<EOF ... EOF`` 外层包裹剥离、``*** Move to:`` 支持。

解析器 ``parse_patch`` 是纯函数,便于单测穷举畸形输入(空 patch、缺 @@、
hunk 行数不符、未知头等)。畸形输入成为**结果**(``invalid_input``),不炸管道。
"""

from __future__ import annotations

import re
import stat
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast

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

PENDING_TOOL_CALL_ID = "pending"
_MAX_PATCH_CHARS = 30_000
_MAX_CONTEXT_LINE_CHARS = 200
_FAILURE_MESSAGE = (
    "patch failed. Earlier hunks may already have changed files; "
    "inspect the workspace before retrying."
)
_HEREDOC_OPENERS = frozenset({"<<EOF", "<<'EOF'", '<<"EOF"'})
_UNIFIED_HUNK = re.compile(r"^-(\d+)(?:,(\d+))?\s+\+(\d+)(?:,(\d+))?\s+@@$")

type HunkLineKind = Literal[" ", "-", "+"]


@dataclass(frozen=True, slots=True)
class HunkLine:
    kind: HunkLineKind
    text: str


@dataclass(frozen=True, slots=True)
class Hunk:
    """一个 ``@@`` hunk;``label`` 为头后的原文(可为空或函数上下文/行号)。"""

    label: str
    lines: tuple[HunkLine, ...]


@dataclass(frozen=True, slots=True)
class AddFile:
    path: str
    lines: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class DeleteFile:
    path: str


@dataclass(frozen=True, slots=True)
class UpdateFile:
    path: str
    move_to: str | None
    hunks: tuple[Hunk, ...]


type PatchOp = AddFile | DeleteFile | UpdateFile


class _HunkMismatchError(Exception):
    """单个 hunk 的上下文在当前文件里找不到(触发非事务失败)。"""

    def __init__(self, *, hunk_index: int, expected: str) -> None:
        super().__init__(expected)
        self.hunk_index = hunk_index
        self.expected = expected


# ---------------------------------------------------------------------------
# 纯函数解析器
# ---------------------------------------------------------------------------


def _parse_error(message: str) -> ToolValidationError:
    return ToolValidationError(path="$.patch", message=message)


def _patch_lines(text: str) -> tuple[str, ...]:
    """规范化换行、剥 ``<<EOF``/``EOF`` 外壳,并去掉边界空白行。"""
    normalized = text.replace("\r\n", "\n").replace("\r", "\n").strip("\n")
    if not normalized:
        return ()
    lines = normalized.split("\n")
    if (
        len(lines) >= 2
        and lines[0].strip() in _HEREDOC_OPENERS
        and lines[-1].strip().endswith("EOF")
    ):
        lines = lines[1:-1]
    while lines and not lines[0].strip():
        lines.pop(0)
    while lines and not lines[-1].strip():
        lines.pop()
    return tuple(lines)


def _header_path(header: str, marker: str) -> str:
    path = header[len(marker) :].strip()
    if not path:
        raise _parse_error(f"missing path after {marker.strip()}")
    return path


def _finish_hunk(label: str, lines: list[HunkLine]) -> Hunk:
    if not lines:
        raise _parse_error("hunk contains no lines")
    match = _UNIFIED_HUNK.match(label)
    if match is not None:
        old_count = int(match.group(2)) if match.group(2) is not None else 1
        new_count = int(match.group(4)) if match.group(4) is not None else 1
        old_actual = sum(1 for line in lines if line.kind != "+")
        new_actual = sum(1 for line in lines if line.kind != "-")
        if old_actual != old_count or new_actual != new_count:
            raise _parse_error(
                f"hunk line count mismatch for '@@ {label}': expected "
                f"old={old_count}/new={new_count}, got old={old_actual}/new={new_actual}"
            )
    return Hunk(label=label, lines=tuple(lines))


def parse_patch(text: str) -> tuple[PatchOp, ...]:
    """把 codex patch 文本解析为不可变操作元组;任何畸形都抛 ``ToolValidationError``。"""
    lines = _patch_lines(text)
    if not lines:
        raise _parse_error("patch is empty")
    if lines[0].strip() != "*** Begin Patch":
        raise _parse_error("patch must start with *** Begin Patch")
    if lines[-1].strip() != "*** End Patch":
        raise _parse_error("patch must end with *** End Patch")

    ops: list[PatchOp] = []
    index = 1
    end = len(lines) - 1
    while index < end:
        header = lines[index].strip()
        if not header:
            index += 1
            continue

        if header.startswith("*** Add File: "):
            path = _header_path(header, "*** Add File: ")
            index += 1
            body: list[str] = []
            while index < end and not lines[index].startswith("*** "):
                raw = lines[index]
                if not raw.startswith("+"):
                    raise _parse_error(
                        f"add file body lines must start with '+': {raw[:80]!r}"
                    )
                body.append(raw[1:])
                index += 1
            ops.append(AddFile(path=path, lines=tuple(body)))
            continue

        if header.startswith("*** Delete File: "):
            path = _header_path(header, "*** Delete File: ")
            index += 1
            ops.append(DeleteFile(path=path))
            continue

        if header.startswith("*** Update File: "):
            path = _header_path(header, "*** Update File: ")
            index += 1
            move_to: str | None = None
            if index < end and lines[index].startswith("*** Move to: "):
                move_to = _header_path(lines[index].strip(), "*** Move to: ")
                index += 1
            hunks: list[Hunk] = []
            current: list[HunkLine] | None = None
            label = ""
            while index < end and not lines[index].startswith("*** "):
                raw = lines[index]
                if raw.startswith("@@"):
                    if current is not None:
                        hunks.append(_finish_hunk(label, current))
                    label = raw[2:].strip()
                    current = []
                elif current is None:
                    if raw.strip():
                        raise _parse_error("update hunk must start with '@@'")
                elif not raw:
                    current.append(HunkLine(kind=" ", text=""))
                elif raw[0] in (" ", "-", "+"):
                    current.append(
                        HunkLine(kind=cast("HunkLineKind", raw[0]), text=raw[1:])
                    )
                else:
                    raise _parse_error(
                        f"hunk line must start with ' ', '-', or '+': {raw[:80]!r}"
                    )
                index += 1
            if current is not None:
                hunks.append(_finish_hunk(label, current))
            if index < end and lines[index].startswith("*** Move to: "):
                if move_to is not None:
                    raise _parse_error("duplicate *** Move to: header")
                move_to = _header_path(lines[index].strip(), "*** Move to: ")
                index += 1
            if not hunks and move_to is None:
                raise _parse_error("update file requires at least one hunk or a move destination")
            ops.append(UpdateFile(path=path, move_to=move_to, hunks=tuple(hunks)))
            continue

        raise _parse_error(f"unknown patch header: {header[:80]}")

    if not ops:
        raise _parse_error("patch contains no file operations")
    return tuple(ops)


# ---------------------------------------------------------------------------
# 路径围栏与完整性检查(权限判定不在这里)
# ---------------------------------------------------------------------------


def _lexical_target(workspace: Path, raw: str) -> Path:
    """绝对化但**不解析 symlink** 的词法目标;``..`` 与越界在此拒绝。"""
    if not raw:
        raise _parse_error("patch path is empty")
    try:
        candidate = Path(raw)
    except (OSError, ValueError) as exc:
        raise _parse_error(f"patch path is invalid: {exc}") from None
    if ".." in candidate.parts:
        raise _parse_error(
            "patch paths must stay within the workspace without parent traversal"
        )
    base = workspace.resolve()
    target = candidate if candidate.is_absolute() else base / candidate
    if not target.is_relative_to(base) or target == base:
        raise _parse_error(f"patch target is outside the workspace: {raw!r}")
    return target


def _assert_regular_path(workspace: Path, target: Path, *, must_exist: bool) -> None:
    """逐级 lstat:任何一级 symlink 拒绝;已存在目标必须是常规文件。"""
    base = workspace.resolve()
    relative = target.relative_to(base)
    current = base
    for part in relative.parts:
        current = current / part
        try:
            info = current.lstat()
        except FileNotFoundError:
            if current == target and must_exist:
                raise _parse_error(
                    f"patch target does not exist: {relative.as_posix()}"
                ) from None
            return  # 后续成分不存在:不可能藏 symlink
        except OSError as exc:
            raise _parse_error(
                f"patch target cannot be inspected: {exc.strerror or exc}"
            ) from None
        except ValueError as exc:
            raise _parse_error(f"patch path is invalid: {exc}") from None
        if stat.S_ISLNK(info.st_mode):
            raise _parse_error("patch target is a symlink")
        if current != target:
            if not stat.S_ISDIR(info.st_mode):
                raise _parse_error(f"patch target parent is not a directory: {part}")
        elif not stat.S_ISREG(info.st_mode):
            raise _parse_error("patch target is not a regular file")


def _preflight(workspace: Path, ops: Sequence[PatchOp]) -> None:
    """写任何文件之前校验全部目标(完整性失败不留部分修改)。"""
    for op in ops:
        target = _lexical_target(workspace, op.path)
        if isinstance(op, AddFile):
            _assert_regular_path(workspace, target, must_exist=False)
        else:
            _assert_regular_path(workspace, target, must_exist=True)
        if isinstance(op, UpdateFile) and op.move_to is not None:
            destination = _lexical_target(workspace, op.move_to)
            _assert_regular_path(workspace, destination, must_exist=False)


def _path_for_use(workspace: Path, raw: str) -> Path:
    if not raw:
        raise _parse_error("patch path is empty")
    if ".." in Path(raw).parts:
        raise _parse_error(
            "patch paths must stay within the workspace without parent traversal"
        )
    return checked_path(workspace, raw)


# ---------------------------------------------------------------------------
# 应用(单文件原子,跨文件非事务)
# ---------------------------------------------------------------------------


def _find_lines(lines: Sequence[str], needle: Sequence[str], *, start: int) -> int | None:
    limit = len(lines) - len(needle)
    for position in range(start, limit + 1):
        if list(lines[position : position + len(needle)]) == list(needle):
            return position
    return None


def _apply_hunks(text: str, hunks: Sequence[Hunk]) -> str:
    """顺序应用同一文件的全部 hunk;不匹配抛 ``_HunkMismatchError``(不改动文件)。"""
    if text:
        lines = text.split("\n")
        final_newline = text.endswith("\n")
        if final_newline:
            lines.pop()
    else:
        lines = []
        final_newline = False
    cursor = 0
    for hunk_index, hunk in enumerate(hunks):
        old = [line.text for line in hunk.lines if line.kind != "+"]
        new = [line.text for line in hunk.lines if line.kind != "-"]
        if old:
            position = _find_lines(lines, old, start=cursor)
            if position is None:
                raise _HunkMismatchError(hunk_index=hunk_index, expected=old[0])
        else:
            position = len(lines)  # 无上下文的纯追加:落在文件末尾
        lines[position : position + len(old)] = new
        cursor = position + len(new)
    result = "\n".join(lines)
    if final_newline:
        result += "\n"
    return result


def _add_content(lines: Sequence[str]) -> str:
    if not lines:
        return ""
    return "\n".join(lines) + "\n"


def _apply_one(workspace: Path, op: PatchOp) -> tuple[str, tuple[str, ...]]:
    """应用单个操作,返回 (模型可见摘要, 触达的绝对路径)。失败以异常上抛。"""
    if isinstance(op, AddFile):
        target = _lexical_target(workspace, op.path)
        target.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_text(target, _add_content(op.lines))
        return f"added {op.path} ({len(op.lines)} lines)", (str(target),)

    if isinstance(op, DeleteFile):
        target = _lexical_target(workspace, op.path)
        target.unlink()
        return f"deleted {op.path}", (str(target),)

    target = _lexical_target(workspace, op.path)
    after = _apply_hunks(target.read_text(encoding="utf-8"), op.hunks)
    if op.move_to is None:
        atomic_write_text(target, after)
        return f"updated {op.path} ({len(op.hunks)} hunks)", (str(target),)
    destination = _lexical_target(workspace, op.move_to)
    destination.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(destination, after)
    target.unlink()
    return (
        f"moved {op.path} -> {op.move_to} ({len(op.hunks)} hunks)",
        (str(target), str(destination)),
    )


def _bounded_line(value: str, *, limit: int = _MAX_CONTEXT_LINE_CHARS) -> str:
    collapsed = " ".join(value.split())
    return collapsed if len(collapsed) <= limit else f"{collapsed[: limit - 1]}…"


def _metadata(
    applied: Sequence[str],
    failed_file: str | None,
    hunk_index: int | None,
    expected_context_line: str | None,
) -> dict[str, JsonValue]:
    return {
        "applied_files": tuple(applied),
        "failed_file": failed_file,
        "hunk_index": hunk_index,
        "expected_context_line": (
            _bounded_line(expected_context_line)
            if expected_context_line is not None
            else None
        ),
    }


def _result(
    *,
    content: str,
    status: ResultStatus,
    metadata: Mapping[str, JsonValue],
    error_code: str | None = None,
    error_message: str | None = None,
    retryable: bool = False,
) -> ToolResult:
    is_error = status in {ResultStatus.ERROR, ResultStatus.TIMEOUT}
    return ToolResult(
        tool_call_id=PENDING_TOOL_CALL_ID,
        tool_name="apply_patch",
        content=(ToolContentBlock(type="text", data={"text": content}),),
        status=status,
        is_error=is_error,
        error_code=error_code if is_error else None,
        error_message=error_message if is_error else None,
        retryable=retryable if is_error else False,
        metadata=metadata,
    )


def _invalid_result(message: str, *, applied: Sequence[str] = ()) -> ToolResult:
    return _result(
        content=message,
        status=ResultStatus.ERROR,
        error_code="invalid_input",
        metadata=_metadata(applied, None, None, None),
    )


def _failure_result(
    applied: Sequence[str],
    *,
    failed_file: str,
    hunk_index: int | None,
    expected_context_line: str | None,
    detail: str,
) -> ToolResult:
    return _result(
        content=f"{_FAILURE_MESSAGE}\n{failed_file}: {detail}",
        status=ResultStatus.ERROR,
        error_code="patch_failed",
        error_message=_FAILURE_MESSAGE,
        retryable=True,
        metadata=_metadata(applied, failed_file, hunk_index, expected_context_line),
    )


def _op_path(workspace: Path, op: PatchOp) -> str:
    try:
        return str(_lexical_target(workspace, op.path))
    except ToolValidationError:
        return op.path


def _apply_ops(workspace: Path, ops: tuple[PatchOp, ...]) -> ToolResult:
    try:
        _preflight(workspace, ops)
    except ToolValidationError as exc:
        return _invalid_result(exc.message)

    applied: list[str] = []
    summaries: list[str] = []
    for op in ops:
        try:
            summary, touched = _apply_one(workspace, op)
        except _HunkMismatchError as mismatch:
            return _failure_result(
                applied,
                failed_file=_op_path(workspace, op),
                hunk_index=mismatch.hunk_index,
                expected_context_line=mismatch.expected,
                detail=f"hunk {mismatch.hunk_index} expected context line {mismatch.expected!r}",
            )
        except ToolValidationError as exc:
            return _invalid_result(exc.message, applied=applied)
        except UnicodeError:
            return _failure_result(
                applied,
                failed_file=_op_path(workspace, op),
                hunk_index=None,
                expected_context_line=None,
                detail="target is not UTF-8 text",
            )
        except (OSError, ValueError) as exc:
            return _failure_result(
                applied,
                failed_file=_op_path(workspace, op),
                hunk_index=None,
                expected_context_line=None,
                detail=str(exc),
            )
        applied.extend(touched)
        summaries.append(summary)

    return _result(
        content="\n".join(summaries),
        status=ResultStatus.OK,
        metadata=_metadata(applied, None, None, None),
    )


# ---------------------------------------------------------------------------
# Tool 构造
# ---------------------------------------------------------------------------


def create_apply_patch_tool(workspace: Path) -> Tool:
    validate_input = json_schema_input(
        {
            "type": "object",
            "properties": {
                "patch": {
                    "type": "string",
                    "minLength": 1,
                    "maxLength": _MAX_PATCH_CHARS,
                },
            },
            "required": ("patch",),
            "additionalProperties": False,
        }
    )

    async def run(
        arguments: Mapping[str, JsonValue],
        exec_ctx: ToolExecContext,  # 进程内工具:不消费沙箱(R9)
    ) -> ToolResult:
        args = validate_input(arguments)
        patch = args.get("patch")
        if not isinstance(patch, str):
            return _invalid_result("patch must be a string")
        try:
            ops = parse_patch(patch)
        except ToolValidationError as exc:
            return _invalid_result(exc.message)
        return _apply_ops(workspace, ops)

    def resolve_use(arguments: Mapping[str, JsonValue]) -> ResolvedUse:
        patch = arguments.get("patch")
        if not isinstance(patch, str):
            raise _parse_error("patch must be a string")
        ops = parse_patch(patch)
        effects = {ToolEffect.READ_WORKSPACE, ToolEffect.WRITE_WORKSPACE}
        targets: list[ToolTarget] = []
        seen: set[str] = set()
        for op in ops:
            if isinstance(op, DeleteFile):
                effects.add(ToolEffect.DESTRUCTIVE)
            paths = (op.path, op.move_to) if isinstance(op, UpdateFile) else (op.path,)
            for raw in paths:
                if raw is None:
                    continue
                value = str(_path_for_use(workspace, raw))
                if value not in seen:
                    seen.add(value)
                    targets.append(ToolTarget(kind="path", value=value))
        return ResolvedUse(effects=frozenset(effects), targets=tuple(targets))

    return Tool(
        definition=ToolDefinition(
            name="apply_patch",
            description=(
                "Edit workspace files with one Codex-style multi-file patch string: "
                "*** Begin Patch ... *** End Patch. Use *** Add File: path (every content "
                "line prefixed with +), *** Delete File: path, or *** Update File: path "
                "followed by @@ hunks using a leading space for unchanged context, - for "
                "removed lines and + for added lines; an update may include "
                "*** Move to: destination. Context must match the file exactly. Each file "
                "is written atomically, but the patch is not transactional: on failure "
                "earlier files may already be changed. Deletions require explicit approval. "
                "Paths must stay inside the workspace, may not contain '..' or traverse "
                "symlinks, and must name regular files. Maximum patch length is 30,000 "
                "characters."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "patch": {
                        "type": "string",
                        "description": "Codex patch text, at most 30,000 characters.",
                    },
                },
                "required": ("patch",),
                "additionalProperties": False,
            },
        ),
        validate_input=validate_input,
        run=run,
        static_effects=frozenset(
            {ToolEffect.READ_WORKSPACE, ToolEffect.WRITE_WORKSPACE, ToolEffect.DESTRUCTIVE}
        ),
        resolve_use=resolve_use,
        idempotent=False,
        concurrency_safe=False,
        cancellation_mode=CancellationMode.MANAGED_PROCESS,
        interrupt_behavior=InterruptBehavior.CANCEL,
        timeout_seconds=35,
        max_model_output_bytes=50_000,
    )