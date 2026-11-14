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
- ``*** Move to:`` 的目标必须**不同于源路径且尚不存在**(自移动会先写后删、
  目标已存在会静默覆盖;二者都是数据丢失),违者 ``invalid_input``。写目标前
  再复核一次(TOCTOU)。Update 目标有 64 MiB 准入上限,保留原文件换行风格
  (CRLF 不被整文件改写)。

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
_MAX_UPDATE_BYTES = 64 * 1024 * 1024
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


class _MoveCleanupError(Exception):
    """Move 的目标已写成,但删除源文件失败(如实报告"已移动未删源")。"""

    def __init__(self, *, destination: str, error: OSError) -> None:
        super().__init__(str(error))
        self.destination = destination
        self.error = error


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


def _hunk_count(value: str | None) -> int:
    """统一 diff 计数;超长数字触发 CPython int 上限时必须仍是 ToolValidationError。"""
    if value is None:
        return 1
    try:
        return int(value)
    except ValueError:
        raise _parse_error(f"hunk line count is not a valid integer: {value[:32]}") from None


def _finish_hunk(label: str, lines: list[HunkLine]) -> Hunk:
    if not lines:
        raise _parse_error("hunk contains no lines")
    match = _UNIFIED_HUNK.match(label)
    if match is not None:
        old_count = _hunk_count(match.group(2))
        new_count = _hunk_count(match.group(4))
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


def _fence_target(workspace: Path, raw: str) -> Path:
    """空路径/``..``/越界的**唯一**校验;返回不解析 symlink 的词法绝对路径。"""
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


def _path_for_use(workspace: Path, raw: str) -> Path:
    """resolve_use 的 target:同一份词法围栏 + 共享 ``checked_path`` 的真实路径检查。"""
    return checked_path(workspace, str(_fence_target(workspace, raw)))


def _validate_move(workspace: Path, source: Path, move_to: str) -> Path:
    """Move 策略:目标必须不同于源、且尚不存在(否则先写后删/静默覆盖)。"""
    destination = _fence_target(workspace, move_to)
    if destination == source:
        raise _parse_error("move destination must differ from the source path")
    if destination.exists() or destination.is_symlink():
        raise _parse_error(f"move destination already exists: {move_to}")
    return destination


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
            if must_exist:
                raise _parse_error(
                    "patch target does not exist (or a parent directory is missing): "
                    f"{relative.as_posix()}"
                ) from None
            return  # 目标不必存在:后续成分不存在,不可能藏 symlink
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


def _assert_update_size(target: Path) -> None:
    try:
        size = target.stat().st_size
    except OSError as exc:
        raise _parse_error(
            f"patch target cannot be inspected: {exc.strerror or exc}"
        ) from None
    if size > _MAX_UPDATE_BYTES:
        raise _parse_error(
            f"update target is too large ({size} bytes; limit {_MAX_UPDATE_BYTES}); "
            "use read/search for targeted inspection instead"
        )


def _preflight(workspace: Path, ops: Sequence[PatchOp]) -> None:
    """写任何文件之前校验全部目标(完整性失败不留部分修改)。"""
    for op in ops:
        target = _fence_target(workspace, op.path)
        if isinstance(op, UpdateFile):
            _assert_regular_path(workspace, target, must_exist=True)
            _assert_update_size(target)
            if op.move_to is not None:
                destination = _validate_move(workspace, target, op.move_to)
                _assert_regular_path(workspace, destination, must_exist=False)
        elif isinstance(op, DeleteFile):
            _assert_regular_path(workspace, target, must_exist=True)
        else:
            _assert_regular_path(workspace, target, must_exist=False)


# ---------------------------------------------------------------------------
# 应用(单文件原子,跨文件非事务)
# ---------------------------------------------------------------------------


def _find_lines(lines: Sequence[str], needle: Sequence[str], *, start: int) -> int | None:
    limit = len(lines) - len(needle)
    for position in range(start, limit + 1):
        if list(lines[position : position + len(needle)]) == list(needle):
            return position
    return None


def _split_terminated(text: str) -> list[tuple[str, str]]:
    """切成 (行内容, 原始终止符) 对;终止符为 ``\\r\\n`` / ``\\n`` / ``\\r`` / ``""``。

    只认这三种换行,Unicode 的其它行分隔符不是文本文件换行约定,按内容保留。
    """
    entries: list[tuple[str, str]] = []
    start = 0
    index = 0
    length = len(text)
    while index < length:
        char = text[index]
        if char == "\r":
            if index + 1 < length and text[index + 1] == "\n":
                entries.append((text[start:index], "\r\n"))
                index += 2
            else:
                entries.append((text[start:index], "\r"))
                index += 1
            start = index
        elif char == "\n":
            entries.append((text[start:index], "\n"))
            index += 1
            start = index
        else:
            index += 1
    if start < length:
        entries.append((text[start:], ""))
    return entries


def _default_ending(endings: Sequence[str]) -> str:
    for candidate in ("\r\n", "\n", "\r"):
        if candidate in endings:
            return candidate
    return "\n"


def _insert_ending(
    entries: Sequence[tuple[str, str]], position: int, old_length: int
) -> str:
    """新增行沿用匹配位置/前一行/后一行的终止符,都不适用时取全文最常见者。"""
    endings = [ending for _, ending in entries]
    if old_length and endings[position]:
        return endings[position]
    if position > 0 and endings[position - 1]:
        return endings[position - 1]
    after = position + old_length
    if after < len(endings) and endings[after]:
        return endings[after]
    return _default_ending(endings)


def _apply_hunks(text: str, hunks: Sequence[Hunk]) -> str:
    """顺序应用同一文件的全部 hunk;不匹配抛 ``_HunkMismatchError``(不改动文件)。

    匹配在**归一化行内容**上进行(逐行剥掉自己的终止符),落盘时逐行还原
    ``\\r\\n`` / ``\\n`` / ``\\r`` / 无尾终止符:未触及的行逐字节不变,混合换行
    文件只改 hunk 所在行,纯 CRLF 文件不会被整文件改写。
    """
    entries = _split_terminated(text)
    had_final_newline = bool(entries) and entries[-1][1] != ""
    cursor = 0
    for hunk_index, hunk in enumerate(hunks):
        contents = [content for content, _ in entries]
        old = [line.text for line in hunk.lines if line.kind != "+"]
        if old:
            position = _find_lines(contents, old, start=cursor)
            if position is None:
                raise _HunkMismatchError(hunk_index=hunk_index, expected=old[0])
        else:
            position = len(entries)  # 无上下文的纯追加:落在文件末尾
        old_endings = [entries[i][1] for i in range(position, position + len(old))]
        insert_ending = _insert_ending(entries, position, len(old))
        rebuilt: list[tuple[str, str]] = []
        old_index = 0
        for line in hunk.lines:
            if line.kind == "+":
                rebuilt.append((line.text, insert_ending))
            elif line.kind == " ":
                # 上下文行保留它自己的原始终止符
                rebuilt.append((line.text, old_endings[old_index]))
                old_index += 1
            else:  # "-":删除,消费一个旧行
                old_index += 1
        entries[position : position + len(old)] = rebuilt
        cursor = position + len(rebuilt)

    if entries:
        default = _default_ending([ending for _, ending in entries])
        # 非末行必须有终止符(否则拼接会黏行);末行沿用原文件"有无尾换行"。
        for index in range(len(entries) - 1):
            if not entries[index][1]:
                entries[index] = (entries[index][0], default)
        last_content, last_ending = entries[-1]
        if had_final_newline and not last_ending:
            entries[-1] = (last_content, default)
        elif not had_final_newline and last_ending:
            entries[-1] = (last_content, "")
    return "".join(content + ending for content, ending in entries)


def _add_content(lines: Sequence[str]) -> str:
    if not lines:
        return ""
    return "\n".join(lines) + "\n"


def _read_update_text(target: Path) -> str:
    """有界读取 Update 目标(保留原换行;超限走结果而不是 OOM/长时间阻塞)。"""
    _assert_update_size(target)
    with target.open("r", encoding="utf-8", newline="") as stream:
        return stream.read()


def _apply_one(workspace: Path, op: PatchOp) -> tuple[str, tuple[str, ...]]:
    """应用单个操作,返回 (模型可见摘要, 触达的绝对路径)。失败以异常上抛。"""
    if isinstance(op, AddFile):
        target = _fence_target(workspace, op.path)
        _assert_regular_path(workspace, target, must_exist=False)  # 写前复核(TOCTOU)
        target.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_text(target, _add_content(op.lines))
        return f"added {op.path} ({len(op.lines)} lines)", (str(target),)

    if isinstance(op, DeleteFile):
        target = _fence_target(workspace, op.path)
        _assert_regular_path(workspace, target, must_exist=True)  # 写前复核(TOCTOU)
        target.unlink()
        return f"deleted {op.path}", (str(target),)

    target = _fence_target(workspace, op.path)
    _assert_regular_path(workspace, target, must_exist=True)  # 写前复核(TOCTOU)
    before = _read_update_text(target)
    after = _apply_hunks(before, op.hunks)
    if op.move_to is None:
        atomic_write_text(target, after)
        return f"updated {op.path} ({len(op.hunks)} hunks)", (str(target),)
    # 写 dest 前再复核一次:自移动/目标已存在都在这里拒绝
    destination = _validate_move(workspace, target, op.move_to)
    _assert_regular_path(workspace, destination, must_exist=False)
    destination.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(destination, after)
    try:
        target.unlink()
    except OSError as exc:
        raise _MoveCleanupError(destination=str(destination), error=exc) from exc
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
    *,
    move_incomplete: bool = False,
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
        "move_incomplete": move_incomplete,
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
    move_incomplete: bool = False,
) -> ToolResult:
    return _result(
        content=f"{_FAILURE_MESSAGE}\n{failed_file}: {detail}",
        status=ResultStatus.ERROR,
        error_code="patch_failed",
        error_message=_FAILURE_MESSAGE,
        retryable=True,
        metadata=_metadata(
            applied,
            failed_file,
            hunk_index,
            expected_context_line,
            move_incomplete=move_incomplete,
        ),
    )


def _op_path(workspace: Path, op: PatchOp) -> str:
    try:
        return str(_fence_target(workspace, op.path))
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
        except _MoveCleanupError as cleanup:
            # dest 已写成但删源失败:如实把 dest 记进 applied_files,并标记 move_incomplete
            return _failure_result(
                [*applied, cleanup.destination],
                failed_file=_op_path(workspace, op),
                hunk_index=None,
                expected_context_line=None,
                detail=(
                    f"moved content to {cleanup.destination} but could not remove the "
                    f"source: {cleanup.error}"
                ),
                move_incomplete=True,
            )
        except ToolValidationError as exc:
            if applied:
                # 已经有文件落盘:此时任何校验失败都是"改了一半",不是纯输入拒绝
                return _failure_result(
                    applied,
                    failed_file=_op_path(workspace, op),
                    hunk_index=None,
                    expected_context_line=None,
                    detail=exc.message,
                )
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
            if isinstance(op, UpdateFile) and op.move_to is not None:
                _validate_move(
                    workspace, _fence_target(workspace, op.path), op.move_to
                )
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
                "*** Move to: destination. Context must match the file exactly; a move "
                "destination must differ from its source and must not already exist. "
                "Each file is written atomically, but the patch is not transactional: on "
                "failure earlier files may already be changed. Deletions require explicit "
                "approval. Paths must stay inside the workspace, may not contain '..' or "
                "traverse symlinks, and must name regular files; update targets are "
                "limited to 64 MiB. Maximum patch length is 30,000 characters."
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
