"""search_text 的可取消 worker:授权范围内的有界读取 + ripgrep 匹配(Task 11)。

由 ``search.py`` 用 ``sys.executable -I -c <code-string>`` 启动
(``start_new_session=True``、``env={"PATH": os.defpath}``、``cwd="/"``)。
只有不可变权限快照与请求跨进程边界;不使用 shell、不做 PATH 查找、不读用户
ripgrep 配置、不用 Python 正则求值(rg 的正则是公开契约)。

移植语义(只读参照 agent_runtime/tools/builtins/search_worker.py,禁止 import):
- 逐文件读取用 ``O_NOFOLLOW`` 且逐级目录 fd 打开,父目录被替换为 symlink 也不能
  把读取面扩大到未授权子树;非常规文件拒;
- symlink 文件/目录一律跳过;``.agent_memory``/``.praxis`` 等 deny 地板条目
  由 ``FLOOR_ENTRIES`` 派生并从遍历中剪除,目标本身落在 deny 子树时显式失败;
- 每个文件作为 stdin 交给打包的 rg,固定参数
  ``--no-config --json --text --encoding utf-8 --no-mmap --max-count <limit+1>``
  (regex 加 ``--crlf``,literal 加 ``--fixed-strings``);``RLIMIT_FSIZE`` 在解码前
  封顶原生输出;
- 阈值常量集中在本模块,``search.py`` 原样写入 ``metadata["limits"]``。
- R-T11-2:遍历触达每个条目时复用 ``protocol.policy.path_access`` 做执行侧
  deny 过滤,被拒文件跳过并计入 ``skipped_denied``(deny 子树整棵剪除)。
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
import unicodedata
from collections.abc import Iterator, Sequence
from pathlib import Path
from types import FrameType
from typing import Any

from praxis.protocol.floor import FLOOR_ENTRIES
from praxis.protocol.policy import PermissionPolicy, ResolvedPolicy, resolve_policy
from praxis.tools.builtins.read_access import is_read_denied

_MAX_FILE_BYTES = 2_000_000
_MAX_TOTAL_BYTES = 64_000_000
_MAX_FILES = 20_000
_MAX_OUTPUT_BYTES = 1_500_000
_MAX_BACKEND_BYTES = 8_000_000
_MAX_LINE_CHARS = 500

LIMITS: dict[str, int] = {
    "max_file_bytes": _MAX_FILE_BYTES,
    "max_total_bytes": _MAX_TOTAL_BYTES,
    "max_files": _MAX_FILES,
    "max_output_bytes": _MAX_OUTPUT_BYTES,
    "max_backend_bytes": _MAX_BACKEND_BYTES,
    "max_line_chars": _MAX_LINE_CHARS,
}

# 地板名由 FLOOR_ENTRIES 派生(NFC + casefold,与 floor_reason 一致);
# 其余是纯性能目录常量(旧 _IGNORED 的非地板部分)。
_FLOOR_DIRECTORY_NAMES = frozenset(
    unicodedata.normalize("NFC", entry.name).casefold() for entry in FLOOR_ENTRIES
)
_DENY_FLOOR_NAMES = tuple(
    entry.name for entry in FLOOR_ENTRIES if entry.access == "deny"
)
_PERFORMANCE_IGNORED = frozenset(
    {
        ".mypy_cache",
        ".nox",
        ".pytest_cache",
        ".rag",
        ".ruff_cache",
        ".tox",
        ".venv",
        "__pycache__",
        "node_modules",
    }
)


class _TraversalLimitError(Exception):
    pass


def _is_ignored(name: str) -> bool:
    return (
        name in _PERFORMANCE_IGNORED
        or unicodedata.normalize("NFC", name).casefold() in _FLOOR_DIRECTORY_NAMES
    )


def _under_deny_floor(root: Path, target: Path) -> str:
    """目标本身落在 deny 地板子树 → 返回拒绝消息;"" = 通过。"""
    try:
        parts = target.relative_to(root).parts
    except ValueError:
        return ""
    normalized = {unicodedata.normalize("NFC", part).casefold() for part in parts}
    for name in _DENY_FLOOR_NAMES:
        if unicodedata.normalize("NFC", name).casefold() in normalized:
            if name == ".agent_memory":
                return "agent memory is not searchable"
            return f"{name} is not searchable"
    return ""


def _policy_sources(payload: dict[str, Any], root: Path) -> tuple[ResolvedPolicy, ...]:
    """worker 侧重建权限来源(policy + ceilings);空 = 无快照(单测直跑)。"""
    raw = payload.get("policies")
    if not raw:
        return ()
    return tuple(
        resolve_policy(PermissionPolicy.model_validate(entry), root) for entry in raw
    )


class _Counters:
    __slots__ = ("skipped_denied",)

    def __init__(self) -> None:
        self.skipped_denied = 0


def _files(
    root: Path,
    target: Path,
    glob: str | None,
    sources: tuple[ResolvedPolicy, ...],
    counters: _Counters,
) -> Iterator[tuple[Path, str]]:
    """确定性前序深搜(子项排序后逆序入栈),逐条目执行 deny 过滤。"""
    pending = [target]
    enumerated = 0
    while pending:
        path = pending.pop()
        if path.is_symlink():
            continue
        if is_read_denied(sources, path):
            counters.skipped_denied += 1
            continue
        if path.is_dir():
            children: list[Path] = []
            try:
                with os.scandir(path) as entries:
                    for entry in entries:
                        enumerated += 1
                        if enumerated > _MAX_FILES:
                            raise _TraversalLimitError
                        if _is_ignored(entry.name):
                            continue
                        children.append(Path(entry.path))
            except OSError:
                raise _TraversalLimitError from None
            pending.extend(sorted(children, reverse=True))
        elif path.is_file():
            label = path.relative_to(root).as_posix()
            if glob is None or fnmatch.fnmatch(path.name, glob) or fnmatch.fnmatch(label, glob):
                yield path, label


def _read_file(path: Path, limit: int) -> bytes:
    """逐级目录 fd + O_NOFOLLOW 打开;父目录被换成 symlink 也不能扩大读取面。"""
    descriptor = os.open(path.anchor, os.O_RDONLY | os.O_DIRECTORY)
    try:
        for component in path.parts[1:-1]:
            following = os.open(
                component, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=descriptor
            )
            os.close(descriptor)
            descriptor = following
        file_descriptor = os.open(
            path.name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=descriptor
        )
        with os.fdopen(file_descriptor, "rb") as handle:
            if not stat.S_ISREG(os.fstat(handle.fileno()).st_mode):
                raise PermissionError("search target is not a regular file")
            return handle.read(limit + 1)
    finally:
        os.close(descriptor)


def _text(value: object) -> str:
    """rg 的行载荷:畸形(非 dict / 非 str / 坏 base64)按空串处理而非炸穿。"""
    if not isinstance(value, dict):
        return ""
    text = value.get("text")
    if isinstance(text, str):
        return text
    encoded = value.get("bytes")
    if not isinstance(encoded, str):
        return ""
    try:
        return base64.b64decode(encoded).decode("utf-8", "replace")
    except (ValueError, TypeError):
        return ""


def _backend(
    binary: str,
    pattern: str,
    regex: bool,
    raw: bytes,
    limit: int,
    context: int,
) -> tuple[list[dict[str, Any]], bool] | dict[str, Any]:
    """把一份文件字节交给打包 rg,解析 JSON 事件;参数非法 → error dict。"""
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
        # 识别 CRLF 行锚点,同时不改变 literal 模式对 CR 的精确匹配。
        arguments.append("--crlf")
    else:
        arguments.append("--fixed-strings")
    arguments.extend(("--regexp", pattern, "--", "-"))

    def limit_output() -> None:
        resource.setrlimit(resource.RLIMIT_FSIZE, (_MAX_BACKEND_BYTES, _MAX_BACKEND_BYTES))

    # JSON 可能为长行的每个字符产出 submatch;在解码前就用 RLIMIT_FSIZE 封顶
    # 原生写出,而不是相信 communicate() 能收下任意大的 JSON 记录。
    with tempfile.TemporaryFile() as output_file, tempfile.TemporaryFile() as error_file:
        process = subprocess.Popen(
            arguments,
            stdin=subprocess.PIPE,
            stdout=output_file,
            stderr=error_file,
            preexec_fn=limit_output,
        )
        try:
            process.communicate(raw)
            output_file.seek(0)
            output = output_file.read(_MAX_BACKEND_BYTES)
            error_file.seek(0)
            errors = error_file.read(2000)
            capped = len(output) >= _MAX_BACKEND_BYTES or process.returncode == -signal.SIGXFSZ
            if process.returncode not in (0, 1) and not capped:
                return {
                    "error_code": "invalid_input",
                    "error_message": errors.decode("utf-8", "replace")[:2000],
                    # 标记这是 rg stderr:search 只把它放 metadata,不入 content。
                    "stderr": True,
                }
            events: list[dict[str, Any]] = []
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
    root = Path(payload["root"])
    sources = _policy_sources(payload, root)
    request = payload["request"]
    matches: list[dict[str, Any]] = []
    truncated = False
    searched_file_path: str | None = None
    total_bytes = 0
    output_bytes = 0
    context = request["context_lines"]
    limit = request["max_results"]
    target = Path(payload["target"])
    exact = target.is_file()

    refusal = _under_deny_floor(root, target)
    if refusal:
        # 目标落在 deny 地板子树:输入本身非法(管道到不了这里,floor 已在
        # decide 拒绝;直构工具路径也应是结果而非 worker 故障)。
        return {"error_code": "invalid_input", "error_message": refusal}

    # 即使没有任何候选文本文件也要验证 pattern:Rust 正则语法是公开契约,
    # 不支持的构造必须在后端显式失败。
    validated = _backend(payload["binary"], request["pattern"], payload["regex"], b"", 1, 0)
    if isinstance(validated, dict):
        return validated

    counters = _Counters()
    try:
        for path, label in _files(root, target, request["glob"], sources, counters):
            if total_bytes >= _MAX_TOTAL_BYTES:
                truncated = True
                break
            remaining = min(_MAX_FILE_BYTES, _MAX_TOTAL_BYTES - total_bytes)
            try:
                raw = _read_file(path, remaining)
            except (OSError, ValueError):
                # 文件消失或被替换属于不完整观察,不是异常。
                truncated = True
                continue
            total_bytes += len(raw)
            if b"\x00" in raw:
                continue
            if exact:
                searched_file_path = label
            if len(raw) > remaining:
                raw = raw[:remaining]
                truncated = True
            raw = raw.decode("utf-8", "replace").encode("utf-8")
            backend_result = _backend(
                payload["binary"], request["pattern"], payload["regex"], raw,
                limit - len(matches), context,
            )
            if isinstance(backend_result, dict):
                return backend_result
            events, backend_truncated = backend_result
            truncated = truncated or backend_truncated
            lines: dict[int, str] = {}
            found: list[dict[str, Any]] = []
            for event in events:
                # rg JSON 行是外部数据:非 dict / 缺字段按畸形跳过并标记不完整,
                # 不让一条坏记录炸掉整次搜索。
                if not isinstance(event, dict):
                    truncated = True
                    continue
                if event.get("type") not in {"match", "context"}:
                    continue
                data = event.get("data")
                number = data.get("line_number") if isinstance(data, dict) else None
                if not isinstance(data, dict) or not isinstance(number, int):
                    truncated = True
                    continue
                lines[number] = _text(data.get("lines")).rstrip("\r\n")
                if event["type"] == "match":
                    found.append(data)
            for data in found:
                if len(matches) >= limit:
                    truncated = True
                    break
                number = data["line_number"]
                line = lines.get(number)
                if line is None:
                    truncated = True
                    continue
                match = {
                    "file_path": label,
                    "line_number": number,
                    "line_content": line[:_MAX_LINE_CHARS],
                    "context_before": [
                        lines[n][:_MAX_LINE_CHARS]
                        for n in range(max(1, number - context), number)
                        if n in lines
                    ],
                    "context_after": [
                        lines[n][:_MAX_LINE_CHARS]
                        for n in range(number + 1, number + context + 1)
                        if n in lines
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
                        "skipped_denied": counters.skipped_denied,
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
        "skipped_denied": counters.skipped_denied,
    }


def main(argv: Sequence[str] | None = None) -> int:
    """stdin 读 payload、stdout 写 JSON 结果;SIGTERM → 退出码 143。"""
    del argv  # 请求只走 stdin;argv 保留为进程入口约定。

    def terminate(_signum: int, _frame: FrameType | None) -> None:
        raise SystemExit(143)

    signal.signal(signal.SIGTERM, terminate)
    try:
        payload = json.load(sys.stdin)
        result = _search(payload)
    except SystemExit:
        raise
    except BaseException as exc:  # noqa: BLE001 — worker 边界:任何异常都转非 0 退出
        sys.stderr.write(f"search worker failed: {type(exc).__name__}: {exc}")
        return 1
    sys.stdout.write(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
