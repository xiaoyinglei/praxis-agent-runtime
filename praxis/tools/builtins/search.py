"""search_text 内置工具:打包 ripgrep + 隔离 worker 子进程(Task 11)。

公开契约(对照旧 agent_runtime/tools/builtins/search.py,只抄语义、禁止 import):
- 输入 ``pattern``(1..2000)、``path``(默认 ".")、``glob``、``regex``(缺省按
  旧标记表自动探测)、``context_lines``(默认 2,0..50)、``max_results``
  (默认 40,1..200);
- 输出 content 每条匹配 ``<相对路径>:<行号>: <行内容>``,上下文行前缀 ``- ``;
  metadata = ``{total_matches, truncated, searched_file_path, backend,
  limits, skipped_denied}``。

执行纪律:
- rg 只从 ``ripgrep-bin`` 发行包里定位(``importlib.metadata``),绝不 PATH 查找;
  拿不到 → ``search_backend_unavailable``(结果,不是异常);
- worker 用 ``sys.executable -I -c <bootstrap>`` 启动:``-I`` 隔离宿主
  PYTHONPATH/用户 site,``env={"PATH": os.defpath}``、``cwd="/"``、
  ``start_new_session=True``;取消/超时走 ``process`` 原语的整组
  SIGTERM → 有界宽限 → SIGKILL(与 bash 同一份纪律);
- worker 非 0 → ``search_worker_failed``;rg 参数非法 → ``invalid_input``,
  且 stderr 前 2000 字符只进 metadata,不进模型可见 content;
- R-T11-2:worker 逐条目复用 ``protocol.policy.path_access`` 过滤 deny,
  被拒文件跳过并计入 ``metadata["skipped_denied"]``;``decide`` 仍是唯一
  裁决函数,这里不是第二套决策入口。
"""

from __future__ import annotations

import asyncio
import contextlib
import importlib.metadata
import json
import os
import signal
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

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
    json_schema_input,
)
from praxis.tools.builtins.filesystem import PENDING_TOOL_CALL_ID
from praxis.tools.builtins.paths import checked_path
from praxis.tools.builtins.process import await_main_exit, signal_group
from praxis.tools.builtins.search_worker import LIMITS

_DEFAULT_CONTEXT_LINES = 2
_DEFAULT_MAX_RESULTS = 40
_MAX_CONTEXT_LINES = 50
_MAX_RESULTS = 200
_MAX_PATTERN_CHARS = 2_000
_MAX_PATH_CHARS = 4_096
_MAX_GLOB_CHARS = 512
_MAX_STDERR_CHARS = 2_000
_WORKER_TIMEOUT_SECONDS = 14.0
_CANCEL_GRACE_SECONDS = 0.5
_POST_KILL_WAIT_SECONDS = 2.0
# worker 的最终 JSON 以 _MAX_OUTPUT_BYTES 为上限;读取缓冲留出序列化/错误余量。
_WORKER_STREAM_LIMIT = 4_000_000


def rg_path() -> Path:
    """Locate our locked dependency, never an unrelated executable on PATH.

    ``ripgrep-bin`` 不提供 Python 模块 API,只把 ``rg`` 安装进发行包的文件表
    (uv: ``<venv>/bin/rg``)。因此定位必须走 ``importlib.metadata`` 的文件表,
    ``locate_file`` 得到绝对路径;任何一环失败都抛 ``FileNotFoundError``,
    由工具映射为 ``search_backend_unavailable``。
    """
    distribution = importlib.metadata.distribution("ripgrep-bin")
    for entry in distribution.files or ():
        if entry.name in {"rg", "rg.exe"}:
            binary = Path(str(distribution.locate_file(entry))).resolve()
            if binary.is_file() and os.access(binary, os.X_OK):
                return binary
    raise FileNotFoundError("installed ripgrep-bin has no executable rg")


def _uses_regular_expression(pattern: str, requested: bool | None) -> bool:
    """旧标记表:未显式指定时按常见正则语法自动探测。"""
    if requested is not None:
        return requested
    return any(
        marker in pattern
        for marker in (
            ".*",
            ".+",
            ".?",
            "|",
            "^",
            "$",
            "(?",
            "[",
            r"\b",
            r"\d",
            r"\s",
            r"\w",
        )
    )


def _bootstrap() -> str:
    """子进程 ``-c`` 代码:显式注入包根后导入 worker(``-I`` 不读宿主路径)。"""
    package_root = Path(__file__).resolve().parents[3]
    return (
        f"import sys; sys.path.insert(0, {str(package_root)!r}); "
        "from praxis.tools.builtins.search_worker import main; "
        "raise SystemExit(main(sys.argv))"
    )


def _policies(exec_ctx: ToolExecContext) -> list[dict[str, Any]] | None:
    """不可变权限快照 → worker(policy + ceilings 的纯 JSON 声明)。"""
    permissions = exec_ctx.permissions
    if permissions is None:
        return None
    return [
        policy.model_dump(mode="json")
        for policy in (permissions.policy, *permissions.ceilings)
    ]


def _result(
    *,
    content: str,
    status: ResultStatus = ResultStatus.OK,
    error_code: str | None = None,
    error_message: str | None = None,
    metadata: Mapping[str, Any] | None = None,
    truncated: bool = False,
) -> ToolResult:
    is_error = status in {ResultStatus.ERROR, ResultStatus.TIMEOUT}
    return ToolResult(
        tool_call_id=PENDING_TOOL_CALL_ID,
        tool_name="search_text",
        content=(ToolContentBlock(type="text", data={"text": content}),),
        status=status,
        is_error=is_error,
        error_code=error_code if is_error else None,
        error_message=error_message if is_error else None,
        truncated=truncated,
        metadata=dict(metadata or {}),
    )


def _format_matches(matches: list[dict[str, Any]]) -> str:
    """模型可读文本:上下文行 ``- `` 前缀,匹配行 ``path:line: content``。"""
    lines: list[str] = []
    for match in matches:
        lines.extend(f"- {line}" for line in match["context_before"])
        lines.append(f"{match['file_path']}:{match['line_number']}: {match['line_content']}")
        lines.extend(f"- {line}" for line in match["context_after"])
    return "\n".join(lines)


async def _stop_worker(proc: asyncio.subprocess.Process) -> None:
    """取消/超时:整组 SIGTERM → 有界宽限 → SIGKILL → 等回收(R-T11-1)。

    宽限期间被二次取消也立即升级 SIGKILL(同 command.py 的取消纪律);
    任何路径都不留下仍在遍历的 worker 进程组。
    """
    killed = False
    try:
        signal_group(proc.pid, signal.SIGTERM)
        try:
            await await_main_exit(proc, _CANCEL_GRACE_SECONDS)
        except TimeoutError:
            pass
        signal_group(proc.pid, signal.SIGKILL)
        killed = True
        with contextlib.suppress(TimeoutError):
            await await_main_exit(proc, _POST_KILL_WAIT_SECONDS)
    except asyncio.CancelledError:
        if not killed:
            signal_group(proc.pid, signal.SIGKILL)
        raise


def create_search_text_tool(workspace: Path) -> Tool:
    validate_input = json_schema_input(
        {
            "type": "object",
            "properties": {
                "pattern": {"type": "string", "minLength": 1, "maxLength": _MAX_PATTERN_CHARS},
                "path": {"type": "string", "maxLength": _MAX_PATH_CHARS},
                "glob": {"type": ("string", "null"), "maxLength": _MAX_GLOB_CHARS},
                "regex": {"type": ("boolean", "null")},
                "context_lines": {
                    "type": "integer",
                    "minimum": 0,
                    "maximum": _MAX_CONTEXT_LINES,
                },
                "max_results": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": _MAX_RESULTS,
                },
            },
            "required": ("pattern",),
            "additionalProperties": False,
        }
    )

    async def run(
        arguments: Mapping[str, JsonValue], exec_ctx: ToolExecContext  # 进程内工具:不消费沙箱
    ) -> ToolResult:
        args = validate_input(arguments)
        pattern = cast("str", args["pattern"])
        path_value = cast("str", args.get("path") or ".")
        glob = cast("str | None", args.get("glob"))
        requested_regex = cast("bool | None", args.get("regex"))
        context_lines = cast("int", args.get("context_lines", _DEFAULT_CONTEXT_LINES))
        max_results = cast("int", args.get("max_results", _DEFAULT_MAX_RESULTS))
        target = checked_path(workspace, path_value, allow_root=True)

        try:
            binary = rg_path()
        except (importlib.metadata.PackageNotFoundError, FileNotFoundError):
            return _result(
                content=(
                    "search_backend_unavailable: Search requires the packaged "
                    "ripgrep-bin dependency; reinstall Praxis."
                ),
                status=ResultStatus.ERROR,
                error_code="search_backend_unavailable",
            )

        payload = json.dumps(
            {
                "root": str(workspace.resolve()),
                "target": str(target),
                "request": {
                    "pattern": pattern,
                    "glob": glob,
                    "context_lines": context_lines,
                    "max_results": max_results,
                },
                "regex": _uses_regular_expression(pattern, requested_regex),
                "binary": str(binary),
                "policies": _policies(exec_ctx),
            }
        ).encode()

        proc = await asyncio.create_subprocess_exec(
            sys.executable,
            "-I",
            "-c",
            _bootstrap(),
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            start_new_session=True,
            env={"PATH": os.defpath},
            cwd="/",
        )
        timed_out = False
        try:
            try:
                stdout, stderr = await asyncio.wait_for(
                    proc.communicate(payload), timeout=_WORKER_TIMEOUT_SECONDS
                )
            except TimeoutError:
                timed_out = True
                await _stop_worker(proc)
                stdout, stderr = b"", b""
        except asyncio.CancelledError:
            await _stop_worker(proc)
            raise

        if timed_out:
            return _result(
                content=f"timeout: search worker exceeded {_WORKER_TIMEOUT_SECONDS:.0f}s",
                status=ResultStatus.TIMEOUT,
                error_code="timeout",
                metadata={"timed_out": True},
            )

        stderr_text = stderr.decode("utf-8", errors="replace")
        if proc.returncode != 0:
            return _result(
                content="search_worker_failed: Search worker failed; retry the search.",
                status=ResultStatus.ERROR,
                error_code="search_worker_failed",
                metadata={"stderr": stderr_text[:_MAX_STDERR_CHARS]},
            )
        try:
            worker_result = json.loads(stdout)
        except (ValueError, UnicodeError):
            return _result(
                content="search_worker_failed: Search worker returned invalid JSON.",
                status=ResultStatus.ERROR,
                error_code="search_worker_failed",
                metadata={"stderr": stderr_text[:_MAX_STDERR_CHARS]},
            )
        if isinstance(worker_result, dict) and "error_code" in worker_result:
            error_code = str(worker_result["error_code"])
            message = str(worker_result.get("error_message", ""))
            if error_code == "invalid_input":
                # rg stderr 只进 metadata(前 2000 字符),模型可见文本保持简短。
                return _result(
                    content="invalid_input: invalid search pattern or arguments",
                    status=ResultStatus.ERROR,
                    error_code="invalid_input",
                    metadata={"stderr": message[:_MAX_STDERR_CHARS]},
                )
            return _result(
                content=f"{error_code}: {message}",
                status=ResultStatus.ERROR,
                error_code=error_code,
            )

        matches = worker_result["matches"]
        truncated = bool(worker_result.get("truncated"))
        return _result(
            content=_format_matches(matches),
            truncated=truncated,
            metadata={
                "total_matches": len(matches),
                "truncated": truncated,
                "searched_file_path": worker_result.get("searched_file_path"),
                "backend": "ripgrep-bin",
                "limits": dict(LIMITS),
                "skipped_denied": worker_result.get("skipped_denied", 0),
            },
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
            name="search_text",
            description=(
                "Search workspace files without an index for literal text or a "
                "ripgrep regular expression. One file or directory path, optional "
                "glob, bounded context lines, and a result limit. Results are "
                "formatted as '<relative path>:<line>: <text>' with context lines "
                "prefixed '- '. Files are searched in deterministic path order using "
                "the bundled ripgrep engine; symlinks are never followed and "
                "permission-denied files are skipped."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "pattern": {
                        "type": "string",
                        "description": "Literal text or regular expression to find.",
                    },
                    "path": {
                        "type": "string",
                        "description": "Workspace-relative file or directory to search (default '.').",
                    },
                    "glob": {
                        "type": ("string", "null"),
                        "description": "Optional file glob such as '*.py' or 'src/**/*.ts'.",
                    },
                    "regex": {
                        "type": ("boolean", "null"),
                        "description": (
                            "true = ripgrep (Rust) regex, false = exact literal text; "
                            "omit to auto-detect common regex syntax."
                        ),
                    },
                    "context_lines": {
                        "type": "integer",
                        "description": (
                            f"Context lines before/after each match, 0..{_MAX_CONTEXT_LINES} "
                            f"(default {_DEFAULT_CONTEXT_LINES})."
                        ),
                    },
                    "max_results": {
                        "type": "integer",
                        "description": (
                            f"Maximum matching lines, 1..{_MAX_RESULTS} "
                            f"(default {_DEFAULT_MAX_RESULTS})."
                        ),
                    },
                },
                "required": ("pattern",),
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
        timeout_seconds=15,
        max_model_output_bytes=2_000_000,
    )