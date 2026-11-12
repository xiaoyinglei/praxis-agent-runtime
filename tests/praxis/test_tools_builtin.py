"""Task 11:read/write/bash 内置工具移植回归(设计 §6 移植清单;执行文档 §3 工具 ACI)。

asyncio.run 包装:仓库无 pytest-asyncio,与 test_model_adapters.py 同风格。
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest

from praxis.protocol.tools import ResultStatus, ToolEffect, ToolValidationError
from praxis.tools.builtins.command import create_bash_tool
from praxis.tools.builtins.filesystem import create_read_tool, create_write_tool
from praxis.tools.builtins.paths import checked_path


def run(tool: Any, arguments: dict[str, Any]) -> Any:
    return asyncio.run(tool.run(arguments))


# ---------------------------------------------------------------------------
# brief Step 1 的五个形态,逐条保留
# ---------------------------------------------------------------------------


def test_read_pages_and_next_offset(tmp_path: Path) -> None:
    (tmp_path / "f.txt").write_text("a" * 100)
    tool = create_read_tool(tmp_path)
    r = run(tool, {"path": "f.txt", "max_bytes": 40})
    assert r.text == "a" * 40
    assert r.metadata["next_offset"] == 40
    assert r.truncated
    # 续页直到读完:offset 链必须精确覆盖 100 字节
    offset = r.metadata["next_offset"]
    total = len(r.text)
    while offset is not None:
        r = run(tool, {"path": "f.txt", "offset": offset, "max_bytes": 40})
        total += len(r.text)
        offset = r.metadata["next_offset"]
    assert total == 100
    assert r.metadata["size_bytes"] == 100


def test_write_atomic_and_confined(tmp_path: Path) -> None:
    tool = create_write_tool(tmp_path)
    run(tool, {"path": "sub/f.txt", "content": "你好"})
    assert (tmp_path / "sub/f.txt").read_text() == "你好"
    with pytest.raises(ToolValidationError):
        run(tool, {"path": "../escape.txt", "content": "x"})
    assert not (tmp_path.parent / "escape.txt").exists()
    # 原子写:同目录不留 tmp 残片
    leftovers = [p.name for p in (tmp_path / "sub").iterdir() if p.name != "f.txt"]
    assert leftovers == []


def test_bash_exit_code_and_output(tmp_path: Path) -> None:
    r = run(create_bash_tool(tmp_path), {"command": "echo out; echo err >&2; exit 3"})
    assert r.metadata["exit_code"] == 3
    assert "out" in r.text
    assert "err" in r.text
    assert not r.is_error
    assert r.status is ResultStatus.OK


def test_bash_timeout_kills_process_group(tmp_path: Path) -> None:
    r = run(create_bash_tool(tmp_path), {"command": "sleep 30", "timeout_seconds": 1})
    assert r.status is ResultStatus.TIMEOUT
    assert r.metadata["timed_out"]
    assert r.is_error
    assert r.error_code == "timeout"


def test_bash_background_child_does_not_hang(tmp_path: Path) -> None:
    # B20 形态:后台子进程持有管道,主命令退出即返回,不得等到 EOF
    async def drive() -> Any:
        return await asyncio.wait_for(
            create_bash_tool(tmp_path).run({"command": "sleep 30 & echo done"}),
            timeout=5,
        )

    r = asyncio.run(drive())
    assert "done" in r.text
    assert r.metadata["exit_code"] == 0


# ---------------------------------------------------------------------------
# 补充用例
# ---------------------------------------------------------------------------


def test_read_missing_file(tmp_path: Path) -> None:
    tool = create_read_tool(tmp_path)
    r = run(tool, {"path": "nope.txt"})
    assert r.status is ResultStatus.ERROR
    assert r.is_error
    assert r.error_code == "not_found"


def test_read_binary_file(tmp_path: Path) -> None:
    (tmp_path / "blob.bin").write_bytes(b"\x00\x01\x02\xff" * 16)
    tool = create_read_tool(tmp_path)
    r = run(tool, {"path": "blob.bin"})
    assert r.status is ResultStatus.ERROR
    assert r.error_code == "binary_file"
    assert r.metadata["is_binary"]
    assert r.metadata["size_bytes"] == 64


def test_write_creates_parent_dirs(tmp_path: Path) -> None:
    run(create_write_tool(tmp_path), {"path": "a/b/c/f.txt", "content": "x"})
    assert (tmp_path / "a/b/c/f.txt").read_text() == "x"


def test_write_overwrites_existing_keeps_file(tmp_path: Path) -> None:
    tool = create_write_tool(tmp_path)
    run(tool, {"path": "f.txt", "content": "one"})
    run(tool, {"path": "f.txt", "content": "two"})
    assert (tmp_path / "f.txt").read_text() == "two"


def test_checked_path_rejects_absolute_outside(tmp_path: Path) -> None:
    workspace = tmp_path / "ws"
    workspace.mkdir()
    with pytest.raises(ToolValidationError):
        checked_path(workspace, "/etc/passwd")
    outside = tmp_path / "outside.txt"
    outside.write_text("secret")
    with pytest.raises(ToolValidationError):
        checked_path(workspace, str(outside), write=True)


def test_checked_path_rejects_symlink_escape(tmp_path: Path) -> None:
    workspace = tmp_path / "ws"
    workspace.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "secret.txt").write_text("s")
    (workspace / "link").symlink_to(outside)
    with pytest.raises(ToolValidationError):
        checked_path(workspace, "link/secret.txt")
    with pytest.raises(ToolValidationError):
        checked_path(workspace, "link/secret.txt", write=True)


def test_bash_timeout_seconds_schema_bounds(tmp_path: Path) -> None:
    tool = create_bash_tool(tmp_path)
    with pytest.raises(ToolValidationError):
        tool.validate_input({"command": "true", "timeout_seconds": 601})
    with pytest.raises(ToolValidationError):
        tool.validate_input({"command": "true", "timeout_seconds": 0})
    with pytest.raises(ToolValidationError):
        run(tool, {"command": "true", "timeout_seconds": 601})
    tool.validate_input({"command": "true", "timeout_seconds": 600})


def test_tool_policy_fields(tmp_path: Path) -> None:
    read = create_read_tool(tmp_path)
    assert read.static_effects == frozenset({ToolEffect.READ_WORKSPACE})
    assert read.idempotent and read.concurrency_safe
    assert read.timeout_seconds == 30
    assert read.max_model_output_bytes == 1_000_000
    write = create_write_tool(tmp_path)
    assert write.static_effects == frozenset({ToolEffect.WRITE_WORKSPACE})
    assert write.idempotent and not write.concurrency_safe
    bash = create_bash_tool(tmp_path)
    assert bash.static_effects == frozenset({ToolEffect.EXECUTE_PROCESS})
    assert not bash.idempotent and not bash.concurrency_safe
    assert bash.timeout_seconds == 605
    assert bash.max_model_output_bytes == 30_000


def test_resolve_use_targets(tmp_path: Path) -> None:
    workspace = tmp_path / "ws"
    workspace.mkdir()
    (workspace / "f.txt").write_text("x")
    read = create_read_tool(workspace)
    use = read.resolve_use({"path": "f.txt"})
    assert use.effects == frozenset({ToolEffect.READ_WORKSPACE})
    assert use.targets[0].value == str(workspace.resolve() / "f.txt")
    bash = create_bash_tool(workspace)
    use = bash.resolve_use({"command": "true"})
    assert use.effects == frozenset({ToolEffect.EXECUTE_PROCESS})


def test_pending_tool_call_id_convention(tmp_path: Path) -> None:
    # Task 12 管道写 item 时替换真实 call id;工具层只放占位符
    (tmp_path / "f.txt").write_text("hello")
    r = run(create_read_tool(tmp_path), {"path": "f.txt"})
    assert r.tool_call_id == "pending"
    assert r.tool_name == "read"
