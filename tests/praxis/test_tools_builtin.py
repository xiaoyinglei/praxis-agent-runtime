"""Task 11:read/write/bash 内置工具移植回归(设计 §6 移植清单;执行文档 §3 工具 ACI)。

asyncio.run 包装:仓库无 pytest-asyncio,与 test_model_adapters.py 同风格。
"""

from __future__ import annotations

import asyncio
import contextlib
import gc
import os
import signal
import time
from pathlib import Path
from typing import Any

import pytest

from praxis.protocol.sandbox import ToolExecContext
from praxis.protocol.tools import ResultStatus, ToolEffect, ToolValidationError
from praxis.tools.builtins.command import create_bash_tool
from praxis.tools.builtins.filesystem import create_read_tool, create_write_tool
from praxis.tools.builtins.paths import checked_path


def run(tool: Any, arguments: dict[str, Any]) -> Any:
    return asyncio.run(tool.run(arguments, ToolExecContext()))


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
    pidfile = tmp_path / "pid"
    r = run(create_bash_tool(tmp_path), {"command": "echo $$ > pid; sleep 30", "timeout_seconds": 1})
    assert r.status is ResultStatus.TIMEOUT
    assert r.metadata["timed_out"]
    assert r.is_error
    assert r.error_code == "timeout"
    assert_dead(pidfile)


def assert_dead(pidfile: Path, budget: float = 6.0) -> None:
    """pidfile 记录的进程必须已被回收:os.kill(pid, 0) → ProcessLookupError。"""
    pid = int(pidfile.read_text().strip())
    deadline = time.monotonic() + budget
    while True:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return
        assert time.monotonic() < deadline, f"process {pid} survived; expected group convergence"
        time.sleep(0.1)


def test_bash_background_child_does_not_hang(tmp_path: Path) -> None:
    # B20 形态:后台子进程持有管道,主命令退出即返回,不得等到 EOF
    async def drive() -> Any:
        return await asyncio.wait_for(
            create_bash_tool(tmp_path).run({"command": "sleep 30 & echo done"}, ToolExecContext()),
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


# ---------------------------------------------------------------------------
# 评审修复轮 1:子进程生命周期三缺陷的回归
# ---------------------------------------------------------------------------


def test_bash_setsid_escaped_grandchild_does_not_hostage(tmp_path: Path, capfd: Any) -> None:
    # Finding 1:逃逸出进程组的孙进程持有管道写端,不得绑架 run() 与 timeout_seconds
    tool = create_bash_tool(tmp_path)
    command = (
        "python3 -c 'import os,time;"
        'f=open("escape.pid","w");f.write(str(os.getpid()));f.close();'
        "os.setsid();time.sleep(30)' & echo done"
    )
    started = time.monotonic()

    async def drive() -> tuple[Any, float]:
        r = await tool.run({"command": command, "timeout_seconds": 2}, ToolExecContext())
        elapsed = time.monotonic() - started
        escaped_pid = int((tmp_path / "escape.pid").read_text().strip())
        # Phase 1 不追杀 setsid 逃逸进程;测试自己收尾
        with contextlib.suppress(ProcessLookupError):
            os.kill(escaped_pid, signal.SIGKILL)
        # 逃逸者死亡 → 管道 EOF,abandoned transport 在循环存活时干净关闭
        # (真实代理循环常驻;单次 asyncio.run 需留在循环内 settle,勿拖到 close 后 __del__)
        await asyncio.sleep(0.3)
        gc.collect()
        await asyncio.sleep(0.3)
        return r, elapsed

    r, elapsed = asyncio.run(drive())
    assert elapsed < 5
    assert "done" in r.text
    assert r.metadata["timed_out"] is False
    assert r.metadata["exit_code"] == 0
    assert "Event loop is closed" not in capfd.readouterr().err


def test_bash_chatty_output_is_capped(tmp_path: Path) -> None:
    # Finding 2:yes 全速刷管道,超时窗口内缓冲必须有硬上限,结果 ≤ max_model_output_bytes
    tool = create_bash_tool(tmp_path)
    r = run(tool, {"command": "yes", "timeout_seconds": 1})
    assert r.status is ResultStatus.TIMEOUT
    assert len(r.text) <= 30_000
    assert r.truncated


def test_bash_large_output_no_pipe_deadlock(tmp_path: Path) -> None:
    # Finding 2 伴随面:边跑边读,200KB 正常退出命令不得因管道灌满而死锁
    tool = create_bash_tool(tmp_path)
    started = time.monotonic()
    r = run(tool, {"command": 'python3 -c "import sys; sys.stdout.write(\'x\' * 200_000)"'})
    assert time.monotonic() - started < 10
    assert r.metadata["exit_code"] == 0
    assert len(r.text) <= 30_000
    assert r.truncated


def test_bash_cancellation_cleans_up(tmp_path: Path, capfd: Any) -> None:
    # Finding 3:执行中取消 → 进程组收敛、任务不悬空、无 Event-loop-closed 噪声
    tool = create_bash_tool(tmp_path)
    pidfile = tmp_path / "pid"

    async def drive() -> None:
        task = asyncio.create_task(tool.run({"command": "echo $$ > pid; sleep 60"}, ToolExecContext()))
        deadline = time.monotonic() + 5
        while not pidfile.exists() and time.monotonic() < deadline:
            await asyncio.sleep(0.05)
        assert pidfile.exists()
        await asyncio.sleep(0.1)  # 确保落在主等待阶段
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        else:
            pytest.fail("cancelled bash run must raise CancelledError")
        await asyncio.sleep(0.2)
        leaked = [
            t for t in asyncio.all_tasks()
            if t is not asyncio.current_task() and not t.done()
        ]
        assert leaked == []

    asyncio.run(drive())
    assert_dead(pidfile)
    captured = capfd.readouterr()
    assert "Event loop is closed" not in captured.err


def test_pending_tool_call_id_convention(tmp_path: Path) -> None:
    # Task 12 管道写 item 时替换真实 call id;工具层只放占位符
    (tmp_path / "f.txt").write_text("hello")
    r = run(create_read_tool(tmp_path), {"path": "f.txt"})
    assert r.tool_call_id == "pending"
    assert r.tool_name == "read"
