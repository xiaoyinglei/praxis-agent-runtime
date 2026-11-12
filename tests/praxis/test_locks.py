"""praxis.core.locks — thread 粒度执行锁单测(设计 §5.1/§5.8)。

TDD:本文件在 locks.py 存在前必须失败。
跨进程与进程死亡(SIGKILL)场景由 acceptance/test_kill9_sweep.py 真实验证,
本文件只测同进程语义:互斥、按 thread 独立、异常/正常路径释放。
"""

from __future__ import annotations

from pathlib import Path
from uuid import uuid4

import pytest

from praxis.core.locks import ThreadBusy, thread_execution_lock


def test_thread_busy_is_runtime_error() -> None:
    assert issubclass(ThreadBusy, RuntimeError)


def test_lock_mutex(tmp_path: Path) -> None:
    """同一 thread 锁被持有时再次进入 → ThreadBusy(flock 不同 fd 相互独立)。"""
    with thread_execution_lock(tmp_path, "t"):
        with pytest.raises(ThreadBusy):
            with thread_execution_lock(tmp_path, "t"):
                pass


def test_lockfile_lives_under_praxis_locks(tmp_path: Path) -> None:
    """lockfile 路径约定:workspace/.praxis/locks/<thread_id>.lock。"""
    with thread_execution_lock(tmp_path, "abc"):
        assert (tmp_path / ".praxis" / "locks" / "abc.lock").is_file()


def test_lock_creates_missing_dir_tree(tmp_path: Path) -> None:
    ws = tmp_path / "fresh" / "workspace"
    with thread_execution_lock(ws, "t"):
        assert (ws / ".praxis" / "locks" / "t.lock").is_file()


def test_lock_different_threads_independent(tmp_path: Path) -> None:
    """锁粒度 = thread:不同 thread 的执行锁互不影响。"""
    with thread_execution_lock(tmp_path, "a"):
        with thread_execution_lock(tmp_path, "b"):
            pass


def test_lock_released_on_normal_exit(tmp_path: Path) -> None:
    with thread_execution_lock(tmp_path, "t"):
        pass
    with thread_execution_lock(tmp_path, "t"):
        pass


def test_lock_released_on_exception(tmp_path: Path) -> None:
    """with 体内异常退出必须解锁——否则一次崩溃就永久锁死该 thread。"""
    with pytest.raises(ZeroDivisionError):
        with thread_execution_lock(tmp_path, "t"):
            raise ZeroDivisionError("boom")  # 异常路径必须解锁
    with thread_execution_lock(tmp_path, "t"):
        pass


# ---------------------------------------------------------------------------
# R6 修复 2:thread_id 路径守卫(SDK Task 14 接受外部 thread_id,文件名不得逃逸)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bad_id", ["", "..", "../evil", "a/b", "a\\b", "x/../y", "/abs"])
def test_lock_rejects_path_unsafe_thread_id(tmp_path: Path, bad_id: str) -> None:
    """含路径分隔符/'..'成分/空的 thread_id → ValueError,且碰文件系统之前就拒绝。"""
    with pytest.raises(ValueError):
        with thread_execution_lock(tmp_path, bad_id):
            pass
    assert not (tmp_path / ".praxis").exists()  # 拒绝发生在 mkdir 之前


def test_lock_accepts_normal_uuid_hex_thread_id(tmp_path: Path) -> None:
    """生产形态 uuid hex 不受守卫影响。"""
    tid = uuid4().hex
    with thread_execution_lock(tmp_path, tid):
        assert (tmp_path / ".praxis" / "locks" / f"{tid}.lock").is_file()
    with thread_execution_lock(tmp_path, "legit-name.with.dots"):  # 非 '..' 成分,合法文件名
        pass
