"""Task 4:command hook 机制层——纯 runner 与协议折叠函数(3B 笔记 §4)。

钉住四件事:
1. ``spawn_hook`` 的 spawn 姿势逐条对齐旧 ``agent_runtime/hooks.py:163-171``:
   ``cwd=spec.directory``、``env={"PATH": os.defpath}``、stdin=真 HookRequest
   JSON、stdout 上限 16_384 字节(``readexactly(16_385)`` 哨兵)、
   ``start_new_session=True``(整组信号的前提)、stderr=DEVNULL;
2. stdout 裁决面:空输出 ⇒ allowed;``{"block":bool,"feedback":str}`` ⇒
   allowed/blocked;未知键、非 object、非 JSON、非 0 退出 ⇒ error;
   feedback 超 2048 字节由 ``HookOutcome`` 构造期截断;
3. 超时/取消/异常路径一律经**注入的** ``ProcessSignaler`` 对整组 SIGKILL
   (SIGKILL 与旧实现一致),再做 0.5s 有界排空;外层取消清理后原样上抛;
4. ``hook_invocation_id`` 的确定性与 ``fold_hook_items`` 三态(settled /
   仅 started ⇒ unknown(阻断)/ 无记录 ⇒ None),malformed payload 跳过不抛。

测试用真实子进程(``/bin/sh`` fixture 脚本);测试模块不受
``praxis.adapters`` 分层契约约束,``signal_group`` 的 import 与
``test_tools_command_sessions.py`` 既有先例一致。
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import shlex
import signal
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from praxis.adapters.hooks.runner import spawn_hook
from praxis.protocol.hooks import (
    HookOutcome,
    HookSpec,
    fold_hook_items,
    hook_invocation_id,
)
from praxis.protocol.items import Item, ItemKind
from praxis.tools.builtins.process import signal_group

_REQUEST: dict[str, Any] = {
    "event": "PreToolUse",
    "invocation_id": "hook_" + "a" * 32,
    "thread_id": "thread-1",
    "turn_id": "turn-1",
    "workspace": "/tmp/workspace",
    "tool_call_id": "call-1",
    "tool_name": "apply_patch",
    "arguments": {"path": "a.txt", "text": "héllo"},
    "effects": ["write_workspace"],
    "input_revision": 3,
}


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _write_hook(tmp_path: Path, name: str, body: str) -> str:
    """落一个绝对路径的 /bin/sh fixture 脚本(env 只有 PATH ⇒ argv[0] 必须绝对)。"""
    path = tmp_path / name
    path.write_text("#!/bin/sh\n" + body + "\n", encoding="utf-8")
    path.chmod(0o755)
    return str(path)


def _spec(
    tmp_path: Path,
    argv: tuple[str, ...],
    *,
    timeout_seconds: float = 5.0,
    event: str = "PreToolUse",
) -> HookSpec:
    # directory 是第 5 个字段:按 T3 留档一律关键字构造。
    return HookSpec(
        name="policy",
        revision="1",
        event=event,  # type: ignore[arg-type]
        argv=argv,
        directory=str(tmp_path),
        timeout_seconds=timeout_seconds,
    )


def _noop_signaler(pgid: int, sig: signal.Signals) -> bool:
    return False


RecordedSignals = list[tuple[int, signal.Signals]]


def _recording_signaler() -> tuple[RecordedSignals, Callable[[int, signal.Signals], bool]]:
    """记录调用并真向下发信号(证明进程组真的消失)。"""
    calls: RecordedSignals = []

    def record(pgid: int, sig: signal.Signals) -> bool:
        calls.append((pgid, sig))
        return signal_group(pgid, sig)

    return calls, record


def _run(
    spec: HookSpec,
    request: dict[str, Any] | None = None,
    *,
    kill_group: Callable[[int, signal.Signals], bool] = _noop_signaler,
) -> HookOutcome:
    return asyncio.run(spawn_hook(spec, request or dict(_REQUEST), kill_group=kill_group))


def _group_alive(pgid: int) -> bool:
    try:
        os.killpg(pgid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _wait_group_gone(pgid: int, timeout: float = 5.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not _group_alive(pgid):
            return True
        time.sleep(0.02)
    return False


async def _wait_for_pid(path: Path, timeout: float = 5.0) -> int:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if path.exists():
            text = path.read_text(encoding="utf-8").strip()
            if text:
                return int(text)
        await asyncio.sleep(0.01)
    raise AssertionError("hook process did not start")


# ---------------------------------------------------------------------------
# spawn 姿势:stdin / env / cwd
# ---------------------------------------------------------------------------


def test_request_json_is_written_to_stdin_and_closed(tmp_path: Path) -> None:
    capture = tmp_path / "request.json"
    argv = _write_hook(tmp_path, "capture.sh", f"cat > {shlex.quote(str(capture))}")
    outcome = _run(_spec(tmp_path, (argv,)))
    assert outcome == HookOutcome("allowed")
    assert json.loads(capture.read_text(encoding="utf-8")) == _REQUEST
    assert "héllo" in capture.read_text(encoding="utf-8")  # ensure_ascii=False 的字节面


def test_environment_contains_only_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PRAXIS_TEST_SECRET", "must-not-leak")
    capture = tmp_path / "env.txt"
    argv = _write_hook(
        tmp_path,
        "env.sh",
        f"printenv > {shlex.quote(str(capture))}\nprintf '%s' '{{\"block\": false}}'",
    )
    assert _run(_spec(tmp_path, (argv,))) == HookOutcome("allowed")
    lines = capture.read_text(encoding="utf-8").splitlines()
    assert f"PATH={os.defpath}" in lines
    assert not any(line.startswith("PRAXIS_TEST_SECRET=") for line in lines)
    assert not any(line.startswith("HOME=") for line in lines)


def test_cwd_is_spec_directory(tmp_path: Path) -> None:
    capture = tmp_path / "cwd.txt"
    argv = _write_hook(
        tmp_path,
        "cwd.sh",
        f"pwd > {shlex.quote(str(capture))}\nprintf '%s' '{{\"block\": false}}'",
    )
    assert _run(_spec(tmp_path, (argv,))) == HookOutcome("allowed")
    assert Path(capture.read_text(encoding="utf-8").strip()).resolve() == tmp_path.resolve()


# ---------------------------------------------------------------------------
# stdout 裁决面
# ---------------------------------------------------------------------------


def test_empty_stdout_is_allowed(tmp_path: Path) -> None:
    argv = _write_hook(tmp_path, "empty.sh", ":")
    assert _run(_spec(tmp_path, (argv,))) == HookOutcome("allowed")


def test_block_false_is_allowed_with_feedback(tmp_path: Path) -> None:
    argv = _write_hook(tmp_path, "allow.sh", "printf '%s' '{\"block\": false, \"feedback\": \"note\"}'")
    assert _run(_spec(tmp_path, (argv,))) == HookOutcome("allowed", "note")


def test_block_true_is_blocked_with_feedback(tmp_path: Path) -> None:
    argv = _write_hook(tmp_path, "block.sh", "printf '%s' '{\"block\": true, \"feedback\": \"no\"}'")
    assert _run(_spec(tmp_path, (argv,))) == HookOutcome("blocked", "no")


def test_unknown_key_is_error(tmp_path: Path) -> None:
    argv = _write_hook(tmp_path, "unknown.sh", "printf '%s' '{\"allow\": true}'")
    assert _run(_spec(tmp_path, (argv,))) == HookOutcome("error")


def test_non_object_json_is_error(tmp_path: Path) -> None:
    argv = _write_hook(tmp_path, "array.sh", "printf '%s' '[true]'")
    assert _run(_spec(tmp_path, (argv,))) == HookOutcome("error")


def test_non_json_output_is_error(tmp_path: Path) -> None:
    argv = _write_hook(tmp_path, "garbage.sh", "printf '%s' 'not json'")
    assert _run(_spec(tmp_path, (argv,))) == HookOutcome("error")


def test_non_boolean_block_is_error(tmp_path: Path) -> None:
    argv = _write_hook(tmp_path, "badblock.sh", "printf '%s' '{\"block\": \"true\"}'")
    assert _run(_spec(tmp_path, (argv,))) == HookOutcome("error")


def test_nonzero_exit_is_error_even_with_valid_output(tmp_path: Path) -> None:
    argv = _write_hook(tmp_path, "fail.sh", "printf '%s' '{\"block\": false}'\nexit 3")
    assert _run(_spec(tmp_path, (argv,))) == HookOutcome("error")


def test_stdout_over_16_kib_is_error(tmp_path: Path) -> None:
    argv = _write_hook(tmp_path, "huge.sh", "printf '%17000s' ''")
    assert _run(_spec(tmp_path, (argv,))) == HookOutcome("error")


def test_feedback_is_truncated_to_2048_bytes(tmp_path: Path) -> None:
    argv = _write_hook(
        tmp_path,
        "feedback.sh",
        "printf '{\"block\": true, \"feedback\": \"%03000d\"}' 0",
    )
    outcome = _run(_spec(tmp_path, (argv,)))
    assert outcome.status == "blocked"
    assert len(outcome.feedback.encode("utf-8")) == 2048


def test_spawn_failure_is_error_not_raise(tmp_path: Path) -> None:
    argv = str(tmp_path / "does-not-exist")
    assert _run(_spec(tmp_path, (argv,))) == HookOutcome("error")


# ---------------------------------------------------------------------------
# 超时与取消:整组 SIGKILL + 有界排空
# ---------------------------------------------------------------------------


def test_timeout_kills_entire_process_group(tmp_path: Path) -> None:
    argv = _write_hook(tmp_path, "sleep.sh", "sleep 30")
    calls, kill_group = _recording_signaler()
    started = time.monotonic()
    outcome = _run(_spec(tmp_path, (argv,), timeout_seconds=0.2), kill_group=kill_group)
    elapsed = time.monotonic() - started

    assert outcome == HookOutcome("timeout")
    assert elapsed < 5.0
    assert len(calls) == 1
    pid, sent = calls[0]
    assert sent is signal.SIGKILL
    assert pid > 0
    assert _wait_group_gone(pid)


def test_outer_cancellation_cleans_up_and_reraises(tmp_path: Path) -> None:
    pidfile = tmp_path / "pid"
    argv = _write_hook(
        tmp_path, "cancel.sh", f"echo $$ > {shlex.quote(str(pidfile))}\nsleep 30"
    )
    spec = _spec(tmp_path, (argv,), timeout_seconds=30.0)
    calls, kill_group = _recording_signaler()

    async def scenario() -> int:
        task = asyncio.create_task(
            spawn_hook(spec, dict(_REQUEST), kill_group=kill_group)
        )
        pid = await _wait_for_pid(pidfile)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        return pid

    pid = asyncio.run(scenario())
    assert (pid, signal.SIGKILL) in calls
    assert _wait_group_gone(pid)


# ---------------------------------------------------------------------------
# hook_invocation_id:确定性身份
# ---------------------------------------------------------------------------


def test_invocation_id_is_deterministic_and_literal() -> None:
    expected = "hook_" + hashlib.sha256(
        "turn-1|PreToolUse|policy|digest".encode("utf-8")
    ).hexdigest()[:32]
    actual = hook_invocation_id("turn-1", "PreToolUse", "policy", "digest")
    assert actual == expected
    assert actual.startswith("hook_")
    assert len(actual) == len("hook_") + 32
    assert actual == hook_invocation_id("turn-1", "PreToolUse", "policy", "digest")


def test_invocation_id_changes_with_every_input() -> None:
    base = hook_invocation_id("turn-1", "PreToolUse", "policy", "digest")
    variants = {
        hook_invocation_id("turn-2", "PreToolUse", "policy", "digest"),
        hook_invocation_id("turn-1", "PostToolUse", "policy", "digest"),
        hook_invocation_id("turn-1", "PreToolUse", "other", "digest"),
        hook_invocation_id("turn-1", "PreToolUse", "policy", "other"),
    }
    assert len(variants) == 4
    assert base not in variants


# ---------------------------------------------------------------------------
# fold_hook_items:settled / 仅 started / 无记录
# ---------------------------------------------------------------------------


def _item(seq: int, payload: Any, *, kind: str = ItemKind.HOOK_INVOCATION) -> Item:
    return Item(
        item_id=f"i{seq}",
        thread_id="thread-1",
        turn_id="turn-1",
        seq=seq,
        kind=kind,
        payload=payload,
        created_at=0.0,
    )


_ID = "hook_" + "b" * 32


def _started(seq: int = 1) -> Item:
    return _item(
        seq,
        {
            "invocation_id": _ID,
            "event": "PreToolUse",
            "hook": "policy",
            "phase": "started",
            "status": "allowed",
            "feedback": "",
        },
    )


def _settled(seq: int = 2, status: str = "blocked", feedback: str = "no") -> Item:
    return _item(
        seq,
        {
            "invocation_id": _ID,
            "event": "PreToolUse",
            "hook": "policy",
            "phase": "settled",
            "status": status,
            "feedback": feedback,
        },
    )


def test_fold_absent_returns_none() -> None:
    assert fold_hook_items([], _ID) is None
    assert fold_hook_items([_item(1, {"invocation_id": "hook_" + "c" * 32})], _ID) is None


def test_fold_started_without_settled_is_unknown() -> None:
    assert fold_hook_items([_started()], _ID) == HookOutcome(
        "unknown", "crash before settlement"
    )


def test_fold_settled_wins_regardless_of_order() -> None:
    assert fold_hook_items([_started(), _settled()], _ID) == HookOutcome("blocked", "no")
    assert fold_hook_items([_settled(), _started(3)], _ID) == HookOutcome("blocked", "no")


def test_fold_settled_feedback_is_truncated() -> None:
    outcome = fold_hook_items([_settled(feedback="x" * 3000)], _ID)
    assert outcome is not None
    assert len(outcome.feedback.encode("utf-8")) == 2048


def test_fold_skips_malformed_payloads_without_raising() -> None:
    malformed = [
        _item(1, "not-a-mapping"),
        _item(2, {"invocation_id": _ID}),  # 无 phase
        _item(3, {"invocation_id": _ID, "phase": "settled", "status": "bogus"}),
        _item(4, {"invocation_id": _ID, "phase": "settled", "status": "allowed", "feedback": 3}),
        _item(5, {"invocation_id": _ID, "phase": "weird"}),
    ]
    assert fold_hook_items(malformed, _ID) is None
    assert fold_hook_items([*malformed, _started(9)], _ID) == HookOutcome(
        "unknown", "crash before settlement"
    )
    assert fold_hook_items([_settled(9), *malformed[:3]], _ID) == HookOutcome("blocked", "no")