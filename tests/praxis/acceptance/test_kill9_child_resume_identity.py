"""R-FINAL-C1 ③ kill -9 端到端:子身份重建跨真实进程死亡存活。

真实 OS 子进程(``_child_resume_identity.py``)spawn 一个 read_only 子并让它
持续 running(child_hold 缝),父测试轮询 store 确认子 tool_call 已落后 SIGKILL
整个进程(无任何用户态收尾)。随后在本测试进程里:

1. 裸 open_session(thread=父) 即 sweep → 子 running 孤儿被收编 interrupted;
2. 裸 open_session(workspace=父, thread=子, db_path=父库, resume_turn_id=子)
   重建子装配身份——read_only 写否决 ceiling 跨进程存活;
3. resume(confirm=True) → child_hold started+非幂等 → outcome_unknown PAUSED;
   reconcile ok 后续跑 → write 仍被 ceiling 拒(静默提权关闭)→ 剧本收尾。

断言全部落在 item 日志与真实文件系统上:write 结果 denied、f.txt 从未出现。
"""

from __future__ import annotations

import asyncio
import json
import os
import signal
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pytest

from praxis.core.statemachine import TurnStatus
from praxis.interfaces.sdk import Item, ItemKind, ModelResponse, ModelToolCall, ScriptedAdapter, Usage, open_session
from praxis.store.log import Store
from praxis.store.turns import get_turn

CHILD_SCRIPT = Path(__file__).resolve().parent / "_child_resume_identity.py"
KILL_WINDOW_TIMEOUT = 30.0


def _resp(
    content: str, calls: Sequence[Mapping[str, Any]] = (), finish: str | None = None
) -> dict[str, Any]:
    return {
        "content": content,
        "tool_calls": list(calls),
        "usage": {"input_tokens": 1, "output_tokens": 1},
        "finish_reason": finish or ("tool_calls" if calls else "stop"),
    }


PARENT_SCRIPTS = {
    "parent": [
        _resp("spawn", [{"call_id": "s1", "name": "spawn_subagent", "arguments": {"task": "[child] hold task"}}]),
        _resp("parent done"),
    ],
    "child": [
        _resp("[child] hold", [{"call_id": "h1", "name": "child_hold", "arguments": {}}]),
        _resp("write", [{"call_id": "w1", "name": "write", "arguments": {"path": "f.txt", "content": "x"}}]),
        _resp("child done"),
    ],
}


def _response(spec: Mapping[str, Any]) -> ModelResponse:
    calls = tuple(
        ModelToolCall(call_id=c["call_id"], name=c["name"], arguments=dict(c["arguments"]))
        for c in spec.get("tool_calls", ())
    )
    usage = spec.get("usage", {"input_tokens": 1, "output_tokens": 1})
    return ModelResponse(
        content=spec["content"],
        tool_calls=calls,
        usage=Usage(input_tokens=int(usage["input_tokens"]), output_tokens=int(usage["output_tokens"])),
        finish_reason=str(spec["finish_reason"]),
    )


def _wait_for_identity(out_file: Path, proc: subprocess.Popen[bytes]) -> dict[str, str]:
    deadline = time.monotonic() + KILL_WINDOW_TIMEOUT
    while True:
        if proc.poll() is not None:
            stderr = proc.stderr.read().decode(errors="replace") if proc.stderr else ""
            pytest.fail(f"child exited before kill window (rc={proc.returncode}): {stderr[-2000:]}")
        if out_file.exists():
            data = json.loads(out_file.read_text(encoding="utf-8"))
            if data.get("child_thread_id") and data.get("child_root_turn_id"):
                return data
        if time.monotonic() > deadline:
            pytest.fail("timeout waiting for child identity out_file")
        time.sleep(0.02)


def _wait_for_child_call(store: Store, child_thread: str, proc: subprocess.Popen[bytes]) -> None:
    deadline = time.monotonic() + KILL_WINDOW_TIMEOUT
    while True:
        if proc.poll() is not None:
            pytest.fail(f"child exited before hold tool_call written (rc={proc.returncode})")
        items = store.read_items(child_thread)
        if any(
            item.kind == ItemKind.TOOL_CALL and item.payload.get("call_id") == "h1"
            for item in items
        ):
            return
        if time.monotonic() > deadline:
            pytest.fail("timeout waiting for child hold tool_call")
        time.sleep(0.02)


def test_kill9_child_resume_rebuilds_read_only_veto(tmp_path: Path) -> None:
    workspace = tmp_path / "ws"
    workspace.mkdir()
    (workspace / ".praxis").mkdir()
    responses = tmp_path / "responses.json"
    responses.write_text(json.dumps(PARENT_SCRIPTS), encoding="utf-8")
    out_file = tmp_path / "child.json"
    db_path = workspace / ".praxis" / "praxis.sqlite"

    proc = subprocess.Popen(
        [sys.executable, str(CHILD_SCRIPT), str(workspace), str(responses), "parent", str(out_file)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    store = Store(db_path)
    try:
        identity = _wait_for_identity(out_file, proc)
        child_thread = identity["child_thread_id"]
        child_turn = identity["child_root_turn_id"]
        _wait_for_child_call(store, child_thread, proc)
        # 子身份已钉进 binding(spawn 写端)。
        turn = get_turn(store, child_turn)
        assert turn is not None and turn.status == "running"
        assert "child_assembly" in turn.binding
        os.kill(proc.pid, signal.SIGKILL)
        proc.wait(timeout=10)
        assert proc.returncode == -signal.SIGKILL
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait(timeout=10)
        store.close()

    # sweep-on-open:子 running 孤儿 → interrupted(无手工 sweep)。
    async def _sweep() -> None:
        session = open_session(workspace, ScriptedAdapter([]), thread_id="parent")
        try:
            assert session.status(child_turn).status == "interrupted"
        finally:
            session.close()

    asyncio.run(_sweep())

    # 裸 open_session 形 resume:子身份经 resume_turn_id 重建,写否决跨进程存活。
    resume_specs = PARENT_SCRIPTS["child"][1:]  # write_resp + done_resp(hold 已消耗)

    async def _resume() -> None:
        child = open_session(
            workspace,
            ScriptedAdapter([_response(s) for s in resume_specs]),
            thread_id=child_thread,
            db_path=db_path,
            resume_turn_id=child_turn,
        )
        try:
            readonly = [c for c in child.permissions.ceilings if c.profile == "read-only"]
            assert len(readonly) == 1  # 写否决 ceiling 跨 kill9 重建
            r1 = await child.resume(child_turn, confirm=True)
            assert r1.status is TurnStatus.PAUSED
            assert r1.reason == "outcome_unknown"
            assert r1.unknown_calls == ("h1",)
            r2 = await child.resume(child_turn, reconcile={"h1": "ok"})
            assert r2.status is TurnStatus.COMPLETED
            assert r2.output == "child done"
        finally:
            await child.aclose()

    asyncio.run(_resume())

    store2 = Store(db_path)
    try:
        items: list[Item] = store2.read_turn_items(child_turn)
        write_results = [
            item
            for item in items
            if item.kind == ItemKind.TOOL_RESULT and item.payload.get("tool_name") == "write"
        ]
        assert len(write_results) == 1
        assert write_results[0].payload["status"] == "denied"  # 写否决跨 resume 承重
        turn = get_turn(store2, child_turn)
        assert turn is not None and turn.status == "completed"
    finally:
        store2.close()
    assert not (workspace / "f.txt").exists()  # 静默提权关闭:文件从未落父工作区