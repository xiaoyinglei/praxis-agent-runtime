"""Task 18 Step 3:kill -9 落在后台命令飞行中(真子进程 + 真 SIGKILL;设计 §5.11/D13)。

不是模拟。子进程(``_child_run_turn.py`` variant d,sys.executable)在真实 Session 里:

1. 模型剧本让 ``bash`` 带 ``yield_time_seconds`` 起 ``touch cmd-marker && sleep 60``:
   marker 真实写入工作区,工具返回 ``state="running"`` 与 ``session_id``
   (宿主进程把 marker 登记到 ``command_jobs_dir()``);
2. 第二个工具调用前有 45s 故障注入缝,父进程等到"后台会话 running + 第二个
   tool_call 已写"即 SIGKILL 宿主——无任何用户态清理。

父进程在**新进程**里按真实恢复顺序验证:

- ``manage_command``(全新 ``CommandSessions``,marker 仍在)报
  ``command_session_lost`` —— 宿主重启后注册表没有终态事实;
- ``sweep_orphan_command_jobs`` 用宿主存活门 + pgid/命令行守卫真的杀掉遗留进程组
  (``ps`` 复查消失),并删除 marker;
- 重开 Session → turn ``interrupted``(sweep-on-open)。

完成后不留活体:清扫成功是测试通过的必要条件。
"""

from __future__ import annotations

import asyncio
import json
import os
import signal
import subprocess
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import pytest

from praxis.adapters.sandbox.base import probe_backend
from praxis.core.recovery import sweep_orphan_command_jobs
from praxis.interfaces.sdk import Item, ItemKind, ScriptedAdapter, open_session
from praxis.protocol.floor import command_jobs_dir
from praxis.protocol.sandbox import ToolExecContext
from praxis.store.log import Store
from praxis.store.turns import list_turns
from praxis.tools.builtins.command_sessions import CommandSessions, create_manage_command_tool
from praxis.tools.builtins.process import signal_group

CHILD_SCRIPT = Path(__file__).resolve().parent / "_child_run_turn.py"
KILL_WINDOW_TIMEOUT = 30.0
_THREAD_ID = "accept-kill9-command-inflight"


def _tool_resp(call_id: str, name: str, arguments: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "content": f"call {name}",
        "tool_calls": [{"call_id": call_id, "name": name, "arguments": dict(arguments)}],
        "usage": {"input_tokens": 2, "output_tokens": 2},
        "finish_reason": "tool_calls",
    }


RESP_BASH_YIELD = _tool_resp(
    "c1", "bash", {"command": "touch cmd-marker && sleep 60", "yield_time_seconds": 5}
)
RESP_READ_TARGET = _tool_resp("c2", "read", {"path": "target.txt"})
RESP_DONE = {
    "content": "done",
    "tool_calls": [],
    "usage": {"input_tokens": 1, "output_tokens": 1},
    "finish_reason": "stop",
}


# ---------------------------------------------------------------------------
# 子进程与观测装置(与 test_kill9_resume.py 同风格)
# ---------------------------------------------------------------------------


def _prepare(tmp_path: Path) -> Path:
    workspace = tmp_path / "ws"
    workspace.mkdir()
    (workspace / ".praxis").mkdir()
    (workspace / "target.txt").write_text("inflight\n", encoding="utf-8")
    return workspace


def _start_child(
    tmp_path: Path, workspace: Path, specs: Sequence[Mapping[str, Any]]
) -> subprocess.Popen[bytes]:
    script = tmp_path / "responses-d.json"
    script.write_text(json.dumps(list(specs)), encoding="utf-8")
    return subprocess.Popen(
        [sys.executable, str(CHILD_SCRIPT), str(workspace), str(script), _THREAD_ID, "d"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def _open_store(workspace: Path) -> Store:
    return Store(workspace / ".praxis" / "praxis.sqlite")


def _calls(items: Sequence[Item], call_id: str) -> list[Item]:
    return [i for i in items if i.kind == ItemKind.TOOL_CALL and i.payload.get("call_id") == call_id]


def _running_session(items: Sequence[Item]) -> Mapping[str, Any] | None:
    for item in items:
        if item.kind != ItemKind.TOOL_RESULT or item.payload.get("call_id") != "c1":
            continue
        metadata = item.payload.get("metadata")
        if isinstance(metadata, Mapping) and metadata.get("state") == "running":
            return metadata
    return None


def _wait_for(
    store: Store,
    proc: subprocess.Popen[bytes],
    pred: Callable[[list[Item]], bool],
    description: str,
) -> list[Item]:
    deadline = time.monotonic() + KILL_WINDOW_TIMEOUT
    while True:
        rc = proc.poll()
        if rc is not None:
            stderr = proc.stderr.read().decode(errors="replace") if proc.stderr else ""
            pytest.fail(f"child exited before kill window (rc={rc}): {stderr[-2000:]}")
        items = store.read_items(_THREAD_ID)
        if pred(items):
            return items
        if time.monotonic() > deadline:
            seen = [(i.seq, i.kind, i.payload.get("call_id")) for i in items]
            pytest.fail(f"timeout waiting for {description}; items so far: {seen}")
        time.sleep(0.02)


def _kill9(proc: subprocess.Popen[bytes]) -> None:
    os.kill(proc.pid, signal.SIGKILL)
    proc.wait(timeout=10)
    assert proc.returncode == -signal.SIGKILL  # 真实 9 号信号,无任何清理


def _cleanup(proc: subprocess.Popen[bytes], store: Store) -> None:
    if proc.poll() is None:
        proc.kill()
        proc.wait(timeout=10)
    store.close()


def _sole_running_turn(store: Store) -> str:
    turns = list_turns(store)
    assert len(turns) == 1, f"expected exactly one turn, got {turns!r}"
    assert turns[0].status == "running"
    return turns[0].turn_id


def _status_after_open(workspace: Path, turn_id: str) -> str:
    """新开 Session 即 sweep(§5.8):不手动调 recovery,状态必须已 interrupted。"""

    async def _go() -> str:
        session = open_session(workspace, ScriptedAdapter([]), thread_id=_THREAD_ID)
        try:
            return session.status(turn_id).status
        finally:
            session.close()

    return asyncio.run(_go())


def _pgid_members(pgid: int) -> list[str]:
    completed = subprocess.run(
        ["ps", "-A", "-o", "pid=,pgid="],
        capture_output=True,
        text=True,
        check=False,
        timeout=10,
    )
    members: list[str] = []
    for line in completed.stdout.splitlines():
        parts = line.split()
        if len(parts) == 2 and parts[1] == str(pgid):
            members.append(parts[0])
    return members


# ---------------------------------------------------------------------------
# 验收
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not probe_backend().available, reason="no sandbox backend")
def test_kill9_during_background_command_sweeps_orphan_group(tmp_path: Path) -> None:
    workspace = _prepare(tmp_path)
    marker_file = workspace / "cmd-marker"

    proc = _start_child(
        tmp_path, workspace, [RESP_BASH_YIELD, RESP_READ_TARGET, RESP_DONE]
    )
    store = _open_store(workspace)
    try:
        # c1 是 yield 的 bash:tool_result 已带 running session_id;第二条
        # model_response 已落 ⇒ 循环已进入 c2 的 execute(变体 d 的 45s 缝),
        # kill 落在这段飞行窗口里。
        items = _wait_for(
            store,
            proc,
            lambda its: _running_session(its) is not None
            and sum(1 for i in its if i.kind == ItemKind.MODEL_RESPONSE) >= 2,
            "background bash session running and second model_response written",
        )
        turn_id = _sole_running_turn(store)
        session_metadata = _running_session(items)
        assert session_metadata is not None
        session_id = str(session_metadata["session_id"])
        pgid = int(session_metadata["pgid"])
        assert marker_file.exists()  # 命令真的开始执行(touch 先于 sleep)
        assert pgid > 0
        _kill9(proc)
    finally:
        _cleanup(proc, store)

    # 宿主 SIGKILL 后:工作区副作用在、marker 登记在(无用户态清理能删它)。
    assert marker_file.exists()
    marker_path = command_jobs_dir() / f"{session_id}.json"
    assert marker_path.exists(), f"expected crash marker {marker_path}"

    # 1) 宿主重启(本进程的全新注册表) + marker 仍在 ⇒ manage_command 报 lost。
    #    注意顺序:必须**先于**清扫,否则注册表看到的是 not_found(两者语义不同)。
    async def _manage() -> Any:
        sessions = CommandSessions(workspace)
        try:
            tool = create_manage_command_tool(sessions)
            return await tool.run(
                {"session_id": session_id, "action": "status"}, ToolExecContext()
            )
        finally:
            await sessions.aclose()

    result = asyncio.run(_manage())
    assert result.is_error is True
    assert result.error_code == "command_session_lost"

    # 2) 真实清扫:宿主已死、pgid/命令行匹配 ⇒ 整组 SIGTERM→SIGKILL、删 marker。
    summaries = sweep_orphan_command_jobs(kill_group=signal_group)
    mine = [summary for summary in summaries if summary.session_id == session_id]
    assert len(mine) == 1, f"expected exactly one summary for {session_id}: {summaries!r}"
    assert mine[0].state == "killed", mine[0]
    assert not marker_path.exists()

    # 3) ps 复查:整个进程组真的消失(不是只删登记)。
    deadline = time.monotonic() + 10.0
    while _pgid_members(pgid) and time.monotonic() < deadline:
        time.sleep(0.05)
    assert _pgid_members(pgid) == []

    # 4) 重开 Session:sweep-on-open 把 dead turn 标 interrupted。
    assert _status_after_open(workspace, turn_id) == "interrupted"


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-v"]))