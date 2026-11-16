"""Task 21 Step 2③:kill -9 落在 PreToolUse hook 飞行中(设计 §5.8/§5.14)。

仿 ``test_kill9_resume.py``  child-driver 风格:真实 OS 子进程
(``_child_run_turn.py``,variant 标签 "hook"——不装任何 seam,默认工具表 +
经 ``PRAXIS_HOOKS_CONFIG`` 装配的真 command hook),父进程轮询 store 确认
``hook_invocation started`` item 落库**且 hook 子进程已在 ps 可见**(飞行中)
后 SIGKILL 宿主——无任何用户态清理。

崩溃面不变量(全部落在 item 日志与真实文件系统上):

1. 重开 Session ⇒ sweep-on-open 自动收编 interrupted(不手动调 recovery);
2. hook 只有 ``started`` 没有 ``settled``(结算事实随宿主死亡丢失);
3. resume(confirm=True) 后模型以**同参数**重发同一调用(同 turn ⇒ 同
   ``hook_invocation_id``)⇒ ``fold_hook_items`` 投影 unknown ⇒ 唯一 decide
   第 1 步 deny(fail-closed 崩溃面:进程是否已产生副作用不可知,绝不假装
   没发生)——reason 逐字 ``hook:sleepy:unknown: crash before settlement``;
4. 重入 **0 次新 spawn**(幂等折叠;counter 文件全程恰一行),fold 不补写
   settled item(phases 恒 ["started"]);调用从未 dispatch(零 tool_call)。

hook 子进程自成新会话(``start_new_session=True``,旧 hooks.py:170 同姿势),
宿主 SIGKILL 不波及它——**产品不为 hook 孤儿建 sweep marker**(与后台命令
会话的 marker 清扫不同);测试装置自行按 ps token 整组回收并复查无残留,
如实记录这一边界(hook 孤儿寿命以其自身命令为界)。

hook 超时/整组 kill/有界排空的运行面由 T4 拥有
(``test_hooks_runner.py``),本文件不重复;伪造 started-only 的管道级
fail-closed 由 T5 ⑧拥有(``test_hooks_pipeline.py::
test_started_without_settled_fails_closed_before_decide``)——本文件证明的是
**真实崩溃**产生同一投影,且经真实 sweep/resume 全链到达。
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

from praxis.interfaces.sdk import Item, ItemKind
from tests.praxis.acceptance.test_kill9_resume import (
    CHILD_SCRIPT,
    RESP_DONE,
    _assert_interrupted_anchor,
    _cleanup,
    _kill9,
    _open_store,
    _resume,
    _sole_running_turn,
    _status_after_open,
    _tool_resp,
    _wait_for,
)

_THREAD_ID = "accept-kill9-hook"
_RESP_READ = _tool_resp("read", {"path": "target.txt"})


def _ps_lines(token: str) -> list[str]:
    completed = subprocess.run(
        ["ps", "-eo", "pid=,args="], capture_output=True, text=True, timeout=20, check=False
    )
    return [row for row in completed.stdout.splitlines() if token in row]


def _wait_hook_process(token: str, timeout: float = 15.0) -> int:
    """同步点:等 hook 子进程真的出现在 ps(不是 sleep 猜时机);返回组长 pid。"""
    deadline = time.monotonic() + timeout
    while True:
        rows = _ps_lines(token)
        if rows:
            return int(rows[0].split()[0])
        if time.monotonic() > deadline:
            pytest.fail(f"hook subprocess never appeared in ps (token={token})")
        time.sleep(0.02)


def _wait_process_gone(token: str, timeout: float = 10.0) -> None:
    deadline = time.monotonic() + timeout
    while _ps_lines(token):
        if time.monotonic() > deadline:
            pytest.fail(f"orphan hook process still alive: {_ps_lines(token)}")
        time.sleep(0.05)


def _hook_items(items: list[Item]) -> list[Item]:
    return [i for i in items if i.kind == ItemKind.HOOK_INVOCATION]


def test_kill9_mid_pretooluse_hook_resume_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace = tmp_path / "ws-hook"
    workspace.mkdir()
    (workspace / ".praxis").mkdir()
    (workspace / "target.txt").write_text("hook crash surface\n", encoding="utf-8")

    # hook 配置与脚本落在 workspace 之外的宿主目录(信任门卫生:0700/0600)。
    token = "kill9hook-" + os.urandom(6).hex()
    host = tmp_path / "host"
    host.mkdir()
    host.chmod(0o700)
    counter = host / "spawns.txt"
    script = host / f"hook-{token}.sh"
    # 真 command hook:记一次 spawn,睡 30s(远大于 kill 窗口),若侥幸活到
    # 终点才返回 allow。timeout_seconds=30(协议上限)保证窗口内不会自行结算。
    script.write_text(
        "#!/bin/sh\n"
        f"echo run >> {counter}\n"
        "sleep 30\n"
        "printf '%s' '{\"block\": false}'\n",
        encoding="utf-8",
    )
    script.chmod(0o755)
    config = host / "hooks.json"
    config.write_text(
        json.dumps(
            [
                {
                    "event": "PreToolUse",
                    "name": "sleepy",
                    "argv": [str(script)],
                    "tool_names": ["read"],
                    "timeout_seconds": 30,
                }
            ]
        ),
        encoding="utf-8",
    )
    config.chmod(0o600)
    # 父进程侧重开/resume 的会话必须装配**同一份** hook 配置(fold 发生在
    # execute 的 ctx.hooks;hooks=None 会整段跳过投影)。conftest 已剥宿主
    # PRAXIS_HOOKS_CONFIG,这里显式指回本测试的配置。
    monkeypatch.setenv("PRAXIS_HOOKS_CONFIG", str(config))

    # 剧本:同参数 read 调用两次 + 收尾。子进程消费第一条后停在 hook 飞行中;
    # resume 侧剧本再发同一调用(同 turn 同参数 ⇒ 同 hook invocation 身份)。
    specs = [_RESP_READ, _RESP_READ, RESP_DONE]
    script_file = tmp_path / "responses-hook.json"
    script_file.write_text(json.dumps(specs), encoding="utf-8")
    child_env = {**os.environ, "PRAXIS_HOOKS_CONFIG": str(config)}
    proc = subprocess.Popen(
        [sys.executable, str(CHILD_SCRIPT), str(workspace), str(script_file), _THREAD_ID, "hook"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=child_env,
    )
    store = _open_store(workspace)
    hook_pid: int | None = None
    try:
        # 崩溃窗口:started item 已落库且 hook 子进程真实在飞。
        items = _wait_for(
            store,
            _THREAD_ID,
            proc,
            lambda its: any(
                i.kind == ItemKind.HOOK_INVOCATION and i.payload.get("phase") == "started"
                for i in its
            ),
            "hook_invocation started item written",
        )
        hook_pid = _wait_hook_process(token)
        # 同步点加固:counter 已写 ⇒ hook 确已执行首行、正停在 sleep 30
        # (仅 ps 可见就 kill 会撞上 sh 启动竞态,echo 可能还没跑)。
        deadline = time.monotonic() + 15.0
        while not counter.exists():
            if time.monotonic() > deadline:
                pytest.fail("hook never wrote its spawn counter")
            time.sleep(0.02)
        turn_id = _sole_running_turn(store)
        started = _hook_items(items)
        assert len(started) == 1
        assert started[0].payload.get("hook") == "sleepy"
        assert started[0].payload.get("event") == "PreToolUse"
        assert not any(i.kind == ItemKind.TOOL_CALL for i in items)  # hook 先于 dispatch
        _kill9(proc)  # 真实 SIGKILL,宿主无任何用户态收尾
    finally:
        _cleanup(proc, store)

    # hook 自成新会话,宿主死亡不波及它:装置按 ps token 整组回收(产品无
    # hook sweep marker——见模块 docstring 的边界注记),并复查无残留。
    assert hook_pid is not None
    try:
        os.killpg(hook_pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    _wait_process_gone(token)
    assert counter.read_text(encoding="utf-8").splitlines() == ["run"]

    # 重开即 sweep:不手动调 recovery,状态必须已 interrupted。
    assert _status_after_open(workspace, _THREAD_ID, turn_id) == "interrupted"
    store2 = _open_store(workspace)
    try:
        items = store2.read_items(_THREAD_ID)
        phases = [i.payload.get("phase") for i in _hook_items(items)]
        assert phases == ["started"]  # 结算事实随宿主死亡丢失
        _assert_interrupted_anchor(items)
    finally:
        store2.close()

    # resume(confirm=True):window#1 未 started 无决策 ⇒ not_executed;循环
    # 继续,模型同参数重发 ⇒ fold unknown ⇒ deny(fail-closed),0 次新 spawn。
    result = _resume(workspace, _THREAD_ID, turn_id, [_RESP_READ, RESP_DONE], confirm=True)
    assert result.status.value == "completed"
    assert result.output == "done"
    assert result.unknown_calls == ()

    store3 = _open_store(workspace)
    try:
        items = store3.read_items(_THREAD_ID)
        # 幂等折叠:全程恰一次 spawn,fold 不补写 settled。
        assert counter.read_text(encoding="utf-8").splitlines() == ["run"]
        assert [i.payload.get("phase") for i in _hook_items(items)] == ["started"]
        # 调用从未 dispatch:零 tool_call;两份结果 = not_executed + denied。
        assert not any(
            i.kind == ItemKind.TOOL_CALL and i.payload.get("call_id") == "c1" for i in items
        )
        results = [
            i for i in items if i.kind == ItemKind.TOOL_RESULT and i.payload.get("call_id") == "c1"
        ]
        assert [r.payload.get("status") for r in results] == ["not_executed", "denied"]
        denied = results[-1]
        assert denied.payload.get("error_code") == "permission_denied"
        assert denied.payload.get("error_message") == "hook:sleepy:unknown: crash before settlement"
        _assert_interrupted_anchor(items)
    finally:
        store3.close()
    # 终局 ps 复查:本实例 hook 无孤儿。
    assert _ps_lines(token) == []


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-v"]))