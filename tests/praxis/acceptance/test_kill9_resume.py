"""Task 16 阶段验收:真实 kill -9 续跑矩阵(执行文档 §4.5/§5.8;设计 §7.1)。

不是模拟:三个变体各起一个真实 OS 子进程(``_child_run_turn.py``,sys.executable),
父进程轮询 store 确认崩溃点后立即 SIGKILL + waitpid——无任何用户态清理;随后
在**本测试进程**里经公开 SDK 入口(open_session/resume)完成 sweep-on-open 与
带确认续跑,断言全部落在 item 日志与真实文件系统上。

三变体(§5.8 崩溃点 × §5.9 决策矩阵的三个分支):

A mid-bash(started+非幂等):kill 落在 ``touch marker && sleep 30`` 执行中
  (tool_call 已写、marker 已真实出现)。resume(confirm=True) 必须停在
  PAUSED/outcome_unknown(绝不自动重放未知副作用),人工 reconcile 后才收尾;
  关键断言:marker mtime 全程不变、c1 只有一条 tool_call——bash 没有第二次执行。
B model_response 已落、dispatch 之前:故障注入缝 = patch 执行循环对管道
  ``execute`` 的调用点(turn.py 按名 from-import,运行时查找的是
  ``praxis.core.turn.execute`` 模块属性,patch 必须打在查找处;brief 所称
  "patch pipeline.execute" 即此落点)。kill 后 resume(confirm=True) 为 c1
  写 not_executed 结果、直接续跑剧本收尾;断言 read 从未执行(零 tool_call item)。
C mid-幂等读:故障注入缝 = dataclasses.replace 把 read.run 换成
  "睡 45s 再真实读"的协程。tool_call 已写、执行中途被 kill;
  resume(confirm=True) 走 started+幂等 → 自动重放(重分发再写一条 tool_call,
  共 2 条,是幂等工具的预期行为),真实读结果正确、一次 outcome_unknown 都没有。

每变体另断言:interrupted turn_status item 携带真实 last_seq(续跑锚点,
Task 8 的 write_status 语义在 SDK 路径上端到端成立),且 interrupted 是父进程
open_session 内 sweep 自动收编的(没有任何手工 sweep 调用)。
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

from praxis.core.statemachine import TurnStatus
from praxis.interfaces.sdk import (
    Item,
    ItemKind,
    ModelResponse,
    ModelToolCall,
    ScriptedAdapter,
    Usage,
    open_session,
)
from praxis.store.log import Store
from praxis.store.turns import get_turn, list_turns

CHILD_SCRIPT = Path(__file__).resolve().parent / "_child_run_turn.py"
# 子进程要冷启动解释器 + import praxis;30s 上限对 CI 富余,失败也能快速定位。
KILL_WINDOW_TIMEOUT = 30.0

# 剧本 spec(与 _child_run_turn.py 的 responses.json 形状一致;父子同源)。
RESP_DONE: dict[str, Any] = {
    "content": "done",
    "tool_calls": [],
    "usage": {"input_tokens": 1, "output_tokens": 1},
    "finish_reason": "stop",
}


def _tool_resp(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    return {
        "content": f"call {name}",
        "tool_calls": [{"call_id": "c1", "name": name, "arguments": arguments}],
        "usage": {"input_tokens": 2, "output_tokens": 2},
        "finish_reason": "tool_calls",
    }


RESP_BASH_MARKER = _tool_resp("bash", {"command": "touch marker && sleep 30"})
RESP_READ_TARGET = _tool_resp("read", {"path": "target.txt"})


# ---------------------------------------------------------------------------
# 子进程与观测装置
# ---------------------------------------------------------------------------


def _prepare(tmp_path: Path, thread_id: str) -> Path:
    workspace = tmp_path / f"ws-{thread_id}"
    workspace.mkdir()
    (workspace / ".praxis").mkdir()
    return workspace


def _start_child(
    tmp_path: Path, workspace: Path, specs: Sequence[Mapping[str, Any]], thread_id: str, variant: str
) -> subprocess.Popen[bytes]:
    script = tmp_path / f"responses-{variant}.json"
    script.write_text(json.dumps(list(specs)), encoding="utf-8")
    return subprocess.Popen(
        [sys.executable, str(CHILD_SCRIPT), str(workspace), str(script), thread_id, variant],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def _open_store(workspace: Path) -> Store:
    return Store(workspace / ".praxis" / "praxis.sqlite")


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


def _wait_for(
    store: Store,
    thread_id: str,
    proc: subprocess.Popen[bytes],
    pred: Callable[[list[Item]], bool],
    description: str,
) -> list[Item]:
    """轮询跨进程可见的 item 日志,直到崩溃条件成立;子进程早退=装置失败。"""
    deadline = time.monotonic() + KILL_WINDOW_TIMEOUT
    while True:
        rc = proc.poll()
        if rc is not None:
            stderr = proc.stderr.read().decode(errors="replace") if proc.stderr else ""
            pytest.fail(f"child exited before kill window (rc={rc}): {stderr[-2000:]}")
        items = store.read_items(thread_id)
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


def _sole_running_turn(store: Store) -> str:
    turns = list_turns(store)
    assert len(turns) == 1, f"expected exactly one turn, got {turns!r}"
    assert turns[0].status == "running"
    return turns[0].turn_id


def _cleanup(proc: subprocess.Popen[bytes], store: Store) -> None:
    if proc.poll() is None:  # 异常路径也不留活体(预期路径下进程已被 SIGKILL)
        proc.kill()
        proc.wait(timeout=10)
    store.close()


# ---------------------------------------------------------------------------
# SDK 公开入口(续跑侧)
# ---------------------------------------------------------------------------


def _status_after_open(workspace: Path, thread_id: str, turn_id: str) -> str:
    """新开 Session 即 sweep(§5.8):不手动调 recovery,状态必须已 interrupted。"""

    async def _go() -> str:
        session = open_session(workspace, ScriptedAdapter([]), thread_id=thread_id)
        try:
            return session.status(turn_id).status
        finally:
            session.close()

    return asyncio.run(_go())


def _resume(
    workspace: Path,
    thread_id: str,
    turn_id: str,
    specs: Sequence[Mapping[str, Any]],
    *,
    confirm: bool = False,
    reconcile: Mapping[str, str] | None = None,
) -> Any:
    async def _go() -> Any:
        session = open_session(workspace, ScriptedAdapter([_response(s) for s in specs]), thread_id=thread_id)
        try:
            return await session.resume(turn_id, confirm=confirm, reconcile=reconcile)
        finally:
            session.close()

    return asyncio.run(_go())


# ---------------------------------------------------------------------------
# item 读取断言助手
# ---------------------------------------------------------------------------


def _calls(items: Sequence[Item], call_id: str) -> list[Item]:
    return [i for i in items if i.kind == ItemKind.TOOL_CALL and i.payload.get("call_id") == call_id]


def _results(items: Sequence[Item], call_id: str) -> list[Item]:
    return [i for i in items if i.kind == ItemKind.TOOL_RESULT and i.payload.get("call_id") == call_id]


def _model_response_with_call(items: Sequence[Item], call_id: str) -> bool:
    for item in items:
        if item.kind != ItemKind.MODEL_RESPONSE:
            continue
        raw = item.payload.get("tool_calls")
        if isinstance(raw, (list, tuple)) and any(
            isinstance(entry, Mapping) and entry.get("call_id") == call_id for entry in raw
        ):
            return True
    return False


def _result_text(item: Item) -> str:
    content = item.payload["content"]
    assert isinstance(content, Sequence) and len(content) == 1
    block = content[0]
    assert isinstance(block, Mapping)
    data = block["data"]
    assert isinstance(data, Mapping)
    text = data["text"]
    assert isinstance(text, str)
    return text


def _assert_interrupted_anchor(items: Sequence[Item]) -> None:
    """sweep 写下的 interrupted turn_status:reason 真实、last_seq = 打断瞬间的尾部 seq。"""
    marks = [
        i
        for i in items
        if i.kind == ItemKind.TURN_STATUS and i.payload.get("to") == "interrupted"
    ]
    assert len(marks) == 1, f"expected one interrupted turn_status, got {len(marks)}"
    mark = marks[0]
    assert mark.payload.get("from") == "running"
    assert mark.payload.get("reason") == "executor_process_died"
    last_seq = mark.payload.get("last_seq")
    assert isinstance(last_seq, int) and last_seq > 0
    prior = [i.seq for i in items if i.seq < mark.seq]
    assert prior and max(prior) == last_seq


# ---------------------------------------------------------------------------
# 变体 A:kill 在 bash 执行中(started+非幂等)→ outcome_unknown → reconcile
# ---------------------------------------------------------------------------


def test_kill9_mid_bash_resume_no_double_side_effect(tmp_path: Path) -> None:
    thread_id = "accept-kill9-a"
    workspace = _prepare(tmp_path, thread_id)
    marker = workspace / "marker"

    proc = _start_child(tmp_path, workspace, [RESP_BASH_MARKER, RESP_DONE], thread_id, "a")
    store = _open_store(workspace)
    mtime_after_kill = 0
    try:
        # touch 在 sleep 之前:marker 出现 ⇒ 副作用已发生且 bash 仍在 sleep 30 中途。
        items = _wait_for(
            store,
            thread_id,
            proc,
            lambda its: bool(_calls(its, "c1")) and marker.exists(),
            "bash tool_call written and marker touched (mid sleep 30)",
        )
        turn_id = _sole_running_turn(store)
        call_item = _calls(items, "c1")[0]
        assert call_item.payload.get("tool_name") == "bash"
        assert call_item.payload.get("idempotent") is False  # 决策矩阵输入:非幂等
        assert _results(items, "c1") == []
        _kill9(proc)
        mtime_after_kill = marker.stat().st_mtime_ns
    finally:
        _cleanup(proc, store)

    assert marker.exists()  # touch 先于 kill:副作用真实发生过了

    assert _status_after_open(workspace, thread_id, turn_id) == "interrupted"

    # resume(confirm=True):started+非幂等 → 只可能对账,绝不自动重放。
    r1 = _resume(workspace, thread_id, turn_id, [], confirm=True)
    assert r1.status is TurnStatus.PAUSED
    assert r1.reason == "outcome_unknown"
    assert r1.unknown_calls == ("c1",)

    # 人工对账 ok → 剧本收尾;r2 用全新 Session(spec 空=若真重放 bash 会先炸剧本耗尽)。
    r2 = _resume(workspace, thread_id, turn_id, [RESP_DONE], reconcile={"c1": "ok"})
    assert r2.status is TurnStatus.COMPLETED
    assert r2.output == "done"
    assert r2.reason == "model_finished"
    assert r2.unknown_calls == ()

    # 关键不变式:marker mtime 从 kill 到终局不变、c1 只有一条 tool_call ⇒
    # bash 从未第二次执行(sleep 30 若重跑,这里的等待早就超时了)。
    assert marker.stat().st_mtime_ns == mtime_after_kill

    store2 = _open_store(workspace)
    try:
        items = store2.read_items(thread_id)
        assert len(_calls(items, "c1")) == 1
        assert [r.payload.get("status") for r in _results(items, "c1")] == ["outcome_unknown", "ok"]
        assert _results(items, "c1")[-1].payload.get("is_error") is False
        _assert_interrupted_anchor(items)
        turn = get_turn(store2, turn_id)
        assert turn is not None and turn.status == "completed"
    finally:
        store2.close()


# ---------------------------------------------------------------------------
# 变体 B:kill 在 model_response 之后、dispatch 之前 → not_executed 自动续跑
# ---------------------------------------------------------------------------


def test_kill9_after_model_before_dispatch_resume_continues(tmp_path: Path) -> None:
    thread_id = "accept-kill9-b"
    workspace = _prepare(tmp_path, thread_id)
    target = workspace / "target.txt"
    target.write_text("before-dispatch\n", encoding="utf-8")

    proc = _start_child(tmp_path, workspace, [RESP_READ_TARGET, RESP_DONE], thread_id, "b")
    store = _open_store(workspace)
    try:
        # 缝 hold 住 execute 45s:model_response 落库时 tool_call 必然还未写。
        items = _wait_for(
            store,
            thread_id,
            proc,
            lambda its: _model_response_with_call(its, "c1"),
            "model_response persisted, seam holding before dispatch",
        )
        turn_id = _sole_running_turn(store)
        assert _calls(items, "c1") == []  # 崩溃点确在 dispatch 之前
        _kill9(proc)
    finally:
        _cleanup(proc, store)

    assert _status_after_open(workspace, thread_id, turn_id) == "interrupted"

    # resume(confirm=True):未 started → not_executed 结果,循环继续,剧本收尾。
    r = _resume(workspace, thread_id, turn_id, [RESP_DONE], confirm=True)
    assert r.status is TurnStatus.COMPLETED
    assert r.output == "done"
    assert r.reason == "model_finished"
    assert r.unknown_calls == () and r.pending_approvals == ()

    store2 = _open_store(workspace)
    try:
        items = store2.read_items(thread_id)
        assert _calls(items, "c1") == []  # read 从未执行(更没被重放)
        results = _results(items, "c1")
        assert len(results) == 1
        assert results[0].payload.get("status") == "not_executed"
        assert "interrupted before dispatch" in _result_text(results[0])
        _assert_interrupted_anchor(items)  # last_seq 恰指向 model_response
        turn = get_turn(store2, turn_id)
        assert turn is not None and turn.status == "completed"
    finally:
        store2.close()
    assert target.read_text(encoding="utf-8") == "before-dispatch\n"


# ---------------------------------------------------------------------------
# 变体 C:kill 在幂等 read 执行中 → resume 自动重跑
# ---------------------------------------------------------------------------

FILE_DATA = "praxis kill-9 idempotent replay\n" * 100  # 3.5KB < read 默认页 16KB


def test_kill9_mid_idempotent_read_resume_auto_retries(tmp_path: Path) -> None:
    thread_id = "accept-kill9-c"
    workspace = _prepare(tmp_path, thread_id)
    (workspace / "target.txt").write_text(FILE_DATA, encoding="utf-8")

    proc = _start_child(tmp_path, workspace, [RESP_READ_TARGET, RESP_DONE], thread_id, "c")
    store = _open_store(workspace)
    try:
        # 慢 read:tool_call 已写、run 睡 45s,kill 必然落在执行中途。
        items = _wait_for(
            store,
            thread_id,
            proc,
            lambda its: bool(_calls(its, "c1")),
            "idempotent read tool_call written (mid slow execution)",
        )
        turn_id = _sole_running_turn(store)
        call_item = _calls(items, "c1")[0]
        assert call_item.payload.get("idempotent") is True  # 决策矩阵输入:可重放
        assert _results(items, "c1") == []
        _kill9(proc)
    finally:
        _cleanup(proc, store)

    assert _status_after_open(workspace, thread_id, turn_id) == "interrupted"

    # resume(confirm=True):started+幂等 → 自动重跑(无需人工对账),一气呵成。
    r = _resume(workspace, thread_id, turn_id, [RESP_DONE], confirm=True)
    assert r.status is TurnStatus.COMPLETED
    assert r.output == "done"
    assert r.reason == "model_finished"
    assert r.unknown_calls == () and r.pending_approvals == ()

    store2 = _open_store(workspace)
    try:
        items = store2.read_items(thread_id)
        # 自动重放 = 重分发:再写一条 tool_call 是幂等工具的预期行为(共 2 条)。
        assert len(_calls(items, "c1")) == 2
        results = _results(items, "c1")
        assert len(results) == 1  # 只有一份结局,无 outcome_unknown
        assert results[0].payload.get("status") == "ok"
        assert results[0].payload.get("is_error") is False
        assert _result_text(results[0]) == FILE_DATA  # 真实文件系统读回的内容
        _assert_interrupted_anchor(items)
        turn = get_turn(store2, turn_id)
        assert turn is not None and turn.status == "completed"
    finally:
        store2.close()


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-v"]))
