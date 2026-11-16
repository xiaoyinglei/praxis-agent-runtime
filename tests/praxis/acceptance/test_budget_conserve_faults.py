"""阶段4 T9 验收:崩溃矩阵与取消竞态(设计 §5 硬规则 8 真实故障注入;旧 §13 矩阵行)。

三组场景,全部真子进程/真多进程/真 git(非 mock):

① kill -9 矩阵(8 点位 × 4 观察点):崩溃点 = {allocation 事务前/后、子首
   dispatch 前/后、子 settle 边界(终态 turn_status 写前/后)、extend item
   前/后};观察点 = {重启后父 exposure、子 turn 状态、孤儿进程(子 driver 是
   父进程内 asyncio task,无独立进程——进程级孤儿只可能来自 provisioner git
   调用,ps 复查 pgid 无残留)、重入后守恒等式}。每个点位真 SIGKILL 后:
   - 守恒等式 ``root_used + Σ child_used + Σ active_grant_unreserved == Σ facts``
     经 build_tree 重建成立(Σ facts 独立逐 turn 折叠,不走树聚合);
   - 无永久 active 分配:interrupted 子占用份额可对账(grant−used 下钳),
     级联 cancel_request 收敛后终态结算、reserved 归零;
   - 不重复执行副作用(阶段1 判据:tool_call 计数/文件 mtime/存在性)。
② 取消竞态双进程(旧 §13 行 praxis 形):A 执行子 turn、B 执行父 turn,真 db
   同库(WAL 多写者),文件 barrier 钉两种时序:级联 item 先落 → 子步边界必拦
   (下一次 dispatch 拒,零 tool_call);子 dispatch 先 → tool_call/model_attempt
   痕迹 + cancel 收敛(stop 可追;再叠加 crash → outcome_unknown 可追)。
   父取消经第三连接 request_cancel → B 步边界收敛 CANCELLED → 生产终态挂钩
   cascade_for 向子落 cancel_request(source=parent)。
③ 预算耗尽场景行(§13):逼近总额 → PAUSED(budget_exhausted) → CLI
   ``resume --extend-tokens`` 续跑端到端(真 CLI 子进程 + scripted)。

缝与先例:_child_run_turn.py 的 T9 矩阵变体(八缝,见其 docstring)、
_race_turn.py 双角色驱动、test_kill9_resume.py 的轮询-kill-sweep 装置。
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
from uuid import uuid4

import pytest

from praxis.core.budget import UsageFacts, build_tree, exposure, fold_turn_facts
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
from praxis.protocol.budget import Limits, limits_from_payload
from praxis.store.log import ItemEntry, Store, append_items_tx
from praxis.store.turns import create_turn_tx, get_turn, list_turns

CHILD_SCRIPT = Path(__file__).resolve().parent / "_child_run_turn.py"
RACE_SCRIPT = Path(__file__).resolve().parent / "_race_turn.py"
KILL_WINDOW_TIMEOUT = 30.0
PARENT_THREAD = "accept-fault-parent"

_DIMENSIONS = ("tokens", "cost_micros", "steps", "web_requests", "browser_requests")
_TERMINAL_CHILD_STATES = {"completed": "done", "failed": "done", "cancelled": "cancelled"}
_WRITE_VARIANTS = frozenset({"alloc-before", "alloc-after", "dispatch-after"})


# ---------------------------------------------------------------------------
# 剧本与 workspace 装置
# ---------------------------------------------------------------------------


def _spec(
    content: str,
    calls: tuple[tuple[str, str, dict[str, Any]], ...] = (),
    usage: tuple[int, int] = (2, 2),
) -> dict[str, Any]:
    return {
        "content": content,
        "tool_calls": [{"call_id": c, "name": n, "arguments": a} for c, n, a in calls],
        "usage": {"input_tokens": usage[0], "output_tokens": usage[1]},
        "finish_reason": "tool_calls" if calls else "stop",
    }


def _git(cwd: Path, *args: str) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(["git", "-C", str(cwd), *args], capture_output=True, check=True)


def _prepare_workspace(tmp_path: Path, variant: str) -> Path:
    workspace = tmp_path / f"ws-{variant}"
    workspace.mkdir()
    (workspace / ".praxis").mkdir()
    if variant in _WRITE_VARIANTS:
        _git(workspace, "init", "-q")
        _git(workspace, "config", "user.email", "praxis-test@example.com")
        _git(workspace, "config", "user.name", "Praxis Test")
        (workspace / "a.txt").write_text("base-a\n", encoding="utf-8")
        _git(workspace, "add", ".")
        _git(workspace, "commit", "-qm", "init")
    else:
        (workspace / "target.txt").write_text("fault matrix target\n", encoding="utf-8")
    return workspace


def _matrix_scripts(variant: str) -> dict[str, Any]:
    mode = "write" if variant in _WRITE_VARIANTS else "read_only"
    spawn = _spec(
        "", (("s1", "spawn_subagent", {"task": f"[child] {variant} task", "workspace_mode": mode}),)
    )
    if variant.startswith("extend-"):
        parent = [
            spawn,
            _spec("", (("c2", "read", {"path": "target.txt"}),)),
            _spec("", (("c3", "read", {"path": "target.txt"}),)),
            _spec("", (("c4", "read", {"path": "target.txt"}),)),
            _spec("parent done"),
        ]
        child = [_spec("", (("h1", "child_hold", {}),)), _spec("child done")]
        budget = {"tokens": 20, "steps": 20, "web_requests": 10, "browser_requests": 5}
    else:
        parent = [
            spawn,
            _spec("", (("w1", "wait_subagent", {"child_thread_id": "PLACEHOLDER-1", "timeout": 30}),)),
            _spec("parent done"),
        ]
        if variant == "dispatch-before":
            child = [_spec("", (("c1", "read", {"path": "target.txt"}),)), _spec("child done")]
        elif variant == "dispatch-after":
            child = [
                _spec("", (("c1", "write", {"path": "side.txt", "content": "child side effect\n"}),)),
                _spec("child done"),
            ]
        else:
            child = [_spec("child done")]
        budget = {"tokens": 1000, "steps": 20, "web_requests": 10, "browser_requests": 5}
    scripts: dict[str, Any] = {"parent": parent, "child": child, "budget": budget, "mode": mode}
    if variant.startswith("extend-"):
        scripts["extend"] = {"tokens": 50}
    return scripts


def _start_driver(
    tmp_path: Path, workspace: Path, payload: Mapping[str, Any], thread_id: str, variant: str
) -> subprocess.Popen[bytes]:
    script = tmp_path / f"scripts-{variant}.json"
    script.write_text(json.dumps(payload), encoding="utf-8")
    return subprocess.Popen(
        [sys.executable, str(CHILD_SCRIPT), str(workspace), str(script), thread_id, variant],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


# ---------------------------------------------------------------------------
# 轮询 / kill / 孤儿进程复查
# ---------------------------------------------------------------------------


def _poll(
    pred: Callable[[], bool],
    description: str,
    procs: Sequence[subprocess.Popen[bytes]] = (),
    timeout: float = KILL_WINDOW_TIMEOUT,
) -> None:
    deadline = time.monotonic() + timeout
    while True:
        for proc in procs:
            rc = proc.poll()
            if rc is not None:
                stderr = proc.stderr.read().decode(errors="replace") if proc.stderr else ""
                stdout = proc.stdout.read().decode(errors="replace") if proc.stdout else ""
                pytest.fail(f"driver exited before window (rc={rc}): {stderr[-2000:]} {stdout[-500:]}")
        if pred():
            return
        if time.monotonic() > deadline:
            pytest.fail(f"timeout waiting for {description}")
        time.sleep(0.02)


def _kill9(proc: subprocess.Popen[bytes]) -> None:
    os.kill(proc.pid, signal.SIGKILL)
    proc.wait(timeout=10)
    assert proc.returncode == -signal.SIGKILL  # 真实 9 号信号,无任何用户态清理


def _cleanup_proc(proc: subprocess.Popen[bytes]) -> None:
    if proc.poll() is None:
        proc.kill()
        proc.wait(timeout=10)


def _assert_no_workspace_processes(tmp_path: Path) -> None:
    """pgid 无残留复查:任何存活进程的命令行不得引用本测试的 workspace 子树。

    子 driver 是父进程内 asyncio task(SIGKILL 即全灭);进程级孤儿只可能来自
    provisioner 的 git 调用(同步 subprocess.run,kill 点位前已完成)——ps 实测
    兜底,不靠假设。driver 已 waitpid 收尸,不在 ps 表里。
    """
    completed = subprocess.run(
        ["ps", "-ax", "-o", "pid=,pgid=,args="], capture_output=True, text=True, timeout=10
    )
    needle = str(tmp_path)
    offenders = [
        line
        for line in completed.stdout.splitlines()
        if needle in line and not line.strip().startswith(f"{os.getpid()} ")
    ]
    assert offenders == [], f"orphan processes referencing workspace: {offenders}"


# ---------------------------------------------------------------------------
# 守恒等式重建(Σ facts 独立逐 turn 折叠,不走 build_tree 的树聚合)
# ---------------------------------------------------------------------------


def _family_facts(
    store: Store, thread_id: str, turn_id: str
) -> tuple[UsageFacts, list[tuple[Any, UsageFacts, str]]]:
    parent_items = [i for i in store.read_items(thread_id) if i.turn_id == turn_id]
    total = fold_turn_facts(parent_items, unit_costs=None)
    children: list[tuple[Any, UsageFacts, str]] = []
    for item in parent_items:
        if item.kind != ItemKind.CHILD_ALLOCATION:
            continue
        payload = item.payload
        child_items = [
            i
            for i in store.read_items(str(payload["child_thread_id"]))
            if i.turn_id == payload["child_root_turn_id"]
        ]
        used = fold_turn_facts(child_items, unit_costs=None)
        total = total + used
        statuses = [i for i in child_items if i.kind == ItemKind.TURN_STATUS]
        state = "active"
        if statuses:
            to = max(statuses, key=lambda i: i.seq).payload.get("to")
            state = _TERMINAL_CHILD_STATES.get(str(to), "active")
        children.append((limits_from_payload(payload["grant"]), used, state))
    return total, children


def _assert_conservation(store: Store, thread_id: str, turn_id: str) -> Any:
    """root_used + Σ child_used + Σ active_grant_unreserved == Σ facts(逐维)。"""
    turn = get_turn(store, turn_id)
    assert turn is not None
    raw_budget = turn.binding.get("budget")
    root_limits = limits_from_payload(raw_budget if isinstance(raw_budget, Mapping) else {})
    node = build_tree(store, thread_id=thread_id, turn_id=turn_id, root_limits=root_limits, unit_costs=None)
    facts, children = _family_facts(store, thread_id, turn_id)
    assert node.used == facts, f"tree used diverges from independent fact fold: {node.used} vs {facts}"
    reserved = {dim: 0 for dim in _DIMENSIONS}
    under = False
    for grant, used, state in children:
        if state != "active":
            continue
        for dim in _DIMENSIONS:
            limit = getattr(grant, dim)
            if limit is None:
                continue
            spent = getattr(used, dim)
            if spent > limit:
                under = True
            reserved[dim] += max(limit - spent, 0)
    expected_reserved = UsageFacts(**reserved)
    assert node.reserved == expected_reserved
    assert exposure(node) == facts + expected_reserved
    assert node.under_accounted is under
    return node


def _assert_fully_settled(store: Store, thread_id: str, turn_id: str) -> None:
    """终局:无永久 active——全部子终态、reserved 逐维归零、exposure == Σ facts。"""
    node = _assert_conservation(store, thread_id, turn_id)
    assert node.reserved == UsageFacts(), f"permanent active reservation: {node.reserved}"
    assert all(state != "active" for _grant, _used, state in _family_facts(store, thread_id, turn_id)[1])
    assert list_turns(store, status="running") == []


# ---------------------------------------------------------------------------
# item 读取 / SDK 续跑助手
# ---------------------------------------------------------------------------


def _calls(items: Sequence[Item], call_id: str) -> list[Item]:
    return [i for i in items if i.kind == ItemKind.TOOL_CALL and i.payload.get("call_id") == call_id]


def _results(items: Sequence[Item], call_id: str) -> list[Item]:
    return [i for i in items if i.kind == ItemKind.TOOL_RESULT and i.payload.get("call_id") == call_id]


def _of_kind(items: Sequence[Item], kind: str) -> list[Item]:
    return [i for i in items if i.kind == kind]


def _first_allocation(store: Store) -> Mapping[str, Any] | None:
    for item in store.read_items(PARENT_THREAD):
        if item.kind == ItemKind.CHILD_ALLOCATION:
            return item.payload
    return None


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


def _sweep_statuses(workspace: Path, thread_id: str) -> dict[str, str]:
    """新开 Session 即 sweep-on-open;返回全库 turn_id → status(不手动调 recovery)。"""

    async def _go() -> dict[str, str]:
        session = open_session(workspace, ScriptedAdapter([]), thread_id=thread_id)
        try:
            return {t.turn_id: t.status for t in list_turns(session.store)}
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
    extend: Mapping[str, int] | None = None,
    db_path: Path | None = None,
    subagents: bool = False,
) -> Any:
    async def _go() -> Any:
        kwargs: dict[str, Any] = {} if db_path is None else {"db_path": db_path}
        session = open_session(
            workspace,
            ScriptedAdapter([_response(s) for s in specs]),
            thread_id=thread_id,
            # 父续跑必须挂 agent 五件:wait 的幂等重放判据来自工具表,
            # 缺表则 unknown_tool fail-safe 按非幂等走 outcome_unknown(假阳性)。
            # 子续跑不挂(生产子表 = 父表减 agent 五件,同形)。
            enable_subagents=subagents or thread_id == PARENT_THREAD,
            **kwargs,
        )
        try:
            return await session.resume(turn_id, confirm=confirm, reconcile=reconcile, extend=extend)
        finally:
            session.close()

    return asyncio.run(_go())


# ---------------------------------------------------------------------------
# kill 矩阵公共装置:窗口轮询 → SIGKILL → 孤儿复查 → sweep 观察
# ---------------------------------------------------------------------------


def _window_pred(store: Store, variant: str, workspace: Path) -> Callable[[], bool]:
    def _child_items() -> list[Item]:
        alloc = _first_allocation(store)
        if alloc is None:
            return []
        return store.read_items(str(alloc["child_thread_id"]))

    def _child_status() -> str:
        alloc = _first_allocation(store)
        if alloc is None:
            return ""
        turn = get_turn(store, str(alloc["child_root_turn_id"]))
        return turn.status if turn is not None else ""

    def _parent_items() -> list[Item]:
        return store.read_items(PARENT_THREAD)

    def _parent_status() -> str:
        turns = list_turns(store, thread_id=PARENT_THREAD)
        return turns[0].status if turns else ""

    if variant == "alloc-before":
        return lambda: (
            any(
                i.kind == ItemKind.TOOL_CALL and i.payload.get("tool_name") == "spawn_subagent"
                for i in _parent_items()
            )
            and _first_allocation(store) is None
        )
    if variant == "alloc-after":
        return lambda: _first_allocation(store) is not None and _child_status() == "running" and _child_items() == []
    if variant == "dispatch-before":
        def _pred() -> bool:
            items = _child_items()
            return (
                any(
                    i.kind == ItemKind.MODEL_RESPONSE and i.payload.get("tool_calls")
                    for i in items
                )
                and not _of_kind(items, ItemKind.TOOL_CALL)
            )
        return _pred
    if variant == "dispatch-after":
        def _pred() -> bool:
            alloc = _first_allocation(store)
            if alloc is None:
                return False
            side = Path(str(alloc["workspace"])) / "side.txt"
            return bool(_calls(_child_items(), "c1")) and not side.exists()
        return _pred
    if variant == "settle-before":
        def _pred() -> bool:
            items = _child_items()
            return (
                any(
                    i.kind == ItemKind.MODEL_RESPONSE and i.payload.get("content") == "child done"
                    for i in items
                )
                and _child_status() == "running"
            )
        return _pred
    if variant == "settle-after":
        return lambda: _child_status() == "completed" and _parent_status() == "running"
    if variant == "extend-before":
        # 缝 hold 在 extension 写事务内(BEGIN IMMEDIATE 持写锁):轮询只能走
        # 无锁 read_items——get_turn/list_turns 的读也经 BEGIN IMMEDIATE,会撞锁。
        def _pred() -> bool:
            items = _parent_items()
            statuses = _of_kind(items, ItemKind.TURN_STATUS)
            if not statuses:
                return False
            last = max(statuses, key=lambda i: i.seq)
            paused = any(
                i.payload.get("to") == "paused" and i.payload.get("reason") == "budget_exhausted"
                for i in statuses
            )
            return (
                paused
                and last.payload.get("to") == "running"
                and not _of_kind(items, ItemKind.BUDGET_EXTENSION)
            )
        return _pred
    if variant == "extend-after":
        def _pred() -> bool:
            items = _parent_items()
            statuses = _of_kind(items, ItemKind.TURN_STATUS)
            return (
                bool(_of_kind(items, ItemKind.BUDGET_EXTENSION))
                and bool(statuses)
                and max(statuses, key=lambda i: i.seq).payload.get("to") == "running"
            )
        return _pred
    raise ValueError(f"unknown matrix variant: {variant}")


def _matrix_kill(tmp_path: Path, variant: str) -> tuple[Path, Store, str, Mapping[str, Any] | None]:
    """起 driver → 轮询崩溃窗口 → SIGKILL → pgid 复查;返回 (ws, store, 父 turn, allocation)。

    store 交给调用方续用(调用方 finally 关闭);异常路径这里收尸,不留活体。
    """
    workspace = _prepare_workspace(tmp_path, variant)
    proc = _start_driver(tmp_path, workspace, _matrix_scripts(variant), PARENT_THREAD, variant)
    store = Store(workspace / ".praxis" / "praxis.sqlite")
    try:
        _poll(_window_pred(store, variant, workspace), f"{variant} kill window", procs=[proc])
        # kill 前只走无锁 read_items(事务内 hold 的点位会持写锁);turn 行核对
        # 放在 SIGKILL 之后(锁随进程死亡释放)。
        items = store.read_items(PARENT_THREAD)
        parent_turn = str(items[0].turn_id)
        statuses = _of_kind(items, ItemKind.TURN_STATUS)
        if statuses:  # 尚无转换的点位没有 turn_status item(turn 行 INSERT running)
            assert max(statuses, key=lambda i: i.seq).payload.get("to") == "running"
        alloc = _first_allocation(store)
        _kill9(proc)
        _assert_no_workspace_processes(tmp_path)
        turns = list_turns(store, thread_id=PARENT_THREAD)
        assert len(turns) == 1 and turns[0].turn_id == parent_turn
        assert turns[0].status == "running"  # kill 瞬间父在执行,死亡不经状态机
        return workspace, store, parent_turn, alloc
    except BaseException:
        _cleanup_proc(proc)
        store.close()
        raise


def _child_resume_workspace(alloc: Mapping[str, Any], workspace: Path) -> tuple[Path, Path | None]:
    """write 子的续跑 Session 必须用子工作区(锁文件族)+ 父库(db_path 同库)。"""
    if alloc["workspace_mode"] == "write":
        return Path(str(alloc["workspace"])), workspace / ".praxis" / "praxis.sqlite"
    return workspace, None


def _assert_cascaded_child(store: Store, alloc: Mapping[str, Any]) -> None:
    child_items = store.read_items(str(alloc["child_thread_id"]))
    cancels = _of_kind(child_items, ItemKind.CANCEL_REQUEST)
    assert len(cancels) == 1
    assert cancels[0].payload.get("source") == "parent"
    turn = get_turn(store, str(alloc["child_root_turn_id"]))
    assert turn is not None and turn.status == "interrupted"


# ---------------------------------------------------------------------------
# ① kill 矩阵:allocation 事务前/后
# ---------------------------------------------------------------------------


def test_kill9_alloc_before_transaction_rolls_back_whole_pair(tmp_path: Path) -> None:
    workspace, store, parent_turn, alloc = _matrix_kill(tmp_path, "alloc-before")
    try:
        assert alloc is None  # 事务未写:allocation 与子根 turn 要么都在要么都不在
        statuses = _sweep_statuses(workspace, PARENT_THREAD)
        assert statuses[parent_turn] == "interrupted"
        assert len(statuses) == 1  # 子根 turn 行不存在(R-T5-3 原子对)
        _assert_conservation(store, PARENT_THREAD, parent_turn)

        # spawn started+非幂等 → outcome_unknown,绝不自动重放;人工对账 failed 收尾。
        r1 = _resume(workspace, PARENT_THREAD, parent_turn, [], confirm=True)
        assert r1.status is TurnStatus.PAUSED and r1.reason == "outcome_unknown"
        assert r1.unknown_calls == ("s1",)
        scripts = _matrix_scripts("alloc-before")
        r2 = _resume(
            workspace, PARENT_THREAD, parent_turn, scripts["parent"][1:], reconcile={"s1": "failed"}
        )
        assert r2.status is TurnStatus.COMPLETED and r2.output == "parent done"

        items = store.read_items(PARENT_THREAD)
        assert _of_kind(items, ItemKind.CHILD_ALLOCATION) == []  # spawn 从未第二次执行
        assert len(_calls(items, "s1")) == 1
        assert [r.payload.get("status") for r in _results(items, "s1")] == ["outcome_unknown", "error"]
        _assert_fully_settled(store, PARENT_THREAD, parent_turn)
    finally:
        store.close()
        _assert_no_workspace_processes(tmp_path)


def test_kill9_alloc_after_transaction_child_interrupted_settles(tmp_path: Path) -> None:
    workspace, store, parent_turn, alloc = _matrix_kill(tmp_path, "alloc-after")
    try:
        assert alloc is not None
        child_root = str(alloc["child_root_turn_id"])
        assert get_turn(store, child_root) is not None  # 原子对:子根 turn 已在
        statuses = _sweep_statuses(workspace, PARENT_THREAD)
        assert statuses[parent_turn] == "interrupted"
        _assert_cascaded_child(store, alloc)  # 父 interrupted → 级联 cancel_request
        # 子 interrupted 保守 active:份额全额占用可对账(子零事实)。
        node = _assert_conservation(store, PARENT_THREAD, parent_turn)
        assert node.children[str(alloc["allocation_id"])].state == "active"
        assert node.reserved.tokens > 0

        r1 = _resume(workspace, PARENT_THREAD, parent_turn, [], confirm=True)
        assert r1.status is TurnStatus.PAUSED and r1.unknown_calls == ("s1",)
        # 续跑 Session 无路由:wait 帧直接给真实 child id(重放/新 dispatch 同据)。
        wait_frame = _spec(
            "", (("w1", "wait_subagent", {"child_thread_id": str(alloc["child_thread_id"]), "timeout": 30}),)
        )
        r2 = _resume(
            workspace, PARENT_THREAD, parent_turn, [wait_frame, _spec("parent done")], reconcile={"s1": "ok"}
        )
        assert r2.status is TurnStatus.COMPLETED and r2.output == "parent done"
        w1 = _results(store.read_items(PARENT_THREAD), "w1")[-1]
        assert w1.payload.get("metadata", {}).get("state") == "interrupted"  # wait 真观察到子状态

        # 子各自确认续跑:步边界读到级联 cancel → CANCELLED → 份额推导式归还。
        child_ws, db_path = _child_resume_workspace(alloc, workspace)
        rc = _resume(child_ws, str(alloc["child_thread_id"]), child_root, [], confirm=True, db_path=db_path)
        assert rc.status is TurnStatus.CANCELLED and rc.reason == "cancelled_by_user"
        assert len(_of_kind(store.read_items(PARENT_THREAD), ItemKind.CHILD_ALLOCATION)) == 1
        _assert_fully_settled(store, PARENT_THREAD, parent_turn)
    finally:
        store.close()
        _assert_no_workspace_processes(tmp_path)


# ---------------------------------------------------------------------------
# ① kill 矩阵:子首 dispatch 前/后
# ---------------------------------------------------------------------------


def test_kill9_child_before_first_dispatch_cancel_blocks_dispatch(tmp_path: Path) -> None:
    workspace, store, parent_turn, alloc = _matrix_kill(tmp_path, "dispatch-before")
    try:
        assert alloc is not None
        child_root = str(alloc["child_root_turn_id"])
        child_thread = str(alloc["child_thread_id"])
        statuses = _sweep_statuses(workspace, PARENT_THREAD)
        assert statuses[parent_turn] == "interrupted"
        _assert_cascaded_child(store, alloc)
        _assert_conservation(store, PARENT_THREAD, parent_turn)

        # 父:wait started+幂等 → 自动重放(子 interrupted 如实返回)→ 收尾。
        scripts = _matrix_scripts("dispatch-before")
        r = _resume(workspace, PARENT_THREAD, parent_turn, scripts["parent"][2:], confirm=True)
        assert r.status is TurnStatus.COMPLETED and r.output == "parent done"
        w1 = _results(store.read_items(PARENT_THREAD), "w1")[-1]
        assert w1.payload.get("metadata", {}).get("state") == "interrupted"

        # 子:c1 从未 dispatch(零 tool_call);级联取消 → not_executed + CANCELLED。
        rc = _resume(workspace, child_thread, child_root, [], confirm=True)
        assert rc.status is TurnStatus.CANCELLED
        child_items = store.read_items(child_thread)
        assert _of_kind(child_items, ItemKind.TOOL_CALL) == []
        assert [i.payload.get("status") for i in _results(child_items, "c1")] == ["not_executed"]
        assert (workspace / "target.txt").read_text(encoding="utf-8") == "fault matrix target\n"
        _assert_fully_settled(store, PARENT_THREAD, parent_turn)
    finally:
        store.close()
        _assert_no_workspace_processes(tmp_path)


def test_kill9_child_after_first_dispatch_no_side_effect_replay(tmp_path: Path) -> None:
    workspace, store, parent_turn, alloc = _matrix_kill(tmp_path, "dispatch-after")
    try:
        assert alloc is not None
        child_root = str(alloc["child_root_turn_id"])
        child_thread = str(alloc["child_thread_id"])
        child_ws = Path(str(alloc["workspace"]))
        side = child_ws / "side.txt"
        assert not side.exists()  # kill 落在 tool_call 已写、真实 write 之前
        statuses = _sweep_statuses(workspace, PARENT_THREAD)
        assert statuses[parent_turn] == "interrupted"
        _assert_cascaded_child(store, alloc)
        _assert_conservation(store, PARENT_THREAD, parent_turn)

        scripts = _matrix_scripts("dispatch-after")
        r = _resume(workspace, PARENT_THREAD, parent_turn, scripts["parent"][2:], confirm=True)
        assert r.status is TurnStatus.COMPLETED and r.output == "parent done"
        w1 = _results(store.read_items(PARENT_THREAD), "w1")[-1]
        assert w1.payload.get("metadata", {}).get("state") == "interrupted"

        # 子:c1 started(write 幂等)→ 重放队列,但级联 cancel 检查点拒 dispatch
        # → 步边界收敛 CANCELLED,清算补 outcome_unknown(可追);副作用从未发生。
        rc1 = _resume(
            child_ws, child_thread, child_root, [], confirm=True,
            db_path=workspace / ".praxis" / "praxis.sqlite",
        )
        assert rc1.status is TurnStatus.CANCELLED and rc1.reason == "cancelled_by_user"
        assert rc1.unknown_calls == ("c1",)
        child_items = store.read_items(child_thread)
        assert len(_calls(child_items, "c1")) == 1  # write 从未第二次执行
        assert [i.payload.get("status") for i in _results(child_items, "c1")] == ["outcome_unknown"]
        assert not side.exists()
        _assert_fully_settled(store, PARENT_THREAD, parent_turn)
    finally:
        store.close()
        _assert_no_workspace_processes(tmp_path)


# ---------------------------------------------------------------------------
# ① kill 矩阵:子 settle 边界(终态 turn_status 写前/后)
# ---------------------------------------------------------------------------


def test_kill9_child_settle_before_status_write_interrupted_holds_share(tmp_path: Path) -> None:
    workspace, store, parent_turn, alloc = _matrix_kill(tmp_path, "settle-before")
    try:
        assert alloc is not None
        child_root = str(alloc["child_root_turn_id"])
        child_thread = str(alloc["child_thread_id"])
        statuses = _sweep_statuses(workspace, PARENT_THREAD)
        assert statuses[parent_turn] == "interrupted"
        _assert_cascaded_child(store, alloc)
        # 终态未写 → 子 interrupted 保守 active:已用事实 + 未用份额都在 exposure。
        node = _assert_conservation(store, PARENT_THREAD, parent_turn)
        child = node.children[str(alloc["allocation_id"])]
        assert child.state == "active" and child.used.tokens > 0 and node.reserved.tokens > 0

        # kill 可能落在 spawn 结果尚未落库的交错(子终态响应已写、父 spawn
        # 续体未跑):s1 或已结算或待对账——两种崩溃现场都必须可恢复。
        wait_frame = _spec(
            "", (("w1", "wait_subagent", {"child_thread_id": str(alloc["child_thread_id"]), "timeout": 30}),)
        )
        tail = [wait_frame, _spec("parent done")]
        r = _resume(workspace, PARENT_THREAD, parent_turn, tail, confirm=True)
        if r.status is TurnStatus.PAUSED:
            assert r.reason == "outcome_unknown" and r.unknown_calls == ("s1",)
            r = _resume(workspace, PARENT_THREAD, parent_turn, tail, reconcile={"s1": "ok"})
        assert r.status is TurnStatus.COMPLETED and r.output == "parent done"
        w1 = _results(store.read_items(PARENT_THREAD), "w1")[-1]
        assert w1.payload.get("metadata", {}).get("state") == "interrupted"
        rc = _resume(workspace, child_thread, child_root, [], confirm=True)
        assert rc.status is TurnStatus.CANCELLED
        _assert_fully_settled(store, PARENT_THREAD, parent_turn)
    finally:
        store.close()
        _assert_no_workspace_processes(tmp_path)


def test_kill9_child_settle_after_status_write_derived_release(tmp_path: Path) -> None:
    workspace, store, parent_turn, alloc = _matrix_kill(tmp_path, "settle-after")
    try:
        assert alloc is not None
        child_root = str(alloc["child_root_turn_id"])
        statuses = _sweep_statuses(workspace, PARENT_THREAD)
        assert statuses[parent_turn] == "interrupted"
        assert statuses[child_root] == "completed"  # 终态子不被 sweep 误伤
        child_items = store.read_items(str(alloc["child_thread_id"]))
        assert _of_kind(child_items, ItemKind.CANCEL_REQUEST) == []  # 终态跳过级联
        # settle 是推导不是写:无 settle item,子终态事实 alone 归还残额。
        node = _assert_conservation(store, PARENT_THREAD, parent_turn)
        assert node.reserved == UsageFacts()
        assert node.used.tokens > 0

        # kill 可能落在 spawn 结果尚未落库的交错(子终态响应已写、父 spawn
        # 续体未跑):s1 或已结算或待对账——两种崩溃现场都必须可恢复。
        wait_frame = _spec(
            "", (("w1", "wait_subagent", {"child_thread_id": str(alloc["child_thread_id"]), "timeout": 30}),)
        )
        tail = [wait_frame, _spec("parent done")]
        r = _resume(workspace, PARENT_THREAD, parent_turn, tail, confirm=True)
        if r.status is TurnStatus.PAUSED:
            assert r.reason == "outcome_unknown" and r.unknown_calls == ("s1",)
            r = _resume(workspace, PARENT_THREAD, parent_turn, tail, reconcile={"s1": "ok"})
        assert r.status is TurnStatus.COMPLETED and r.output == "parent done"
        w1 = _results(store.read_items(PARENT_THREAD), "w1")[-1]
        assert w1.payload.get("metadata", {}).get("state") == "done"
        marks = [
            i for i in store.read_items(str(alloc["child_thread_id"]))
            if i.kind == ItemKind.TURN_STATUS and i.payload.get("to") == "completed"
        ]
        assert len(marks) == 1  # 子终态只写一次,重入不重复结算
        _assert_fully_settled(store, PARENT_THREAD, parent_turn)
    finally:
        store.close()
        _assert_no_workspace_processes(tmp_path)


# ---------------------------------------------------------------------------
# ① kill 矩阵:extend item 前/后(活跃子占用下补预算)
# ---------------------------------------------------------------------------


def _extend_child_converges(store: Store, workspace: Path, alloc: Mapping[str, Any]) -> None:
    child_root = str(alloc["child_root_turn_id"])
    child_thread = str(alloc["child_thread_id"])
    rc1 = _resume(workspace, child_thread, child_root, [], confirm=True)
    assert rc1.status is TurnStatus.PAUSED and rc1.reason == "outcome_unknown"
    assert rc1.unknown_calls == ("h1",)  # hold started+非幂等 → 未知可对账
    rc2 = _resume(workspace, child_thread, child_root, [], reconcile={"h1": "ok"})
    assert rc2.status is TurnStatus.CANCELLED
    child_items = store.read_items(child_thread)
    assert len(_calls(child_items, "h1")) == 1  # hold 从未第二次执行
    assert not (workspace / "hold-marker").exists()


def test_kill9_extend_before_item_no_phantom_budget(tmp_path: Path) -> None:
    workspace, store, parent_turn, alloc = _matrix_kill(tmp_path, "extend-before")
    try:
        assert alloc is not None
        items = store.read_items(PARENT_THREAD)
        assert _of_kind(items, ItemKind.BUDGET_EXTENSION) == []  # item 未写:不凭空多预算
        statuses = _sweep_statuses(workspace, PARENT_THREAD)
        assert statuses[parent_turn] == "interrupted"
        _assert_cascaded_child(store, alloc)
        # 活跃子(interrupted)占用份额:exposure = Σ facts + 未用份额,逐维可对账。
        node = _assert_conservation(store, PARENT_THREAD, parent_turn)
        assert node.reserved.tokens > 0

        # 重新补预算续跑:extension 只落一条,折叠后准入放行。
        r = _resume(
            workspace, PARENT_THREAD, parent_turn, [_spec("parent done")],
            confirm=True, extend={"tokens": 50},
        )
        assert r.status is TurnStatus.COMPLETED and r.output == "parent done"
        items = store.read_items(PARENT_THREAD)
        extensions = _of_kind(items, ItemKind.BUDGET_EXTENSION)
        assert len(extensions) == 1
        assert extensions[0].payload == {"limits_delta": {"tokens": 50}}
        assert get_turn(store, parent_turn).binding["budget"] == {
            "tokens": 20, "steps": 20, "web_requests": 10, "browser_requests": 5,
        }
        _extend_child_converges(store, workspace, alloc)
        _assert_fully_settled(store, PARENT_THREAD, parent_turn)
        assert _assert_conservation(store, PARENT_THREAD, parent_turn).limits.tokens == 70
    finally:
        store.close()
        _assert_no_workspace_processes(tmp_path)


def test_kill9_extend_after_item_folds_exactly_once(tmp_path: Path) -> None:
    workspace, store, parent_turn, alloc = _matrix_kill(tmp_path, "extend-after")
    try:
        assert alloc is not None
        items = store.read_items(PARENT_THREAD)
        assert len(_of_kind(items, ItemKind.BUDGET_EXTENSION)) == 1  # 已提交
        statuses = _sweep_statuses(workspace, PARENT_THREAD)
        assert statuses[parent_turn] == "interrupted"
        _assert_cascaded_child(store, alloc)
        node = _assert_conservation(store, PARENT_THREAD, parent_turn)
        assert node.limits.tokens == 70  # 崩溃前增量已折叠,不重不漏
        assert node.reserved.tokens > 0

        # confirm 续跑(无需再 extend):同一 extension 只折叠一次。
        r = _resume(workspace, PARENT_THREAD, parent_turn, [_spec("parent done")], confirm=True)
        assert r.status is TurnStatus.COMPLETED and r.output == "parent done"
        assert len(_of_kind(store.read_items(PARENT_THREAD), ItemKind.BUDGET_EXTENSION)) == 1
        _extend_child_converges(store, workspace, alloc)
        _assert_fully_settled(store, PARENT_THREAD, parent_turn)
        assert _assert_conservation(store, PARENT_THREAD, parent_turn).limits.tokens == 70
    finally:
        store.close()
        _assert_no_workspace_processes(tmp_path)


# ---------------------------------------------------------------------------
# ② 取消竞态:双进程真 db 同库 + 文件 barrier
# ---------------------------------------------------------------------------

RACE_ROOT_BUDGET = {"tokens": 1000, "steps": 20}
RACE_GRANT = {"tokens": 400, "steps": 9}


def _race_setup(tmp_path: Path) -> tuple[Path, Store, str, str, Path]:
    """预创建父/子 turn + allocation(spawn 事务同形);返回 (ws, store, 父turn, 子turn, barrier)。"""
    workspace = tmp_path / "race-ws"
    workspace.mkdir()
    (workspace / ".praxis").mkdir()
    store = Store(workspace / ".praxis" / "praxis.sqlite")
    parent_turn = uuid4().hex
    child_turn = uuid4().hex
    store.ensure_thread(PARENT_THREAD)
    store.ensure_thread("race-child")
    payload: dict[str, Any] = {
        "allocation_id": uuid4().hex,
        "parent_turn_id": parent_turn,
        "child_thread_id": "race-child",
        "child_root_turn_id": child_turn,
        "workspace_mode": "read_only",
        "workspace": str(workspace),
        "grant": dict(RACE_GRANT),
        "requested": {},
        "source": "model",
    }

    def _tx(conn: Any) -> None:
        append_items_tx(conn, PARENT_THREAD, parent_turn, [ItemEntry(kind=ItemKind.CHILD_ALLOCATION, payload=payload)])
        create_turn_tx(conn, turn_id=parent_turn, thread_id=PARENT_THREAD, binding={"budget": dict(RACE_ROOT_BUDGET)})
        create_turn_tx(
            conn, turn_id=child_turn, thread_id="race-child",
            binding={"budget": dict(RACE_GRANT), "parent_turn_id": parent_turn, "root_turn_id": child_turn, "depth": 1},
        )

    store.transact(_tx)
    barrier = tmp_path / "barrier"
    barrier.mkdir()
    return workspace, store, parent_turn, child_turn, barrier


def _start_race(
    tmp_path: Path, workspace: Path, config: Mapping[str, Any], name: str
) -> subprocess.Popen[bytes]:
    path = tmp_path / f"race-{name}.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    return subprocess.Popen(
        [sys.executable, str(RACE_SCRIPT), str(workspace), str(path)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def _race_result(proc: subprocess.Popen[bytes]) -> dict[str, Any]:
    stdout, stderr = proc.communicate(timeout=30)
    assert proc.returncode == 0, f"race driver failed rc={proc.returncode}: {stderr.decode()[-2000:]}"
    return json.loads(stdout.decode())


def _cancel_parent_via_third_connection(workspace: Path, parent_turn: str) -> None:
    async def _go() -> None:
        observer = open_session(workspace, ScriptedAdapter([]), thread_id=PARENT_THREAD)
        try:
            assert observer.request_cancel(parent_turn) == "queued"  # B 执行中:仅落 item
        finally:
            observer.close()

    asyncio.run(_go())


def _wait_child_cascade(store: Store, child_turn: str, procs: Sequence[subprocess.Popen[bytes]]) -> None:
    _poll(
        lambda: any(
            i.kind == ItemKind.CANCEL_REQUEST and i.turn_id == child_turn
            for i in store.read_items("race-child")
        ),
        "parent cascade cancel_request on child",
        procs=procs,
    )


def test_cancel_race_cascade_first_blocks_child_dispatch(tmp_path: Path) -> None:
    workspace, store, parent_turn, child_turn, barrier = _race_setup(tmp_path)
    child_frames = [_spec("", (("c1", "race_noop", {}),)), _spec("child done")]
    parent_frames = [_spec("", (("cp1", "parent_hold", {}),)), _spec("parent done")]
    a = _start_race(tmp_path, workspace, {
        "role": "child", "thread_id": "race-child", "turn_id": child_turn,
        "frames": child_frames, "barrier": str(barrier), "mode": "hold-before-dispatch",
    }, "a")
    b: subprocess.Popen[bytes] | None = None
    try:
        _poll(lambda: (barrier / "child-ready").exists(), "child ready before dispatch", procs=[a])
        child_items = store.read_items("race-child")
        assert _of_kind(child_items, ItemKind.MODEL_RESPONSE) and not _of_kind(child_items, ItemKind.TOOL_CALL)
        b = _start_race(tmp_path, workspace, {
            "role": "parent", "thread_id": PARENT_THREAD, "turn_id": parent_turn,
            "frames": parent_frames, "barrier": str(barrier), "mode": "parent-hold",
        }, "b")
        _poll(
            lambda: bool(_calls(store.read_items(PARENT_THREAD), "cp1")),
            "parent hold dispatched", procs=[a, b],
        )
        _cancel_parent_via_third_connection(workspace, parent_turn)
        _wait_child_cascade(store, child_turn, [a, b])  # B 收敛 CANCELLED → cascade_for
        (barrier / "go").write_text("go", encoding="utf-8")  # 级联已落,放行子 dispatch

        ra = _race_result(a)
        assert ra["status"] == "cancelled" and ra["reason"] == "cancelled_by_user"
        rb = _race_result(b)
        assert rb["status"] == "cancelled"

        child_items = store.read_items("race-child")
        assert _of_kind(child_items, ItemKind.TOOL_CALL) == []  # 下一次 dispatch 必拒
        assert [i.payload.get("status") for i in _results(child_items, "c1")] == ["not_executed"]
        assert _of_kind(child_items, ItemKind.MODEL_ATTEMPT)  # 模型尝试痕迹可追
        cancels = _of_kind(child_items, ItemKind.CANCEL_REQUEST)
        assert len(cancels) == 1 and cancels[0].payload.get("source") == "parent"
        assert get_turn(store, child_turn).status == "cancelled"
        assert get_turn(store, parent_turn).status == "cancelled"
        # 预算不凭空归零:子已用事实仍在 exposure;终态结算后 reserved 归零。
        node = _assert_conservation(store, PARENT_THREAD, parent_turn)
        assert node.used.tokens == 8 and node.reserved == UsageFacts()
        _assert_fully_settled(store, PARENT_THREAD, parent_turn)
    finally:
        _cleanup_proc(a)
        if b is not None:
            _cleanup_proc(b)
        store.close()
        _assert_no_workspace_processes(tmp_path)


def _race_dispatch_first_setup(
    tmp_path: Path, *, expect_kill: bool
) -> tuple[Path, Store, str, str, Path, subprocess.Popen[bytes], subprocess.Popen[bytes], Path]:
    workspace, store, parent_turn, child_turn, barrier = _race_setup(tmp_path)
    marker = tmp_path / "race-marker"
    child_frames = [
        _spec("", (("c1", "race_touch", {}), ("c2", "race_noop", {}))),
        _spec("child done"),
    ]
    parent_frames = [_spec("", (("cp1", "parent_hold", {}),)), _spec("parent done")]
    a = _start_race(tmp_path, workspace, {
        "role": "child", "thread_id": "race-child", "turn_id": child_turn,
        "frames": child_frames, "barrier": str(barrier), "mode": "touch-hold",
        "marker": str(marker), "expect_kill": expect_kill,
    }, "a")
    b: subprocess.Popen[bytes] | None = None
    try:
        _poll(
            lambda: (barrier / "child-dispatched").exists() and marker.exists()
            and bool(_calls(store.read_items("race-child"), "c1")),
            "child c1 dispatched with side effect", procs=[a],
        )
        b = _start_race(tmp_path, workspace, {
            "role": "parent", "thread_id": PARENT_THREAD, "turn_id": parent_turn,
            "frames": parent_frames, "barrier": str(barrier), "mode": "parent-hold",
        }, "b")
        _poll(lambda: bool(_calls(store.read_items(PARENT_THREAD), "cp1")), "parent hold dispatched", procs=[a, b])
        _cancel_parent_via_third_connection(workspace, parent_turn)
        _wait_child_cascade(store, child_turn, [a, b])
        assert b is not None
        return workspace, store, parent_turn, child_turn, barrier, a, b, marker
    except BaseException:
        _cleanup_proc(a)
        if b is not None:
            _cleanup_proc(b)
        store.close()
        raise


def test_cancel_race_dispatch_first_traces_and_stop_settlement(tmp_path: Path) -> None:
    workspace, store, parent_turn, child_turn, barrier, a, b, marker = _race_dispatch_first_setup(
        tmp_path, expect_kill=False
    )
    try:
        mtime = marker.stat().st_mtime_ns
        rb = _race_result(b)  # 父先收敛:CANCELLED + 级联
        assert rb["status"] == "cancelled"
        (barrier / "release").write_text("release", encoding="utf-8")
        ra = _race_result(a)
        assert ra["status"] == "cancelled" and ra["reason"] == "cancelled_by_user"

        child_items = store.read_items("race-child")
        # dispatch 先:痕迹齐全(tool_call + model_attempt),stop 结算可追。
        assert len(_calls(child_items, "c1")) == 1
        assert _of_kind(child_items, ItemKind.MODEL_ATTEMPT)
        by_id = {i.payload.get("call_id"): i.payload.get("status") for i in _of_kind(child_items, ItemKind.TOOL_RESULT)}
        assert by_id == {"c1": "ok", "c2": "not_executed"}  # c2 检查点拒,从未 dispatch
        assert _calls(child_items, "c2") == []
        assert marker.stat().st_mtime_ns == mtime  # 副作用只发生一次
        assert get_turn(store, child_turn).status == "cancelled"
        node = _assert_conservation(store, PARENT_THREAD, parent_turn)
        assert node.used.tokens == 8 and node.reserved == UsageFacts()
        _assert_fully_settled(store, PARENT_THREAD, parent_turn)
    finally:
        _cleanup_proc(a)
        _cleanup_proc(b)
        store.close()
        _assert_no_workspace_processes(tmp_path)


def test_cancel_race_dispatch_first_crash_unknown_traceable(tmp_path: Path) -> None:
    workspace, store, parent_turn, child_turn, barrier, a, b, marker = _race_dispatch_first_setup(
        tmp_path, expect_kill=True
    )
    try:
        mtime = marker.stat().st_mtime_ns
        rb = _race_result(b)
        assert rb["status"] == "cancelled"
        # 级联已落、c1 仍在飞:SIGKILL 子执行者 → unknown 经 resume 对账可追。
        _kill9(a)
        _assert_no_workspace_processes(tmp_path)
        statuses = _sweep_statuses(workspace, PARENT_THREAD)
        assert statuses[child_turn] == "interrupted"
        child_items = store.read_items("race-child")
        cancels = _of_kind(child_items, ItemKind.CANCEL_REQUEST)
        assert len(cancels) == 1 and cancels[0].payload.get("source") == "parent"  # sweep 级联幂等跳过

        rc1 = _resume(workspace, "race-child", child_turn, [], confirm=True)
        assert rc1.status is TurnStatus.PAUSED and rc1.reason == "outcome_unknown"
        assert rc1.unknown_calls == ("c1",)  # started+非幂等 → 未知副作用不自动重放
        rc2 = _resume(workspace, "race-child", child_turn, [], reconcile={"c1": "ok"})
        assert rc2.status is TurnStatus.CANCELLED  #  queued cancel 步边界收敛
        child_items = store.read_items("race-child")
        assert len(_calls(child_items, "c1")) == 1
        assert [i.payload.get("status") for i in _results(child_items, "c1")] == ["outcome_unknown", "ok"]
        assert [i.payload.get("status") for i in _results(child_items, "c2")] == ["not_executed"]
        assert marker.stat().st_mtime_ns == mtime
        _assert_fully_settled(store, PARENT_THREAD, parent_turn)
    finally:
        _cleanup_proc(a)
        _cleanup_proc(b)
        store.close()
        _assert_no_workspace_processes(tmp_path)


# ---------------------------------------------------------------------------
# ③ 预算耗尽场景行:逼近总额 → PAUSED → CLI --extend-tokens 续跑(真 CLI 子进程)
# ---------------------------------------------------------------------------

_CLI_SHIM = "from praxis.interfaces.cli import app; app()"


def _cli(workspace: Path, args: Sequence[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-c", _CLI_SHIM, *args, "--workspace", str(workspace)],
        capture_output=True, text=True, timeout=45, cwd=str(Path(__file__).resolve().parents[3]),
    )


def test_cli_budget_near_limit_pauses_then_extend_tokens_resumes(tmp_path: Path) -> None:
    """§13 预算耗尽行:逼近总额 → PAUSED(budget_exhausted) → --extend-tokens 续跑。

    耗尽腿经 SDK scripted(真实 usage 事实)——CLI ``--scripted`` 加载器目前
    硬编码 Usage(0,0)(findings 补充节:F-T9-1,非 B2/B14/B29,不顺手修),
    tokens 耗尽无法用纯 CLI 剧本表达;暂停观察与续跑腿都是**真 CLI 子进程**
    (status 重建文案 + resume --extend-tokens 端到端收尾)。
    """
    workspace = tmp_path / "cli-ws"
    workspace.mkdir()
    # 每帧 10 tokens;限额 22(保护尾 2 → 有效 20):两帧后 20/22 逼近总额,
    # 第三帧 dispatch 前准入拒绝 → PAUSED(budget_exhausted, tokens)。
    frames = [
        _spec("", (("c1", "list_files", {}),), usage=(5, 5)),
        _spec("", (("c2", "list_files", {}),), usage=(5, 5)),
        _spec("done", usage=(5, 5)),
    ]

    async def _exhaust() -> str:
        session = open_session(
            workspace,
            ScriptedAdapter([_response(s) for s in frames]),
            thread_id="budget-cli",
            budget=Limits(tokens=22),
        )
        try:
            result = await session.submit("go")
            assert result.status is TurnStatus.PAUSED
            assert result.reason == "budget_exhausted"
            assert result.budget_dimension == "tokens"
            return result.turn_id
        finally:
            session.close()

    turn_id = asyncio.run(_exhaust())

    # 真 CLI 观察暂停现场:status 重建文案点名 tokens 维与 --extend-tokens 出路。
    detail = _cli(workspace, ["status", turn_id, "--thread", "budget-cli"])
    assert detail.returncode == 0, detail.stdout + detail.stderr
    assert "budget_exhausted" in detail.stdout
    assert "budget exhausted" in detail.stdout and "--extend-tokens" in detail.stdout

    # 真 CLI 续跑:--extend-tokens 30 写 budget_extension item → 同一续跑路径收尾。
    script2 = tmp_path / "resume.json"
    script2.write_text(json.dumps(frames[2:]), encoding="utf-8")
    done = _cli(workspace, [
        "resume", turn_id, "--thread", "budget-cli", "--scripted", str(script2), "--extend-tokens", "30",
    ])
    assert done.returncode == 0, done.stdout + done.stderr
    assert "done" in done.stdout

    store = Store(workspace / ".praxis" / "praxis.sqlite")
    try:
        turn = get_turn(store, turn_id)
        assert turn is not None and turn.status == "completed"
        assert turn.binding["budget"] == {"tokens": 22}
        items = store.read_items("budget-cli")
        extensions = _of_kind(items, ItemKind.BUDGET_EXTENSION)
        assert len(extensions) == 1
        assert extensions[0].payload == {"limits_delta": {"tokens": 30}}
        reasons = [i.payload.get("reason") for i in _of_kind(items, ItemKind.TURN_STATUS)]
        assert "budget_extended" in reasons
        # 守恒:无子分配,exposure == Σ facts,补预算只增 limits。已用 20 = SDK
        # 耗尽腿两帧真实 usage;CLI 续跑腿的 done 帧经 --scripted 加载器 usage=0
        # (F-T9-1),不贡献 tokens——事实账诚实,准入放行由收尾成功证明。
        node = _assert_conservation(store, "budget-cli", turn_id)
        assert node.used.tokens == 20 and node.limits.tokens == 52
        assert node.reserved == UsageFacts()
    finally:
        store.close()


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-v"]))