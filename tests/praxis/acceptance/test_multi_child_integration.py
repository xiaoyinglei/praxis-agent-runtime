"""阶段4 T8 验收:真实双子工作区集成(§13"真实多子代理"行的阶段 4 子集)。

三场景(真 git repo + 真子 Session,非 mock):

① 两 write 子**并行**改相邻文件(asyncio.Barrier 钉双 child_edit 同时在飞)
   → 双 integrate apply 全成(逐次 DESTRUCTIVE 显式审批);
② 两 write 子并行改**同文件同行** → 第一个 apply 成,第二个 integrate_conflict
   且父/子工作区字节都在(整组零写);
③ apply 中途 kill 父(真 SIGKILL,``_child_integrate_turn.py`` 独立 OS 进程,
   缝 = provisioner.apply 先睡 45s):重启后 allocation 完整、子终态不被 sweep
   误伤、integrate 调用 started+非幂等 → resume(confirm=True) PAUSED/
   outcome_unknown → reconcile(ok) 收尾(T4 既有矩阵完整链);随后全新 turn
   再 integrate apply 成功——崩溃现场可解释、工作区仍可交付。

R-T8-1:子经**生产工具**改自身 worktree(write/bash;子工具按子工作区根
重构 + worktree 迁出地板私有集 host_worktrees_dir)。``child_barrier`` 是
纯编排测试缝(只等 barrier、零写入,T6 child_hang 同先例),与 asyncio.Barrier
一起钉双子编辑**同时在飞**;编辑字节全部来自真 write/bash 管道。
子是同库真 Session、真终态 turn、真 git worktree diff——integrate 全程真实管道。
"""

from __future__ import annotations

import asyncio
import json
import os
import signal
import subprocess
import sys
import time
from collections.abc import AsyncIterator, Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from praxis.core.statemachine import TurnStatus
from praxis.interfaces.sdk import ScriptedAdapter, open_session
from praxis.protocol.items import Item, ItemKind
from praxis.protocol.models import (
    ModelEvent,
    ModelMessage,
    ModelResponse,
    ModelSettings,
    ModelToolCall,
    ModelToolCallDelta,
    Usage,
    WireRequest,
    assemble_model_events,
    collect_events,
)
from praxis.protocol.policy import NetworkPolicy, PermissionPolicy
from praxis.protocol.tools import (
    CancellationMode,
    InterruptBehavior,
    ResolvedUse,
    ResultStatus,
    Tool,
    ToolContentBlock,
    ToolDefinition,
    ToolResult,
)
from praxis.store.log import Store
from praxis.store.turns import get_turn

CHILD_SCRIPT = Path(__file__).resolve().parent / "_child_integrate_turn.py"
KILL_WINDOW_TIMEOUT = 30.0


def _git(cwd: Path, *args: str) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(["git", "-C", str(cwd), *args], capture_output=True, check=True)


def _resp(content: str, calls: tuple[tuple[str, str, dict[str, Any]], ...] = ()) -> ModelResponse:
    return ModelResponse(
        content=content,
        tool_calls=tuple(ModelToolCall(call_id=c, name=n, arguments=a) for c, n, a in calls),
        usage=Usage(input_tokens=1, output_tokens=1),
        finish_reason="tool_calls" if calls else "stop",
    )


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    root.mkdir()
    _git(root, "init", "-q")
    _git(root, "config", "user.email", "praxis-test@example.com")
    _git(root, "config", "user.name", "Praxis Test")
    (root / "a.txt").write_text("base-a\n", encoding="utf-8")
    (root / "b.txt").write_text("base-b\n", encoding="utf-8")
    _git(root, "add", ".")
    _git(root, "commit", "-qm", "init")
    return root


class _Router:
    """标记路由(并行子不抢帧)+ PLACEHOLDER-n → 第 n 条 allocation 的真实 id。"""

    def __init__(
        self,
        default: Sequence[ModelResponse],
        holder: list[Any],
        marked: Mapping[str, Sequence[ModelResponse]],
        thread_id: str = "parent",
    ) -> None:
        self._default = ScriptedAdapter(default)
        self._marked = {m: ScriptedAdapter(frames) for m, frames in marked.items()}
        self._holder = holder
        self._thread = thread_id

    def _child_id(self, placeholder: str) -> str:
        index = int(placeholder.rsplit("-", 1)[1]) - 1
        allocations = [
            item.payload
            for item in self._holder[0].store.read_items(self._thread)
            if item.kind == ItemKind.CHILD_ALLOCATION
        ]
        return str(allocations[index]["child_thread_id"])

    def build_request(
        self, messages: Sequence[ModelMessage], tools: Sequence[Any], settings: ModelSettings
    ) -> WireRequest:
        return self._default.build_request(messages, tools, settings)

    async def stream(self, req: WireRequest) -> AsyncIterator[ModelEvent]:
        text = " ".join(message.content for message in req.messages)
        adapter = self._default
        for marker, marked in self._marked.items():
            if marker in text:
                adapter = marked
                break
        async for event in adapter.stream(req):
            if isinstance(event, ModelToolCallDelta) and event.name_fragment in (
                "wait_subagent",
                "integrate_child_workspace",
            ):
                args = json.loads(event.arguments_fragment)
                args["child_thread_id"] = self._child_id(str(args["child_thread_id"]))
                event = replace(event, arguments_fragment=json.dumps(args))
            yield event

    async def complete(self, req: WireRequest) -> ModelResponse:
        return assemble_model_events(await collect_events(self.stream(req)))


def _barrier_tool(barrier: asyncio.Barrier, arrivals: list[str]) -> Tool:
    """child_barrier 纯编排测试缝:只等 barrier 并记录到达,零写入。

    与 asyncio.Barrier(2) 一起钉双子在生产编辑前同时在飞(并行承重);
    编辑本身由子的真 write/bash 工具完成(R-T8-1)。
    """

    async def run(arguments: Mapping[str, Any], exec_ctx: Any) -> ToolResult:
        await asyncio.wait_for(barrier.wait(), timeout=15)
        arrivals.append(exec_ctx.permissions.workspace)
        return ToolResult(
            tool_call_id="pending",
            tool_name="child_barrier",
            content=(ToolContentBlock(type="text", data={"text": "barrier passed"}),),
            status=ResultStatus.OK,
        )

    return Tool(
        definition=ToolDefinition(
            name="child_barrier",
            description="TEST SEAM: wait for the sibling child (orchestration only, no writes).",
            input_schema={"type": "object"},
        ),
        validate_input=lambda arguments: arguments,
        run=run,
        static_effects=frozenset(),
        resolve_use=lambda arguments: ResolvedUse(frozenset(), ()),
        idempotent=True,
        concurrency_safe=True,
        cancellation_mode=CancellationMode.COOPERATIVE,
        interrupt_behavior=InterruptBehavior.CANCEL,
        timeout_seconds=30,
        max_model_output_bytes=4_000,
    )


def _spawn(call_id: str, marker: str) -> ModelResponse:
    return _resp(
        "",
        ((call_id, "spawn_subagent", {"task": f"{marker} edit the file", "workspace_mode": "write"}),),
    )


def _wait(call_id: str, n: int) -> ModelResponse:
    return _resp("", ((call_id, "wait_subagent", {"child_thread_id": f"PLACEHOLDER-{n}", "timeout": 10}),))


def _integrate(call_id: str, n: int, action: str = "apply") -> ModelResponse:
    return _resp(
        "",
        ((call_id, "integrate_child_workspace", {"child_thread_id": f"PLACEHOLDER-{n}", "action": action}),),
    )


def _open(repo: Path, frames: Sequence[ModelResponse], marked: Mapping[str, Sequence[ModelResponse]], barrier_tool: Tool) -> Any:
    holder: list[Any] = []
    session = open_session(
        repo, _Router(frames, holder, marked),
        thread_id="parent", enable_subagents=True, extra_tools=(barrier_tool,),
        # 子的 bash 继承父网络策略(kill9 先例:默认 ask 会暂停命令网络)。
        policy=PermissionPolicy(network=NetworkPolicy(decision="allow")),
    )
    holder.append(session)
    return session


async def _approve(session: Any, result: Any) -> Any:
    if result.status is TurnStatus.PAUSED and result.reason == "approval_required":
        call_id = result.pending_approvals[0].call_id
        return await session.resume(result.turn_id, approval={call_id: "allow"})
    return result


def _allocations(session: Any) -> list[Mapping[str, JsonValue]]:
    return [
        item.payload
        for item in session.store.read_items("parent")
        if item.kind == ItemKind.CHILD_ALLOCATION
    ]


def _integrate_payloads(session: Any, turn_id: str) -> list[Mapping[str, JsonValue]]:
    return [
        item.payload
        for item in session.store.read_turn_items(turn_id)
        if item.kind == ItemKind.TOOL_RESULT
        and item.payload.get("tool_name") == "integrate_child_workspace"
    ]


# ---------------------------------------------------------------------------
# ① 双子并行改相邻文件 → 双 integrate 全成
# ---------------------------------------------------------------------------


def test_two_write_children_parallel_adjacent_files_both_integrate(repo: Path) -> None:
    async def scenario() -> None:
        barrier = asyncio.Barrier(2)
        arrivals: list[str] = []
        marked = {
            "[child-a]": [
                _resp("", (("b1", "child_barrier", {}),)),
                _resp("", (("e1", "write", {"path": "a.txt", "content": "child-a\n"}),)),
                _resp("done a"),
            ],
            "[child-b]": [
                _resp("", (("b2", "child_barrier", {}),)),
                _resp("", (("e2", "bash", {"command": "printf 'child-b\\n' > b.txt"}),)),
                _resp("done b"),
            ],
        }
        frames = [
            _spawn("s1", "[child-a]"),
            _spawn("s2", "[child-b]"),
            _wait("w1", 1),
            _wait("w2", 2),
            _integrate("ia", 1),
            _integrate("ib", 2),
            _resp("parent done"),
        ]
        session = _open(repo, frames, marked, _barrier_tool(barrier, arrivals))
        try:
            result = await session.submit("go")
            result = await _approve(session, result)
            result = await _approve(session, result)
            assert result.status is TurnStatus.COMPLETED
            # 并行承重:两个子的 barrier 曾同时在飞(双侧到达),编辑是真工具。
            assert len(arrivals) == 2
            allocations = _allocations(session)
            assert len(allocations) == 2
            assert all(a["workspace_mode"] == "write" for a in allocations)
            for allocation in allocations:
                turn = get_turn(session.store, str(allocation["child_root_turn_id"]))
                assert turn is not None and turn.status == "completed"
            assert [c.state for c in result.children_summary] == ["done", "done"]
            payloads = _integrate_payloads(session, result.turn_id)
            assert [p["status"] for p in payloads] == ["ok", "ok"]
            assert payloads[0]["metadata"]["applied"] == ("a.txt",)
            assert payloads[1]["metadata"]["applied"] == ("b.txt",)
            assert (repo / "a.txt").read_text(encoding="utf-8") == "child-a\n"
            assert (repo / "b.txt").read_text(encoding="utf-8") == "child-b\n"
        finally:
            await session.aclose()

    asyncio.run(scenario())


# ---------------------------------------------------------------------------
# ② 双子并行改同文件同行 → 第二个 conflicts,父/子字节都在
# ---------------------------------------------------------------------------


def test_two_write_children_same_line_second_conflicts_bytes_preserved(repo: Path) -> None:
    async def scenario() -> None:
        barrier = asyncio.Barrier(2)
        arrivals: list[str] = []
        marked = {
            "[child-a]": [
                _resp("", (("b1", "child_barrier", {}),)),
                _resp("", (("e1", "write", {"path": "a.txt", "content": "child-a\n"}),)),
                _resp("done a"),
            ],
            "[child-b]": [
                _resp("", (("b2", "child_barrier", {}),)),
                _resp("", (("e2", "bash", {"command": "printf 'child-b\\n' > a.txt"}),)),
                _resp("done b"),
            ],
        }
        frames = [
            _spawn("s1", "[child-a]"),
            _spawn("s2", "[child-b]"),
            _wait("w1", 1),
            _wait("w2", 2),
            _integrate("ia", 1),
            _integrate("ib", 2),
            _resp("parent done"),
        ]
        session = _open(repo, frames, marked, _barrier_tool(barrier, arrivals))
        try:
            result = await session.submit("go")
            result = await _approve(session, result)
            result = await _approve(session, result)
            assert result.status is TurnStatus.COMPLETED
            assert len(arrivals) == 2
            payloads = _integrate_payloads(session, result.turn_id)
            assert payloads[0]["status"] == "ok"
            assert payloads[0]["metadata"]["applied"] == ("a.txt",)
            assert payloads[1]["status"] == "error"
            assert payloads[1]["error_code"] == "integrate_conflict"
            assert payloads[1]["metadata"]["conflicts"] == ("a.txt",)
            assert payloads[1]["metadata"]["applied"] == ()
            allocations = _allocations(session)
            ws_a = Path(str(allocations[0]["workspace"]))
            ws_b = Path(str(allocations[1]["workspace"]))
            # 父 = 第一个子的字节;第二个子快照字节原样保留(双方都在)。
            assert (repo / "a.txt").read_text(encoding="utf-8") == "child-a\n"
            assert (ws_a / "a.txt").read_text(encoding="utf-8") == "child-a\n"
            assert (ws_b / "a.txt").read_text(encoding="utf-8") == "child-b\n"
            assert (repo / "b.txt").read_text(encoding="utf-8") == "base-b\n"
        finally:
            await session.aclose()

    asyncio.run(scenario())


# ---------------------------------------------------------------------------
# ③ apply 中途 kill 父(真 SIGKILL)→ outcome_unknown → reconcile 完整链
# ---------------------------------------------------------------------------


def _spec(response: ModelResponse) -> dict[str, Any]:
    return {
        "content": response.content,
        "tool_calls": [
            {"call_id": c.call_id, "name": c.name, "arguments": dict(c.arguments)}
            for c in response.tool_calls
        ],
        "usage": {"input_tokens": 1, "output_tokens": 1},
        "finish_reason": response.finish_reason,
    }


def _calls(items: Sequence[Item], call_id: str) -> list[Item]:
    return [i for i in items if i.kind == ItemKind.TOOL_CALL and i.payload.get("call_id") == call_id]


def _results(items: Sequence[Item], call_id: str) -> list[Item]:
    return [i for i in items if i.kind == ItemKind.TOOL_RESULT and i.payload.get("call_id") == call_id]


def test_kill9_mid_integrate_apply_reconcile_chain(repo: Path, tmp_path: Path) -> None:
    thread_id = "accept-kill9-integrate"
    scripts = {
        "parent": [
            _spec(_spawn("s1", "[kill-child]")),
            _spec(_wait("w1", 1)),
            _spec(_integrate("ci1", 1)),
            _spec(_resp("parent done")),
        ],
        "child": [
            _spec(_resp("", (("e1", "write", {"path": "a.txt", "content": "child-killed\n"}),))),
            _spec(_resp("child done")),
        ],
    }
    script_path = tmp_path / "scripts.json"
    script_path.write_text(json.dumps(scripts), encoding="utf-8")
    # 父测试先开 Store 轮询:.praxis 目录必须先于 driver 建库存在。
    (repo / ".praxis").mkdir(exist_ok=True)
    proc = subprocess.Popen(
        [sys.executable, str(CHILD_SCRIPT), str(repo), str(script_path), thread_id],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    store = Store(repo / ".praxis" / "praxis.sqlite")
    child_ws: Path | None = None
    try:
        deadline = time.monotonic() + KILL_WINDOW_TIMEOUT
        while True:
            rc = proc.poll()
            if rc is not None:
                stderr = proc.stderr.read().decode(errors="replace") if proc.stderr else ""
                stdout = proc.stdout.read().decode(errors="replace") if proc.stdout else ""
                pytest.fail(
                    f"driver exited before kill window (rc={rc}): {stderr[-2000:]} {stdout[-500:]}"
                )
            items = store.read_items(thread_id)
            allocations = [i.payload for i in items if i.kind == ItemKind.CHILD_ALLOCATION]
            if allocations and _calls(items, "ci1"):
                child_root = str(allocations[0]["child_root_turn_id"])
                child_turn = get_turn(store, child_root)
                child_ws = Path(str(allocations[0]["workspace"]))
                if (
                    child_turn is not None
                    and child_turn.status == "completed"
                    and (child_ws / "a.txt").exists()
                ):
                    break
            if time.monotonic() > deadline:
                pytest.fail("timeout waiting for integrate tool_call + terminal child")
            time.sleep(0.02)
        turn_id = str(_calls(items, "ci1")[0].turn_id)
        assert _results(items, "ci1") == []
        assert _calls(items, "ci1")[0].payload.get("idempotent") is False
        os.kill(proc.pid, signal.SIGKILL)
        proc.wait(timeout=10)
        assert proc.returncode == -signal.SIGKILL
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait(timeout=10)
        store.close()

    assert child_ws is not None
    # kill 落在缝内(真实 apply 之前):父仓字节未动,子工作产物已在。
    assert (repo / "a.txt").read_text(encoding="utf-8") == "base-a\n"
    assert (child_ws / "a.txt").read_text(encoding="utf-8") == "child-killed\n"

    # sweep-on-open:父 turn interrupted;子终态不被误伤;allocation 完整。
    async def _open_and_check() -> str:
        session = open_session(repo, ScriptedAdapter([]), thread_id=thread_id)
        try:
            parent_status = session.status(turn_id).status
            allocations = [
                i.payload
                for i in session.store.read_items(thread_id)
                if i.kind == ItemKind.CHILD_ALLOCATION
            ]
            assert len(allocations) == 1
            child_turn = get_turn(session.store, str(allocations[0]["child_root_turn_id"]))
            assert child_turn is not None and child_turn.status == "completed"
            assert Path(str(allocations[0]["workspace"])).is_dir()
            return parent_status
        finally:
            session.close()

    assert asyncio.run(_open_and_check()) == "interrupted"

    # resume(confirm=True):started+非幂等 → PAUSED/outcome_unknown,绝不自动重放。
    async def _resume_confirm() -> Any:
        session = open_session(repo, ScriptedAdapter([]), thread_id=thread_id)
        try:
            return await session.resume(turn_id, confirm=True)
        finally:
            session.close()

    r1 = asyncio.run(_resume_confirm())
    assert r1.status is TurnStatus.PAUSED
    assert r1.reason == "outcome_unknown"
    assert r1.unknown_calls == ("ci1",)

    # 人工 reconcile ok → 剧本收尾(T4 矩阵;全新 Session,空剧本防重放)。
    async def _reconcile() -> Any:
        session = open_session(repo, ScriptedAdapter([_resp("parent done")]), thread_id=thread_id)
        try:
            return await session.resume(turn_id, reconcile={"ci1": "ok"})
        finally:
            session.close()

    r2 = asyncio.run(_reconcile())
    assert r2.status is TurnStatus.COMPLETED
    assert r2.output == "parent done"
    assert r2.unknown_calls == ()

    store2 = Store(repo / ".praxis" / "praxis.sqlite")
    try:
        items = store2.read_items(thread_id)
        assert len(_calls(items, "ci1")) == 1  # integrate 从未第二次执行
        assert [r.payload.get("status") for r in _results(items, "ci1")] == ["outcome_unknown", "ok"]
        marks = [
            i for i in items
            if i.kind == ItemKind.TURN_STATUS and i.payload.get("to") == "interrupted"
        ]
        assert len(marks) == 1
        assert marks[0].payload.get("reason") == "executor_process_died"
        assert marks[0].payload.get("from") == "running"
    finally:
        store2.close()
    assert (repo / "a.txt").read_text(encoding="utf-8") == "base-a\n"

    # 崩溃现场可解释的最终证明:全新 turn 再 integrate apply + discard 全成。
    async def _fresh_integrate() -> None:
        holder: list[Any] = []
        frames = [_integrate("ci2", 1), _integrate("ci3", 1, action="discard"), _resp("done")]
        session = open_session(
            repo, _Router(frames, holder, {}, thread_id), thread_id=thread_id, enable_subagents=True
        )
        holder.append(session)
        try:
            result = await session.submit("integrate after recovery")
            result = await _approve(session, result)
            result = await _approve(session, result)
            assert result.status is TurnStatus.COMPLETED
            payloads = _integrate_payloads(session, result.turn_id)
            assert payloads[0]["status"] == "ok"
            assert payloads[0]["metadata"]["applied"] == ("a.txt",)
            assert payloads[1]["status"] == "ok"
            assert not child_ws.exists()
        finally:
            await session.aclose()

    asyncio.run(_fresh_integrate())
    assert (repo / "a.txt").read_text(encoding="utf-8") == "child-killed\n"


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-v"]))
