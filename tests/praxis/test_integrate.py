"""阶段4 T8:integrate_child_workspace 实装(inspect/apply/discard)。

覆盖面(逐条对应 brief 与控制器裁决):
- 工具面形状与错误码(stub surface 直达):inspect {paths, diff_preview, sha 表}、
  apply applied/conflicts、conflicts 非空 → ``integrate_conflict`` + 双方字节保留、
  discard 幂等、``child_still_active``/``child_not_found`` 映射、其余 ValueError
  如实上抛交管道归一 tool_error;
- 真 git repo + 真 Session(scripted 父子,真实管道 + DESTRUCTIVE 显式审批):
  inspect/apply/discard 端到端;apply 子须终态;paths 子集;read_only 子拒绝;
- diff_preview ≤200_000 字节截断旗标(旧 diff_limit parity);大结果经 artifacts
  写时外部化(既有机制,内容寻址全文可回读);
- T6① 端到端:子 worktree ``.praxis/permissions.json``(untracked 与 git add 两态)
  经 integrate inspect/apply 不渗父;
- R-T7-3 经 integrate 承重:父仓祖先 symlink 指仓外 → apply conflicts,绝不写出
  parent_root(provisioner 层承重见 test_child_workspace.py)。

R-T8-1 后 write 子经**生产工具**(write/bash/apply_patch)改自身 worktree:
子工具按子工作区根重构 + worktree 迁出地板私有集(host_worktrees_dir)。
旗舰端到端与生产工具测试用真子编辑;其余 integrate 机制测试(冲突/子集/
祖先 symlink 等)仍由测试直接写 worktree 字节(T7 先例,聚焦交付语义)。

仓库无 pytest-asyncio:与 test_agent_tools.py 同风格,asyncio.run 包装。
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import shlex
import subprocess
from collections.abc import AsyncIterator, Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any
from uuid import uuid4

import pytest

from praxis.adapters.sandbox.base import probe_backend
from praxis.adapters.workspaces.git import GitWorktreeProvisioner
from praxis.composition import build_child_session
from praxis.core.statemachine import TurnStatus
from praxis.interfaces.sdk import ScriptedAdapter, open_session
from praxis.protocol.agents import ChildActiveError
from praxis.protocol.items import ItemKind
from praxis.protocol.json import JsonValue
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
from praxis.protocol.budget import UNLIMITED
from praxis.protocol.policy import NetworkPolicy, PermissionPolicy
from praxis.protocol.sandbox import ToolExecContext
from praxis.protocol.tools import (
    CancellationMode,
    InterruptBehavior,
    ResolvedUse,
    ResultStatus,
    Tool,
    ToolContentBlock,
    ToolDefinition,
    ToolEffect,
    ToolResult,
)
from praxis.protocol.workspaces import ApplyResult, WorkspaceInspection
from praxis.store.artifacts import ArtifactStore
from praxis.tools.builtins.child_env import child_env
from praxis.tools.builtins.agent_tools import create_agent_tools

_ZERO_USAGE = Usage(input_tokens=0, output_tokens=0)


def _git(cwd: Path, *args: str) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(["git", "-C", str(cwd), *args], capture_output=True, check=True)


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _resp(content: str, calls: tuple[tuple[str, str, dict[str, Any]], ...] = ()) -> ModelResponse:
    return ModelResponse(
        content=content,
        tool_calls=tuple(ModelToolCall(call_id=c, name=n, arguments=a) for c, n, a in calls),
        usage=_ZERO_USAGE,
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


class _ScriptRouter:
    """确定性剧本路由:按任务标记把父/子请求分到各自剧本(并行子不抢帧)。

    子的 user_message 含任务标记且贯穿子 turn 上下文;父上下文不含标记
    (spawn/wait/integrate 的结果文本都不回显任务)。PLACEHOLDER child id
    重写为 store 最近一条 allocation 的真实 id(T6 _ChildIdRewritingAdapter
    同先例,扩到 integrate)。``rescript`` 换默认剧本供跨 turn 测试复用会话。
    """

    def __init__(
        self,
        default: Sequence[ModelResponse],
        session_holder: list[Any],
        thread_id: str,
        marked: Mapping[str, Sequence[ModelResponse]] | None = None,
    ) -> None:
        self._default = ScriptedAdapter(default)
        self._marked = {
            marker: ScriptedAdapter(frames) for marker, frames in (marked or {}).items()
        }
        self._holder = session_holder
        self._thread = thread_id

    def rescript(self, frames: Sequence[ModelResponse]) -> None:
        self._default = ScriptedAdapter(frames)

    def _latest_child_id(self) -> str:
        child_id = ""
        for item in self._holder[0].store.read_items(self._thread):
            if item.kind == ItemKind.CHILD_ALLOCATION:
                raw = item.payload.get("child_thread_id")
                if isinstance(raw, str):
                    child_id = raw
        return child_id

    def _pick(self, req: WireRequest) -> ScriptedAdapter:
        text = " ".join(message.content for message in req.messages)
        for marker, adapter in self._marked.items():
            if marker in text:
                return adapter
        return self._default

    def build_request(
        self, messages: Sequence[ModelMessage], tools: Sequence[Any], settings: ModelSettings
    ) -> WireRequest:
        return self._default.build_request(messages, tools, settings)

    async def stream(self, req: WireRequest) -> AsyncIterator[ModelEvent]:
        async for event in self._pick(req).stream(req):
            if isinstance(event, ModelToolCallDelta) and event.name_fragment in (
                "wait_subagent",
                "integrate_child_workspace",
            ):
                args = json.loads(event.arguments_fragment)
                if str(args.get("child_thread_id", "")).startswith("PLACEHOLDER"):
                    args["child_thread_id"] = self._latest_child_id()
                    event = replace(event, arguments_fragment=json.dumps(args))
            yield event

    async def complete(self, req: WireRequest) -> ModelResponse:
        return assemble_model_events(await collect_events(self.stream(req)))


def _spawn_wait_frames(mode: str = "write", wait_timeout: float = 10) -> list[ModelResponse]:
    return [
        _resp("", (("s1", "spawn_subagent", {"task": "child task", "workspace_mode": mode}),)),
        _resp("child done"),
        _resp(
            "",
            (("w1", "wait_subagent", {"child_thread_id": "PLACEHOLDER", "timeout": wait_timeout}),)),
    ]


def _integrate_frame(action: str, paths: list[str] | None = None) -> ModelResponse:
    args: dict[str, Any] = {"child_thread_id": "PLACEHOLDER", "action": action}
    if paths is not None:
        args["paths"] = paths
    return _resp("", (("i1", "integrate_child_workspace", args),))


def _open(repo: Path, frames: Sequence[ModelResponse], **kwargs: Any) -> Any:
    holder: list[Any] = []
    marked = kwargs.pop("marked", None)
    session = open_session(
        repo, _ScriptRouter(frames, holder, "parent", marked),
        thread_id="parent", enable_subagents=True, **kwargs,
    )
    holder.append(session)
    return session


def _rescript(session: Any, frames: Sequence[ModelResponse]) -> None:
    adapter = session.adapter
    assert isinstance(adapter, _ScriptRouter)
    adapter.rescript(frames)


def _allocations(session: Any) -> list[Mapping[str, JsonValue]]:
    return [
        item.payload
        for item in session.store.read_items(session.thread_id)
        if item.kind == ItemKind.CHILD_ALLOCATION
    ]


def _integrate_payloads(session: Any, turn_id: str) -> list[Mapping[str, JsonValue]]:
    return [
        item.payload
        for item in session.store.read_turn_items(turn_id)
        if item.kind == ItemKind.TOOL_RESULT
        and item.payload.get("tool_name") == "integrate_child_workspace"
    ]


async def _approve(session: Any, result: Any) -> Any:
    """DESTRUCTIVE 显式审批(S5):PAUSED(approval_required)→ once allow 续跑。"""
    if result.status is TurnStatus.PAUSED and result.reason == "approval_required":
        assert result.pending_approvals[0].tool_name == "integrate_child_workspace"
        return await session.resume(result.turn_id, approval={"i1": "allow"})
    return result


# ---------------------------------------------------------------------------
# ① 工具面形状与错误码(stub surface 直达)
# ---------------------------------------------------------------------------


class _IntegrateStub:
    """SubagentSurface 的 integrate 面测试替身(canned 结果 + 异常注入)。"""

    def __init__(self) -> None:
        self.exc: Exception | None = None
        self.calls: list[tuple[str, str, tuple[str, ...] | None]] = []
        self.inspection = WorkspaceInspection(
            paths=("a.txt",),
            preimages={"a.txt": "pre-sha"},
            postimages={"a.txt": "post-sha"},
            diff_preview="diff --git a/a.txt b/a.txt\n",
            diff_truncated=False,
            diff_bytes=27,
        )
        self.apply_result = ApplyResult(applied=["a.txt"], conflicts=[])

    def integrate_inspect(
        self, child_thread_id: str, *, paths: Sequence[str] | None = None
    ) -> WorkspaceInspection:
        self.calls.append(("inspect", child_thread_id, tuple(paths) if paths is not None else None))
        if self.exc is not None:
            raise self.exc
        return self.inspection

    def integrate_apply(
        self, child_thread_id: str, *, paths: Sequence[str] | None = None
    ) -> ApplyResult:
        self.calls.append(("apply", child_thread_id, tuple(paths) if paths is not None else None))
        if self.exc is not None:
            raise self.exc
        return self.apply_result

    def integrate_discard(self, child_thread_id: str) -> None:
        self.calls.append(("discard", child_thread_id, None))
        if self.exc is not None:
            raise self.exc


def _stub_tools(stub: _IntegrateStub) -> dict[str, Tool]:
    return {tool.definition.name: tool for tool in create_agent_tools(lambda: stub)}


def test_integrate_tool_surface_shapes_and_error_codes() -> None:
    async def scenario() -> None:
        stub = _IntegrateStub()
        tools = _stub_tools(stub)
        integrate = tools["integrate_child_workspace"]
        ctx = ToolExecContext()
        assert integrate.static_effects == frozenset(
            {ToolEffect.DESTRUCTIVE, ToolEffect.WRITE_WORKSPACE}
        )

        result = await integrate.run(
            {"child_thread_id": "c", "action": "inspect", "paths": ["a.txt"]}, ctx
        )
        assert result.status is ResultStatus.OK
        assert result.metadata["paths"] == ("a.txt",)
        assert result.metadata["diff_preview"] == stub.inspection.diff_preview
        assert result.metadata["diff_truncated"] is False
        assert result.metadata["postimages"] == {"a.txt": "post-sha"}
        assert result.metadata["preimages"] == {"a.txt": "pre-sha"}
        assert stub.calls[-1] == ("inspect", "c", ("a.txt",))
        assert stub.inspection.diff_preview in result.text

        result = await integrate.run({"child_thread_id": "c", "action": "apply"}, ctx)
        assert result.status is ResultStatus.OK
        assert result.metadata["applied"] == ("a.txt",)
        assert result.metadata["conflicts"] == ()
        assert stub.calls[-1] == ("apply", "c", None)

        stub.apply_result = ApplyResult(applied=[], conflicts=["a.txt", "b.txt"])
        result = await integrate.run({"child_thread_id": "c", "action": "apply"}, ctx)
        assert result.status is ResultStatus.ERROR
        assert result.error_code == "integrate_conflict"
        assert result.metadata["conflicts"] == ("a.txt", "b.txt")
        assert result.metadata["applied"] == ()
        assert "preserved" in result.text

        result = await integrate.run({"child_thread_id": "c", "action": "discard"}, ctx)
        assert result.status is ResultStatus.OK
        assert stub.calls[-1] == ("discard", "c", None)

        stub.exc = ChildActiveError("child 'c' is still active")
        result = await integrate.run({"child_thread_id": "c", "action": "apply"}, ctx)
        assert result.error_code == "child_still_active"

        stub.exc = ValueError("unknown child thread 'ghost' (no allocation in parent thread)")
        result = await integrate.run({"child_thread_id": "ghost", "action": "inspect"}, ctx)
        assert result.error_code == "child_not_found"

        # 其余 ValueError(如 read_only 子拒绝)如实上抛,管道唯一归一 tool_error。
        stub.exc = ValueError("child 'c' is read_only; no isolated workspace")
        with pytest.raises(ValueError, match="read_only"):
            await integrate.run({"child_thread_id": "c", "action": "apply"}, ctx)

    asyncio.run(scenario())


# ---------------------------------------------------------------------------
# ② 真 git + 真 Session:inspect/apply/discard 端到端(显式审批)
# ---------------------------------------------------------------------------


def test_inspect_apply_end_to_end_with_explicit_approval(repo: Path) -> None:
    async def scenario() -> None:
        # R-T8-1:子经**生产 write 工具**改自身 worktree(子工具按子根重构)。
        marked = {
            "[edit-child]": [
                _resp("", (("ew1", "write", {"path": "a.txt", "content": "child-a\n"}),)),
                _resp("", (("ew2", "write", {"path": "new.txt", "content": "new\n"}),)),
                _resp("child done"),
            ]
        }
        frames = [
            _resp(
                "",
                (("s1", "spawn_subagent", {"task": "[edit-child] edit files", "workspace_mode": "write"}),)),
            _resp("", (("w1", "wait_subagent", {"child_thread_id": "PLACEHOLDER", "timeout": 10}),)),
            _resp("turn one done"),
        ]
        session = _open(repo, frames, marked=marked)
        try:
            r1 = await session.submit("go")
            assert r1.status is TurnStatus.COMPLETED
            allocation = _allocations(session)[0]
            child_ws = Path(str(allocation["workspace"]))
            child_results = [
                item.payload
                for item in session.store.read_turn_items(str(allocation["child_root_turn_id"]))
                if item.kind == ItemKind.TOOL_RESULT
            ]
            assert [p["status"] for p in child_results] == ["ok", "ok"]
            assert (child_ws / "a.txt").read_text(encoding="utf-8") == "child-a\n"
            assert (child_ws / "new.txt").read_text(encoding="utf-8") == "new\n"
            _rescript(session, [_integrate_frame("inspect"), _integrate_frame("apply"), _resp("parent done")])
            r2 = await session.submit("integrate")
            # 第一个 integrate(inspect)即 DESTRUCTIVE 显式审批(S5)。
            assert r2.status is TurnStatus.PAUSED
            assert r2.reason == "approval_required"
            r3 = await _approve(session, r2)
            # inspect 结果落库后 apply 再次 ask → 再批(逐次显式批)。
            assert r3.status is TurnStatus.PAUSED
            r4 = await _approve(session, r3)
            assert r4.status is TurnStatus.COMPLETED
            inspect_payload, apply_payload = _integrate_payloads(session, r2.turn_id)
            assert inspect_payload["status"] == "ok"
            assert inspect_payload["metadata"]["paths"] == ("a.txt", "new.txt")
            assert inspect_payload["metadata"]["preimages"] == {
                "a.txt": _sha(b"base-a\n"),
                "new.txt": None,
            }
            assert inspect_payload["metadata"]["postimages"] == {
                "a.txt": _sha(b"child-a\n"),
                "new.txt": _sha(b"new\n"),
            }
            assert "child-a" in str(inspect_payload["metadata"]["diff_preview"])
            assert inspect_payload["metadata"]["diff_truncated"] is False
            assert apply_payload["status"] == "ok"
            assert apply_payload["metadata"]["applied"] == ("a.txt", "new.txt")
            assert apply_payload["metadata"]["conflicts"] == ()
            assert (repo / "a.txt").read_text(encoding="utf-8") == "child-a\n"
            assert (repo / "new.txt").read_text(encoding="utf-8") == "new\n"
        finally:
            await session.aclose()

    asyncio.run(scenario())


def test_apply_conflict_preserves_both_sides(repo: Path) -> None:
    async def scenario() -> None:
        session = _open(repo, [*_spawn_wait_frames(), _resp("turn one done")])
        try:
            r1 = await session.submit("go")
            assert r1.status is TurnStatus.COMPLETED
            child_ws = Path(str(_allocations(session)[0]["workspace"]))
            (child_ws / "a.txt").write_text("child-a\n", encoding="utf-8")
            (child_ws / "new.txt").write_text("new\n", encoding="utf-8")
            # 快照后父并发改了 a.txt(preimage 不一致)。
            (repo / "a.txt").write_text("parent-moved\n", encoding="utf-8")
            _rescript(session, [_integrate_frame("apply"), _resp("turn two done")])
            r2 = await session.submit("integrate")
            r3 = await _approve(session, r2)
            assert r3.status is TurnStatus.COMPLETED
            payload = _integrate_payloads(session, r2.turn_id)[0]
            assert payload["status"] == "error"
            assert payload["error_code"] == "integrate_conflict"
            assert payload["metadata"]["conflicts"] == ("a.txt",)
            assert payload["metadata"]["applied"] == ()
            # 双方字节都在:父保留自己的新字节,子快照保留子字节;整组零写。
            assert (repo / "a.txt").read_text(encoding="utf-8") == "parent-moved\n"
            assert (child_ws / "a.txt").read_text(encoding="utf-8") == "child-a\n"
            assert not (repo / "new.txt").exists()
        finally:
            await session.aclose()

    asyncio.run(scenario())


def _hang_tool(release: asyncio.Event) -> Tool:
    async def hang_run(arguments: Mapping[str, JsonValue], exec_ctx: ToolExecContext) -> ToolResult:
        await release.wait()
        return ToolResult(
            tool_call_id="pending",
            tool_name="child_hang",
            content=(ToolContentBlock(type="text", data={"text": "released"}),),
            status=ResultStatus.OK,
        )

    return Tool(
        definition=ToolDefinition(
            name="child_hang", description="hang until released", input_schema={"type": "object"}
        ),
        validate_input=lambda arguments: arguments,
        run=hang_run,
        static_effects=frozenset(),
        resolve_use=lambda arguments: ResolvedUse(frozenset(), ()),
        idempotent=True,
        concurrency_safe=True,
        cancellation_mode=CancellationMode.COOPERATIVE,
        interrupt_behavior=InterruptBehavior.CANCEL,
        timeout_seconds=30,
        max_model_output_bytes=4_000,
    )


def test_apply_requires_terminal_child(repo: Path) -> None:
    release = asyncio.Event()

    async def scenario() -> None:
        marked = {
            "[hang-child]": [
                _resp("", (("h1", "child_hang", {}),)),
                _resp("child done"),
            ]
        }
        frames = [
            _resp(
                "",
                (
                    (
                        "s1",
                        "spawn_subagent",
                        {"task": "[hang-child] stay active", "workspace_mode": "write"},
                    ),
                ),
            ),
            _resp("", (("w1", "wait_subagent", {"child_thread_id": "PLACEHOLDER", "timeout": 0}),)),
            _integrate_frame("apply"),
            _resp("parent done"),
        ]
        session = _open(repo, frames, marked=marked, extra_tools=(_hang_tool(release),))
        try:
            r1 = await session.submit("go")
            r2 = await _approve(session, r1)
            assert r2.status is TurnStatus.COMPLETED
            payload = _integrate_payloads(session, r1.turn_id)[0]
            assert payload["error_code"] == "child_still_active"
            assert (repo / "a.txt").read_text(encoding="utf-8") == "base-a\n"
        finally:
            release.set()
            await session.aclose()

    asyncio.run(scenario())


def test_discard_idempotent_and_inspect_after_discard_honest(repo: Path) -> None:
    async def scenario() -> None:
        frames = [
            *_spawn_wait_frames(),
            _integrate_frame("discard"),
            _integrate_frame("discard"),
            _integrate_frame("inspect"),
            _resp("parent done"),
        ]
        session = _open(repo, frames)
        try:
            result = await session.submit("go")
            for _ in range(3):
                result = await _approve(session, result)
            assert result.status is TurnStatus.COMPLETED
            payloads = _integrate_payloads(session, result.turn_id)
            assert len(payloads) == 3
            assert payloads[0]["status"] == "ok"
            assert payloads[1]["status"] == "ok"  # 幂等:第二次 discard 同样 OK
            child_ws = Path(str(_allocations(session)[0]["workspace"]))
            assert not child_ws.exists()
            assert str(child_ws) not in _git(repo, "worktree", "list").stdout.decode()
            # discard 后 inspect:worktree 已不在,如实 tool_error(不发明第二套码)。
            assert payloads[2]["status"] == "error"
            assert payloads[2]["error_code"] == "tool_error"
        finally:
            await session.aclose()

    asyncio.run(scenario())


def test_apply_paths_subset(repo: Path) -> None:
    async def scenario() -> None:
        session = _open(repo, [*_spawn_wait_frames(), _resp("turn one done")])
        try:
            r1 = await session.submit("go")
            assert r1.status is TurnStatus.COMPLETED
            child_ws = Path(str(_allocations(session)[0]["workspace"]))
            (child_ws / "a.txt").write_text("child-a\n", encoding="utf-8")
            (child_ws / "b.txt").write_text("child-b\n", encoding="utf-8")
            (child_ws / "new.txt").write_text("new\n", encoding="utf-8")
            _rescript(
                session,
                [
                    _integrate_frame("apply", paths=["a.txt"]),
                    _integrate_frame("apply", paths=["b.txt", "new.txt"]),
                    _resp("parent done"),
                ],
            )
            r2 = await session.submit("integrate")
            result = await _approve(session, r2)
            # 第一次子集 apply 之后:只有 a.txt 落父仓。
            payloads = _integrate_payloads(session, r2.turn_id)
            assert payloads[0]["metadata"]["applied"] == ("a.txt",)
            assert (repo / "a.txt").read_text(encoding="utf-8") == "child-a\n"
            assert (repo / "b.txt").read_text(encoding="utf-8") == "base-b\n"
            assert not (repo / "new.txt").exists()
            result = await _approve(session, result)
            assert result.status is TurnStatus.COMPLETED
            payloads = _integrate_payloads(session, r2.turn_id)
            assert payloads[1]["metadata"]["applied"] == ("b.txt", "new.txt")
            assert (repo / "b.txt").read_text(encoding="utf-8") == "child-b\n"
            assert (repo / "new.txt").read_text(encoding="utf-8") == "new\n"
        finally:
            await session.aclose()

    asyncio.run(scenario())


def test_read_only_child_integrate_rejected(repo: Path) -> None:
    async def scenario() -> None:
        frames = [
            *_spawn_wait_frames(mode="read_only"),
            _integrate_frame("inspect"),
            _resp("parent done"),
        ]
        session = _open(repo, frames)
        try:
            r1 = await session.submit("go")
            r2 = await _approve(session, r1)
            assert r2.status is TurnStatus.COMPLETED
            payload = _integrate_payloads(session, r1.turn_id)[0]
            assert payload["error_code"] == "tool_error"
            assert "read_only" in str(payload["error_message"])
        finally:
            await session.aclose()

    asyncio.run(scenario())


def test_unknown_child_integrate_not_found(repo: Path) -> None:
    async def scenario() -> None:
        frames = [
            _resp(
                "",
                (("i1", "integrate_child_workspace", {"child_thread_id": "ghost", "action": "apply"}),)),
            _resp("parent done"),
        ]
        session = _open(repo, frames)
        try:
            r1 = await session.submit("go")
            r2 = await _approve(session, r1)
            assert r2.status is TurnStatus.COMPLETED
            payload = _integrate_payloads(session, r1.turn_id)[0]
            assert payload["error_code"] == "child_not_found"
        finally:
            await session.aclose()

    asyncio.run(scenario())


def test_integrate_never_mode_denied(repo: Path) -> None:
    async def scenario() -> None:
        frames = [*_spawn_wait_frames(), _integrate_frame("apply"), _resp("parent done")]
        session = _open(repo, frames, mode="never")
        try:
            r1 = await session.submit("go")
            assert r1.status is TurnStatus.COMPLETED
            payload = _integrate_payloads(session, r1.turn_id)[0]
            assert payload["status"] == "denied"
            assert payload["error_code"] == "permission_denied"
        finally:
            await session.aclose()

    asyncio.run(scenario())


# ---------------------------------------------------------------------------
# ②b R-T8-1:write 子经生产 write/bash/apply_patch 改自身 worktree + 父 confinement
# ---------------------------------------------------------------------------


def test_write_child_production_tools_edit_own_worktree(repo: Path) -> None:
    """子工具按子工作区根重构(R-T8-1(i))+ worktree 迁出地板(iii):
    write/bash/apply_patch(Add File,非 DESTRUCTIVE)三条生产路径全通。"""

    async def scenario() -> None:
        patch_text = "*** Begin Patch\n*** Add File: patched.txt\n+patched\n*** End Patch\n"
        # 父不 submit;剧本只被子 Session 消费(适配器继承父,公开构造)。
        adapter = ScriptedAdapter(
            [
                _resp("", (("cw1", "write", {"path": "a.txt", "content": "child-a\n"}),)),
                _resp("", (("cb1", "bash", {"command": "printf 'child-b\\n' > b.txt"}),)),
                _resp("", (("cp1", "apply_patch", {"patch": patch_text}),)),
                _resp("child done"),
            ]
        )
        parent = open_session(
            repo, adapter, thread_id="parent",
            policy=PermissionPolicy(network=NetworkPolicy(decision="allow")),
        )
        provisioner = GitWorktreeProvisioner(child_env)
        ws = provisioner.prepare(repo, uuid4().hex, "write")
        child = build_child_session(
            parent=parent, workspace=ws, thread_id=uuid4().hex, limits=UNLIMITED,
            tool_subset=tuple(parent.tools.values()), workspace_mode="write",
        )
        try:
            result = await child.submit("edit my workspace")
            assert result.status is TurnStatus.COMPLETED
            payloads = [
                item.payload
                for item in child.store.read_turn_items(result.turn_id)
                if item.kind == ItemKind.TOOL_RESULT
            ]
            assert [p["status"] for p in payloads] == ["ok", "ok", "ok"]
            assert (ws / "a.txt").read_text(encoding="utf-8") == "child-a\n"
            assert (ws / "b.txt").read_text(encoding="utf-8") == "child-b\n"
            assert (ws / "patched.txt").read_text(encoding="utf-8") == "patched\n"
            # 交付:inspect preimages + diff + apply 全真实管道语义。
            inspection = provisioner.inspect(ws)
            applied = provisioner.apply(repo, provisioner.diff(ws), preimages=inspection.preimages)
            assert applied.conflicts == []
            assert applied.applied == ["a.txt", "b.txt", "patched.txt"]
            assert (repo / "a.txt").read_text(encoding="utf-8") == "child-a\n"
            assert (repo / "patched.txt").read_text(encoding="utf-8") == "patched\n"
        finally:
            await child.aclose()
            await parent.aclose()

    asyncio.run(scenario())


@pytest.mark.skipif(not probe_backend().available, reason="no sandbox backend")
def test_parent_production_tools_cannot_reach_child_worktree(repo: Path) -> None:
    """R-T8-1 安全承重:迁出地板**不开父旁路**——父 bash 读/写子 worktree
    绝对路径被沙箱拒,父 write/read 工具被根围栏拒(invalid_input)。"""

    async def scenario() -> None:
        # 父 bash 需网络 allow(默认 ask 会暂停命令网络;kill9 先例)。
        session = _open(
            repo,
            [*_spawn_wait_frames(), _resp("turn one done")],
            policy=PermissionPolicy(network=NetworkPolicy(decision="allow")),
        )
        try:
            r1 = await session.submit("go")
            assert r1.status is TurnStatus.COMPLETED
            child_ws = Path(str(_allocations(session)[0]["workspace"]))
            target = shlex.quote(str(child_ws / "a.txt"))
            _rescript(
                session,
                [
                    _resp("", (("pb1", "bash", {"command": f"cat {target}"}),)),
                    _resp("", (("pb2", "bash", {"command": f"printf hacked > {target}"}),)),
                    _resp("", (("pw1", "write", {"path": str(child_ws / "a.txt"), "content": "hacked\n"}),)),
                    _resp("", (("pr1", "read", {"path": str(child_ws / "a.txt")}),)),
                    _resp("parent done"),
                ],
            )
            r2 = await session.submit("probe")
            assert r2.status is TurnStatus.COMPLETED
            payloads = {
                item.payload.get("call_id"): item.payload
                for item in session.store.read_turn_items(r2.turn_id)
                if item.kind == ItemKind.TOOL_RESULT
            }
            # 沙箱拒:命令真实执行但非零退出(读/写都不可达)。
            assert int(payloads["pb1"]["metadata"]["exit_code"]) != 0
            assert int(payloads["pb2"]["metadata"]["exit_code"]) != 0
            # 工具根围栏:绝对路径越出父工作区 → invalid_input。
            assert payloads["pw1"]["error_code"] == "invalid_input"
            assert payloads["pr1"]["error_code"] == "invalid_input"
            assert (child_ws / "a.txt").read_text(encoding="utf-8") == "base-a\n"
        finally:
            await session.aclose()

    asyncio.run(scenario())


# ---------------------------------------------------------------------------
# ③ diff_preview 截断旗标(≤200_000 parity)+ 大结果 artifacts 外部化
# ---------------------------------------------------------------------------


def test_inspect_preview_truncation_flag_and_artifact_externalization(repo: Path) -> None:
    async def scenario() -> None:
        session = _open(repo, [*_spawn_wait_frames(), _resp("turn one done")])
        try:
            r1 = await session.submit("go")
            assert r1.status is TurnStatus.COMPLETED
            child_ws = Path(str(_allocations(session)[0]["workspace"]))
            # tracked 大 diff(>200_000 字节)+ 大量 untracked(sha 表撑大结果)。
            (child_ws / "a.txt").write_text("x" * 300_000 + "\n", encoding="utf-8")
            for index in range(600):
                name = f"file-{index:04d}-" + "y" * 40 + ".txt"
                (child_ws / name).write_text(f"u{index}\n", encoding="utf-8")
            _rescript(session, [_integrate_frame("inspect"), _resp("parent done")])
            r2 = await session.submit("inspect")
            r3 = await _approve(session, r2)
            assert r3.status is TurnStatus.COMPLETED
            payload = _integrate_payloads(session, r2.turn_id)[0]
            metadata = payload["metadata"]
            assert metadata["diff_truncated"] is True
            assert int(metadata["diff_bytes"]) > 200_000
            preview = str(metadata["diff_preview"])
            assert len(preview.encode("utf-8")) <= 200_000
            assert len(metadata["paths"]) == 601
            # 超 max_model_output_bytes(256_000)→ 写时外部化(既有机制)。
            assert payload["truncated"] is True
            digest = str(metadata["artifact_digest"])
            artifacts = ArtifactStore(repo / ".praxis" / "artifacts")
            full = artifacts.get(digest)
            assert full is not None
            text = full.decode("utf-8")
            assert "file-0599-" in text  # 全文尾部可回读(item 里只有头部)
            assert len(text) > 256_000
            assert int(metadata["artifact_bytes"]) == len(full)
        finally:
            await session.aclose()

    asyncio.run(scenario())


# ---------------------------------------------------------------------------
# ④ T6① 端到端:子 .praxis/permissions.json 经 integrate 不渗父
# ---------------------------------------------------------------------------


def test_child_praxis_permissions_never_leak_through_integrate(repo: Path) -> None:
    async def scenario() -> None:
        parent_policy = repo / ".praxis" / "permissions.json"
        parent_policy.parent.mkdir(parents=True, exist_ok=True)
        parent_policy.write_text(PermissionPolicy().model_dump_json(indent=2), encoding="utf-8")
        before = parent_policy.read_bytes()
        session = _open(repo, [*_spawn_wait_frames(), _resp("turn one done")])
        try:
            r1 = await session.submit("go")
            assert r1.status is TurnStatus.COMPLETED
            child_ws = Path(str(_allocations(session)[0]["workspace"]))
            # 子 always 落盘形态(T6 已钉 write 子写自己的隔离文件)+ git add 态。
            (child_ws / ".praxis").mkdir(exist_ok=True)
            (child_ws / ".praxis" / "permissions.json").write_text(
                '{"always": ["*"]}\n', encoding="utf-8"
            )
            _git(child_ws, "add", ".praxis/permissions.json")
            (child_ws / "a.txt").write_text("child-a\n", encoding="utf-8")
            _rescript(
                session,
                [_integrate_frame("inspect"), _integrate_frame("apply"), _resp("parent done")],
            )
            r2 = await session.submit("integrate")
            result = await _approve(session, r2)
            result = await _approve(session, result)
            assert result.status is TurnStatus.COMPLETED
            inspect_payload, apply_payload = _integrate_payloads(session, r2.turn_id)
            assert ".praxis/permissions.json" not in inspect_payload["metadata"]["paths"]
            assert ".praxis" not in str(inspect_payload["metadata"]["diff_preview"])
            assert apply_payload["metadata"]["applied"] == ("a.txt",)
            assert (repo / "a.txt").read_text(encoding="utf-8") == "child-a\n"
            # 父策略文件字节不动(子的 always 绝不渗父)。
            assert parent_policy.read_bytes() == before
        finally:
            await session.aclose()

    asyncio.run(scenario())


# ---------------------------------------------------------------------------
# ⑤ R-T7-3 经 integrate 承重:祖先 symlink 指仓外 → conflict,零越界写
# ---------------------------------------------------------------------------


def test_integrate_apply_conflicts_on_ancestor_symlink_escape(repo: Path, tmp_path: Path) -> None:
    async def scenario() -> None:
        outside = tmp_path / "outside"
        outside.mkdir()
        session = _open(repo, [*_spawn_wait_frames(), _resp("turn one done")])
        try:
            r1 = await session.submit("go")
            assert r1.status is TurnStatus.COMPLETED
            child_ws = Path(str(_allocations(session)[0]["workspace"]))
            (child_ws / "link").mkdir()
            (child_ws / "link" / "pwned.txt").write_text("pwned\n", encoding="utf-8")
            (repo / "link").symlink_to(outside, target_is_directory=True)
            _rescript(session, [_integrate_frame("apply"), _resp("parent done")])
            r2 = await session.submit("integrate")
            r3 = await _approve(session, r2)
            assert r3.status is TurnStatus.COMPLETED
            payload = _integrate_payloads(session, r2.turn_id)[0]
            assert payload["error_code"] == "integrate_conflict"
            assert payload["metadata"]["conflicts"] == ("link/pwned.txt",)
            assert not (outside / "pwned.txt").exists()
            assert (child_ws / "link" / "pwned.txt").read_text(encoding="utf-8") == "pwned\n"
        finally:
            await session.aclose()

    asyncio.run(scenario())
