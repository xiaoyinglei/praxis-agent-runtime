"""kill -9 mid-integrate-apply 的子进程驱动(阶段4 T8;设计 §5.8/§5.13)。

由父测试用 ``sys.executable`` 以独立 OS 进程启动,等着被 SIGKILL:

    python _child_integrate_turn.py <workspace> <scripts.json> <thread_id>

scripts.json 形状:{"parent": [spec...], "child": [spec...]}(spec 与
_child_run_turn.py 的 responses.json 条目同构)。父剧本 spawn 一个 write 子
(任务含标记 ``[kill-child]``),子剧本经生产 write 工具改 a.txt 后收尾;
父 wait 子终态后调 integrate apply。

故障注入缝:``GitWorktreeProvisioner.apply`` 类级替换为"先睡 _HOLD_SECONDS
再进原函数"——kill 恰好落在 integrate 的 tool_call item 已落库、真实 apply
尚未写父仓的窗口(started+非幂等 → outcome_unknown → reconcile,T4 矩阵)。

R-T8-1:子经**生产 write 工具**改自身 worktree(子工具按子工作区根重构 +
worktree 迁出地板私有集),无测试缝工具。

子进程绝不自求多福地清理:SIGKILL 不留任何用户态收尾正是被验收的性质。
若它侥幸跑完 resume(缝失效=测试装置失败),打印结局并以退出码 1 报给父测试。
"""

from __future__ import annotations

import asyncio
import json
import sys
import time
from collections.abc import AsyncIterator, Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

# 脚本位于 tests/praxis/acceptance/,parents[3] = 仓库根;先修 path 再 import praxis。
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from praxis.adapters.models.scripted import ScriptedAdapter  # noqa: E402
from praxis.adapters.workspaces.git import GitWorktreeProvisioner  # noqa: E402
from praxis.interfaces.sdk import open_session  # noqa: E402
from praxis.protocol.items import ItemKind  # noqa: E402
from praxis.protocol.models import (  # noqa: E402
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

_HOLD_SECONDS = 45.0  # 必须远大于父进程轮询间隔:保证 kill 落在窗口正中
_MARKER = "[kill-child]"


def _to_response(spec: Mapping[str, Any]) -> ModelResponse:
    calls = tuple(
        ModelToolCall(call_id=c["call_id"], name=c["name"], arguments=dict(c["arguments"]))
        for c in spec.get("tool_calls", [])
    )
    usage = spec.get("usage", {"input_tokens": 1, "output_tokens": 1})
    return ModelResponse(
        content=spec["content"],
        tool_calls=calls,
        usage=Usage(input_tokens=int(usage["input_tokens"]), output_tokens=int(usage["output_tokens"])),
        finish_reason=str(spec.get("finish_reason", "tool_calls" if calls else "stop")),
    )


class _Router:
    """标记路由 + PLACEHOLDER child id 重写(与 test_integrate.py 同纪律)。"""

    def __init__(
        self,
        parent: Sequence[ModelResponse],
        child: Sequence[ModelResponse],
        holder: list[Any],
        thread_id: str,
    ) -> None:
        self._parent = ScriptedAdapter(parent)
        self._child = ScriptedAdapter(child)
        self._holder = holder
        self._thread = thread_id

    def _latest_child_id(self) -> str:
        child_id = ""
        for item in self._holder[0].store.read_items(self._thread):
            if item.kind == ItemKind.CHILD_ALLOCATION:
                raw = item.payload.get("child_thread_id")
                if isinstance(raw, str):
                    child_id = raw
        return child_id

    def build_request(
        self, messages: Sequence[ModelMessage], tools: Sequence[Any], settings: ModelSettings
    ) -> WireRequest:
        return self._parent.build_request(messages, tools, settings)

    async def stream(self, req: WireRequest) -> AsyncIterator[ModelEvent]:
        text = " ".join(message.content for message in req.messages)
        adapter = self._child if _MARKER in text else self._parent
        async for event in adapter.stream(req):
            if isinstance(event, ModelToolCallDelta) and event.name_fragment in (
                "wait_subagent",
                "integrate_child_workspace",
            ):
                args = json.loads(event.arguments_fragment)
                args["child_thread_id"] = self._latest_child_id()
                event = replace(event, arguments_fragment=json.dumps(args))
            yield event

    async def complete(self, req: WireRequest) -> ModelResponse:
        return assemble_model_events(await collect_events(self.stream(req)))


def _install_apply_seam() -> None:
    real_apply = GitWorktreeProvisioner.apply

    def slow_apply(
        self: GitWorktreeProvisioner,
        parent_root: Path,
        patch: bytes,
        *,
        preimages: Mapping[str, str | None],
    ) -> Any:
        time.sleep(_HOLD_SECONDS)  # kill 窗口:tool_call 已落库,父仓尚未写
        return real_apply(self, parent_root, patch, preimages=preimages)

    GitWorktreeProvisioner.apply = slow_apply  # type: ignore[method-assign]


async def _amain(workspace: Path, scripts: Mapping[str, Any], thread_id: str) -> int:
    _install_apply_seam()
    holder: list[Any] = []
    router = _Router(
        [_to_response(spec) for spec in scripts["parent"]],
        [_to_response(spec) for spec in scripts["child"]],
        holder,
        thread_id,
    )
    session = open_session(
        workspace, router, thread_id=thread_id, enable_subagents=True
    )
    holder.append(session)
    try:
        r1 = await session.submit("acceptance: kill -9 mid integrate apply")
        if r1.status.value != "paused":
            print(json.dumps({"stage": "submit", "status": r1.status.value, "reason": r1.reason}))
            return 1
        # DESTRUCTIVE 显式审批后 dispatch integrate apply → 缝 hold → 等 SIGKILL。
        r2 = await session.resume(r1.turn_id, approval={"ci1": "allow"})
    finally:
        session.close()
    print(json.dumps({"stage": "resume", "turn_id": r2.turn_id, "status": r2.status.value}))
    return 1


def main() -> None:
    if len(sys.argv) != 4:
        raise SystemExit(f"usage: {sys.argv[0]} <workspace> <scripts.json> <thread_id>")
    workspace = Path(sys.argv[1])
    scripts = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
    exit_code = asyncio.run(_amain(workspace, scripts, sys.argv[3]))
    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
