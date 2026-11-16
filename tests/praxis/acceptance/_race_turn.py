"""取消竞态双进程驱动(阶段4 T9;旧 §13 取消竞态行的 praxis 形)。

由父测试用 ``sys.executable`` 以独立 OS 进程启动;两个实例(A 执行子 turn、
B 执行父 turn)真 db 同库(WAL 多写者),经 barrier 目录的文件钉时序:

    python _race_turn.py <workspace> <config.json>

config.json 形状:
    {"role": "child"|"parent", "thread_id": str, "turn_id": str,
     "frames": [spec...], "barrier": str(dir),
     "mode": "hold-before-dispatch"|"touch-hold"|"parent-hold",
     "marker": str(path),          # touch-hold 的副作用落点
     "expect_kill": bool}          # true = 本进程应被 SIGKILL,跑完即装置失败

三种模式(缝都是纯编排测试缝,_barrier_tool 同先例):

hold-before-dispatch  子首 dispatch 前挂起:turn.execute 查找处缝,首次调用
                      先写 ``child-ready`` 再轮询 ``go``——父测试借此钉
                      "级联 item 先落 → 子下一次 dispatch 必拒"。
touch-hold            子已 dispatch:race_touch.run 先真实 touch marker(副作用
                      已发生)、写 ``child-dispatched``、再轮询 ``release``——
                      钉"子 dispatch 先 → 痕迹 + cancel 收敛"。
parent-hold           父执行者:parent_hold.run 轮询本 turn 的 cancel_request
                      (父测试经第三连接 request_cancel 写入)→ 返回 → 父步边界
                      收敛 CANCELLED → Session._execute 终态挂钩 cascade_for
                      向子 thread 落 cancel_request(source=parent)。

两个执行进程都 ``run_sweep=False`` + ``submit(turn_id=预创建 turn)``——与生产
spawn 的子 Session 装配/收养完全同形(子根 turn 先于执行者存在,sweep-on-open
不得把预创建的 running turn 误标 interrupted)。

正常收敛(取消)打印结局 JSON 退出 0;expect_kill 的进程跑完即退出 1。
"""

from __future__ import annotations

import asyncio
import json
import sys
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

# 脚本位于 tests/praxis/acceptance/,parents[3] = 仓库根;先修 path 再 import praxis。
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import praxis.core.turn as turn_module  # noqa: E402
from praxis.adapters.models.scripted import ScriptedAdapter  # noqa: E402
from praxis.core.turn import execute as _real_execute  # noqa: E402
from praxis.interfaces.sdk import open_session  # noqa: E402
from praxis.protocol.items import ItemKind  # noqa: E402
from praxis.protocol.json import JsonValue  # noqa: E402
from praxis.protocol.models import ModelResponse, ModelToolCall, Usage  # noqa: E402
from praxis.protocol.policy import NetworkPolicy, PermissionPolicy  # noqa: E402
from praxis.protocol.sandbox import ToolExecContext  # noqa: E402
from praxis.protocol.tools import (  # noqa: E402
    CancellationMode,
    InterruptBehavior,
    ResolvedUse,
    ResultStatus,
    Tool,
    ToolContentBlock,
    ToolDefinition,
    ToolResult,
)

_BARRIER_TIMEOUT = 30.0


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


def _wait_file(path: Path) -> None:
    deadline = time.monotonic() + _BARRIER_TIMEOUT
    while not path.exists():
        if time.monotonic() > deadline:
            raise SystemExit(f"barrier file never appeared: {path}")
        time.sleep(0.02)


def _ok_result(name: str, text: str) -> ToolResult:
    return ToolResult(
        tool_call_id="pending",
        tool_name=name,
        content=(ToolContentBlock(type="text", data={"text": text}),),
        status=ResultStatus.OK,
    )


def _seam_tool(name: str, run: Any, *, idempotent: bool) -> Tool:
    return Tool(
        definition=ToolDefinition(
            name=name,
            description="TEST SEAM: cancel-race orchestration only.",
            input_schema={"type": "object"},
        ),
        validate_input=lambda arguments: arguments,
        run=run,
        static_effects=frozenset(),
        resolve_use=lambda arguments: ResolvedUse(frozenset(), ()),
        idempotent=idempotent,
        concurrency_safe=True,
        cancellation_mode=CancellationMode.COOPERATIVE,
        interrupt_behavior=InterruptBehavior.CANCEL,
        timeout_seconds=60,
        max_model_output_bytes=4_000,
    )


def _install_hold_before_dispatch(barrier: Path) -> None:
    state = {"held": False}

    async def seam(call: Any, ctx: Any) -> Any:
        if not state["held"]:
            state["held"] = True
            (barrier / "child-ready").write_text("ready", encoding="utf-8")
            _wait_file(barrier / "go")  # 级联 item 先落,再放行 dispatch
        return await _real_execute(call, ctx)

    turn_module.execute = seam


def _touch_hold_tool(barrier: Path, marker: Path) -> Tool:
    async def run(arguments: Mapping[str, JsonValue], exec_ctx: ToolExecContext) -> ToolResult:
        marker.write_text("race side effect\n", encoding="utf-8")  # 真实副作用先发生
        (barrier / "child-dispatched").write_text("dispatched", encoding="utf-8")
        _wait_file(barrier / "release")
        return _ok_result("race_touch", "touched and released")

    return _seam_tool("race_touch", run, idempotent=False)


def _noop_tool() -> Tool:
    async def run(arguments: Mapping[str, JsonValue], exec_ctx: ToolExecContext) -> ToolResult:
        return _ok_result("race_noop", "noop ran")

    return _seam_tool("race_noop", run, idempotent=True)


def _parent_hold_tool(holder: list[Any], turn_id: str) -> Tool:
    async def run(arguments: Mapping[str, JsonValue], exec_ctx: ToolExecContext) -> ToolResult:
        deadline = time.monotonic() + _BARRIER_TIMEOUT
        while True:
            items = holder[0].store.read_turn_items(turn_id)
            if any(item.kind == ItemKind.CANCEL_REQUEST for item in items):
                return _ok_result("parent_hold", "cancel observed")
            if time.monotonic() > deadline:
                raise SystemExit("parent never saw cancel_request")
            time.sleep(0.02)

    return _seam_tool("parent_hold", run, idempotent=True)


async def _amain(workspace: Path, config: Mapping[str, Any]) -> int:
    barrier = Path(str(config["barrier"]))
    mode = str(config["mode"])
    holder: list[Any] = []
    extra: tuple[Tool, ...] = ()
    if mode == "hold-before-dispatch":
        _install_hold_before_dispatch(barrier)
        extra = (_noop_tool(),)  # 被检查点拒的调用也必须是已知工具(拒在 dispatch 前)
    elif mode == "touch-hold":
        extra = (_touch_hold_tool(barrier, Path(str(config["marker"]))), _noop_tool())
    elif mode == "parent-hold":
        extra = (_parent_hold_tool(holder, str(config["turn_id"])),)
    session = open_session(
        workspace,
        ScriptedAdapter([_to_response(spec) for spec in config["frames"]]),
        thread_id=str(config["thread_id"]),
        run_sweep=False,  # 收养预创建 running turn——生产 spawn 子装配同形
        extra_tools=extra,
        policy=PermissionPolicy(network=NetworkPolicy(decision="allow")),
    )
    holder.append(session)
    try:
        result = await session.submit(f"acceptance: cancel race {mode}", turn_id=str(config["turn_id"]))
    finally:
        session.close()
    print(
        json.dumps(
            {
                "status": result.status.value,
                "reason": result.reason,
                "unknown_calls": list(result.unknown_calls),
            }
        )
    )
    return 1 if config.get("expect_kill") else 0


def main() -> None:
    if len(sys.argv) != 3:
        raise SystemExit(f"usage: {sys.argv[0]} <workspace> <config.json>")
    config = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
    raise SystemExit(asyncio.run(_amain(Path(sys.argv[1]), config)))


if __name__ == "__main__":
    main()