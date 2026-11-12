"""kill -9 续跑矩阵的子进程驱动(Task 16;设计 §5.8;旧 §13 纪律)。

由父测试用 ``sys.executable`` 以独立 OS 进程启动,等着被 SIGKILL:

    python _child_run_turn.py <workspace> <responses.json> <thread_id> <variant>

variant 三档(与 test_kill9_resume.py 一一对应):

a  无注入:bash 执行 ``touch marker && sleep 30`` 本身就慢。父进程等
   "tool_call item 落库且 marker 真实出现"(touch 在 sleep 之前)才 kill,
   此刻回合停在 started+非幂等 的执行中。
b  故障注入缝(patch 执行循环对管道唯一 ``execute`` 的调用点)。注意:
   turn.py 是 ``from praxis.core.pipeline import execute``——按名导入,
   运行时查找的是 ``praxis.core.turn.execute`` 这个模块属性,patch 必须
   打在查找处才生效(brief 所述"patch pipeline.execute"的落点即此)。
   首次调用先睡 _HOLD_SECONDS 再进原函数:kill 恰好落在 model_response
   item 已落库之后、管道写 tool_call item 之前(§5.8"已生成未分发")。
c  故障注入缝(dataclasses.replace 换掉 read 工具的 run):协程先睡
   _HOLD_SECONDS 再真实读。tool_call item 已写、幂等工具执行中,kill
   落在执行中途;resume 应自动重跑(重放安全性由 read.idempotent=True 给出)。

responses.json 形状(剧本每条一个对象,与 model_response item payload 同构):
    {"content": str,
     "tool_calls": [{"call_id": str, "name": str, "arguments": {}}],
     "usage": {"input_tokens": int, "output_tokens": int},
     "finish_reason": str}

子进程绝不自求多福地清理:SIGKILL 不留任何用户态收尾正是被验收的性质。
若它侥幸跑完 submit(缝失效=测试装置失败),打印结局并以退出码 1 报给父测试。
"""

from __future__ import annotations

import asyncio
import dataclasses
import json
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

# 脚本位于 tests/praxis/acceptance/,parents[3] = 仓库根;先修 path 再 import praxis。
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from praxis.adapters.models.scripted import ScriptedAdapter  # noqa: E402
from praxis.core.turn import execute as _real_execute  # noqa: E402
from praxis.interfaces.sdk import open_session  # noqa: E402
from praxis.protocol.json import JsonValue  # noqa: E402
from praxis.protocol.models import ModelResponse, ModelToolCall, Usage  # noqa: E402
from praxis.protocol.tools import Tool  # noqa: E402
from praxis.tools.builtins.command import create_bash_tool  # noqa: E402
from praxis.tools.builtins.filesystem import create_read_tool, create_write_tool  # noqa: E402

_HOLD_SECONDS = 45.0  # 必须远大于父进程轮询间隔:保证 kill 落在窗口正中


def _load_specs(path: str) -> list[Mapping[str, Any]]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, list) or not all(isinstance(entry, Mapping) for entry in raw):
        raise ValueError(f"responses.json must be a list of objects: {path}")
    return list(raw)


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


def _install_dispatch_seam() -> None:
    """变体 b:turn 循环执行首个工具调用前挂起(见模块 docstring 的落点说明)。"""
    state = {"first": True}

    async def _seam_execute(call: Any, ctx: Any) -> Any:
        if state["first"]:
            state["first"] = False
            await asyncio.sleep(_HOLD_SECONDS)
        return await _real_execute(call, ctx)

    import praxis.core.turn as turn_module

    turn_module.execute = _seam_execute


def _slow_read_tools(workspace: Path) -> list[Tool]:
    """变体 c:read.run 换成"睡 _HOLD_SECONDS 再真实读"的协程,其余字段原样。"""
    read_tool = create_read_tool(workspace)
    real_run = read_tool.run

    async def _slow_run(arguments: Mapping[str, JsonValue]) -> Any:
        await asyncio.sleep(_HOLD_SECONDS)
        return await real_run(arguments)

    slow_read = dataclasses.replace(read_tool, run=_slow_run)
    return [slow_read, create_write_tool(workspace), create_bash_tool(workspace)]


async def _amain(workspace: Path, specs: list[Mapping[str, Any]], thread_id: str, variant: str) -> int:
    if variant == "b":
        _install_dispatch_seam()
    tools = _slow_read_tools(workspace) if variant == "c" else None
    adapter = ScriptedAdapter([_to_response(spec) for spec in specs])
    session = open_session(workspace, adapter, thread_id=thread_id, tools=tools)
    try:
        result = await session.submit(f"acceptance: kill -9 resume matrix, variant {variant}")
    finally:
        session.close()
    # 走到这里说明没被 kill 住:装置失败,如实报告结局。
    print(json.dumps({"turn_id": result.turn_id, "status": result.status.value}))
    return 1


def main() -> None:
    if len(sys.argv) != 5:
        raise SystemExit(f"usage: {sys.argv[0]} <workspace> <responses.json> <thread_id> <a|b|c>")
    workspace = Path(sys.argv[1])
    specs = _load_specs(sys.argv[2])
    exit_code = asyncio.run(_amain(workspace, specs, sys.argv[3], sys.argv[4]))
    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
